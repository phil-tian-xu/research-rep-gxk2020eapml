"""Figure 4 from saved annual fits. Reporting only: never train or tune models."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.data_query.parquet_dataset import ParquetPanel
from core.figure4_importance import characteristic_groups, streaming_zero_out
from core.run_state import dataset_identity, load_run_config, read_checkpoints


def restore_predictor(tuning):
    """Return (predict, feature_names, optional linear coefficients).

    Future nonlinear saved-model loaders belong here and return coefficients=None.
    The importance evaluator and plotter are independent of model family.
    Only explicitly recognized linear models may use the coefficient shortcut.
    """
    model = tuning["specification"]
    if model not in {"OLS", "OLS_HUBER", "OLS3_HUBER", "ENET_HUBER", "PCR", "PLS"}:
        raise NotImplementedError(f"No saved-predictor loader for {model}; no refit will be attempted.")
    if not {"coefficients", "intercept", "feature_names"} <= tuning.keys():
        raise ValueError(f"{model} lacks saved coefficients/intercept/feature names. "
                         "Existing PCR/PLS imports cannot support Figure 4; no refit attempted.")
    names = list(tuning["feature_names"])
    beta = np.asarray(tuning["coefficients"], dtype=float)
    intercept = float(tuning["intercept"])
    if (not names or len(set(names)) != len(names) or beta.shape != (len(names),)
            or not np.isfinite(beta).all() or not np.isfinite(intercept)):
        raise ValueError("Invalid saved linear parameters.")
    if tuning.get("converged") is False:
        raise ValueError("Saved model did not converge.")
    return lambda X: X @ beta + intercept, names, beta


def verify_predictions(predict, window, names, record):
    """Check every saved test prediction, observation and month against local data."""
    expected = pd.DataFrame(record["predictions"])
    year, model = record["metrics"]["test_year"], record["metrics"]["specification"]
    if (expected.empty or expected.isna().any().any() or expected.row_position.duplicated().any()
            or not expected.test_year.eq(year).all() or not expected.specification.eq(model).all()):
        raise ValueError("Invalid saved prediction identities.")
    expected = expected.set_index("row_position")
    count, max_error = 0, 0.0
    for batch in window.batches(feature_columns=names):
        rows = expected.loc[batch.row_position]
        fitted = np.asarray(predict(batch.X))
        if (not np.array_equal(rows.return_month.to_numpy(), batch.return_month)
                or not np.allclose(rows.observed, batch.y, rtol=0, atol=1e-14)
                or fitted.shape != batch.y.shape
                or not np.allclose(rows.prediction, fitted, rtol=1e-10, atol=1e-10)):
            raise ValueError(f"Restored predictions/observations mismatch: {model}, {year}.")
        max_error = max(max_error, float(np.max(np.abs(rows.prediction.to_numpy() - fitted))))
        count += len(rows)
    if count != len(expected) or count != record["metrics"]["n_test"]:
        raise ValueError("Test observation counts differ.")
    return dict(model=model, year=year, n_test=count, max_absolute_error=max_error,
                prediction_atol=1e-10, prediction_rtol=1e-10)


def summarize_and_plot(annual, output, stamp):
    """Equal-weight annual means; preserve negatives and avoid silent clipping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if annual.duplicated(["model", "year", "predictor"]).any():
        raise ValueError("Duplicate annual importance records.")
    summaries = []
    for model, records in annual.groupby("model", sort=False):
        coverage = records.groupby("year").predictor.apply(set)
        if any(names != coverage.iloc[0] for names in coverage):
            raise ValueError("Predictor groups differ across years.")
        summary = records.groupby("predictor", sort=False).importance.mean().rename("mean_importance").reset_index()
        summary.insert(0, "model", model)
        summary["n_years"] = len(coverage)
        total = summary.mean_importance.sum()
        normalized = bool((summary.mean_importance >= 0).all() and total > 0)
        summary["relative_importance"] = summary.mean_importance / total if normalized else np.nan
        summary["normalization_status"] = "normalized" if normalized else "raw_only_negative_or_zero_total"
        summary = summary.sort_values(["mean_importance", "predictor"], ascending=[False, True])
        summaries.append(summary)
        value = "relative_importance" if normalized else "mean_importance"
        top = summary.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        ax.barh(top.predictor, top[value], color="#246B8E")
        ax.set(title=f"{model} | {records.year.min()}–{records.year.max()} ({len(coverage)} annual fits)",
               xlabel="Relative importance (all groups sum to 1)" if normalized else "Mean signed R² reduction (unnormalized)")
        ax.axvline(0, color="0.4", linewidth=.7)
        ax.grid(axis="x", alpha=.2)
        fig.suptitle("Figure 4 diagnostic: characteristic + interactions grouped\n"
                     f"Author grouping unverified | Generated {stamp}", fontsize=10)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{model.lower()}.{extension}", dpi=180)
        plt.close(fig)
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(output / "mean_importance.csv", index=False)
    return result


def make_report(experiments, data_dir, output_root, years, batch_size=10000):
    years = list(years)
    if not years or len(set(years)) != len(years):
        raise ValueError("Require unique test years.")
    panel = ParquetPanel(data_dir, batch_size=batch_size)
    local_identity = dataset_identity(panel.info)
    jobs, models = [], set()
    # Preflight all selected years before any large data reads or output creation.
    for experiment in experiments:
        folder = Path(experiment).resolve()
        if (folder / "annual").is_dir():
            folder = folder / "annual"
        experiment_model, common_groups = None, None
        for year in years:
            runs = sorted((folder / "years" / str(year)).glob("*/manifest.json"))
            if len(runs) != 1:
                raise ValueError(f"Expected one annual run for {folder}, {year}; found {len(runs)}.")
            run = runs[0].parent
            cfg = load_run_config(run)
            manifest = json.loads(runs[0].read_text(encoding="utf-8"))
            if manifest["dataset"] != local_identity:
                raise ValueError(f"Local data identity differs from saved fit: {run}")
            records = read_checkpoints(run, cfg, lazy=True)
            if len(records) != 1:
                raise ValueError("Require one model/year checkpoint per annual run.")
            key = next(iter(records))
            record = records[key]  # Existing reader verifies checkpoint SHA-256.
            if key[0] != year:
                raise ValueError("Checkpoint year differs from requested year.")
            predict, names, beta = restore_predictor(record["tuning"])
            if not set(names) <= set(panel.features):
                raise ValueError("Saved features are absent from the local panel.")
            groups = characteristic_groups(names)
            model = key[1]
            if experiment_model is not None and (model != experiment_model or set(groups) != common_groups):
                raise ValueError("Model or characteristic groups changed within an experiment.")
            experiment_model, common_groups = model, set(groups)
            first = cfg.RUN.SPLIT.FIRST_TRAIN_YEAR
            last = year - cfg.RUN.SPLIT.VALIDATION_YEARS - 1
            if first > last or cfg.RUN.SPLIT.VALIDATION_YEARS < 1:
                raise ValueError("Invalid saved training window.")
            jobs.append((run, cfg, year, model, first, last))
        if experiment_model in models:
            raise ValueError("Duplicate model experiments; select one source per model.")
        models.add(experiment_model)

    generated = datetime.now().astimezone()
    output = Path(output_root) / generated.strftime("%Y%m%dT%H%M%S%f%z") / "figure4"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="incomplete", generated_at=generated.isoformat(), years=years,
        dataset=local_identity, sources=[str(Path(p).resolve()) for p in experiments],
        grouping="base characteristic plus all macro interactions; SIC dummies jointly",
        author_grouping_verified=False, aggregation="equal-weight annual mean, then normalize all groups",
        negative_policy="preserve signed values; raw plot if any mean is negative or total is zero",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups={}, annual_sources=[])
    metadata_path = output / "methodology.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    annual, checks = [], []
    for run, cfg, year, model, first, last in jobs:
        started = perf_counter()
        record = read_checkpoints(run, cfg, lazy=True)[year, model]
        predict, names, beta = restore_predictor(record["tuning"])
        groups = characteristic_groups(names)
        metadata["groups"][model] = {group: [names[i] for i in indices] for group, indices in groups.items()}
        print(f"{model} {year}: verifying saved test predictions", flush=True)
        check = verify_predictions(predict, panel.window(year, year), names, record)
        window = panel.window(first, last)
        if window.n_rows != record["metrics"]["n_train"]:
            raise ValueError("Training sample count differs from the saved fit.")
        print(f"{model} {year}: importance on training {first}–{last} ({window.n_rows:,} rows)", flush=True)
        batches = ((batch.X, batch.y) for batch in window.batches(feature_columns=names))
        scores, diagnostics = streaming_zero_out(predict, batches, n_features=len(names),
                                                 groups=groups, coefficients=beta)
        if diagnostics["n_train"] != window.n_rows:
            raise ValueError("Incomplete training data stream.")
        scores.insert(0, "year", year)
        scores.insert(0, "model", model)
        annual.append(scores)
        check.update(diagnostics, first_train_year=first, last_train_year=last,
                     elapsed_seconds=perf_counter() - started)
        checks.append(check)
        metadata["annual_sources"].append(dict(model=model, year=year, run=str(run)))
        pd.concat(annual, ignore_index=True).to_csv(output / "annual_importance.csv", index=False)
        pd.DataFrame(checks).to_csv(output / "verification.csv", index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{model} {year}: done in {check['elapsed_seconds']:.1f}s", flush=True)
    summarize_and_plot(pd.concat(annual, ignore_index=True), output, generated.isoformat(timespec="seconds"))
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Figure 4 diagnostics: {output.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, nargs="+", help="One or more imported experiment folders.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/gkx_parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("results/replication_comparison"))
    parser.add_argument("--years", default="1987:2016", help="Single test year or inclusive FIRST:LAST.")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    bounds = args.years.split(":")
    first, last = int(bounds[0]), int(bounds[-1])
    if len(bounds) > 2 or first > last:
        parser.error("--years must be YEAR or FIRST:LAST with FIRST <= LAST")
    make_report(args.experiment, args.data_dir, args.output_root, range(first, last + 1), args.batch_size)
