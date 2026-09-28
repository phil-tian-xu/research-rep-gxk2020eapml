"""Separate OLS-3 Huber threshold sensitivity experiment; no model selection.

python -B -m core.data_examine.ols3_thresholds --sample full --years 1987 1988 2011 2012
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from core.data_query.parquet_dataset import ParquetPanel, load_window
from core.losses import mse_loss, validate_huber_delta
from core.models import LinearRegressor
from utils.path_info import data_path, root_path


FEATURES = ["mvel1", "bm", "mom12m"]
QUANTILE = 0.999
BASELINE = "ols_residual_q999"
VARIANTS = [BASELINE, "absolute_return_q999", "demeaned_return_q999", "ols"]
SETTINGS = dict(fit_intercept=True, standardize=False, penalty_strength=0.0,
                solver="apg", max_iter=50000, tol=1e-8, initial_step=1.0)


def fit_variants(X, y):
    """Thresholds use this training window only; reuse the unchanged estimator."""
    ols = LinearRegressor(**SETTINGS).fit(X, y)
    thresholds = dict(zip(VARIANTS[:3], [
        float(np.quantile(values, QUANTILE, method="linear")) for values in
        (np.abs(y - ols.predict(X)), np.abs(y), np.abs(y - y.mean()))
    ]))
    models, records = {}, []
    for name in VARIANTS:
        started = perf_counter()
        delta = thresholds.get(name)
        if delta is not None:
            validate_huber_delta(delta)
            model = LinearRegressor(loss="huber", huber_delta=delta, **SETTINGS).fit(X, y)
        else:
            model = ols
        if not model.converged_:
            raise RuntimeError(f"{name} did not converge after {model.n_iter_} iterations.")
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
            raise FloatingPointError(f"Nonfinite coefficients: {name}")
        residual = y - model.predict(X)
        result = model.optimization_result_
        records.append({
            "variant": name, "threshold": delta, "n_train": len(y),
            "training_mse": float(np.mean(residual**2)),
            "training_tail_fraction": None if delta is None else float(np.mean(np.abs(residual) > delta)),
            "converged": bool(model.converged_), "iterations": int(model.n_iter_),
            "objective": float(model.objective_),
            "stationarity": None if result is None else float(result.stationarity),
            "step_size": None if result is None else float(result.step_size),
            "huber_fit_seconds": None if delta is None else perf_counter() - started,
            "intercept": float(model.intercept_),
            **{f"coef_{feature}": float(value) for feature, value in zip(FEATURES, model.coef_)},
        })
        models[name] = model
    return models, records


def write_reports(records, output):
    """Pool sums of squared errors, never average annual R-squared values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    annual = pd.DataFrame(records)
    annual.to_csv(output / "diagnostics" / "annual.csv", index=False)
    pooled = annual.groupby("variant", sort=False)[["test_sse", "test_sum_y2", "n_test"]].sum()
    pooled["test_mse"] = pooled.test_sse / pooled.n_test
    pooled["r2_percent"] = 100 * (1 - pooled.test_sse / pooled.test_sum_y2)
    pooled["difference_from_baseline_pp"] = pooled.r2_percent - pooled.loc[BASELINE, "r2_percent"]
    pooled["first_test_year"] = int(annual.test_year.min())
    pooled["last_test_year"] = int(annual.test_year.max())
    pooled["number_of_test_years"] = annual.test_year.nunique()
    pooled.to_csv(output / "table1" / "comparison.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(pooled.index, pooled.r2_percent)
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(ylabel="Pooled test R² (%)", title="OLS-3 threshold sensitivity: All stocks")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(output / "table1" / "comparison.png", dpi=160)
    plt.close(fig)
    for column, label in [("threshold", "Threshold (decimal return units)"),
                          ("training_tail_fraction", "Training fraction beyond fitted threshold"),
                          ("test_r2_percent", "Annual test R² (%)")]:
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, group in annual.groupby("variant", sort=False):
            if group[column].notna().any():
                ax.plot(group.test_year, group[column], marker="o", label=name)
        ax.set(xlabel="Test year", ylabel=label, title="OLS-3 threshold sensitivity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "diagnostics" / f"{column}.png", dpi=160)
        plt.close(fig)


def experiment(parquet_dir, years, output_dir="results/huber_threshold_sensitivity"):
    years = sorted(years)
    if not years or len(years) != len(set(years)) or any(y < 1987 or y > 2016 for y in years):
        raise ValueError("Supply unique test years within 1987-2016.")
    panel = ParquetPanel(parquet_dir)
    required = set(range(1957, max(years) + 1))
    if not required <= {int(y) for y in panel.metadata["year_counts"]}:
        raise ValueError("Panel is missing required training/validation/test years.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() / stamp
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("annual", "table1", "diagnostics"):
        (output / folder).mkdir()
    sources = ["core/data_examine/ols3_thresholds.py", "core/models/linear.py",
               "core/models/_helpers.py", "core/model_inputs.py", "core/losses.py",
               "core/optimizers.py", "core/data_query/parquet_dataset.py"]
    manifest = {
        "status": "running", "created_utc": stamp, "test_years": years, "completed_years": [],
        "features": FEATURES, "quantile": QUANTILE, "quantile_method": "linear",
        "variants": VARIANTS, "settings": SETTINGS, "first_train_year": 1957,
        "validation_years": 12, "return_units": "decimal", "selection": "none; report every variant",
        "dataset": panel.info, "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib")},
        "sources_sha256": {p: hashlib.sha256((Path(root_path) / p).read_bytes()).hexdigest() for p in sources},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    all_records = []
    started = perf_counter()
    try:
        for year in years:
            print(f"[{year}] Training 1957-{year - 13}; validation {year - 12}-{year - 1}", flush=True)
            X, y, positions, months = load_window(panel, 1957, year - 13, feature_columns=FEATURES)
            models, records = fit_variants(X, y)
            del X, y, positions, months
            folder = output / "annual" / str(year)
            folder.mkdir()
            for split, first, last in [("validation", year - 12, year - 1), ("test", year, year)]:
                X, y, positions, months = load_window(panel, first, last, feature_columns=FEATURES)
                predictions = {name: model.predict(X) for name, model in models.items()}
                if not all(np.isfinite(p).all() for p in predictions.values()):
                    raise FloatingPointError(f"Nonfinite {split} prediction in {year}.")
                sum_y2 = float(y @ y)
                if sum_y2 <= 0:
                    raise ValueError(f"Zero return benchmark denominator in {year} {split}.")
                for record in records:
                    prediction = predictions[record["variant"]]
                    residual = y - prediction
                    difference = prediction - predictions[BASELINE]
                    sse = float(residual @ residual)
                    record.update({"test_year": year, f"n_{split}": len(y),
                                   f"{split}_mse": mse_loss(y, prediction),
                                   f"{split}_sse": sse, f"{split}_sum_y2": sum_y2,
                                   f"{split}_r2_percent": 100 * (1 - sse / sum_y2),
                                   f"{split}_prediction_rmse_vs_baseline": float(np.sqrt(np.mean(difference**2))),
                                   f"{split}_prediction_max_abs_diff": float(np.max(np.abs(difference)))})
                np.savez_compressed(folder / f"{split}_predictions.npz", observed=y,
                                    row_position=positions, return_month=months, **predictions)
                del X, y, positions, months, predictions, prediction, residual, difference
            (folder / "fits.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
            all_records.extend(records)
            manifest["completed_years"].append(year)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            print(f"[{year}] Saved all four variants", flush=True)
        write_reports(all_records, output)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["elapsed_seconds"] = perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Complete: {output / 'table1' / 'comparison.csv'}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=["full", "p05"], default="full")
    parser.add_argument("--years", nargs="+", default=["1987", "1988", "2011", "2012"],
                        help="Explicit years, or one inclusive range such as 1987:2016.")
    parser.add_argument("--output-dir", default="results/huber_threshold_sensitivity")
    args = parser.parse_args(argv)
    try:
        if len(args.years) == 1 and ":" in args.years[0]:
            first, last = map(int, args.years[0].split(":"))
            years = list(range(first, last + 1))
        else:
            years = [int(y) for y in args.years]
        if not years or len(years) != len(set(years)) or any(y < 1987 or y > 2016 for y in years):
            raise ValueError
    except ValueError:
        parser.error("Use unique years within 1987-2016, or an ascending FIRST:LAST range.")
    source = "gkx_parquet" if args.sample == "full" else "gkx_dev_p05_seed2020_parquet"
    experiment(data_path / "processed" / source, years, args.output_dir)


if __name__ == "__main__":
    main()
