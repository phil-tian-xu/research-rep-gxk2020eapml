"""OLS-3 + Huber Table 1 size groups from saved forecasts; optional window check.

python -B -m core.data_examine.ols3_size_groups --experiment PATH
Add --refit-train-validation to fit one additional full-universe model per year.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.combined_experiments import resolve_annual_run, validate_scores
from core.data_query.parquet_dataset import ParquetPanel, load_window
from core.dense_linear_run import diagnostics, fit_candidates
from core.evaluation.prediction_metrics import stock_prediction_summary, zero_benchmark_r2
from core.models import LinearRegressor
from core.run_state import dataset_identity, load_run_config, read_checkpoint, runtime_identity


FEATURES = ["mvel1", "bm", "mom12m"]
# GKX Table 1, printed p. 2250, OLS-3 + Huber; percentages, not decimal R2.
PAPER_R2_PERCENT = {"all": .16, "top": .31, "bottom": .17}


def read_sizes(path, years, chunksize=100_000):
    """Read raw, already-lagged size; reproduce its pre-target-exclusion ranks.

    Raw missing/nonpositive sizes are retained for an explicit exclusion audit.
    No absolute-value correction, CRSP substitution or size imputation is used
    for selecting stocks. ranked_size only verifies the source against the panel.
    """
    frames = []
    for chunk in pd.read_csv(path, usecols=["permno", "DATE", "mvel1"], chunksize=chunksize):
        months = pd.to_datetime(chunk.DATE.astype(str), format="%Y%m%d", errors="raise").dt.to_period("M")
        selected = months.dt.year.isin(years)
        if selected.any():
            frame = chunk.loc[selected, ["permno", "mvel1"]].rename(
                columns={"permno": "stock_id", "mvel1": "market_equity"})
            frame["return_month"] = months[selected]
            frames.append(frame)
    if not frames:
        raise ValueError("No raw size observations in the requested years.")
    sizes = pd.concat(frames, ignore_index=True)
    identifiers = pd.to_numeric(sizes.stock_id, errors="raise")
    if (identifiers.isna().any() or (identifiers <= 0).any()
            or (identifiers != np.floor(identifiers)).any()):
        raise ValueError("Raw size requires positive integer PERMNOs.")
    sizes["stock_id"] = identifiers.astype("int64")
    sizes["market_equity"] = pd.to_numeric(sizes.market_equity, errors="raise")
    if np.isinf(sizes.market_equity).any() or sizes.duplicated(["stock_id", "return_month"]).any():
        raise ValueError("Infinite raw size or duplicate security-month keys.")
    expected = {month for year in years for month in pd.period_range(f"{year}-01", f"{year}-12", freq="M")}
    if set(sizes.return_month) != expected:
        raise ValueError("Raw size does not cover every requested test month.")
    groups = sizes.groupby("return_month", sort=False)
    filled = sizes.market_equity.fillna(groups.market_equity.transform("median")).fillna(0.)
    counts = groups.stock_id.transform("size")
    ranks = filled.groupby(sizes.return_month).rank(method="average")
    sizes["ranked_size"] = (2 * (ranks - 1) / (counts - 1) - 1).where(counts > 1, 0.)
    return sizes


def load_test_frame(panel, year, record, sizes):
    """Verify positions, targets, saved coefficients, source ranks and size keys."""
    validate_scores(record)
    tuning = record["tuning"]
    if tuning["feature_names"] != FEATURES or not tuning["converged"]:
        raise ValueError("Expected a converged OLS-3 fit on mvel1, bm, mom12m.")
    metadata = panel.metadata
    columns = ["permno", metadata["row_position_column"], metadata["return_month_column"],
               metadata["feature_month_column"], metadata["target_column"], *FEATURES]
    pieces = []
    window = panel.window(year, year)
    for part in window.parts:
        frame = pq.ParquetFile(panel.verify_part(part)).read(columns=columns, use_threads=False).to_pandas()
        if len(frame) != part["n_rows"]:
            raise ValueError("Parquet row count differs from its manifest.")
        pieces.append(frame)
    data = pd.concat(pieces, ignore_index=True).rename(columns={
        "permno": "stock_id", metadata["row_position_column"]: "row_position",
        metadata["return_month_column"]: "return_month", metadata["target_column"]: "target",
        metadata["feature_month_column"]: "feature_month"})
    identifiers = pd.to_numeric(data.stock_id, errors="raise")
    if identifiers.isna().any() or (identifiers <= 0).any() or (identifiers != np.floor(identifiers)).any():
        raise ValueError("Panel requires positive integer PERMNOs.")
    data["stock_id"] = identifiers.astype("int64")
    data["return_month"] = pd.PeriodIndex(data.return_month, freq="M")
    formation = pd.PeriodIndex(data.feature_month, freq="M")
    expected = set(pd.period_range(f"{year}-01", f"{year}-12", freq="M"))
    if (set(data.return_month) != expected or not (formation + 1 == data.return_month).all()
            or data.duplicated(["stock_id", "return_month"]).any()):
        raise ValueError("Invalid panel security keys or forecast dates.")
    prediction = pd.DataFrame(record["predictions"])
    prediction["return_month"] = pd.PeriodIndex(prediction.return_month, freq="M")
    data = data.merge(prediction[["row_position", "return_month", "observed", "prediction"]],
                      on=["row_position", "return_month"], how="left", validate="one_to_one")
    if (len(data) != len(prediction) or data[["observed", "prediction"]].isna().any().any()
            or not np.isfinite(data[["target", *FEATURES]].to_numpy()).all()
            or not np.allclose(data.target, data.observed, rtol=0, atol=1e-14)):
        raise ValueError("Saved predictions do not match the complete local test panel.")
    reconstructed = data[FEATURES].to_numpy() @ np.asarray(tuning["coefficients"]) + tuning["intercept"]
    if not np.allclose(reconstructed, data.prediction, rtol=1e-10, atol=1e-10):
        raise ValueError("Saved coefficients do not reproduce the test predictions.")
    data = data.merge(sizes, on=["stock_id", "return_month"], how="left", validate="one_to_one",
                      indicator=True)
    if not data._merge.eq("both").all():
        raise ValueError("Some test stocks have no raw characteristic row; this is not ordinary missing size.")
    if not np.allclose(data.ranked_size, data.mvel1, rtol=0, atol=1e-14):
        raise ValueError("Raw size ranks differ from the model panel; check the characteristic source.")
    return data.drop(columns="_merge")


def size_group_summary(frame, size_count=1000):
    """Keep all forecasts in All; select size groups only from valid raw size.

    Membership is monthly, among the model's existing test observations. Ties
    follow stock_prediction_summary: ascending numeric PERMNO in both groups.
    Fail on insufficient coverage or overlapping groups instead of shrinking N.
    """
    if isinstance(size_count, bool) or not isinstance(size_count, int) or size_count < 1:
        raise ValueError("size_count must be a positive integer.")
    if frame.duplicated(["stock_id", "return_month"]).any():
        raise ValueError("Duplicate test security-month keys.")
    valid = np.isfinite(frame.market_equity) & frame.market_equity.gt(0)
    eligible = frame.loc[valid]
    coverage = eligible.groupby("return_month").size().reindex(frame.return_month.unique(), fill_value=0)
    if (coverage < 2 * size_count).any():
        raise ValueError("Each test month needs at least twice size_count valid sizes.")
    top = eligible.sort_values(["return_month", "market_equity", "stock_id"],
                              ascending=[True, False, True]).groupby("return_month", sort=False).head(size_count)
    bottom = eligible.sort_values(["return_month", "market_equity", "stock_id"]).groupby(
        "return_month", sort=False).head(size_count)
    if not top.merge(bottom, on=["stock_id", "return_month"]).empty:
        raise ValueError("Size ties cause overlapping top/bottom groups; inspect this cross-section.")
    summary = stock_prediction_summary(eligible, size_count=size_count)
    observed, prediction = frame.observed.to_numpy(), frame.prediction.to_numpy()
    r2 = zero_benchmark_r2(observed, prediction)
    summary.loc["all"] = [len(frame), frame.return_month.nunique(),
                          float(np.sum((observed - prediction)**2)), float(observed @ observed), r2, 100 * r2]
    audit = frame.groupby("return_month").size().rename("n_test").to_frame()
    audit["n_size_eligible"] = coverage
    audit["n_missing_size"] = frame.market_equity.isna().groupby(frame.return_month).sum()
    audit["n_nonpositive_size"] = frame.market_equity.le(0).groupby(frame.return_month).sum()
    audit["top_count"] = top.groupby("return_month").size()
    audit["bottom_count"] = bottom.groupby("return_month").size()
    audit["top_cutoff"] = top.groupby("return_month").market_equity.min()
    audit["bottom_cutoff"] = bottom.groupby("return_month").market_equity.max()
    invalid = frame.loc[~valid, ["stock_id", "return_month", "row_position", "market_equity"]].copy()
    invalid["reason"] = np.where(invalid.market_equity.isna(), "missing_size", "nonpositive_size")
    return summary.reset_index(), audit.reset_index(), invalid


def make_report(experiment, data_dir, characteristics_csv, output_root, years,
                *, size_count=1000, refit_train_validation=False):
    years = list(years)
    if not years or years != sorted(set(years)):
        raise ValueError("Select distinct test years in increasing order.")
    experiment, characteristics_csv = Path(experiment).resolve(), Path(characteristics_csv).resolve()
    panel = ParquetPanel(data_dir)
    identity = dataset_identity(panel.info)
    stat = characteristics_csv.stat()
    source_stat = (stat.st_size, stat.st_mtime_ns)
    print("Reading raw market capitalization (not ranked/imputed model inputs)", flush=True)
    sizes = read_sizes(characteristics_csv, years)
    sizes_by_year = {year: rows for year, rows in sizes.groupby(sizes.return_month.dt.year, sort=False)}
    del sizes
    output = Path(output_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", experiment=str(experiment), dataset=identity, years=years,
        size_count=size_count, refit_train_validation=refit_train_validation,
        characteristics_csv=str(characteristics_csv), characteristics_bytes=source_stat[0],
        characteristics_mtime_ns=source_stat[1],
        size_definition="Raw mvel1, already lagged to forecast formation; verified against model-panel ranks",
        size_policy="Finite positive raw size only for Top/Bottom; All retains every saved test observation",
        universe="Existing test observations with observed returns; group membership reselected each month",
        tie_rule="Ascending numeric PERMNO in both groups; fail if groups overlap",
        evaluation="Pooled 1 - sum((y-prediction)^2)/sum(y^2); stock-month observations equally weighted",
        author_refit_rule_verified=False, annual_sources=[],
        runtime=runtime_identity(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    annual, audits, invalid_sizes, fits = [], [], [], []
    training_data = None
    metadata_path = output / "methodology.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    try:
        for year in years:
            run = resolve_annual_run(experiment, year)
            cfg = load_run_config(run)
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if manifest["dataset"] != identity or list(cfg.RUN.MODELS.ENABLED) != ["OLS3_HUBER"]:
                raise ValueError("Saved run must be OLS3_HUBER on the selected local dataset.")
            checkpoint = run / "years" / str(year) / "OLS3_HUBER.json"
            record = read_checkpoint(checkpoint, cfg)
            first = cfg.RUN.SPLIT.FIRST_TRAIN_YEAR
            train_end = year - cfg.RUN.SPLIT.VALIDATION_YEARS - 1
            if (record["metrics"]["n_train"] != panel.window(first, train_end).n_rows
                    or record["metrics"]["n_validation"] != panel.window(train_end+1, year-1).n_rows):
                raise ValueError("Saved fitting counts do not match the configured train/validation windows.")
            frame = load_test_frame(panel, year, record, sizes_by_year[year])
            summary, audit, invalid = size_group_summary(frame, size_count)
            summary.insert(0, "variant", "saved_training_only")
            summary.insert(1, "test_year", year)
            annual.append(summary)
            audits.append(audit)
            invalid_sizes.append(invalid)
            tuning = record["tuning"]
            fits.append(dict(variant="saved_training_only", test_year=year,
                first_train_year=first, last_train_year=train_end,
                n_train=record["metrics"]["n_train"], feature_names=FEATURES,
                coefficients=tuning["coefficients"], intercept=tuning["intercept"],
                huber_delta=tuning["huber_delta"]))
            if refit_train_validation:
                print(f"{year}: diagnostic refit on {first}-{year-1}, all training/validation stocks", flush=True)
                if training_data is None:
                    print("Loading the three predictors once for the optional annual refits", flush=True)
                    all_X, all_y, positions, months = load_window(panel, first, max(years)-1, feature_columns=FEATURES)
                    fitting_years = months.astype("datetime64[Y]").astype(int) + 1970
                    training_data = (first, all_X, all_y, fitting_years)
                    del positions, months
                if first != training_data[0]:
                    raise ValueError("Annual sources must use the same initial training year.")
                # The cache includes later observations, but each fit strictly excludes its test year and future years.
                selected = training_data[3] < year
                X, y = training_data[1][selected], training_data[2][selected]
                if len(y) != record["metrics"]["n_train"] + record["metrics"]["n_validation"]:
                    raise ValueError("Combined fitting count differs from saved train + validation counts.")
                model = fit_candidates(X, y, LinearRegressor, [tuning["best_params"]], label=f"[size check {year}]")[0]
                fits.append(dict(variant="refit_train_validation", test_year=year,
                    first_train_year=first, last_train_year=year-1, n_train=len(y), feature_names=FEATURES,
                    coefficients=model.coef_.tolist(), intercept=float(model.intercept_), **diagnostics(model)))
                del X, y
                frame = frame.assign(prediction=model.predict(frame[FEATURES].to_numpy()))
                alternative, _, _ = size_group_summary(frame, size_count)
                if not np.array_equal(alternative[["n_obs", "benchmark_sse"]], summary[["n_obs", "benchmark_sse"]]):
                    raise ValueError("Refit changed evaluation observations or size-group membership.")
                alternative.insert(0, "variant", "refit_train_validation")
                alternative.insert(1, "test_year", year)
                annual.append(alternative)
            metadata["annual_sources"].append(dict(year=year, run=str(run),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            pd.concat(annual, ignore_index=True).to_csv(output / "annual_metrics.csv", index=False)
            pd.concat(audits, ignore_index=True).to_csv(output / "monthly_size_audit.csv", index=False)
            pd.concat(invalid_sizes, ignore_index=True).to_csv(output / "invalid_sizes.csv", index=False)
            (output / "fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False), encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            print(f"{year}: verified forecasts; scored All, Top {size_count}, Bottom {size_count}", flush=True)
        stat = characteristics_csv.stat()
        if (stat.st_size, stat.st_mtime_ns) != source_stat:
            raise ValueError("Raw characteristic source changed during reporting.")
        table = pd.concat(annual, ignore_index=True).groupby(["variant", "sample"], sort=False)[
            ["n_obs", "n_months", "sse", "benchmark_sse"]].sum().reset_index()
        table["r2_zero"] = 1 - table.sse / table.benchmark_sse
        table["r2_percent"] = 100 * table.r2_zero
        comparable = years == list(range(1987, 2017)) and size_count == 1000 and identity["name"] == "gkx_parquet"
        table["paper_r2_percent"] = table["sample"].map(PAPER_R2_PERCENT) if comparable else np.nan
        table["difference_pp"] = table.r2_percent - table.paper_r2_percent
        table.to_csv(output / "comparison.csv", index=False)
        metadata["paper_comparable_horizon_and_group_count"] = comparable
        metadata["status"] = "complete"
        print(table.to_string(index=False))
        print(f"Saved: {output}")
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="Saved OLS-3 annual experiment folder.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/gkx_parquet"))
    parser.add_argument("--characteristics-csv", type=Path, default=Path("data/raw/datashare.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/ols3_size_groups"))
    parser.add_argument("--years", default="1987:2016", help="YEAR or inclusive FIRST:LAST.")
    parser.add_argument("--size-count", type=int, default=1000)
    parser.add_argument("--refit-train-validation", action="store_true")
    args = parser.parse_args()
    try:
        bounds = [int(value) for value in args.years.split(":")]
        if len(bounds) not in (1, 2) or bounds[0] > bounds[-1]:
            raise ValueError
        years = range(bounds[0], bounds[-1]+1)
        if args.size_count < 1:
            raise ValueError
    except ValueError:
        parser.error("Use YEAR or FIRST:LAST with FIRST <= LAST, and a positive --size-count.")
    make_report(args.experiment, args.data_dir, args.characteristics_csv, args.output_root, years,
                size_count=args.size_count, refit_train_validation=args.refit_train_validation)


if __name__ == "__main__":
    main()
