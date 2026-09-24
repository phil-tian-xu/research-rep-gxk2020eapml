"""Simple dense PCR experiment on the prepared 5% Parquet sample.

python -B -m core.data_examine.pcr_memory

Load the complete training window, fit the original PCRRegressor once, then
load validation and test windows for prediction. No batch fitting or QR summary.
Default K=10 is a fixed demonstration setting, not a tuned/paper parameter.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.data_examine.csv_memory import process_memory
from core.data_query.parquet_dataset import ParquetPanel
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.losses import mse_loss
from core.models import PCRRegressor
from core.run_state import atomic_write, resolve_run_dir
from utils.path_info import data_path


def load_window(panel, first_year, last_year):
    """Fill ordinary arrays from Parquet pieces; return the complete window."""
    window = panel.window(first_year, last_year)
    X = np.empty((window.n_rows, len(panel.features)), dtype=np.float64)
    y = np.empty(window.n_rows, dtype=np.float64)
    positions = np.empty(window.n_rows, dtype=np.int64)
    months = np.empty(window.n_rows, dtype="U7")
    start = 0
    for batch in window.batches():
        end = start + len(batch.y)
        X[start:end], y[start:end] = batch.X, batch.y
        positions[start:end], months[start:end] = batch.row_position, batch.return_month
        start = end
    if start != window.n_rows:
        raise ValueError("Loaded row count differs from the window metadata.")
    return X, y, positions, months


def experiment(parquet_dir, *, test_year=1987, components=10, first_train_year=1957,
               validation_years=12, output_dir="results/pcr_memory"):
    if (isinstance(components, bool) or not isinstance(components, int) or components < 0 or
            validation_years < 1 or first_train_year >= test_year - validation_years):
        raise ValueError("Invalid component count or train/validation/test years.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = resolve_run_dir(output_dir) / stamp
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "parquet_dir": str(Path(parquet_dir).resolve()),
              "model": "PCR", "n_components": components, "fit_intercept": True,
              "standardize": False, "tuning": False, "first_train_year": first_train_year,
              "test_year": test_year, "validation_years": validation_years,
              "memory_stages": [], "windows": {}}
    started = perf_counter()

    def snapshot(label):
        counters = process_memory()
        report["memory_stages"].append({"stage": label, "bytes": counters})
        if counters:
            print(f"{label}: Python RAM={counters['resident_bytes'] / 1024**3:.3f} GiB; "
                  f"MAX SO FAR={counters['peak_resident_bytes'] / 1024**3:.3f} GiB", flush=True)
        else:
            print(f"{label}: process RAM counters unavailable on this OS.", flush=True)
        return counters

    try:
        print(f"Simple PCR experiment: fixed K={components}, test year={test_year}\nOutput: {output}", flush=True)
        panel = ParquetPanel(parquet_dir)
        report["dataset"] = {key: panel.info[key] for key in ("name", "panel_sha256", "metadata_sha256")}
        snapshot("Before loading training data")
        train_end = test_year - validation_years - 1
        print(f"Loading complete training sample: {first_train_year}-{train_end}", flush=True)
        X_train, y_train, positions, months = load_window(panel, first_train_year, train_end)
        del positions, months
        train_bytes = X_train.nbytes + y_train.nbytes
        report["windows"]["training"] = {"first_year": first_train_year, "last_year": train_end,
            "n_rows": len(y_train), "n_features": X_train.shape[1], "dtype": str(X_train.dtype),
            "X_bytes": X_train.nbytes, "y_bytes": y_train.nbytes, "X_y_bytes": train_bytes}
        print(f"TRAINING DATA: X shape={X_train.shape}, dtype={X_train.dtype}\n"
              f"X={X_train.nbytes / 1024**3:.6f} GiB; y={y_train.nbytes / 1024**2:.3f} MiB; "
              f"X+y={train_bytes / 1024**3:.6f} GiB", flush=True)
        snapshot("After loading complete training arrays")

        # The original, full-array estimator. This experiment does not import
        # core.batch_run or core.models.batched and does not tune/refit on validation.
        model = PCRRegressor(n_components=components, fit_intercept=True, standardize=False)
        print("Fitting original PCR on the complete training array...", flush=True)
        fit_started = perf_counter()
        model.fit(X_train, y_train)
        report["fit_seconds"] = perf_counter() - fit_started
        snapshot("After PCR fit")
        report["training_mse"] = mse_loss(y_train, model.predict(X_train))
        # The fitted model retains coefficients, not the training array.
        del X_train, y_train

        for name, first, last in (("validation", test_year - validation_years, test_year - 1),
                                  ("test", test_year, test_year)):
            print(f"Loading complete {name} sample: {first}-{last}", flush=True)
            X, y, positions, months = load_window(panel, first, last)
            snapshot(f"After loading {name} arrays")
            prediction = model.predict(X)
            metrics = {"first_year": first, "last_year": last, "n_rows": len(y),
                       "X_bytes": X.nbytes, "y_bytes": y.nbytes,
                       "mse": mse_loss(y, prediction), "r2_zero": zero_benchmark_r2(y, prediction)}
            if not np.isfinite(metrics["r2_zero"]):
                metrics["r2_zero"] = None
            report["windows"][name] = metrics
            if name == "test":
                pd.DataFrame({"row_position": positions, "return_month": months,
                              "observed": y, "prediction": prediction}).to_csv(output / "test_predictions.csv", index=False)
            print(f"{name}: n={len(y):,}, MSE={metrics['mse']:.9g}, R2={metrics['r2_zero']}", flush=True)
            snapshot(f"After {name} prediction")
            del X, y, positions, months, prediction
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        counters = snapshot("Final measurement")
        report["peak_resident_gib"] = counters["peak_resident_bytes"] / 1024**3 if counters else None
        atomic_write(output / "report.json", json.dumps(report, indent=2, allow_nan=False))
        if counters:
            print(f"MAXIMUM OBSERVED PYTHON RAM: {report['peak_resident_gib']:.3f} GiB", flush=True)
        print(f"Report: {output / 'report.json'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=data_path / "processed" / "gkx_dev_p05_seed2020_parquet")
    parser.add_argument("--test-year", type=int, default=1987)
    parser.add_argument("--components", type=int, default=10, help="Fixed PCR K; no hyperparameter search.")
    parser.add_argument("--output-dir", default="results/pcr_memory")
    args = parser.parse_args()
    experiment(args.parquet_dir, test_year=args.test_year, components=args.components, output_dir=args.output_dir)
