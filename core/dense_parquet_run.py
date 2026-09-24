"""Annual PCR: complete float64 windows, original dense SVD, saved model-years."""

import numpy as np
import pandas as pd

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_examine.pcr_memory import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.run import configured_loss, configured_specifications
from core.run_state import read_checkpoints, resolve_run_dir, run_lock, save_checkpoint, validate_run


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    split = cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or
            split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("Invalid annual train/validation/test years.")
    if cfg.RUN.MODELS.ENABLED != ["PCR"]:
        raise ValueError("The complete-array Parquet workflow currently supports PCR only.")
    if checkpoint_dir is None or panel.info["name"] != cfg.DATA.DATASET:
        raise ValueError("Dense Parquet runs require an output directory and matching dataset identity.")
    _, factory, candidates = configured_specifications(cfg.RUN.MODELS)[0]
    for parameters in candidates:
        k = parameters["n_components"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("PCR grid must contain nonnegative integers.")
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    folder = resolve_run_dir(checkpoint_dir)

    def snapshot(stage, X=None, y=None):
        if memory_snapshot is not None:
            memory_snapshot(stage, array_bytes=None if X is None else X.nbytes + y.nbytes)

    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        completed = read_checkpoints(folder, cfg, lazy=True)
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                if (year, "PCR") in completed:
                    _check_saved(completed[year, "PCR"], panel.window(year, year), year, "PCR")
                    print(f"[PCR {year}] Already saved; skipping.", flush=True)
                    continue
                train_end = year - split.VALIDATION_YEARS - 1
                print(f"[PCR {year}] Loading complete training window {split.FIRST_TRAIN_YEAR}-{train_end}", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, train_end)
                n_train = len(y)
                del positions, months
                snapshot(f"{year}: training loaded", X, y)
                parameters = max(candidates, key=lambda p: p["n_components"])
                print(f"[PCR {year}] Fitting original dense PCR through K={parameters['n_components']}", flush=True)
                path = factory(**parameters).fit(X, y)
                snapshot(f"{year}: training fit complete")
                del X, y

                print(f"[PCR {year}] Loading validation {year - split.VALIDATION_YEARS}-{year - 1}", flush=True)
                X, y, positions, months = load_window(panel, year - split.VALIDATION_YEARS, year - 1)
                n_validation = len(y)
                snapshot(f"{year}: validation loaded", X, y)
                records, best, best_params, best_score = [], None, None, float("inf")
                for parameters in candidates:
                    model = path.with_components(parameters["n_components"])
                    score = float(validation_loss(y, model.predict(X)))
                    if not np.isfinite(score):
                        raise FloatingPointError(f"Nonfinite validation loss: PCR {year}, {parameters}")
                    records.append({"params": parameters.copy(), "validation_loss": score,
                                    "n_components": model.n_components_})
                    if score < best_score:  # Same first-candidate tie rule as the original runner.
                        best, best_params, best_score = model, parameters.copy(), score
                del X, y, positions, months, path, model
                snapshot(f"{year}: validation complete")

                X, y, positions, months = load_window(panel, year, year)
                snapshot(f"{year}: test loaded", X, y)
                prediction = best.predict(X)
                metrics = {"specification": "PCR", "test_year": year, "n_train": n_train,
                    "n_validation": n_validation, "n_test": len(y),
                    "validation_loss_name": validation_label, "best_validation_loss": best_score,
                    "test_loss_name": test_label, "test_loss": float(test_loss(y, prediction)),
                    "r2_zero": zero_benchmark_r2(y, prediction)}
                predictions = pd.DataFrame({"specification": "PCR", "test_year": year,
                    "row_position": positions, "return_month": months, "observed": y, "prediction": prediction})
                tuning = {"specification": "PCR", "test_year": year,
                          "best_params": best_params, "validation_results": records}
                save_checkpoint(folder, metrics, predictions, tuning)
                print(f"[PCR {year}] Saved; selected K={best_params['n_components']}; "
                      f"test R2={100 * metrics['r2_zero']:.6f}%", flush=True)
                del X, y, positions, months, prediction, predictions, best
                snapshot(f"{year}: checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model=PCR] [test_year={year}] Complete-array run; completed checkpoints can be resumed.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)
