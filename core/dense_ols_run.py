"""Squared-error OLS benchmarks using the existing annual checkpoint protocol."""

import numpy as np
import pandas as pd

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_query.parquet_dataset import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.models import LinearRegressor
from core.models.saved_fit import linear_state
from core.run import configured_loss
from core.run_state import (read_checkpoints, resolve_run_dir, run_lock,
                            save_checkpoint, validate_run)


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    split = cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    names = list(cfg.RUN.MODELS.ENABLED)
    if (names not in (["OLS"], ["OLS3"]) or checkpoint_dir is None or
            panel.info["name"] != cfg.DATA.DATASET or isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("OLS requires one benchmark, matching data, checkpoints and valid annual windows.")
    name = names[0]
    features = list(cfg.RUN.MODELS.OLS3.FEATURES) if name == "OLS3" else list(panel.features)
    if (not features or len(set(features)) != len(features) or not set(features) <= set(panel.features) or
            (name == "OLS3" and len(features) != 3)):
        raise ValueError("Select existing unique predictors; OLS3 requires exactly three.")
    params = dict(loss="squared", penalty_strength=0., fit_intercept=cfg.RUN.MODELS.FIT_INTERCEPT,
                  standardize=cfg.RUN.MODELS.STANDARDIZE)
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    folder = resolve_run_dir(checkpoint_dir)
    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        saved = read_checkpoints(folder, cfg, lazy=True)
        year = None
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                label = f"[{name} {year}]"
                if (year, name) in saved:
                    _check_saved(saved[year, name], panel.window(year, year), year, name)
                    print(f"{label} Already saved; skipping.", flush=True)
                    continue
                end = year - split.VALIDATION_YEARS - 1
                print(f"{label} Training {split.FIRST_TRAIN_YEAR}-{end}; {len(features)} predictors", flush=True)
                X, y, _, _ = load_window(panel, split.FIRST_TRAIN_YEAR, end, feature_columns=features)
                n_train = len(y)
                if memory_snapshot:
                    memory_snapshot(f"{year}: OLS training loaded", array_bytes=X.nbytes + y.nbytes)
                model = LinearRegressor(**params).fit(X, y)
                del X, y
                V, vy, _, _ = load_window(panel, end + 1, year - 1, feature_columns=features)
                n_validation = len(vy)
                score = float(validation_loss(vy, model.predict(V)))
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite OLS validation loss.")
                del V, vy
                X, y, positions, months = load_window(panel, year, year, feature_columns=features)
                prediction = model.predict(X)
                loss = float(test_loss(y, prediction))
                if not np.isfinite(prediction).all() or not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite OLS test prediction/loss.")
                metrics = dict(specification=name, test_year=year, n_train=n_train,
                    n_validation=n_validation, n_test=len(y), validation_loss_name=validation_label,
                    best_validation_loss=score, test_loss_name=test_label, test_loss=loss,
                    r2_zero=zero_benchmark_r2(y, prediction))
                predictions = pd.DataFrame(dict(specification=name, test_year=year,
                    row_position=positions, return_month=months, observed=y, prediction=prediction))
                tuning = dict(specification=name, test_year=year, best_params=params,
                    validation_results=[dict(params=params, validation_loss=score)],
                    objective="mean_squared_error", return_units="decimal", rank=model.rank_,
                    singular_values=model.singular_values_.tolist(), **linear_state(model, features))
                save_checkpoint(folder, metrics, predictions, tuning)
                print(f"{label} Saved; test R2={100 * metrics['r2_zero']:.6f}%", flush=True)
                del X, y, prediction, model, predictions, tuning
                if memory_snapshot:
                    memory_snapshot(f"{year}: OLS checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model={name}] [test_year={year}]; completed annual checkpoints can resume.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial OLS reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)
