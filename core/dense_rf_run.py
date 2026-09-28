"""Complete annual arrays; validation selects one RF without training/validation refit."""

import numpy as np
import pandas as pd
import sklearn

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_query.parquet_dataset import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.models.random_forest import cpu_budget, diagnostics, make_forest
from core.run import configured_loss
from core.run_state import read_checkpoints, resolve_run_dir, run_lock, save_checkpoint, validate_run


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    settings, split = cfg.RUN.MODELS.RF, cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (list(cfg.RUN.MODELS.ENABLED) != ["RF"] or checkpoint_dir is None or
            panel.info["name"] != cfg.DATA.DATASET or isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("RF requires matching data, checkpoints and valid annual windows.")
    depths, features = list(settings.DEPTH_GRID), list(settings.MAX_FEATURES_GRID)
    for label, values, upper in (("depth", depths, None), ("max_features", features, len(panel.features))):
        if (not values or len(set(values)) != len(values) or
                any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or
                    (upper is not None and v > upper) for v in values)):
            raise ValueError(f"Invalid RF {label} grid.")
    if isinstance(settings.N_TREES, bool) or not isinstance(settings.N_TREES, int) or settings.N_TREES < 1:
        raise ValueError("RF N_TREES must be a positive integer.")
    if (isinstance(settings.SEED, bool) or not isinstance(settings.SEED, int) or
            not 0 <= settings.SEED <= 2**32 - 1 - split.LAST_TEST_YEAR):
        raise ValueError("RF SEED plus test year must fit an unsigned 32-bit integer.")
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    if validation_label != "MSE":
        raise ValueError("RF pilot requires MSE validation.")
    workers = cpu_budget()
    folder = resolve_run_dir(checkpoint_dir)

    def snapshot(stage, X=None, y=None):
        if memory_snapshot is not None:
            memory_snapshot(stage, array_bytes=None if X is None else X.nbytes + y.nbytes)

    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        saved = read_checkpoints(folder, cfg, lazy=True)
        year = None
        params = None
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                label = f"[RF {year}]"
                if (year, "RF") in saved:
                    _check_saved(saved[year, "RF"], panel.window(year, year), year, "RF")
                    print(f"{label} Already saved; skipping.", flush=True)
                    continue
                train_end = year - split.VALIDATION_YEARS - 1
                print(f"{label} Loading training {split.FIRST_TRAIN_YEAR}-{train_end}", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, train_end)
                X = X.astype(np.float32)
                del positions, months
                n_train = len(y)
                snapshot(f"{year}: training loaded (float32 predictors)", X, y)
                V, vy, positions, months = load_window(panel, train_end + 1, year - 1)
                V = V.astype(np.float32)
                del positions, months
                n_validation = len(vy)
                if not all(np.isfinite(a).all() for a in (X, y, V, vy)):
                    raise ValueError("Nonfinite RF training/validation values after float32 conversion.")
                snapshot(f"{year}: training and validation loaded")
                candidates, best, best_score = [], None, np.inf
                for depth in depths:
                    for count in features:
                        params = dict(n_estimators=settings.N_TREES, max_depth=depth,
                                      max_features=count, random_state=settings.SEED + year)
                        print(f"{label} Fitting {params}; tree workers={workers}", flush=True)
                        model = make_forest(**params, n_jobs=workers).fit(X, y)
                        # Serial accumulation makes scores independent of thread completion order.
                        model.n_jobs = 1
                        prediction = model.predict(V)
                        score = float(validation_loss(vy, prediction))
                        if not np.isfinite(score) or not np.isfinite(prediction).all():
                            raise FloatingPointError("Nonfinite RF validation results.")
                        candidates.append(dict(params=params, validation_loss=score, **diagnostics(model)))
                        if score < best_score:  # First grid entry wins exact ties.
                            best, best_score = model, score
                        print(f"{label} Validation MSE={score:.10g}", flush=True)
                        del model, prediction
                selected = int(np.argmin([c["validation_loss"] for c in candidates]))
                params = candidates[selected]["params"]
                del X, y, V, vy
                snapshot(f"{year}: validation complete")
                X, y, positions, months = load_window(panel, year, year)
                X = X.astype(np.float32)
                if not np.isfinite(X).all() or not np.isfinite(y).all():
                    raise ValueError("Nonfinite RF test data.")
                prediction = best.predict(X)
                loss = float(test_loss(y, prediction))
                if not np.isfinite(loss) or not np.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite RF test results.")
                info = diagnostics(best)
                metrics = dict(specification="RF", test_year=year, n_train=n_train,
                    n_validation=n_validation, n_test=len(y), validation_loss_name=validation_label,
                    best_validation_loss=best_score, test_loss_name=test_label, test_loss=loss,
                    r2_zero=zero_benchmark_r2(y, prediction), average_tree_depth=info["average_tree_depth"],
                    max_depth=best.max_depth, max_features=best.max_features)
                predictions = pd.DataFrame(dict(specification="RF", test_year=year, row_position=positions,
                    return_month=months, observed=y, prediction=prediction))
                tuning = dict(specification="RF", test_year=year, best_params=candidates[selected]["params"],
                    validation_results=candidates, feature_names=list(panel.features),
                    sklearn_version=sklearn.__version__, fit_n_jobs=workers, prediction_n_jobs=1,
                    estimator_params=best.get_params(), predictor_dtype="float32", target_dtype="float64",
                    bootstrap_rows=n_train, feature_importances=best.feature_importances_.tolist(), **info)
                save_checkpoint(folder, metrics, predictions, tuning)
                print(f"{label} Saved; average depth={info['average_tree_depth']:.4f}; test R2={100 * metrics['r2_zero']:.6f}%", flush=True)
                del best, X, y, positions, months, prediction, predictions
                snapshot(f"{year}: checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model=RF] [test_year={year}] [candidate={params}]; completed years remain resumable.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)
