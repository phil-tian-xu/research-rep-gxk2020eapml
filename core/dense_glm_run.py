"""Annual GLM+H pilot: squared group selection, then training-only Huber refinement."""

import numpy as np
import pandas as pd

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_query.parquet_dataset import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.losses import validate_huber_delta
from core.models import LinearRegressor
from core.models.glm import SplineBasis, fit_group_loss
from core.run import configured_loss
from core.run_state import read_checkpoints, resolve_run_dir, run_lock, save_checkpoint, validate_run


def diagnostics(result, groups):
    boundaries = np.r_[0, np.cumsum(groups)]
    return {"converged": bool(result.converged), "iterations": int(result.n_iter),
            "objective": float(result.objective), "stationarity": float(result.stationarity),
            "step_size": float(result.step_size),
            "nonzero_slopes": int(np.count_nonzero(result.coefficients)),
            "active_groups": sum(int(np.any(result.coefficients[a:b] != 0))
                                 for a, b in zip(boundaries[:-1], boundaries[1:]))}


def checked_fit(Z, y, groups, strength, *, label, **options):
    result = fit_group_loss(Z, y, groups, strength, **options)
    print(f"{label}: lambda={strength:.8g}; iterations={result.n_iter}; stationarity={result.stationarity:.3g}", flush=True)
    if not result.converged:
        raise RuntimeError(f"{label}: lambda={strength} did not converge; iterations={result.n_iter}, stationarity={result.stationarity}.")
    return result


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    name = "GLM_HUBER"
    settings, split = cfg.RUN.MODELS.GLM_HUBER, cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (list(cfg.RUN.MODELS.ENABLED) != [name] or checkpoint_dir is None or
            panel.info["name"] != cfg.DATA.DATASET or isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("GLM requires a matching dense Parquet dataset, checkpoint directory and valid annual windows.")
    if (settings.CENTERING != "training" or settings.INTERCEPT != "training_mean" or
            settings.TUNING != "squared_then_huber" or settings.DUMMY_PREFIX != "sic2_" or
            settings.OBJECTIVE != "mean_half_loss" or list(settings.KNOT_PROBABILITIES) != [0., .25, .5, .75]):
        raise ValueError("Unsupported GLM pilot convention; see the documented specification.")
    penalties = list(settings.PENALTY_GRID)
    if not penalties or not np.isfinite(penalties).all() or min(penalties) <= 0 or len(set(penalties)) != len(penalties):
        raise ValueError("GLM penalties must be finite, positive and unique.")
    quantile = settings.HUBER_QUANTILE
    if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("GLM Huber quantile must lie in (0,1).")
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    if validation_label != "MSE":
        raise ValueError("The GLM squared-selection pilot requires MSE validation.")
    solver = cfg.RUN.MODELS.LINEAR_SOLVER
    options = dict(max_iter=solver.MAX_ITER, tol=solver.TOL, initial_step=solver.INITIAL_STEP)
    features = list(panel.features)
    binary = [i for i, feature in enumerate(features) if feature.startswith(settings.DUMMY_PREFIX)]
    folder = resolve_run_dir(checkpoint_dir)

    def snapshot(stage, X=None, y=None):
        if memory_snapshot is not None:
            memory_snapshot(stage, array_bytes=None if X is None else X.nbytes + y.nbytes)

    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        saved = read_checkpoints(folder, cfg, lazy=True)
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                label = f"[{name} {year}]"
                if (year, name) in saved:
                    _check_saved(saved[year, name], panel.window(year, year), year, name)
                    print(f"{label} Already saved; skipping.", flush=True)
                    continue
                train_end = year - split.VALIDATION_YEARS - 1
                print(f"{label} Loading training {split.FIRST_TRAIN_YEAR}-{train_end}", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, train_end)
                n_train, intercept = len(y), float(y.mean())
                snapshot(f"{year}: raw training loaded", X, y)
                print(f"{label} Computing OLS threshold on original inputs", flush=True)
                ols = LinearRegressor().fit(X, y)
                delta = float(np.quantile(abs(ols.predict(X) - y), quantile, method="linear"))
                validate_huber_delta(delta)
                del ols
                basis = SplineBasis()
                Z = basis.fit_transform(X, binary)
                del X, positions, months
                target = y - intercept
                del y
                snapshot(f"{year}: expanded training loaded", Z, target)
                print(f"{label} {len(features)} groups, {Z.shape[1]} basis columns; Huber delta={delta:.8g}", flush=True)
                gram_stats = (Z.T @ Z / len(Z), Z.T @ target / len(Z), .5 * np.mean(target**2))
                fits = [checked_fit(Z, target, basis.group_sizes_, strength, label=label + " squared",
                                    gram_stats=gram_stats, **options) for strength in penalties]
                del Z, target, gram_stats
                X, y, positions, months = load_window(panel, year - split.VALIDATION_YEARS, year - 1)
                n_validation = len(y)
                Z = basis.transform(X)
                del X, positions, months
                snapshot(f"{year}: validation basis loaded", Z, y)
                records = []
                for strength, fit in zip(penalties, fits):
                    prediction = Z @ fit.coefficients + intercept
                    score = float(validation_loss(y, prediction))
                    if not np.isfinite(score) or not np.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite GLM validation results.")
                    records.append({"params": {"penalty_strength": strength}, "validation_loss": score,
                                    "stage": "squared", "huber_delta": delta, **diagnostics(fit, basis.group_sizes_)})
                selected = int(np.argmin([r["validation_loss"] for r in records]))
                strength, initial = penalties[selected], fits[selected].coefficients.copy()
                del Z, y, fits, fit, prediction
                # Reload only training data; never include validation in the robust fit.
                print(f"{label} Selected lambda={strength:.8g}; reloading training for Huber refinement", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, train_end)
                Z = basis.transform(X)
                del X, positions, months
                snapshot(f"{year}: Huber training basis loaded", Z, y)
                best = checked_fit(Z, y - intercept, basis.group_sizes_, strength,
                                   delta=delta, initial=initial, label=label + " Huber", **options)
                del Z, y, initial
                X, y, positions, months = load_window(panel, year, year)
                Z = basis.transform(X)
                del X
                prediction = Z @ best.coefficients + intercept
                del Z
                loss = float(test_loss(y, prediction))
                if not np.isfinite(prediction).all() or not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite GLM test results.")
                info = diagnostics(best, basis.group_sizes_)
                metrics = dict(specification=name, test_year=year, n_train=n_train, n_validation=n_validation,
                    n_test=len(y), validation_loss_name=validation_label,
                    best_validation_loss=records[selected]["validation_loss"], test_loss_name=test_label,
                    test_loss=loss, r2_zero=zero_benchmark_r2(y, prediction),
                    penalty_strength=strength, huber_delta=delta, **info)
                predictions = pd.DataFrame(dict(specification=name, test_year=year, row_position=positions,
                                                return_month=months, observed=y, prediction=prediction))
                tuning = dict(specification=name, test_year=year, best_params=records[selected]["params"],
                    validation_results=records, feature_names=features, coefficients=best.coefficients.tolist(),
                    intercept=intercept, spline_state=basis.state(), huber_delta=delta,
                    huber_threshold_rule="training_original_input_OLS_absolute_residual_quantile",
                    selection_stage="squared", final_stage="huber", objective_normalization="mean_half_loss",
                    nonzero_rule="any coefficient != 0 per original input group; intercept excluded", **info)
                save_checkpoint(folder, metrics, predictions, tuning)
                print(f"{label} Saved; active groups={info['active_groups']}; test R2={100 * metrics['r2_zero']:.6f}%", flush=True)
                del best, basis, y, positions, months, prediction, predictions
                snapshot(f"{year}: checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model={name}] [test_year={year}]; completed years remain resumable.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)
