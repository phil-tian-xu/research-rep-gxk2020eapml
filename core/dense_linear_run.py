"""Complete-array annual Huber regressions; no train/validation refit."""

import numpy as np
import pandas as pd

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_query.parquet_dataset import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.losses import validate_huber_delta
from core.models import LinearRegressor
from core.run import configured_loss, configured_specifications
from core.run_state import read_checkpoints, resolve_run_dir, run_lock, save_checkpoint, validate_run


def fit_candidates(X, y, factory, candidates, *, label):
    """One training OLS threshold per window, shared by all penalty candidates."""
    candidates = [dict(p) for p in candidates]
    first = candidates[0]
    quantile = first.get("huber_quantile")
    if any((p.get("huber_quantile"), p.get("huber_delta"), p["fit_intercept"], p["standardize"]) !=
           (quantile, first.get("huber_delta"), first["fit_intercept"], first["standardize"])
           for p in candidates):
        raise ValueError("Candidates must share the Huber rule and feature transforms.")
    if quantile is not None:
        if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0 < quantile < 1 or first.get("huber_delta") is not None:
            raise ValueError("Supply a quantile in (0,1), without a fixed Huber delta.")
        print(f"{label} Computing training OLS residual quantile={quantile}", flush=True)
        ols = LinearRegressor(fit_intercept=first["fit_intercept"], standardize=first["standardize"]).fit(X, y)
        delta = float(np.quantile(np.abs(ols.predict(X) - y), quantile, method="linear"))
        del ols
    else:
        delta = first.get("huber_delta")
    validate_huber_delta(delta)
    print(f"{label} Huber delta={delta:.10g} (decimal return units)", flush=True)
    fitted = []
    for index, parameters in enumerate(candidates, 1):
        resolved = {**parameters, "huber_quantile": None, "huber_delta": delta}
        print(f"{label} Fitting candidate {index}/{len(candidates)}; lambda={parameters['penalty_strength']:.8g}", flush=True)
        try:
            model = factory(**resolved).fit(X, y)
            result = model.optimization_result_
            if not model.converged_:
                raise RuntimeError(f"Candidate did not converge: iterations={model.n_iter_}, "
                                   f"stationarity={result.stationarity:.6g}, tol={model.tol}.")
            if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
                raise FloatingPointError("Nonfinite fitted coefficients.")
            print(f"{label} Converged in {model.n_iter_} iterations; "
                  f"nonzero slopes={np.count_nonzero(model.coef_)}", flush=True)
            fitted.append(model)
        except Exception as error:
            error.add_note(f"{label} candidate={index}; parameters={resolved}")
            raise
    return fitted


def diagnostics(model):
    result = model.optimization_result_
    return {"huber_delta": model.huber_delta_, "converged": bool(model.converged_),
            "iterations": int(model.n_iter_), "objective": float(model.objective_),
            "stationarity": float(result.stationarity), "step_size": float(result.step_size),
            "nonzero_slopes": int(np.count_nonzero(model.coef_))}


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    split = cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or
            split.VALIDATION_YEARS < 1 or split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("Invalid annual train/validation/test years.")
    if list(cfg.RUN.MODELS.ENABLED) not in (["ENET_HUBER"], ["OLS3_HUBER"]):
        raise ValueError("Select one Huber linear specification.")
    if checkpoint_dir is None or panel.info["name"] != cfg.DATA.DATASET:
        raise ValueError("Dense linear runs require checkpoints and matching dataset identity.")
    name, factory, candidates = configured_specifications(cfg.RUN.MODELS)[0]
    features = (list(cfg.RUN.MODELS.OLS3_HUBER.FEATURES) if name == "OLS3_HUBER" else list(panel.features))
    if name == "OLS3_HUBER" and len(features) != 3:
        raise ValueError("OLS3_HUBER requires exactly three features.")
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    folder = resolve_run_dir(checkpoint_dir)

    def snapshot(stage, X=None, y=None):
        if memory_snapshot:
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
                print(f"{label} Loading training {split.FIRST_TRAIN_YEAR}-{train_end}; {len(features)} features", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, train_end, feature_columns=features)
                n_train = len(y)
                snapshot(f"{year}: training loaded", X, y)
                models = fit_candidates(X, y, factory, candidates, label=label)
                del X, y, positions, months
                snapshot(f"{year}: training fit complete")
                print(f"{label} Loading validation {year - split.VALIDATION_YEARS}-{year - 1}", flush=True)
                X, y, positions, months = load_window(panel, year - split.VALIDATION_YEARS, year - 1, feature_columns=features)
                n_validation = len(y)
                snapshot(f"{year}: validation loaded", X, y)
                records = []
                for parameters, model in zip(candidates, models):
                    prediction = model.predict(X)
                    score = float(validation_loss(y, prediction))
                    if not np.isfinite(prediction).all() or not np.isfinite(score):
                        raise FloatingPointError(f"{label} Nonfinite validation prediction/loss: {parameters}")
                    records.append({"params": parameters.copy(), "validation_loss": score, **diagnostics(model)})
                selected = int(np.argmin([r["validation_loss"] for r in records]))
                best, best_params = models[selected], candidates[selected].copy()
                best_score = records[selected]["validation_loss"]
                del X, y, positions, months, models, model, prediction
                snapshot(f"{year}: validation complete")
                X, y, positions, months = load_window(panel, year, year, feature_columns=features)
                snapshot(f"{year}: test loaded", X, y)
                prediction = best.predict(X)
                loss = float(test_loss(y, prediction))
                if not np.isfinite(prediction).all() or not np.isfinite(loss):
                    raise FloatingPointError(f"{label} Nonfinite test prediction/loss.")
                metrics = {"specification": name, "test_year": year, "n_train": n_train,
                           "n_validation": n_validation, "n_test": len(y),
                           "validation_loss_name": validation_label, "best_validation_loss": best_score,
                           "test_loss_name": test_label, "test_loss": loss,
                           "r2_zero": zero_benchmark_r2(y, prediction),
                           "penalty_strength": best_params["penalty_strength"], **diagnostics(best)}
                predictions = pd.DataFrame({"specification": name, "test_year": year,
                    "row_position": positions, "return_month": months, "observed": y, "prediction": prediction})
                tuning = {"specification": name, "test_year": year, "best_params": best_params,
                          "validation_results": records, "feature_names": features,
                          "coefficients": best.coef_.tolist(), "intercept": best.intercept_,
                          "huber_threshold_rule": "training_OLS_absolute_residual_quantile" if best_params.get("huber_quantile") else "fixed",
                          "nonzero_rule": "coef != 0; intercept excluded", **diagnostics(best)}
                save_checkpoint(folder, metrics, predictions, tuning)
                print(f"{label} Saved; lambda={best_params['penalty_strength']:.8g}; "
                      f"nonzero slopes={tuning['nonzero_slopes']}; test R2={100 * metrics['r2_zero']:.6f}%", flush=True)
                del X, y, positions, months, prediction, predictions, best
                snapshot(f"{year}: checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model={name}] [test_year={year}] Complete-array Huber run; saved years can be resumed.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)
