"""Shared annual model runs on an already prepared prediction panel.

Uses the existing CfgNode configuration; models themselves receive ordinary
constructor arguments. Every enabled specification is tuned and evaluated in
every test year. This module does not build financial features or targets.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.losses import huber_loss, mse_loss, validate_huber_delta
from core.model_selection import tune_hyperparameters
from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.model_inputs import validate_model_inputs
from core.splits import annual_splits
from core.run_state import (atomic_write, read_checkpoints, resolve_run_dir, run_lock,
                            save_checkpoint, validate_run)
from utils.path_info import root_path


@dataclass
class RunResult:
    annual_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    tuning_history: list[dict]
    config_yaml: str
    dataset_info: dict | None = None


def configured_loss(settings):
    """Translate a configured name/threshold to its display label and callable."""
    if settings.NAME == "mse":
        if settings.DELTA is not None:
            raise ValueError("DELTA is only used for Huber evaluation loss.")
        return "MSE", mse_loss
    if settings.NAME == "huber":
        validate_huber_delta(settings.DELTA)
        return f"Huber(delta={settings.DELTA:g})", partial(huber_loss, delta=settings.DELTA)
    raise ValueError(f"Unknown evaluation loss: {settings.NAME!r}; use 'mse' or 'huber'.")


def configured_specifications(settings):
    """Build separate candidate lists; no registry or cross-specification selection."""
    factories = {
        "OLS": LinearRegressor,
        "OLS_HUBER": LinearRegressor,
        "ENET_HUBER": LinearRegressor,
        "OLS3_HUBER": LinearRegressor,
        "PCR": PCRRegressor,
        "PLS": PLSRegressor,
    }
    if not settings.ENABLED or len(set(settings.ENABLED)) != len(settings.ENABLED):
        raise ValueError("MODELS.ENABLED must contain at least one name, without duplicates.")
    common = {"fit_intercept": settings.FIT_INTERCEPT, "standardize": settings.STANDARDIZE}
    solver = settings.LINEAR_SOLVER
    linear = {**common, "max_iter": solver.MAX_ITER, "tol": solver.TOL, "initial_step": solver.INITIAL_STEP}
    specifications = []
    for name in settings.ENABLED:
        if name not in factories:
            raise ValueError(f"Unknown model specification {name!r}; available: {list(factories)}.")
        if name == "OLS":
            candidates = [{**linear, "loss": "squared", "penalty_strength": 0.0}]
        elif name == "OLS_HUBER":
            quantile = settings.OLS_HUBER.get("HUBER_QUANTILE")
            if quantile is None:
                validate_huber_delta(settings.OLS_HUBER.HUBER_DELTA)
            elif (settings.OLS_HUBER.HUBER_DELTA is not None or isinstance(quantile, bool) or
                  not np.isfinite(quantile) or not 0 < quantile < 1):
                raise ValueError("OLS_HUBER requires either HUBER_DELTA or HUBER_QUANTILE in (0, 1).")
            candidates = [{**linear, "loss": "huber", "huber_delta": settings.OLS_HUBER.HUBER_DELTA,
                           "huber_quantile": quantile, "solver": settings.OLS_HUBER.get("SOLVER", "apg"),
                           "penalty_strength": 0.0}]
        elif name == "ENET_HUBER":
            enet = settings.ENET_HUBER
            quantile = enet.get("HUBER_QUANTILE")
            if quantile is None:
                validate_huber_delta(enet.HUBER_DELTA)
            elif (enet.HUBER_DELTA is not None or isinstance(quantile, bool) or
                  not np.isfinite(quantile) or not 0 < quantile < 1):
                raise ValueError("ENET_HUBER requires either HUBER_DELTA or HUBER_QUANTILE in (0, 1).")
            candidates = [{**linear, "loss": "huber", "huber_delta": enet.HUBER_DELTA,
                           "huber_quantile": quantile,
                           "rho": enet.RHO, "penalty_strength": strength} for strength in enet.PENALTY_GRID]
        elif name == "OLS3_HUBER":
            candidates = [{**linear, "loss": "huber", "huber_quantile": settings.OLS3_HUBER.HUBER_QUANTILE,
                           "penalty_strength": 0.0}]
        elif name == "PCR":
            candidates = [{**common, "n_components": k} for k in settings.PCR.COMPONENT_GRID]
        else:
            candidates = [{**common, "n_components": k, "tol": settings.PLS.TOL}
                          for k in settings.PLS.COMPONENT_GRID]
        if not candidates:
            raise ValueError(f"Supply an explicit hyperparameter grid for {name}.")
        specifications.append((name, factories[name], candidates))
    return specifications


def run(X, y, return_month, cfg, *, dataset_info=None, checkpoint_dir=None, stop_after_year=None):
    """Run in memory, or resume an initialized directory of model-year results."""
    if "OLS3_HUBER" in cfg.RUN.MODELS.ENABLED:
        raise ValueError("OLS3_HUBER requires the dense Parquet runner's explicit feature selection.")
    window = cfg.RUN.SPLIT
    if stop_after_year is None:
        stop_after_year = window.LAST_TEST_YEAR
    if (isinstance(stop_after_year, bool) or not isinstance(stop_after_year, int) or
            not window.FIRST_TEST_YEAR <= stop_after_year <= window.LAST_TEST_YEAR):
        raise ValueError("stop_after_year must be within the configured test years.")
    if checkpoint_dir is None:
        return _run(X, y, return_month, cfg, dataset_info=dataset_info,
                    stop_after_year=stop_after_year)
    folder = resolve_run_dir(checkpoint_dir)
    with run_lock(folder):
        validate_run(folder, cfg, dataset_info)
        completed = read_checkpoints(folder, cfg)
        try:
            _run(X, y, return_month, cfg, dataset_info=dataset_info,
                 checkpoint_dir=folder, completed=completed, stop_after_year=stop_after_year)
        except BaseException as error:
            # Preserve completed work and publish partial reports even on Ctrl+C.
            try:
                rebuild_reports(folder, cfg, dataset_info)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_reports(folder, cfg, dataset_info)


def _run(X, y, return_month, cfg, *, dataset_info=None, checkpoint_dir=None,
         completed=None, stop_after_year=None):
    """Fit and tune each specification annually, then pool its test predictions.

    Inputs follow the estimators' row/column contract. return_month labels the
    realized target month and must correspond positionally to X and y. If X and
    dates are both pandas objects with row labels, their indices must agree.
    No fitting uses validation/test observations. Fitted estimators are released
    between runs; predictions and all candidate validation scores are retained.
    """
    array, target, _ = validate_model_inputs(X, y)
    if dataset_info is not None and dataset_info.get("name") != cfg.DATA.DATASET:
        raise ValueError("Loaded dataset identity does not match DATA.DATASET in the run config.")
    if isinstance(X, pd.DataFrame) and isinstance(return_month, pd.Series):
        if not X.index.equals(return_month.index):
            raise ValueError("X and return_month indices must match in the same order.")
    dates = np.asarray(return_month)
    if dates.ndim != 1 or len(dates) != len(target):
        raise ValueError("return_month must have one date per observation.")
    # Preserve DataFrame schemas at the estimator boundary, otherwise use arrays.
    features = X if isinstance(X, pd.DataFrame) else array
    settings = cfg.RUN
    specifications = configured_specifications(settings.MODELS)
    validation_loss_name, validation_loss_fn = configured_loss(settings.EVALUATION.VALIDATION)
    test_loss_name, test_loss_fn = configured_loss(settings.EVALUATION.TEST)
    window = settings.SPLIT
    predictions, annual_metrics, tuning_history = [], [], []
    completed = {} if completed is None else completed

    for split in annual_splits(return_month, first_train_year=window.FIRST_TRAIN_YEAR,
                               first_test_year=window.FIRST_TEST_YEAR,
                               last_test_year=stop_after_year,
                               validation_years=window.VALIDATION_YEARS):
        train, validation, test = split.train_indices, split.validation_indices, split.test_indices
        inputs_ready = False
        for name, factory, candidates in specifications:
            if (split.test_year, name) in completed:
                record = completed[split.test_year, name]
                saved = pd.DataFrame(record["predictions"])
                if (len(saved) != len(test) or
                        not np.array_equal(saved["row_position"].to_numpy(), test) or
                        not np.array_equal(saved["observed"].to_numpy(), target[test]) or
                        not np.array_equal(saved["return_month"].to_numpy(), dates[test].astype(str)) or
                        not np.isfinite(saved["prediction"].to_numpy()).all()):
                    raise ValueError(f"Checkpoint observations differ: model={name}, test_year={split.test_year}")
                annual_metrics.append(record["metrics"])
                predictions.append(saved)
                tuning_history.append(record["tuning"])
                print(f"[model={name}] [test_year={split.test_year}] Already saved; skipping.", flush=True)
                continue
            if not inputs_ready:
                if isinstance(features, pd.DataFrame):
                    X_train, X_validation, X_test = features.iloc[train], features.iloc[validation], features.iloc[test]
                else:
                    X_train, X_validation, X_test = features[train], features[validation], features[test]
                inputs_ready = True
            tuning_result = tune_hyperparameters(
                factory, candidates, X_train, target[train], X_validation, target[validation],
                validation_loss_fn=validation_loss_fn,
                model_name=name, test_year=split.test_year,
            )
            tuned_model = tuning_result.best_estimator
            try:
                prediction = tuned_model.predict(X_test)
                observed = target[test]
                test_loss = test_loss_fn(observed, prediction)
            except Exception as error:
                error.add_note(f"[model={name}] [test_year={split.test_year}] "
                               f"stage=test prediction/scoring; parameters={tuning_result.best_params}")
                raise
            annual_metrics.append({
                "specification": name, "test_year": split.test_year,
                "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
                "validation_loss_name": validation_loss_name,
                "best_validation_loss": min(r["validation_loss"] for r in tuning_result.validation_results),
                "test_loss_name": test_loss_name, "test_loss": test_loss,
                "r2_zero": zero_benchmark_r2(observed, prediction),
            })
            predictions.append(pd.DataFrame({
                "specification": name, "test_year": split.test_year,
                "row_position": test, "return_month": dates[test].astype(str),
                "observed": observed, "prediction": prediction,
            }))
            tuning_history.append({
                "specification": name, "test_year": split.test_year,
                "best_params": tuning_result.best_params,
                "validation_results": tuning_result.validation_results,
            })
            if checkpoint_dir is not None:
                save_checkpoint(checkpoint_dir, annual_metrics[-1], predictions[-1], tuning_history[-1])
                print(f"[model={name}] [test_year={split.test_year}] Saved.", flush=True)
            del tuning_result, tuned_model

    return assemble_result(annual_metrics, predictions, tuning_history, cfg, dataset_info)


def assemble_result(annual_metrics, predictions, tuning_history, cfg, dataset_info):
    predictions = (pd.concat(predictions, ignore_index=True) if predictions else
                   pd.DataFrame(columns=["specification", "test_year", "row_position", "return_month",
                                         "observed", "prediction"]))
    test_loss_name, test_loss_fn = configured_loss(cfg.RUN.EVALUATION.TEST)
    summary = []
    for name, group in predictions.groupby("specification", sort=False):
        observed, prediction = group["observed"].to_numpy(), group["prediction"].to_numpy()
        summary.append({
            "specification": name, "n_test": len(group),
            "first_test_year": int(group["test_year"].min()),
            "last_test_year": int(group["test_year"].max()),
            "test_loss_name": test_loss_name, "test_loss": test_loss_fn(observed, prediction),
            "r2_zero": zero_benchmark_r2(observed, prediction),
        })
    return RunResult(pd.DataFrame(annual_metrics), pd.DataFrame(summary),
                     predictions, tuning_history, cfg.dump(), dataset_info)


def rebuild_reports(folder, cfg, dataset_info):
    """Derived reports can always be reconstructed from completed checkpoints."""
    records = read_checkpoints(folder, cfg)
    ordered = [records[year, name]
               for year in range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1)
               for name in cfg.RUN.MODELS.ENABLED if (year, name) in records]
    result = assemble_result([r["metrics"] for r in ordered],
                             [pd.DataFrame(r["predictions"]) for r in ordered],
                             [r["tuning"] for r in ordered], cfg, dataset_info)
    for name in ("annual_metrics", "summary", "predictions"):
        atomic_write(Path(folder) / f"{name}.csv", getattr(result, name).to_csv(index=False))
    atomic_write(Path(folder) / "tuning_history.json", json.dumps(result.tuning_history, indent=2, allow_nan=False))
    status = [{"specification": name, "test_year": year,
               "status": "complete" if (year, name) in records else "pending"}
              for year in range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1)
              for name in cfg.RUN.MODELS.ENABLED]
    atomic_write(Path(folder) / "status.csv", pd.DataFrame(status).to_csv(index=False))
    print(f"Completed {len(records)}/{len(status)} model-years. Reports: {folder}", flush=True)
    if len(records) < len(status):
        quoted_folder = str(Path(folder).resolve()).replace("'", "''")
        print(
            "Run incomplete. To continue, please use --resume mode:\n"
            f"$runDir = '{quoted_folder}'\n"
            'python -B main.py --resume "$runDir"',
            flush=True,
        )
    return result


def save_run(run_result, output_dir):
    """Save a separate run directory; existing run files are never overwritten.

    Relative output paths are resolved against the project root, not the shell's
    working directory. This is called only when OUTPUT_DIR is nonempty.
    """
    parent = Path(output_dir)
    if not parent.is_absolute():
        parent = Path(root_path) / parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(run_result.config_yaml, encoding="utf-8")
    (run_dir / "dataset.json").write_text(
        json.dumps(run_result.dataset_info, indent=2, allow_nan=False), encoding="utf-8",
    )
    run_result.annual_metrics.to_csv(run_dir / "annual_metrics.csv", index=False)
    run_result.summary.to_csv(run_dir / "summary.csv", index=False)
    run_result.predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "tuning_history.json").write_text(
        json.dumps(run_result.tuning_history, indent=2, allow_nan=False), encoding="utf-8",
    )
    metadata = {"run_id": run_id, "python": platform.python_version(),
                "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "PyYAML", "iopath")}}
    (run_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir
