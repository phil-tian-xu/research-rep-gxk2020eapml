"""Annual Parquet workflow with bounded training, prediction, and report memory."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.models.batched import reduce_training, fit_components, fit_huber
from core.run import configured_loss, configured_specifications
from core.run_state import (atomic_write, read_checkpoints, resolve_run_dir, run_lock,
                            save_checkpoint, validate_run)


@dataclass
class BatchRunResult:
    annual_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions_path: Path


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None):
    """Fit one model per specification/year, saving each before starting the next."""
    split = cfg.RUN.SPLIT
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or
            split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("Invalid annual train/validation/test years.")
    specs = configured_specifications(cfg.RUN.MODELS)
    if any(name not in {"PCR", "PLS", "OLS_HUBER"} for name, _, _ in specs):
        raise ValueError("Batch fitting currently supports PCR, PLS, and OLS_HUBER only.")
    if checkpoint_dir is None:
        raise ValueError("Batch runs require RUN.OUTPUT_DIR for incremental results.")
    if panel.info["name"] != cfg.DATA.DATASET:
        raise ValueError("Parquet dataset identity differs from DATA.DATASET.")
    folder = resolve_run_dir(checkpoint_dir)
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        completed = read_checkpoints(folder, cfg, lazy=True)
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                test = panel.window(year, year)
                pending = []
                for name, factory, candidates in specs:
                    if (year, name) in completed:
                        _check_saved(completed[year, name], test, year, name)
                        print(f"[model={name}] [test_year={year}] Already saved; skipping.", flush=True)
                    else:
                        pending.append((name, factory, candidates))
                if not pending:
                    continue
                train = panel.window(split.FIRST_TRAIN_YEAR, year - split.VALIDATION_YEARS - 1)
                validation = panel.window(year - split.VALIDATION_YEARS, year - 1)
                print(f"[test_year={year}] train={train.n_rows:,}, validation={validation.n_rows:,}, "
                      f"test={test.n_rows:,}; batch_size={panel.batch_size:,}", flush=True)
                reduction = reduce_training(train, fit_intercept=cfg.RUN.MODELS.FIT_INTERCEPT,
                                            standardize=cfg.RUN.MODELS.STANDARDIZE)
                for name, _, candidates in pending:
                    models, records = [], []
                    for index, parameters in enumerate(candidates, 1):
                        try:
                            print(f"[model={name}] [test_year={year}] Candidate {index}/{len(candidates)}", flush=True)
                            model = (fit_huber(parameters, reduction, train) if name == "OLS_HUBER" else
                                     fit_components(name, parameters, reduction))
                            if not getattr(model, "converged_", True):
                                result = model.optimization_result_
                                threshold = model.tol * (1 + np.linalg.norm(result.coefficients, ord=np.inf))
                                raise RuntimeError(f"Candidate did not converge: iterations={model.n_iter_}, "
                                    f"stationarity={result.stationarity:.6g}, stopping_threshold={threshold:.6g}.")
                        except Exception as error:
                            error.add_note(f"[model={name}] [test_year={year}] stage=fit; parameters={parameters}")
                            raise
                        record = {"params": parameters.copy(), "n_components": getattr(model, "n_components_", None)}
                        if name == "OLS_HUBER":
                            result = model.optimization_result_
                            record.update(huber_delta=model.huber_delta_, n_iter=model.n_iter_,
                                          stationarity=result.stationarity,
                                          stopping_threshold=float(model.tol * (1 + np.linalg.norm(result.coefficients, ord=np.inf))))
                        models.append(model)
                        records.append(record)
                    try:
                        # Only batch_rows x candidate_count predictions, never the
                        # full validation design or its score/latent matrices.
                        coefficients = np.column_stack([model.coef_ for model in models])
                        intercepts = np.array([model.intercept_ for model in models])
                        scores = np.zeros(len(models))
                        for batch in validation.batches():
                            forecasts = batch.X @ coefficients + intercepts
                            for index in range(len(models)):
                                scores[index] += validation_loss(batch.y, forecasts[:, index]) * len(batch.y) / validation.n_rows
                        if not np.isfinite(scores).all():
                            raise FloatingPointError("Validation scores must be finite.")
                        selected = int(np.argmin(scores))
                        for record, score in zip(records, scores):
                            record["validation_loss"] = float(score)
                        best, parameters = models[selected], candidates[selected]
                        del models, coefficients, intercepts
                        # Retain only this year's compact prediction columns for
                        # the existing atomic model-year checkpoint format.
                        frames, score = [], 0.
                        for batch in test.batches():
                            prediction = best.predict(batch.X)
                            score += test_loss(batch.y, prediction) * len(batch.y) / test.n_rows
                            frames.append(pd.DataFrame({"specification": name, "test_year": year,
                                "row_position": batch.row_position, "return_month": batch.return_month,
                                "observed": batch.y, "prediction": prediction}))
                        predictions = pd.concat(frames, ignore_index=True)
                        metrics = {"specification": name, "test_year": year, "n_train": train.n_rows,
                            "n_validation": validation.n_rows, "n_test": test.n_rows,
                            "validation_loss_name": validation_label, "best_validation_loss": float(scores[selected]),
                            "test_loss_name": test_label, "test_loss": float(score),
                            "r2_zero": zero_benchmark_r2(predictions.observed, predictions.prediction)}
                        tuning = {"specification": name, "test_year": year,
                                  "best_params": parameters.copy(), "validation_results": records}
                        save_checkpoint(folder, metrics, predictions, tuning)
                        print(f"[model={name}] [test_year={year}] Saved; selected={parameters}, "
                              f"validation_loss={scores[selected]:.9g}", flush=True)
                        del frames, predictions, best, model, batch, forecasts
                    except Exception as error:
                        error.add_note(f"[model={name}] [test_year={year}] stage=validation/test/checkpoint")
                        raise
                del reduction
        except BaseException as error:
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)


def _check_saved(record, test, year, name):
    saved = pd.DataFrame(record["predictions"])
    if len(saved) != test.n_rows or not np.isfinite(saved.prediction).all():
        raise ValueError(f"Invalid checkpoint observations: {name}, {year}")
    start = 0
    for batch in test.batches(features=False):
        rows = saved.iloc[start:start + len(batch.y)]
        if (not np.array_equal(rows.row_position, batch.row_position) or
                not np.array_equal(rows.observed, batch.y) or
                not np.array_equal(rows.return_month, batch.return_month) or
                not (rows.specification == name).all() or not (rows.test_year == year).all()):
            raise ValueError(f"Checkpoint observations differ: model={name}, test_year={year}")
        start += len(batch.y)


def rebuild_batch_reports(folder, cfg):
    """Write aggregate CSV incrementally and pool sums, holding one model-year."""
    folder = Path(folder)
    records = read_checkpoints(folder, cfg, lazy=True)
    status, metrics, tuning, pooled = [], [], [], {}
    label, loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    temporary = folder / f".predictions.{uuid4().hex}.tmp"
    columns = ["specification", "test_year", "row_position", "return_month", "observed", "prediction"]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            pd.DataFrame(columns=columns).to_csv(stream, index=False)
            for year in range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1):
                for name in cfg.RUN.MODELS.ENABLED:
                    exists = (year, name) in records
                    status.append({"specification": name, "test_year": year,
                                   "status": "complete" if exists else "pending"})
                    if not exists:
                        continue
                    record = records[year, name]
                    frame = pd.DataFrame(record["predictions"])
                    frame.to_csv(stream, index=False, header=False, columns=columns)
                    metrics.append(record["metrics"])
                    tuning.append(record["tuning"])
                    y, prediction = frame.observed.to_numpy(), frame.prediction.to_numpy()
                    total = pooled.setdefault(name, {"n": 0, "sse": 0., "denom": 0., "loss": 0., "first": year, "last": year})
                    total["n"] += len(y)
                    total["sse"] += float(np.sum((y - prediction)**2))
                    total["denom"] += float(y @ y)
                    total["loss"] += loss(y, prediction) * len(y)
                    total["last"] = year
                    del record, frame, y, prediction
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / "predictions.csv")
    finally:
        temporary.unlink(missing_ok=True)
    summary = pd.DataFrame([{"specification": name, "n_test": t["n"], "first_test_year": t["first"],
        "last_test_year": t["last"], "test_loss_name": label, "test_loss": t["loss"] / t["n"],
        "r2_zero": 1 - t["sse"] / t["denom"] if t["denom"] else np.nan} for name, t in pooled.items()])
    annual = pd.DataFrame(metrics)
    for name, frame in (("annual_metrics", annual), ("summary", summary), ("status", pd.DataFrame(status))):
        atomic_write(folder / f"{name}.csv", frame.to_csv(index=False))
    atomic_write(folder / "tuning_history.json", json.dumps(tuning, indent=2, allow_nan=False))
    print(f"Completed {len(records)}/{len(status)} model-years. Reports: {folder}", flush=True)
    if len(records) < len(status):
        quoted = str(folder.resolve()).replace("'", "''")
        print(f"Run incomplete. To continue, please use --resume mode:\n$runDir = '{quoted}'\n"
              'python -B main.py --resume "$runDir"', flush=True)
    return BatchRunResult(annual, summary, folder / "predictions.csv")
