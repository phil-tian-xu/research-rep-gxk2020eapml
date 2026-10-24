"""Annual GKX NN experiments; model/training details live in models/neural_network.

Select a common learning-rate/L1 pair by ensemble validation MSE. Every seed
uses its own best validation epoch; the test window is loaded only after tuning.
"""

import itertools
import os
import time

import numpy as np
import pandas as pd
import torch

from core.batch_run import _check_saved, rebuild_batch_reports
from core.data_query.parquet_dataset import load_window
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.model_inputs import positive_integer
from core.models.neural_network import ARCHITECTURES, NeuralNetworkRegressor
from core.run import configured_loss
from core.run_state import (atomic_write, read_checkpoints, resolve_run_dir, run_lock,
                            save_checkpoint, validate_run)


def run_parquet(panel, cfg, *, checkpoint_dir, stop_after_year=None, memory_snapshot=None):
    settings, split = cfg.RUN.MODELS.NN, cfg.RUN.SPLIT
    names = list(cfg.RUN.MODELS.ENABLED)
    stop = split.LAST_TEST_YEAR if stop_after_year is None else stop_after_year
    if (len(names) != 1 or names[0] not in ARCHITECTURES or checkpoint_dir is None or
            panel.info["name"] != cfg.DATA.DATASET or isinstance(stop, bool) or not isinstance(stop, int) or
            not split.FIRST_TEST_YEAR <= stop <= split.LAST_TEST_YEAR or split.VALIDATION_YEARS < 1 or
            split.FIRST_TRAIN_YEAR >= split.FIRST_TEST_YEAR - split.VALIDATION_YEARS):
        raise ValueError("NN requires one architecture, matching data, checkpoints and valid annual windows.")
    name = names[0]
    if cfg.RUN.MODELS.STANDARDIZE or not cfg.RUN.MODELS.FIT_INTERCEPT:
        raise ValueError("The NN baseline uses supplied predictors and learned layer biases; no extra scaling.")
    rates, penalties = list(settings.LEARNING_RATE_GRID), list(settings.L1_GRID)
    for label, grid in (("learning rate", rates), ("L1", penalties)):
        if (not grid or len(set(grid)) != len(grid) or any(isinstance(v, bool) or
                not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in grid)):
            raise ValueError(f"Provide a unique positive NN {label} grid.")
    positive_integer(settings.ENSEMBLE_SIZE, "ENSEMBLE_SIZE")
    if isinstance(settings.SEED, bool) or not isinstance(settings.SEED, int) or settings.SEED < 0:
        raise ValueError("NN SEED must be a nonnegative integer.")
    seeds = [settings.SEED + 1000 * stop + i for i in range(settings.ENSEMBLE_SIZE)]
    # Validate all training options before reading any large data arrays.
    NeuralNetworkRegressor(hidden_sizes=ARCHITECTURES[name], learning_rate=rates[0], l1=penalties[0],
        seeds=seeds, batch_size=settings.BATCH_SIZE, max_epochs=settings.MAX_EPOCHS,
        patience=settings.PATIENCE, device=settings.DEVICE, predict_batch_size=settings.PREDICT_BATCH_SIZE)
    validation_label, validation_loss = configured_loss(cfg.RUN.EVALUATION.VALIDATION)
    test_label, test_loss = configured_loss(cfg.RUN.EVALUATION.TEST)
    if validation_label != "MSE" or test_label != "MSE":
        raise ValueError("The GKX NN baseline requires MSE validation and test evaluation.")
    if settings.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NN CUDA is unavailable; no CPU fallback. Check the NN extra and allocated GPU.")
    workers = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, workers))
    folder = resolve_run_dir(checkpoint_dir)

    def snapshot(stage):
        if memory_snapshot is not None:
            memory_snapshot(stage)

    with run_lock(folder):
        validate_run(folder, cfg, panel.info)
        saved = read_checkpoints(folder, cfg, lazy=True)
        year, params = None, None
        try:
            for year in range(split.FIRST_TEST_YEAR, stop + 1):
                label = f"[{name} {year}]"
                if (year, name) in saved:
                    _check_saved(saved[year, name], panel.window(year, year), year, name)
                    print(f"{label} Already saved; skipping.", flush=True)
                    continue
                started = time.perf_counter()
                if settings.DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                end = year - split.VALIDATION_YEARS - 1
                print(f"{label} Loading training {split.FIRST_TRAIN_YEAR}-{end}, validation {end+1}-{year-1}", flush=True)
                X, y, positions, months = load_window(panel, split.FIRST_TRAIN_YEAR, end, feature_dtype=np.float32)
                del positions, months
                V, vy, positions, months = load_window(panel, end + 1, year - 1, feature_dtype=np.float32)
                del positions, months
                n_train, n_validation = len(y), len(vy)
                training_mean = float(y.mean())
                snapshot(f"{year}: NN training/validation loaded")
                seeds = [settings.SEED + 1000 * year + i for i in range(settings.ENSEMBLE_SIZE)]
                candidates, best, best_score = [], None, np.inf
                for index, (rate, penalty) in enumerate(itertools.product(rates, penalties), 1):
                    params = dict(learning_rate=rate, l1=penalty)
                    print(f"{label} Candidate {index}/{len(rates)*len(penalties)}: {params}; device={settings.DEVICE}", flush=True)
                    model = NeuralNetworkRegressor(hidden_sizes=ARCHITECTURES[name], **params,
                        seeds=seeds, batch_size=settings.BATCH_SIZE, max_epochs=settings.MAX_EPOCHS,
                        patience=settings.PATIENCE, device=settings.DEVICE,
                        predict_batch_size=settings.PREDICT_BATCH_SIZE)
                    model.fit(X, y, X_validation=V, y_validation=vy,
                              progress=lambda message: print(f"{label} {message}", flush=True))
                    validation_prediction = model.predict(V)
                    score = float(validation_loss(vy, validation_prediction))
                    if not np.isfinite(score):
                        raise FloatingPointError("Nonfinite NN ensemble validation loss.")
                    candidates.append(dict(params=params.copy(), validation_loss=score,
                        prediction_mean=float(validation_prediction.mean()),
                        prediction_std=float(validation_prediction.std()), members=model.diagnostics_))
                    if score < best_score:
                        best, best_score = model, score
                    print(f"{label} Ensemble validation MSE={score:.10g}", flush=True)
                    del model, validation_prediction
                selected = int(np.argmin([c["validation_loss"] for c in candidates]))
                baseline_validation = dict(zero=float(np.mean(vy**2)),
                                           training_mean=float(np.mean((vy - training_mean)**2)))
                del X, y, V, vy
                snapshot(f"{year}: NN tuning complete")
                X, y, positions, months = load_window(panel, year, year, feature_dtype=np.float32)
                prediction = best.predict(X)
                metrics = dict(specification=name, test_year=year, n_train=n_train,
                    n_validation=n_validation, n_test=len(y), validation_loss_name=validation_label,
                    best_validation_loss=best_score, test_loss_name=test_label,
                    test_loss=float(test_loss(y, prediction)), r2_zero=zero_benchmark_r2(y, prediction))
                predictions = pd.DataFrame(dict(specification=name, test_year=year,
                    row_position=positions, return_month=months, observed=y, prediction=prediction))
                tuning = dict(specification=name, test_year=year, best_params=candidates[selected]["params"],
                    validation_results=candidates, feature_names=list(panel.features), nn_state=best.export_state(),
                    seeds=seeds, hidden_sizes=list(ARCHITECTURES[name]),
                    objective="mean_half_squared_error_plus_l1_dense_kernels", return_units="decimal",
                    validation_baselines=baseline_validation, training_mean=training_mean,
                    test_training_mean_r2=zero_benchmark_r2(y, np.full(len(y), training_mean)),
                    test_prediction_mean=float(prediction.mean()), test_prediction_std=float(prediction.std()),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda, device=settings.DEVICE,
                    device_name=torch.cuda.get_device_name() if settings.DEVICE == "cuda" else "CPU",
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if settings.DEVICE == "cuda" else 0,
                    elapsed_seconds=time.perf_counter() - started)
                save_checkpoint(folder, metrics, predictions, tuning)
                # Human-readable diagnostics supplement the authoritative checksummed checkpoint.
                rows = [dict(candidate=i, learning_rate=c["params"]["learning_rate"], l1=c["params"]["l1"],
                             seed=m["seed"], **epoch)
                        for i, c in enumerate(candidates, 1) for m in c["members"] for epoch in m["history"]]
                atomic_write(folder / "years" / str(year) / f"{name}_training.csv", pd.DataFrame(rows).to_csv(index=False))
                plot_training(candidates[selected]["members"], folder / "years" / str(year) / f"{name}_training.png")
                print(f"{label} Saved; selected={tuning['best_params']}; test R2={100*metrics['r2_zero']:.6f}%; "
                      f"forecast std={prediction.std():.6g}", flush=True)
                del best, X, y, prediction, predictions, candidates, tuning
                snapshot(f"{year}: NN checkpoint saved")
        except BaseException as error:
            error.add_note(f"[model={name}] [test_year={year}] [candidate={params}]; completed annual checkpoints can resume.")
            try:
                rebuild_batch_reports(folder, cfg)
            except Exception as report_error:
                error.add_note(f"Could not rebuild partial NN reports: {report_error}")
            raise
        return rebuild_batch_reports(folder, cfg)


def plot_training(members, path):
    """Selected candidate only; show all seeds and their selected validation epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for member in members:
        history = pd.DataFrame(member["history"])
        axes[0].plot(history.epoch, history.train_mse, alpha=.65)
        line, = axes[1].plot(history.epoch, history.validation_mse, alpha=.65)
        best = history.loc[history.epoch == member["best_epoch"]].iloc[0]
        axes[1].scatter([best.epoch], [best.validation_mse], color=line.get_color(), s=18)
    for axis, title in zip(axes, ("Training MSE (evaluation mode)", "Validation MSE; dots mark retained epochs")):
        axis.set(title=title, xlabel="Epoch", ylabel="Mean squared error (decimal returns)")
        axis.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
