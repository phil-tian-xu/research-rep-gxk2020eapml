"""Inspect saved NN training and retained parameters without fitting or loading data."""

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.annual_jobs import load_plan
from core.combined_experiments import (load_combination, load_source_bundle,
                                       resolve_annual_run, source_directory)
from core.models.nn_architectures import ARCHITECTURES
from core.run_state import atomic_write, load_run_config, read_checkpoints


NEAR_ZERO_THRESHOLDS = (1e-8, 1e-6, 1e-4)
PREDICTION_STD_THRESHOLD = 1e-8  # Decimal returns; a review threshold, not a selection rule.
HISTORY_COLUMNS = ("epoch", "train_mse", "validation_mse", "online_half_mse",
                   "online_l1_penalty", "first_batch_gradient_l2",
                   "train_prediction_std", "validation_prediction_std")


def parameter_summary(state, seeds):
    """Check the saved sequential architecture and summarize each tensor, without torch."""
    if state.get("version") != 1 or state.get("dtype") != "float32":
        raise ValueError("Unsupported NN parameter state.")
    widths = [state["n_features"], *state["hidden_sizes"], 1]
    if len(widths) < 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in widths):
        raise ValueError("Invalid saved NN layer widths.")
    if len(state["members"]) != len(seeds):
        raise ValueError("Saved NN members disagree with seeds.")
    expected = {}
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        output = index == len(widths) - 2
        layer = "output" if output else f"hidden_{index + 1}"
        expected[f"layers.{3 * index}.weight"] = (layer, "dense_kernel", (outputs, inputs))
        expected[f"layers.{3 * index}.bias"] = (layer, "dense_bias", (outputs,))
        if not output:
            for suffix, kind in (("weight", "bn_scale"), ("bias", "bn_offset"),
                                 ("running_mean", "bn_running_mean"), ("running_var", "bn_running_variance"),
                                 ("num_batches_tracked", "bn_batch_counter")):
                expected[f"layers.{3 * index + 2}.{suffix}"] = (
                    layer, kind, () if suffix == "num_batches_tracked" else (outputs,))
    rows = []
    for seed, member in zip(seeds, state["members"]):
        if set(member) != set(expected):
            raise ValueError("Saved NN tensor names disagree with its architecture.")
        for name, (layer, kind, shape) in expected.items():
            values = np.asarray(member[name], dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved NN tensor: {name}.")
            if kind == "bn_batch_counter":
                if values < 0 or values != np.floor(values):
                    raise ValueError("Invalid BN batch counter.")
            elif np.max(np.abs(values)) > np.finfo(np.float32).max:
                raise ValueError(f"Saved NN tensor is outside float32 range: {name}.")
            else:
                values = values.astype(np.float32).astype(np.float64)
            if kind == "bn_running_variance" and (values < 0).any():
                raise ValueError("Negative BN running variance.")
            absolute = np.abs(values).ravel()
            rows.append(dict(seed=seed, layer=layer, tensor=name, kind=kind,
                trainable=kind in {"dense_kernel", "dense_bias", "bn_scale", "bn_offset"},
                shape=str(list(shape)), count=values.size, zero_count=int(np.count_nonzero(values == 0)),
                zero_fraction=float(np.mean(values == 0)), l1_norm=float(absolute.sum()),
                l2_norm=float(np.linalg.norm(absolute)), rms=float(np.sqrt(np.mean(values**2))),
                median_abs=float(np.median(absolute)), max_abs=float(absolute.max()),
                **{f"near_zero_fraction_{threshold:.0e}": float(np.mean(absolute <= threshold))
                   for threshold in NEAR_ZERO_THRESHOLDS}))
    return pd.DataFrame(rows)


def diagnostic_tables(record):
    """All candidate histories; parameter checks concern only the retained ensemble."""
    tuning, metrics = record["tuning"], record["metrics"]
    model, year = tuning["specification"], tuning["test_year"]
    if (model not in ARCHITECTURES or metrics["specification"] != model or metrics["test_year"] != year
            or tuning["objective"] != "mean_half_squared_error_plus_l1_dense_kernels"
            or tuning["return_units"] != "decimal"):
        raise ValueError("Require a supported NN checkpoint with the recorded squared-error/L1 objective.")
    seeds, state = tuning["seeds"], tuning["nn_state"]
    names = tuning["feature_names"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
            or tuple(tuning["hidden_sizes"]) != ARCHITECTURES[model]
            or state["hidden_sizes"] != tuning["hidden_sizes"] or state["n_features"] != len(names)
            or len(set(names)) != len(names) or not names
            or state.get("feature_names") not in (None, names)):
        raise ValueError("Saved NN architecture, features or seeds disagree.")
    candidates = tuning["validation_results"]
    scores = np.asarray([c["validation_loss"] for c in candidates], dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid ensemble validation scores.")
    selected = int(np.argmin(scores))
    if (candidates[selected]["params"] != tuning["best_params"]
            or not np.isclose(scores[selected], metrics["best_validation_loss"], rtol=1e-12, atol=0)):
        raise ValueError("Selected NN candidate disagrees with saved validation scores.")
    parameters = parameter_summary(state, seeds)
    baselines = tuning["validation_baselines"]
    if (set(baselines) != {"zero", "training_mean"}
            or not np.isfinite(list(baselines.values())).all() or min(baselines.values()) < 0):
        raise ValueError("Invalid saved NN validation baselines.")
    epoch_rows, member_rows, candidate_rows, pairs = [], [], [], set()
    for number, candidate in enumerate(candidates, 1):
        rate, penalty = candidate["params"]["learning_rate"], candidate["params"]["l1"]
        if (not np.isfinite([rate, penalty]).all() or rate <= 0 or penalty < 0
                or (rate, penalty) in pairs or [m["seed"] for m in candidate["members"]] != seeds):
            raise ValueError("Invalid or duplicate NN candidate/seed identities.")
        if (not np.isfinite([candidate["prediction_mean"], candidate["prediction_std"]]).all()
                or candidate["prediction_std"] < 0):
            raise ValueError("Invalid saved ensemble prediction variation.")
        pairs.add((rate, penalty))
        identity = dict(model=model, test_year=year, candidate=number,
                        selected=number == selected + 1, learning_rate=rate, l1=penalty)
        candidate_rows.append(dict(**identity, ensemble_validation_mse=float(scores[number - 1]),
            mse_above_best=float(scores[number - 1] - scores[selected]),
            ensemble_validation_prediction_mean=candidate["prediction_mean"],
            ensemble_validation_prediction_std=candidate["prediction_std"]))
        for member in candidate["members"]:
            history = pd.DataFrame(member["history"])
            if (not set(HISTORY_COLUMNS) <= set(history) or history.empty
                    or history.epoch.tolist() != list(range(1, member["epochs_run"] + 1))
                    or not np.isfinite(history[list(HISTORY_COLUMNS)].to_numpy()).all()
                    or (history[list(HISTORY_COLUMNS)] < 0).any().any()
                    or member["best_epoch"] != int(np.argmin(history.validation_mse)) + 1
                    or member["stop_reason"] not in {"patience", "max_epochs"}):
                raise ValueError("Invalid NN epoch history or retained epoch.")
            retained = history.iloc[member["best_epoch"] - 1]
            if not np.isclose(retained.validation_mse, member["best_validation_mse"], rtol=1e-12, atol=0):
                raise ValueError("Retained NN epoch disagrees with its validation MSE.")
            summary = {key: value for key, value in member.items() if key != "history"}
            scalar_fields = ("initial_train_mse", "initial_validation_mse", "parameter_update_l2",
                             "dense_weight_l1", "nonzero_dense_weights", "dense_weight_count")
            scalars = np.asarray([summary[k] for k in scalar_fields], dtype=float)
            if (not np.isfinite(scalars).all() or (scalars < 0).any() or member["dense_weight_count"] < 1
                    or member["nonzero_dense_weights"] > member["dense_weight_count"]):
                raise ValueError("Invalid NN optimization/parameter summary.")
            warnings = []
            if member["parameter_update_l2"] == 0:
                warnings.append("zero_parameter_update")
            if member["nonzero_dense_weights"] == 0:
                warnings.append("all_dense_weights_zero")
            for label in ("train", "validation"):
                if retained[f"{label}_prediction_std"] <= PREDICTION_STD_THRESHOLD:
                    warnings.append(f"near_constant_{label}_prediction")
            if history.first_batch_gradient_l2.eq(0).all():
                warnings.append("all_recorded_first_batch_gradients_zero")
            if retained.train_mse >= member["initial_train_mse"]:
                warnings.append("retained_train_mse_not_below_initial")
            if identity["selected"]:
                kernels = parameters.loc[(parameters.seed == member["seed"]) & parameters.kind.eq("dense_kernel")]
                if (int(kernels["count"].sum()) != member["dense_weight_count"]
                        or int((kernels["count"] - kernels.zero_count).sum()) != member["nonzero_dense_weights"]
                        or not np.isclose(kernels.l1_norm.sum(), member["dense_weight_l1"], rtol=1e-5, atol=1e-12)):
                    raise ValueError("Saved dense weights disagree with retained-member diagnostics.")
                for layer in kernels.loc[kernels.zero_fraction.eq(1), "layer"]:
                    warnings.append(f"zero_dense_kernel:{layer}")
                scales = parameters.loc[parameters.seed.eq(member["seed"]) & parameters.kind.eq("bn_scale")]
                for layer in scales.loc[scales.zero_fraction.eq(1), "layer"]:
                    warnings.append(f"zero_bn_scale:{layer}")
            summary.update(identity, retained_train_mse=float(retained.train_mse),
                retained_validation_mse=float(retained.validation_mse),
                retained_train_prediction_std=float(retained.train_prediction_std),
                retained_validation_prediction_std=float(retained.validation_prediction_std),
                final_epoch_train_mse=float(history.iloc[-1].train_mse),
                final_epoch_validation_mse=float(history.iloc[-1].validation_mse),
                zero_first_batch_gradient_epochs=int(history.first_batch_gradient_l2.eq(0).sum()),
                parameter_state_available=identity["selected"], diagnostic_warnings=";".join(warnings))
            member_rows.append(summary)
            for epoch in member["history"]:
                epoch_rows.append(dict(**identity, seed=member["seed"], **epoch,
                    retained=epoch["epoch"] == member["best_epoch"],
                    online_objective=epoch["online_half_mse"] + epoch["online_l1_penalty"]))

    # Only saved forecast variation is inspected here; no test accuracy/comparison statistics.
    predictions = pd.DataFrame(record["predictions"])
    if (len(predictions) != metrics["n_test"] or predictions.empty
            or predictions.row_position.duplicated().any() or not np.isfinite(predictions.prediction).all()
            or not predictions.specification.eq(model).all() or not predictions.test_year.eq(year).all()):
        raise ValueError("Invalid saved prediction identities/counts.")
    months = pd.PeriodIndex(predictions.return_month, freq="M")
    if months.isna().any() or not (months.year == year).all():
        raise ValueError("Saved forecast months differ from the test year.")
    predictions["month"] = months.astype(str)
    variation = []
    for month, rows in [("all", predictions), *predictions.groupby("month", sort=True)]:
        values = rows.prediction.to_numpy(dtype=float)
        variation.append(dict(month=month, n_predictions=len(values), prediction_mean=float(values.mean()),
            prediction_std=float(values.std()), prediction_min=float(values.min()), prediction_max=float(values.max()),
            prediction_range=float(np.ptp(values)), assessable=len(values) >= 2,
            constant=bool(len(values) >= 2 and np.ptp(values) == 0),
            near_constant=bool(len(values) >= 2 and values.std() <= PREDICTION_STD_THRESHOLD)))
    return dict(epochs=pd.DataFrame(epoch_rows), members=pd.DataFrame(member_rows),
                candidates=pd.DataFrame(candidate_rows), parameters=parameters,
                prediction_variation=pd.DataFrame(variation))


def plot_diagnostics(tables, tuning, output, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = tables["epochs"].loc[lambda x: x.candidate == candidate]
    members = tables["members"].loc[lambda x: x.candidate == candidate]
    title = f"{tuning['specification']} · {tuning['test_year']} · candidate {candidate}"
    palette = plt.get_cmap("tab10")
    figures = []
    for name in ("learning_curves", "training_objective", "optimization", "prediction_variation"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for index, member in enumerate(members.to_dict("records")):
            history = epochs.loc[epochs.seed == member["seed"]]
            color = palette(index % 10)
            label = str(member["seed"])
            best = history.loc[history.retained].iloc[0]
            if name == "learning_curves":
                for axis, split in zip(axes, ("train", "validation")):
                    axis.plot(history.epoch, history[f"{split}_mse"], color=color, alpha=.8, label=label)
                    axis.scatter([best.epoch], [best[f"{split}_mse"]], color=color, s=20)
                axes[0].set(title="Training MSE · evaluation mode", ylabel="MSE (decimal returns squared)")
                axes[1].set(title="Validation MSE · evaluation mode")
            elif name == "training_objective":
                axes[0].plot(history.epoch, history.online_half_mse, color=color, alpha=.8, label=label)
                axes[0].plot(history.epoch, history.online_l1_penalty, color=color, linestyle="--", alpha=.8)
                axes[1].plot(history.epoch, history.online_objective, color=color, alpha=.8)
                axes[0].set(title="Solid: online half-MSE; dashed: online L1", ylabel="Online batch-weighted average")
                axes[1].set(title="Online training objective = half-MSE + L1")
            elif name == "optimization":
                axes[0].plot(history.epoch, history.first_batch_gradient_l2, color=color, label=label)
                axes[1].bar(index, member["parameter_update_l2"], color=color)
                axes[0].set(title="First-batch gradient norm per epoch", ylabel="L2 norm (data + penalty gradient)")
                axes[1].set(title="Retained parameters minus initialization", ylabel="L2 norm (all trainable parameters)")
            else:
                for axis, split in zip(axes, ("train", "validation")):
                    axis.plot(history.epoch, history[f"{split}_prediction_std"], color=color, label=label)
                    axis.scatter([best.epoch], [best[f"{split}_prediction_std"]], color=color, s=20)
                    axis.set(title=f"{split.capitalize()} member prediction variation", ylabel="Prediction std (decimal returns)")
        if name == "learning_curves":
            for label, value in tuning["validation_baselines"].items():
                axes[1].axhline(value, color="0.3", linestyle=":" if label == "zero" else "--", label=f"{label} baseline")
            for axis, split in zip(axes, ("train", "validation")):
                initial = members[f"initial_{split}_mse"]
                axis.set_title(axis.get_title() + f"\nInitial MSE range: {initial.min():.4g}–{initial.max():.4g}", fontsize=11)
        if name == "optimization":
            axes[0].set_yscale("symlog", linthresh=1e-10)
            axes[1].set_xticks(range(len(members)), members.seed.astype(str), rotation=60)
            axes[1].set_xlabel("Seed")
        for axis in axes:
            axis.grid(alpha=.2)
            if name != "optimization" or axis is axes[0]:
                axis.set_xlabel("Epoch")
            if name in {"training_objective", "prediction_variation"}:
                axis.set_yscale("symlog", linthresh=1e-10)
        axes[0].legend(title="Seed", fontsize=7, ncol=2)
        if name == "learning_curves":
            axes[1].legend(fontsize=7, ncol=2)
        fig.suptitle(title + (" · epoch 1 onward; dots: retained epoch" if name == "learning_curves" else ""), fontsize=10)
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
        figures.append(name)

    candidates = tables["candidates"]
    grid = candidates.pivot(index="learning_rate", columns="l1", values="ensemble_validation_mse").sort_index().sort_index(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    image = axes[0].imshow(np.ma.masked_invalid(grid.to_numpy()), aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
    axes[0].set_yticks(range(len(grid.index)), [f"{x:g}" for x in grid.index])
    for row in candidates.itertuples():
        x, y = grid.columns.get_loc(row.l1), grid.index.get_loc(row.learning_rate)
        axes[0].text(x, y, f"{row.ensemble_validation_mse:.6g}\n#{row.candidate}{' *' if row.selected else ''}",
                     ha="center", va="center", color="white", fontsize=9,
                     bbox=dict(facecolor="black", alpha=.35, edgecolor="none"))
    axes[0].set(xlabel="L1 penalty", ylabel="Learning rate", title="Ensemble validation MSE (* selected)")
    fig.colorbar(image, ax=axes[0])
    axes[1].bar(candidates.candidate, candidates.mse_above_best,
                color=["#B85B32" if s else "#246B8E" for s in candidates.selected])
    best_score = candidates.ensemble_validation_mse.min()
    for label, value in tuning["validation_baselines"].items():
        axes[1].axhline(value - best_score, color="0.3", linestyle=":" if label == "zero" else "--",
                       label=f"{label} baseline minus selected")
    axes[1].set(xlabel="Candidate", ylabel="Validation MSE difference", title="Candidate MSE minus selected ensemble MSE")
    axes[1].set_xticks(candidates.candidate)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=.2)
    fig.suptitle(f"{tuning['specification']} · {tuning['test_year']} · hyperparameter selection")
    fig.savefig(output / "hyperparameters.png", dpi=160)
    plt.close(fig)
    figures.append("hyperparameters")

    kernels = tables["parameters"].loc[lambda x: x.kind.eq("dense_kernel")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, column in zip(axes, ("zero_fraction", "near_zero_fraction_1e-06")):
        grid = kernels.pivot(index="seed", columns="layer", values=column).reindex(
            index=tuning["seeds"], columns=kernels.layer.drop_duplicates())
        image = axis.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="magma")
        axis.set_xticks(range(len(grid.columns)), grid.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(grid.index)), grid.index.astype(str))
        axis.set(title="Exact zero fraction" if column == "zero_fraction" else "Fraction |weight| ≤ 1e-6", ylabel="Seed")
        fig.colorbar(image, ax=axis)
    fig.suptitle("Retained selected ensemble · dense kernels only; biases and BN reported separately")
    fig.savefig(output / "parameters.png", dpi=160)
    plt.close(fig)
    figures.append("parameters")
    return figures


def write_page(path, title, body):
    atomic_write(path, '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{escape(title)}</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.5}}'
        'table{border-collapse:collapse;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #ddd;text-align:left}'
        'img{max-width:100%;height:auto}.scroll{overflow:auto}a{color:#246b8e}section{margin:28px 0}</style>'
        f'<body><h1>{escape(title)}</h1>{body}</body></html>')


def make_report(experiment, output_root="results/nn_diagnostics", years=None, candidate=None):
    """One checkpoint at a time. Source fits, data, RNG and execution hashes stay untouched."""
    experiment = Path(experiment).expanduser().resolve()
    direct = (experiment / "manifest.json").is_file()
    inventory, protected = {}, [experiment]
    if direct:
        cfg = load_run_config(experiment)
        planned = list(range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1))
    elif (experiment / "collection.json").is_file():
        plan = load_combination(experiment)
        planned = plan["years"]
        bundle = load_source_bundle(experiment)
        for source in plan["sources"]:
            protected.append(source_directory(experiment, source).resolve())
        if bundle:
            for source in bundle["sources"]:
                parent = experiment / "sources" / source["directory"]
                inventory.update({(parent / name).resolve(): digest for name, digest in source["files"].items()})
    else:
        annual = experiment / "annual" if (experiment / "annual").is_dir() else experiment
        _, plan = load_plan(annual)
        planned = plan["years"]
    years = planned if years is None else list(years)
    if not years or len(set(years)) != len(years) or not set(years) <= set(planned):
        raise ValueError("Select unique test years within the experiment's recorded horizon.")
    if candidate is not None and (isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1):
        raise ValueError("Candidate numbers start at 1.")
    output_root = Path(output_root).expanduser().resolve()
    if any(output_root.is_relative_to(path) for path in protected):
        raise ValueError("Write diagnostics outside the source experiment folders.")
    jobs = []
    for year in sorted(years):
        run = experiment if direct else resolve_annual_run(experiment, year)
        cfg = load_run_config(run)
        if len(cfg.RUN.MODELS.ENABLED) != 1 or cfg.RUN.MODELS.ENABLED[0] not in ARCHITECTURES:
            raise ValueError("NN diagnostics require one saved NN1-NN10 specification.")
        model = cfg.RUN.MODELS.ENABLED[0]
        checkpoints = read_checkpoints(run, cfg, lazy=True)
        if (year, model) not in checkpoints:
            raise ValueError(f"No completed {model} checkpoint for {year}; no fitting attempted.")
        jobs.append((year, run, cfg, checkpoints.paths[year, model]))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="incomplete", experiment=str(experiment), planned_years=planned, years=sorted(years),
        generated_utc=stamp, requested_candidate=candidate,
        near_zero_weight_thresholds=list(NEAR_ZERO_THRESHOLDS), prediction_std_threshold=PREDICTION_STD_THRESHOLD,
        online_objective="batch-size-weighted mean of half-MSE + summed dense-kernel L1 during updates",
        retained_state="each seed's best validation epoch, not the last executed epoch",
        gradient_scope="first batch per epoch; aggregate data-plus-L1 gradient, not layer-level gradients",
        parameter_scope="all candidate scalar diagnostics; full tensors only for the selected candidate",
        limitations=["No raw panel, forward-pass reproduction, retraining or test-accuracy comparison.",
                     "Near-zero weights and prediction warnings are review flags, never exclusion/selection rules.",
                     "Full weight/update/activation trajectories were not saved and cannot be reconstructed.",
                     "Initialization is diagnostic epoch 0 and was not a selectable retained epoch."],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=[])
    atomic_write(output / "methodology.json", json.dumps(metadata, indent=2))
    overview = []
    for year, run, cfg, checkpoint in jobs:
        source_hashes = {}
        for path in (run / "manifest.json", run / "resolved_config.yaml", checkpoint):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if inventory and inventory.get(path.resolve()) != digest:
                raise ValueError(f"NN diagnostic source differs from the frozen bundle inventory: {path}")
            source_hashes[str(path)] = digest
        record = read_checkpoints(run, cfg, lazy=True)[year, cfg.RUN.MODELS.ENABLED[0]]
        tables = diagnostic_tables(record)
        selected = int(tables["candidates"].loc[lambda x: x.selected, "candidate"].iloc[0])
        plotted = selected if candidate is None else candidate
        if plotted not in set(tables["candidates"].candidate):
            raise ValueError(f"Candidate {plotted} is absent from the {year} checkpoint.")
        folder = output / str(year)
        folder.mkdir()
        for name, frame in tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.17g")
        figures = plot_diagnostics(tables, record["tuning"], folder, plotted)
        chosen = tables["members"].loc[lambda x: x.selected]
        variation = tables["prediction_variation"]
        row = dict(test_year=year, model=record["tuning"]["specification"], dataset=cfg.DATA.DATASET,
            n_features=len(record["tuning"]["feature_names"]), selected_candidate=selected,
            plotted_candidate=plotted, **record["tuning"]["best_params"], members=len(chosen),
            members_with_warnings=int(chosen.diagnostic_warnings.ne("").sum()),
            zero_update_members=int(chosen.parameter_update_l2.eq(0).sum()),
            all_dense_weights_zero_members=int(chosen.nonzero_dense_weights.eq(0).sum()),
            zero_kernel_members=int(chosen.diagnostic_warnings.str.contains("zero_dense_kernel").sum()),
            near_constant_train_members=int(chosen.retained_train_prediction_std.le(PREDICTION_STD_THRESHOLD).sum()),
            near_constant_validation_members=int(chosen.retained_validation_prediction_std.le(PREDICTION_STD_THRESHOLD).sum()),
            near_constant_test_months=int(variation.loc[variation.month.ne("all"), "near_constant"].sum()))
        overview.append(row)
        links = " · ".join(f'<a href="{name}.csv">{escape(name)} CSV</a>' for name in tables)
        body = (f'<p><a href="../index.html">All reported years</a> · {escape(cfg.DATA.DATASET)} · {row["n_features"]} inputs</p>'
            f'<p>Selected candidate: {selected}; plotted candidate: {plotted}. Parameter tables always concern the selected ensemble.</p>'
            '<p>Loss curves show evaluation-mode MSE from epoch 1 onward; dots mark retained epochs. '
            'Initialization values are shown in the annotations and member table, so they do not compress the trained-epoch curves. '
            'The online objective uses changing weights and training-mode batch normalization; its scale is distinct from evaluation MSE.</p>'
            '<p>Only the first-batch aggregate gradient is recorded. Parameter update norms compare initialization with the retained state. '
            'BN buffers are not trainable parameters. Near-zero fractions are descriptive and are not pruning rules.</p>'
            f'<p>{links}</p><div class="scroll">{pd.DataFrame([row]).to_html(index=False, border=0)}</div>'
            '<h2>Retained ensemble members</h2><div class="scroll">'
            + chosen[["seed", "best_epoch", "epochs_run", "stop_reason", "initial_train_mse", "retained_train_mse",
                      "parameter_update_l2", "nonzero_dense_weights", "diagnostic_warnings"]].to_html(
                          index=False, border=0, float_format=lambda value: f"{value:.9g}") + '</div>')
        for name in figures:
            body += f'<section><h2>{escape(name.replace("_", " ").capitalize())}</h2><img src="{name}.png" alt="{escape(name)}"></section>'
        for name in ("candidates", "members", "parameters", "prediction_variation"):
            body += (f'<details><summary>{escape(name.replace("_", " ").capitalize())} table</summary>'
                     f'<div class="scroll">{tables[name].to_html(index=False, border=0, float_format=lambda value: f"{value:.9g}")}</div></details>')
        write_page(folder / "index.html", f"{row['model']} · {year} · NN fitting diagnostics", body)
        metadata["sources"].append(dict(test_year=year, hashes=source_hashes))
        atomic_write(output / "methodology.json", json.dumps(metadata, indent=2))
        print(f"{row['model']} {year}: {len(chosen)} retained members; {row['members_with_warnings']} with review flags", flush=True)
        del record, tables
    summary = pd.DataFrame(overview)
    summary.to_csv(output / "summary.csv", index=False, float_format="%.17g")
    body = (f'<p>Source: {escape(str(experiment))}</p><p>Reported years: {escape(str(sorted(years)))}; '
        f'recorded experiment horizon: {min(planned)}–{max(planned)}. This is a fitting report; warnings do not change model selection.</p>'
        '<p><a href="summary.csv">Summary CSV</a> · <a href="methodology.json">Scope, thresholds and source hashes</a></p>'
        f'<div class="scroll">{summary.to_html(index=False, border=0)}</div><ul>'
        + "".join(f'<li><a href="{y}/index.html">{y}: curves, optimization, variation, tuning and parameters</a></li>' for y in sorted(years))
        + '</ul>')
    write_page(output / "index.html", "NN fitting diagnostics", body)
    metadata["status"] = "complete"
    atomic_write(output / "methodology.json", json.dumps(metadata, indent=2))
    print(f"Saved NN fitting diagnostics: {output}")
    return output
