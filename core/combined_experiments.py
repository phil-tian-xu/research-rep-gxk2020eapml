"""One logical experiment over disjoint annual jobs; never fit or load features."""

from collections import ChainMap, Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from core.annual_jobs import annual_config, load_plan
from core.run_state import CheckpointIndex, atomic_write, dataset_identity, load_run_config, run_lock
from utils.config_utils.config import CfgNode


def research_config(plan):
    """Only output locations and requested test horizons may differ."""
    config = deepcopy(plan["config"])
    config["RUN"].pop("OUTPUT_DIR", None)
    for key in ("FIRST_TEST_YEAR", "LAST_TEST_YEAR"):
        config["RUN"]["SPLIT"].pop(key)
    return config


def combination_plan(experiments, years):
    sources = []
    for experiment in experiments:
        folder = Path(experiment).resolve()
        if (folder / "collection.json").exists():
            raise ValueError("Nested combined runs are not supported; select the original source experiments.")
        _, plan = load_plan(folder / "annual")
        sources.append(dict(experiment=str(folder), plan=plan))
    if len(sources) < 2 or len({s["experiment"] for s in sources}) != len(sources):
        raise ValueError("Select at least two distinct source experiments.")
    first = sources[0]["plan"]
    for source in sources[1:]:
        plan = source["plan"]
        if research_config(plan) != research_config(first):
            raise ValueError("Research configurations differ across source experiments.")
        if (plan["runtime"] != first["runtime"] or
                plan["orchestrator_sha256"] != first["orchestrator_sha256"]):
            raise ValueError("Recorded execution identities differ across source experiments.")
    counts = Counter(year for s in sources for year in s["plan"]["years"])
    expected = list(range(years[0], years[1] + 1))
    duplicates = sorted(y for y, count in counts.items() if count != 1)
    missing, extra = sorted(set(expected) - counts.keys()), sorted(counts.keys() - set(expected))
    if not expected or duplicates or missing or extra:
        raise ValueError(f"Invalid year coverage: duplicate={duplicates}, missing={missing}, out-of-range={extra}.")
    return dict(schema_version=1, years=expected, sources=sources)


def load_combination(folder):
    saved = json.loads((Path(folder) / "collection.json").read_text(encoding="utf-8"))
    if saved.get("schema_version") != 1:
        raise ValueError("Unsupported combined experiment schema.")
    actual = combination_plan([s["experiment"] for s in saved["sources"]],
                              (saved["years"][0], saved["years"][-1]))
    if actual != saved:
        raise ValueError("A source experiment plan changed after combination.")
    return saved


def resolve_annual_run(experiment, year):
    """Resolve a source fit for reporting, including Figures 4 and 5."""
    folder = Path(experiment).resolve()
    if (folder / "collection.json").is_file():
        plan = load_combination(folder)
        owners = [s for s in plan["sources"] if year in s["plan"]["years"]]
        if len(owners) != 1:
            raise ValueError(f"Year {year} is outside the combined run.")
        folder = Path(owners[0]["experiment"])
    if (folder / "annual").is_dir():
        folder /= "annual"
    runs = sorted((folder / "years" / str(year)).glob("*/manifest.json"))
    if len(runs) != 1:
        raise ValueError(f"Expected one annual run for {folder}, {year}; found {len(runs)}.")
    return runs[0].parent


def validate_scores(record):
    metrics = record["metrics"]
    year, model = metrics["test_year"], metrics["specification"]
    panel = pd.DataFrame(record["predictions"])
    months = pd.to_datetime(panel.return_month, format="%Y-%m", errors="coerce")
    positions = pd.to_numeric(panel.row_position, errors="coerce").to_numpy()
    if (panel.empty or panel.isna().any().any() or panel.row_position.duplicated().any()
            or not np.isfinite(positions).all() or np.any(positions < 0)
            or np.any(positions != np.floor(positions))
            or not panel.test_year.eq(year).all() or not panel.specification.eq(model).all()
            or not months.dt.year.eq(year).all()
            or not np.isfinite(panel[["observed", "prediction"]].to_numpy()).all()
            or metrics["n_test"] != len(panel)):
        raise ValueError(f"Invalid prediction panel/count/date: {model}, {year}.")
    observed, prediction = panel.observed.to_numpy(), panel.prediction.to_numpy()
    mse = float(np.mean((observed - prediction)**2))
    denominator = float(observed @ observed)
    r2 = 1 - float(np.sum((observed - prediction)**2)) / denominator if denominator else np.nan
    if (metrics["test_loss_name"] != "MSE"
            or not np.isclose(mse, metrics["test_loss"], rtol=1e-10, atol=1e-14)
            or not np.isclose(r2, metrics["r2_zero"], rtol=1e-10, atol=1e-12, equal_nan=True)):
        raise ValueError(f"Saved test metrics disagree with predictions: {model}, {year}.")
    tuning = record["tuning"]
    candidates = tuning["validation_results"]
    scores = np.asarray([c["validation_loss"] for c in candidates], dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid validation scores: {model}, {year}.")
    selected = candidates[int(np.argmin(scores))]
    if (selected["params"] != tuning["best_params"]
            or not np.isclose(scores.min(), metrics["best_validation_loss"], rtol=1e-10, atol=1e-14)):
        raise ValueError(f"Saved validation selection disagrees: {model}, {year}.")
    return selected


def nn_training_rows(record, selected, source):
    """Retained member diagnostics, distinct from ensemble validation MSE."""
    tuning = record["tuning"]
    rows = []
    for member in selected["members"]:
        history = member["history"]
        if ([h["epoch"] for h in history] != list(range(1, member["epochs_run"] + 1))
                or not 1 <= member["best_epoch"] <= len(history)):
            raise ValueError("Invalid NN epoch history.")
        retained = history[member["best_epoch"] - 1]
        values = [value for h in history for value in h.values()]
        if (not np.isfinite(values).all() or
                member["best_epoch"] != int(np.argmin([h["validation_mse"] for h in history])) + 1
                or not np.isclose(retained["validation_mse"], member["best_validation_mse"], rtol=1e-12)):
            raise ValueError("NN retained epoch disagrees with its validation history.")
        row = {key: value for key, value in member.items() if key != "history"}
        row.update(specification=tuning["specification"], test_year=tuning["test_year"],
                   source_experiment=source, **tuning["best_params"],
                   retained_train_mse=retained["train_mse"],
                   retained_validation_mse=retained["validation_mse"],
                   retained_train_prediction_std=retained["train_prediction_std"],
                   retained_validation_prediction_std=retained["validation_prediction_std"],
                   final_epoch_train_mse=history[-1]["train_mse"],
                   final_epoch_validation_mse=history[-1]["validation_mse"])
        warnings = []
        if member["parameter_update_l2"] == 0:
            warnings.append("zero_parameter_update")
        if member["nonzero_dense_weights"] == 0:
            warnings.append("all_dense_weights_zero")
        if retained["train_prediction_std"] == 0:
            warnings.append("constant_train_prediction")
        if retained["train_mse"] >= member["initial_train_mse"]:
            warnings.append("train_mse_not_below_initial")
        row["diagnostic_warnings"] = ";".join(warnings)
        rows.append(row)
    if ([r["seed"] for r in rows] != tuning["seeds"]
            or len(rows) != len(tuning["nn_state"]["members"])):
        raise ValueError("NN retained ensemble members/seeds disagree.")
    return rows


def collect_combination(experiment):
    from core.replication_report import report_tables, plot_complexity, plot_validation_curves

    folder = Path(experiment).resolve()
    with ExitStack() as locks:
        locks.enter_context(run_lock(folder))
        plan = load_combination(folder)
        cfg = CfgNode(plan["sources"][0]["plan"]["config"])
        cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR = plan["years"][0], plan["years"][-1]
        output = folder / "annual" / "combined"
        cfg.RUN.OUTPUT_DIR = str(output)
        records, sources, annual, training = ChainMap(), [], [], []
        dataset, features, nn_runtime = None, None, None
        for source in sorted(plan["sources"], key=lambda s: s["experiment"]):
            parent = Path(source["experiment"])
            locks.enter_context(run_lock(parent))
            locks.enter_context(run_lock(parent / "annual"))
        # Recheck after taking the source locks.
        load_combination(folder)
        for year in plan["years"]:
            source, = (s for s in plan["sources"] if year in s["plan"]["years"])
            parent, source_plan = Path(source["experiment"]), source["plan"]
            year_dir = parent / "annual" / "years" / str(year)
            if not year_dir.is_dir():
                raise ValueError(f"Missing/incomplete annual task: {year} ({parent.name}).")
            locks.enter_context(run_lock(year_dir))
            run = resolve_annual_run(parent, year)
            locks.enter_context(run_lock(run))
            source_cfg = load_run_config(run)
            if source_cfg.dump() != annual_config(source_plan, parent / "annual", year).dump():
                raise ValueError(f"Annual configuration differs from its source plan: {year}.")
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if manifest["runtime"] != source_plan["runtime"]:
                raise ValueError(f"Execution identity differs from its source plan: {year}.")
            dataset_identity(manifest["dataset"])
            if dataset is not None and dataset != manifest["dataset"]:
                raise ValueError("Dataset identities differ across source fits.")
            dataset = manifest["dataset"]
            index = CheckpointIndex(run, source_cfg)
            model, = source_cfg.RUN.MODELS.ENABLED
            if set(index) != {(year, model)}:
                raise ValueError(f"Missing/incomplete or unexpected annual checkpoint: {year}.")
            record = index[year, model]
            selected = validate_scores(record)
            tuning = record["tuning"]
            names = tuning["feature_names"]
            if not names or len(set(names)) != len(names) or (features is not None and features != names):
                raise ValueError("Feature order differs across source fits.")
            features = names
            launch = json.loads((parent / "launch.json").read_text(encoding="utf-8"))
            task_index = source_plan["years"].index(year)
            attempts = [a for a in launch["submissions"] if task_index in a.get("indices", [])
                        and a["state"] in {"submitted", "running", "complete", "failed"}]
            attempt = attempts[-1] if attempts else {}
            resources = attempt.get("resources", launch["resources"])
            checkpoint = index.paths[year, model]
            provenance = dict(test_year=year, specification=model, source_experiment=parent.name,
                experiment_directory=str(parent), run_directory=str(run), checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                requested_gpu_type=resources.get("gpu_type"), resources=resources,
                slurm_job_id=attempt.get("job_id"),
                device=tuning.get("device"), device_name=tuning.get("device_name"),
                training_csv=str(checkpoint.with_name(f"{model}_training.csv")) if model.startswith("NN") else None,
                training_plot=str(checkpoint.with_name(f"{model}_training.png")) if model.startswith("NN") else None)
            sources.append(provenance)
            row = dict(record["metrics"], source_experiment=parent.name,
                       device_name=tuning.get("device_name"), best_params=json.dumps(tuning["best_params"], sort_keys=True))
            if model.startswith("NN"):
                identity = {key: tuning[key] for key in ("torch_version", "cuda_version", "objective", "return_units", "hidden_sizes")}
                if nn_runtime is not None and identity != nn_runtime:
                    raise ValueError("NN runtime/method identities differ across source fits.")
                nn_runtime = identity
                training.extend(nn_training_rows(record, selected, parent.name))
                row.update(test_prediction_std=tuning["test_prediction_std"],
                           validation_zero_mse=tuning["validation_baselines"]["zero"])
            annual.append(row)
            records.maps.append(index)
            del record
        table, complexity, curves = report_tables(records, cfg)
        identity = dict(combination=plan, dataset=dataset, feature_names=features, sources=sources)
        # Repeated collection verifies all inputs again and returns the same published report.
        if output.exists():
            saved = json.loads((output / "source_identity.json").read_text(encoding="utf-8"))
            if saved != identity:
                raise ValueError("Published combined report has different source identities.")
            print(f"Verified combined reports: {output}")
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".collect-", dir=output.parent) as temporary:
            staging = Path(temporary) / "combined"
            staging.mkdir()
            annual_frame = pd.DataFrame(annual)
            annual_frame["r2_percent"] = annual_frame.r2_zero * 100
            for name, frame in (("table1_all", table), ("complexity", complexity),
                                ("tuning_curves", curves), ("annual_metrics", annual_frame),
                                ("year_sources", pd.DataFrame(sources))):
                atomic_write(staging / f"{name}.csv", frame.to_csv(index=False))
            if training:
                atomic_write(staging / "nn_training_summary.csv", pd.DataFrame(training).to_csv(index=False))
            atomic_write(staging / "source_identity.json", json.dumps(identity, indent=2))
            atomic_write(staging / "validation_summary.json", json.dumps(dict(
                status="complete", years=plan["years"], n_annual_fits=len(sources),
                checks=["fixed_disjoint_coverage", "config_runtime_dataset_feature_identity", "checkpoint_sha256",
                        "prediction_dates_counts_finiteness", "recomputed_mse_r2", "validation_selection", "nn_retained_epochs"],
                diagnostic_warning_members=sum(bool(r["diagnostic_warnings"]) for r in training)), indent=2))
            atomic_write(staging / "comparison_notes.txt",
                "Pooled R2 = 1 - sum annual squared forecast errors / sum annual observed returns squared.\n"
                "GPU models may differ; saved research settings, fitting sources and environments must agree.\n"
                "NN summary contains retained members of the validation-selected ensemble, not all grid candidates.\n"
                "Original checkpoints contain all candidate epoch histories; year_sources.csv links original curves.\n"
                "Negative R2 or training warnings do not themselves indicate corrupt results or justify selecting another fit.\n"
                "Integrity checks do not establish predictive success or exact paper replication.\n")
            plot_complexity(complexity, cfg, staging)
            plot_validation_curves(curves, staging)
            staging.rename(output)
        print(table.to_string(index=False))
        print(f"Combined reports: {output}")
        return output
