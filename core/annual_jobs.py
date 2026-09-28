"""Prepare independent annual runs, execute one task, and collect replication reports.

Uses the existing main workflow unchanged. Preparation does not load the panel.
"""

import argparse
from collections import ChainMap
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import pandas as pd
import yaml

from core.run_state import (CheckpointIndex, atomic_write, load_run_config,
                            run_lock, runtime_identity)
from utils.config_utils.config import CfgNode
from utils.general_utils import load_config


def prepare(config, first_year, last_year, output):
    cfg = load_config(config)
    split = cfg.RUN.SPLIT
    if not split.FIRST_TEST_YEAR <= first_year <= last_year <= split.LAST_TEST_YEAR:
        raise ValueError("Requested years must be within the base configuration's test horizon.")
    if cfg.DATA.get("FORMAT") != "parquet" or cfg.RUN.get("FITTING") != "dense":
        raise ValueError("Annual jobs currently require the complete-array Parquet workflow.")
    if list(cfg.RUN.MODELS.ENABLED) not in (["PCR"], ["PLS"], ["ENET_HUBER"], ["OLS3_HUBER"], ["GLM_HUBER"], ["RF"]):
        raise ValueError("Select exactly one supported model: PCR, PLS, ENET_HUBER, OLS3_HUBER, GLM_HUBER or RF.")
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cfg.RUN.SPLIT.FIRST_TEST_YEAR = first_year
    cfg.RUN.SPLIT.LAST_TEST_YEAR = last_year
    cfg.RUN.OUTPUT_DIR = str(folder / "combined")
    plan = {"schema_version": 1, "config": yaml.safe_load(cfg.dump()),
            "runtime": runtime_identity(),
            "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "years": list(range(first_year, last_year + 1))}
    for year in plan["years"]:
        annual = annual_config(plan, folder, year)
        atomic_write(folder / "configs" / f"{year}.yaml", annual.dump())
    atomic_write(folder / "experiment.json", json.dumps(plan, indent=2))
    print(f"Prepared {len(plan['years'])} annual tasks: {folder}")
    return folder


def annual_config(plan, folder, year):
    cfg = CfgNode(plan["config"])
    cfg.RUN.SPLIT.FIRST_TEST_YEAR = year
    cfg.RUN.SPLIT.LAST_TEST_YEAR = year
    cfg.RUN.OUTPUT_DIR = str(folder / "years" / str(year))
    return cfg


def load_plan(experiment):
    folder = Path(experiment).resolve()
    plan = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    if plan["schema_version"] != 1:
        raise ValueError("Unsupported annual experiment schema.")
    split = plan["config"]["RUN"]["SPLIT"]
    if plan["years"] != list(range(split["FIRST_TEST_YEAR"], split["LAST_TEST_YEAR"] + 1)):
        raise ValueError("Experiment years disagree with its configuration.")
    return folder, plan


def run_task(experiment, task_index):
    from main import main
    from core.data_examine.csv_memory import process_memory

    folder, plan = load_plan(experiment)
    if not 0 <= task_index < len(plan["years"]):
        raise ValueError("Task index is outside the prepared year range.")
    if (runtime_identity() != plan["runtime"] or
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["orchestrator_sha256"]):
        raise ValueError("Execution code/environment changed since preparation; prepare a new experiment.")
    year = plan["years"][task_index]
    cfg = annual_config(plan, folder, year)
    config_path = folder / "configs" / f"{year}.yaml"
    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != yaml.safe_load(cfg.dump()):
        raise ValueError("Generated annual configuration was changed.")
    year_dir = folder / "years" / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    # Serializes accidental duplicate task submissions before a timestamped run exists.
    with run_lock(year_dir):
        runs = sorted(path.parent for path in year_dir.glob("*/manifest.json"))
        if len(runs) > 1:
            raise ValueError(f"Multiple runs for {year}; select a single run before continuing.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        started = time.perf_counter()
        status = {"test_year": year, "status": "failed", "started_utc": stamp}
        try:
            if runs:
                if load_run_config(runs[0]).dump() != cfg.dump():
                    raise ValueError("Existing run configuration differs from the prepared task.")
                main(resume=str(runs[0]), memory=True)
            else:
                main(str(config_path.with_suffix("")), memory=True)
            status["status"] = "complete"
        except BaseException as error:
            status["error"] = repr(error)
            raise
        finally:
            status["elapsed_seconds"] = time.perf_counter() - started
            try:
                counters = process_memory()
                status["peak_resident_gib"] = counters["peak_resident_bytes"] / 1024**3 if counters else None
            except Exception:
                status["peak_resident_gib"] = None
            atomic_write(year_dir / f"attempt_{stamp}.json", json.dumps(status, indent=2))


def collect(experiment):
    from core.replication_report import report_tables, plot_complexity, plot_validation_curves

    folder, plan = load_plan(experiment)
    cfg = CfgNode(plan["config"])
    output = folder / "combined"
    # Refuse to overwrite an already published report.
    if output.exists():
        raise FileExistsError(f"Combined report already exists: {output}")
    records = ChainMap()  # Lazy checkpoint indexes; never retain all predictions.
    sources, summary = [], []
    dataset = None
    with ExitStack() as locks:
        locks.enter_context(run_lock(folder))
        if output.exists():
            raise FileExistsError(f"Combined report already exists: {output}")
        for year in plan["years"]:
            year_dir = folder / "years" / str(year)
            if year_dir.exists():
                locks.enter_context(run_lock(year_dir))
            runs = sorted(path.parent for path in year_dir.glob("*/manifest.json"))
            if len(runs) > 1:
                raise ValueError(f"Duplicate runs for {year}: {runs}")
            attempts = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(year_dir.glob("attempt_*.json"))]
            row = {"test_year": year, "status": "missing", "attempts": len(attempts),
                   "elapsed_seconds_all_attempts": sum(a["elapsed_seconds"] for a in attempts),
                   "peak_resident_gib": max((a["peak_resident_gib"] for a in attempts
                                             if a.get("peak_resident_gib") is not None), default=None)}
            if runs:
                run_dir = runs[0]
                locks.enter_context(run_lock(run_dir))
                saved = load_run_config(run_dir)
                if saved.dump() != annual_config(plan, folder, year).dump():
                    raise ValueError(f"Configuration mismatch for {year}.")
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                if manifest["runtime"] != plan["runtime"]:
                    raise ValueError(f"Execution identity mismatch for {year}.")
                if dataset is not None and dataset != manifest["dataset"]:
                    raise ValueError(f"Dataset identity mismatch for {year}.")
                dataset = manifest["dataset"]
                index = CheckpointIndex(run_dir, saved)
                expected = {(year, model) for model in cfg.RUN.MODELS.ENABLED}
                if set(index) == expected:
                    row["status"] = "complete"
                else:
                    row["status"] = "incomplete"
                records.maps.append(index)
                sources.append({"test_year": year, "run_directory": str(run_dir)})
            summary.append(row)
        atomic_write(folder / "task_summary.csv", pd.DataFrame(summary).to_csv(index=False))
        if any(row["status"] != "complete" for row in summary):
            raise ValueError("Missing/incomplete years; see task_summary.csv. No combined report published.")
        # Existing reporting validates prediction identity, finite values and tuning selection.
        table, complexity, curves = report_tables(records, cfg)
        annual = pd.DataFrame([record["metrics"] for record in records.values()]).sort_values(
            ["test_year", "specification"])
        annual["r2_percent"] = annual.r2_zero * 100
        # Publish only when every CSV and plot has been written successfully.
        staging = locks.enter_context(TemporaryDirectory(prefix=".collect-", dir=folder))
        output = Path(staging)
        for name, frame in (("table1_all", table), ("complexity", complexity),
                            ("tuning_curves", curves), ("annual_metrics", annual)):
            atomic_write(output / f"{name}.csv", frame.to_csv(index=False))
        plot_complexity(complexity, cfg, output)
        plot_validation_curves(curves, output)
        atomic_write(output / "source_identity.json", json.dumps({
            "experiment": plan, "dataset": dataset, "sources": sources}, indent=2))
        atomic_write(output / "comparison_notes.txt",
                     "R2 is pooled across saved observations, not averaged across annual R2.\n"
                     "Paper references cover the full universe in 1987-2016. A shorter experiment\n"
                     "is only a partial-period diagnostic; complete_horizon refers to its requested years.\n"
                     "Peak RAM and elapsed time in task_summary.csv cover recorded worker attempts;\n"
                     "an externally killed worker may not save an attempt. Consult Slurm/GNU time logs.\n")
        output.rename(folder / "combined")
        output = folder / "combined"
    print(table.to_string(index=False))
    print(f"Combined reports: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--first-year", type=int, required=True)
    prep.add_argument("--last-year", type=int, required=True)
    prep.add_argument("--output", required=True, help="New experiment directory; must not exist.")
    worker = commands.add_parser("run")
    worker.add_argument("--experiment", required=True)
    worker.add_argument("--task-index", required=True, type=int)
    report = commands.add_parser("collect")
    report.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.first_year, args.last_year, args.output)
    elif args.command == "run":
        run_task(args.experiment, args.task_index)
    else:
        collect(args.experiment)
