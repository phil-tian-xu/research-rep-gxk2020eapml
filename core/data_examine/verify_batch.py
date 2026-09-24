"""Verify batch fits against saved dense results, then measure full-window RAM.

Run --help for the compare and memory commands. Each annual fit runs in a fresh
Python subprocess, with the production grids, validation, prediction and reports.
No reference results are overwritten. This does not run the full experiment by default.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import yaml

from core.data_examine.csv_memory import process_memory
from core.data_query.datasets import DATASET_PATHS
from core.data_query.parquet_dataset import ParquetPanel
from core.run_state import (atomic_write, create_run, load_run_config, read_checkpoints,
                            resolve_run_dir)
from utils.config_utils.config import CfgNode
from utils.general_utils import load_config
from utils.path_info import root_path


def memory_summary(report):
    """Separate an observed exceedance from windows that have not completed."""
    requested = set(report["years"])
    results = report["results"]
    completed = {r["test_year"] for r in results if r["status"] == "complete"}
    measured = [r for r in results if r.get("peak_resident_gib") is not None]
    measured_years = {r["test_year"] for r in measured}
    peak = max((r["peak_resident_gib"] for r in measured), default=None)
    if peak is not None and peak >= report["memory_target_gib"]:
        status = "exceeded"
    elif requested <= completed and requested <= measured_years:
        status = "passed"
    elif requested <= completed:
        status = "unavailable"
    else:
        status = "incomplete"
    return {"peak_resident_gib": peak, "memory_status": status,
            "memory_target_passed": True if status == "passed" else False if status == "exceeded" else None,
            "completed_years": sorted(completed), "pending_or_failed_years": sorted(requested - completed),
            "largest_window_completed": bool(completed & set(report["largest_training_test_years"]))}


def print_memory_summary(report):
    summary = memory_summary(report)
    for result in report["results"]:
        peak = result.get("peak_resident_gib")
        reading = f"{peak:.3f} GiB" if peak is not None else "unavailable"
        print(f"  {result['test_year']}: peak Python RAM={reading}; fit={result['status']}", flush=True)
    peak = summary["peak_resident_gib"]
    print("MAXIMUM OBSERVED PYTHON RAM: " + (f"{peak:.3f} GiB" if peak is not None else "unavailable"), flush=True)
    print(f"Memory check: {summary['memory_status'].upper()} (target < {report['memory_target_gib']:g} GiB); "
          f"completed {len(summary['completed_years'])}/{len(report['years'])} requested windows; "
          f"largest window completed={summary['largest_window_completed']}.", flush=True)
    if summary["memory_status"] == "incomplete":
        print("No completed full-scope memory verdict: some requested fits have not finished. "
              "This does not mean the RAM target was exceeded.", flush=True)
    return summary


def watch_memory(stop, output, year, interval=30):
    """Publish OS-maintained peak-so-far while the fitting process is alive."""
    while not stop.wait(interval):
        try:
            counters = process_memory()
            if counters is None:
                return
            snapshot = {"pid": os.getpid(), "test_year": year,
                        "measured_utc": datetime.now(timezone.utc).isoformat(),
                        "memory_bytes": counters,
                        "peak_resident_gib": counters["peak_resident_bytes"] / 1024**3}
            atomic_write(output, json.dumps(snapshot, indent=2))
            print(f"[RAM] year={year} PID={os.getpid()}: current={counters['resident_bytes'] / 1024**3:.3f} GiB; "
                  f"MAX SO FAR={snapshot['peak_resident_gib']:.3f} GiB", flush=True)
        except Exception as error:
            print(f"Live RAM display unavailable: {error}. Final measurement will still be attempted.", flush=True)
            return


def check_reference(panel, folder, cfg):
    """Require conversion of precisely the CSV/metadata used in the saved run."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["dataset"]
    if (source["name"] != cfg.DATA.DATASET or
            panel.metadata["source_panel_sha256"] != source["panel_sha256"] or
            panel.metadata["source_metadata_sha256"] != source["metadata_sha256"]):
        raise ValueError("Parquet source hashes do not match the reference run's dataset. "
                         "Convert the exact prepared 5% CSV used by that run.")


def compare_records(reference, actual, *, atol, rtol):
    """Compare one model-year, aligning row positions before numerical checks."""
    rows = []

    def exact(field, passed):
        rows.append({"field": field, "passed": bool(passed), "max_abs_difference": None,
                     "rmse_difference": None, "outside_tolerance": None})

    def numeric(field, expected, observed):
        expected, observed = np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)
        if expected.shape != observed.shape:
            exact(field, False)
            return
        close = np.isclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)
        # Annual R2 is legitimately undefined only if both targets are all zero.
        if field == "r2_zero":
            close |= np.isnan(expected) & np.isnan(observed)
        difference = (observed - expected).ravel()
        finite = difference[np.isfinite(difference)]
        rows.append({"field": field, "passed": bool(close.all()),
                     "max_abs_difference": float(np.max(np.abs(finite))) if finite.size else None,
                     "rmse_difference": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
                     "outside_tolerance": int(np.count_nonzero(~close))})

    old = pd.DataFrame(reference["predictions"]).sort_values("row_position").reset_index(drop=True)
    new = pd.DataFrame(actual["predictions"]).sort_values("row_position").reset_index(drop=True)
    identity = ["specification", "test_year", "row_position", "return_month", "observed"]
    aligned = (not old.row_position.duplicated().any() and not new.row_position.duplicated().any()
               and old[identity].equals(new[identity]))
    exact("test_observation_identity", aligned)
    if aligned:
        numeric("predictions", old.prediction, new.prediction)
    for field in ("specification", "test_year", "n_train", "n_validation", "n_test",
                  "validation_loss_name", "test_loss_name"):
        exact(field, reference["metrics"][field] == actual["metrics"][field])
    for field in ("best_validation_loss", "test_loss", "r2_zero"):
        numeric(field, reference["metrics"][field], actual["metrics"][field])
    old_tuning, new_tuning = reference["tuning"], actual["tuning"]
    exact("selected_parameters", old_tuning["best_params"] == new_tuning["best_params"])
    old_candidates, new_candidates = old_tuning["validation_results"], new_tuning["validation_results"]
    same_grid = [c["params"] for c in old_candidates] == [c["params"] for c in new_candidates]
    exact("candidate_grid", same_grid)
    if same_grid:
        numeric("all_candidate_validation_losses", [c["validation_loss"] for c in old_candidates],
                [c["validation_loss"] for c in new_candidates])
        exact("all_fitted_component_counts", [c["n_components"] for c in old_candidates] ==
              [c["n_components"] for c in new_candidates])
        if any("huber_delta" in c for c in old_candidates):
            numeric("huber_thresholds", [c.get("huber_delta", np.nan) for c in old_candidates],
                    [c.get("huber_delta", np.nan) for c in new_candidates])
    return rows


def run_worker(job_path):
    """One fresh process: fit all configured models for one complete test year."""
    from core.batch_run import run_parquet
    from core.replication_report import make_report

    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    report = {"test_year": job["year"], "status": "failed", "run_dir": None,
              "memory_target_gib": job["memory_target_gib"]}
    started = perf_counter()
    exit_code = 0
    stop_monitor = Event()
    monitor = Thread(target=watch_memory, args=(stop_monitor, job_path.with_name("memory_latest.json"), job["year"]),
                     daemon=True)
    monitor.start()
    try:
        cfg = CfgNode(job["config"])
        panel = ParquetPanel(job["parquet_dir"], batch_size=cfg.DATA.BATCH_SIZE)
        folder = create_run(cfg, panel.info)
        report["run_dir"] = str(folder)
        print(f"Verification model checkpoints: {folder}", flush=True)
        result = run_parquet(panel, cfg, checkpoint_dir=folder)
        if cfg.RUN.get("REPORT_TABLE1_FIGURE3", False):
            make_report(folder)
        if len(result.annual_metrics) != len(cfg.RUN.MODELS.ENABLED):
            raise ValueError("Not all requested model-years completed.")
        report["status"] = "complete"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        exit_code = 1
    finally:
        stop_monitor.set()
        monitor.join()
        report["elapsed_seconds"] = perf_counter() - started
        counters = process_memory()
        report["memory_bytes"] = counters
        peak = counters["peak_resident_bytes"] / 1024**3 if counters else None
        report["peak_resident_gib"] = peak
        report["memory_target_passed"] = peak < job["memory_target_gib"] if peak is not None else None
        atomic_write(job_path.with_name("measurement.json"), json.dumps(report, indent=2, allow_nan=False))
        print(f"Year {job['year']}: status={report['status']}, peak resident GiB={peak}, "
              f"below target={report['memory_target_passed']}", flush=True)
    return exit_code


def verify(args):
    if (not np.isfinite(args.target_gib) or args.target_gib <= 0 or args.batch_size <= 0 or
            not np.isfinite(args.atol) or args.atol < 0 or not np.isfinite(args.rtol) or args.rtol < 0):
        raise ValueError("Memory target/batch size must be positive; tolerances finite and nonnegative.")
    comparing = args.mode == "compare"
    reference_folder = resolve_run_dir(args.reference_run) if comparing else None
    cfg = load_run_config(reference_folder) if comparing else load_config(args.config)
    if set(cfg.RUN.MODELS.ENABLED) != {"PCR", "PLS", "OLS_HUBER"}:
        raise ValueError("This verification requires all three models: PCR, PLS, OLS_HUBER.")
    parquet_dir = (resolve_run_dir(args.parquet_dir) if args.parquet_dir else
                   DATASET_PATHS[cfg.DATA.DATASET])
    panel = ParquetPanel(parquet_dir, batch_size=args.batch_size)
    reference = None
    if comparing:
        check_reference(panel, reference_folder, cfg)
        reference = read_checkpoints(reference_folder, cfg, lazy=True)
        available = sorted({year for year, _ in reference})
        years = sorted(set(args.years)) if args.years else available
        if not years or any((year, name) not in reference for year in years for name in cfg.RUN.MODELS.ENABLED):
            raise ValueError("Every requested comparison year needs saved checkpoints for all three models.")
    else:
        if cfg.DATA.get("FORMAT", "csv") != "parquet":
            raise ValueError("Memory verification requires a Parquet workflow config.")
        if panel.info["name"] != cfg.DATA.DATASET:
            raise ValueError("Parquet dataset identity differs from the memory-test config.")
        years = sorted(set(args.years or [cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR]))
    if any(not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= year <= cfg.RUN.SPLIT.LAST_TEST_YEAR for year in years):
        raise ValueError("Verification years must be within the saved/configured test horizon.")
    # Preflight all windows before starting expensive fits. Counts identify the
    # actual largest expanding training sample, including ties if years are empty.
    counts = {year: panel.window(cfg.RUN.SPLIT.FIRST_TRAIN_YEAR,
                                year - cfg.RUN.SPLIT.VALIDATION_YEARS - 1).n_rows
              for year in range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1)}
    for year in years:
        panel.window(year - cfg.RUN.SPLIT.VALIDATION_YEARS, year - 1)
        panel.window(year, year)
    largest = max(counts.values())
    largest_years = [year for year, count in counts.items() if count == largest]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = resolve_run_dir(args.output_dir) / f"{args.mode}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": args.mode, "reference_run": str(reference_folder) if comparing else None,
              "parquet_dir": str(parquet_dir), "years": years, "batch_size": args.batch_size,
              "atol": args.atol, "rtol": args.rtol, "memory_target_gib": args.target_gib,
              "largest_training_rows": largest, "largest_training_test_years": largest_years,
              "largest_window_selected": bool(set(years) & set(largest_years)),
              "complete_reference_horizon": years == available if comparing else None,
              "verification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": [], "status": "running"}
    atomic_write(output / "report.json", json.dumps(report, indent=2))
    checks = []
    print(f"Verification output: {output}\nYears: {years}; largest training sample: {largest:,} rows "
          f"for test year(s) {largest_years}", flush=True)
    for year in years:
        job_folder = output / str(year)
        job_folder.mkdir()
        # Preserve the reference model/evaluation settings exactly, replacing
        # only backend/output identity and restricting execution to this year.
        job_cfg = CfgNode(yaml.safe_load(cfg.dump()))
        job_cfg.DATA.DATASET = panel.info["name"]
        job_cfg.DATA.FORMAT, job_cfg.DATA.BATCH_SIZE = "parquet", args.batch_size
        job_cfg.RUN.SPLIT.FIRST_TEST_YEAR = job_cfg.RUN.SPLIT.LAST_TEST_YEAR = year
        job_cfg.RUN.OUTPUT_DIR = str(job_folder / "runs")
        job = {"year": year, "parquet_dir": str(parquet_dir), "config": yaml.safe_load(job_cfg.dump()),
               "memory_target_gib": args.target_gib}
        job_path = job_folder / "job.json"
        atomic_write(job_path, json.dumps(job, indent=2))
        # Inherit the console for live progress; never use shell=True. Running
        # years serially prevents several memory-heavy workers at once.
        completed = subprocess.run([sys.executable, "-B", "-m", "core.data_examine.verify_batch",
                                    "_worker", str(job_path)], cwd=root_path, check=False)
        measurement_path = job_folder / "measurement.json"
        measurement = (json.loads(measurement_path.read_text(encoding="utf-8")) if measurement_path.exists()
                       else {"test_year": year, "status": "failed", "error": "Worker exited without a measurement",
                             "memory_target_passed": None})
        measurement["worker_exit_code"] = completed.returncode
        if completed.returncode != 0:
            measurement["status"] = "failed"
        if comparing and measurement["status"] == "complete":
            try:
                actual_cfg = load_run_config(measurement["run_dir"])
                actual = read_checkpoints(measurement["run_dir"], actual_cfg, lazy=True)
                for name in cfg.RUN.MODELS.ENABLED:
                    rows = compare_records(reference[year, name], actual[year, name], atol=args.atol, rtol=args.rtol)
                    checks.extend({"test_year": year, "model": name, **row} for row in rows)
                measurement["correctness_passed"] = all(row["passed"] for row in checks if row["test_year"] == year)
            except Exception as error:
                measurement["correctness_passed"] = False
                measurement["comparison_error"] = f"{type(error).__name__}: {error}"
        report["results"].append(measurement)
        report.update(memory_summary(report))
        atomic_write(output / "comparison.csv", pd.DataFrame(checks).to_csv(index=False))
        atomic_write(output / "report.json", json.dumps(report, indent=2, allow_nan=False))
        # A failed fit or comparison is a reason to investigate before spending
        # hours on the remaining windows. Partial artifacts remain reviewable.
        if measurement["status"] != "complete" or (comparing and not measurement.get("correctness_passed")):
            break
    all_finished = len(report["results"]) == len(years) and all(r["status"] == "complete" for r in report["results"])
    report["correctness_passed"] = (all_finished and all(r.get("correctness_passed") for r in report["results"])) if comparing else None
    report.update(memory_summary(report))
    report["reference_horizon_completed"] = all_finished and report["complete_reference_horizon"] if comparing else None
    passed = (report["correctness_passed"] if comparing else
              report["memory_target_passed"] is True and report["largest_window_completed"])
    report["status"] = "passed" if passed else "failed"
    atomic_write(output / "report.json", json.dumps(report, indent=2, allow_nan=False))
    print(f"Verification {report['status']}. Report: {output / 'report.json'}", flush=True)
    if comparing:
        print(f"Correctness={report['correctness_passed']}; full saved horizon completed={report['reference_horizon_completed']}. "
              f"Details: {output / 'comparison.csv'}", flush=True)
    print_memory_summary(report)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    summary = commands.add_parser("summary", help="Show peak RAM from an existing verification; no fitting.")
    summary.add_argument("--run-dir", required=True, help="Directory containing verification report.json.")
    compare = commands.add_parser("compare", help="Refit Parquet data and compare to saved dense checkpoints.")
    compare.add_argument("--reference-run", required=True, help="Existing timestamped 5% run directory.")
    compare.add_argument("--parquet-dir", required=True, help="Conversion of exactly that run's 5% dataset.")
    memory = commands.add_parser("memory", help="Measure complete first/largest full-universe annual fits.")
    memory.add_argument("--config", default="gkx_table1_figure3_v4")
    memory.add_argument("--parquet-dir", help="Defaults to the dataset directory registered for the config.")
    for command in (compare, memory):
        command.add_argument("--years", type=int, nargs="+", help="Compare defaults to all saved years; memory to first/last.")
        command.add_argument("--batch-size", type=int, default=10000)
        command.add_argument("--target-gib", type=float, default=8., help="Peak resident RAM target, not an enforced cap.")
        command.add_argument("--atol", type=float, default=1e-7)
        command.add_argument("--rtol", type=float, default=1e-6)
        command.add_argument("--output-dir", default="results/batch_verification")
    worker = commands.add_parser("_worker", help="Internal fresh-process worker; invoked by compare/memory.")
    worker.add_argument("job_path")
    args = parser.parse_args(argv)
    if args.mode == "summary":
        folder = resolve_run_dir(args.run_dir)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        print(f"Saved verification: {report.get('status')}; correctness={report.get('correctness_passed')}", flush=True)
        print_memory_summary(report)
        return 0
    return run_worker(args.job_path) if args.mode == "_worker" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
