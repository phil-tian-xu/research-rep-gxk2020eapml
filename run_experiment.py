"""Local/SCRP PCR, PLS, ENet+H and OLS-3+H. See docs/Running Experiments.md."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from core.annual_jobs import collect, load_plan, prepare
from core.data_query.datasets import DATASET_PATHS
from core.run_state import atomic_write, load_run_config, read_checkpoints, run_lock, runtime_identity
from utils.general_utils import load_config


ROOT = Path(__file__).resolve().parent
BASELINES = {"pcr": "gkx_pcr_full_v5", "pls": "gkx_pls_p05_v1",
             "enet": "gkx_enet_huber_v1", "ols3": "gkx_ols3_huber_v1"}
MODEL_NAMES = {"pcr": "PCR", "pls": "PLS", "enet": "ENET_HUBER", "ols3": "OLS3_HUBER"}
SAMPLES = {"full": "gkx_parquet", "p05": "gkx_dev_p05_seed2020_parquet"}


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def results_root(value=None, *, scrp=False):
    if value or os.environ.get("GKX_EXPERIMENT_ROOT"):
        return Path(value or os.environ["GKX_EXPERIMENT_ROOT"]).expanduser().resolve()
    if scrp or shutil.which("sbatch"):
        # Reuse this project's existing deployment setting; no second site config.
        return Path(load_config("gkx_pcr_scrp").RUN.OUTPUT_DIR).parent / "experiments"
    return ROOT / "results" / "experiments"


def resolve_experiment(value, root=None):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    path = results_root(root) / value
    if not path.is_dir():
        raise ValueError(f"Experiment not found: {value}. Use list or provide its full path.")
    return path.resolve()


def read_launch(folder):
    record = json.loads((folder / "launch.json").read_text(encoding="utf-8"))
    if record["schema_version"] != 1:
        raise ValueError("Unsupported launcher record version.")
    return record


def year_range(value):
    if not re.fullmatch(r"\d{4}(:\d{4})?", value):
        raise argparse.ArgumentTypeError("Use YEAR or FIRST:LAST, e.g. 2016 or 1987:2016.")
    years = list(map(int, value.split(":")))
    first, last = years[0], years[-1]
    if not 1987 <= first <= last <= 2016:
        raise argparse.ArgumentTypeError("Choose a test horizon within 1987–2016.")
    return first, last


def execution_options(args, previous=None):
    options = dict(previous or {})
    for key in ("parallel", "cpus", "mem", "time", "account", "partition", "qos"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    options.setdefault("parallel", 4 if args.scrp else 1)
    options.setdefault("cpus", 8)
    model = getattr(args, "command", "")
    options.setdefault("mem", {"pcr": "128G", "ols3": "8G"}.get(model, "192G"))
    options.setdefault("time", {"pcr": "02:00:00", "ols3": "01:00:00"}.get(model, "12:00:00"))
    options.setdefault("account", "pg")
    options.setdefault("partition", "scrp")
    options.setdefault("qos", "c32g4")
    if options["cpus"] < 1 or options["parallel"] < 1:
        raise ValueError("CPUs and parallel tasks must be positive.")
    if not re.fullmatch(r"[1-9]\d*[MG]", options["mem"]):
        raise ValueError("Memory must be a positive amount in M or G, e.g. 128G.")
    if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", options["time"]):
        raise ValueError("Time must use HH:MM:SS or D-HH:MM:SS.")
    if not args.scrp and options["parallel"] != 1:
        raise ValueError("Local execution runs annual tasks sequentially; use --parallel 1.")
    return options


def prepare_experiment(args, argv):
    options = execution_options(args)
    if not args.scrp and not args.dry_run and shutil.which("sbatch") and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use --scrp on a cluster login node, or run --local inside a compute allocation.")
    if args.scrp and not args.dry_run and not shutil.which("sbatch"):
        raise ValueError("Run --scrp on the SCRP login node with sbatch available. No SSH submission is performed.")
    config_name = args.config or BASELINES[args.command]
    cfg = load_config(config_name)
    if list(cfg.RUN.MODELS.ENABLED) != [MODEL_NAMES[args.command]]:
        raise ValueError("The research config must enable only the requested model.")
    cfg.DATA.DATASET = SAMPLES[args.sample]
    if not (DATASET_PATHS[cfg.DATA.DATASET] / "metadata.json").is_file():
        raise ValueError(f"Prepared dataset is missing: {DATASET_PATHS[cfg.DATA.DATASET]}")
    first, last = args.years
    if not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= first <= last <= cfg.RUN.SPLIT.LAST_TEST_YEAR:
        raise ValueError("Requested years exceed this research config's test horizon.")
    run_id = f"{args.command}_{args.sample}_{first}-{last}_{timestamp()}"
    folder = results_root(args.root, scrp=args.scrp) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    # This resolved input and generated annual configs live with results, not config/.
    atomic_write(folder / "research.yaml", cfg.dump())
    prepare(str((folder / "research").resolve()), first, last, folder / "annual")
    record = {"schema_version": 1, "id": run_id, "model": MODEL_NAMES[args.command],
              "sample": args.sample, "years": [first, last], "baseline_config": config_name,
              "backend": "scrp" if args.scrp else "local", "created_utc": timestamp(),
              "command": [sys.executable, str(Path(__file__).resolve()), *argv],
              "repository": str(ROOT), "python": sys.executable,
              "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "resources": options, "submissions": []}
    try:
        record["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        record["git_status_at_preparation"] = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        record["git_commit"] = None
    atomic_write(folder / "launch.json", json.dumps(record, indent=2))
    print(f"Experiment: {folder}\nID: {run_id}", flush=True)
    return folder


def saved_years(folder, *, verify=False):
    _, plan = load_plan(folder / "annual")
    saved = []
    for year in plan["years"]:
        runs = sorted((folder / "annual" / "years" / str(year)).glob("*/manifest.json"))
        if len(runs) > 1:
            raise ValueError(f"Duplicate run directories for {year}.")
        if not runs:
            continue
        run_dir = runs[0].parent
        model, = plan["config"]["RUN"]["MODELS"]["ENABLED"]
        if (run_dir / "years" / str(year) / f"{model}.json").is_file():
            if verify:
                records = read_checkpoints(run_dir, load_run_config(run_dir), lazy=True)
                records[year, model]  # Validate the checksum before omitting a completed year.
            saved.append(year)
    return saved


def ensure_inactive(record):
    jobs = [s["job_id"] for s in record["submissions"] if s.get("job_id")]
    if any(s["state"] in {"submitting", "unknown"} for s in record["submissions"]):
        raise ValueError("An earlier submission has an uncertain outcome. Reconcile it with Slurm before retrying.")
    if not jobs:
        return
    result = subprocess.run(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%F"],
                            check=True, capture_output=True, text=True, timeout=30)
    if set(result.stdout.split()) & set(jobs):
        raise ValueError("This experiment still has queued/running Slurm tasks. Wait for them before resuming.")


def execute(folder, args, argv):
    with run_lock(folder):
        record = read_launch(folder)
        args.scrp = record["backend"] == "scrp"
        options = execution_options(args, record["resources"])
        if args.after and not args.scrp:
            raise ValueError("--after applies only to SCRP submissions.")
        _, plan = load_plan(folder / "annual")
        if (plan["runtime"] != runtime_identity() or plan["orchestrator_sha256"] !=
                hashlib.sha256((ROOT / "core" / "annual_jobs.py").read_bytes()).hexdigest()):
            raise ValueError("Execution code/environment changed; use the original checkout/environment to resume.")
        completed = saved_years(folder, verify=True)
        indices = [i for i, year in enumerate(plan["years"]) if year not in completed]
        if not indices:
            print("Every annual checkpoint is saved. Use collect to generate/locate reports.")
            return
        if not args.dry_run and args.scrp:
            ensure_inactive(record)
        submission_dir = folder / "submissions" / timestamp()
        submission_dir.mkdir(parents=True)
        attempt = {"created_utc": timestamp(), "state": "prepared", "indices": indices,
                   "resources": options, "command": argv, "directory": str(submission_dir)}
        record["submissions"].append(attempt)
        record["resources"] = options
        record_path = folder / "launch.json"
        if args.scrp:
            script = submission_dir / "run_annual_scrp.sh"
            # Snapshot the generic script without changing the legacy entry point.
            script.write_bytes((ROOT / "run_annual_scrp.sh").read_bytes().replace(b"\r\n", b"\n"))
            array = ",".join(map(str, indices)) + f"%{options['parallel']}"
            command = ["sbatch", "--parsable", f"--array={array}",
                       f"--cpus-per-task={options['cpus']}", f"--mem={options['mem']}",
                       f"--time={options['time']}", f"--account={options['account']}",
                       f"--partition={options['partition']}", f"--qos={options['qos']}",
                       f"--job-name=gkx-{record['model'].lower()}",
                       f"--output={submission_dir}/slurm-%A_%a.out"]
            if args.after:
                command.append(f"--dependency=afterany:{args.after}")
            command += [str(script), str(folder / "annual")]
            attempt["sbatch_command"] = command
            atomic_write(record_path, json.dumps(record, indent=2))
            print(shlex.join(command), flush=True)
            if args.dry_run:
                print("Prepared only; no job submitted. Use resume with this experiment ID to submit.")
                return
            attempt["state"] = "submitting"
            atomic_write(record_path, json.dumps(record, indent=2))
            try:
                result = subprocess.run(command, cwd=ROOT, check=True, text=True,
                                        capture_output=True, timeout=60)
                attempt["stdout"], attempt["stderr"] = result.stdout, result.stderr
                if not re.fullmatch(r"\d+(?:;[\w.-]+)?", result.stdout.strip()):
                    raise ValueError("Unrecognized sbatch response; check Slurm before submitting again.")
                attempt["job_id"] = result.stdout.strip().split(";")[0]
                attempt["state"] = "submitted"
            except (OSError, subprocess.CalledProcessError) as error:
                attempt["state"] = "submission_failed"
                attempt["error"] = str(error)
                attempt["stderr"] = getattr(error, "stderr", None)
                raise
            except BaseException as error:
                attempt["state"] = "unknown"
                attempt["error"] = str(error)
                raise
            finally:
                atomic_write(record_path, json.dumps(record, indent=2))
            print(f"Submitted job {attempt['job_id']}. Results: {folder}")
        else:
            if not args.dry_run and shutil.which("sbatch") and not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("Local fitting requires a compute allocation on SCRP.")
            atomic_write(record_path, json.dumps(record, indent=2))
            if args.dry_run:
                print("Prepared only; no fitting performed. Use resume with this experiment ID to run.")
                return
            env = os.environ.copy()
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = str(options["cpus"])
            env["MPLBACKEND"] = "Agg"
            attempt["state"] = "running"
            atomic_write(record_path, json.dumps(record, indent=2))
            try:
                for index in indices:
                    log = submission_dir / f"{plan['years'][index]}.log"
                    print(f"Running {plan['years'][index]}; log: {log}", flush=True)
                    with log.open("w", encoding="utf-8") as stream:
                        subprocess.run([sys.executable, "-u", "-B", "-m", "core.annual_jobs", "run",
                                        "--experiment", str(folder / "annual"), "--task-index", str(index)],
                                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                attempt["state"] = "complete"
            except BaseException as error:
                attempt["state"], attempt["error"] = "failed", str(error)
                raise
            finally:
                atomic_write(record_path, json.dumps(record, indent=2))
            collect(folder / "annual")


def scheduler_states(job_ids):
    """Read live array states, falling back to accounting for finished jobs."""
    if not job_ids:
        return {}
    try:
        live = subprocess.run(["squeue", "--array", "--noheader", "--user", getpass.getuser(),
                               "--format=%F|%T"], capture_output=True, text=True, check=True, timeout=30)
        states = defaultdict(Counter)
        for line in live.stdout.splitlines():
            job, state = line.strip().split("|", 1)
            states[job][state] += 1
        missing = [job for job in job_ids if job not in states]
        if missing:
            history = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(missing),
                                      "--format=JobID,State"], capture_output=True, text=True, check=True, timeout=30)
            for line in history.stdout.splitlines():
                job, state, *_ = line.strip().split("|")
                if "." not in job:
                    states[job.split("_")[0]][state] += 1
        return {job: ",".join(f"{state}:{count}" for state, count in sorted(states[job].items()))
                or "unknown" for job in job_ids}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Scheduler status unavailable: {error}", file=sys.stderr)
        return {job: "unknown" for job in job_ids}


def list_experiments(root, refresh=False):
    rows = []
    for path in sorted(root.glob("*/launch.json")):
        try:
            record = read_launch(path.parent)
            rows.append({"experiment": record["id"], "model": record["model"], "sample": record["sample"],
                         "years": f"{record['years'][0]}:{record['years'][1]}",
                         "saved": f"{len(saved_years(path.parent))}/{record['years'][1]-record['years'][0]+1}",
                         "reports": "ready" if (path.parent / "annual/combined/source_identity.json").is_file() else "pending",
                         "last_action": record["submissions"][-1]["state"] if record["submissions"] else "prepared",
                         "job_ids": ",".join(s["job_id"] for s in record["submissions"] if s.get("job_id")),
                         "path": str(path.parent)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Could not inspect {path}: {error}", file=sys.stderr)
    if refresh:
        jobs = sorted({job for row in rows for job in row["job_ids"].split(",") if job})
        states = scheduler_states(jobs)
        for row in rows:
            row["scheduler_now"] = "; ".join(f"{job}={states[job]}" for job in row["job_ids"].split(",") if job) or "n/a"
    if rows:
        import pandas as pd
        print(f"Results root: {root}\nEach experiment directory is ROOT/EXPERIMENT_ID.")
        print(pd.DataFrame(rows).drop(columns="path").to_string(index=False))
    else:
        print(f"No unified-launcher experiments in {root}.")
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in (*BASELINES, "resume"):
        cmd = commands.add_parser(name)
        if name == "resume":
            cmd.add_argument("experiment", help="Saved experiment ID or directory.")
            cmd.set_defaults(scrp=False)
        else:
            backend = cmd.add_mutually_exclusive_group(required=True)
            backend.add_argument("--local", dest="scrp", action="store_false")
            backend.add_argument("--scrp", action="store_true")
            cmd.add_argument("--sample", choices=SAMPLES, required=True)
            cmd.add_argument("--years", type=year_range, required=True)
            cmd.add_argument("--config", help="Optional versioned research config; defaults listed in the run guide.")
        cmd.add_argument("--root", help="Results root; otherwise GKX_EXPERIMENT_ROOT or the site default.")
        cmd.add_argument("--parallel", type=int, help="Maximum concurrent Slurm annual jobs (default 4); local uses 1.")
        cmd.add_argument("--cpus", type=int, help="CPUs/BLAS threads per annual task (default 8).")
        cmd.add_argument("--mem", help="RAM/task: PCR 128G, PLS/ENet 192G, OLS-3 8G; initial allocations.")
        cmd.add_argument("--time", help="Time/task: PCR 02:00:00, PLS/ENet 12:00:00, OLS-3 01:00:00.")
        for key in ("account", "partition", "qos"):
            cmd.add_argument(f"--{key}", help="Override the SCRP deployment default.")
        cmd.add_argument("--after", help="Start after this entire Slurm job/array ends, even if it fails.")
        cmd.add_argument("--dry-run", action="store_true", help="Save the plan and print submission; no fitting/submission.")
    listing = commands.add_parser("list")
    listing.add_argument("--root")
    listing.add_argument("--refresh", action="store_true", help="Also query Slurm; otherwise read saved records only.")
    report = commands.add_parser("collect")
    report.add_argument("experiment")
    report.add_argument("--root")
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if getattr(args, "after", None) and not re.fullmatch(r"\d+", args.after):
        raise ValueError("--after requires a numeric Slurm job/array ID.")
    if args.command in BASELINES and not args.scrp and args.after:
        raise ValueError("--after applies only to SCRP submissions.")
    if args.command == "list":
        return list_experiments(results_root(args.root), args.refresh)
    if args.command == "collect":
        folder = resolve_experiment(args.experiment, args.root)
        if (folder / "annual/combined/source_identity.json").is_file():
            print(f"Reports already saved: {folder / 'annual/combined'}")
        else:
            collect(folder / "annual")
        return
    folder = (resolve_experiment(args.experiment, args.root) if args.command == "resume"
              else prepare_experiment(args, argv))
    execute(folder, args, argv)
    return folder


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Experiment command failed: {error}", file=sys.stderr)
        sys.exit(1)
