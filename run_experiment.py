"""Run annual research experiments by model, dataset and config. See docs/Running Experiments.md."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
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
from core.combined_experiments import (combination_plan, load_combination, collect_combination,
                                       load_source_bundle, source_directory, verify_source_bundle)
from core.data_query.datasets import DATASET_PATHS, validate_feature_dataset
from core.models.nn_architectures import ARCHITECTURES
from core.run_state import atomic_write, load_run_config, read_checkpoints, run_lock, runtime_identity
from utils.general_utils import load_config


ROOT = Path(__file__).resolve().parent
BASELINES = {"pcr": "gkx_pcr_full_v5", "pls": "gkx_pls_v2",
             "enet": "gkx_enet_huber_v1", "ols3": "gkx_ols3_huber_v1", "glm": "gkx_glm_huber_v1", "rf": "gkx_rf_v1"}
MODEL_NAMES = {"pcr": "PCR", "pls": "PLS", "enet": "ENET_HUBER", "ols3": "OLS3_HUBER", "glm": "GLM_HUBER", "rf": "RF"}
BASELINES.update({name.lower(): f"gkx_{name.lower()}_v1" for name in ARCHITECTURES})
MODEL_NAMES.update({name.lower(): name for name in ARCHITECTURES})
BASELINES.update({"ols": "gkx_ols_v1", "ols3-mse": "gkx_ols3_mse_v1"})
MODEL_NAMES.update({"ols": "OLS", "ols3-mse": "OLS3"})
SAMPLES = {"full": "gkx_parquet", "p05": "gkx_dev_p05_seed2020_parquet"}


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def results_root(value=None, *, scrp=False, aihub=False):
    if value or os.environ.get("GKX_EXPERIMENT_ROOT"):
        return Path(value or os.environ["GKX_EXPERIMENT_ROOT"]).expanduser().resolve()
    if aihub:
        return ROOT / "results" / "aihub_experiments"
    if scrp or shutil.which("sbatch"):
        # Reuse this project's existing deployment setting; no second site config.
        return Path(load_config("gkx_pcr_scrp").RUN.OUTPUT_DIR).parent / "experiments"
    return ROOT / "results" / "experiments"


def resolve_experiment(value, root=None, *, aihub=False):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    path = results_root(root, aihub=aihub) / value
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
    aihub = getattr(args, "aihub", False)
    options = dict(previous or {})
    for key in ("parallel", "cpus", "mem", "time", "account", "partition", "qos", "gpus", "gpu_type"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    model = getattr(args, "command", "")
    is_nn = MODEL_NAMES.get(model) in ARCHITECTURES or options.get("gpus", 0) > 0
    options.setdefault("parallel", 1 if is_nn else 4 if args.scrp else 1)
    options.setdefault("cpus", 8)
    options.setdefault("gpus", 1 if is_nn and (args.scrp or aihub) else 0)
    options.setdefault("gpu_type", "a100" if is_nn and args.scrp else "any" if is_nn and aihub else None)
    options.setdefault("mem", "64G" if is_nn else {"pcr": "128G", "ols3": "8G", "ols3-mse": "8G"}.get(model, "192G"))
    options.setdefault("time", {"pcr": "02:00:00", "ols3": "01:00:00", "ols3-mse": "01:00:00"}.get(model, "12:00:00"))
    options.setdefault("account", None if aihub else "pg")
    # A different GPU type on resume must not retain the previous type's partition.
    if (previous and getattr(args, "gpu_type", None) is not None and
            options["gpu_type"] != previous.get("gpu_type", "a100" if args.scrp else "any") and
            getattr(args, "partition", None) is None):
        options.pop("partition", None)
    options.setdefault("partition", None if aihub else
                       "a100" if is_nn and args.scrp and options["gpu_type"] == "a100" else "scrp")
    options.setdefault("qos", None if aihub else "c32g4")
    if options["cpus"] < 1 or options["parallel"] < 1:
        raise ValueError("CPUs and parallel tasks must be positive.")
    if options["gpus"] not in (0, 1):
        raise ValueError("Annual jobs support zero GPUs or one GPU per task; use --parallel for concurrent GPU jobs.")
    if options["gpu_type"] is not None and not options["gpus"]:
        raise ValueError("--gpu-type requires a cluster NN job with one GPU per task.")
    if not re.fullmatch(r"[1-9]\d*[MG]", options["mem"]):
        raise ValueError("Memory must be a positive amount in M or G, e.g. 128G.")
    if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", options["time"]):
        raise ValueError("Time must use HH:MM:SS or D-HH:MM:SS.")
    if aihub:
        memory_mib = int(options["mem"][:-1]) * (1024 if options["mem"][-1] == "G" else 1)
        days, _, clock = options["time"].rpartition("-")
        hours, minutes, seconds = map(int, clock.split(":"))
        duration = int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds
        if options["parallel"] * memory_mib > 240 * 1024:
            raise ValueError("AI Hub array requests exceed the documented 240G per-user memory budget; reduce --parallel.")
        if options["parallel"] * options["cpus"] > 60:
            raise ValueError("AI Hub array requests exceed the documented 60-CPU per-user budget.")
        if options["parallel"] * options["gpus"] > 4:
            raise ValueError("AI Hub array requests exceed the documented four-GPU per-user budget.")
        if not 0 < duration <= 86400 or minutes >= 60 or seconds >= 60:
            raise ValueError("AI Hub task time must be positive and no more than 24 hours.")
    if not (args.scrp or aihub) and options["parallel"] != 1:
        raise ValueError("Local execution runs annual tasks sequentially; use --parallel 1.")
    return options


def prepare_experiment(args, argv):
    options = execution_options(args)
    slurm = args.scrp or args.aihub
    if not slurm and not args.dry_run and shutil.which("sbatch") and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use --scrp or --aihub on a cluster login node, or run --local inside a compute allocation.")
    if slurm and not args.dry_run and not shutil.which("sbatch"):
        raise ValueError("Run on the selected cluster with sbatch available. No SSH submission is performed.")
    config_name = args.config or BASELINES[args.command]
    cfg = load_config(config_name)
    if list(cfg.RUN.MODELS.ENABLED) != [MODEL_NAMES[args.command]]:
        raise ValueError("The research config must enable only the requested model.")
    is_nn = MODEL_NAMES[args.command] in ARCHITECTURES
    if slurm and is_nn and (options["gpus"] != 1 or cfg.RUN.MODELS.NN.DEVICE != "cuda"):
        raise ValueError("Cluster NN jobs require one GPU and NN.DEVICE=cuda.")
    if not is_nn and options["gpus"]:
        raise ValueError("Only NN jobs use GPUs; do not allocate unused GPUs to CPU models.")
    dataset = args.dataset or SAMPLES[args.sample]
    cfg.DATA.DATASET = dataset
    if not (DATASET_PATHS[cfg.DATA.DATASET] / "metadata.json").is_file():
        raise ValueError(f"Prepared dataset is missing: {DATASET_PATHS[cfg.DATA.DATASET]}")
    if args.dataset:
        metadata = json.loads((DATASET_PATHS[dataset] / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("dataset_name") != dataset or metadata.get("format") != "parquet" or
                metadata.get("status") != "complete"):
            raise ValueError("--dataset requires a complete Parquet dataset with a matching metadata name.")
        validate_feature_dataset(metadata)
    sample = args.sample or next((key for key, value in SAMPLES.items() if value == dataset), dataset)
    first, last = args.years
    if not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= first <= last <= cfg.RUN.SPLIT.LAST_TEST_YEAR:
        raise ValueError("Requested years exceed this research config's test horizon.")
    run_id = f"{args.command}_{sample}_{first}-{last}_{timestamp()}"
    folder = results_root(args.root, scrp=args.scrp, aihub=args.aihub) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    # This resolved input and generated annual configs live with results, not config/.
    atomic_write(folder / "research.yaml", cfg.dump())
    prepare(str((folder / "research").resolve()), first, last, folder / "annual")
    record = {"schema_version": 1, "id": run_id, "model": MODEL_NAMES[args.command],
              "sample": sample, "dataset": dataset, "years": [first, last], "baseline_config": config_name,
              "backend": "aihub" if args.aihub else "scrp" if args.scrp else "local", "created_utc": timestamp(),
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


def active_jobs(record):
    jobs = [s["job_id"] for s in record["submissions"] if s.get("job_id")]
    if any(s["state"] in {"submitting", "unknown"} for s in record["submissions"]):
        raise ValueError("An earlier submission has an uncertain outcome. Reconcile it with Slurm before retrying.")
    if any(s["state"] == "submitted" and not s.get("job_id") for s in record["submissions"]):
        raise ValueError("Submitted job ID is missing; reconcile its status before retrying.")
    if not jobs:
        return set()
    result = subprocess.run(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%F"],
                            check=True, capture_output=True, text=True, timeout=30)
    return set(result.stdout.split()) & set(jobs)


def ensure_inactive(record):
    if active_jobs(record):
        raise ValueError("This experiment still has queued/running Slurm tasks. Wait for them before resuming.")


def combine_experiments(values, args):
    parents = [resolve_experiment(value, args.root, aihub=args.aihub) for value in values]
    if args.years is None:
        raise ValueError("Combining experiments requires --years FIRST:LAST to declare the expected coverage.")
    plan = combination_plan(parents, args.years)
    launches = [read_launch(parent) for parent in parents]
    if len({r["backend"] for r in launches}) != 1:
        raise ValueError("A combined run must use one execution backend; prepare its sources on the same cluster.")
    if len({r["sample"] for r in launches}) != 1:
        raise ValueError("Source sample labels differ.")
    validate_combined_resources(launches)
    for source, record in zip(plan["sources"], launches):
        expected_model, = source["plan"]["config"]["RUN"]["MODELS"]["ENABLED"]
        if record["model"] != expected_model or record["years"] != [source["plan"]["years"][0], source["plan"]["years"][-1]]:
            raise ValueError("Source launch record disagrees with its annual plan.")
    first = launches[0]
    run_id = f"combined_{first['model'].lower()}_{first['sample']}_{args.years[0]}-{args.years[1]}_{timestamp()}"
    folder = results_root(args.root, scrp=first["backend"] == "scrp", aihub=first["backend"] == "aihub") / run_id
    folder.mkdir(parents=True, exist_ok=False)
    atomic_write(folder / "collection.json", json.dumps(plan, indent=2))
    record = dict(schema_version=1, kind="combined", id=run_id, model=first["model"],
                  sample=first["sample"], years=list(args.years), backend=first["backend"],
                  created_utc=timestamp(), resources={}, submissions=[])
    atomic_write(folder / "launch.json", json.dumps(record, indent=2))
    print(f"Combined experiment: {folder}\nID: {run_id}", flush=True)
    return folder


def validate_combined_resources(records):
    if records[0]["backend"] == "aihub":
        options = [r["resources"] for r in records]
        memory = sum(o["parallel"] * int(o["mem"][:-1]) * (1024 if o["mem"][-1] == "G" else 1) for o in options)
        cpus = sum(o["parallel"] * o["cpus"] for o in options)
        gpus = sum(o["parallel"] * o["gpus"] for o in options)
        if memory > 240 * 1024 or cpus > 60 or gpus > 4:
            raise ValueError("Combined AI Hub resource pools exceed the 240G/60-CPU/four-GPU per-user budget.")


def resume_combination(folder, args, argv):
    overrides = ("parallel", "cpus", "mem", "time", "account", "partition", "qos", "gpus", "gpu_type")
    if any(getattr(args, key, None) is not None for key in overrides):
        raise ValueError("Combined resume preserves each source's resources. Change resources with resume SOURCE_ID first.")
    with run_lock(folder):
        plan = load_combination(folder)
        if args.aihub and read_launch(folder)["backend"] != "aihub":
            raise ValueError("--aihub cannot resume an experiment from another backend.")
        validate_combined_resources([read_launch(source_directory(folder, s)) for s in plan["sources"]])
        bundled = load_source_bundle(folder) is not None
        pending = []
        # Preflight every source before any submission; one uncertain receipt stops the run.
        for source in plan["sources"]:
            parent = source_directory(folder, source)
            source_plan, record = source["plan"], read_launch(parent)
            completed = saved_years(parent, verify=True)
            if completed == source_plan["years"]:
                continue
            if bundled:
                raise ValueError("Embedded source is incomplete; restore the combined snapshot. It cannot submit fitting jobs.")
            if (source_plan["runtime"] != runtime_identity() or source_plan["orchestrator_sha256"] !=
                    hashlib.sha256((ROOT / "core" / "annual_jobs.py").read_bytes()).hexdigest()):
                raise ValueError("Execution code/environment changed; use the original checkout/environment to resume.")
            if record["backend"] == "local":
                if any(s["state"] in {"running", "submitting", "unknown"} for s in record["submissions"]):
                    raise ValueError("A local source is running or uncertain; reconcile it before resuming.")
            elif not args.dry_run and active_jobs(record):
                print(f"Source already queued/running; skipped: {parent.name}")
                continue
            pending.append(parent)
        for parent in pending:
            execute(parent, argparse.Namespace(**vars(args)), argv)
    if not args.dry_run and all(saved_years(source_directory(folder, s)) == s["plan"]["years"] for s in plan["sources"]):
        collect_combination(folder)
    else:
        print(f"Track this combined ID with status; collect when complete: {folder.name}")


def execute(folder, args, argv):
    with run_lock(folder):
        record = read_launch(folder)
        if args.aihub and record["backend"] != "aihub":
            raise ValueError("--aihub cannot resume an experiment from another backend; prepare a separate experiment.")
        args.scrp = record["backend"] == "scrp"
        args.aihub = record["backend"] == "aihub"
        slurm = args.scrp or args.aihub
        options = execution_options(args, record["resources"])
        if slurm and record["model"] in ARCHITECTURES and options["gpus"] != 1:
            raise ValueError("Cluster NN jobs require one GPU per annual task.")
        if record["model"] not in ARCHITECTURES and options["gpus"]:
            raise ValueError("Only NN jobs use GPUs.")
        if args.after and not slurm:
            raise ValueError("--after applies only to Slurm submissions.")
        _, plan = load_plan(folder / "annual")
        if (plan["runtime"] != runtime_identity() or plan["orchestrator_sha256"] !=
                hashlib.sha256((ROOT / "core" / "annual_jobs.py").read_bytes()).hexdigest()):
            raise ValueError("Execution code/environment changed; use the original checkout/environment to resume.")
        completed = saved_years(folder, verify=True)
        indices = [i for i, year in enumerate(plan["years"]) if year not in completed]
        if not indices:
            print("Every annual checkpoint is saved. Use collect to generate/locate reports.")
            return
        if not args.dry_run and slurm:
            ensure_inactive(record)
        submission_dir = folder / "submissions" / timestamp()
        submission_dir.mkdir(parents=True)
        attempt = {"created_utc": timestamp(), "state": "prepared", "indices": indices,
                   "resources": options, "command": argv, "directory": str(submission_dir)}
        record["submissions"].append(attempt)
        record["resources"] = options
        record_path = folder / "launch.json"
        if slurm:
            scrp_nn = args.scrp and record["model"] in ARCHITECTURES
            script_name = ("run_annual_aihub.sh" if args.aihub else
                           "run_nn_scrp.sh" if scrp_nn else "run_annual_scrp.sh")
            script = submission_dir / script_name
            # Snapshot the generic script without changing the legacy entry point.
            script.write_bytes((ROOT / script_name).read_bytes().replace(b"\r\n", b"\n"))
            array = ",".join(map(str, indices)) + f"%{options['parallel']}"
            command = ["sbatch", "--parsable", f"--array={array}",
                       f"--cpus-per-task={options['cpus']}", f"--mem={options['mem']}",
                       f"--time={options['time']}",
                       *[f"--{key}={options[key]}" for key in ("account", "partition", "qos") if options[key]],
                       f"--job-name=gkx-{record['model'].lower()}",
                       f"--output={submission_dir}/slurm-%A_%a.out"]
            if options["gpus"]:
                gpu_request = (str(options["gpus"]) if options["gpu_type"] == "any" else
                               f"{options['gpu_type']}:{options['gpus']}")
                command.append(f"--gres=gpu:{gpu_request}" if args.aihub else f"--gpus={gpu_request}")
                if scrp_nn and options["gpu_type"] == "a100":
                    command.append("--constraint=a100")
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


def list_experiments(root, refresh=False, experiment=None, *, verify=False):
    rows = []
    paths = [experiment / "launch.json"] if experiment else sorted(root.glob("*/launch.json"))
    for path in paths:
        try:
            record = read_launch(path.parent)
            reports = "ready" if (path.parent / "annual/combined/source_identity.json").is_file() else "pending"
            source_status = "n/a"
            if record.get("kind") == "combined":
                submissions, saved = [], 0
                action = "combined"
                try:
                    plan = load_combination(path.parent)
                    parents = [source_directory(path.parent, s) for s in plan["sources"]]
                    submissions = [s for parent in parents for s in read_launch(parent)["submissions"]]
                    saved = sum(len(saved_years(parent)) for parent in parents)
                    bundle = verify_source_bundle(path.parent, checksums=verify)
                    source_status = ("verified" if verify else "present") if bundle else "linked"
                    if not bundle and reports == "ready":
                        reports = "linked"
                except (OSError, ValueError, KeyError, TypeError) as error:
                    source_status, reports = "incomplete", "incomplete"
                    print(f"Combined source bundle is incomplete: {record['id']}: {error}", file=sys.stderr)
            else:
                submissions = record["submissions"]
                saved = len(saved_years(path.parent))
                action = submissions[-1]["state"] if submissions else "prepared"
            rows.append({"experiment": record["id"], "model": record["model"], "sample": record["sample"],
                         "dataset": record.get("dataset", SAMPLES.get(record["sample"], record["sample"])),
                         "years": f"{record['years'][0]}:{record['years'][1]}",
                         "saved": f"{saved}/{record['years'][1]-record['years'][0]+1}",
                         "reports": reports, "sources": source_status,
                         "last_action": action,
                         "job_ids": ",".join(s["job_id"] for s in submissions if s.get("job_id")),
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


def remove_experiment(value, root=None, *, aihub=False):
    """Archive one inactive experiment without changing its saved identities."""
    root = results_root(root, aihub=aihub).resolve()
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = root / source
    if source.is_symlink() or source.resolve().parent != root:
        raise ValueError("Select a direct experiment folder under the results root; symlinks are not supported.")
    folder = source.resolve()
    if not folder.is_dir():
        raise ValueError(f"Experiment not found: {value}. Use list or provide its full path.")
    for descriptor in root.glob("*/collection.json"):
        if descriptor.parent != folder:
            plan = json.loads(descriptor.read_text(encoding="utf-8"))
            if any(Path(s["experiment"]).resolve() == folder for s in plan["sources"]):
                raise ValueError(f"Source is referenced by {descriptor.parent.name}; archive the combined run first.")
    archive = root.with_name(root.name + "_archive")
    if archive.is_symlink():
        raise ValueError("The archive directory must not be a symlink.")
    destination = archive / folder.name
    if os.path.lexists(destination):
        raise ValueError(f"Archive destination already exists: {destination}")

    with ExitStack() as locks:
        locks.enter_context(run_lock(folder))
        try:
            record = read_launch(folder)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid launcher record; cannot archive this directory.") from error
        if (not isinstance(record.get("id"), str) or
                record.get("backend") not in {"local", "scrp", "aihub"} or
                not isinstance(record.get("submissions"), list) or
                any(not isinstance(s, dict) or "state" not in s for s in record["submissions"])):
            raise ValueError("Invalid launcher record; cannot archive this directory.")
        if aihub and record["backend"] != "aihub":
            raise ValueError("--aihub cannot remove an experiment from another backend.")
        if record["backend"] == "local":
            if any(s["state"] in {"running", "submitting", "unknown"} for s in record["submissions"]):
                raise ValueError("Local execution is running or uncertain; reconcile its status before archiving.")
        else:
            if any(s["state"] == "submitted" and not s.get("job_id") for s in record["submissions"]):
                raise ValueError("Submitted job ID is missing; reconcile its status before archiving.")
            try:
                ensure_inactive(record)
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError("Cannot verify Slurm inactivity; run remove on the original cluster when squeue is available.") from error
        # Annual workers and collection use locks below the launcher directory.
        for lock in sorted(folder.rglob(".run.lock")):
            if lock.parent != folder:
                locks.enter_context(run_lock(lock.parent))
        archive.mkdir(exist_ok=True)
        # Serialize archive operations so another remove cannot claim this name.
        with run_lock(archive):
            if os.path.lexists(destination):
                raise ValueError(f"Archive destination already exists: {destination}")
            if os.name == "nt":
                # Windows cannot rename a directory containing our open lock files.
                # Check them first, then close immediately before the rename.
                locks.close()
            folder.rename(destination)
    print(f"Archived experiment (files preserved):\nFrom: {folder}\nTo:   {destination}")
    print("Move it back to the original path before resuming or collecting.")
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in (*BASELINES, "resume"):
        cmd = commands.add_parser(name)
        if name == "resume":
            cmd.add_argument("experiment", help="Saved experiment ID or directory.")
            cmd.set_defaults(scrp=False)
            cmd.add_argument("--aihub", action="store_true", help="Look up the ID in the AI Hub results root.")
        else:
            backend = cmd.add_mutually_exclusive_group(required=True)
            backend.add_argument("--local", dest="scrp", action="store_false")
            backend.add_argument("--scrp", action="store_true")
            backend.add_argument("--aihub", action="store_true")
            cmd.set_defaults(scrp=False)
            data = cmd.add_mutually_exclusive_group(required=True)
            data.add_argument("--sample", choices=SAMPLES, help="Existing full/p05 dataset shorthand.")
            data.add_argument("--dataset", choices=sorted(DATASET_PATHS), help="Explicit prepared Parquet dataset name.")
            cmd.add_argument("--years", type=year_range, required=True)
            cmd.add_argument("--config", help="Optional versioned research config; defaults listed in the run guide.")
        cmd.add_argument("--root", help="Results root; otherwise GKX_EXPERIMENT_ROOT or the site default.")
        cmd.add_argument("--parallel", type=int, help="Concurrent annual jobs / total GPUs for NN: NN defaults to 1; SCRP CPU models to 4.")
        cmd.add_argument("--cpus", type=int, help="CPUs/BLAS threads per annual task (default 8).")
        cmd.add_argument("--gpus", type=int, help="GPUs per annual task: NN cluster jobs use 1; CPU models use 0.")
        cmd.add_argument("--gpu-type", choices=("a100", "rtx3090", "rtx3060", "any"),
                         help="GPU model per NN task: SCRP defaults to a100, AI Hub to any. Use --parallel for the number of GPU jobs.")
        cmd.add_argument("--mem", help="RAM/task: PCR 128G, PLS/ENet 192G, OLS-3 8G; initial allocations.")
        cmd.add_argument("--time", help="Time/task: PCR 02:00:00, PLS/ENet 12:00:00, OLS-3 01:00:00.")
        for key in ("account", "partition", "qos"):
            cmd.add_argument(f"--{key}", help="Override site settings; AI Hub omits this flag unless supplied.")
        cmd.add_argument("--after", help="Start after this entire Slurm job/array ends, even if it fails.")
        cmd.add_argument("--dry-run", action="store_true", help="Save the plan and print submission; no fitting/submission.")
    listing = commands.add_parser("list")
    listing.add_argument("--root")
    listing.add_argument("--aihub", action="store_true")
    listing.add_argument("--refresh", action="store_true", help="Also query Slurm; otherwise read saved records only.")
    listing.add_argument("--verify", action="store_true", help="Also verify every embedded combined-source file's SHA-256.")
    report = commands.add_parser("collect")
    report.add_argument("experiment", nargs="+")
    report.add_argument("--years", type=year_range, help="Required when combining multiple source experiments.")
    report.add_argument("--root")
    report.add_argument("--aihub", action="store_true")
    report.add_argument("--repair-sources", action="store_true",
                        help="For a combined ID, restore frozen source files and shorten embedded paths; never refit.")
    combination = commands.add_parser("combine", help="Register disjoint source experiments as one logical run, before or after fitting.")
    combination.add_argument("experiment", nargs="+")
    combination.add_argument("--years", type=year_range, required=True)
    combination.add_argument("--root")
    combination.add_argument("--aihub", action="store_true")
    status = commands.add_parser("status", help="Inspect one experiment or combined run.")
    status.add_argument("experiment")
    status.add_argument("--root")
    status.add_argument("--aihub", action="store_true")
    status.add_argument("--refresh", action="store_true")
    status.add_argument("--verify", action="store_true", help="Verify every embedded combined-source file's SHA-256.")
    removal = commands.add_parser("remove", help="Move one inactive experiment to the sibling archive directory.")
    removal.add_argument("experiment", help="Exact folder name or absolute path under the results root.")
    removal.add_argument("--root")
    removal.add_argument("--aihub", action="store_true")
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if getattr(args, "after", None) and not re.fullmatch(r"\d+", args.after):
        raise ValueError("--after requires a numeric Slurm job/array ID.")
    if args.command in BASELINES and not (args.scrp or args.aihub) and args.after:
        raise ValueError("--after applies only to Slurm submissions.")
    if args.command == "list":
        rows = list_experiments(results_root(args.root, aihub=args.aihub), args.refresh, verify=args.verify)
        if args.verify and any(row["sources"] == "incomplete" for row in rows):
            raise ValueError("Combined source verification failed; inspect the incomplete rows above.")
        return rows
    if args.command == "status":
        folder = resolve_experiment(args.experiment, args.root, aihub=args.aihub)
        rows = list_experiments(folder.parent, args.refresh, folder, verify=args.verify)
        if args.verify and (not rows or rows[0]["sources"] == "incomplete"):
            raise ValueError("Combined source verification failed; use collect --repair-sources on the original server.")
        return rows
    if args.command == "combine":
        return combine_experiments(args.experiment, args)
    if args.command == "remove":
        return remove_experiment(args.experiment, args.root, aihub=args.aihub)
    if args.command == "collect":
        folder = (combine_experiments(args.experiment, args) if len(args.experiment) > 1 else
                  resolve_experiment(args.experiment[0], args.root, aihub=args.aihub))
        if (folder / "collection.json").is_file():
            if args.years and list(args.years) != read_launch(folder)["years"]:
                raise ValueError("Requested years differ from the combined run's fixed horizon.")
            return collect_combination(folder, repair_sources=args.repair_sources)
        if args.repair_sources:
            raise ValueError("--repair-sources applies only to a combined experiment ID.")
        if args.years:
            raise ValueError("--years applies to combined collection; a single source uses its saved horizon.")
        if (folder / "annual/combined/source_identity.json").is_file():
            print(f"Reports already saved: {folder / 'annual/combined'}")
        else:
            collect(folder / "annual")
        return
    folder = (resolve_experiment(args.experiment, args.root, aihub=args.aihub) if args.command == "resume"
              else prepare_experiment(args, argv))
    if (folder / "collection.json").is_file():
        resume_combination(folder, args, argv)
    else:
        execute(folder, args, argv)
    return folder


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Experiment command failed: {error}", file=sys.stderr)
        sys.exit(1)
