# Running experiments

New dense runs save the selected fitted model for post-run Figures 4 and 5.
PCR/PLS now save coefficients, intercept, feature order and fitted transforms;
RF saves its selected trees. ENet/OLS-3 and GLM already retain their fitted state.
Keep the complete annual checkpoints and matching prepared data. Generate
Figure 4 from saved fits, then Figure 5 from its annual scores using the
[Figures 4/5 workflow](working_notes/Figure%204%20Workflow.md).
Older PCR/PLS/RF checkpoints lack this state; use fresh runs after this update.

RF local pilot ([settings and outputs](working_notes/Random%20Forest%20Workflow.md)):

```powershell
python -B run_experiment.py rf --local --sample p05 --years 1987 --cpus 8
```

Uses `gkx_rf_v1`: 300 trees and 60 depth/feature-count candidates; saves annual
R2 and actual average tree depth. This is a one-year check with the full pilot grid.

GLM+H local pilot (approved settings and diagnostics in
[GLM Huber Workflow](working_notes/GLM%20Huber%20Workflow.md)):

```powershell
python -B run_experiment.py glm --local --sample p05 --years 1987 --cpus 8
```

Test one year before extending the horizon. GLM uses the same list/resume/collect
commands. Full-sample GLM memory and timing are not yet measured.

## Daily command reference: your PCR job 397900

This section uses Phil's actual submission on 2026-09-26 (Hong Kong time).
Submission was confirmed; completion is not assumed. Commands here run in
**Bash on SCRP**, not Windows PowerShell. Re-enter these variables after logging
in again. For another run, replace the experiment ID, job ID and date.

```bash
cd ~/research-rep-gxk2020eapml
exp_id="pcr_full_1987-2016_20260925T182454226202Z"
experiment="/data/users/s1155269217/GKX2020/results/experiments/$exp_id"
job=397900
since=2026-09-25
```

Experiment names contain UTC timestamps, so the date can differ from the local
date. `since` starts before submission and prevents historical accounting queries
from missing the run after midnight. A retry gets a new Slurm job ID but retains
the experiment directory.

| Need | Command |
| --- | --- |
| Find experiments | `.venv/bin/python -B run_experiment.py list` |
| Also query Slurm | `.venv/bin/python -B run_experiment.py list --refresh` |
| Check this array now | `squeue -r -j "$job" -o "%.20i %.10T %.10M %R"` |
| Check finished jobs | Use `sacct` below |
| See the current stage | Read the task's `run.log` below |
| Build Table 1/Figure 3 results | `.venv/bin/python -B run_experiment.py collect "$experiment"` |
| Retry unfinished years | `.venv/bin/python -B run_experiment.py resume "$experiment"` |

### Start a new experiment

The command that submitted job **397900** was:

```bash
.venv/bin/python -B run_experiment.py pcr --scrp --sample full \
  --years 1987:2016 --parallel 4 --cpus 8 --mem 128G
```

The returned identifiers were:

```text
ID: pcr_full_1987-2016_20260925T182454226202Z
Submitted job 397900
Experiment: /data/users/s1155269217/GKX2020/results/experiments/pcr_full_1987-2016_20260925T182454226202Z
```

**Do not repeat this command just to check or continue the run:** it creates a
new experiment. Use `list`, `resume` or `collect` for existing experiments.
Local checks and PLS examples are in the settings section below.

### Find results when only the job ID is remembered

```bash
.venv/bin/python -B run_experiment.py list
```

The job ID appears in `job_ids`. The directory is the printed results root plus
the experiment ID. To look up an exact job ID directly from saved records:

```bash
.venv/bin/python - "$job" <<'PY'
import json
import sys
from run_experiment import results_root

matches = []
for path in results_root().glob("*/launch.json"):
    record = json.loads(path.read_text())
    if any(s.get("job_id") == sys.argv[1] for s in record["submissions"]):
        matches.append(path.parent)
for path in matches:
    print(path)
if not matches:
    print("No unified-launcher record for this job in the configured results root.")
PY
```

This searches all recorded submissions, including retries. Legacy jobs launched
outside this interface are not included. For jobs still known to Slurm,
`scontrol show job 397900_0` can show the working directory, command and stdout
path. Slurm's live job record can disappear after completion.

### Is the job running, waiting, or finished?

```bash
# All my jobs, including other experiments
squeue -u "$USER" -o "%.20i %.10T %.12q %.10m %R"

# One row per annual task; full pending reason
squeue -r -j "$job" -o "%.20i %.10T %.10M %R"

# Refresh every 30 seconds; Ctrl+C stops watching, not the jobs
squeue -r -j "$job" -o "%.20i %.10T %.10M %R" --iterate=30

# Historical status, including completed/failed tasks; omit step rows
sacct -X -j "$job" --starttime "$since" \
  --format=JobID%22,State%24,ExitCode,Elapsed

# Check just tasks 0 and 1, including their steps
sacct -j "${job}_0,${job}_1" --starttime "$since" \
  --format=JobID%22,State%24,ExitCode,Elapsed
```

| Status | Meaning / next step |
| --- | --- |
| `PENDING` | Queued; read the reason column |
| `RUNNING` | Allocated and executing |
| `COMPLETING` | Slurm is cleaning up the task |
| `COMPLETED`, `0:0` | Process succeeded; collect and validate research outputs |
| `FAILED` | Inspect the worker log and exit code |
| `OUT_OF_MEMORY` | Inspect measured memory and requested allocation |
| `TIMEOUT` | The requested time limit was reached |
| `CANCELLED` | Cancelled; saved annual checkpoints remain |
| Empty queue / invalid job ID | Use `sacct`; absence does not prove success |

For **this 1987–2016 experiment**, task 0 is 1987, task 1 is 1988, task 22 is
2009, and task 29 is 2016. All 30 tasks need to finish. `.batch` and `.extern`
rows are job steps, not additional test years. The mapping is saved in
`annual/experiment.json`; the earlier eight-year experiment had a different mapping.

### Print the log and see the current stage

```bash
# For this experiment task 0 is 1987; use task=29 for 2016
task=0

# Locate the log without guessing the internal child job ID
find "$experiment/annual/logs" -type f \
  -path "*/${job}_${task}_*/run.log" -print

# Last 80 lines
tail -n 80 "$experiment"/annual/logs/"${job}_${task}_"*/run.log

# Follow output; Ctrl+C only stops following
tail -f "$experiment"/annual/logs/"${job}_${task}_"*/run.log
```

Progress messages include `Loading complete training window`, `Fitting`,
`Loading validation`, and `Saved; selected K=...; test R2=...`. A long gap during
SVD fitting is possible; there is no within-fit percentage indicator.
`Completed 1/1 model-years` refers to one annual task, not the entire array.

Pending tasks normally have no worker log yet. If a task started/failed before
creating `run.log`, check its scheduler stdout for an early launcher error:

```bash
find "$experiment/submissions" -type f -name "slurm-${job}_${task}.out" -print
tail -n 80 "$experiment"/submissions/*/"slurm-${job}_${task}.out"
```

For the original submission, task 0's exact scheduler stdout path is:

```text
/data/users/s1155269217/GKX2020/results/experiments/pcr_full_1987-2016_20260925T182454226202Z/submissions/20260925T182454778708Z/slurm-397900_0.out
```

An empty scheduler stdout file can be normal because the launcher redirects
Python output into `annual/logs/.../run.log`. After a retry use its **new job ID**.
If a pattern matches several files, inspect the paths individually. A wildcard
"no such file" error means that log does not exist yet; verify task status and ID.

### Check peak RAM and elapsed time

After the selected task ends:

```bash
cat "$experiment"/annual/logs/"${job}_${task}_"*/time.txt

# Keep .batch/step rows here: MaxRSS is commonly recorded on those rows
sacct -j "$job" --starttime "$since" \
  --format=JobID%22,State%24,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

GNU time's `Maximum resident set size (kbytes)` is an observed process peak.
Divide by **1,048,576** to obtain GiB on this Linux system: 100,952,100 KiB is
about 96.275 GiB. `ReqMem` is requested RAM, not actual use. Slurm MaxRSS is
accounting data and may differ from GNU time or be unavailable; blank is not zero.
GNU time's final summary may not exist until the task ends. Live worker logs
also print `[memory] ... MAX SO FAR=...`.

### Collect and read Table 1/Figure 3 outputs

After all tasks have ended and their checkpoints are saved:

```bash
.venv/bin/python -B run_experiment.py collect "$experiment"

# Pooled Table 1 R2
cat "$experiment/annual/combined/table1_all.csv"

# Annual R2, losses and sample sizes
head -n 12 "$experiment/annual/combined/annual_metrics.csv"

# Requested and fitted component counts over time
cat "$experiment/annual/combined/complexity.csv"

# Recorded per-year peak RAM and timing
cat "$experiment/annual/task_summary.csv"
```

Figure 3 is saved as `annual/combined/figure3_complexity.png` and `.svg`.
Use MobaXterm's SFTP browser to download/open the PNG. Validation diagnostics are
in `tuning_curves.csv` and `validation_YEAR.png`. Pooled R2 uses all test
observations; it is not an average of annual R2.

Collection refuses missing/incompatible years. If it reports missing years,
inspect `task_summary.csv` before resuming. `reports=pending` is normal before
collection. `last_action=submitted` records the launch action, not current job
state; use `--refresh` or `sacct` for scheduler status.

### Retry or cancel

Wait until the experiment's recorded submissions have no queued/running tasks:

```bash
.venv/bin/python -B run_experiment.py resume "$experiment"
```

If the evidence calls for more resources, use this **instead**:

```bash
.venv/bin/python -B run_experiment.py resume "$experiment" \
  --mem 192G --time 04:00:00 --parallel 2
```

Completed checkpoints are retained; only missing years are submitted. The new
job ID is added to `launch.json`. Keep the original code and environment for
resume. To cancel deliberately, choose **one** of these alternatives:

```bash
# Cancel one task (1987 in this example)
scancel "${job}_0"

# Cancel the entire array
scancel "$job"
```

Cancellation retains saved checkpoints but interrupts the current unsaved fit.
Do not cancel a healthy full run simply because other jobs are waiting for RAM.
For `QOSMaxMemoryPerUser`, inspect allocations and quota:

```bash
squeue -u "$USER" -o "%.20i %.10T %.12q %.10m %R"
sacctmgr -P show qos c32g4 format=Name,MaxTRESPU
```

Requested allocations count against the per-user limit, regardless of idle RAM
elsewhere in the cluster. Eight workers in one job still need their combined RAM.

### Tomorrow's short routine

After setting the variables at the top:

```bash
.venv/bin/python -B run_experiment.py list --refresh
sacct -X -j "$job" --starttime "$since" --format=JobID%22,State%24,ExitCode,Elapsed
```

If tasks failed, read their logs and address the cause before resuming. If all
30 annual tasks succeeded and their checkpoints are saved:

```bash
.venv/bin/python -B run_experiment.py collect "$experiment"
cat "$experiment/annual/combined/table1_all.csv"
cat "$experiment/annual/combined/complexity.csv"
```

References: [SCRP guide](https://scrp.econ.cuhk.edu.hk/guide/slurm),
[squeue](https://slurm.schedmd.com/squeue.html),
[sacct](https://slurm.schedmd.com/sacct.html).

## Launcher settings and background

Use the project Python environment. `run_experiment.py` is the normal interface
for **new PCR, PLS, ENet+H and OLS-3+H experiments**. It prepares annual configurations inside
the result directory and reuses the existing complete-array fitting workflow.
No server-only YAML, model-specific batch script, or manual array-index calculation
is needed. Run commands from the repository directory.

## Local development

```powershell
python -B run_experiment.py pcr --local --sample p05 --years 1987:1988
python -B run_experiment.py pls --local --sample p05 --years 1987:1988
python -B run_experiment.py enet --local --sample p05 --years 1987:1988
python -B run_experiment.py ols3 --local --sample p05 --years 1987:1988
python -B run_experiment.py list
```

Local annual tasks run sequentially in fresh Python processes with eight BLAS
threads by default (`--cpus` changes this). Each year's live output is written
to the printed log file. RAM measurements are recorded by the existing annual
worker; there is no local RAM cap. Combined reports are collected automatically
after all local years succeed. `--parallel` refers to Slurm jobs, not a local
multiprocessing pool; local runs require `--parallel 1`.

## SCRP

**Deployment note for the Huber additions:** Phil's PCR run is reported finished
and PLS is running. Do not update the checkout used by queued/running PLS tasks.
The new execution code changes source hashes. Finish/collect the existing run
using its original checkout, or deploy new methods in a separate checkout.
Ordinary resume intentionally refuses execution-code changes.

Run the command **on SCRP**, using `.venv/bin/python`. The wrapper calls `sbatch`
and records the response; it does not use SSH or fit models on the login node.
The existing generic Bash script executes the annual Python worker on compute
nodes. After submission returns a job ID, disconnecting does not stop the jobs.

Full PCR replication:

```bash
.venv/bin/python -B run_experiment.py pcr --scrp --sample full --years 1987:2016 \
  --parallel 4 --cpus 8 --mem 128G
```

ENet+H and OLS-3+H: first benchmark the largest annual window individually.
These are initial resource requests, not measured full-sample requirements:

ENet defaults to 192G/12 hours per task; OLS-3 defaults to 8G/1 hour. Both
inherit the launcher default of four concurrent SCRP tasks, so specify
`--parallel 1` explicitly for the first resource check.

```bash
.venv/bin/python -B run_experiment.py enet --scrp --sample full --years 2016 \
  --parallel 1 --cpus 8 --mem 192G --time 12:00:00
.venv/bin/python -B run_experiment.py ols3 --scrp --sample full --years 2016 \
  --parallel 1 --cpus 8 --mem 8G --time 01:00:00
```

After checking convergence, memory and runtime, use `--years 1987:2016` for a
fresh full-horizon experiment. `list`, `resume` and `collect` work identically
for all four methods. ENet produces `complexity.csv` and
`figure3_complexity.png` with nonzero slope counts, plus penalty validation
curves. OLS-3 has no tuned-complexity figure. Both produce pooled Table 1 and
annual metrics; coefficients and diagnostics are saved in each annual checkpoint
and run's `tuning_history.json`. See [Huber linear workflows](working_notes/Huber%20Linear%20Workflows.md)
for exact settings and interpretation.

PLS largest-window resource check, optionally queued after the current PCR array:

```bash
.venv/bin/python -B run_experiment.py pls --scrp --sample full --years 2016 \
  --parallel 1 --cpus 8 --mem 192G --time 12:00:00 --after 397886
```

Omit `--after` if no dependency is intended. Its meaning is `afterany`: wait for
the entire specified job/array to end, even if it fails. Job 397886 is an example
from Phil's current run; choose the appropriate current ID for future experiments.

Each array task gets the requested CPUs and RAM. `--parallel 4` caps concurrent
annual tasks at four; it does not guarantee four allocations. Defaults:

| Setting | PCR | PLS |
| --- | --- | --- |
| Research config | `gkx_pcr_full_v5` | `gkx_pls_v2` |
| Tuning grid | K=0–100 | K=0–29 |
| Additional predictor standardization | Off | Training-only standard deviation |
| SCRP concurrency | 4 | 4 |
| CPUs/task | 8 | 8 |
| RAM/task | 128G | 192G |
| Time/task | 02:00:00 | 12:00:00 |

PLS v2 enables training-only standardization and expands the grid; see
[PLS workflow](working_notes/PLS%20Small%20Sample%20Workflow.md) for evidence,
limitations and the two-year check. Start a new experiment to use v2: resuming
an existing experiment retains its saved specification. To reproduce the old
baseline, supply `--config gkx_pls_p05_v1` explicitly.

The PLS RAM/time defaults are initial test allocations, not measured requirements.
For its first full-universe run explicitly select one year and `--parallel 1`.
Account/partition/QoS default to `pg`/`scrp`/`c32g4`; the corresponding options
override them. `--mem`, `--time`, account, partition and QoS apply to SCRP, not
local process limits. Model fitting and sample settings are identical between
backends when the same research config, sample and years are selected.

The observed `QOSMaxMemoryPerUser` queue reason is consistent with a 512 GiB
per-user ceiling, but the exact quota still needs confirmation from account
output. Four 128G tasks reserve 512 GiB before counting other jobs. Allow Slurm
to queue tasks instead of reducing memory below the tested requirement.

`--dry-run` prepares a saved experiment and prints the submission command without
submitting or fitting. Submit that prepared experiment using `resume` below.
Do not repeat the original new-experiment command unless a separate experiment
is intended; each invocation gets a new unique ID.

## Find runs, retry, and collect

```bash
python -B run_experiment.py list
python -B run_experiment.py list --refresh
python -B run_experiment.py resume EXPERIMENT_ID
python -B run_experiment.py collect EXPERIMENT_ID
```

An absolute experiment directory can be used instead of its ID. The list shows
all unified-launcher experiments under the selected results root, one row per
experiment, including model, sample, horizon, saved-year count, report readiness,
recorded submission state, job IDs and path. It never loads feature data or
reruns a model. Saved-year counts indicate checkpoint files present; `collect`
performs checksum, identity and result validation before publishing reports.

`last_action` is what the launcher last recorded, not current scheduler status.
For example it can remain `submitted` after a job finishes. `--refresh` queries
`squeue`, then `sacct` for finished jobs, and adds a separate `scheduler_now`
column. Missing accounting or an unavailable Slurm connection means `unknown`,
not successful completion. The command does not run continuously in the background.

`resume` preserves research settings and the original backend, submits only
unfinished years, and stores each retry under the same experiment. For SCRP it
first checks that no recorded job is still queued/running. Completed checkpoints
are validated before being skipped. Execution code/package changes are rejected
by the existing identity checks. Resource options can change on a retry, e.g.:

```bash
python -B run_experiment.py resume EXPERIMENT_ID --mem 192G --parallel 2
```

A submission error is recorded. If sbatch times out or returns an unrecognized
response, the record is marked `unknown` and automatic retry is blocked: reconcile
the saved submission with Slurm first to avoid duplicate jobs. A lost response
does not prove that no job was submitted.

After SCRP jobs finish, `collect` explicitly generates pooled Table 1 R2, annual
metrics, selected components and Figure 3 plots under `annual/combined/`. It
refuses incomplete or incompatible results. It does not average annual R2.
For an existing complete report, the command prints its location without rewriting it.

## Remove an experiment from the active list

`remove` archives one explicitly selected experiment. It preserves all results,
checkpoints, settings and logs, and does not cancel jobs or erase Slurm history.
Run it on the cluster where the experiment was submitted:

```bash
cd ~/research-rep-gxk2020eapml
.venv/bin/python -B run_experiment.py remove pls_full_1987-1988_20260928T125719130953Z
.venv/bin/python -B run_experiment.py list --refresh
```

On SCRP this moves the selected folder from
`/data/users/s1155269217/GKX2020/results/experiments/` to
`/data/users/s1155269217/GKX2020/results/experiments_archive/`.
The normal list no longer scans it. The command prints both paths.

Use `--root PATH` for another results root, or `--aihub` for AI Hub. The archive
is always a sibling directory named `<results-root-name>_archive`; for example,
local `results/experiments` uses `results/experiments_archive`, and AI Hub uses
`results/aihub_experiments_archive`.

```bash
.venv/bin/python -B run_experiment.py remove EXPERIMENT_ID --aihub
.venv/bin/python -B run_experiment.py remove EXPERIMENT_ID --root /path/to/experiments
```

The argument is an exact folder name or absolute path directly under the selected
results root. For a manually renamed copy, use its actual folder name/path, not
the duplicate ID inside `launch.json`. Only that selected folder is moved.
Symlink experiment folders and existing archive destinations are refused.

The command checks the launcher record, existing execution/collection locks and,
for submitted cluster runs, the live Slurm queue. Active jobs, uncertain submissions,
or unavailable Slurm checks block archival. A completed run with saved results
may be archived too; nothing is permanently deleted. Do not launch, resume or
collect the same experiment concurrently with removal. On Windows, lock checks
finish immediately before the move because open lock files prevent renaming.

Archived records retain their original absolute paths. Before resuming or
collecting, move the folder back to its exact original location, provided that
location is unoccupied. Do not rewrite saved identities to relocate a run.
For example, on SCRP:

```bash
mv -n -T \
  /data/users/s1155269217/GKX2020/results/experiments_archive/EXPERIMENT_ID \
  /data/users/s1155269217/GKX2020/results/experiments/EXPERIMENT_ID
```

`remove` is a launcher-only change. To deploy it while other jobs are running,
transfer only the updated launcher and documentation; avoid a broad checkout
update that also changes the fitting code used by those jobs.

## Where everything is saved

Defaults:

- Local: the repository's `results/experiments/`.
- SCRP: `/data/users/s1155269217/GKX2020/results/experiments/`, derived from the
  existing `gkx_pcr_scrp` deployment config.

Use `--root PATH` on commands or set `GKX_EXPERIMENT_ROOT` once to choose another
location. `list` detects SCRP through the availability of `sbatch`; without Slurm,
use `--root` to inspect downloaded server results. No automatic synchronization
or remote discovery is performed.

```text
results/experiments/pcr_full_1987-2016_TIMESTAMP/
  launch.json                 command, resources, Git/code identity, submission history
  research.yaml               resolved baseline plus selected dataset
  submissions/TIMESTAMP/      exact batch script, submission command in launch.json, logs
  annual/
    experiment.json           effective research settings and environment/source identities
    configs/                  generated yearly configs; never maintained manually
    years/                    annual checkpoints and memory measurements
    logs/                     SCRP worker output and GNU time reports
    task_summary.csv          collected memory/time summary
    combined/                 pooled R2, annual metrics and Figure 3 diagnostics
```

The effective first/last test years are saved in `annual/experiment.json`; the
baseline's original horizon remains in `research.yaml`. The command and fully
resolved settings are preserved even if the baseline config is later edited.
Git commit/status and source hashes identify execution code, but do not archive
uncommitted source contents: commit research changes before production runs and
retain that checkout/environment for queued tasks and retries. Keep experiment
directories at their original paths while executing/resuming; copied directories
can be listed, but embedded absolute run paths are intentionally not migrated.

## AI Hub experiments

Use `--aihub` instead of `--scrp` for a new AI Hub experiment. Use `--aihub`
also on `list`, `resume ID` and `collect ID` to select its separate default root,
`results/aihub_experiments/`. Explicit `--root` or `GKX_EXPERIMENT_ROOT` wins.
AI Hub starts one annual task at a time and uses `run_annual_aihub.sh`; account,
partition and QoS are omitted unless supplied. SCRP defaults remain unchanged.
See [Running Python on AI Hub](Running%20Python%20on%20AI%20Hub.md) for setup,
resource checks and the deferred comparison tests. Prepare a fresh experiment
on AI Hub; do not migrate an unfinished SCRP experiment or its active checkout.

## Existing workflows

The legacy `main.py`, `core.annual_jobs`, `run_pcr_scrp.sh` and
`run_annual_scrp.sh` remain unchanged. Existing running jobs need no migration.
Finish or resume them through their original commands. Their result directories
do not have the new `launch.json` and are not silently added to `list`.

New experiments use the two baseline configs above; old pilot/memory configs
remain for provenance. Use `--config NAME` for an explicitly versioned research
specification, rather than creating configs just to change years or resources.
The CLI requires the selected model alone and bounds years to that config's horizon.

Sources: [SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm),
[Slurm sbatch](https://slurm.schedmd.com/sbatch.html).
