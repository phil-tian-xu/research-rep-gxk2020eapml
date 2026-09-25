# Running experiments

Use the project Python environment. `run_experiment.py` is the normal interface
for **new PCR and PLS experiments**. It prepares annual configurations inside
the result directory and reuses the existing complete-array fitting workflow.
No server-only YAML, model-specific batch script, or manual array-index calculation
is needed. Run commands from the repository directory.

## Local development

```powershell
python -B run_experiment.py pcr --local --sample p05 --years 1987:1988
python -B run_experiment.py pls --local --sample p05 --years 1987:1988
python -B run_experiment.py list
```

Local annual tasks run sequentially in fresh Python processes with eight BLAS
threads by default (`--cpus` changes this). Each year's live output is written
to the printed log file. RAM measurements are recorded by the existing annual
worker; there is no local RAM cap. Combined reports are collected automatically
after all local years succeed. `--parallel` refers to Slurm jobs, not a local
multiprocessing pool; local runs require `--parallel 1`.

## SCRP

Run the command **on SCRP**, using `.venv/bin/python`. The wrapper calls `sbatch`
and records the response; it does not use SSH or fit models on the login node.
The existing generic Bash script executes the annual Python worker on compute
nodes. After submission returns a job ID, disconnecting does not stop the jobs.

Full PCR replication:

```bash
.venv/bin/python -B run_experiment.py pcr --scrp --sample full --years 1987:2016 \
  --parallel 4 --cpus 8 --mem 128G
```

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
| Research config | `gkx_pcr_full_v5` | `gkx_pls_p05_v1` |
| Tuning grid | K=0–100 | K=0–10 |
| SCRP concurrency | 4 | 4 |
| CPUs/task | 8 | 8 |
| RAM/task | 128G | 192G |
| Time/task | 02:00:00 | 12:00:00 |

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
