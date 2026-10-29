# SCRP command reference

Run these commands **on SCRP, in Bash**. Each project uses its own `.venv`.
Copy the individual block you need.

For the full research and transfer cycle, see the
[research experiment workflow](research_workflow.md).

The `run_experiment.py` commands assume the project uses our launcher. The Slurm
commands apply to other projects too. Model names, samples and years below are
examples from the current replication.

## 1. Enter the project

```bash
project_dir="$HOME/PROJECT_NAME"
cd "$project_dir"
```

Set this again after reconnecting. The launcher uses the project's configured
SCRP results root; `list` prints its location. Use `--root /PATH/TO/EXPERIMENTS`
consistently if overriding that location.

## 2. Run an experiment

Create and submit a **new experiment**. Example for a GPU model:

```bash
.venv/bin/python -B run_experiment.py nn4 --scrp \
  --sample full --years 1987:2016 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 8 --mem 96G --time 12:00:00
```

Change the model, sample, years and resources as needed.

- Add `--dry-run` to prepare without submitting.
- Use `--gpu-type a100` for A100 jobs; the launcher selects the A100 partition and constraint.
- CPU models omit the GPU flags and use appropriate CPU/RAM allocations.
- Resource requests are **per task**; `--parallel` controls concurrent tasks.

To submit a prepared experiment or continue an **existing experiment**:

```bash
run="PASTE_EXPERIMENT_OR_COMBINED_ID"

.venv/bin/python -B run_experiment.py resume "$run"
```

Completed annual fits remain saved. An interrupted annual fit restarts.
Repeating the new-experiment command creates another experiment.

## 3. Check status

Find experiments and their recorded Slurm job IDs:

```bash
.venv/bin/python -B run_experiment.py list
```

Check the selected experiment and refresh scheduler information:

```bash
.venv/bin/python -B run_experiment.py status "$run" --refresh
```

Check all experiments with scheduler information:

```bash
.venv/bin/python -B run_experiment.py list --refresh
```

If scheduler/accounting queries fail, inspect saved progress:

```bash
.venv/bin/python -B run_experiment.py status "$run"
```

For Slurm monitoring, select the job/array ID from the launch output or listing:

```bash
job="PASTE_SLURM_JOB_ID"
```

```bash
# All my queued/running jobs
squeue -u "$USER"

# Each task in this array
squeue -r -j "$job" -o "%.18i %.12T %.10M %R"

# Refresh every 30 seconds
squeue -r -j "$job" -o "%.18i %.12T %.10M %R" --iterate=30
```

`Ctrl+C` stops refreshing; submitted jobs continue.

| Output | Meaning |
| --- | --- |
| `PENDING` / `PD` | Waiting; read the reason |
| `RUNNING` / `R` | Executing |
| `saved=30/30` | All 30 annual checkpoints are present |
| `reports=pending` | Aggregate reports have not been collected |
| Empty `squeue` | Check finished-job history and saved results |

`last_action=submitted` records a previous launch action, not current job status.

## 4. Collect results

After all required tasks finish:

```bash
.venv/bin/python -B run_experiment.py collect "$run"
```

The command prints the report directory. To inspect outputs, set the experiment's
full directory:

```bash
exp="/FULL/PATH/TO/EXPERIMENT"
```

```bash
# Pooled performance
cat "$exp/annual/combined/table1_all.csv"

# Annual performance
head -n 15 "$exp/annual/combined/annual_metrics.csv"
```

For combined NN runs:

```bash
# Training diagnostics
head -n 15 "$exp/annual/combined/nn_training_summary.csv"

# Collection validation
cat "$exp/annual/combined/validation_summary.json"
```

For a **combined** ID, collection also copies the complete source experiment
folders into `$exp/sources/`, verifies their file hashes and preserves the originals
unchanged. This also upgrades older combined runs whose reports are already ready.
After collection succeeds, download **the entire combined folder only**, including
`sources/`. It contains the saved models, predictions, training curves, configs
and logs as well as pooled reports. Raw/processed datasets are separate.

Paths in `annual/combined/year_sources.csv` are relative to `$exp`;
`original_*` columns retain the original server paths. Status, collection and
completed resume work with the downloaded combined folder's full path. Figure 4
also needs the matching local panel. A repeated collect verifies the stored copies;
it does not refresh them from the originals.

To restore missing/corrupt copies from the original sub-runs, or shorten an older
combined bundle's paths, run on SCRP:

```bash
.venv/bin/python -B run_experiment.py collect "$run" --repair-sources
.venv/bin/python -B run_experiment.py status "$run" --verify
```

The same combined ID and scores are retained. Repair uses only files matching the
recorded SHA-256 values and never changes the originals or submits training.
New/repaired bundles use `sources/00/`, `sources/01/`, etc. Allow temporary disk
space for a second bundle during repair. A missing inventory manifest must be
restored from backup; it is never recreated from guessed contents.

`sources=present` means the file inventory matches; `sources=verified` additionally
means all hashes were checked. `reports=incomplete` / `sources=incomplete` indicate
a damaged bundle even when an old report exists. Use `list --verify` to check all
combined bundles. These checks do not query Slurm.

For Windows, transfer the archive intact between PCs, extract to a short directory
such as `D:\data_transfer`, then run `status /FULL/PATH/TO/COMBINED --verify` with
the local project's Python. Verify the extracted files before analysis.

| ID | Use |
| --- | --- |
| Experiment or combined ID | `resume`, `status`, `collect` |
| Slurm job/array ID | `squeue`, `sacct`, `scancel` |

## 5. Combine experiment parts

Prepare each part with its own years/resources using `--dry-run`, then combine
their IDs. Sources must use the same cluster and research settings, with disjoint
years covering the declared horizon.

```bash
early="PASTE_FIRST_EXPERIMENT_ID"
late="PASTE_SECOND_EXPERIMENT_ID"

.venv/bin/python -B run_experiment.py combine \
  "$early" "$late" --years 1987:2016
```

Copy the printed combined ID:

```bash
run="PASTE_COMBINED_ID"
```

Use the usual commands from then onward:

```bash
.venv/bin/python -B run_experiment.py resume "$run"

.venv/bin/python -B run_experiment.py status "$run" --refresh

.venv/bin/python -B run_experiment.py collect "$run"
```

Each part retains its resource settings. Keep the original source directories.
Change resources through an inactive **source** experiment's `resume`, rather
than adding resource overrides to the combined ID.

For completed parts that were never combined, register and collect together:

```bash
.venv/bin/python -B run_experiment.py collect \
  "$early" "$late" --years 1987:2016
```

Once a combined ID exists, reuse `collect "$run"`.

## 6. Read logs

While jobs run, set `exp` to the **source experiment**. After combined collection,
you can instead point it to the copied source directory under `COMBINED_ID/sources/`
to read the saved logs.

```bash
exp="/FULL/PATH/TO/SOURCE_EXPERIMENT"
job="PASTE_SLURM_JOB_ID"
task=0
```

```bash
# Last 80 lines
tail -n 80 "$exp"/annual/logs/"${job}_${task}_"*/run.log

# Follow live output
tail -f "$exp"/annual/logs/"${job}_${task}_"*/run.log

# Early startup errors
tail -n 80 "$exp"/submissions/*/"slurm-${job}_${task}.out"

# Timing and peak RAM after completion
cat "$exp"/annual/logs/"${job}_${task}_"*/time.txt
```

Pending tasks normally have no worker log yet. After resubmission, use the new
job ID. If a wildcard matches multiple attempts, inspect the paths individually.

For all tasks in this submission:

```bash
grep -H 'Elapsed (wall clock)' "$exp"/annual/logs/"${job}_"*/time.txt
grep -H 'Maximum resident set size' "$exp"/annual/logs/"${job}_"*/time.txt
```

## 7. Other Slurm commands

Detailed job information:

```bash
scontrol show job "$job"
```

Estimated starting time; the estimate may change:

```bash
squeue --start -j "$job"
```

Finished-task status and runtime:

```bash
sacct -X --array -j "$job" \
  --format=JobID%22,State%20,ExitCode,Elapsed,Start,End
```

Measured RAM, including job-step records:

```bash
sacct -j "$job" \
  --format=JobID%22,State%20,Elapsed,ReqMem,MaxRSS,ExitCode
```

`Elapsed` excludes queue waiting. `ReqMem` is requested RAM; `MaxRSS` is measured
RAM and may appear on `.batch` rows. Blank accounting fields mean unavailable.
If `sacct` cannot connect to its database, use saved progress and worker logs.

Cancel the **whole array**:

```bash
scancel "$job"
```

Cancel **one task**:

```bash
scancel "${job}_${task}"
```

Cancel only the array's **pending tasks**:

```bash
scancel --state=PENDING "$job"
```

Check GPU allocations:

```bash
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50
```

Compare `GRES` with `GRES_USED`. Exclude down/drained nodes; allocated GPUs are
not available for new tasks even if their instantaneous utilization is low.

SCRP resource and storage information:

```bash
scrp-info
qos
scrp-quota
```

## 8. Updates and help

Before preparing a new batch, when an update is needed:

```bash
cd "$project_dir"

git status --short
git pull --ff-only
uv sync --locked --extra nn
```

Use the project's required dependency extras; `--extra nn` is for this project's
neural networks. Preserve the original fitting code and environment for existing
runs that may need resumption. Install dependencies on the login node.

Command help:

```bash
.venv/bin/python -B run_experiment.py --help
.venv/bin/python -B run_experiment.py resume --help
.venv/bin/python -B run_experiment.py collect --help
```

For another project using a conventional Slurm script:

```bash
sbatch run_job.sh
```

Further references: [project run guide](../Running%20Experiments.md),
[SCRP cluster guide](https://scrp.econ.cuhk.edu.hk/guide/slurm), and
[Slurm accounting reference](https://slurm.schedmd.com/sacct.html).

The matching [AI Hub reference](run_aihub.md) uses its
own backend and result lookup flags. Site details last reviewed on 2026-10-26;
use current account output for actual quotas and availability.
