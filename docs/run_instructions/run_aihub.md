# AI Hub command reference

Run these commands **on AI Hub, in Bash**. Each project uses its own `.venv`.
Copy the individual block you need.

The `../../run_experiment.py` commands assume the project uses our launcher. The Slurm
commands apply to other projects too. Model names, samples and years below are
examples from the current replication.

## 1. Enter the project

```bash
project_dir="$HOME/PROJECT_NAME"
cd "$project_dir"
```

Set this again after reconnecting. Use `--aihub` when creating experiments and
looking up their IDs. The default results root is
`$project_dir/results/aihub_experiments`; `list` prints its location. Use
`--root /PATH/TO/EXPERIMENTS` consistently if overriding that location.

## 2. Run an experiment

Create and submit a **new experiment**. Example for a GPU model:

```bash
.venv/bin/python -B run_experiment.py nn2 --aihub \
  --sample full --years 1987:2016 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 3 \
  --cpus 8 --mem 80G --time 12:00:00
```

Change the model, sample, years and resources as needed.

- Add `--dry-run` to prepare without submitting.
- `--gpus 1` requests one GPU per annual task; `--parallel 3` allows three tasks.
- CPU/RAM/time requests are **per task**. The example totals up to 24 CPUs,
  240G RAM and three GPUs.
- `--gpu-type any` lets Slurm select the GPU. The observed `gpuh` partition had
  H100 nodes on 2026-10-26; check `sinfo` for current configuration.
- `normal` was rejected on this account. Use `gpuh` for these GPU examples.
- CPU models omit GPU flags and use a partition confirmed to allow CPU jobs.

The launcher enforces a 240G aggregate RAM budget, 60 CPUs, four GPUs and at most
24 hours per task, based on the documented RPG limits. Three 96G tasks exceed
that RAM budget. Other active experiments also consume resources. See the
[AI Hub resource guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#resource-quota)
and current account limits before changing allocations.

To submit a prepared experiment or continue an **existing experiment**:

```bash
run="PASTE_EXPERIMENT_OR_COMBINED_ID"

.venv/bin/python -B run_experiment.py resume "$run" --aihub
```

Completed annual fits remain saved. An interrupted annual fit restarts.
Repeating the new-experiment command creates another experiment.

## 3. Check status

Find experiments and their recorded Slurm job IDs:

```bash
.venv/bin/python -B run_experiment.py list --aihub
```

Check the selected experiment and refresh scheduler information:

```bash
.venv/bin/python -B run_experiment.py status "$run" --aihub --refresh
```

Check all experiments with scheduler information:

```bash
.venv/bin/python -B run_experiment.py list --aihub --refresh
```

If scheduler/accounting queries fail, inspect saved progress:

```bash
.venv/bin/python -B run_experiment.py status "$run" --aihub
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
Array concurrency is a ceiling: an array allowing three tasks can start one at
a time as resources become available.

## 4. Collect results

After all required tasks finish:

```bash
.venv/bin/python -B run_experiment.py collect "$run" --aihub
```

For the default results layout:

```bash
exp="$project_dir/results/aihub_experiments/$run"
```

If using a custom root, set `exp` to the full experiment directory printed by
the launcher instead.

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

| ID | Use |
| --- | --- |
| Experiment or combined ID | `resume`, `status`, `collect`, with `--aihub` |
| Slurm job/array ID | `squeue`, `sacct`, `scancel` |

## 5. Combine experiment parts

Prepare each part with its own years/resources using `--dry-run`, then combine
their IDs. Sources must all be on AI Hub, use matching research settings and
recorded fitting environments, and have disjoint years covering the declared
horizon. SCRP and AI Hub sources cannot currently be combined by this launcher.

```bash
early="PASTE_FIRST_EXPERIMENT_ID"
late="PASTE_SECOND_EXPERIMENT_ID"

.venv/bin/python -B run_experiment.py combine \
  "$early" "$late" --years 1987:2016 --aihub
```

Copy the printed combined ID:

```bash
run="PASTE_COMBINED_ID"
```

Use the usual commands from then onward:

```bash
.venv/bin/python -B run_experiment.py resume "$run" --aihub

.venv/bin/python -B run_experiment.py status "$run" --aihub --refresh

.venv/bin/python -B run_experiment.py collect "$run" --aihub
```

Each part retains its resource settings. The source pools' total requested
concurrency must fit the launcher's combined resource budget. Keep the original
source directories. Change resources through an inactive **source** experiment's
`resume`, rather than adding resource overrides to the combined ID.

For completed parts that were never combined, register and collect together:

```bash
.venv/bin/python -B run_experiment.py collect \
  "$early" "$late" --years 1987:2016 --aihub
```

Once a combined ID exists, reuse `collect "$run" --aihub`.

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

AI Hub workers use GNU time at `$HOME/.local/bin/time`, previously verified on
a compute node. Keep that executable available for the batch worker.

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
If accounting queries fail, use saved progress and worker logs.

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

Check GPU allocations and detailed node resources:

```bash
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

scontrol show nodes
```

Compare `GRES` with `GRES_USED`. Exclude down/drained nodes; allocated GPUs are
not available for new tasks even if their instantaneous utilization is low.

Inspect partitions and account associations:

```bash
scontrol show partition

sacctmgr -P show assoc where user="$USER" \
  format=Cluster,Account,Partition,QOS,DefaultQOS
```

Accounting queries may be restricted or unavailable. These commands do not use
the SCRP-only `scrp-info`, `qos` or `scrp-quota` shortcuts.

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
runs that may need resumption. Install dependencies on the head/login node;
compute nodes have no internet access.

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
[AI Hub cluster guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide),
and [Slurm accounting reference](https://slurm.schedmd.com/sacct.html).

The matching [SCRP reference](run_scrp.md) uses its
own backend and result lookup defaults. Site details last reviewed on 2026-10-26;
use current account output for actual quotas and availability.
