# Running the replication on AI Hub

Prepared 2026-09-27. Cluster execution and agreement with SCRP remain untested.
Use a separate AI Hub checkout and new experiments. Do not update the active
SCRP checkout, copy unfinished runs for resumption, or replace its environment.
The SCRP scripts/config and the shared annual worker/model code are unchanged.

## Site settings to confirm before the first submission

The [AI Hub organization guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide)
lists RPG limits of 60 logical CPUs, 240 GB memory and 24 hours per job, with
1,000 GB persistent storage under `/hpchome/<user>`. It says `/hpclarge` was
scheduled for removal on September 15, 2026. The
[older guide](https://github.com/cuhk-fba-server/ai_lab_HPC_user_guid) differs;
confirm actual account limits and storage before deployment.

On AI Hub, inspect the available partitions and your account associations:

```bash
sinfo
scontrol show partition
sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS
```

If accounting information is restricted, obtain the account/QoS and CPU job
partition from AI Hub support. The launcher does not invent their names: it
omits account/partition/QoS unless you supply `--account`, `--partition`, `--qos`.
Confirm that site defaults permit CPU-only jobs before using those defaults.
The present models do not use GPUs; the guide prohibits reserving unused GPUs.
Compute nodes have no internet, so install dependencies before submission.

## Separate checkout, environment and data

Place a separate copy of the project under your verified `/hpchome` storage.
Keep prepared data at the same repository-relative paths:

```text
research-rep-gxk2020eapml/
  .venv/bin/python
  data/processed/gkx_dev_p05_seed2020_parquet/
  data/processed/gkx_parquet/
  results/aihub_experiments/
```

Recreate the Linux environment with the exact SCRP Python and package versions
used for the comparison. Do not copy the Windows virtual environment. Use
`.venv/bin/python` both to prepare experiments and in batch jobs. The existing
project dependency declarations alone are not proof of an identical numerical
environment. Compare installed versions and NumPy BLAS/LAPACK configuration:

```bash
.venv/bin/python -V
.venv/bin/python -B -c 'import numpy; numpy.show_config()'
```

Check large-matrix/ILP64 support before full-window PCR; SCRP used an ILP64
OpenBLAS build. Copy the existing Parquet partitions and metadata, verifying
checksums against the source. Do not regenerate the fixed 5% sample or alter
features, precision, model settings or annual splits for this deployment.
Keep checkpoints on persistent storage, not job scratch that is deleted on exit.

## Prepare the first experiment without submitting

Run from the AI Hub project root after environment/data setup:

```bash
.venv/bin/python -B run_experiment.py pcr --aihub --sample p05 --years 1987 --dry-run
```

This creates a new saved experiment and prints its `sbatch` command. It does
not fit a model or submit a job. Add confirmed site flags if needed. The saved
record identifies the backend as `aihub` and snapshots `run_annual_aihub.sh`.
Prepare on AI Hub, not Windows: plans contain absolute paths and runtime identities.

AI Hub defaults to `results/aihub_experiments/` under its checkout, even when
`sbatch` is present. Explicit `--root` and `GKX_EXPERIMENT_ROOT` still take
precedence; do not point them at SCRP results. Use the same root on subsequent
commands. Existing SCRP root detection and defaults are retained.

## Submit and inspect later

Replace `EXPERIMENT_ID` with the ID printed by preparation:

```bash
.venv/bin/python -B run_experiment.py resume EXPERIMENT_ID --aihub
.venv/bin/python -B run_experiment.py list --aihub --refresh
.venv/bin/python -B run_experiment.py collect EXPERIMENT_ID --aihub
```

An absolute experiment path also works. Resume retains the saved backend and
research settings, and submits only unfinished years. Account/partition/QoS
overrides persist on retries. Existing duplicate-submission protection and
code/environment compatibility checks apply. Use this interface for submission;
direct `sbatch` bypasses launch records and launcher resource checks.

| Model command | CPUs/task | RAM/task | Time/task | Concurrent tasks |
| --- | ---: | ---: | --- | ---: |
| `pcr` | 8 | 128G | 02:00:00 | 1 |
| `pls` | 8 | 192G | 12:00:00 | 1 |
| `enet` | 8 | 192G | 12:00:00 | 1 |
| `ols3` | 8 | 8G | 01:00:00 | 1 |

These are initial allocations, not measured AI Hub requirements. Resource flags
can override them. The launcher rejects an array's requested concurrency times
RAM above 240G, concurrency times CPUs above 60, or task times above 24 hours.
Those checks use the documented student budget; they do not query your live
quota or account for other experiments/jobs. Two 128G tasks exceed that budget.
Slurm still decides eligibility and availability. A changed/approved quota would
require revisiting these checks. No artificial memory cap changes model fitting.

The script runs the same `core.annual_jobs run` worker as SCRP and caps BLAS
threads to allocated CPUs. Logs under `annual/logs/` include `run.log`, GNU
`time.txt`, `packages.txt`, `numpy_config.txt` and the submitted script.

## Later comparison with SCRP

1. Compare the same 1987 fixed-5% model-year using matching research sources,
   data checksums and environment. Check selected hyperparameters, all validation
   losses, predictions and test R2, allowing justified floating-point tolerance.
2. Run a new full-sample PCR 2016 experiment only after that comparison passes:

   ```bash
   .venv/bin/python -B run_experiment.py pcr --aihub --sample full --years 2016 --dry-run
   ```

   Compare with the matching SCRP specification and measure memory/runtime.
   The recorded SCRP peak was 96.275 GiB; it is not an AI Hub measurement.
3. Then assign independent years or models to AI Hub. Keep experiment identities
   and outputs separate. Cross-cluster result merging/migration is not implemented
   by this addition.

Local mocked launcher tests cover research-setting equivalence, distinct site
defaults, limits, submission/resume and result lookup. They do not establish
cluster access, scheduling, package compatibility or numerical equivalence.
