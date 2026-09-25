# Independent annual runs on SCRP

For new experiments, use the unified commands in
[Running Experiments](../Running%20Experiments.md). This document describes the
underlying annual runner and remains applicable to arrays already submitted.

The annual launcher reuses `main.py`, the full-array Parquet reader, and the
original PCR/PLS estimators. Training windows, validation, tuning grids, float64
precision and test evaluation are unchanged. Each task runs exactly one test
year by setting both FIRST_TEST_YEAR and LAST_TEST_YEAR; FIRST_TRAIN_YEAR and
VALIDATION_YEARS stay unchanged. No warm start or cross-year fitted state is used.

## First experiment: PCR, 2009–2016, up to eight simultaneous tasks

Use the checkout containing these new files. Keep the checkout used by any
currently running job unchanged. If necessary deploy these additions in a
separate checkout with its own environment and the same data link. The launcher
uses the directory from which `sbatch` is submitted, not a hard-coded checkout.
Prepare on SCRP, not Windows: generated paths and the environment identity are
specific to the machine used for preparation. Do not update code/packages after
preparation until its tasks and retries finish.

From that repository directory on SCRP:

```bash
experiment="/data/users/s1155269217/GKX2020/results/pcr_parallel_2009_2016_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -B -m core.annual_jobs prepare \
  --config gkx_pcr_full_v5 --first-year 2009 --last-year 2016 \
  --output "$experiment"

sbatch --array=0-7%8 run_annual_scrp.sh "$experiment"
echo "$experiment"
```

Preparation is lightweight; it does not load the panel. Save the printed
experiment path and Slurm job ID. The array maps index 0 to 2009 and index 7 to
2016. `%8` permits up to eight concurrent tasks; actual starts depend on the
scheduler and account limits. Existing `run_pcr_scrp.sh` remains unchanged.

Each task requests 8 CPUs, 128G RAM and two hours; BLAS threads are capped at
the allocation. Eight tasks request 64 CPUs and 1 TiB, excluding other jobs.
Phil's successful standalone 2016 test measured 100,952,100 KiB = 96.275 GiB
peak RSS and 730.68 seconds elapsed. 128 GiB provides about 33% headroom over
that observation; concurrent I/O and accounting may differ. These requests are
PCR-specific starting values, not a guarantee or established PLS requirements.
Override resources with sbatch options when needed, e.g. `--mem=192G`.

The launcher enables the existing optional memory report and also uses GNU
`/usr/bin/time -v`. Logs are kept under
`EXPERIMENT/logs/ARRAYID_TASKINDEX_JOBID/{run.log,time.txt}`.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

## Collect after the array finishes

Restore the experiment variable to the saved absolute path if using a new shell:

```bash
.venv/bin/python -B -m core.annual_jobs collect --experiment "$experiment"
```

The collector reads checkpoints one model-year at a time, never the feature
panel. It requires every requested year exactly once, matching configurations,
data identities and execution identities, and verifies checkpoint checksums.
It reuses the existing prediction/tuning validation and reporting calculations.
Pooled R2 is `1 - sum((y-prediction)^2) / sum(y^2)` across observations, not the
average annual R2. A missing/incomplete year prevents publication of a combined
report; `task_summary.csv` identifies missing years. Active workers hold locks
and cannot be collected concurrently.

Outputs:

- `task_summary.csv`: checkpoint completion, attempt count, recorded elapsed
  seconds summed across attempts, and maximum recorded process peak GiB per year.
  An OOM kill may prevent the attempt record; check GNU time and Slurm accounting.
  Summed task times are not the concurrent experiment wall time.
- `combined/table1_all.csv`: pooled R2 for the requested horizon.
- `combined/annual_metrics.csv`: annual R2, row counts and validation/test losses.
- `combined/complexity.csv`, `figure3_complexity.png` and `.svg`: selected and
  fitted component counts by test year.
- `combined/tuning_curves.csv` and `validation_YEAR.png`: validation diagnostics.
- `combined/source_identity.json`: experiment configuration, execution/data
  identities and source run directories.

Reports are published together only after successful collection and an existing
`combined` directory is never overwritten. The eight-year experiment is a
partial-period diagnostic, not the complete 1987–2016 Table 1 replication.
`complete_horizon` refers to the requested experiment horizon. The known
standalone 2016 result is K=6, test R2 approximately 0.709179%; compare the new
2016 result before using the array as the standard execution workflow.

## Retry one year, or run the full horizon

Resubmit the same experiment and task index. The worker automatically resumes
its sole timestamped run, validates compatibility, and skips a completed fit.
Duplicate submissions of the same year are refused while its lock is held.
For example, to retry 2016 (index 7 in the eight-year experiment):

```bash
sbatch --array=7 run_annual_scrp.sh "$experiment"
```

For a new full 1987–2016 experiment, prepare with those bounds and a new output
directory, then submit `--array=0-29%8`. Do not reuse the eight-year experiment
directory or change its saved plan. The same script/config preparation can be
used with a dense PLS-only config after its full-sample resource needs are tested.

## Local bounded smoke test

Use the existing fixed 5% Parquet sample without modifying the production config.
In PowerShell, from the repository:

```powershell
$experiment = "results/pcr_parallel_local_$(Get-Date -Format yyyyMMddTHHmmss)"
python -B -m core.annual_jobs prepare --config gkx_pcr_p05_check --first-year 1987 --last-year 1988 --output "$experiment"
python -B -m core.annual_jobs run --experiment "$experiment" --task-index 0
python -B -m core.annual_jobs run --experiment "$experiment" --task-index 1
python -B -m core.annual_jobs collect --experiment "$experiment"
```

Automated tests use a small synthetic Parquet panel to compare independent years
with the serial workflow, including exact predictions/tuning, pooled R2,
out-of-order completion, retry without refitting, and rejection of missing years,
duplicates and incompatible identities. Actual eight-way scheduling and shared
storage throughput must be verified on SCRP.

Verification completed locally on 2026-09-26: all 155 tests passed and the
launcher passed Bash syntax checking. The real 5% smoke experiment is saved at
`results/pcr_parallel_local_check_20260926`: 1987 selected K=14, R2=0.463759%,
peak 0.980 GiB; 1988 selected K=16, R2=0.495089%, peak 1.100 GiB. Combined
R2 is 0.477986% across 7,772 observations. Both component choices match the
saved v1 pilot at `results/gkx_table1_figure3_v1/20260922T084211035119Z`;
prediction positions match and absolute prediction differences are below
1e-10. The collector generated CSVs and plots successfully. This validates
the local entry points and reporting, not eight-way SCRP throughput.

Sources: [SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm) and
[Slurm job arrays](https://slurm.schedmd.com/job_array.html).
