# Resumable annual runs

Runs launched through `main.py` with a nonempty `RUN.OUTPUT_DIR` now save each
completed model-year immediately. The configured test horizon does not change
between invocations. `--stop-after-year` is an inclusive execution limit, and
`--resume` selects the exact existing run directory and loads its saved config.
Do not pass `--config` together with `--resume`.

## workflow_v2: five years in three stages

The existing `gkx_all_models_workflow_v2` config retains test years 1987–1991,
the 5% development dataset, and OLS, PCR, PLS. No estimation settings changed.

First stage, 1987–1988:

```powershell
python -B main.py --config gkx_all_models_workflow_v2 --stop-after-year 1988
```

The program prints `Run directory:` and a complete resume command before
training starts. Copy that exact timestamped directory into a PowerShell variable:

```powershell
$runDir = 'results/gkx_all_models_workflow_v2/REPLACE_WITH_PRINTED_RUN_ID'
```

When work remains after a stage or a handled training failure, the program also
prints `Run incomplete`, the exact `$runDir` assignment, and the `--resume`
command. A fully completed run does not print this incomplete-run reminder.

Second stage, 1989–1990 (completed units from 1987–1988 are skipped):

```powershell
python -B main.py --resume "$runDir" --stop-after-year 1990
```

Remaining stage, 1991:

```powershell
python -B main.py --resume "$runDir"
```

For an independent uninterrupted comparison covering all five configured years:

```powershell
python -B main.py --config gkx_all_models_workflow_v2
```

This creates a new timestamped run directory using the same 5% dataset and
estimation settings. Compare its predictions, tuning history, and metrics with
the completed staged run; run IDs and execution metadata naturally differ.

Reissuing the resume command after completion does not refit anything. After
an interruption, reissue the same command to complete any missing model-years
through that invocation's stopping year. Do not use `--config` to resume: it
creates a separate new run.

For a 20-year experiment, define all 20 years in a separate config before the
first run and choose successive stopping years in the same way. Changing the
saved horizon or any other saved config field is rejected. Existing runs from
before checkpoint support contain aggregate outputs only and cannot be resumed.

## Saved state and reports

Each run contains:

- `resolved_config.yaml`, `dataset.json`, `manifest.json`: experiment identity.
- `years/<year>/<model>.json`: one complete model-year, containing predictions,
  observations, panel row positions, tuning records, and annual metrics. A
  checksum protects the payload against accidental corruption. JSON retains
  full Python float round-trip precision.
- `predictions.csv`, `annual_metrics.csv`, `summary.csv`, `tuning_history.json`:
  combined reports rebuilt from all completed checkpoints after each invocation.
- `status.csv`: complete/pending status for every planned model-year.

A checkpoint is flushed to a temporary file and atomically renamed into place.
Unpublished temporary files do not count as completed results. A published but
corrupt checkpoint raises an error rather than silently being reused or replaced.
An OS file lock prevents two processes from writing the same run concurrently;
the lock is released if the process exits, even if `.run.lock` remains on disk.

Reports are rebuilt after normal stage completion, ordinary failures, and Ctrl+C.
An abrupt process kill can leave reports stale; the completed model-year files
remain authoritative and reports are rebuilt on the next successful resume.
Report files are individually replaced atomically; readers should wait until
the invocation ends before treating all aggregate files as one consistent set.

Pooled metrics use all saved observations for each model, not averages of annual
metrics. A partial run can have different completed years for different models;
inspect `status.csv` before comparing performance across models.

## Compatibility and limits

Resume verifies panel and metadata hashes, the complete resolved config,
relevant execution source file hashes (including uncommitted code), and Python/
NumPy/pandas/SciPy/PyYAML/iopath versions. It rejects mismatches before fitting.
The saved ordered feature schema is protected by the metadata hash. Data loading
still verifies the panel and loads it into memory on every invocation.

Each year fits fresh models using that year's expanding training and rolling
validation windows. No previous-year estimator is needed. An unfinished
model-year restarts its tuning; candidate/optimizer state is not checkpointed.
The number of completed models is therefore the recovery boundary, not elapsed
training time.

`run()` remains usable in memory without checkpointing. To use checkpoints from
Python, call `create_run(cfg, dataset_info)` once, then pass the returned directory
as `checkpoint_dir` to `run()`. Verified dataset provenance is required. The
existing `save_run()` remains an aggregate-only export for direct Python callers;
`main.py` now uses incremental checkpointing instead.

## Verification

The 101-test suite passed. Resume tests use the unchanged workflow_v2 model grids
and dates with a small generated panel: 1987–1988, 1989–1990, then 1991 produce
exactly the same predictions, tuning history, and metrics as an uninterrupted run.
Tests verify skipped completed fits, recovery within a partially completed year,
write interruptions, abandoned temporary files, checkpoint corruption, changed
config/data/code/package identities, and concurrent-writer rejection. These are
software integration tests; the five-year real-data experiment was not rerun.
