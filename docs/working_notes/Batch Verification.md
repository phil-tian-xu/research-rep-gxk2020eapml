# Verification before the full replication run

The existing `tests/test_batch_workflow.py` compares small synthetic dense and
batch fits. `core/data_examine/verify_batch.py` adds reproducible checks against
saved real-data results and fresh-process memory measurements. It calls the
production batch runner; it does not implement another fitting algorithm.
Original datasets and reference results remain unchanged. Verification results
are separate from the main replication outputs.

## 1. Convert the same 5% sample to Parquet once

The full-universe Parquet conversion cannot be substituted for the 5% sample.
Use the already prepared fixed-security sample, preserving all its rows/values:

```powershell
python -B -m core.data_cleaning.partition_gkx --source-dir data/processed/gkx_dev_p05_seed2020 --output-dir data/processed/gkx_dev_p05_seed2020_parquet
```

Skip this command if that exact conversion already exists. The converter refuses
an existing destination. No new random sample is drawn. The verifier checks both
source CSV and source metadata hashes against the saved run manifest before fitting.
The current 5% source metadata matches the saved reference's recorded hashes.

## 2. Compare with the existing 5% results

This reference contains all 30 test years and all three models:

```powershell
$referenceRun = 'results/gkx_table1_figure3_v1/20260922T084211035119Z'
python -B -m core.data_examine.verify_batch compare --reference-run "$referenceRun" --parquet-dir data/processed/gkx_dev_p05_seed2020_parquet
```

By default this reruns **all saved test years**, with the exact saved model,
split, loss, and tuning settings. Each year runs in its own fresh Python process.
The execution horizon is narrowed to that test year; its training start and
rolling validation length remain unchanged. Each annual job fits PCR, PLS and
OLS+H, evaluates all candidates, predicts, checkpoints, and generates the
configured reports. It does not resume or mutate the old run.

For a preliminary check, append `--years 1987 2016`. The report explicitly marks
this as partial coverage of the saved horizon. Passing those two years alone
does not establish equivalence for the intervening Figure 3 path. A later command
without `--years` starts fresh verification jobs, including those years again.

The comparison checks:

- Identical test row positions, return months, observed targets, model identities,
  and train/validation/test counts. Predictions are aligned by row position.
- Prediction values, annual test loss and zero-benchmark R2.
- Every candidate's validation loss and the selected parameters.
- Every candidate's actual component count, covering Figure 3 complexity.
- Huber's fitted training-residual threshold.

Default numeric tolerances are `atol=1e-7`, `rtol=1e-6`, using
`abs(new - reference) <= atol + rtol * abs(reference)`. These are explicit
verification choices, not paper settings. For a decimal-return prediction,
the absolute term is 0.001 basis point. The same numeric tolerances apply to
reported losses, R2, and thresholds in their respective units. Identity,
candidate parameters, and component counts must match exactly. Iteration counts
and solver coordinates need not match; the production runner rejects any new
fit that fails its convergence checks.

`--atol` and `--rtol` can set stricter tolerances. A failure should be investigated,
especially for ill-conditioned OLS+H or near-tied component choices; do not simply
loosen tolerances until the test passes. Max absolute differences, RMSE of the
differences, and counts outside tolerance are saved in `comparison.csv`.
The verifier stops after the first failed annual fit/comparison, preserving the
completed checks and model checkpoints. Its exit code is nonzero on failure.
Compare-mode success refers to correctness for the selected years; its separately
reported memory result is informational until the full-universe test below.

## 3. Measure full-universe annual fits

After investigating any comparison failures:

```powershell
python -B -m core.data_examine.verify_batch memory --config gkx_table1_figure3_v4
```

By default this runs the first and last configured test years, **1987 and 2016**,
one after the other in separate processes. It uses the existing full-universe
Parquet files and all configured candidates/models, without row limits:

| Test year | Training return years | Training rows | Validation return years |
| --- | --- | ---: | --- |
| 1987 | 1957–1974 | 475,906 | 1975–1986 |
| 2016 | 1957–2003 | 2,796,159 | 2004–2015 |

These counts were checked against the current full conversion manifest. The
script independently calculates the largest training count across the configured
horizon and records which test years attain it. This is a substantial computation,
particularly Huber's repeated training passes. It is not a quick 10,000-row smoke test.

The Windows OS process-lifetime peak working set measures **resident RAM of each
fitting Python process**, including imports, decoding, numerical workspace,
validation, prediction, checkpointing and configured reporting. It captures brief
spikes between Python observations. Fresh workers keep earlier jobs from inflating
later measurements. The small controlling Python process, other applications, and
system/file-cache memory are outside this number. End-of-job private committed
memory is recorded separately; it is not resident RAM or a peak measurement.

The default target is **strictly below 8 GiB** (`8 * 1024**3` bytes). The target
is checked after execution, not enforced as a memory cap. It does not kill a job
at 8 GiB or guarantee that the machine cannot run out of memory. Fits must complete
for a passing result; a failed fit with a low measured peak is still a failure.
Unsupported OS memory counters are reported as unavailable, never as zero/pass.
Memory mode exits nonzero if a requested fit fails, its memory target is not met,
or the selected years omit the largest training window.

Optional flags: `--batch-size 5000`, `--target-gib 8`, or `--years 2016`.
The last command measures the largest window only and reports that selected scope.
A pass covers only the measured windows, machine, config, and batch size. It is
not a numerical comparison to a dense full-universe fit or an unconditional claim
about every year. Use the default batch size for the main run if validating it.

## Outputs and current verification status

Each command creates a timestamped directory under `results/batch_verification/`:

- `report.json`: scope, tolerances, per-year completion, correctness and memory
  outcomes, timings, largest-window coverage, and verification-script hash.
  Top-level `peak_resident_gib` is the maximum measured worker peak across years.
  `memory_status` distinguishes `passed`, `exceeded`, `incomplete`, and `unavailable`;
  `memory_target_passed` is null when the full-scope verdict is not yet known.
- `comparison.csv`: detailed differences (compare mode).
- One directory per test year, containing `job.json`, `measurement.json`, and
  production run artifacts/config/dataset/runtime identity under `runs/`.

During future fits a background thread prints current and **MAX SO FAR** resident
RAM with the worker PID approximately every 30 seconds. It also updates that
year's `memory_latest.json`. These readings use Windows' OS-maintained lifetime
peak, so a short spike between prints is still included. A forced process kill
can prevent the final measurement; the last live snapshot then gives only the
peak known at that timestamp. No separate CMD/Task Manager monitor is necessary.

Read a finished or interrupted verification's saved measurements without refitting:

```powershell
python -B -m core.data_examine.verify_batch summary --run-dir results/batch_verification/compare_20260923T053443957803Z
```

This works with older reports and does not overwrite them. It recomputes the
memory verdict from the actual per-year measurements, even if an older summary
incorrectly used False to mean incomplete. The final console line includes
`MAXIMUM OBSERVED PYTHON RAM`, followed by completed window counts and whether
the largest window actually completed (selection alone is not completion).

Results are written after each year. Unexpected worker termination without a
measurement is recorded as failure. Verification jobs are fresh measurements;
there is no verification-level resume command. Existing production model-year
checkpoints remain usable for inspection, but resuming them would not measure
the peak of the complete annual fit.

The initial verifier passed the 132-test repository suite. After the memory
summary/live-display update, all 11 focused verification-tool tests pass.
The verifier itself was tested on synthetic fixtures, including real subprocess
execution, numerical comparisons, mismatch detection, provenance rejection,
largest-window selection, memory-target failure, and saving measurements after
fit failure.

Phil's real comparison `compare_20260923T053443957803Z` completed 1987 and 1988:

| Test year | Peak worker resident RAM (GiB) | Correctness comparison |
| --- | ---: | --- |
| 1987 | 0.391117 | Passed |
| 1988 | 0.387238 | OLS_HUBER predictions failed |

Only the 1988 OLS_HUBER prediction check failed in the recorded comparisons:
maximum absolute difference 0.0168642591 and RMSE difference 0.00640390857,
with 3,838 predictions outside tolerance. This remains unresolved; tolerances
have not been relaxed. Other recorded comparisons passed. The previous final
memory False meant the other 28 years never ran, not that 8 GiB was exceeded.
All-30-year correctness and full-universe memory therefore remain unverified.

To isolate the largest full-universe memory check, use:

```powershell
python -B -m core.data_examine.verify_batch memory --config gkx_table1_figure3_v4 --years 2016
```

This can characterize memory independently of the unresolved correctness
comparison; it does not approve the replication's numerical results. The earlier
0.563-GiB smoke result and 5% peaks are not substitutes for this full-window test.
