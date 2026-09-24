# Full-universe PCR through 2016

Config: `gkx_pcr_full_v5`. This selects all prepared stocks from
`data/processed/gkx_parquet`, PCR only, and test years 1987-2016.

## Estimation

For test year Y, training is 1957 through Y-13, validation is Y-12 through
Y-1, and testing is Y. Each window is loaded into complete float64 NumPy
arrays, using the same loader as the simple PCR experiment. Training data
are released before loading validation; validation is released before testing.

The original PCR estimator fits a dense training SVD through K=100 once per
year. Its existing `with_components` method supplies K=0..100 candidates.
Validation MSE selects K, retaining the first candidate in an exact tie.
There is no train-plus-validation refit. Intercept is enabled; additional
standardization is disabled. The grid and K=0 training-mean convention are
our previously agreed choices, not independently verified author settings.
The fixed K=10 memory experiment cannot supply a time-varying complexity path.

This workflow does not use the batch QR fitting implementation. It reuses
the existing checkpoint validation and incremental report writing utilities.

## Commands

The complete configured run:

```powershell
python -B main.py --config gkx_pcr_full_v5 --memory
```

Omit `--memory` to disable RAM measurements. It is an invocation option, not
an estimation parameter, and can be toggled on resume. It imposes no RAM cap.
It prints current resident RAM, the process-lifetime peak, and complete-array
sizes at loading stages. Measurements are saved after each stage in
`memory_<invocation_timestamp>.json`; the final measurement also runs on Python
exceptions. An OS process kill cannot execute that final measurement.
Each resumed Python process has its own lifetime peak. Optional counter or
logging failures do not abort estimation. For CSV and legacy batch runs,
measurements bracket loading and the model run; dense PCR also has annual stages.

To inspect the first annual fit before proceeding:

```powershell
python -B main.py --config gkx_pcr_full_v5 --stop-after-year 1987 --memory
```

Copy the printed timestamped run directory. Continue the same run, with no
repeated fits for saved years:

```powershell
$runDir = 'COPY_THE_PRINTED_RUN_DIRECTORY_HERE'
python -B main.py --resume "$runDir" --stop-after-year 1996 --memory
python -B main.py --resume "$runDir" --stop-after-year 2006 --memory
python -B main.py --resume "$runDir" --memory
```

These commands select execution stages, not smaller training windows. Later
windows may require more RAM than this machine has; staging does not remove
that requirement. A failed annual fit restarts that year on resume.

## Saved replication diagnostics

Under `results/gkx_pcr_full_v5/<timestamp>/table1_figure3/`:

- `table1_all.csv`: pooled out-of-sample zero-benchmark R2, also in percent,
  for the All-stocks row of Table 1. Uses sums of squared errors/returns
  over completed years, not an average of annual R2. Includes the existing
  GKX PCR reference (0.26%), observation count, and horizon-completion flag.
- `complexity.csv` and `figure3_complexity.png` / `.svg`: selected PCR K by
  test year, with pending years as gaps. Includes boundary flags and the
  runner-up validation loss gap.
- `tuning_curves.csv` and `validation_YEAR.png`: validation loss for every K.
- `annual_metrics.csv`: yearly train/validation/test counts, losses and R2.

Annual checkpoints retain predictions, selected parameters and all validation
scores. The run root also saves combined predictions, configuration, dataset
hashes and completion status. These reports cover the Table 1 All-stocks row;
the large/small stock rows are not added here. Plots show PCR only for this config.

## Verification and memory evidence

143 tests passed, including complete-array versus original-run predictions,
validation scores and selected K; staged versus uninterrupted execution;
memory on/off invariance; failure recovery; and PCR-only diagnostic generation.

Phil's full-universe fixed-K=10 run for 1987 reported a 16.715481 GiB resident
peak: 475,906 training rows and 920 features. The largest training window
(test 2016) has 2,796,159 rows. Its feature array alone is about 19.17 GiB.
A simple row-count extrapolation of the measured dense-fit peak is about
98 GiB, not a measured peak or guarantee. The 64-GB machine's ability to finish
the unchanged dense workflow through 2016 is therefore unverified.
