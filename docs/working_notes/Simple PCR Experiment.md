# Simple full-array PCR experiment

Run from the repository root:

```powershell
python -B -m core.data_examine.pcr_memory
```

This standalone experiment uses the existing 5% Parquet conversion. It avoids
the separate batch-fitting workflow selected by the current v4 config.

The procedure is deliberately direct:

1. Read return years 1957-1974 into one complete float64 training feature array
   and target vector. Parquet pieces fill preallocated arrays, avoiding full
   DataFrame/concatenation copies. All training observations are present before fitting.
2. Print the training shape, exact array bytes, and current/peak process RAM.
3. Call the original `PCRRegressor(n_components=10, fit_intercept=True,
   standardize=False).fit(X_train, y_train)` once. This uses the original dense SVD.
4. Report training MSE and release the training arrays.
5. Load complete validation years 1975-1986, predict and score, then release them.
6. Load test year 1987, predict and score, and save test predictions and a JSON report.

K=10 is a fixed demonstration parameter, not a newly tuned or paper-specified
choice. Validation is used for assessment; it does not update the model. To use
another fixed K or test year:

```powershell
python -B -m core.data_examine.pcr_memory --components 14 --test-year 1987
```

Training always starts in 1957 and validation covers the 12 years before the
selected test year. Float64 precision, original feature order and row positions
are retained. There is no added QR reduction, batch model training, worker
orchestration, or 8-GiB threshold. The source data and original model are unchanged.

Outputs go into a new timestamped directory under `results/pcr_memory/`:
`report.json` contains parameters, dataset hashes, array sizes, stage memory
counters, losses, and timing; `test_predictions.csv` contains test forecasts.
Windows' lifetime peak resident counter includes temporary allocations during
fitting. Array sizes and total process RAM are different quantities. A final
memory report is also attempted if fitting fails.

## Full-universe experiment

Run the same single-year experiment on all stocks in the prepared full panel:

```powershell
python -B -m core.data_examine.pcr_full_memory
```

This entry point calls the exact same experiment function, selecting
`data/processed/gkx_parquet` and saving under `results/pcr_full_memory/<timestamp>/`.
Defaults remain training 1957-1974, validation 1975-1986, test 1987, fixed K=10,
intercept enabled and no additional standardization. It loads the entire training
window before the original dense PCR fit; validation and test are loaded and
scored sequentially. No 8-GiB limit is imposed.

The full universe means all stocks within these windows, not a loop over all
30 test years. To examine another annual window, use `--test-year YEAR`;
training still begins in 1957 and validation covers the preceding 12 years.
Later windows are larger, so their peak cannot be established by the 1987 run.
`--components` and `--output-dir` are also supported.

Read `MAXIMUM OBSERVED PYTHON RAM` in the terminal or `peak_resident_gib` in
`report.json` for the Windows process-lifetime peak. The training X/y sizes
exclude PCR's temporary allocations; Parquet disk size is not a RAM estimate.
The full-universe run is left for Phil to execute; no full-data peak is claimed.

## Executed 5% experiment

Run: `results/pcr_memory/20260923T152603763633Z`.

| Item | Result |
| --- | ---: |
| Training rows x features | 22,395 x 920 |
| Training X alone | 0.153507 GiB |
| Training X + y | 0.153674 GiB |
| Process RAM after training load | 0.333 GiB |
| Peak process resident RAM | 0.993053 GiB |
| Validation observations | 37,680 |
| Test observations | 3,892 |
| Validation MSE | 0.02719238594763845 |
| Test MSE | 0.03541520292680378 |
| Test zero-benchmark R2 | 0.456731% |
| Model fitting time | 4.10 seconds |
| Total experiment time | 29.52 seconds |

Validation MSE equals the saved original 1987 dense-PCR candidate for K=10.
The older Table 1/Figure 3 run selected K=14, so its selected-model test result
is not the same experiment as this fixed-K=10 check. Three focused tests verify
complete-array loading, one original-estimator fit, held-out prediction results,
array-memory accounting, and saving the report after a simulated fit failure.
This measurement covers this 5% annual window, not full-universe fitting.
