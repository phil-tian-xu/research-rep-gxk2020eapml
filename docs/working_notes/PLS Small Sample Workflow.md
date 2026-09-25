# PLS on the fixed 5% sample

`gkx_pls_p05_v1` uses the existing `gkx_dev_p05_seed2020_parquet` conversion.
It does not resample securities or change the prepared features/returns.

Run a two-year local check:

```powershell
python -B main.py --config gkx_pls_p05_v1 --stop-after-year 1988 --memory
```

Continue the saved run to 2016 using the directory printed by that command:

```powershell
python -B main.py --resume "results/gkx_pls_p05_v1/<timestamp>" --memory
```

Or start a fresh complete 1987–2016 run:

```powershell
python -B main.py --config gkx_pls_p05_v1 --memory
```

`--memory` is optional; intermediate stops can use `--stop-after-year` on resume.
Do not add `--config` when resuming: the saved configuration is used.

For test year t, training uses 1957 through t−13, validation uses t−12 through
t−1, and testing uses t. Parquet batches fill complete float64 arrays for one
window at a time. The original `PLSRegressor` fits each K from 0 through 10
separately on the complete training window. Training arrays are released before
loading validation. Validation MSE selects K (first candidate wins a tie), and
that training-fitted model predicts the test year without refitting on validation.
There is no QR reduction or batch estimator in this workflow.

Intercept is enabled, additional standardization is disabled, and PLS tolerance
is 1e-12. K=0 predicts the training mean. The grid and zero-component convention
are agreed pilot choices, not verified author settings. Requested and realized
component counts are saved (PLS can stop early if its covariance is exhausted).

Each completed year saves predictions and tuning results before moving on.
The existing `table1_figure3/` reports contain pooled zero-benchmark R2 and annual
component counts/plots. These are development results for 5% of securities;
the paper's full-universe numbers are only reference values. Partial runs are
labelled with their completed horizon. This workflow does not provide the
paper's largest/smallest-1,000-stock Table 1 columns.

The shared dense runner now accepts one PCR or PLS specification per run.
The full-window loader lives in `core/data_query/parquet_dataset.py`; the PCR
memory demonstration imports it there too. PCR's existing SVD fit is unchanged.

Keep the running SCRP PCR checkout at its existing revision. Source hashes are
checked on resume, so these local changes should be deployed in a separate
checkout or after that PCR run is finished.

Verification on 2026-09-25: 144 tests passed, including original-versus-Parquet
PLS predictions, candidate validation scores, selected/realized components, and
staged-versus-uninterrupted results. The real sample run
`results/gkx_pls_p05_v1/20260925T062656118605Z` completed 1987 and 1988:
selected K=2 and K=1, annual test R2=0.581663% and 0.414895%, pooled
R2=0.505934%. Maximum observed Python resident RAM was 0.756 GiB for these
two windows only; later windows have not been measured in this check.
The remaining 1989–2016 years are ready to resume.
