# PLS workflow

## Current baseline: v2 (2026-09-28)

New `run_experiment.py pls` experiments default to `gkx_pls_v2`. It inherits
the v1 annual split and dense SIMPLS estimator, enables `STANDARDIZE: true`,
and searches actual component counts 0 through 29. The CLI still explicitly
selects the full or fixed 5% sample. Direct `main.py` use defaults to the 5%
sample inherited by this config.

For each annual training window, the existing estimator centers all 920
predictors and divides each nonconstant column by its training population
standard deviation (ddof=0). Constant columns use scale 1. Validation and test
use those same training means and scales. This includes industry dummies and
characteristic-macro interactions. The fitted intercept, training-mean K=0
forecast, validation MSE, annual windows and no-refit convention are unchanged.

The [author simulation script](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/MC_Reg.m)
scales predictors by training standard deviations. Its
[PLS routine](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/pls.m)
returns a zero-coefficient candidate followed by 1-29 components when called
with A=30. This motivates v2; it does not establish the exact empirical
preprocessing or grid. MATLAB uses ddof=1, but its common scale factor relative
to ddof=0 does not change fixed-K PLS predictions in exact arithmetic. We retain
our fitted-intercept convention rather than copy the simulation's prediction
offset convention. Neither a solver change nor a change to other models is made.

The old full-universe run selected the maximum K=10 in 2004-2012. In a read-only
1987 fixed-5% diagnostic, widening to 0-29 left unscaled K=2 unchanged, while
training-only scaling selected K=0. This is a diagnostic, not proof of improved
full-sample replication. Inspect complete validation curves and boundary hits,
not only whether the selected K resembles the paper.

Local two-year check:

```powershell
python -B run_experiment.py pls --local --sample p05 --years 1987:1988 --cpus 8
```

Full-sample two-year check on SCRP, using the existing launcher:

```bash
.venv/bin/python -B run_experiment.py pls --scrp --sample full --years 1987:1988 --parallel 1 --cpus 8 --mem 192G --time 12:00:00
```

Verification completed on 2026-09-28: all 197 unit tests passed. The local
two-year run completed under
`results/experiments/pls_p05_1987-1988_20260928T120101559178Z`.
Selected K was 0 in both years; test R2 was 0.216555% in 1987 and 0.026491%
in 1988, with pooled R2=0.130248% over 7,772 test observations. This verifies
execution and reporting on the fixed 5% sample, not improved full-sample fit.

Start fresh experiments; old checkpoints must not be combined with v2.
The legacy config remains available via `--config gkx_pls_p05_v1`.
No cluster job was submitted for this change. Preserve any remote checkout
still needed by active jobs. The wider grid fits candidates separately and
will cost more than v1; no runtime improvement is claimed.

Table 1 and Figure 3 reporting remain available. The subsequent 2026-09-28
saved-state update now saves selected fitted PLS parameters for Figures 4/5.
See [Figures 4/5 workflow](Figure%204%20Workflow.md). Start a new fitting run
with that update before the planned final 1987-2016 rerun.

## Legacy v1: PLS on the fixed 5% sample

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
