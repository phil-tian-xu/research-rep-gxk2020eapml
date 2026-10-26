# OLS-3 + Huber: Table 1 size groups

The separate diagnostic `core.data_examine.ols3_size_groups` evaluates saved
full-universe OLS-3 + Huber forecasts for All stocks, Top 1,000 and Bottom 1,000.
The default never fits models. It does not change the production workflow or
original experiments.

From the repository root:

```powershell
.venv/Scripts/python.exe -B -m core.data_examine.ols3_size_groups --experiment results/scrp_imports/ols3_full_1987-2016_20260926T140325486437Z
```

Add `--refit-train-validation` to compare the saved training-only forecasts with
an additional annual fit using all training and validation observations before
the test year. This reuses the existing estimator and saved model settings,
recomputing the same OLS-residual Huber quantile on the expanded fitting sample.
It does not fit separate models for the size groups or use test returns in fitting.
The optional comparison loads the three predictors once, then selects only the
observations strictly before each test year for that year's fit.
This is a sensitivity check: the authors' empirical OLS-3 refit rule remains
unverified, and matching the published score does not establish that rule.

## Size and evaluation conventions

- Use raw `mvel1` from `data/raw/datashare.csv`, already lagged to formation time.
  Do not use the ranked/imputed predictor as market capitalization.
- Each month, select groups among existing test observations with finite,
  strictly positive raw size. Both fitting variants use identical memberships.
- Exclude missing, zero and negative size only from size-group selection. Keep
  these observations in All; record exclusions without filling or taking abs(size).
  This is an explicit diagnostic convention, not verified author treatment.
- Break size ties by ascending numeric PERMNO, as in the existing evaluation
  helper. Fail if groups overlap or a month has fewer than 2,000 eligible stocks.
- Pool stock-month squared errors and squared excess returns over the requested
  period; do not average annual/monthly R2 or weight observations by size.
- Verify dataset identity, checkpoint checksums and metrics, Parquet checksums,
  source-row joins, targets, forecast dates, reconstructed predictions and raw
  size ranks. Size ranks are reconstructed on the full source cross-section
  before target-return exclusions, matching the existing panel preparation.

## Output

Each invocation creates a new UTC-timestamped folder under
`results/ols3_size_groups/` containing:

- `comparison.csv`: pooled results and differences from the paper.
- `annual_metrics.csv`: annual SSE, benchmark SSE, counts and R2 by variant/group.
- `monthly_size_audit.csv`: eligible/excluded counts, group counts and size cutoffs.
- `invalid_sizes.csv`: excluded security-months and reasons.
- `fits.json`: saved/diagnostic coefficients, thresholds and fitting windows.
- `methodology.json`: conventions, dataset/source/runtime identities and completion status.

Paper references are All = 0.16%, Top = 0.31%, Bottom = 0.17%, visually checked
against Table 1, printed p. 2250. They appear only for the full 1987-2016 horizon,
`gkx_parquet`, and 1,000-stock groups. `--years YEAR` or `--years FIRST:LAST`
supports partial checks; partial runs do not display those full-period references.
Other paths can be selected with `--data-dir`, `--characteristics-csv` and
`--output-root`. Parquet checksum verification can dominate runtime even though
only narrow columns are decoded.

## Verification and current results (2026-10-26)

Six new tests and 13 existing evaluation tests passed. The new tests cover
monthly membership, numeric tie-breaking, invalid sizes retained in All,
pre-target-exclusion ranking, pooled SSE accounting, source/checkpoint failures,
no fitting by default, and cached refits that strictly exclude each test year.

The complete full-universe run is
`results/ols3_size_groups/20261026T153516820520Z/`:

| Sample | Saved training only R2 (%) | Train + validation R2 (%) | Paper R2 (%) |
| --- | ---: | ---: | ---: |
| All | 0.212829 | 0.155809 | 0.16 |
| Top 1,000 | 0.222102 | 0.239630 | 0.31 |
| Bottom 1,000 | 0.226620 | 0.155460 | 0.17 |

All covers 2,494,470 observations; each size group covers 360,000 observations
over 360 months. There are 478 missing sizes and 2,286 zero sizes, with no negative
sizes among the test observations. These rows remain in All. All saved test
forecasts were reconstructed and all 30 optional annual fits converged. Cached
loading gave identical coefficients for the 17 years also completed with the
initial repeated-loading implementation; that partial output is explicitly
marked incomplete in its own folder.

The expanded fitting window brings All and Bottom closer to the paper, but Top
remains about 0.07037 percentage points below the paper. Matching the rounded
All-stock number does not resolve the replication difference or identify the
authors' estimation procedure. Original experiments and production fitting
sources remain unchanged.
