# Table 1 and Figure 3 pilot

Milestone 1 is complete based on Phil's confirmation that staged and uninterrupted
real-data runs produced the same results. Milestone 2 examines predictive performance
and selected complexity together. Similar performance alone does not establish
that estimation or tuning agrees with the paper.

## Experiment

`config/gkx_table1_figure3_v1.yaml` uses the existing 5% development sample,
1957 training start, rolling 12-year validation, and all 30 test years 1987–2016.
The enabled specifications are PCR, PLS, and OLS_HUBER. PCR/PLS run first, so a
Huber failure cannot prevent that year's complexity results from being checkpointed.
The old workflow_v2 settings are unchanged.

Approved pilot choices:

- PCR component grid: every integer 0–100. PLS: every integer 0–10.
- Intercept enabled; no additional feature standardization.
- K=0 predicts the training-sample mean. Without an intercept the estimators'
  K=0 behavior is a zero forecast. This is an explicit baseline, not a fitted
  component that is counted as one.
- Candidate selection minimizes validation MSE, with first-in-grid ties retained.
- Models are fitted on training observations only; no train+validation refit.

The grid, zero-component convention, and development sample were explicitly
approved by Phil. They are not claimed as recovered empirical author settings.
The public author simulation code includes a mean-forecast PCR candidate, but
simulation code does not establish the precise empirical procedure.

## OLS+H threshold: evidence versus implementation

Sources checked:

- `docs/main_paper.pdf`, printed p. 2234, equation (6): squared residuals inside
  the threshold, and 2*delta*absolute_residual - delta^2 outside it.
- `docs/online_appendix.pdf`, p. 13, Table A.5: the threshold is a **99.9% quantile**.
  The table does not identify the variable, sample, or recomputation rule.
- [Author repository](https://github.com/xiubooth/ML_Codes), linked from
  [Dacheng Xiu's website](https://dachxiu.chicagobooth.edu/).
  Its [simulation regression script](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/MC_Reg.m)
  sets a fixed simulation threshold, so it does not resolve the empirical quantile.

After this ambiguity was presented, Phil approved the following interpretation:
fit ordinary least squares on that year's training sample and feature matrix;
set delta to the 99.9th percentile of the **absolute training OLS residuals**;
hold it fixed throughout the Huber fit. Recompute it at each annual fit.
NumPy's `quantile(..., method="linear")` fixes the interpolation convention.
Validation and test returns never enter this calculation. The quantile level
comes from the paper; the residual/sample/interpolation construction is our
explicit interpretation, not a verified author convention. A zero quantile raises
an error instead of receiving an arbitrary positive floor.

The config expresses this as HUBER_DELTA=null and HUBER_QUANTILE=0.999. Fixed
thresholds remain available for previous configs. Supplying both is an error.
Resolved delta, iteration count, stationarity, and stopping threshold are saved
with the candidate's validation result.

## Numerical implementation

PCR grid evaluation fits one SVD through the largest requested K per annual
training sample. Smaller candidates reuse prefixes of its orthogonal directions
and regression coefficients. The optimization is used only if candidates differ
solely in K. Invalid/rank-exceeding candidates still fail. Numerical equivalence
with independent candidate fits is tested for centered/uncentered and scaled/
unscaled inputs. Validation/test data never affect the decomposition.

The OLS_HUBER pilot uses `solver="svd_apg"`. The previous APG implementation on
raw feature coordinates failed at 50,000 iterations. The new option uses a
training-only SVD to express the same unpenalized Huber objective in orthogonal,
unit-second-moment coordinates. It starts at the OLS solution, optimizes the
intercept jointly, and transforms coefficients back to the input coordinates.
Numerically null directions use the standard least-squares rank threshold;
there is no tunable component truncation or added penalty in this Huber solver.

Convergence requires both the coordinate-space APG criterion and the original
feature-coordinate gradient criterion. It does not mark a fit converged merely
because objective changes are small. The solver is restricted to unpenalized
Huber; the existing APG path for penalized models remains unchanged. Independent
BFGS comparisons test fitted predictions, objective values, and stationarity.
This is our numerical implementation of the objective, not the author's exact
optimizer path. No general claim of convergence for every future year is made.

## Commands and outputs

First stage (one year to inspect the initial fit):

```powershell
python -B main.py --config gkx_table1_figure3_v1 --stop-after-year 1987
```

Copy the printed run directory into `$runDir`. Continue in stages:

```powershell
python -B main.py --resume "$runDir" --stop-after-year 1991
python -B main.py --resume "$runDir"
```

After each successful stage the config automatically generates diagnostics under
`<run-directory>/table1_figure3/`. To regenerate them from saved checkpoints,
including older runs or a run whose later model failed:

```powershell
python -B -m core.replication_report --run-dir "$runDir"
```

This command neither loads the wide panel nor fits models. It verifies checkpoint
integrity and acquires the run lock, but intentionally does not require the current
estimation source fingerprint to equal the saved version: plotting older results
is allowed even when retraining/resuming them is not.

- `table1_all.csv`: pooled zero-benchmark R2, both raw and percentage, using the
  intersection of completed years across all enabled models. The same observed
  returns and panel row identities are required for every comparison column.
- `annual_metrics.csv`: all completed model-year metrics, including years that
  are not yet common across models. R2 percentages are included.
- `complexity.csv`: all planned years for PCR/PLS, status, requested and actual
  fitted components, validation loss, runner-up loss gap, and grid-boundary flags.
- `figure3_complexity.png` and `.svg`: PCR and PLS paths by test year. Pending
  years remain gaps; they are not zero or interpolated. Grid maxima are flagged.
- `tuning_curves.csv` and `validation_<year>.png`: every candidate's validation
  loss and distance from the best loss. These help distinguish a sharply identified
  component count from a nearly flat validation objective.
- `source_identity.json` and `comparison_notes.txt`: dataset/config identity and
  interpretation of the comparison.

Matplotlib is declared as a project dependency. Plotting is separate from model
checkpoint publication, so a plot error cannot invalidate saved model results.

## Interpretation

Table 1's All row gives PLS 0.27%, PCR 0.26%, and OLS+H -3.46% for the full
universe over 1987–2016 (printed p. 2250). These are reference values, not pass
thresholds. Our table labels the dataset and exact common test years. Plain OLS
is not assigned the OLS+H reference. Top/bottom-1,000 rows are deliberately absent.

Figure 3's published complexity paths are not digitized or fabricated in our
chart. Compare against printed p. 2251. A development sample can change both the
best component count and the shape of its validation-loss curve. Inspect candidate
scores, preprocessing, timing, and sample composition before attributing a different
path to incorrect code or adjusting settings to match the published plot.

Execution source changes for this milestone invalidate old runs for resumption
under the strict source check. Existing saved results remain usable for reporting.

## First real-data verification: 1987

Run `results/gkx_table1_figure3_v1/20260922T084211035119Z` completed all three
models for 1987, leaving 1988–2016 pending. There are 22,395 training,
37,680 validation, and 3,892 test observations. All 110 automated tests pass,
including comparison of the Huber solver against an independent optimizer,
training-only threshold construction, shared versus separate PCR decompositions,
and report reconstruction from checkpoints. The generated complexity and
validation-loss figures were visually inspected.

| Model | Selected components | 1987 test R2 (%) |
| --- | ---: | ---: |
| PCR | 14 | 0.463759 |
| PLS | 2 | 0.581663 |
| OLS+H | Not applicable | -2378.176271 |

The fitted Huber threshold is 0.7279323781 in decimal-return units. OLS+H
converged in 11 iterations, with original-coordinate stationarity approximately
6.11e-12. Nevertheless, its test predictions range from -5.3362 to 6.9723
(about -534% to +697%). The ten largest squared errors account for only 5.84%
of its total squared error, so this is not explained by a single bad prediction.
The cause of this out-of-sample instability remains unresolved. Numerical
convergence does not establish empirical correctness; investigate feature
scaling, weakly identified directions, and sample/timing before interpreting
this model's comparison with the paper. No clipping or regularization was added
to improve the reported score.

These are one-year development-sample diagnostics, not a completed Table 1
comparison or a Figure 3 trend. PCR's runner-up validation MSE differs from
the minimum by only about 1.84e-7; the saved tuning curve helps assess how
sharply the component count is selected.

## v4: full prepared universe in ten-year stages

`gkx_table1_figure3_v4` now selects `gkx_parquet`, the full prepared panel's
yearly Parquet conversion, and the batch execution backend (10,000 rows).
Its model, split, and evaluation settings remain identical to v1. It includes PCR, PLS, and OLS_HUBER. This
isolates the universe change; it does not establish that the universe caused
the pilot's Huber instability. Other documented assumptions remain in force.

Run these commands from the repository root in the project environment:

```powershell
# First stage: 1987-1996, inclusive.
python -B main.py --config gkx_table1_figure3_v4 --stop-after-year 1996

# Replace the placeholder with the exact timestamped directory printed above.
$runDir = 'results/gkx_table1_figure3_v4/REPLACE_WITH_PRINTED_RUN_ID'

# Second stage: 1997-2006; completed model-years are skipped.
python -B main.py --resume "$runDir" --stop-after-year 2006

# Third stage: 2007-2016.
python -B main.py --resume "$runDir" --stop-after-year 2016
```

Each year still has its own annual training, validation, and test windows.
Ten-year stages only limit execution; they do not change the statistical splits.
After an interruption, repeat the same stage's command using `--resume` and
that stage's stopping year. Start a new v4 run after the batch-backend change;
saved runs from older execution code cannot be resumed under new source hashes.
Table 1 and Figure 3 diagnostics update automatically after successful stages.

The full panel is approximately 52.5 GB on disk. Its 3,740,946 by 920 float64
feature matrix alone requires approximately 27.5 GB (25.6 GiB); loading copies,
annual slices, and decomposition workspaces require additional memory. The
batch backend avoids these full-sized allocations. Staging limits execution
time; batch fitting limits working memory. See
[Batch Parquet Workflow](Batch%20Parquet%20Workflow.md) for methods, measured
smoke-test memory, numerical equivalence, and remaining verification limits.
