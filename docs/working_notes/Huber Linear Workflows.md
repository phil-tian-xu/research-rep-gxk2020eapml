# ENet+H and OLS-3+H

Use `run_experiment.py enet/ols3 --local --sample p05 --years 1987:1988`.
The primary [run guide](../Running%20Experiments.md) covers submission, monitoring,
collection and retries. No model-specific batch scripts are needed.

## Research specification

Both methods retain full annual float64 training arrays, training from 1957,
a rolling 12-year validation window, and one test year (1987–2016). They fit
on training only; validation selects candidates, with no train+validation refit.
The pilot configs default to the existing fixed 5% security sample. `--sample full`
selects the full Parquet panel without changing the estimation specification.

- ENet uses all 920 prepared features, the original accelerated proximal gradient
  estimator, Huber loss and the elastic-net penalty. Appendix B.1 defines
  lambda*(1-rho)*sum(abs(beta)) + lambda*rho/2*sum(beta**2).
- Table A.5 fixes rho=0.5 and gives the lambda range (1e-4, 1e-1). Our explicit
  ten-point logarithmic grid includes both endpoints. The density and endpoint
  inclusion are replication choices, not a recovered author grid.
- OLS-3 uses ranked `mvel1` (size), `bm`, and `mom12m`, in that order, without
  macro interactions or industry dummies. Appendix A.11 identifies size, bm and
  mom12m. Parquet projection reads only those three features, not 920 columns.
- Both use an unpenalized, jointly fitted intercept and no additional feature
  standardization, continuing the existing pilot conventions. The panel's
  cross-sectional ranking remains unchanged.
- Huber loss is mean(e^2) within delta, and mean(2*delta*abs(e)-delta^2) outside.
  Returns and delta are in decimal units. There is no 1/2 factor on the loss;
  lambda refers to this exact objective. A package with a different loss or
  penalty normalization cannot use lambda unchanged. The appendix does not
  fully resolve the authors' empirical units/scaling recipe, so the numerical
  grid is explicitly a pilot interpretation; no silent unit conversion is made.
- Table A.5 specifies a 99.9% Huber quantile but does not fully specify its
  construction. Phil approved absolute in-sample OLS residuals from that year's
  training window. We compute their linear-interpolated 0.999 quantile once,
  using the specification's own feature set, and hold it fixed across all ENet
  candidates. A zero/invalid threshold raises instead of inventing a floor.
- Validation uses MSE; Table 1 uses pooled zero-benchmark R2. Training Huber loss
  does not change the evaluation definition. Exact ties select the first grid
  candidate, as in the existing workflow.
- Each candidate starts from zero. Solver cap 50,000 iterations, tolerance 1e-8,
  initial step 1 with existing backtracking. Nonconvergence stops the year with
  model/candidate diagnostics; it is never silently selected or skipped.

Sources: main paper Sections 1.2–1.4 and Table 1; online appendix B.1,
Table A.5 (p.13), and Table A.11 (p.27), stored in `docs/`.
These workflows do not add full-feature OLS+H or GLM+H.

## Saved results

One annual checkpoint retains predictions, chosen penalty, all candidate validation
scores, actual Huber delta, iterations, convergence, objective, stationarity and
step size. Selected slope coefficients, ordered feature names and intercept are
saved with the tuning record so predictions can be reconstructed.

ENet Figure 3 counts exactly nonzero slopes, excluding the intercept; the proximal
operator produces exact zeros. This counts individual expanded features, not
groups of base characteristics. Save the chosen lambda and whether it lies on a
grid boundary; inspect validation curves before interpreting a complexity trend.
Do not adjust the grid using held-out test R2 to imitate the paper.

OLS-3 has a fixed three-predictor specification, so no Figure 3 complexity plot
is generated. Its `complexity.csv` records the realized nonzero slope count for
diagnostics only. Paper full-period All-stock reference R2 values are 0.11% for
ENet+H and 0.16% for OLS-3+H, not targets for individual years or the 5% sample.

## Validation and deployment

Tests compare Huber regression with SciPy robust least-squares and ENet with an
independent constrained SLSQP objective. Annual tests check feature order,
training-only thresholds shared across penalties, saved-coefficient prediction,
staged/complete equality, resume skipping, nonconvergence rejection and reports.

PLS is currently running on SCRP according to Phil. Keep its execution checkout
unchanged: new source hashes intentionally invalidate ordinary resume under the
modified implementation. Develop locally, then use a separate checkout or wait
for PLS to finish before updating the server code.

Full-sample ENet runtime and memory are unmeasured. Each penalty requires an
iterative fit and the threshold also needs a training OLS fit. Benchmark 2016
before requesting a full array; PCR timing is not an ENet timing estimate.

Verification on 2026-09-26: all 171 tests passed. Real unified-launcher runs for
1987 on the fixed 5% sample completed and produced combined reports:

| Method | Test R2 (%) | Selected lambda | Nonzero slopes | Peak Python GiB | Worker seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| ENet+H | 0.523733 | 0.01 | 11 | 0.641 | 190.15 |
| OLS-3+H | 0.345153 | 0 (unpenalized) | 3 | 0.140 | 1.74 |

Experiments: `enet_p05_1987-1987_20260926T072037893220Z` and
`ols3_p05_1987-1987_20260926T072039517817Z`, under `results/experiments/`.
The weakest ENet penalty converged after 7,621 iterations; every candidate
converged. Timing includes local I/O and fitting and is not a cluster forecast.
