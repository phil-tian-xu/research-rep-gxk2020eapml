# GLM+H empirical pilot

Phil approved training-only spline centering/scaling, singleton SIC groups, a
fixed training-mean intercept, and squared-loss tuning followed by Huber refinement.
The separate Python–MATLAB reference check passed on the shared synthetic dataset;
that reference remains unchanged. This empirical implementation deliberately adapts
its preprocessing and uses the project's convergence-checked APG solver.

## Local test commands

From the repository root with the project Python environment active:

```powershell
python -B run_experiment.py glm --local --sample p05 --years 1987 --cpus 8
```

The launcher runs a separate Python worker with capped BLAS threads and memory
reporting. It collects results under the printed experiment directory. Inspect
the worker log, convergence, peak memory, and collected reports before extending:

```powershell
python -B run_experiment.py glm --local --sample p05 --years 1987:1988 --cpus 8
```

That second command creates a new two-year experiment. For an interrupted existing
experiment use `python -B run_experiment.py resume EXPERIMENT_ID`; completed years
are skipped. `list` and `collect EXPERIMENT_ID` work as for the other models.

Alternative direct entry point, with a 30-year configured horizon:

```powershell
python -B main.py --config gkx_glm_huber_v1 --stop-after-year 1987 --memory
```

Use the printed `--resume` command for that direct run. Do not mix a direct-run
directory with a launcher experiment ID. No full-sample cluster job has been
submitted. A largest-window resource check should precede production deployment;
the launcher's inherited memory/time defaults are not measured GLM requirements.

## Model and saved diagnostics

- Same annual train/validation/test dates and zero-benchmark pooled R² as the
  existing replication. No training-plus-validation refit.
- For each continuous input: `[x, (x−minimum)^2, (x−q25)_+^2,
  (x−q50)_+^2, (x−q75)_+^2]`, with MATLAB/Hazen training quantiles. Each five-term
  block is one group. Macro-interaction columns are separate inputs/groups.
- Columns named `sic2_*` must be binary and contribute one term/group. All inputs
  are divided by training sample standard deviations before expansion. Expanded
  terms are centered and scaled using training statistics, reused unchanged for
  validation and test. Zero standard deviations use scale one. SIC terms are
  also centered/scaled; they are not left in unscaled binary coordinates.
- Intercept is the training return mean and remains fixed during Huber refinement.
- Ten log-spaced penalties from 1e-4 through 1e-1; density is our pilot choice.
  Fit squared group lasso at each penalty from zero, choose lowest validation MSE
  (first candidate on ties), then refine only the selected fit with Huber loss.
  Saved validation curves describe the **squared stage**, not final Huber fits.
- Huber delta is the .999 quantile of absolute training OLS residuals on the
  **original inputs**, with an intercept, before spline expansion. This retains
  the agreed residual-quantile interpretation and is an explicit GLM adaptation.
- Objective is `mean(half-Huber loss) + lambda * sum(group L2 norms)`; squared
  stage uses `mean(residual²)/2`. This retains the reference's half-loss convention
  but averages over training rows so the penalty grid does not grow with n.
  The authors' simulation uses summed loss: matching it would require
  `lambda_mean = lambda_simulation / n`. Our empirical grid is not that simulation
  grid translated numerically. Against this project's ENet full-loss convention,
  the same effective penalty would have twice this lambda. Normalization is an
  explicit pilot convention, not a recovered empirical author setting.
- APG uses backtracking and proximal-gradient stationarity, max 50,000 iterations,
  tol 1e-8. Unconverged fits fail with model/year/penalty context; no bad candidate
  is silently selected. Squared candidates share exact Gram statistics. Huber
  evaluates the complete training array. This is not a batch/QR estimator.

For 920 inputs (846 continuous and 74 SIC dummies), the design has 4,304 columns.
Only one window's expanded matrix is retained at a time. Training is reloaded
after validation to refine the selected model; validation data never enter fitting.
Column standard deviations are computed per group to avoid a full-sized temporary.

Each annual checkpoint includes the coefficients, feature order, group sizes,
knots, both scaling stages, training basis means, fixed intercept, actual Huber
threshold, candidate scores, selected penalty and final convergence diagnostics.
These suffice to reconstruct predictions. The Table 1 output reports pooled
zero-benchmark R²; the GLM paper reference is 0.19% (full universe, 1987–2016).
Figure 3 plots **active original-input groups**, defined by any nonzero coefficient
within a group; it excludes the intercept. Also saves the expanded nonzero-slope
count. This grouping convention is explicit; it does not merge macro interactions
back into their parent characteristic.

## Verification

Tests cover agreement of the training transform with the verified reference,
prediction independence from evaluation sample composition, singleton indicators,
independent constrained optimization, nonconvergence rejection, staged versus
uninterrupted execution, checkpoint skipping, plots and launcher configuration.
The real 5% empirical GLM run and full-sample peak RAM remain for Phil to test.

```powershell
python -B -m unittest discover -s tests -p test_glm_workflow.py -v
```
