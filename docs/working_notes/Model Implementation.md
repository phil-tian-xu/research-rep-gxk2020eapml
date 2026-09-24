# Linear and dimension-reduction implementation

Implemented 2026-09-20. This note describes the added model code, its common
interface, numerical choices, and remaining empirical decisions. It does not
establish that the paper's empirical results have been replicated.

## Files

- `core/models/linear.py`: `LinearRegressor` for OLS, Huber linear regression,
  lasso, ridge, and elastic net.
- `core/models/dimension_reduction.py`: `PCRRegressor` and `PLSRegressor`.
- `core/losses.py`: shared prediction losses, residual kernels/gradients, and
  the paper's elastic-net penalty.
- `core/optimizers.py`: elastic-net proximal operator and accelerated
  proximal gradient with backtracking.
- `core/model_inputs.py`: shared input checks and numeric conversion via
  `validate_model_inputs(X, y)`; no fitting, scaling, or sample splitting.
- `core/models/_helpers.py`: internal feature transforms and fitted-state helpers.
- `core/model_selection.py`: hyperparameter tuning within one specification, with no test input.
- `core/splits.py`: expanding training / rolling validation / annual test indices.
- `core/experiment.py`: configured annual model runs and optional run saving.
- `core/model_demo.py`: experiment on a saved dataset selected by YAML.
- `core/data_query/generate_experimental_data.py`: separate, reproducible toy-data generation.
- `core/data_query/datasets.py`: prepared-panel loading and dataset provenance.
- `core/evaluation/prediction_metrics.py`: zero-benchmark R-squared, reported on
  out-of-sample predictions; the metric itself does not verify sample separation.
- `tests/test_models.py`, `tests/test_model_selection.py`: standard-library
  unittest numerical and workflow checks.
- `tests/test_evaluation_losses.py`: paired-input loss values and shape/index checks.
- `tests/test_experiment.py`: config integration, annual reporting, and saved outputs.
- `tests/test_datasets.py`: saved-panel loading, metadata, reproducibility, and immutability checks.

The estimators are independent small classes; there is no project base class,
registry, universal trainer, or dependency on scikit-learn. They need NumPy,
SciPy, and pandas, already available in the current environment. The existing
run configuration uses the existing CfgNode loader. PyYAML and iopath
are declared in the dependency manifest and installed for that loader; declaring
direct NumPy/SciPy dependencies and tracking a reproducible lockfile remain future
repository work. See `Experiment Configuration.md` for configuration and usage.

## Shared input and output contract

| Item | Convention |
|---|---|
| `X` | Dense numeric matrix of shape `(n_observations, n_features)` |
| Rows | Pooled stock-month observations in a caller-defined order |
| Columns | Predictors in an identical order at fit and prediction |
| `y` | One-dimensional `(n_observations,)` next-period excess returns |
| Units | Prefer decimal returns, e.g. `0.01` for 1%; predictions keep these units |
| Accepted containers | NumPy/array-like matrices, or pandas DataFrames; Series targets supported |
| Numeric computation | Real float64; NaN/infinity rejected; no silent imputation |
| `fit(X, y)` | Fits on these observations only; returns the estimator |
| `predict(X)` | NumPy vector `(n_observations,)` in the supplied row order |
| `transform(X)` | PCR/PLS component scores `(n_observations, n_components_)` |

An `(n, 1)` target is rejected to prevent accidental broadcasting. A DataFrame
and Series must have identical indices in the same order; automatic alignment
could hide a data error. After DataFrame fitting, prediction requires a DataFrame
with the same ordered column labels, including for component transforms. Duplicate
columns are rejected. Array fitting uses positional columns: the caller is
responsible for their meaning and order. Inputs are not mutated. Empty prediction
batches are supported; empty fitting samples are not. A failed refit invalidates
the previous fitted state.

Do not pass PERMNO, dates, realized future returns, or raw eligibility flags as
features by accident. Dates/identifiers stay alongside the matrix for splitting
and joining predictions. A common full GKX matrix would have 920 columns; these
estimators deliberately support arbitrary feature counts. OLS-3 uses the same
linear estimator with only the approved size, book-to-market, and momentum columns.

The caller must construct information-available-at-t predictors paired with
excess returns at t+1, apply the paper's monthly rank/imputation conventions,
build interactions and industry indicators, and exclude ineligible padded rows.
None of that financial data preparation is inferred inside a model.

## Paper requirements and implementation choices

Sources: main paper Sections 1.1-1.4, especially Equations (4), (6), (8), and
(10)-(12); online appendix B.1, Algorithm 1, Table A.5, and Appendix D. See the
local `docs/main_paper.pdf` and `docs/online_appendix.pdf` and the existing
`Tech Details - Part A.md` / `Tech Details - Part B.md` notes.

### Linear family

For residual `e = prediction - y`, the objective is

```text
mean(loss(e)) + lambda*(1-rho)*sum(abs(beta)) + lambda*rho/2*sum(beta**2)
```

Squared loss is `e**2`, without a leading 1/2. Huber is `e**2` for
`abs(e) <= delta` and `2*delta*abs(e) - delta**2` otherwise. The API name for
lambda is `penalty_strength`; `rho=0` is lasso, `rho=1` ridge, and `rho=.5`
the appendix's empirical elastic-net mix.

Use `loss="huber"` for the paper's OLS+H / ENet+H specifications. The plain
constructor defaults to squared-loss, unpenalized OLS; it is not implicitly OLS+H.
Huber fits require an explicit positive finite `huber_delta`. The appendix's
99.9% quantile does not fully identify the underlying variable/sample/update rule,
so the code deliberately does not invent one. Derive a threshold from an approved
training-only rule, or explicitly tune candidate thresholds using validation.
Thresholds and penalty grids are in the target's units; converting percent returns
to decimals changes the appropriate settings.

Plain OLS uses `numpy.linalg.lstsq` for a minimum-norm solution in the fitted
coordinates, including rank-deficient designs. Its `rank_` describes the transformed
predictor matrix, excluding the separate intercept. Penalized and Huber fits use
accelerated proximal gradient from zero coefficients, with:

- FISTA momentum, restarting acceleration after an objective increase;
- initial step 1.0, halved until the smooth quadratic majorization check passes;
- at most 60 backtracking attempts per iteration;
- default 5,000 iterations and tolerance `1e-8`;
- convergence based on the proximal gradient mapping infinity norm at the
  accepted iterate, at most `tol * (1 + ||parameters||_inf)`.

These are explicit numerical implementation choices, not recovered author settings.
The momentum recurrence differs from the printed Algorithm 1. The exact elastic-net
proximal operator also gives the standard soft-thresholding lasso endpoint; it does
not reproduce the apparent extra lambda factor in the appendix's printed lasso row,
which is inconsistent with the stated penalty.

Nonconvergence produces a warning and `converged_=False`; direct users can inspect
the last iterate. Validation selection refuses unconverged candidates. Numerical
failures raise. `optimization_result_` contains objective history, iterations,
accepted step, convergence flag, and stationarity. `objective_` includes the penalty.
OLS has `optimization_result_=None`, `n_iter_=0`, and `converged_=True`.

### Centering, intercepts, and scaling

Common defaults are `fit_intercept=True`, `standardize=False`. They are our
configurable choices because the empirical preprocessing/intercept recipe is
incompletely specified. With an intercept, training predictors are centered;
without it, they are not. Optional scaling divides each column by its training
population standard deviation (`ddof=0`); exactly constant columns use scale 1.
With `fit_intercept=False`, scaling does not also center the features.

The intercept is never penalized. For Huber fits it is optimized jointly with
the slopes, rather than being incorrectly fixed to the arithmetic target mean.
When scaling is enabled, penalties apply to coefficients in the scaled feature
coordinates. This changes the penalized problem and must be recorded for real runs.

`coef_` and `intercept_` always predict in the original supplied feature units:
`X @ coef_ + intercept_`. Training means/scales are saved as `feature_offset_`
and `feature_scale_`; prediction never fits new transformations. These generic
training transforms do not implement the paper's monthly cross-sectional ranks.

### Dimension reduction

PCR uses a deterministic economy SVD, retains the first K right singular vectors,
then regresses on their scores using OLS. K must be a positive integer no larger
than the numerical rank, using `eps * max(X.shape) * largest_singular_value` as the
rank cutoff. Excessive K raises instead of silently clipping it. Constant targets
are valid, but a predictor matrix with zero centered rank cannot support K >= 1.

PLS uses single-response SIMPLS: predictor-target covariance directions,
orthogonal loading deflation, and a final OLS score regression. It follows the
paper's covariance formulation. No separate predictor-by-predictor variance
division is applied beyond optional common standardization. The informal univariate
slope description and scaling conventions do not fully establish an author code
path; this chosen implementation should be included in replication comparisons.

SIMPLS uses two-pass numerical reorthogonalization; `tol=1e-12` controls covariance
and rank exhaustion. Extraction stops if there is no remaining predictive covariance
or independent numerical direction. `n_components_` reports the actual retained
count and `stop_reason_` explains an early stop. Constant targets with an intercept
give zero components and a constant forecast. K is an upper bound for PLS; a request
above `min(p, n-fit_intercept)` still raises. There is no user-selectable K=0 model.

`components_` has shape `(k, p)` in transformed feature coordinates, and
`component_coef_` holds the OLS coefficients on those scores. PLS scores have unit
training Euclidean norm. Component signs are arbitrary; predictions are the meaningful
comparison across SVD/PLS implementations.

These are dense, in-memory estimators. Economy SVD still needs substantial working
memory at the full GKX sample size. Full-sample memory/runtime have not been measured;
there is no out-of-core solver, GPU path, or model persistence layer in this stage.

## Validation and annual windows

`tune_hyperparameters` accepts an explicit list/iterator of constructor dictionaries and
creates a fresh estimator for each. No penalty/component grid is silently filled in.
The keyword `validation_loss_fn` accepts a callable with the interface
`loss_fn(y_true, y_pred) -> float`, returning a finite real scalar where smaller
is better. It defaults to `mse_loss`, unpenalized MSE pooled equally over observations.
Use `partial(huber_loss, delta=chosen_delta)` for Huber validation scoring, with
one fixed threshold shared across all candidates. Exact ties retain the first
candidate. These scoring/tie conventions are our choices. The former string
`scoring` and separate scoring `huber_delta` arguments have been replaced by this
callable; estimator training arguments such as `huber_delta` remain unchanged.

The returned `TuningResult` contains `best_estimator`, `best_params`, and
`validation_results` (all candidate dictionaries, scores, and actual component counts).
The best estimator is retained from its training-only fit. There is no combined
train-plus-validation refit and no test-set argument. The caller must provide
disjoint chronological samples; the tuning helper cannot infer dates from X.

Tuning runs separately for each specification: select lambda within ENet+H,
K within PCR, and K within PLS. Evaluate and report every tuned specification
on the test sample. A fixed specification has a single candidate and is still
evaluated. The outer experiment loop repeats this for each test year and collects
out-of-sample predictions separately for every specification; it does not discard
specifications after comparing their validation or test performance. The current
demo now runs both 1987 and 1988 and reports annual and pooled performance.

### Callable evaluation losses (updated 2026-09-21)

`core.losses` provides `mse_loss(y_true, y_pred)` and
`huber_loss(y_true, y_pred, *, delta)`. Both require matching, finite, nonempty
one-dimensional arrays and return an observation mean. Column vectors are rejected
to prevent accidental broadcasting. When both inputs are pandas Series, their
indices must match in the same order. Otherwise pairing is positional. No missing
observations are silently dropped. Huber uses the same normalization as training.

These evaluation functions reuse the residual-based kernels in the same module.
The numerical training losses and gradients are unchanged. Public prediction
scoring takes observed/predicted values; `squared_loss` and `huber_residual_loss`
take residuals, with separate gradient functions for fitting. Imports formerly
under `core.models.losses`, `core.models.optimizers`, and `core.evaluation.losses`
have moved to `core.losses` and `core.optimizers`.

Configure `validation_loss_fn` and `test_loss_fn` independently. Changing either
evaluation callable does not change a model's training objective. Custom validation
callables must follow the smaller-is-better convention; a raw R-squared score would
require conversion to a quantity to minimize. Keep the test loss definition fixed
across specifications when comparing their reported performance.

```python
from functools import partial
from core.losses import mse_loss, huber_loss

validation_loss_fn = mse_loss
test_loss_name, test_loss_fn = "Huber (delta=0.1)", partial(huber_loss, delta=0.1)
test_loss = test_loss_fn(observed, prediction)
```

The threshold above is illustrative. The demo defaults both callables to MSE,
with names and parameters read from the configuration. The runner builds
the display label from those settings. Zero-benchmark R-squared remains a
separate squared-error metric regardless of the chosen test loss.

### Annual windows

`annual_splits` defaults to 30 tests, 1987-2016. The first window trains on observed
1957-1974 rows and validates on 1975-1986. Each year training expands by one year,
validation rolls forward with length 12 years, and test is that single year.
The final window trains through 2003 and validates on 2004-2015.

Dates label **realized target return months**. A December feature row predicting
January returns is assigned using January, consistently throughout the pipeline.
This target-month labeling is an explicit implementation convention for the paper's
reported calendar windows. Fit-time information availability still depends on correct
feature construction; the splitter alone cannot prevent delayed-release leakage.

The splitter returns integer row positions and preserves input order; use NumPy
indexing or pandas `.iloc`, not label-based `.loc`. Repeated dates across stocks are
normal. Missing dates, numeric YYYYMM dates, nonmonthly PeriodIndex inputs, invalid
year bounds, and empty windows raise. It does not require every month to be present
or implement annual/overlapping return targets.

## Usage

```python
from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.losses import mse_loss
from core.model_selection import tune_hyperparameters

ols = LinearRegressor().fit(X_train, y_train)
prediction = ols.predict(X_test)  # (n_test,), in original return units

# Threshold is supplied by our approved rule, not guessed inside the estimator.
robust = LinearRegressor(loss="huber", huber_delta=chosen_delta)
robust.fit(X_train, y_train)

pcr = PCRRegressor(n_components=chosen_k).fit(X_train, y_train)
pls = PLSRegressor(n_components=chosen_k).fit(X_train, y_train)

tuning_result = tune_hyperparameters(
    LinearRegressor,
    [{"loss": "huber", "huber_delta": chosen_delta,
      "penalty_strength": strength, "rho": .5}
     for strength in approved_lambda_grid],
    X_train, y_train, X_validation, y_validation,
    validation_loss_fn=mse_loss,
)
tuned_model = tuning_result.best_estimator
test_prediction = tuned_model.predict(X_test)
test_loss_fn = mse_loss
test_loss = test_loss_fn(y_test, test_prediction)
```

Run from the repository root in the project environment:

```powershell
python -B -m unittest discover -s tests -v
python -B -m core.data_query.generate_experimental_data  # Once, if datasets do not exist.
python -B main.py --config model_demo
python -B main.py --config model_demo_nonlinear
```

The runner loads a saved dataset via `DATA.DATASET`; generation is a separate
preparation step. The example Huber threshold, grids, and generated feature
distribution are not empirical replication defaults. By default the run
writes no files. Setting `RUN.OUTPUT_DIR` enables saving the resolved config,
dataset identity/hashes, predictions, all tuning scores, annual/pooled metrics,
and environment versions. Generation settings are recorded with the dataset,
not in the run YAML or `default.py`.

## Remaining replication work

Before empirical runs: approve feature construction/eligibility, intercept/scaling
choices, the Huber threshold rule, exact lambda/K grids, and validation scoring.
Build the final dated panel and benchmark memory/runtime on realistic dimensions.
The annual model run loop and prediction outputs are implemented; large-sample
streaming, resumable fits, and estimator checkpoint persistence remain future work.
Trees, spline/group lasso, neural networks, portfolio evaluation, and replication
table generation are not part of this implementation.
