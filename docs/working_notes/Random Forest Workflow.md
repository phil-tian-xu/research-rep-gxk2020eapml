# Random Forest pilot

Local development uses the same annual Parquet workflow and launcher:

```powershell
python -B run_experiment.py rf --local --sample p05 --years 1987 --cpus 8
```

Baseline: `config/gkx_rf_v1.yaml`. This one-year check uses the actual pilot
grid: 60 candidate forests, each with 300 trees. It is not an instant smoke test.
Install updated project dependencies with `uv sync` before running in another
environment. The tested scikit-learn version is recorded in every annual result.

## Specification and sources

The main paper, Section 1.6, and online appendix Algorithms 2–3 specify
least-squares CART, bootstrap aggregation and random candidate features at each
split. Table A.5 specifies 300 trees, depth 1–6 and the feature-count sequence
`3, 5, 10, 20, 30, 50, ...`. Figure 3 reports average tree depth.

Our explicit completion of that incomplete grid is
`[3, 5, 10, 20, 30, 50, 100, 200, 400, 920]`; values after 50 are pilot choices.
The remaining conventional choices are:

- Scikit-learn RandomForestRegressor; bootstrap n_train stock-month rows with
  replacement separately for each tree. No date blocks, weights or OOB selection.
- Squared-error splits, arithmetic mean leaves, average tree predictions.
- Minimum split size 2, leaf size 1; no pruning or positive impurity threshold.
  Nodes can stop before the requested depth. The root has depth zero.
- Library split search can inspect extra features when no valid split is found
  among the sampled features. Random seeds are `2020 + test_year`, shared across
  candidates for that year; exact ties choose the first candidate (depth first).
- Existing panel transformations are preserved. No extra scaling, separately
  fitted intercept, Huber loss or new missing-value imputation is applied.
- Float64 Parquet files are unchanged. Predictors are converted to float32,
  matching scikit-learn's internal tree representation; targets remain float64.
  Nonfinite values are rejected. This precision choice can affect close splits.

Training expands from 1957, validation is the preceding 12 years, and testing
is one year. For 1987 this means training 1957–1974, validation 1975–1986.
All candidates fit training only; validation MSE chooses the forest. There is
no train-plus-validation refit and no use of test outcomes for selection.

See the [scikit-learn estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
MATLAB simulation settings are not used to override the empirical paper.

## Execution and outputs

Training and validation predictor arrays coexist as float32 during tuning.
Only the best forest and current candidate are retained. Training/validation
arrays are released before loading test data. Trees fit in parallel using
SLURM_CPUS_PER_TASK, otherwise the launcher's OMP_NUM_THREADS, otherwise one CPU.
Candidates run sequentially; prediction accumulates trees serially so scores do
not depend on worker completion order. No multiprocessing copies of the panel
are introduced by this workflow. Actual peak memory still requires measurement.

The launcher saves resolved config, software/source identity, logs and memory
reports in a timestamped experiment under `results/experiments/`. Existing
list/resume/collect commands apply; checkpoints are per completed model-year,
not per candidate. Interrupted tuning restarts that year.

Table 1 uses pooled zero-benchmark R2 (not the mean of annual R2). Figure 3 uses
the selected forest's mean realized tree depth. Reports also retain the depth
cap, features per split, every validation score, each tree's depth and leaf
count, selected estimator parameters, seeds, feature names and importances.
Selected fitted tree objects are not serialized; predictions and diagnostics
are saved, and the recorded seed/environment support rerunning the fit.

Use a fresh experiment after updating execution sources. Leave active server
checkouts unchanged until their existing jobs finish. Full-universe performance
and resource requirements have not yet been verified for RF.

Verification: 192 tests passed with scikit-learn 1.9.1. RF checks cover bootstrap
leaf means, forest averaging, thread-count reproducibility, independent candidate
validation scores, training-only selected predictions, complete/staged equality,
resume skipping, invalid grids and report generation. The real 5% annual run
has not been executed as part of this implementation.
