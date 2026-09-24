# Evaluation methods

Implemented 2026-09-21 under `core/evaluation`. The functions are independent of
the annual runner and perform no fitting, file saving, or plotting. Existing
`zero_benchmark_r2` imports remain compatible. No full empirical replication has
been run as part of this change.

## Table 1

`stock_prediction_summary(predictions, size_count=1000)` consumes ONE model's
complete OOS panel with `stock_id`, `return_month`, `market_equity`, `observed`,
and `prediction`. Returns all/top/bottom sample rows with observation/month counts,
SSE, zero-benchmark SSE, raw R2 and percentage R2. Membership changes monthly;
errors are pooled over observations, never averaged from monthly R2s. Size does
not weight the errors. All three groups use predictions from the full-sample fit.

Caller must supply formation-date market equity and the intended test period
(1987-2016 for the paper). Missing/nonpositive equity and duplicate stock-months
fail explicitly. Our tie convention is ascending stock ID in both groups; if
fewer than 1,000 stocks are available, all are used. Groups may overlap. These
are explicit implementation choices, not verified author conventions. Ensure
all models are evaluated on the same eligible panel before comparing columns.

## Figure 3

Use selected fitted state at each annual reestimation:

- ENet: `nonzero_count(model.coef_, tolerance=...)`, excluding intercept.
- PCR/PLS: `selected_component_count(model)` uses realized `n_components_`.
- GLM: `nonzero_count(coefficients, groups=...)`, mapping predictor groups to
  spline coefficient positions rather than counting basis coefficients.
- RF: `average_tree_depth(depths)` consumes realized depth per tree, root = 0.
- GBRT: `distinct_split_count(split_features, feature_groups=...)`; supply actual
  split IDs without leaf sentinels, optionally mapped to base characteristics.

Future tree/GLM implementations must extract their fitted state for these small
numeric functions. No speculative estimator adapters have been added. Exact
feature-versus-characteristic counting remains an author-code verification item.

## Figures 4/5 and Table 4

`zero_out_importance(predict, X_train, y_train, groups=..., batch_size=10000)`
computes signed R2 reductions on training data with the fitted predictor fixed.
It accepts a DataFrame and explicit groups of column names; columns are jointly
zeroed within each group. For a neural ensemble, pass its averaged-prediction
callable. Batch prediction must not depend on other rows in a batch (e.g. use
neural-network evaluation mode). Original feature data are not changed.

The function computes (ablated SSE - baseline SSE) / sum(y**2), pooling batches
before dividing. One baseline prediction per batch is reused across groups.
It returns a Series of raw importance by group, including negative values;
an all-zero target produces NaN. It neither fits nor normalizes.

Grouping is deliberately caller-supplied. A possible characteristic grouping
jointly removes a characteristic and its eight macro interactions; industry can
group all SIC dummies; macro grouping can remove all characteristic interactions
involving one macro variable. This is NOT established as the author's exact
group aggregation. Joint ablation differs from summing separate ablations.
Normalization/averaging order and ensemble aggregation also remain replication
choices rather than fully verified paper details.

Attach model and year to each Series and concatenate annual records:
`model, year, predictor, importance`. Pass characteristic and macro panels
separately to `summarize_importance(annual, negative_policy=...)`.

- Figure 4/Table 4 convention: average raw annual scores, THEN normalize within
  each model over all supplied predictors. Select top 20 only AFTER normalization
  for Figure 4. `negative_policy='raise'` refuses negative mean scores;
  `'clip'` clips mean scores only for shares. Neither changes raw means or ranks.
  An all-zero normalization total returns NaN shares, not invented equal weights.
- Figure 5: rank signed importance within each model/year (1 = best, average
  ranks for ties), then average ranks over years. Sum these average ranks across
  models for a common ascending row order. Figure 5 has 94 characteristics plus
  industry and 11 model columns. Rank colors should darken toward rank 1.
- The summary requires a complete common predictor/year panel across models so
  that missing runs do not silently distort cross-model rankings.

The output contains `model, predictor, mean_importance, mean_rank,
relative_importance, overall_rank_sum`. Graph rendering is left to a future
reporting step. SSD/impurity robustness analysis is deferred.

## Deferred work and runner integration

`diebold_mariano_test(..., max_lag=...)` is a Table 3 placeholder that raises
`NotImplementedError`. Future implementation will cross-sectionally average
squared-loss differences each month and use a Newey-West standard error. The
bandwidth must be explicit. No fake test statistic is returned.

Annual targets/Table 2 and portfolio/Sharpe evaluation remain outside this scope.
The runner has NOT been modified. When integrating, record complexity and
training importance before discarding the selected annual estimator. Saved test
predictions alone cannot reconstruct importance. Join stock identifiers and
formation-date equity to test predictions before Table 1 evaluation.

Validation and final evaluation are distinct: minimizing validation MSE is
equivalent to maximizing zero-benchmark R2 on a fixed validation panel. Huber
fitting does not change the squared-error Table 1 metric.

Verification: `python -B -m unittest discover -s tests -p test_evaluation.py -v`.
Tests include hand-calculated pooling, monthly size membership, nonlinear/group
ablation, signed scores, batch invariance, fitted-model compatibility, and the
difference between ranking average scores and averaging annual ranks.
