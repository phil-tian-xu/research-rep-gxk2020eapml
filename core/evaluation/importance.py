"""Training-sample zero-out importance and summaries for GKX Figures 4/5, Table 4."""

import numpy as np
import pandas as pd

from core.losses import mse_loss
from core.model_inputs import positive_integer, target_vector, feature_matrix


def zero_out_importance(predict, X, y, *, groups, batch_size=10000):
    """Return signed R2 reductions indexed by explicit predictor-group names.

    predict is a fixed fitted model/ensemble's prediction callable. X must be a
    numeric DataFrame; groups maps names to nonempty lists of X column names.
    All columns in a group are jointly zeroed in the supplied input coordinates.
    Groups may overlap or omit columns. No automatic interaction/SIC grouping,
    normalization, refitting, or clipping is performed. Use the training sample
    for the paper's importance figures. Work is batched without copying full X.
    Predictions must be row-independent for batching to preserve results.
    """
    if not callable(predict):
        raise ValueError("predict must be callable.")
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or X.shape[1] == 0:
        raise ValueError("X must be a DataFrame with unique feature columns.")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y indices must match in the same order.")
    target = target_vector(y, len(X))
    positive_integer(batch_size, "batch_size")
    if not groups:
        raise ValueError("Supply explicit predictor groups.")
    selected = {}
    for name, columns in groups.items():
        if isinstance(columns, (str, bytes)):
            raise ValueError("Each group must contain a list of column names.")
        columns = list(columns)
        if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(X.columns):
            raise ValueError(f"Invalid feature columns for group {name!r}.")
        selected[name] = columns
    baseline_sse = 0.0
    removed_sse = pd.Series(0.0, index=list(selected), name="importance")
    denominator = float(np.sum(target**2))
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start:start + batch_size]
        feature_matrix(batch)
        observed = pd.Series(target[start:start + len(batch)], index=batch.index)
        baseline_sse += len(batch) * mse_loss(observed, predict(batch))
        altered = batch.copy()
        for name, columns in selected.items():
            altered[columns] = 0.0
            removed_sse.loc[name] += len(batch) * mse_loss(observed, predict(altered))
            altered[columns] = batch[columns]
    if denominator == 0:
        return removed_sse * np.nan
    return (removed_sse - baseline_sse) / denominator


def summarize_importance(annual, *, negative_policy):
    """Summarize a complete model/year/predictor importance panel.

    Required columns: model, year, predictor, importance. Pass characteristic
    and macro panels separately: normalization is over the supplied predictors.
    Every model/year must contain the same predictors and every model the same
    years. Returns one row per model/predictor with mean_importance,
    relative_importance, mean_rank and overall_rank_sum.

    Convention: average raw scores over years THEN normalize (Figure 4/Table 4).
    Rank raw scores within each year THEN average ranks (Figure 5); 1 is best,
    ties receive average ranks. overall_rank_sum orders heatmap rows ascending.
    negative_policy must explicitly be 'raise' or 'clip' for normalization;
    clipping affects only averaged scores used as shares, never raw ranks/scores.
    All-zero shares are NaN because a relative allocation is undefined.
    """
    if negative_policy not in {"raise", "clip"}:
        raise ValueError("negative_policy must be 'raise' or 'clip'.")
    frame = annual.loc[:, ["model", "year", "predictor", "importance"]].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Importance records must be nonempty and complete.")
    if frame.duplicated(["model", "year", "predictor"]).any():
        raise ValueError("Duplicate model/year/predictor records.")
    frame["importance"] = target_vector(frame.importance, len(frame))
    counts = frame.groupby(["model", "year"]).size()
    expected_pairs = frame.model.nunique() * frame.year.nunique()
    if len(counts) != expected_pairs or not (counts == frame.predictor.nunique()).all():
        raise ValueError("Require the same years and predictors for every model.")
    frame["rank"] = frame.groupby(["model", "year"]).importance.rank(
        ascending=False, method="average",
    )
    summary = frame.groupby(["model", "predictor"], sort=False).agg(
        mean_importance=("importance", "mean"), mean_rank=("rank", "mean"),
    ).reset_index()
    scores = summary.mean_importance
    if negative_policy == "raise" and (scores < 0).any():
        raise ValueError("Negative mean importance: explicitly choose a normalization policy.")
    weights = scores.clip(lower=0) if negative_policy == "clip" else scores
    totals = weights.groupby(summary.model).transform("sum")
    summary["relative_importance"] = weights / totals.replace(0, np.nan)
    summary["overall_rank_sum"] = summary.groupby("predictor").mean_rank.transform("sum")
    return summary
