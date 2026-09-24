"""Metrics for evaluating predictions, separate from fitting objectives."""

import numpy as np
import pandas as pd

from core.losses import mse_loss
from core.model_inputs import positive_integer


def zero_benchmark_r2(y_true, y_pred):
    """1 - SSE / sum(y_true**2); undefined (NaN) when the denominator is zero.

    Compute on all concatenated out-of-sample observations for pooled reporting,
    rather than averaging the annual R-squared values.
    """
    error = mse_loss(y_true, y_pred)
    benchmark_error = mse_loss(y_true, np.zeros_like(y_true, dtype=float))
    return float("nan") if benchmark_error == 0 else 1.0 - error / benchmark_error


def stock_prediction_summary(predictions, *, size_count=1000):
    """Table 1 statistics for ONE model's complete OOS panel.

    Required columns: stock_id, return_month, market_equity, observed, prediction.
    market_equity must be positive formation-date equity supplied by the caller.
    Months are normalized to monthly periods. Ties use ascending stock_id for
    both size groups. Months with fewer than size_count stocks use all available
    stocks; top/bottom groups can overlap. No missing rows are silently dropped.
    Returns pooled raw R2, percentage R2, counts, SSE and zero-benchmark SSE.
    This function neither fits models nor imposes a particular OOS date range.
    """
    positive_integer(size_count, "size_count")
    columns = ["stock_id", "return_month", "market_equity", "observed", "prediction"]
    panel = predictions.loc[:, columns].copy()
    if panel.empty or panel.isna().any().any():
        raise ValueError("Predictions must be nonempty with no missing required values.")
    panel["return_month"] = pd.PeriodIndex(panel["return_month"], freq="M")
    if panel["return_month"].isna().any() or panel.duplicated(["stock_id", "return_month"]).any():
        raise ValueError("Require one observation per stock and return month.")
    mse_loss(panel.observed, panel.prediction)
    if np.iscomplexobj(panel.market_equity.to_numpy()):
        raise ValueError("market_equity must be real.")
    equity = panel.market_equity.to_numpy(dtype=float)
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("market_equity must be finite and positive.")
    panel["market_equity"] = equity
    largest = panel.sort_values(
        ["return_month", "market_equity", "stock_id"], ascending=[True, False, True],
    ).groupby("return_month", sort=False).head(size_count)
    smallest = panel.sort_values(
        ["return_month", "market_equity", "stock_id"], ascending=[True, True, True],
    ).groupby("return_month", sort=False).head(size_count)
    records = []
    for sample, rows in (("all", panel), ("top", largest), ("bottom", smallest)):
        observed = rows.observed.to_numpy(dtype=float)
        predicted = rows.prediction.to_numpy(dtype=float)
        r2 = zero_benchmark_r2(observed, predicted)
        records.append({"sample": sample, "n_obs": len(rows),
                        "n_months": rows.return_month.nunique(),
                        "sse": float(np.sum((observed - predicted)**2)),
                        "benchmark_sse": float(np.sum(observed**2)),
                        "r2_zero": r2, "r2_percent": 100 * r2})
    return pd.DataFrame(records).set_index("sample")
