"""Reserved forecast comparison interface for GKX Table 3."""


def diebold_mariano_test(y_true, prediction_1, prediction_2, return_month, *, max_lag):
    """Future implementation: monthly mean squared-error differences and HAC SE.

    Planned output: dict with statistic, two-sided p_value, mean_loss_difference,
    n_months. Positive means prediction_2 is better. max_lag will explicitly
    select the Newey-West bandwidth. No estimate is returned until implemented.
    """
    raise NotImplementedError("GKX monthly cross-sectional Diebold-Mariano test is deferred.")
