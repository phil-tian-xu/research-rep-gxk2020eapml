"""Dense Huber regression with the unweighted group-lasso penalty (GKX B.1)."""

import warnings
import numpy as np

from core.model_inputs import validate_model_inputs
from core.losses import huber_residual_loss, huber_loss_gradient, validate_huber_delta, validate_penalty
from core.optimizers import accelerated_proximal_gradient
from ._helpers import clear_fit, prediction_data, prepare_features


def validate_groups(groups, n_features):
    """Every coefficient belongs to exactly one nonempty, contiguous group."""
    sizes = np.asarray(groups)
    if (sizes.ndim != 1 or not sizes.size or sizes.dtype.kind not in "iu" or
            (sizes <= 0).any() or sizes.sum() != n_features):
        raise ValueError("Group sizes must be positive integers partitioning all features.")
    return np.r_[0, np.cumsum(sizes)].astype(int)


def group_prox(values, step, strength, boundaries):
    """Block soft thresholding; no sqrt(group size) weighting."""
    result = values.copy()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:end]
        norm = np.linalg.norm(block)
        result[start:end] = block * max(0., 1 - step * strength / norm) if norm else 0.
    return result


class GroupHuberRegressor:
    """Penalizes spline groups; optional intercept is jointly fitted and unpenalized.

    Input is an already constructed basis matrix. Knots are fitted outside this
    estimator, once per training window. No implicit spline or group convention.
    """
    def __init__(self, *, groups, penalty_strength, huber_delta, huber_quantile=None,
                 loss="huber", fit_intercept=True, standardize=False,
                 max_iter=50000, tol=1e-8, initial_step=1.):
        self.groups = groups
        self.penalty_strength = penalty_strength
        self.huber_delta = huber_delta
        self.huber_quantile = huber_quantile
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter, self.tol, self.initial_step = max_iter, tol, initial_step

    def fit(self, X, y):
        clear_fit(self)
        X, y, names = validate_model_inputs(X, y)
        if self.loss != "huber" or self.huber_quantile is not None:
            raise ValueError("Supply a resolved fixed Huber threshold for group regression.")
        validate_huber_delta(self.huber_delta)
        validate_penalty(self.penalty_strength, 0.)
        boundaries = validate_groups(self.groups, X.shape[1])
        Z, offset, scale = prepare_features(X, fit_intercept=self.fit_intercept, standardize=self.standardize)
        p = Z.shape[1]

        def smooth(parameters):
            residual = Z @ parameters[:p] - y
            if self.fit_intercept:
                residual += parameters[-1]
            derivative = huber_loss_gradient(residual, self.huber_delta)
            gradient = Z.T @ derivative
            if self.fit_intercept:
                gradient = np.r_[gradient, derivative.sum()]
            return huber_residual_loss(residual, self.huber_delta), gradient

        def penalty(parameters):
            return self.penalty_strength * sum(np.linalg.norm(parameters[a:b])
                                               for a, b in zip(boundaries[:-1], boundaries[1:]))

        # Boundaries exclude the final intercept; the prox copies it unchanged.
        result = accelerated_proximal_gradient(smooth,
            lambda values, step: group_prox(values, step, self.penalty_strength, boundaries),
            penalty, np.zeros(p + int(self.fit_intercept)), max_iter=self.max_iter,
            tol=self.tol, initial_step=self.initial_step)
        self.coef_ = result.coefficients[:p] / scale
        intercept = float(result.coefficients[-1]) if self.fit_intercept else 0.
        self.intercept_ = float(intercept - offset @ self.coef_)
        self.n_features_in_, self.feature_names_in_ = p, names
        self.feature_offset_, self.feature_scale_ = offset, scale
        self.optimization_result_ = result
        self.objective_, self.n_iter_, self.converged_ = result.objective, result.n_iter, result.converged
        self.huber_delta_ = float(self.huber_delta)
        self.n_active_groups_ = sum(int(np.any(self.coef_[a:b] != 0))
                                   for a, b in zip(boundaries[:-1], boundaries[1:]))
        if not result.converged:
            warnings.warn(f"Group Huber fit did not converge in {result.n_iter} iterations; "
                          f"stationarity={result.stationarity:.6g}.", RuntimeWarning, stacklevel=2)
        return self

    def predict(self, X):
        return prediction_data(self, X) @ self.coef_ + self.intercept_
