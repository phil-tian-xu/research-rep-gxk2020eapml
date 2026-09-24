"""Shared GKX training kernels, prediction losses, and parameter penalties.

Prediction API: mse_loss(y_true, y_pred), huber_loss(y_true, y_pred, delta=...).
Training kernels and gradients take residuals defined as prediction - target.
Losses are observation means, without a factor of 1/2 on squared loss.
"""

import numpy as np
import pandas as pd


def _residual_vector(residuals):
    if np.iscomplexobj(np.asarray(residuals)):
        raise ValueError("Residuals must be real.")
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Residuals must be a finite, nonempty one-dimensional vector.")
    return residuals


def validate_huber_delta(delta):
    if delta is None or not np.isscalar(delta) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("huber_delta must be an explicit finite positive number.")


def validate_penalty(penalty_strength, rho):
    if not np.isscalar(penalty_strength) or not np.isfinite(penalty_strength) or penalty_strength < 0:
        raise ValueError("penalty_strength must be finite and nonnegative.")
    if not np.isscalar(rho) or not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]: 0 is lasso and 1 is ridge.")


def squared_loss(residuals):
    residuals = _residual_vector(residuals)
    return float(np.mean(residuals**2))


def squared_loss_gradient(residuals):
    residuals = _residual_vector(residuals)
    return 2.0 * residuals / residuals.size


def huber_residual_loss(residuals, delta):
    validate_huber_delta(delta)
    absolute = np.abs(_residual_vector(residuals))
    quadratic = np.minimum(absolute, delta)
    return float(np.mean(quadratic**2 + 2.0 * delta * (absolute - quadratic)))


def huber_loss_gradient(residuals, delta):
    validate_huber_delta(delta)
    residuals = _residual_vector(residuals)
    return 2.0 * np.clip(residuals, -delta, delta) / residuals.size


def elastic_net_penalty(coefficients, penalty_strength, rho):
    """lambda * (1-rho) * ||beta||_1 + lambda * rho/2 * ||beta||_2^2."""
    validate_penalty(penalty_strength, rho)
    coefficients = _residual_vector(coefficients)
    return float(penalty_strength * (
        (1.0 - rho) * np.abs(coefficients).sum() + 0.5 * rho * (coefficients @ coefficients)
    ))


def _prediction_residuals(y_true, y_pred):
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred indices must match in the same order.")
    observed, predicted = np.asarray(y_true), np.asarray(y_pred)
    if observed.ndim != 1 or predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same one-dimensional shape (n,).")
    if observed.size == 0:
        raise ValueError("Prediction losses require at least one observation.")
    if np.iscomplexobj(observed) or np.iscomplexobj(predicted):
        raise ValueError("y_true and y_pred must contain real numbers.")
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must be finite; missing observations are not dropped.")
    return predicted - observed


def mse_loss(y_true, y_pred):
    """Mean squared prediction error in squared target units."""
    return squared_loss(_prediction_residuals(y_true, y_pred))


def huber_loss(y_true, y_pred, *, delta):
    """Mean GKX Huber error, with delta in the same units as the target."""
    return huber_residual_loss(_prediction_residuals(y_true, y_pred), delta)
