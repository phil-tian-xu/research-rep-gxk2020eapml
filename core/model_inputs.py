"""Shared checks and numeric conversion for model data and integer settings.

These checks do not fit models, transform features, or select validation samples.
"""

import numpy as np
import pandas as pd


def feature_matrix(X):
    """Return a finite, dense float64 matrix and optional ordered column names."""
    names = None
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")
        names = tuple(X.columns)
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError("X must contain real numbers.")
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a dense numeric matrix.") from exc
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must have shape (n_observations, n_features), with features > 0.")
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinity; prepare/impute features before fitting.")
    return array, names


def target_vector(y, n_observations):
    if np.iscomplexobj(np.asarray(y)):
        raise ValueError("y must contain real numbers.")
    array = np.asarray(y, dtype=np.float64)
    if array.ndim != 1 or len(array) != n_observations:
        raise ValueError("y must have shape (n_observations,); column vectors are not accepted.")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("y must be nonempty and finite.")
    return array


def validate_model_inputs(X, y):
    """Check aligned X/y and return dense float64 arrays plus feature names.

    X must be a finite (n, p) matrix and y a finite (n,) vector. Pandas indices
    must already match; no alignment, imputation, scaling, or fitting is done.
    The same checks apply to training, validation, and test observations.
    """
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match in the same order; no automatic alignment.")
    array, names = feature_matrix(X)
    target = target_vector(y, len(array))
    return array, target, names


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
