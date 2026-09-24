"""Internal feature transforms and fitted-state helpers for the estimators."""

import numpy as np

from core.model_inputs import feature_matrix


def prediction_data(model, X):
    if not hasattr(model, "coef_"):
        raise RuntimeError("Call fit before predict or transform.")
    array, names = feature_matrix(X)
    if array.shape[1] != model.n_features_in_:
        raise ValueError(f"Expected {model.n_features_in_} features, received {array.shape[1]}.")
    if model.feature_names_in_ is not None and names != model.feature_names_in_:
        raise ValueError("Prediction requires a DataFrame with the fitted columns in the same order.")
    return array


def prepare_features(X, *, fit_intercept, standardize):
    """Center iff fitting an intercept; optional population SD scaling (ddof=0)."""
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be boolean.")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be boolean.")
    constant = X.min(axis=0) == X.max(axis=0)
    offset = X.mean(axis=0) if fit_intercept else np.zeros(X.shape[1])
    if fit_intercept:
        # Repeated decimal values can have a rounded mean and a spurious tiny SD.
        offset[constant] = X[0, constant]
    scale = X.std(axis=0) if standardize else np.ones(X.shape[1])
    scale = np.where(constant | (scale == 0), 1.0, scale)
    transformed = (X - offset) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Feature centering/scaling overflowed; check input magnitudes.")
    return transformed, offset, scale


def clear_fit(model):
    """A failed refit must not leave stale fitted coefficients usable."""
    for name in list(vars(model)):
        if name.endswith("_"):
            delattr(model, name)
