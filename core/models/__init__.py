"""Small, independent estimators sharing fit(X, y) and predict(X)."""

from .dimension_reduction import PCRRegressor, PLSRegressor
from .linear import LinearRegressor

__all__ = ["LinearRegressor", "PCRRegressor", "PLSRegressor"]
