"""PCR (economy SVD) and single-target PLS (SIMPLS), followed by OLS.

Both use only training observations to learn feature transforms and directions.
With fit_intercept=False these operate on uncentered second moments; the default
True uses centered covariances. No automatic component grid is assumed.
"""

import numpy as np
from copy import copy
from numpy.linalg import svd

from core.model_inputs import validate_model_inputs
from ._helpers import clear_fit, prediction_data, prepare_features


class PCRRegressor:
    """Regress on the n_components highest-variance training predictor directions.

    components_ has shape (k, p) in the centered/scaled feature coordinates.
    coef_ and intercept_ operate directly on original input feature units.
    Requests exceeding numerical rank raise rather than silently changing k.
    """

    def __init__(self, n_components, *, fit_intercept=True, standardize=False):
        self.n_components = n_components
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    def fit(self, X, y):
        clear_fit(self)
        X, y, names = validate_model_inputs(X, y)
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(self.n_components, (int, np.integer)) or self.n_components < 0:
            raise ValueError("n_components must be a nonnegative integer.")
        Z, offset, scale = prepare_features(
            X, fit_intercept=self.fit_intercept, standardize=self.standardize,
        )
        max_components = min(Z.shape[1], Z.shape[0] - int(self.fit_intercept))
        if self.n_components > max_components:
            raise ValueError(f"n_components exceeds the maximum possible rank {max_components}.")
        # Economy SVD avoids forming X'X, which squares the condition number.
        if self.n_components == 0:
            left, singular_values, right = np.empty((len(Z), 0)), np.empty(0), np.empty((0, Z.shape[1]))
        else:
            # NumPy's ILP64 backend on SCRP supports matrices exceeding 2**31 elements.
            # Keep economy SVD and float64; large runs require an ILP64 NumPy build.
            left, singular_values, right = svd(Z, full_matrices=False)
        rank_tolerance = np.finfo(float).eps * max(Z.shape) * (singular_values[0] if singular_values.size else 0)
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if self.n_components > rank:
            raise ValueError(f"n_components={self.n_components} exceeds numerical rank {rank}.")
        k = self.n_components
        components = right[:k].copy()
        scores = left[:, :k] * singular_values[:k]
        del left, right
        target_offset = float(y.mean()) if self.fit_intercept else 0.0
        component_coef = np.linalg.lstsq(scores, y - target_offset, rcond=None)[0]

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = names
        self.feature_offset_ = offset
        self.feature_scale_ = scale
        self.n_components_ = k
        self.components_ = components
        self.component_coef_ = component_coef
        self.target_offset_ = target_offset
        self.singular_values_ = singular_values[:k].copy()
        self.rank_ = rank
        self.coef_ = (components.T @ component_coef) / scale
        self.intercept_ = float(target_offset - offset @ self.coef_)
        return self

    def with_components(self, n_components):
        """Reuse this training SVD for a smaller PCR candidate, without refitting.

        Only valid for an already fitted model. Centering, scaling, and component
        regression all remain training-only. Zero components predicts the training
        mean when fitting an intercept, and zero otherwise.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before selecting components.")
        if (isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, (int, np.integer)) or
                not 0 <= n_components <= self.n_components_):
            raise ValueError("Requested component count is outside the fitted PCR path.")
        model = copy(self)
        model.n_components = model.n_components_ = n_components
        model.components_ = self.components_[:n_components].copy()
        model.component_coef_ = self.component_coef_[:n_components].copy()
        model.singular_values_ = self.singular_values_[:n_components].copy()
        model.coef_ = (model.components_.T @ model.component_coef_) / model.feature_scale_
        model.intercept_ = float(model.target_offset_ - model.feature_offset_ @ model.coef_)
        return model

    def transform(self, X):
        X = prediction_data(self, X)
        return ((X - self.feature_offset_) / self.feature_scale_) @ self.components_.T

    def predict(self, X):
        X = prediction_data(self, X)
        return X @ self.coef_ + self.intercept_


class PLSRegressor:
    """Single-response SIMPLS using covariance with y to extract components.

    n_components is an upper bound: extraction stops if predictive covariance
    or an independent direction is exhausted. Inspect n_components_ and
    stop_reason_. Constant training targets yield the intercept-only forecast
    when fit_intercept=True. tol governs numerical exhaustion, not tuning of k.
    """

    def __init__(self, n_components, *, fit_intercept=True, standardize=False, tol=1e-12):
        self.n_components = n_components
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.tol = tol

    def fit(self, X, y):
        clear_fit(self)
        X, y, names = validate_model_inputs(X, y)
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(self.n_components, (int, np.integer)) or self.n_components < 0:
            raise ValueError("n_components must be a nonnegative integer.")
        if not np.isfinite(self.tol) or not 0 < self.tol < 1:
            raise ValueError("tol must be finite and between 0 and 1.")
        Z, offset, scale = prepare_features(
            X, fit_intercept=self.fit_intercept, standardize=self.standardize,
        )
        n, p = Z.shape
        if self.n_components > min(p, n - int(self.fit_intercept)):
            raise ValueError("n_components exceeds min(n_features, n_rows - fit_intercept).")
        target_offset = float(y.mean()) if self.fit_intercept else 0.0
        centered_y = y - target_offset
        if self.fit_intercept and np.all(y == y[0]):
            target_offset = float(y[0])
            centered_y = np.zeros_like(y)
        covariance = Z.T @ centered_y
        initial_covariance_norm = np.linalg.norm(covariance)
        feature_norm = np.linalg.norm(Z)
        directions = np.empty((p, self.n_components))
        scores = np.empty((n, self.n_components))
        orthogonal_loadings = np.empty((p, self.n_components))
        k = 0
        stop_reason = "requested_components"

        for j in range(self.n_components):
            covariance_norm = np.linalg.norm(covariance)
            if covariance_norm <= self.tol * initial_covariance_norm:
                stop_reason = "covariance_exhausted"
                break
            direction = covariance / covariance_norm
            score = Z @ direction
            # Reorthogonalization limits numerical drift and preserves score = Z @ direction.
            for _ in range(2):
                projection = scores[:, :j].T @ score
                score -= scores[:, :j] @ projection
                direction -= directions[:, :j] @ projection
            score_norm = np.linalg.norm(score)
            if score_norm <= self.tol * feature_norm:
                stop_reason = "rank_exhausted"
                break
            score /= score_norm
            direction /= score_norm
            loading = Z.T @ score
            orthogonal = loading.copy()
            for _ in range(2):
                orthogonal -= orthogonal_loadings[:, :j] @ (orthogonal_loadings[:, :j].T @ orthogonal)
            loading_norm = np.linalg.norm(orthogonal)
            if loading_norm <= self.tol * np.linalg.norm(loading):
                stop_reason = "rank_exhausted"
                break
            directions[:, j] = direction
            scores[:, j] = score
            orthogonal_loadings[:, j] = orthogonal / loading_norm
            k += 1
            for _ in range(2):
                basis = orthogonal_loadings[:, :k]
                covariance -= basis @ (basis.T @ covariance)

        component_coef = np.linalg.lstsq(scores[:, :k], centered_y, rcond=None)[0]
        self.n_features_in_ = p
        self.feature_names_in_ = names
        self.feature_offset_ = offset
        self.feature_scale_ = scale
        self.target_offset_ = target_offset
        self.n_components_ = k
        self.stop_reason_ = stop_reason
        self.components_ = directions[:, :k].T.copy()
        self.component_coef_ = component_coef
        self.coef_ = (self.components_.T @ component_coef) / scale
        self.intercept_ = float(target_offset - offset @ self.coef_)
        return self

    def transform(self, X):
        X = prediction_data(self, X)
        return ((X - self.feature_offset_) / self.feature_scale_) @ self.components_.T

    def predict(self, X):
        X = prediction_data(self, X)
        return X @ self.coef_ + self.intercept_
