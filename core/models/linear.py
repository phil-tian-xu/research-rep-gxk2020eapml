"""Pooled linear regression with GKX squared/Huber loss and elastic-net penalty."""

import warnings

import numpy as np
from scipy.linalg import svd

from core.model_inputs import positive_integer, validate_model_inputs
from ._helpers import clear_fit, prediction_data, prepare_features
from core.losses import (
    elastic_net_penalty, huber_residual_loss, huber_loss_gradient,
    squared_loss, squared_loss_gradient, validate_huber_delta, validate_penalty,
)
from core.optimizers import OptimizationResult, accelerated_proximal_gradient, elastic_net_prox


class LinearRegressor:
    """OLS, robust linear, lasso, ridge, or elastic net via explicit settings.

    X has shape (n, p), y shape (n,), and predict returns shape (n,).
    Rows are equally weighted. penalty_strength is the paper's lambda; rho is
    its L2 share (0 = lasso, 1 = ridge). An intercept, when enabled, is unpenalized
    and jointly optimized for Huber fits. Scaling is fitted on training X only;
    when enabled, the penalty applies to coefficients in the scaled coordinates.

    fit_intercept=True and standardize=False are our configurable defaults,
    not fully specified empirical choices in the paper. Huber requires delta.
    """

    def __init__(
        self, *, loss="squared", penalty_strength=0.0, rho=0.5,
        huber_delta=None, fit_intercept=True, standardize=False,
        max_iter=5000, tol=1e-8, initial_step=1.0, huber_quantile=None, solver="apg",
    ):
        self.loss = loss
        self.penalty_strength = penalty_strength
        self.rho = rho
        self.huber_delta = huber_delta
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.initial_step = initial_step
        self.huber_quantile = huber_quantile
        self.solver = solver

    def fit(self, X, y):
        clear_fit(self)
        X, y, names = validate_model_inputs(X, y)
        if self.loss not in {"squared", "huber"}:
            raise ValueError("loss must be 'squared' or 'huber'.")
        validate_penalty(self.penalty_strength, self.rho)
        if self.solver not in {"apg", "svd_apg"}:
            raise ValueError("solver must be 'apg' or 'svd_apg'.")
        if self.solver == "svd_apg" and (self.loss != "huber" or self.penalty_strength != 0):
            raise ValueError("svd_apg is only for unpenalized Huber regression.")
        if self.loss == "huber":
            if self.huber_quantile is None:
                validate_huber_delta(self.huber_delta)
            elif (self.huber_delta is not None or isinstance(self.huber_quantile, bool) or
                  not np.isfinite(self.huber_quantile) or not 0 < self.huber_quantile < 1):
                raise ValueError("Supply either huber_delta or a huber_quantile strictly between 0 and 1.")
        elif self.huber_delta is not None or self.huber_quantile is not None:
            raise ValueError("huber_delta is only used with loss='huber'.")
        positive_integer(self.max_iter, "max_iter")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite.")
        if not np.isfinite(self.initial_step) or self.initial_step <= 0:
            raise ValueError("initial_step must be positive and finite.")
        Z, offset, scale = prepare_features(
            X, fit_intercept=self.fit_intercept, standardize=self.standardize,
        )
        p = Z.shape[1]
        delta = self.huber_delta
        if self.solver == "svd_apg":
            # Preconditioning changes coordinates, not the unpenalized Huber objective.
            # Drop only numerically null directions using the usual least-squares rank rule.
            left, singular, right = svd(Z, full_matrices=False, check_finite=False)
            rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(Z.shape) * singular[0]))
            root_n = np.sqrt(len(Z))
            design = left[:, :rank] * root_n
            if self.fit_intercept:
                design = np.column_stack((design, np.ones(len(Z))))
            initial = design.T @ y / len(Z)
            ols_residuals = design @ initial - y
            del left
        elif self.loss == "huber" and self.huber_quantile is not None:
            target_offset = float(y.mean()) if self.fit_intercept else 0.0
            ols_coef = np.linalg.lstsq(Z, y - target_offset, rcond=None)[0]
            ols_residuals = Z @ ols_coef + target_offset - y
        if self.loss == "huber" and self.huber_quantile is not None:
            delta = float(np.quantile(np.abs(ols_residuals), self.huber_quantile, method="linear"))
            validate_huber_delta(delta)  # Do not silently invent a floor for a zero quantile.
        result = None
        if self.loss == "squared" and self.penalty_strength == 0:
            target_offset = float(y.mean()) if self.fit_intercept else 0.0
            coefficients, _, rank, singular_values = np.linalg.lstsq(Z, y - target_offset, rcond=None)
            intercept = target_offset
            self.rank_ = int(rank)
            self.singular_values_ = singular_values
            objective = squared_loss(Z @ coefficients + intercept - y)
        elif self.solver == "svd_apg":
            def whitened_smooth(parameters):
                residuals = design @ parameters - y
                return huber_residual_loss(residuals, delta), design.T @ huber_loss_gradient(residuals, delta)

            # Tighten the coordinate-space tolerance to control the original gradient.
            gradient_bound = max(1., singular[0] / root_n * np.sqrt(p))
            if initial.size:
                whitened = accelerated_proximal_gradient(
                    whitened_smooth, lambda values, step: values, lambda values: 0., initial,
                    initial_step=self.initial_step, max_iter=self.max_iter, tol=self.tol / gradient_bound,
                )
            else:
                value = huber_residual_loss(-y, delta)
                whitened = OptimizationResult(initial, True, 0, value, 0., self.initial_step, np.array([value]))
            coefficients = right[:rank].T @ (whitened.coefficients[:rank] * root_n / singular[:rank])
            intercept = float(whitened.coefficients[-1]) if self.fit_intercept else 0.0
            residuals = Z @ coefficients + intercept - y
            derivative = huber_loss_gradient(residuals, delta)
            gradient = Z.T @ derivative
            parameters = coefficients.copy()
            if self.fit_intercept:
                gradient = np.append(gradient, derivative.sum())
                parameters = np.append(parameters, intercept)
            stationarity = float(np.linalg.norm(gradient, ord=np.inf))
            objective = huber_residual_loss(residuals, delta)
            converged = whitened.converged and stationarity <= self.tol * (1 + np.linalg.norm(parameters, ord=np.inf))
            result = OptimizationResult(parameters, bool(converged), whitened.n_iter, objective,
                                        stationarity, whitened.step_size, whitened.objective_history)
            self.rank_ = rank
            self.singular_values_ = singular[:rank].copy()
        else:
            def smooth(parameters):
                residuals = Z @ parameters[:p] - y
                if self.fit_intercept:
                    residuals = residuals + parameters[-1]
                if self.loss == "squared":
                    value = squared_loss(residuals)
                    derivative = squared_loss_gradient(residuals)
                else:
                    value = huber_residual_loss(residuals, delta)
                    derivative = huber_loss_gradient(residuals, delta)
                gradient = Z.T @ derivative
                if self.fit_intercept:
                    gradient = np.append(gradient, derivative.sum())
                return value, gradient

            def penalty(parameters):
                return elastic_net_penalty(parameters[:p], self.penalty_strength, self.rho)

            def prox(parameters, step):
                updated = parameters.copy()
                updated[:p] = elastic_net_prox(parameters[:p], step, self.penalty_strength, self.rho)
                return updated

            result = accelerated_proximal_gradient(
                smooth, prox, penalty, np.zeros(p + int(self.fit_intercept)),
                initial_step=self.initial_step, max_iter=self.max_iter, tol=self.tol,
            )
            coefficients = result.coefficients[:p]
            intercept = float(result.coefficients[-1]) if self.fit_intercept else 0.0
            objective = result.objective
        if result is not None and not result.converged:
            warnings.warn(
                f"Linear fit did not converge in {result.n_iter} iterations; "
                f"stationarity={result.stationarity:.3g}. Check scaling/tolerance/max_iter.",
                RuntimeWarning, stacklevel=2,
            )

        self.n_features_in_ = p
        self.feature_names_in_ = names
        self.feature_offset_ = offset
        self.feature_scale_ = scale
        self.coef_ = coefficients / scale
        self.intercept_ = float(intercept - offset @ self.coef_)
        self.objective_ = objective
        self.optimization_result_ = result
        self.n_iter_ = 0 if result is None else result.n_iter
        self.converged_ = result is None or result.converged
        if self.loss == "huber":
            self.huber_delta_ = float(delta)
        return self

    def predict(self, X):
        X = prediction_data(self, X)
        return X @ self.coef_ + self.intercept_
