"""Float64 batch fits for the existing PCR, SIMPLS, and unpenalized Huber models.

Blocked QR preserves the training least-squares geometry without forming X'X
or retaining n-by-p scores. Huber still evaluates its nonlinear loss on every
training observation; the QR summary is only its OLS initialization/preconditioner.
"""

from dataclasses import dataclass
from time import monotonic

import numpy as np
from scipy.linalg import qr, svd

from core.losses import huber_residual_loss, huber_loss_gradient, validate_huber_delta
from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.optimizers import OptimizationResult, accelerated_proximal_gradient


@dataclass
class TrainingReduction:
    n_rows: int
    offset: np.ndarray
    scale: np.ndarray
    target_offset: float
    R: np.ndarray
    left: np.ndarray
    singular: np.ndarray
    right: np.ndarray
    rank: int
    fit_intercept: bool
    standardize: bool


def reduce_training(window, *, fit_intercept=True, standardize=False):
    """Two passes: training-only moments, then QR of [transformed X, centered y]."""
    if not isinstance(fit_intercept, bool) or not isinstance(standardize, bool):
        raise ValueError("fit_intercept and standardize must be boolean.")
    p = len(window.panel.features)
    count, mean_y = 0, 0.
    last_print = monotonic()
    mean, m2 = np.zeros(p), np.zeros(p)
    low, high = np.full(p, np.inf), np.full(p, -np.inf)
    low_y, high_y = np.inf, -np.inf
    print(f"Training moments: {window.n_rows:,} rows, {p} features.", flush=True)
    for batch in window.batches():
        X, y = batch.X, batch.y
        n = len(y)
        batch_mean = X.mean(axis=0)
        difference = batch_mean - mean
        if standardize:
            centered = X - batch_mean
            m2 += np.einsum("ij,ij->j", centered, centered) + difference**2 * count * n / (count + n)
            del centered
        mean += difference * n / (count + n)
        mean_y += (float(y.mean()) - mean_y) * n / (count + n)
        low, high = np.minimum(low, X.min(axis=0)), np.maximum(high, X.max(axis=0))
        low_y, high_y = min(low_y, y.min()), max(high_y, y.max())
        count += n
        if monotonic() - last_print > 30:
            print(f"Training moments: {count:,}/{window.n_rows:,} rows.", flush=True)
            last_print = monotonic()
    constant = low == high
    mean[constant] = low[constant]
    offset = mean if fit_intercept else np.zeros(p)
    scale = np.sqrt(m2 / count) if standardize else np.ones(p)
    scale[constant | (scale == 0)] = 1.
    target_offset = (float(low_y) if low_y == high_y else mean_y) if fit_intercept else 0.
    R = np.empty((0, p + 1))
    seen, last_print = 0, monotonic()
    print("Reducing training batches with QR...", flush=True)
    for batch in window.batches():
        block = np.empty((len(R) + len(batch.y), p + 1), order="F")
        block[:len(R)] = R
        block[len(R):, :p] = (batch.X - offset) / scale
        block[len(R):, p] = batch.y - target_offset
        if not np.isfinite(block).all():
            raise ValueError("Training transform overflowed.")
        # raw mode returns the small R without constructing Q. Copy releases
        # LAPACK's tall work array before reading the next batch.
        raw_qr, triangular = qr(block, mode="raw", overwrite_a=True, check_finite=False)
        R = triangular.copy()
        del raw_qr, triangular, block
        seen += len(batch.y)
        if monotonic() - last_print > 30:
            print(f"Training QR: {seen:,}/{count:,} rows.", flush=True)
            last_print = monotonic()
    left, singular, right = svd(R[:, :p], full_matrices=False, check_finite=False)
    cutoff = np.finfo(float).eps * max(count, p) * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    print(f"Training reduction complete: numerical rank={rank}.", flush=True)
    return TrainingReduction(count, offset, scale, target_offset, R, left, singular,
                             right, rank, fit_intercept, standardize)


def fit_components(name, parameters, reduction):
    """Fit one component candidate on the shared, lossless QR reduction."""
    r, k = reduction, parameters["n_components"]
    p = len(r.offset)
    if (isinstance(k, bool) or not isinstance(k, int) or
            not 0 <= k <= min(p, r.n_rows - int(r.fit_intercept))):
        raise ValueError("n_components is outside the possible training rank.")
    if name == "PCR":
        if k > r.rank:
            raise ValueError(f"n_components={k} exceeds numerical rank {r.rank}.")
        model = PCRRegressor(**parameters)
        model.components_ = r.right[:k].copy()
        model.component_coef_ = (r.left[:, :k].T @ r.R[:, p]) / r.singular[:k]
        model.singular_values_ = r.singular[:k].copy()
        model.rank_ = r.rank
        model.n_components_ = k
        coefficients = model.components_.T @ model.component_coef_
    elif name == "PLS":
        # Orthogonal QR is an isometry: SIMPLS covariances, score inner products,
        # and final least squares are preserved on these p+1 rows.
        model = PLSRegressor(k, fit_intercept=False, standardize=False, tol=parameters["tol"])
        model.fit(r.R[:, :p], r.R[:, p])
        coefficients = model.coef_.copy()
        model.fit_intercept, model.standardize = r.fit_intercept, r.standardize
    else:
        raise ValueError(f"Unsupported component model: {name}")
    model.n_features_in_, model.feature_names_in_ = p, None
    model.feature_offset_, model.feature_scale_ = r.offset, r.scale
    model.target_offset_ = r.target_offset
    model.coef_ = coefficients / r.scale
    model.intercept_ = float(r.target_offset - r.offset @ model.coef_)
    return model


def fit_huber(parameters, reduction, window):
    """Same svd_apg objective/threshold as LinearRegressor, using streamed passes."""
    model = LinearRegressor(**parameters)
    if model.solver != "svd_apg" or model.loss != "huber" or model.penalty_strength != 0:
        raise ValueError("Batch OLS_HUBER requires unpenalized Huber with solver='svd_apg'.")
    r, p, n = reduction, len(reduction.offset), reduction.n_rows
    rank, root_n = r.rank, np.sqrt(n)
    transform = r.right[:rank].T * (root_n / r.singular[:rank])
    initial = r.left[:, :rank].T @ r.R[:, p] / root_n
    if r.fit_intercept:
        initial = np.append(initial, r.target_offset)
    delta = model.huber_delta
    if model.huber_quantile is not None:
        if delta is not None or not 0 < model.huber_quantile < 1:
            raise ValueError("Supply only a Huber quantile strictly between 0 and 1.")
        # One scalar per training row (8*n bytes), not an n-by-p matrix.
        residuals, start = np.empty(n), 0
        beta = transform @ initial[:rank]
        intercept = initial[-1] if r.fit_intercept else 0.
        for batch in window.batches():
            end = start + len(batch.y)
            residuals[start:end] = np.abs(((batch.X - r.offset) / r.scale) @ beta + intercept - batch.y)
            start = end
        delta = float(np.quantile(residuals, model.huber_quantile, method="linear", overwrite_input=True))
        del residuals
    validate_huber_delta(delta)
    print(f"Huber threshold={delta:.9g}; fitting with repeated training passes.", flush=True)
    passes, last_print = 0, monotonic()

    def original_smooth(coefficients, intercept):
        nonlocal passes, last_print
        value, gradient, intercept_gradient = 0., np.zeros(p), 0.
        seen = 0
        for batch in window.batches():
            Z = (batch.X - r.offset) / r.scale
            residual = Z @ coefficients + intercept - batch.y
            weight = len(residual) / n
            value += huber_residual_loss(residual, delta) * weight
            derivative = huber_loss_gradient(residual, delta) * weight
            gradient += Z.T @ derivative
            intercept_gradient += derivative.sum()
            seen += len(residual)
            if monotonic() - last_print > 30:
                print(f"Huber objective pass {passes + 1}: {seen:,}/{n:,} rows.", flush=True)
                last_print = monotonic()
        passes += 1
        if monotonic() - last_print > 30:
            print(f"Huber objective pass {passes}: loss={value:.9g}", flush=True)
            last_print = monotonic()
        return value, gradient, intercept_gradient

    def smooth(values):
        value, gradient, intercept_gradient = original_smooth(
            transform @ values[:rank], float(values[-1]) if r.fit_intercept else 0.)
        transformed_gradient = transform.T @ gradient
        if r.fit_intercept:
            transformed_gradient = np.append(transformed_gradient, intercept_gradient)
        return value, transformed_gradient

    gradient_bound = max(1., r.singular[0] / root_n * np.sqrt(p))
    if initial.size:
        whitened = accelerated_proximal_gradient(
            smooth, lambda values, step: values, lambda values: 0., initial,
            initial_step=model.initial_step, max_iter=model.max_iter, tol=model.tol / gradient_bound)
    else:
        value = smooth(initial)[0]
        whitened = OptimizationResult(initial, True, 0, value, 0., model.initial_step, np.array([value]))
    coefficients = transform @ whitened.coefficients[:rank]
    intercept = float(whitened.coefficients[-1]) if r.fit_intercept else 0.
    value, gradient, intercept_gradient = original_smooth(coefficients, intercept)
    params = coefficients.copy()
    if r.fit_intercept:
        gradient, params = np.append(gradient, intercept_gradient), np.append(params, intercept)
    stationarity = float(np.linalg.norm(gradient, ord=np.inf))
    converged = bool(whitened.converged and stationarity <= model.tol * (1 + np.linalg.norm(params, ord=np.inf)))
    model.optimization_result_ = OptimizationResult(params, converged, whitened.n_iter, value,
        stationarity, whitened.step_size, whitened.objective_history)
    model.n_features_in_, model.feature_names_in_ = p, None
    model.feature_offset_, model.feature_scale_ = r.offset, r.scale
    model.coef_, model.intercept_ = coefficients / r.scale, float(intercept - r.offset @ (coefficients / r.scale))
    model.rank_, model.singular_values_ = rank, r.singular[:rank].copy()
    model.objective_, model.n_iter_, model.converged_ = value, whitened.n_iter, converged
    model.huber_delta_ = float(delta)
    return model
