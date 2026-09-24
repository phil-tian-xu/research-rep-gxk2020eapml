"""Accelerated proximal gradient with backtracking and explicit convergence checks.

This is a FISTA variant, not a literal transcription of appendix Algorithm 1.
Numerical choices are documented in docs/working_notes/Model Implementation.md.
"""

from dataclasses import dataclass

import numpy as np

from core.losses import validate_penalty


@dataclass
class OptimizationResult:
    coefficients: np.ndarray
    converged: bool
    n_iter: int
    objective: float
    stationarity: float
    step_size: float
    objective_history: np.ndarray


def elastic_net_prox(values, step_size, penalty_strength, rho):
    """Exact prox for the paper's elastic-net penalty, including pure lasso/ridge."""
    validate_penalty(penalty_strength, rho)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be positive and finite.")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector.")
    threshold = step_size * penalty_strength * (1.0 - rho)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0) / (
        1.0 + step_size * penalty_strength * rho
    )


def accelerated_proximal_gradient(
    value_and_gradient, prox, penalty, initial,
    *, initial_step=1.0, max_iter=5000, tol=1e-8, max_backtracks=60,
):
    """Minimize smooth(x) + penalty(x), given prox(x, step) for the penalty.

The stopping statistic is the infinity norm of the proximal gradient mapping
at the returned iterate, compared with tol * (1 + ||x||_inf). Small parameter
updates alone do not establish convergence when the step size is small.
"""
    for name, value in (("max_iter", max_iter), ("max_backtracks", max_backtracks)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite.")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be positive and finite.")
    x = np.array(initial, dtype=np.float64, copy=True)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("initial must be a finite, nonempty vector.")
    extrapolated = x.copy()
    momentum = 1.0
    step = float(initial_step)
    initial_value, _ = value_and_gradient(x)
    history = [float(initial_value + penalty(x))]
    if not np.isfinite(history[0]):
        raise FloatingPointError("Initial objective is not finite.")
    converged = False

    for iteration in range(1, max_iter + 1):
        value, gradient = value_and_gradient(extrapolated)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite smooth objective or gradient.")
        for _ in range(max_backtracks):
            candidate = prox(extrapolated - step * gradient, step)
            candidate_value, candidate_gradient = value_and_gradient(candidate)
            difference = candidate - extrapolated
            upper_bound = value + gradient @ difference + (difference @ difference) / (2.0 * step)
            slack = 16 * np.finfo(float).eps * max(1.0, abs(value), abs(candidate_value))
            if np.isfinite(candidate_value) and candidate_value <= upper_bound + slack:
                break
            step *= 0.5
        else:
            raise RuntimeError("Proximal-gradient backtracking failed to find a descent step.")

        objective = float(candidate_value + penalty(candidate))
        if not np.isfinite(objective) or not np.isfinite(candidate_gradient).all():
            raise FloatingPointError("Nonfinite objective or gradient at accepted iterate.")
        mapping = (candidate - prox(candidate - step * candidate_gradient, step)) / step
        stationarity = float(np.linalg.norm(mapping, ord=np.inf))
        history.append(objective)
        if stationarity <= tol * (1.0 + np.linalg.norm(candidate, ord=np.inf)):
            x = candidate
            converged = True
            break

        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        if objective > history[-2]:
            # Restart acceleration after an objective increase (a numerical choice).
            extrapolated = candidate.copy()
            next_momentum = 1.0
        else:
            extrapolated = candidate + ((momentum - 1.0) / next_momentum) * (candidate - x)
        x = candidate
        momentum = next_momentum

    return OptimizationResult(x, converged, iteration, objective, stationarity, step, np.asarray(history))
