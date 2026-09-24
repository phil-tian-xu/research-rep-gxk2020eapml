"""Tune hyperparameters within each specification, then evaluate every specification."""

from dataclasses import dataclass

import numpy as np

from core.losses import mse_loss
from core.model_inputs import validate_model_inputs
from core.models import PCRRegressor


@dataclass
class TuningResult:
    best_estimator: object
    best_params: dict
    validation_results: list[dict]


def tune_hyperparameters(
    estimator_factory, candidates, X_train, y_train, X_validation, y_validation,
    *, validation_loss_fn=mse_loss, model_name=None, test_year=None,
):
    """Tune one model specification and return its best fitted estimator.

    Call separately for each specification (e.g., ENet+H, PCR, PLS), then report
    test performance for every specification. This function chooses settings
    within that specification; it does not choose a winner across specifications.
    candidates is a caller-supplied iterable of constructor keyword dictionaries.
    validation_loss_fn(y_true, y_pred) must return one finite real scalar, with
    smaller values better. Built-in losses are unpenalized observation means;
    exact ties retain the first candidate. MSE is the default, including for
    Huber-trained models. For Huber scoring, pass partial(huber_loss, delta=...)
    from core.losses with one fixed threshold across candidates.
    Changing validation loss does not change the estimators' training objectives.
    No combined train+validation refit and no implicit grid generation.
    Caller supplies disjoint chronological samples (e.g., using annual_splits).
    Supplying model_name enables flushed candidate progress. Exceptions retain
    their original type/traceback with model, year, candidate, and stage notes.
    """
    _, validation_y, _ = validate_model_inputs(X_validation, y_validation)
    if not callable(validation_loss_fn):
        raise ValueError("validation_loss_fn must be callable as loss_fn(y_true, y_pred).")
    records = []
    best_estimator = None
    best_params = None
    best_score = float("inf")
    candidates = list(candidates)
    # Reuse one training SVD only when candidates differ solely in component count.
    pcr_path = None
    reuse_pcr = (estimator_factory is PCRRegressor and bool(candidates) and
                 all(isinstance(c, dict) and "n_components" in c for c in candidates) and
                 all({k: v for k, v in c.items() if k != "n_components"} ==
                     {k: v for k, v in candidates[0].items() if k != "n_components"} for c in candidates) and
                 all(isinstance(c["n_components"], int) and not isinstance(c["n_components"], bool)
                     and c["n_components"] >= 0 for c in candidates))
    for candidate_index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a dictionary of constructor arguments.")
        parameters = candidate.copy()
        label = model_name or getattr(estimator_factory, "__name__", "estimator")
        context = f"[model={label}] [test_year={test_year}] [candidate={candidate_index}/{len(candidates)}]"
        if model_name is not None:
            print(f"{context} Fitting...", flush=True)
        stage = "construction"
        try:
            estimator = estimator_factory(**parameters)
            stage = "fit"
            if reuse_pcr:
                if pcr_path is None:
                    path_parameters = max(candidates, key=lambda c: c["n_components"])
                    stage = f"training PCR path through K={path_parameters['n_components']}"
                    pcr_path = estimator_factory(**path_parameters).fit(X_train, y_train)
                estimator = pcr_path.with_components(parameters["n_components"])
            else:
                estimator.fit(X_train, y_train)
            if not getattr(estimator, "converged_", True):
                diagnostics = f"iterations={getattr(estimator, 'n_iter_', None)}"
                optimization = getattr(estimator, "optimization_result_", None)
                if optimization is not None:
                    threshold = estimator.tol * (1 + np.linalg.norm(optimization.coefficients, ord=np.inf))
                    diagnostics += (f", stationarity={optimization.stationarity:.6g}"
                                    f", tolerance={estimator.tol:.6g}, stopping_threshold={threshold:.6g}")
                raise RuntimeError(f"Candidate did not converge: {diagnostics}. Resolve before selection.")
            stage = "validation prediction"
            prediction = np.asarray(estimator.predict(X_validation), dtype=np.float64)
            if prediction.shape != validation_y.shape or not np.isfinite(prediction).all():
                raise ValueError("Model predictions must be finite and have shape (n_observations,).")
            stage = "validation scoring"
            score = np.asarray(validation_loss_fn(validation_y, prediction))
            if score.ndim != 0 or score.dtype.kind not in "iuf":
                raise ValueError("validation_loss_fn must return one real scalar, not per-observation losses.")
            score = float(score)
            if not np.isfinite(score):
                raise FloatingPointError("Validation score is not finite.")
        except Exception as error:
            error.add_note(f"{context} stage={stage}; parameters={parameters}")
            raise
        if model_name is not None:
            print(f"{context} Validation loss={score:.6g}", flush=True)
        record = {"params": parameters, "validation_loss": score,
                  "n_components": getattr(estimator, "n_components_", None)}
        if hasattr(estimator, "huber_delta_"):
            record["huber_delta"] = estimator.huber_delta_
            record["n_iter"] = estimator.n_iter_
            record["stationarity"] = estimator.optimization_result_.stationarity
            record["stopping_threshold"] = estimator.tol * (1 + np.linalg.norm(estimator.optimization_result_.coefficients, ord=np.inf))
        records.append(record)
        if score < best_score:
            best_estimator, best_params, best_score = estimator, parameters.copy(), score
    if best_estimator is None:
        raise ValueError("Provide at least one hyperparameter candidate.")
    return TuningResult(best_estimator, best_params, records)
