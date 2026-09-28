"""Zero-out importance from a stream of (X, y) arrays; no model fitting.

Kept outside evaluation/ so adding reporting does not change fit-resume hashes.
"""

import numpy as np
import pandas as pd


def characteristic_groups(feature_names):
    """Proposed convention: base + macro interactions; all SIC dummies jointly.

    This is an explicit replication choice, not a verified author-code convention.
    Return column indices in the predictor's saved feature order.
    """
    names = list(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Require nonempty, unique feature names.")
    groups = {}
    for index, name in enumerate(names):
        base = name.split("__macro_", 1)[0]
        if "__macro_" in name and base not in names:
            raise ValueError(f"Interaction lacks its base characteristic: {name}")
        group = "sic2" if name.startswith("sic2_") else base
        groups.setdefault(group, []).append(index)
    return groups


def streaming_zero_out(predict, batches, *, n_features, groups, coefficients=None):
    """Return signed importance and pooled diagnostics for ONE training window.

    `predict(X)` must be row-independent and operate in the supplied coordinates.
    `groups` maps names to nonempty integer column-index lists (overlap allowed).
    Optional coefficients enable the exact linear shortcut; callers must ensure
    predict(X) == intercept + X @ coefficients. Nonlinear models omit coefficients.
    Aggregate SSE differences before dividing; never average batch R2 values.
    """
    if not callable(predict) or not groups or n_features < 1:
        raise ValueError("Require a predictor, groups and positive feature count.")
    selected = {}
    for name, columns in groups.items():
        indices = np.asarray(columns)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                or len(np.unique(indices)) != len(indices)
                or (indices < 0).any() or (indices >= n_features).any()):
            raise ValueError(f"Invalid columns for {name}.")
        selected[name] = indices
    beta = None if coefficients is None else np.asarray(coefficients, dtype=float)
    if beta is not None and (beta.shape != (n_features,) or not np.isfinite(beta).all()):
        raise ValueError("Invalid linear coefficients.")
    numerator = np.zeros(len(selected))
    denominator, baseline_sse, count = 0.0, 0.0, 0
    for X, y in batches:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if (X.ndim != 2 or X.shape[1] != n_features or y.shape != (len(X),)
                or not np.isfinite(X).all() or not np.isfinite(y).all()):
            raise ValueError("Invalid training batch.")
        fitted = np.asarray(predict(X), dtype=float)
        if fitted.shape != y.shape or not np.isfinite(fitted).all():
            raise ValueError("Invalid baseline predictions.")
        residual = y - fitted
        count += len(y)
        denominator += float(y @ y)
        baseline_sse += float(residual @ residual)
        for j, indices in enumerate(selected.values()):
            if beta is not None:
                active = indices[beta[indices] != 0]
                if not len(active):
                    continue
                contribution = X[:, active] @ beta[active]
            else:
                altered = X.copy()
                altered[:, indices] = 0
                muted = np.asarray(predict(altered), dtype=float)
                if muted.shape != y.shape or not np.isfinite(muted).all():
                    raise ValueError("Invalid zero-out predictions.")
                contribution = fitted - muted
            numerator[j] += float(contribution @ contribution + 2 * residual @ contribution)
    if not count or denominator <= 0 or not np.isfinite([denominator, baseline_sse, *numerator]).all():
        raise ValueError("Empty sample, zero benchmark, or nonfinite importance totals.")
    result = pd.DataFrame({"predictor": list(selected), "sse_increase": numerator,
                           "importance": numerator / denominator})
    return result, dict(n_train=count, benchmark_sse=denominator, baseline_sse=baseline_sse,
                        training_r2=1 - baseline_sse / denominator)
