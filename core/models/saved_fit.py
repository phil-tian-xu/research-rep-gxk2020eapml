"""Portable selected-fit state for post-estimation diagnostics (no refitting)."""

import numpy as np


def linear_state(model, feature_names):
    """Slopes/intercept act on original panel coordinates, including scaled PLS."""
    names = list(feature_names)
    if len(names) != len(model.coef_) or len(set(names)) != len(names):
        raise ValueError("Saved feature names must match the fitted coefficient order.")
    state = dict(fitted_state_version=1, prediction_coordinates="original_panel",
                 feature_names=names, coefficients=model.coef_.tolist(),
                 intercept=float(model.intercept_))
    # Preserve transforms/directions for auditing the selected PCR/PLS fit.
    for attribute in ("feature_offset_", "feature_scale_", "target_offset_",
                      "components_", "component_coef_", "n_components_", "stop_reason_"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            state[attribute.rstrip("_")] = value.tolist() if isinstance(value, np.ndarray) else value
    return state


def forest_state(model):
    """Store numeric tree arrays, avoiding executable pickle and private loaders."""
    return dict(version=1, n_features=int(model.n_features_in_), input_dtype="float32",
                trees=[dict(children_left=t.tree_.children_left.tolist(),
                            children_right=t.tree_.children_right.tolist(),
                            feature=t.tree_.feature.tolist(), threshold=t.tree_.threshold.tolist(),
                            value=t.tree_.value[:, 0, 0].tolist()) for t in model.estimators_])


def restore_forest(state):
    if state.get("version") != 1 or state.get("input_dtype") != "float32":
        raise ValueError("Unsupported saved forest format.")
    width = state["n_features"]
    trees = []
    for tree in state["trees"]:
        left, right, feature = (np.asarray(tree[k], dtype=np.int64)
                                for k in ("children_left", "children_right", "feature"))
        threshold, value = (np.asarray(tree[k], dtype=float) for k in ("threshold", "value"))
        size = len(left)
        if (not size or any(a.shape != (size,) for a in (right, feature, threshold, value))
                or not np.isfinite(threshold).all() or not np.isfinite(value).all()):
            raise ValueError("Invalid saved tree arrays.")
        leaf = left == -1
        internal = np.flatnonzero(~leaf)
        # sklearn numbers children after parents; this also excludes cycles.
        if (np.any(right[leaf] != -1) or np.any(left[internal] <= internal)
                or np.any(right[internal] <= internal) or np.any(left[internal] >= size)
                or np.any(right[internal] >= size) or np.any(feature[internal] < 0)
                or np.any(feature[internal] >= width)):
            raise ValueError("Invalid saved tree topology.")
        trees.append((left, right, feature, threshold, value))
    if not trees:
        raise ValueError("Saved forest is empty.")

    def predict(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != width or not np.isfinite(X).all():
            raise ValueError("Forest inputs must be finite and match saved features.")
        prediction = np.zeros(len(X))
        for left, right, feature, threshold, value in trees:
            nodes = np.zeros(len(X), dtype=np.int64)
            active = np.flatnonzero(left[nodes] != -1)
            while len(active):
                node = nodes[active]
                # Compare float32 inputs with float64 thresholds, as sklearn does.
                go_left = X[active, feature[node]].astype(float) <= threshold[node]
                nodes[active] = np.where(go_left, left[node], right[node])
                active = active[left[nodes[active]] != -1]
            prediction += value[nodes]
        return prediction / len(trees)

    return predict


def restore_spline(state, n_features):
    from core.models.glm import SplineBasis

    basis = SplineBasis()
    basis.n_features_ = n_features
    basis.binary_indices_ = set(state["binary_indices"])
    basis.group_sizes_ = list(state["group_sizes"])
    expected = [1 if i in basis.binary_indices_ else 5 for i in range(n_features)]
    if basis.group_sizes_ != expected or not basis.binary_indices_ <= set(range(n_features)):
        raise ValueError("Invalid saved spline groups.")
    for key, attribute, shape in (("raw_scale", "raw_scale_", (n_features,)),
            ("knots", "knots_", (4, n_features)),
            ("basis_mean", "mean_", (sum(expected),)),
            ("basis_scale", "scale_", (sum(expected),))):
        value = np.asarray(state[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid saved spline transform.")
        if "scale" in key and (value <= 0).any():
            raise ValueError("Spline scales must be positive.")
        setattr(basis, attribute, value)
    return basis
