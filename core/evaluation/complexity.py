"""Small numerical diagnostics for Figure 3; caller supplies selected fitted state."""

import numpy as np


def nonzero_count(coefficients, *, tolerance=0.0, groups=None):
    """ENet feature count, or GLM active-group count; exclude the intercept.

    coefficients must be a finite vector. groups optionally maps names to lists
    of coefficient positions (e.g. each predictor's spline basis). A group is
    active if any coefficient exceeds tolerance in absolute value. Coefficient
    units and the tolerance are caller choices; the default counts exact zeros.
    """
    values = np.asarray(coefficients)
    if values.ndim != 1 or not values.size or np.iscomplexobj(values):
        raise ValueError("coefficients must be a nonempty real vector.")
    values = values.astype(float)
    if not np.isfinite(values).all() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Require finite coefficients and a nonnegative finite tolerance.")
    active = np.abs(values) > tolerance
    if groups is None:
        return int(active.sum())
    if not groups:
        raise ValueError("groups must be nonempty.")
    count = 0
    for positions in groups.values():
        indices = np.asarray(positions)
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or (indices < 0).any() or (indices >= len(values)).any()
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Each group must contain distinct valid coefficient positions.")
        count += int(active[indices].any())
    return count


def selected_component_count(model):
    """PCR/PLS realized component count (n_components_), not the requested cap."""
    count = getattr(model, "n_components_", None)
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("Require a fitted model with nonnegative integer n_components_.")
    return int(count)


def average_tree_depth(depths):
    """RF mean realized tree depth; supply one depth per tree, root-only = 0."""
    values = np.asarray(depths)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()):
        raise ValueError("depths must be a nonempty vector of nonnegative integers.")
    return float(values.mean())


def distinct_split_count(split_features, *, feature_groups=None):
    """GBRT count of distinct split inputs across all trees.

    Supply a flattened iterable of actual split feature names/IDs, excluding
    leaf sentinels. Optional feature_groups maps each used input to its original
    characteristic. No feature-to-characteristic mapping is inferred.
    """
    used = set(split_features)
    if feature_groups is None:
        return len(used)
    if not used <= feature_groups.keys():
        raise ValueError("Every used split feature must have a group mapping.")
    return len({feature_groups[feature] for feature in used})
