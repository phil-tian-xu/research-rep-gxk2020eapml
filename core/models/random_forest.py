"""Explicit GKX RF pilot conventions around scikit-learn's CART forest."""

import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def cpu_budget():
    """Slurm allocation takes precedence; the local launcher exports OMP threads."""
    value = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))
    if value < 1:
        raise ValueError("RF CPU budget must be positive.")
    return value


def make_forest(*, n_estimators, max_depth, max_features, random_state, n_jobs):
    return RandomForestRegressor(
        n_estimators=n_estimators, criterion="squared_error", max_depth=max_depth,
        max_features=max_features, bootstrap=True, max_samples=None,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
        min_impurity_decrease=0., ccp_alpha=0., oob_score=False, warm_start=False,
        random_state=random_state, n_jobs=n_jobs)


def diagnostics(model):
    depths = [int(tree.tree_.max_depth) for tree in model.estimators_]
    return {"average_tree_depth": float(np.mean(depths)), "tree_depths": depths,
            "tree_leaf_counts": [int(tree.get_n_leaves()) for tree in model.estimators_]}
