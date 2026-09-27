"""Quadratic spline basis and group-lasso solver for the empirical GLM pilot."""

import numpy as np

from core.losses import validate_huber_delta, validate_penalty
from core.optimizers import accelerated_proximal_gradient


class SplineBasis:
    """Training-only knots, means and sample standard deviations.

    Binary indices are explicit: do not infer dummies from observed unique values.
    Each continuous input has five terms; each supplied binary input has one.
    """
    def fit_transform(self, X, binary_indices=()):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError("Spline training data must be a finite matrix with at least two rows.")
        self.n_features_ = X.shape[1]
        self.binary_indices_ = set(binary_indices)
        if any(i < 0 or i >= X.shape[1] for i in self.binary_indices_):
            raise ValueError("Invalid binary feature index.")
        self.raw_scale_ = X.std(axis=0, ddof=1)
        self.raw_scale_[self.raw_scale_ == 0] = 1.
        self.knots_ = np.zeros((4, X.shape[1]))
        self.group_sizes_ = [1 if i in self.binary_indices_ else 5 for i in range(X.shape[1])]
        for i in range(X.shape[1]):
            if i not in self.binary_indices_:
                self.knots_[:, i] = np.quantile(X[:, i] / self.raw_scale_[i], [0, .25, .5, .75], method="hazen")
        Z = self._expand(X)
        self.mean_ = Z.mean(axis=0)
        Z -= self.mean_
        # NumPy std on the entire expanded matrix creates another full matrix.
        # Compute the identical column statistics one input group at a time.
        self.scale_ = np.empty(Z.shape[1])
        start = 0
        for size in self.group_sizes_:
            self.scale_[start:start + size] = Z[:, start:start + size].std(axis=0, ddof=1)
            start += size
        self.scale_[self.scale_ == 0] = 1.
        Z /= self.scale_
        return Z

    def _expand(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("Prediction inputs must match the fitted finite feature matrix.")
        Z = np.empty((len(X), sum(self.group_sizes_)), dtype=np.float64)
        start = 0
        for i, size in enumerate(self.group_sizes_):
            x = X[:, i] / self.raw_scale_[i]
            if size == 1:
                if not np.isin(X[:, i], [0., 1.]).all():
                    raise ValueError(f"Industry indicator {i} is not binary.")
                Z[:, start] = x
            else:
                knots = self.knots_[:, i]
                Z[:, start] = x
                Z[:, start + 1] = (x - knots[0])**2
                for j in range(1, 4):
                    Z[:, start + 1 + j] = np.maximum(x - knots[j], 0.)**2
            start += size
        return Z

    def transform(self, X):
        Z = self._expand(X)
        Z -= self.mean_
        Z /= self.scale_
        return Z

    def state(self):
        return {"raw_scale": self.raw_scale_.tolist(), "knots": self.knots_.tolist(),
                "basis_mean": self.mean_.tolist(), "basis_scale": self.scale_.tolist(),
                "group_sizes": self.group_sizes_, "binary_indices": sorted(self.binary_indices_)}


def fit_group_loss(Z, target, groups, strength, *, delta=None, initial=None,
                   max_iter=50000, tol=1e-8, initial_step=1., gram_stats=None):
    """Mean half-squared/half-Huber loss plus unweighted L2 group penalty.

    The target is already centered at the fixed training mean. Squared fits
    may share exact Gram statistics across candidates; Huber uses full arrays.
    """
    validate_penalty(strength, 0.)
    if delta is not None:
        validate_huber_delta(delta)
    groups = np.asarray(groups)
    if groups.dtype.kind not in "iu" or groups.ndim != 1 or (groups <= 0).any() or groups.sum() != Z.shape[1]:
        raise ValueError("Group sizes must partition the design matrix.")
    starts = np.r_[0, np.cumsum(groups)[:-1]]

    def norms(v):
        return np.sqrt(np.add.reduceat(v*v, starts))

    def prox(v, step):
        lengths = norms(v)
        multiplier = np.zeros(len(groups))
        active = lengths > step * strength
        multiplier[active] = 1 - step * strength / lengths[active]
        return v * np.repeat(multiplier, groups)

    if delta is None:
        gram, cross, constant = (gram_stats if gram_stats is not None else
                                (Z.T @ Z / len(Z), Z.T @ target / len(Z), .5 * np.mean(target**2)))

        def smooth(v):
            product = gram @ v
            return .5 * v @ product - cross @ v + constant, product - cross
    else:
        def smooth(v):
            residual = Z @ v - target
            absolute = abs(residual)
            loss = np.where(absolute <= delta, .5 * residual**2, delta * absolute - .5 * delta**2).mean()
            return loss, Z.T @ np.clip(residual, -delta, delta) / len(Z)

    return accelerated_proximal_gradient(smooth, prox, lambda v: strength * norms(v).sum(),
        np.zeros(Z.shape[1]) if initial is None else initial, max_iter=max_iter,
        tol=tol, initial_step=initial_step)
