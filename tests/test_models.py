"""Numerical reference and input-contract tests; run with standard-library unittest."""

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize

from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.losses import (
    elastic_net_penalty, huber_residual_loss as huber_loss, huber_loss_gradient,
    squared_loss, squared_loss_gradient,
)
from core.optimizers import accelerated_proximal_gradient, elastic_net_prox


class LossAndOptimizerTests(unittest.TestCase):
    def test_paper_loss_normalization_and_huber_boundary(self):
        residuals = np.array([-3., -1., 0., 1., 3.])
        self.assertAlmostEqual(squared_loss(residuals), 4.0)
        self.assertAlmostEqual(huber_loss(residuals, 1.0), 12.0 / 5.0)
        assert_allclose(huber_loss_gradient(residuals, 1.0), [-.4, -.4, 0, .4, .4])
        self.assertAlmostEqual(elastic_net_penalty(np.array([2., -3.]), .4, .5), 2.3)

    def test_loss_gradients_against_finite_differences(self):
        residuals = np.array([-.8, -.1, .2, 1.2])
        for loss, gradient in (
            (squared_loss, squared_loss_gradient),
            (lambda r: huber_loss(r, .5), lambda r: huber_loss_gradient(r, .5)),
        ):
            derivative = []
            for j in range(len(residuals)):
                direction = np.eye(len(residuals))[j] * 1e-6
                derivative.append((loss(residuals + direction) - loss(residuals - direction)) / 2e-6)
            assert_allclose(gradient(residuals), derivative, rtol=1e-8, atol=1e-10)

    def test_prox_lasso_and_ridge_endpoints(self):
        values = np.array([-2., .1, 3.])
        assert_allclose(elastic_net_prox(values, .5, .4, 0), [-1.8, 0, 2.8])
        assert_allclose(elastic_net_prox(values, .5, .4, 1), values / 1.2)
        assert_allclose(elastic_net_prox(values, .5, 0, .5), values)

    def test_backtracking_on_badly_scaled_quadratic(self):
        curvature = np.array([2., 2000.])
        target = np.array([.5, -1.])

        def objective(x):
            residual = x - target
            return .5 * np.sum(curvature * residual**2), curvature * residual

        result = accelerated_proximal_gradient(
            objective, lambda x, step: x, lambda x: 0., np.zeros(2), tol=1e-9,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.step_size, 1.)
        assert_allclose(result.coefficients, target, atol=1e-8)

    def test_tiny_step_does_not_fake_convergence(self):
        result = accelerated_proximal_gradient(
            lambda x: (.5 * np.sum((x - 1)**2), x - 1),
            lambda x, step: x, lambda x: 0., np.zeros(2), initial_step=1e-10, max_iter=2,
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.stationarity, .9)

    def test_invalid_loss_inputs(self):
        for residuals in ([], [[1.]], [np.nan], [1j]):
            with self.subTest(residuals=residuals), self.assertRaises(ValueError):
                squared_loss(residuals)
        for delta in (None, 0, -1, np.inf):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                huber_loss([1.], delta)


class LinearTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(314)
        self.X = rng.normal(size=(150, 5)) + np.arange(5)
        self.y = .2 + self.X @ np.array([.3, 0, -.2, .1, 0]) + rng.normal(0, .03, 150)

    def test_ols_matches_augmented_lstsq_with_collinearity(self):
        X = np.column_stack((self.X, self.X[:, 0], np.ones(len(self.X))))
        expected = np.linalg.lstsq(np.column_stack((X, np.ones(len(X)))), self.y, rcond=None)[0]
        model = LinearRegressor().fit(X, self.y)
        assert_allclose(model.predict(X), np.column_stack((X, np.ones(len(X)))) @ expected, atol=1e-12)

    def test_no_intercept_matches_lstsq(self):
        expected = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        model = LinearRegressor(fit_intercept=False).fit(self.X, self.y)
        assert_allclose(model.coef_, expected, atol=1e-12)
        self.assertEqual(model.intercept_, 0.)

    def test_ridge_matches_closed_form_with_paper_factor_two(self):
        penalty = .4
        centered = self.X - self.X.mean(axis=0)
        expected = np.linalg.solve(
            centered.T @ centered / len(centered) + penalty / 2 * np.eye(centered.shape[1]),
            centered.T @ (self.y - self.y.mean()) / len(centered),
        )
        model = LinearRegressor(penalty_strength=penalty, rho=1, tol=1e-11).fit(self.X, self.y)
        self.assertTrue(model.converged_)
        assert_allclose(model.coef_, expected, atol=1e-9)
        self.assertAlmostEqual(model.intercept_, self.y.mean() - self.X.mean(axis=0) @ expected, places=9)

    def test_lasso_and_enet_against_orthogonal_closed_form(self):
        q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(70, 5)))
        X = q * np.sqrt(len(q))
        beta = np.array([.6, -.3, .01, 0, 1.2])
        for rho in (0., .5, 1.):
            expected = np.sign(beta) * np.maximum(np.abs(beta) - .2 * (1 - rho) / 2, 0) / (1 + .2 * rho / 2)
            model = LinearRegressor(fit_intercept=False, penalty_strength=.2, rho=rho, tol=1e-11).fit(X, X @ beta)
            assert_allclose(model.coef_, expected, atol=1e-9)

    def test_huber_with_l2_matches_independent_scipy_solver(self):
        y = self.y.copy()
        y[:5] += 10
        delta, penalty = .15, .2
        design = np.column_stack((self.X, np.ones(len(self.X))))

        def objective(parameters):
            residuals = design @ parameters - y
            value = huber_loss(residuals, delta) + penalty / 2 * (parameters[:-1] @ parameters[:-1])
            gradient = design.T @ huber_loss_gradient(residuals, delta)
            gradient[:-1] += penalty * parameters[:-1]
            return value, gradient

        reference = minimize(objective, np.zeros(6), jac=True, method="BFGS", options={"gtol": 1e-10})
        self.assertLess(np.max(np.abs(reference.jac)), 1e-7)
        model = LinearRegressor(loss="huber", huber_delta=delta, penalty_strength=penalty, rho=1, tol=1e-10).fit(self.X, y)
        self.assertTrue(model.converged_)
        assert_allclose(np.append(model.coef_, model.intercept_), reference.x, atol=2e-7)

    def test_huber_elastic_net_satisfies_kkt(self):
        y = self.y.copy()
        y[:3] += 4
        model = LinearRegressor(loss="huber", huber_delta=.1, penalty_strength=.02, rho=.5, tol=1e-10).fit(self.X, y)
        gradient = self.X.T @ huber_loss_gradient(model.predict(self.X) - y, .1) + .01 * model.coef_
        active = np.abs(model.coef_) > 1e-8
        assert_allclose(gradient[active] + .01 * np.sign(model.coef_[active]), 0, atol=1e-8)
        self.assertTrue(np.all(np.abs(gradient[~active]) <= .01 + 1e-8))
        self.assertLess(abs(huber_loss_gradient(model.predict(self.X) - y, .1).sum()), 1e-8)

    def test_huber_intercept_is_optimized_not_fixed_to_target_mean(self):
        X = np.zeros((11, 1))
        y = np.array([0.] * 10 + [100.])
        model = LinearRegressor(loss="huber", huber_delta=1., tol=1e-10).fit(X, y)
        self.assertAlmostEqual(model.intercept_, .1, places=8)
        self.assertGreater(abs(model.intercept_ - y.mean()), 8)

    def test_huber_explicit_delta_and_no_ignored_delta(self):
        with self.assertRaises(ValueError):
            LinearRegressor(loss="huber").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            LinearRegressor(huber_delta=1).fit(self.X, self.y)

    def test_scaling_train_only_and_coef_original_units(self):
        model = LinearRegressor(standardize=True, penalty_strength=.1, rho=.5).fit(self.X, self.y)
        expected = LinearRegressor(penalty_strength=.1, rho=.5).fit(
            (self.X - self.X.mean(axis=0)) / self.X.std(axis=0), self.y,
        )
        shifted = self.X[:3] + 100
        assert_allclose(model.predict(shifted), expected.predict((shifted - self.X.mean(axis=0)) / self.X.std(axis=0)))
        assert_allclose(model.predict(shifted), shifted @ model.coef_ + model.intercept_)
        assert_allclose(model.feature_offset_, self.X.mean(axis=0))

    def test_unconverged_fit_reports_failure(self):
        with self.assertWarns(RuntimeWarning):
            model = LinearRegressor(penalty_strength=.1, max_iter=1, initial_step=1e-5).fit(self.X, self.y)
        self.assertFalse(model.converged_)
        self.assertEqual(model.n_iter_, 1)


class DimensionReductionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(712)
        self.X = rng.normal(size=(100, 7)) * np.arange(1, 8) + 3
        self.y = .1 + self.X @ rng.normal(size=7) + rng.normal(size=100) * .02

    def test_pcr_against_explicit_svd_regression(self):
        X = self.X - self.X.mean(axis=0)
        _, _, right = np.linalg.svd(X, full_matrices=False)
        scores = X @ right[:3].T
        coefficients = np.linalg.lstsq(scores, self.y - self.y.mean(), rcond=None)[0]
        model = PCRRegressor(3).fit(self.X, self.y)
        assert_allclose(model.predict(self.X), scores @ coefficients + self.y.mean(), atol=1e-11)

    def test_full_components_recover_ols(self):
        for cls in (PCRRegressor, PLSRegressor):
            with self.subTest(model=cls.__name__):
                model = cls(7).fit(self.X, self.y)
                assert_allclose(model.predict(self.X), LinearRegressor().fit(self.X, self.y).predict(self.X), atol=1e-9)

    def test_pls_matches_independent_residual_deflation(self):
        # Independent PLS1/NIPALS reference, not the production SIMPLS routine.
        X = self.X - self.X.mean(axis=0)
        E, f = X.copy(), self.y - self.y.mean()
        weights, loadings, target_loadings = [], [], []
        for _ in range(4):
            w = E.T @ f
            w /= np.linalg.norm(w)
            t = E @ w
            p = E.T @ t / (t @ t)
            q = f @ t / (t @ t)
            weights.append(w)
            loadings.append(p)
            target_loadings.append(q)
            E -= np.outer(t, p)
            f -= t * q
        W, P = np.array(weights).T, np.array(loadings).T
        coefficients = W @ np.linalg.solve(P.T @ W, np.array(target_loadings))
        model = PLSRegressor(4).fit(self.X, self.y)
        assert_allclose(model.coef_, coefficients, atol=1e-10)
        assert_allclose(model.transform(self.X).T @ model.transform(self.X), np.eye(4), atol=1e-11)

    def test_pcr_directions_ignore_target_but_pls_uses_target(self):
        reversed_y = self.y[::-1]
        pcr_a, pcr_b = PCRRegressor(2).fit(self.X, self.y), PCRRegressor(2).fit(self.X, reversed_y)
        assert_allclose(pcr_a.components_, pcr_b.components_)
        pls_a, pls_b = PLSRegressor(2).fit(self.X, self.y), PLSRegressor(2).fit(self.X, reversed_y)
        self.assertGreater(np.linalg.norm(pls_a.components_ - pls_b.components_), .01)

    def test_transform_and_predict_agree_with_training_only_scaling(self):
        for cls in (PCRRegressor, PLSRegressor):
            model = cls(3, standardize=True).fit(self.X, self.y)
            test = self.X[:8] + 10
            assert_allclose(model.predict(test), model.transform(test) @ model.component_coef_ + model.target_offset_, atol=1e-10)
            assert_allclose(model.feature_offset_, self.X.mean(axis=0))
            assert_allclose(model.feature_scale_, self.X.std(axis=0))

    def test_no_intercept_full_components(self):
        for cls in (PCRRegressor, PLSRegressor):
            model = cls(7, fit_intercept=False, standardize=True).fit(self.X, self.y)
            expected = LinearRegressor(fit_intercept=False).fit(self.X, self.y)
            assert_allclose(model.predict(self.X), expected.predict(self.X), atol=1e-9)
            self.assertEqual(model.intercept_, 0.)

    def test_pcr_rejects_rank_deficient_request(self):
        X = np.column_stack((self.X[:, 0], self.X[:, 0], np.ones(len(self.X))))
        with self.assertRaisesRegex(ValueError, "rank"):
            PCRRegressor(2).fit(X, self.y)

    def test_pls_rank_and_covariance_exhaustion(self):
        X = np.column_stack((self.X[:, 0], self.X[:, 0], np.ones(len(self.X))))
        model = PLSRegressor(3).fit(X, self.y)
        self.assertEqual(model.n_components_, 1)
        assert_allclose(model.predict(X), LinearRegressor().fit(X, self.y).predict(X), atol=1e-9)

    def test_pls_constant_target(self):
        model = PLSRegressor(3).fit(self.X, np.full(len(self.X), .02))
        self.assertEqual(model.n_components_, 0)
        assert_allclose(model.predict(self.X + 100), .02, atol=1e-15)

    def test_component_count_validation(self):
        for cls in (PCRRegressor, PLSRegressor):
            for k in (-1, 1.5, True, 8):
                with self.subTest(model=cls.__name__, k=k), self.assertRaises(ValueError):
                    cls(k).fit(self.X, self.y)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(65).normal(size=(20, 4))
        self.y = self.X @ np.array([1., 2., 3., 4.])

    def factories(self):
        return (LinearRegressor, lambda: PCRRegressor(2), lambda: PLSRegressor(2))

    def test_arrays_not_mutated_and_prediction_shape(self):
        X, y = self.X.copy(), self.y.copy()
        for factory in self.factories():
            model = factory().fit(X, y)
            self.assertEqual(model.predict(X).shape, (20,))
            self.assertEqual(model.predict(X[:0]).shape, (0,))
            self.assertEqual(model.predict(X[:1]).shape, (1,))
            assert_array_equal(X, self.X)
            assert_array_equal(y, self.y)

    def test_dataframe_schema_and_index_alignment(self):
        X = pd.DataFrame(self.X, columns=list("abcd"))
        y = pd.Series(self.y, index=X.index)
        for factory in self.factories():
            model = factory().fit(X, y)
            self.assertEqual(model.predict(X).shape, (20,))
            for wrong in (X[list("bacd")], X.to_numpy(), X.rename(columns={"a": "other"})):
                with self.assertRaises(ValueError):
                    model.predict(wrong)
            with self.assertRaises(ValueError):
                factory().fit(X, y.iloc[::-1])

    def test_invalid_shapes_and_missing_values(self):
        for factory in self.factories():
            with self.assertRaises(RuntimeError):
                factory().predict(self.X)
            for X, y in ((self.X.T, self.y), (self.X, self.y[:, None]),
                         (self.X * np.nan, self.y), (self.X, self.y * np.inf),
                         (self.X[:0], self.y[:0]), (self.X.astype(complex) + 1j, self.y)):
                with self.subTest(model=factory, shape=X.shape), self.assertRaises(ValueError):
                    factory().fit(X, y)
            model = factory().fit(self.X, self.y)
            with self.assertRaises(ValueError):
                model.predict(self.X[:, :3])

    def test_failed_refit_invalidates_old_fit(self):
        for factory in self.factories():
            model = factory().fit(self.X, self.y)
            with self.assertRaises(ValueError):
                model.fit(self.X, self.y[:, None])
            with self.assertRaises(RuntimeError):
                model.predict(self.X)

    def test_constant_columns_remain_zero_after_centering(self):
        X = np.column_stack((self.X[:, 0], np.full(20, .1)))
        for factory in (LinearRegressor, lambda **kw: PCRRegressor(1, **kw), lambda **kw: PLSRegressor(1, **kw)):
            model = factory(standardize=True).fit(X, self.y)
            self.assertEqual(model.feature_scale_[1], 1.)
            self.assertEqual(model.coef_[1], 0.)


if __name__ == "__main__":
    unittest.main()
