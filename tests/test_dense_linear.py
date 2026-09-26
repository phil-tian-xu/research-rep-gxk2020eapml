from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import least_squares, minimize

import test_batch_workflow as fixtures
from core.data_query.parquet_dataset import load_window
from core.dense_linear_run import fit_candidates
from core.dense_parquet_run import run_parquet
from core.models import LinearRegressor
from core.replication_report import make_report, report_tables
from core.run import configured_specifications
from core.run_state import create_run, read_checkpoints


class HuberNumericsTests(unittest.TestCase):
    def test_ols_huber_matches_independent_scipy_solver(self):
        rng = np.random.default_rng(902)
        X = rng.normal(size=(120, 3))
        y = .2 + X @ np.array([.1, -.2, .3]) + rng.normal(scale=.04, size=120)
        y[::10] += 2
        model = LinearRegressor(loss="huber", huber_delta=.1, tol=1e-10).fit(X, y)
        reference = least_squares(lambda b: X @ b[:3] + b[3] - y,
                                  np.zeros(4), loss="huber", f_scale=.1,
                                  gtol=1e-12, ftol=1e-12, xtol=1e-12)
        self.assertTrue(model.converged_)
        assert_allclose(model.predict(X), X @ reference.x[:3] + reference.x[3], atol=1e-7)

    def test_elastic_net_matches_independent_constrained_objective(self):
        rng = np.random.default_rng(331)
        X = rng.normal(size=(150, 4))
        y = .05 + .3 * X[:, 0] + rng.normal(scale=.03, size=150)
        y[::9] += 1
        delta, strength, rho = .1, .05, .5
        model = LinearRegressor(loss="huber", huber_delta=delta,
                                penalty_strength=strength, rho=rho, tol=1e-10).fit(X, y)
        # Epigraph form of L1: independent SLSQP fit, no shared loss/prox code.
        def objective(v):
            residual = X @ v[:4] + v[4] - y
            loss = np.where(abs(residual) <= delta, residual**2,
                            2 * delta * abs(residual) - delta**2).mean()
            return loss + strength * (1-rho) * v[5:].sum() + strength*rho/2 * (v[:4] @ v[:4])
        reference = minimize(objective, np.zeros(9), method="SLSQP",
                             constraints={"type": "ineq", "fun": lambda v: np.r_[v[5:]-v[:4], v[5:]+v[:4]]},
                             options={"ftol": 1e-12, "maxiter": 1000})
        self.assertTrue(reference.success, reference.message)
        self.assertTrue(model.converged_)
        assert_allclose(model.predict(X), X @ reference.x[:4] + reference.x[4], atol=2e-6)
        self.assertAlmostEqual(model.objective_, reference.fun, places=10)


class DenseLinearWorkflowTests(unittest.TestCase):
    setUp = fixtures.BatchWorkflowTests.setUp

    def configure(self, name):
        self.cfg.RUN.FITTING = "dense"
        self.cfg.RUN.MODELS.ENABLED = [name]
        self.cfg.RUN.MODELS.OLS3_HUBER.FEATURES = ["x2", "x0", "x4"]
        enet = self.cfg.RUN.MODELS.ENET_HUBER
        enet.HUBER_DELTA, enet.HUBER_QUANTILE = None, .999
        enet.PENALTY_GRID = [.001, .01, .1]
        self.cfg.RUN.MODELS.LINEAR_SOLVER.TOL = 1e-9

    def test_projection_retains_requested_order_and_rejects_missing(self):
        X, y, _, _ = load_window(self.panel, 2000, 2001, feature_columns=["x4", "x0", "x2"])
        mask = self.months.year <= 2001
        assert_allclose(X, self.X[mask][:, [4, 0, 2]], atol=1e-15)
        assert_allclose(y, self.y[mask], atol=1e-15)
        for columns in (["missing"], ["x0", "x0"], []):
            with self.assertRaises(ValueError):
                load_window(self.panel, 2000, 2001, feature_columns=columns)

    def test_both_models_staged_equal_complete_and_coefficients_predict(self):
        for name in ("OLS3_HUBER", "ENET_HUBER"):
            self.configure(name)
            with redirect_stdout(StringIO()):
                whole = create_run(self.cfg, self.panel.info)
                run_parquet(self.panel, self.cfg, checkpoint_dir=whole)
                staged = create_run(self.cfg, self.panel.info)
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
                make_report(staged)
            left, right = read_checkpoints(whole, self.cfg), read_checkpoints(staged, self.cfg)
            for key in left:
                self.assertEqual(left[key], right[key])
                year, _ = key
                tuning = left[key]["tuning"]
                X, _, _, _ = load_window(self.panel, year, year, feature_columns=tuning["feature_names"])
                assert_allclose(X @ tuning["coefficients"] + tuning["intercept"], left[key]["predictions"]["prediction"])
                trainX, trainy, _, _ = load_window(self.panel, 2000, year-2, feature_columns=tuning["feature_names"])
                design = np.column_stack([trainX, np.ones(len(trainy))])
                ols = np.linalg.lstsq(design, trainy, rcond=None)[0]
                expected = np.quantile(abs(design @ ols - trainy), .999)
                for c in tuning["validation_results"]:
                    self.assertAlmostEqual(c["huber_delta"], expected, places=12)
            table, complexity, curves = report_tables(left, self.cfg)
            self.assertTrue(table.complete_horizon.all())
            self.assertTrue((complexity.status == "complete").all())
            self.assertEqual(len(curves), 3 * (3 if name == "ENET_HUBER" else 1))
            with patch("core.dense_linear_run.fit_candidates", side_effect=AssertionError("Refit forbidden")), redirect_stdout(StringIO()):
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)

    def test_threshold_ols_fit_once_and_nonconvergence_rejected(self):
        self.configure("ENET_HUBER")
        _, factory, candidates = configured_specifications(self.cfg.RUN.MODELS)[0]
        original = LinearRegressor.fit
        calls = []
        def tracked(model, X, y):
            calls.append(model.loss)
            return original(model, X, y)
        with patch.object(LinearRegressor, "fit", tracked), redirect_stdout(StringIO()):
            models = fit_candidates(self.X, self.y, factory, candidates, label="test")
        self.assertEqual(calls.count("squared"), 1)
        self.assertEqual(len({m.huber_delta_ for m in models}), 1)
        candidates[0]["max_iter"] = 1
        with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(RuntimeError, "did not converge"), redirect_stdout(StringIO()):
            fit_candidates(self.X, self.y, factory, candidates[:1], label="test")


if __name__ == "__main__":
    unittest.main()
