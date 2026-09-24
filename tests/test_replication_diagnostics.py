from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import svd
from scipy.optimize import minimize

from core.losses import huber_loss_gradient, huber_residual_loss
from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.model_selection import tune_hyperparameters
from core.replication_report import make_report, report_tables
from core.run import configured_specifications, run
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


class EmpiricalModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(704)
        self.X = rng.normal(size=(150, 5)) * np.arange(1, 6) + 2
        self.y = .05 + self.X[:, 0] * .02 + rng.normal(scale=.02, size=len(self.X))
        self.y[::17] += .6

    def test_zero_components_has_explicit_training_mean_or_zero_forecast(self):
        for cls in (PCRRegressor, PLSRegressor):
            for intercept in (True, False):
                model = cls(0, fit_intercept=intercept).fit(self.X, self.y)
                assert_allclose(model.predict(self.X + 100), self.y.mean() if intercept else 0.)
                self.assertEqual(model.n_components_, 0)
                self.assertEqual(model.transform(self.X).shape, (150, 0))

    def test_pcr_shared_svd_matches_independent_candidate_fits(self):
        validation = self.X[:20] + .3
        for intercept in (True, False):
            for standardize in (True, False):
                candidates = [{"n_components": k, "fit_intercept": intercept, "standardize": standardize}
                              for k in (3, 0, 1, 2)]
                with patch("core.models.dimension_reduction.svd", wraps=svd) as decomposition:
                    result = tune_hyperparameters(PCRRegressor, candidates, self.X, self.y, validation, self.y[:20])
                    self.assertEqual(decomposition.call_count, 1)
                scores = []
                for candidate in candidates:
                    fitted = PCRRegressor(**candidate).fit(self.X, self.y)
                    scores.append(np.mean((fitted.predict(validation) - self.y[:20]) ** 2))
                assert_allclose([r["validation_loss"] for r in result.validation_results], scores, rtol=1e-12)
                self.assertEqual(result.best_params, candidates[int(np.argmin(scores))])
                expected = PCRRegressor(**result.best_params).fit(self.X, self.y)
                assert_allclose(result.best_estimator.predict(validation), expected.predict(validation), atol=1e-12)

    def test_svd_huber_matches_independent_bfgs_objective_and_predictions(self):
        X = np.column_stack((self.X, self.X[:, 0], np.zeros(len(self.X))))
        delta = .08
        model = LinearRegressor(loss="huber", huber_delta=delta, solver="svd_apg", tol=1e-10).fit(X, self.y)
        design = np.column_stack((self.X, np.ones(len(X))))

        def objective(parameters):
            residuals = design @ parameters - self.y
            return huber_residual_loss(residuals, delta), design.T @ huber_loss_gradient(residuals, delta)

        reference = minimize(objective, np.zeros(6), jac=True, method="BFGS", options={"gtol": 1e-10})
        self.assertLess(np.max(np.abs(reference.jac)), 1e-8)
        self.assertTrue(model.converged_)
        assert_allclose(model.predict(X), design @ reference.x, atol=1e-7)
        self.assertAlmostEqual(model.objective_, reference.fun, places=12)
        gradient = np.append(X.T @ huber_loss_gradient(model.predict(X) - self.y, delta),
                             huber_loss_gradient(model.predict(X) - self.y, delta).sum())
        self.assertLess(np.linalg.norm(gradient, ord=np.inf), 1e-8)

    def test_svd_huber_handles_constant_design_and_no_intercept(self):
        X = np.zeros((11, 2))
        y = np.array([0.] * 10 + [100.])
        model = LinearRegressor(loss="huber", huber_delta=1., solver="svd_apg", tol=1e-10).fit(X, y)
        self.assertAlmostEqual(model.intercept_, .1, places=7)
        model = LinearRegressor(loss="huber", huber_delta=1., solver="svd_apg", fit_intercept=False).fit(X, y)
        self.assertTrue(model.converged_)
        assert_array_equal(model.predict(X), np.zeros(11))

    def test_huber_quantile_is_from_training_ols_residuals_only(self):
        ols = LinearRegressor().fit(self.X, self.y)
        expected = np.quantile(np.abs(ols.predict(self.X) - self.y), .999, method="linear")
        for solver in ("apg", "svd_apg"):
            model = LinearRegressor(loss="huber", huber_quantile=.999, solver=solver).fit(self.X, self.y)
            self.assertAlmostEqual(model.huber_delta_, expected, places=12)
        candidate = {"loss": "huber", "huber_quantile": .999, "solver": "svd_apg"}
        a = tune_hyperparameters(LinearRegressor, [candidate], self.X, self.y, self.X, self.y)
        b = tune_hyperparameters(LinearRegressor, [candidate], self.X, self.y, self.X + 10, self.y + 100)
        self.assertEqual(a.validation_results[0]["huber_delta"], b.validation_results[0]["huber_delta"])
        assert_array_equal(a.best_estimator.coef_, b.best_estimator.coef_)
        self.assertLessEqual(a.validation_results[0]["stationarity"], a.validation_results[0]["stopping_threshold"])

    def test_huber_invalid_rules_are_not_silently_accepted(self):
        for settings in ({"huber_delta": .1, "huber_quantile": .999}, {"huber_quantile": 1.},
                         {"huber_quantile": True}, {"huber_delta": .1, "solver": "unknown"},
                         {"huber_delta": .1, "solver": "svd_apg", "penalty_strength": .1}):
            with self.assertRaises(ValueError):
                LinearRegressor(loss="huber", **settings).fit(self.X, self.y)


class ReplicationReportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("gkx_table1_figure3_v1")

    def test_approved_pilot_settings_and_model_construction(self):
        specs = configured_specifications(self.cfg.RUN.MODELS)
        self.assertEqual([(name, len(candidates)) for name, _, candidates in specs],
                         [("PCR", 101), ("PLS", 11), ("OLS_HUBER", 1)])
        self.assertEqual(self.cfg.RUN.MODELS.PCR.COMPONENT_GRID, list(range(101)))
        self.assertEqual(self.cfg.RUN.MODELS.PLS.COMPONENT_GRID, list(range(11)))
        self.assertEqual(specs[-1][2][0]["huber_quantile"], .999)
        self.assertIsNone(specs[-1][2][0]["huber_delta"])

    def test_saved_report_uses_common_years_fitted_counts_and_pending_gaps(self):
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 1989
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = [0, 1, 2]
        self.cfg.RUN.MODELS.PLS.COMPONENT_GRID = [0, 1, 2]
        months = pd.period_range("1957-03", "1989-12", freq="M")
        rng = np.random.default_rng(43)
        X = rng.normal(size=(len(months), 3))
        y = X[:, 0] * .03 + rng.normal(scale=.01, size=len(months))
        info = {"name": self.cfg.DATA.DATASET, "panel_sha256": "test-panel", "metadata_sha256": "test-meta"}
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            self.cfg.RUN.OUTPUT_DIR = temporary
            folder = create_run(self.cfg, info)
            run(X, y, months, self.cfg, dataset_info=info, checkpoint_dir=folder, stop_after_year=1987)
            records = read_checkpoints(folder, self.cfg)
            table, complexity, curves = report_tables(records, self.cfg)
            self.assertEqual(table.common_test_years.tolist(), ["1987"] * 3)
            self.assertTrue(complexity.loc[complexity.test_year == 1988, "fitted_components"].isna().all())
            for _, row in table.iterrows():
                p = records[1987, row.model]["predictions"]
                observed, predicted = np.array(p["observed"]), np.array(p["prediction"])
                expected = 100 * (1 - np.sum((observed - predicted) ** 2) / np.sum(observed ** 2))
                self.assertAlmostEqual(row.r2_percent, expected)
            with patch("core.model_selection.tune_hyperparameters", side_effect=AssertionError("refit")):
                output = make_report(folder)
            self.assertTrue((output / "figure3_complexity.png").is_file())
            self.assertTrue((output / "validation_1987.png").is_file())
            # A missing model cannot silently give the Table 1 columns different samples.
            del records[1987, "OLS_HUBER"]
            table, _, _ = report_tables(records, self.cfg)
            self.assertTrue((table.n_test == 0).all())

    def test_report_rejects_misaligned_models(self):
        record = {"predictions": {"row_position": [0], "observed": [.1], "prediction": [.1],
                                  "return_month": ["1987-01"], "test_year": [1987]}}
        import copy
        records = {}
        for i, name in enumerate(self.cfg.RUN.MODELS.ENABLED):
            records[1987, name] = copy.deepcopy(record)
            records[1987, name]["predictions"]["specification"] = [name]
            records[1987, name]["predictions"]["row_position"] = [i]
        with self.assertRaisesRegex(ValueError, "same test observations"):
            report_tables(records, self.cfg)


if __name__ == "__main__":
    unittest.main()
