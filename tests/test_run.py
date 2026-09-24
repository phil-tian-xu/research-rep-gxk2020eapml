import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
import yaml

from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.data_query.generate_experimental_data import generate_panel
from core.run import configured_loss, configured_specifications, run, save_run
from core.losses import huber_loss, mse_loss
from utils.config_utils.config import get_cfg
from utils.general_utils import load_config


class ConfiguredRunTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("model_demo")
        self.generator_settings = {"start_month": "2000-01", "end_month": "2004-12",
                                   "stocks_per_month": 2, "coefficients": [.1, -.05, .01]}
        self.cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
        self.cfg.RUN.SPLIT.FIRST_TEST_YEAR = 2003
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004
        self.cfg.RUN.SPLIT.VALIDATION_YEARS = 1
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = [1, 2]
        self.cfg.RUN.MODELS.PLS.COMPONENT_GRID = [1, 2]
        panel, _ = generate_panel(**self.generator_settings)
        self.X, self.y = panel[["x0", "x1", "x2"]], panel["excess_return"]
        self.months = pd.PeriodIndex(panel.return_month, freq="M")

    def test_existing_loader_merges_defaults_without_changing_showcase(self):
        demo = load_config("demo")
        self.assertEqual(demo.GENERAL.TASK_TYPE, "market_data_update")
        self.assertEqual(demo.DATA.SOURCE.NAME, "lseg")
        config = load_config("model_demo")
        self.assertEqual(config.GENERAL.TASK_TYPE, "run")
        self.assertEqual(config.DATA.DATASET, "synthetic_linear_v1")
        self.assertNotIn("SYNTHETIC", config)
        self.assertNotIn("SEED", config.RUN)
        self.assertEqual(config.RUN.SPLIT.LAST_TEST_YEAR, 1988)
        self.assertEqual(config.RUN.MODELS.LINEAR_SOLVER.MAX_ITER, 5000)
        config.RUN.MODELS.PCR.COMPONENT_GRID.append(6)
        self.assertEqual(load_config("model_demo").RUN.MODELS.PCR.COMPONENT_GRID, [2, 4])
        self.assertEqual(get_cfg().RUN.MODELS.PCR.COMPONENT_GRID, [])

    def test_settings_build_independent_candidate_grids(self):
        settings = self.cfg.RUN.MODELS
        settings.ENABLED = ["ENET_HUBER", "PCR"]
        settings.STANDARDIZE = True
        settings.ENET_HUBER.RHO = .25
        settings.ENET_HUBER.PENALTY_GRID = [.002, .02, .2]
        specifications = configured_specifications(settings)
        self.assertEqual([s[0] for s in specifications], settings.ENABLED)
        enet, pcr = specifications[0][2], specifications[1][2]
        self.assertEqual([p["penalty_strength"] for p in enet], [.002, .02, .2])
        self.assertTrue(all(p["rho"] == .25 and p["standardize"] for p in enet))
        self.assertEqual([p["n_components"] for p in pcr], [1, 2])

    def test_all_specifications_run_each_year_and_pool_predictions(self):
        result = run(self.X, self.y, self.months, self.cfg)
        self.assertEqual(len(result.annual_metrics), 10)
        self.assertEqual(len(result.summary), 5)
        self.assertEqual(len(result.tuning_history), 10)
        self.assertEqual(len(result.predictions), 5 * 2 * 24)
        self.assertTrue((result.annual_metrics.n_validation == 24).all())
        self.assertTrue((result.annual_metrics.query("test_year == 2003").n_train == 48).all())
        self.assertTrue((result.annual_metrics.query("test_year == 2004").n_train == 72).all())
        for _, summary in result.summary.iterrows():
            group = result.predictions[result.predictions.specification == summary.specification]
            self.assertEqual(summary.n_test, 48)
            self.assertAlmostEqual(summary.test_loss, mse_loss(group.observed, group.prediction))
            expected = 1 - np.sum((group.observed - group.prediction)**2) / np.sum(group.observed**2)
            self.assertAlmostEqual(summary.r2_zero, expected)
            assert_allclose(group.observed, self.y.iloc[group.row_position])
        for history in result.tuning_history:
            count = 1 if history["specification"] in ("OLS", "OLS_HUBER") else 2
            self.assertEqual(len(history["validation_results"]), count)

    def test_test_loss_switch_does_not_change_fits(self):
        self.cfg.RUN.MODELS.ENABLED = ["OLS", "PCR"]
        baseline = run(self.X, self.y, self.months, self.cfg)
        self.cfg.RUN.EVALUATION.TEST.NAME = "huber"
        self.cfg.RUN.EVALUATION.TEST.DELTA = .04
        alternative = run(self.X, self.y, self.months, self.cfg)
        assert_allclose(baseline.predictions.prediction, alternative.predictions.prediction)
        self.assertEqual(baseline.tuning_history, alternative.tuning_history)
        for _, row in alternative.summary.iterrows():
            group = alternative.predictions[alternative.predictions.specification == row.specification]
            self.assertEqual(row.test_loss_name, "Huber(delta=0.04)")
            self.assertAlmostEqual(row.test_loss, huber_loss(group.observed, group.prediction, delta=.04))

    def test_final_test_targets_do_not_affect_fits_or_tuning(self):
        self.cfg.RUN.MODELS.ENABLED = ["ENET_HUBER", "PLS"]
        baseline = run(self.X, self.y, self.months, self.cfg)
        changed_y = self.y.copy()
        changed_y.loc[self.months.year == 2004] += 100
        changed = run(self.X, changed_y, self.months, self.cfg)
        assert_allclose(baseline.predictions.prediction, changed.predictions.prediction)
        self.assertEqual(baseline.tuning_history, changed.tuning_history)

    def test_configured_validation_loss_used_for_every_candidate(self):
        self.cfg.RUN.MODELS.ENABLED = ["OLS"]
        loss_settings = self.cfg.RUN.EVALUATION.VALIDATION
        loss_settings.NAME, loss_settings.DELTA = "huber", .03
        result = run(self.X, self.y, self.months, self.cfg)
        from core.models import LinearRegressor
        for record in result.tuning_history:
            year = record["test_year"]
            train = self.months.year < year - 1
            validation = self.months.year == year - 1
            model = LinearRegressor(**record["best_params"]).fit(self.X.loc[train], self.y.loc[train])
            expected = huber_loss(self.y.loc[validation], model.predict(self.X.loc[validation]), delta=.03)
            self.assertAlmostEqual(record["validation_results"][0]["validation_loss"], expected)

    def test_synthetic_seed_reproducible(self):
        panel, _ = generate_panel(**self.generator_settings)
        assert_array_equal(panel[["x0", "x1", "x2"]], self.X)
        assert_array_equal(panel.excess_return, self.y)
        assert_array_equal(pd.PeriodIndex(panel.return_month, freq="M"), self.months)

    def test_missing_choices_and_unknown_names_fail(self):
        for enabled in ([], ["OLS", "OLS"], ["UNKNOWN"]):
            self.cfg.RUN.MODELS.ENABLED = enabled
            with self.assertRaises(ValueError):
                configured_specifications(self.cfg.RUN.MODELS)
        self.cfg.RUN.MODELS.ENABLED = ["PCR"]
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = []
        with self.assertRaises(ValueError):
            configured_specifications(self.cfg.RUN.MODELS)
        self.cfg.RUN.EVALUATION.TEST.NAME = "unknown"
        with self.assertRaises(ValueError):
            configured_loss(self.cfg.RUN.EVALUATION.TEST)

    def test_date_row_alignment(self):
        with self.assertRaises(ValueError):
            run(self.X, self.y, self.months[:-1], self.cfg)
        dates = pd.Series(self.months, index=self.X.index).iloc[::-1]
        with self.assertRaises(ValueError):
            run(self.X, self.y, dates, self.cfg)

    def test_saved_run_contains_config_predictions_and_all_tuning_scores(self):
        self.cfg.RUN.MODELS.ENABLED = ["OLS"]
        result = run(self.X, self.y, self.months, self.cfg)
        with TemporaryDirectory() as temporary:
            first = save_run(result, temporary)
            second = save_run(result, temporary)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_relative_to(Path(temporary)))
            self.assertEqual(yaml.safe_load((first / "resolved_config.yaml").read_text(encoding="utf-8"))["RUN"]["MODELS"]["ENABLED"], ["OLS"])
            saved_predictions = pd.read_csv(first / "predictions.csv")
            assert_allclose(saved_predictions.prediction, result.predictions.prediction)
            self.assertEqual(json.loads((first / "tuning_history.json").read_text()), result.tuning_history)
            self.assertIn("numpy", json.loads((first / "environment.json").read_text())["packages"])

    def test_zero_benchmark_definition_and_zero_denominator(self):
        self.assertAlmostEqual(zero_benchmark_r2([1., 3.], [0., 2.]), .8)
        self.assertTrue(np.isnan(zero_benchmark_r2([0., 0.], [1., 1.])))


if __name__ == "__main__":
    unittest.main()
