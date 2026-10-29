from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import test_batch_workflow as fixtures
from core.data_query.parquet_dataset import load_window
from core.dense_parquet_run import run_parquet
from core.figure4_report import restore_predictor
from core.replication_report import report_tables
from core.run_state import create_run, read_checkpoints


class OLSWorkflowTests(unittest.TestCase):
    setUp = fixtures.BatchWorkflowTests.setUp

    def test_ols_and_ols3_match_independent_training_only_fit_and_resume(self):
        for name, columns in (("OLS", list(self.panel.features)), ("OLS3", ["x2", "x0", "x4"])):
            self.cfg.RUN.FITTING = "dense"
            self.cfg.RUN.MODELS.ENABLED = [name]
            self.cfg.RUN.MODELS.OLS3.FEATURES = columns
            full, staged = create_run(self.cfg, self.panel.info), create_run(self.cfg, self.panel.info)
            with redirect_stdout(StringIO()):
                run_parquet(self.panel, self.cfg, checkpoint_dir=full)
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
                with patch("core.dense_ols_run.load_window", side_effect=AssertionError("Refitting forbidden")):
                    run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            left, right = read_checkpoints(full, self.cfg), read_checkpoints(staged, self.cfg)
            for key in left:
                self.assertEqual(left[key], right[key])
                year, _ = key
                X, y, _, _ = load_window(self.panel, 2000, year-2, feature_columns=columns)
                design = np.column_stack((np.ones(len(X)), X))
                beta = np.linalg.lstsq(design, y, rcond=None)[0]
                V, vy, _, _ = load_window(self.panel, year-1, year-1, feature_columns=columns)
                T, _, _, _ = load_window(self.panel, year, year, feature_columns=columns)
                record = left[key]
                assert_allclose(record["predictions"]["prediction"], beta[0] + T @ beta[1:], atol=1e-13)
                self.assertAlmostEqual(record["metrics"]["best_validation_loss"], np.mean((vy-beta[0]-V@beta[1:])**2))
                predict, names, _ = restore_predictor(record["tuning"])
                self.assertEqual(names, columns)
                assert_allclose(predict(T), record["predictions"]["prediction"], atol=0, rtol=0)
                self.assertEqual(record["metrics"]["n_train"], len(y))
            table, complexity, _ = report_tables(left, self.cfg)
            self.assertTrue(table.complete_horizon.all())
            self.assertTrue(complexity.empty)

    def test_invalid_ols3_features_fail_before_loading(self):
        self.cfg.RUN.MODELS.ENABLED = ["OLS3"]
        for features in (["x0", "x1"], ["x0", "x0", "x1"], ["x0", "x1", "missing"]):
            self.cfg.RUN.MODELS.OLS3.FEATURES = features
            folder = create_run(self.cfg, self.panel.info)
            with patch("core.dense_ols_run.load_window", side_effect=AssertionError("Must validate first")):
                with self.assertRaisesRegex(ValueError, "predictors"):
                    run_parquet(self.panel, self.cfg, checkpoint_dir=folder)


if __name__ == "__main__":
    unittest.main()
