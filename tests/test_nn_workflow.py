import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose

import test_batch_workflow as fixtures
from core.data_query.parquet_dataset import load_window
from core.dense_parquet_run import run_parquet
from core.figure4_report import restore_predictor, verify_predictions
from core.replication_report import report_tables
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install the optional NN dependencies")
class NNWorkflowTests(unittest.TestCase):
    setUp = fixtures.BatchWorkflowTests.setUp

    def configure(self):
        self.cfg.RUN.MODELS = load_config("gkx_nn1_v1").RUN.MODELS
        settings = self.cfg.RUN.MODELS.NN
        settings.DEVICE = "cpu"
        settings.LEARNING_RATE_GRID = [.001, .01]
        settings.L1_GRID = [.00001]
        settings.ENSEMBLE_SIZE = 2
        settings.MAX_EPOCHS = 4
        settings.PATIENCE = 2
        settings.BATCH_SIZE = 16
        self.cfg.RUN.FITTING = "dense"
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004

    def test_selection_checkpoints_resume_saved_predictions_and_training_logs(self):
        self.configure()
        full, staged = create_run(self.cfg, self.panel.info), create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()):
            result = run_parquet(self.panel, self.cfg, checkpoint_dir=full)
            run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
            resumed = run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            with patch("core.nn_run.load_window", side_effect=AssertionError("must skip")):
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
        assert_array_equal(pd.read_csv(result.predictions_path).prediction, pd.read_csv(resumed.predictions_path).prediction)
        records = read_checkpoints(full, self.cfg)
        table, complexity, curves = report_tables(records, self.cfg)
        self.assertTrue(table.complete_horizon.all())
        self.assertTrue(complexity.empty)  # Paper Figure 3 excludes fixed NN architectures.
        for (year, _), record in records.items():
            tuning = record["tuning"]
            scores = [r["validation_loss"] for r in tuning["validation_results"]]
            self.assertEqual(tuning["best_params"], tuning["validation_results"][int(np.argmin(scores))]["params"])
            predict, names, beta = restore_predictor(tuning)
            self.assertIsNone(beta)
            X, y, _, _ = load_window(self.panel, year, year)
            assert_allclose(predict(X), record["predictions"]["prediction"], atol=0, rtol=0)
            verify_predictions(predict, self.panel.window(year, year), names, record)
            V, vy, _, _ = load_window(self.panel, year-1, year-1)
            self.assertAlmostEqual(np.mean((vy - predict(V))**2), min(scores))
            self.assertTrue((full / "years" / str(year) / "NN1_training.csv").is_file())

    def test_invalid_settings_fail_before_loading(self):
        self.configure()
        self.cfg.RUN.MODELS.NN.LEARNING_RATE_GRID = [0.]
        folder = create_run(self.cfg, self.panel.info)
        with patch("core.nn_run.load_window", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "learning rate"):
                run_parquet(self.panel, self.cfg, checkpoint_dir=folder)


if __name__ == "__main__":
    unittest.main()
