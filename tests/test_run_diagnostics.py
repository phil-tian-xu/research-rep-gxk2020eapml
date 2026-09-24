from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from core.model_selection import tune_hyperparameters
from core.models import LinearRegressor
from core.run import run
from utils.general_utils import load_config


class RunDiagnosticsTests(unittest.TestCase):
    def test_nonconvergence_reports_context_and_stopping_diagnostics(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        output = StringIO()
        with redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(RuntimeError) as caught:
                tune_hyperparameters(
                    LinearRegressor,
                    [{"loss": "huber", "huber_delta": .1, "max_iter": 1}],
                    X, X[:, 0], X, X[:, 0], model_name="OLS_HUBER", test_year=1987,
                )
        self.assertIn("iterations=1", str(caught.exception))
        self.assertIn("stationarity=", str(caught.exception))
        self.assertIn("stopping_threshold=", str(caught.exception))
        note = " ".join(caught.exception.__notes__)
        for expected in ("model=OLS_HUBER", "test_year=1987", "candidate=1/1", "stage=fit", "parameters="):
            self.assertIn(expected, note)
        self.assertIn("Fitting...", output.getvalue())

    def test_validation_error_retains_type_and_original_traceback(self):
        def broken_score(observed, predicted):
            raise ValueError("scoring failed")

        X = np.arange(10, dtype=float).reshape(-1, 1)
        with redirect_stdout(StringIO()), self.assertRaises(ValueError) as caught:
            tune_hyperparameters(LinearRegressor, [{}], X, X[:, 0], X, X[:, 0],
                                 validation_loss_fn=broken_score, model_name="OLS", test_year=1990)
        self.assertEqual(str(caught.exception), "scoring failed")
        self.assertIn("stage=validation scoring", caught.exception.__notes__[0])
        self.assertIn("model=OLS", caught.exception.__notes__[0])

    def test_active_config_runs_three_models_for_five_years(self):
        cfg = load_config("gkx_all_models_workflow_v2")
        self.assertEqual(cfg.RUN.MODELS.ENABLED, ["OLS", "PCR", "PLS"])
        self.assertEqual(cfg.RUN.SPLIT.FIRST_TEST_YEAR, 1987)
        self.assertEqual(cfg.RUN.SPLIT.LAST_TEST_YEAR, 1991)
        months = pd.period_range("1957-03", "1991-12", freq="M")
        rng = np.random.default_rng(42)
        X = rng.normal(size=(len(months), 4))
        y = X[:, 0] * .1 + rng.normal(scale=.01, size=len(months))
        with redirect_stdout(StringIO()):
            result = run(X, y, months, cfg)
        self.assertEqual(len(result.annual_metrics), 15)
        self.assertEqual(len(result.predictions), 3 * 5 * 12)
        self.assertEqual(sum(len(item["validation_results"]) for item in result.tuning_history), 35)

    def test_test_prediction_error_identifies_model_and_year(self):
        cfg = load_config("gkx_all_models_workflow_v2")
        months = pd.period_range("1957-03", "1991-12", freq="M")
        X = np.ones((len(months), 1))
        with patch("core.run.tune_hyperparameters") as tune:
            tune.return_value.best_estimator.predict.side_effect = ValueError("prediction failed")
            tune.return_value.best_params = {"loss": "squared"}
            with self.assertRaises(ValueError) as caught:
                run(X, np.ones(len(months)), months, cfg)
        note = caught.exception.__notes__[0]
        self.assertIn("model=OLS", note)
        self.assertIn("test_year=1987", note)
        self.assertIn("stage=test prediction/scoring", note)


if __name__ == "__main__":
    unittest.main()
