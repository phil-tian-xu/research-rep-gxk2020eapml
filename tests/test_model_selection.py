import unittest
import warnings
from functools import partial

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from core.losses import huber_loss
from core.model_selection import tune_hyperparameters
from core.models import LinearRegressor, PCRRegressor
from core.splits import annual_splits


class SplitTests(unittest.TestCase):
    def test_first_and_last_paper_windows(self):
        months = pd.period_range("1957-03", "2016-12", freq="M")
        splits = list(annual_splits(months))
        self.assertEqual(len(splits), 30)
        first, last = splits[0], splits[-1]
        self.assertEqual(months[first.train_indices[-1]], pd.Period("1974-12"))
        self.assertEqual(months[first.validation_indices[0]], pd.Period("1975-01"))
        self.assertEqual(months[first.test_indices[0]], pd.Period("1987-01"))
        self.assertEqual(months[last.train_indices[-1]], pd.Period("2003-12"))
        self.assertEqual(months[last.validation_indices[0]], pd.Period("2004-01"))
        self.assertEqual(months[last.validation_indices[-1]], pd.Period("2015-12"))
        for split in splits:
            self.assertEqual(len(split.validation_indices), 144)
            self.assertEqual(len(split.test_indices), 12)
            self.assertLess(months[split.train_indices].max(), months[split.validation_indices].min())
            self.assertLess(months[split.validation_indices].max(), months[split.test_indices].min())

    def test_unsorted_duplicate_months_preserve_row_order(self):
        dates = pd.to_datetime(["1987-01-31", "1974-12-31", "1975-01-31", "1987-01-31", "1957-03-31"])
        split = next(annual_splits(dates, last_test_year=1987))
        assert_array_equal(split.train_indices, [1, 4])
        assert_array_equal(split.validation_indices, [2])
        assert_array_equal(split.test_indices, [0, 3])

    def test_reject_invalid_or_missing_dates_and_windows(self):
        for dates in ([195703, 198701], ["1957-03-31", None], [], pd.period_range("1957", periods=3, freq="Y")):
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                list(annual_splits(dates, last_test_year=1987))
        with self.assertRaises(ValueError):
            list(annual_splits(["1987-01-31"], last_test_year=1987))


class HyperparameterTuningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.X = rng.normal(size=(120, 5))
        self.y = self.X[:, 0] * .2 + .1
        self.X_validation = rng.normal(size=(40, 5)) + 10
        self.y_validation = self.X_validation[:, 0] * .2 + .1

    def test_scores_choose_best_and_do_not_refit_validation(self):
        candidates = [{"penalty_strength": .5}, {"penalty_strength": 0.}]
        result = tune_hyperparameters(LinearRegressor, candidates, self.X, self.y, self.X_validation, self.y_validation)
        self.assertEqual(result.best_params, candidates[1])
        assert_allclose(result.best_estimator.feature_offset_, self.X.mean(axis=0))
        reference = LinearRegressor().fit(self.X, self.y)
        assert_allclose(result.best_estimator.coef_, reference.coef_)
        for record in result.validation_results:
            model = LinearRegressor(**record["params"]).fit(self.X, self.y)
            self.assertAlmostEqual(record["validation_loss"], np.mean((model.predict(self.X_validation) - self.y_validation)**2))

    def test_first_exact_tie_wins(self):
        # rho has no effect when lambda is zero, making an exact objective tie.
        result = tune_hyperparameters(LinearRegressor, [{"rho": 0.}, {"rho": 1.}], self.X, self.y, self.X_validation, self.y_validation)
        self.assertEqual(result.best_params, {"rho": 0.})

    def test_pcr_candidates_and_explicit_huber_scoring(self):
        validation_loss_fn = partial(huber_loss, delta=.1)
        result = tune_hyperparameters(PCRRegressor, [{"n_components": 1}, {"n_components": 3}],
                                      self.X, self.y, self.X_validation, self.y_validation,
                                      validation_loss_fn=validation_loss_fn)
        best_score = validation_loss_fn(self.y_validation, result.best_estimator.predict(self.X_validation))
        self.assertAlmostEqual(best_score, min(r["validation_loss"] for r in result.validation_results))

    def test_empty_grid_invalid_score_and_unconverged_candidate(self):
        with self.assertRaises(ValueError):
            tune_hyperparameters(LinearRegressor, [], self.X, self.y, self.X_validation, self.y_validation)
        with self.assertRaises(ValueError):
            tune_hyperparameters(LinearRegressor, [{}], self.X, self.y, self.X_validation, self.y_validation,
                                 validation_loss_fn="huber")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(RuntimeError):
                tune_hyperparameters(LinearRegressor, [{"penalty_strength": .01, "max_iter": 1, "initial_step": 1e-5}],
                                      self.X, self.y, self.X_validation, self.y_validation)

    def test_custom_loss_receives_targets_first_and_controls_selection(self):
        calls = []

        def asymmetric_loss(y_true, y_pred):
            assert_array_equal(y_true, self.y_validation)
            calls.append(y_pred.copy())
            residuals = y_pred - y_true
            return float(np.mean(np.where(residuals > 0, 3 * residuals, -residuals)))

        candidates = [{"penalty_strength": .1}, {"penalty_strength": 0.}]
        result = tune_hyperparameters(LinearRegressor, candidates, self.X, self.y,
                                      self.X_validation, self.y_validation,
                                      validation_loss_fn=asymmetric_loss)
        self.assertEqual(len(calls), len(candidates))
        expected_scores = [np.mean(np.where(prediction > self.y_validation,
                                            3 * (prediction - self.y_validation),
                                            self.y_validation - prediction)) for prediction in calls]
        assert_allclose([r["validation_loss"] for r in result.validation_results], expected_scores)
        self.assertEqual(result.best_params, candidates[int(np.argmin(expected_scores))])

    def test_custom_loss_rejects_nonscalar_or_nonfinite_scores(self):
        for invalid in ([1.], np.array([1., 2.]), 1j, True, "1", None, np.nan, np.inf):
            with self.subTest(score=invalid), self.assertRaises((ValueError, FloatingPointError)):
                tune_hyperparameters(LinearRegressor, [{}], self.X, self.y,
                                     self.X_validation, self.y_validation,
                                     validation_loss_fn=lambda observed, predicted: invalid)


if __name__ == "__main__":
    unittest.main()
