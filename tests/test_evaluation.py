"""Hand-calculated checks for the replication evaluation API."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from core.evaluation import (
    stock_prediction_summary, zero_out_importance, summarize_importance,
    nonzero_count, selected_component_count, average_tree_depth,
    distinct_split_count, diebold_mariano_test,
)
from core.models import LinearRegressor


class StockSummaryTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "stock_id": [1, 2, 1, 2], "return_month": ["2000-01"] * 2 + ["2000-02"] * 2,
            "market_equity": [20., 10., 5., 30.],
            "observed": [1., 2., 3., 4.], "prediction": [0., 2., 3., 0.],
        })

    def test_monthly_membership_and_pooled_denominators(self):
        original = self.panel.copy()
        result = stock_prediction_summary(self.panel, size_count=1)
        self.assertAlmostEqual(result.loc["all", "r2_zero"], 1 - 17 / 30)
        self.assertEqual(result.loc["top", "r2_zero"], 0.)
        self.assertEqual(result.loc["bottom", "r2_zero"], 1.)
        self.assertEqual(result.loc["top", "benchmark_sse"], 17.)
        self.assertEqual(result.loc["bottom", "n_obs"], 2)
        self.assertAlmostEqual(result.loc["all", "r2_percent"], 100 * (1 - 17 / 30))
        self.assertNotAlmostEqual(result.loc["all", "r2_zero"], (.8 + .36) / 2)
        assert_frame_equal(self.panel, original)

    def test_ties_are_independent_of_row_order_and_small_samples_overlap(self):
        self.panel["market_equity"] = 1.
        first = stock_prediction_summary(self.panel, size_count=1)
        shuffled = stock_prediction_summary(self.panel.iloc[::-1], size_count=1)
        assert_frame_equal(first, shuffled)
        self.assertEqual(first.loc["top", "benchmark_sse"], 10.)
        small = stock_prediction_summary(self.panel)
        self.assertEqual(small.loc["top", "n_obs"], 4)

    def test_missing_duplicate_invalid_equity_and_zero_denominator(self):
        for column, value in (("observed", np.nan), ("market_equity", 0.),
                              ("market_equity", np.inf)):
            bad = self.panel.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                stock_prediction_summary(bad)
        duplicate = pd.concat([self.panel, self.panel.iloc[:1]])
        with self.assertRaises(ValueError):
            stock_prediction_summary(duplicate)
        self.panel["observed"] = 0.
        self.assertTrue(stock_prediction_summary(self.panel).r2_zero.isna().all())


class ImportanceTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"a": [1., 2., 3.], "b": [2., 1., 4.]}, index=[7, 3, 5])
        self.y = self.X.a + self.X.b

    def test_group_zeroing_hand_values_and_batch_invariance(self):
        def predict(frame):
            return frame.a + frame.b
        groups = {"a": ["a"], "b": ["b"], "both": ["a", "b"]}
        original = self.X.copy()
        scores = zero_out_importance(predict, self.X, self.y, groups=groups, batch_size=1)
        assert_allclose(scores, [14 / 67, 21 / 67, 1.])
        self.assertNotAlmostEqual(scores["both"], scores["a"] + scores["b"])
        assert_series_equal(scores, zero_out_importance(predict, self.X, self.y,
                                                       groups=groups, batch_size=10))
        assert_frame_equal(self.X, original)

    def test_negative_scores_and_zero_targets(self):
        scores = zero_out_importance(lambda x: 2 * x.a, self.X, self.X.a * .1,
                                     groups={"a": ["a"]})
        self.assertLess(scores["a"], 0.)
        zeros = zero_out_importance(lambda x: x.a, self.X, self.y * 0,
                                    groups={"a": ["a"]})
        self.assertTrue(zeros.isna().all())

    def test_existing_estimator_preserves_fitted_state(self):
        model = LinearRegressor().fit(self.X, self.y)
        coefficients, intercept = model.coef_.copy(), model.intercept_
        scores = zero_out_importance(model.predict, self.X, self.y,
                                     groups={"a": ["a"]}, batch_size=2)
        self.assertAlmostEqual(scores["a"], 14 / 67)
        assert_allclose(model.coef_, coefficients)
        self.assertEqual(model.intercept_, intercept)

    def test_alignment_groups_and_prediction_contract(self):
        with self.assertRaises(ValueError):
            zero_out_importance(lambda x: x.a, self.X, self.y.iloc[::-1], groups={"a": ["a"]})
        for groups in ({}, {"a": []}, {"a": ["missing"]}, {"a": "a"}):
            with self.assertRaises(ValueError):
                zero_out_importance(lambda x: x.a, self.X, self.y, groups=groups)
        for predict in (lambda x: np.zeros((len(x), 1)), lambda x: x.a.iloc[::-1]):
            with self.assertRaises(ValueError):
                zero_out_importance(predict, self.X, self.y, groups={"a": ["a"]})

    def test_nonlinear_predictor_group_changes_actual_prediction(self):
        y = self.X.a * self.X.b
        result = zero_out_importance(lambda x: x.a * x.b, self.X, y,
                                     groups={"a": ["a"], "b": ["b"]})
        assert_allclose(result, [1., 1.])


class ImportanceSummaryTests(unittest.TestCase):
    def setUp(self):
        # B wins two years by a little, while A wins one year by a lot.
        self.annual = pd.DataFrame([
            (model, year, predictor, value)
            for model in ("m1", "m2")
            for year, values in enumerate(((100., 0.), (0., 1.), (0., 1.)))
            for predictor, value in zip(("a", "b"), values)
        ], columns=["model", "year", "predictor", "importance"])

    def test_average_ranks_differ_from_ranking_average_scores(self):
        result = summarize_importance(self.annual, negative_policy="raise")
        a = result.query("model == 'm1' and predictor == 'a'").iloc[0]
        b = result.query("model == 'm1' and predictor == 'b'").iloc[0]
        self.assertAlmostEqual(a.relative_importance, 100 / 102)
        self.assertAlmostEqual(a.mean_rank, 5 / 3)
        self.assertAlmostEqual(b.mean_rank, 4 / 3)
        self.assertAlmostEqual(b.overall_rank_sum, 8 / 3)
        assert_allclose(result.groupby("model").relative_importance.sum(), [1., 1.])

    def test_ties_zero_totals_and_explicit_negative_policy(self):
        self.annual["importance"] = 0.
        result = summarize_importance(self.annual, negative_policy="raise")
        self.assertTrue(result.relative_importance.isna().all())
        assert_allclose(result.mean_rank, 1.5)
        self.annual.loc[self.annual.predictor == "a", "importance"] = -1.
        with self.assertRaises(ValueError):
            summarize_importance(self.annual, negative_policy="raise")
        result = summarize_importance(self.annual, negative_policy="clip")
        self.assertEqual(result.mean_importance.min(), -1.)
        self.assertTrue(result.relative_importance.isna().all())

    def test_incomplete_panel_and_duplicates_fail(self):
        for bad in (self.annual.iloc[1:], pd.concat([self.annual, self.annual.iloc[:1]])):
            with self.assertRaises(ValueError):
                summarize_importance(bad, negative_policy="raise")


class ComplexityTests(unittest.TestCase):
    def test_counts_grouping_and_realized_components(self):
        self.assertEqual(nonzero_count([0., 1., 1e-14], tolerance=1e-10), 1)
        self.assertEqual(nonzero_count([0., 1., 2.], groups={"a": [0, 1], "b": [2]}), 2)
        self.assertEqual(selected_component_count(SimpleNamespace(n_components=10, n_components_=2)), 2)
        self.assertEqual(selected_component_count(SimpleNamespace(n_components_=0)), 0)
        self.assertEqual(average_tree_depth([0, 2, 4]), 2.)
        self.assertEqual(distinct_split_count(["a", "a", "a_x", "b"]), 3)
        self.assertEqual(distinct_split_count(["a", "a_x"],
                                              feature_groups={"a": "a", "a_x": "a"}), 1)
        self.assertEqual(distinct_split_count([]), 0)

    def test_invalid_state_and_deferred_test_fail_explicitly(self):
        with self.assertRaises(ValueError):
            selected_component_count(SimpleNamespace(n_components=3))
        with self.assertRaises(ValueError):
            average_tree_depth([1.5])
        with self.assertRaises(ValueError):
            nonzero_count([1.], groups={"bad": [-1]})
        with self.assertRaises(NotImplementedError):
            diebold_mariano_test([1.], [1.], [0.], ["2000-01"], max_lag=0)


if __name__ == "__main__":
    unittest.main()
