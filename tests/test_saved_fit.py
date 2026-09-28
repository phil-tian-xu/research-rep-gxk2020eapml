"""Round-trip selected models and independently verify the Figure 5 ranking order."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from core.figure4_importance import characteristic_groups, streaming_zero_out
from core.figure4_report import restore_predictor
from core.figure5_report import rank_importance, make_report
from core.models import PCRRegressor, PLSRegressor
from core.models.glm import SplineBasis
from core.models.random_forest import make_forest
from core.models.saved_fit import linear_state, forest_state


class SavedFitTests(unittest.TestCase):
    def test_scaled_pcr_pls_round_trip_on_unseen_and_zeroed_inputs(self):
        rng = np.random.default_rng(24)
        X = rng.normal(size=(90, 4)) * [1, .001, 100, 3] + [2, .02, 10, -1]
        y = X @ [.2, 20, .003, .1] + rng.normal(scale=.1, size=len(X))
        names = ["a", "a__macro_dp", "b", "c"]
        for cls in (PCRRegressor, PLSRegressor):
            for k in (0, 2):
                with self.subTest(model=cls.__name__, k=k):
                    model = cls(k, standardize=True).fit(X[:60], y[:60])
                    state = json.loads(json.dumps(dict(specification=cls.__name__[:3], **linear_state(model, names))))
                    predict, actual_names, beta = restore_predictor(state)
                    self.assertEqual(names, actual_names)
                    for values in (X[60:], np.zeros_like(X[60:])):
                        assert_allclose(predict(values), model.predict(values), atol=1e-14)
                    scores, _ = streaming_zero_out(predict, [(X[:60], y[:60])], n_features=4,
                        groups=characteristic_groups(names), coefficients=beta)
                    self.assertEqual(len(scores), 3)
                    if k == 0:
                        assert_allclose(scores.importance, 0, atol=0)

    def test_forest_round_trip_matches_sklearn_including_zero_out(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(70, 3))
        y = X[:, 0] ** 2 + X[:, 1] * X[:, 2]
        model = make_forest(n_estimators=7, max_depth=3, max_features=2, random_state=1, n_jobs=1).fit(X, y)
        state = json.loads(json.dumps(dict(specification="RF", feature_names=list("abc"), forest_state=forest_state(model))))
        predict, _, beta = restore_predictor(state)
        self.assertIsNone(beta)
        for values in (X, X + .4, np.zeros_like(X)):
            assert_allclose(predict(values), model.predict(values), atol=0, rtol=0)
        state["forest_state"]["trees"][0]["children_left"][0] = 0
        with self.assertRaises(ValueError):
            restore_predictor(state)

    def test_spline_round_trip_zeroes_original_inputs_before_transform(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(30, 3))
        X[:, 2] = rng.integers(0, 2, len(X))
        basis = SplineBasis()
        Z = basis.fit_transform(X, [2])
        beta = rng.normal(size=Z.shape[1])
        state = json.loads(json.dumps(dict(specification="GLM_HUBER", feature_names=["a", "b", "sic2_10"],
            coefficients=beta.tolist(), intercept=.1, spline_state=basis.state(), converged=True)))
        predict, _, shortcut = restore_predictor(state)
        self.assertIsNone(shortcut)
        for values in (X, np.zeros_like(X)):
            assert_allclose(predict(values), basis.transform(values) @ beta + .1, atol=1e-14)


class Figure5Tests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(model=["PLS"] * 9, year=np.repeat([2000, 2001, 2002], 3),
            predictor=["a", "b", "c"] * 3, importance=[100., 2., -1., 0., 3., -1., 0., 3., -1.]))

    def test_rank_before_averaging_and_preserve_negatives_and_ties(self):
        annual, summary, matrix = rank_importance(self.fixture())
        # a has the highest mean score, but b wins in two out of three years.
        self.assertEqual(list(matrix.index), ["b", "a", "c"])
        self.assertAlmostEqual(matrix.loc["b", "PLS"], 4 / 3)
        tied = self.fixture()
        tied["importance"] = 0.
        annual, _, matrix = rank_importance(tied)
        assert_allclose(annual.annual_rank, 2.)
        self.assertEqual(list(matrix.index), ["a", "b", "c"])

    def test_reject_mismatched_coverage_and_duplicates(self):
        first = self.fixture()
        second = first.assign(model="PCR")
        for frame in (pd.concat([first, second.iloc[:-1]]), pd.concat([first, second.iloc[3:]]),
                      pd.concat([first, first])):
            with self.assertRaises(ValueError):
                rank_importance(frame)

    def test_report_writes_auditable_ranks_and_plot(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "figure4"
            folder.mkdir()
            self.fixture().to_csv(folder / "annual_importance.csv", index=False)
            (folder / "methodology.json").write_text(json.dumps(dict(status="complete", dataset={"name": "test"},
                grouping="test", author_grouping_verified=False, years=[2000, 2001, 2002])))
            output = make_report([folder], root / "output")
            self.assertTrue((output / "characteristic_ranks.png").is_file())
            self.assertEqual(pd.read_csv(output / "rank_matrix.csv").predictor.tolist(), ["b", "a", "c"])
            self.assertEqual(json.loads((output / "methodology.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
