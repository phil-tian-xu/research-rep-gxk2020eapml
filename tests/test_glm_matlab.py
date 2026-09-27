"""Reference math tests; these do NOT claim MATLAB execution/equivalence."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import savemat
from scipy.optimize import minimize

from core.data_examine.glm_matlab import (block_threshold, calculate, compare_arrays,
                                         compare, make_fixture, prepare, read_mat,
                                         reference_fit, spline_basis)


class GLMReferenceTests(unittest.TestCase):
    def test_matlab_quantiles_differ_from_numpy_default(self):
        # Published MathWorks example: seven evenly spaced values.
        values = np.arange(2., 15., 2.)
        assert_allclose(np.quantile(values, [.25, .5, .75], method="hazen"), [4.5, 8, 11.5])
        self.assertFalse(np.allclose(np.quantile(values, [.25, .5, .75]), [4.5, 8, 11.5]))

    def test_hand_calculated_hinges_and_centering(self):
        X = np.array([[-1.], [0.], [1.], [2.]])
        anchors = np.array([[-1.], [0.], [1.], [2.]])
        expected = np.array([[-1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                             [1, 4, 1, 0, 0], [2, 9, 4, 1, 0]], dtype=float)
        assert_allclose(spline_basis(X, anchors), expected - expected.mean(axis=0))
        # Original code re-centers evaluation data, even a single observation.
        assert_allclose(spline_basis(X[:1], anchors), np.zeros((1, 5)))

    def test_group_threshold_analytic_solution(self):
        values = np.array([3., 4., 0, 0, 0, .1, 0, 0, 0, 0])
        assert_allclose(block_threshold(values, 1), [2.4, 3.2, 0, 0, 0, 0, 0, 0, 0, 0])
        beta, info = reference_fit(np.eye(10), values, 1., 1., 1e-12)
        assert_allclose(beta, block_threshold(values, 1))
        self.assertTrue(info["iterate_stop_met"])

    def test_huber_fit_against_independent_epigraph_optimization(self):
        rng = np.random.default_rng(25)
        X = rng.normal(size=(80, 10))
        y = X[:, 0] * .5 + rng.normal(0, .05, 80)
        y[::8] += 2
        strength, delta = .4, .2
        L = np.linalg.norm(X, 2)**2
        beta, info = reference_fit(X, y, strength, L, 1e-18, delta=delta)
        # Independent group-norm epigraph solved by SLSQP, not our prox loop.
        def objective(v):
            residual = abs(X @ v[:10] - y)
            return np.where(residual <= delta, .5 * residual**2,
                            delta * residual - .5 * delta**2).sum() + strength * v[10:].sum()
        reference = minimize(objective, np.r_[np.zeros(10), 1., 1.], method="SLSQP",
                             constraints={"type": "ineq", "fun": lambda v: v[10:] - np.linalg.norm(v[:10].reshape(2, 5), axis=1)},
                             options={"ftol": 1e-12, "maxiter": 2000})
        self.assertTrue(reference.success, reference.message)
        self.assertTrue(info["iterate_stop_met"])
        assert_allclose(X @ beta, X @ reference.x[:10], atol=2e-5)

    def test_nonconvergence_is_reported(self):
        beta, info = reference_fit(np.eye(5), np.ones(5), 0., 1e12, 0., delta=.2)
        self.assertEqual(info["iterations"], 3000)
        self.assertFalse(info["iterate_stop_met"])

    def test_comparison_rejects_missing_nan_shape_and_selection(self):
        expected = {"knots": np.ones((4, 2)), "selected_index": 2, "beta_huber": np.ones(10)}
        self.assertTrue(all(r["passed"] for r in compare_arrays(expected, expected)))
        for actual in [{}, {**expected, "knots": np.full((4, 2), np.nan)},
                       {**expected, "knots": np.ones((2, 4))},
                       {**expected, "selected_index": 3},
                       {**expected, "beta_huber": np.ones(10) + .01}]:
            self.assertFalse(all(r["passed"] for r in compare_arrays(expected, actual)))

    def test_fixture_full_path_is_nontrivial(self):
        result, info = calculate(make_fixture())
        self.assertEqual(result["coefficient_path"].shape, (20, 26))
        self.assertTrue(np.isfinite(result["beta_huber"]).all())
        assert_allclose(result["basis_train"][:, -5:], 0.)
        self.assertGreater(np.linalg.norm(result["beta_huber"] - result["beta_squared"]), 1e-3)
        self.assertTrue(np.any(result["coefficient_path"] == 0))
        self.assertEqual(len(info["squared_candidates"]), 26)

    def test_mat_file_exchange_and_artifact_integrity(self):
        # Synthetic reference ONLY tests the file protocol; no MATLAB claim.
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            run = Path(directory) / "run"
            prepare(run)
            with self.assertRaises(FileExistsError):
                prepare(run)
            with self.assertRaises(FileNotFoundError):
                compare(run)
            actual = read_mat(run / "python_results.mat")
            actual["fixture"] = read_mat(run / "fixture.mat")
            actual["matlab_version"] = "UNIT TEST ONLY"
            actual["author_sources"] = dict.fromkeys(
                ["cut_knots_degree2", "proximal", "proximalH", "f_grad", "f_gradh",
                 "soft_threshodg", "lossh", "fw1"], "unit test placeholder")
            savemat(run / "matlab_results.mat", actual)
            self.assertTrue(compare(run))
            self.assertTrue((run / "comparison.csv").is_file())
            actual["fixture"]["X_test"][0, 0] += 1e-12
            savemat(run / "matlab_results.mat", actual)
            self.assertFalse(compare(run))
            (run / "fixture.mat").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                compare(run)


if __name__ == "__main__":
    unittest.main()
