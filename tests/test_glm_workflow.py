import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import test_batch_workflow as fixtures
from core.data_examine.glm_matlab import calculate, make_fixture
from core.dense_glm_run import checked_fit
from core.dense_parquet_run import run_parquet
from core.models.glm import SplineBasis, fit_group_loss
from core.replication_report import make_report, report_tables
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


class GLMModelTests(unittest.TestCase):
    def test_training_basis_matches_verified_reference(self):
        f = make_fixture()
        reference, _ = calculate(f)
        basis = SplineBasis()
        assert_allclose(basis.fit_transform(f['X_train']), reference['Z_train'], atol=2e-14)
        assert_allclose(basis.knots_, reference['knots'], atol=1e-14)
        # Empirical prediction is invariant to the other observations supplied.
        X = f['X_test']
        assert_allclose(basis.transform(X)[:1], basis.transform(X[:1]), atol=1e-14)
        assert_allclose(basis.transform(X), np.vstack([basis.transform(X[:50]), basis.transform(X[50:])]))

    def test_explicit_industry_singleton(self):
        X = np.column_stack([np.arange(20.) / 20, np.arange(20) % 2])
        basis = SplineBasis()
        Z = basis.fit_transform(X, [1])
        self.assertEqual(basis.group_sizes_, [5, 1])
        self.assertEqual(Z.shape, (20, 6))
        assert_allclose(Z.mean(axis=0), 0, atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'binary'):
            basis.transform(np.array([[.1, 2.]]))

    def test_both_losses_against_independent_constrained_solver(self):
        rng = np.random.default_rng(43)
        X = rng.normal(size=(90, 6))
        y = .3 * X[:, 0] + rng.normal(0, .04, 90)
        y[::10] += 1
        for delta in (None, .2):
            strength = .005
            fit = fit_group_loss(X, y, [5, 1], strength, delta=delta, tol=1e-10)
            def objective(v):
                r = abs(X @ v[:6] - y)
                loss = .5 * r**2 if delta is None else np.where(r <= delta, .5*r**2, delta*r-.5*delta**2)
                return loss.mean() + strength * v[6:].sum()
            reference = minimize(objective, np.r_[np.zeros(6), 1, 1], method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda v: v[6:] - [np.linalg.norm(v[:5]), abs(v[5])]},
                options={'ftol': 1e-12, 'maxiter': 2000})
            self.assertTrue(reference.success, reference.message)
            self.assertTrue(fit.converged)
            assert_allclose(X @ fit.coefficients, X @ reference.x[:6], atol=2e-5)

    def test_nonconvergence_rejected(self):
        rng = np.random.default_rng(44)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, 'did not converge'):
            checked_fit(rng.normal(size=(40, 5)), rng.normal(size=40), [5], .001,
                        label='GLM fixture', max_iter=1, tol=1e-14)


class GLMWorkflowTests(unittest.TestCase):
    setUp = fixtures.BatchWorkflowTests.setUp

    def test_complete_staged_resume_and_replication_reports(self):
        settings = load_config('gkx_glm_huber_v1').RUN.MODELS
        settings.GLM_HUBER.PENALTY_GRID = [.001, .01]
        self.cfg.RUN.MODELS = settings
        self.cfg.RUN.FITTING = 'dense'
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004
        full = create_run(self.cfg, self.panel.info)
        staged = create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()):
            result = run_parquet(self.panel, self.cfg, checkpoint_dir=full)
            run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
            resumed = run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            with patch('core.dense_glm_run.checked_fit', side_effect=AssertionError('must skip')):
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            make_report(full)
        assert_allclose(pd.read_csv(result.predictions_path).prediction, pd.read_csv(resumed.predictions_path).prediction)
        records = read_checkpoints(full, self.cfg)
        table, complexity, curves = report_tables(records, self.cfg)
        self.assertTrue(table.complete_horizon.all())
        self.assertTrue((complexity.active_groups <= 5).all())
        self.assertEqual(len(curves), 4)
        for (year, _), record in records.items():
            tuning = record['tuning']
            self.assertEqual(tuning['selection_stage'], 'squared')
            self.assertEqual(tuning['final_stage'], 'huber')
            self.assertEqual(len(tuning['spline_state']['group_sizes']), 5)
            train = self.y[self.months.year <= year - 2]
            self.assertAlmostEqual(tuning['intercept'], train.mean())


if __name__ == '__main__':
    unittest.main()
