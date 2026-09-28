import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

import test_batch_workflow as fixtures
from core.data_query.parquet_dataset import load_window
from core.dense_parquet_run import run_parquet
from core.models.random_forest import cpu_budget, diagnostics, make_forest
from core.replication_report import make_report, report_tables
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


class RFModelTests(unittest.TestCase):
    def test_bootstrap_leaf_means_and_average_prediction(self):
        rng = np.random.default_rng(4)
        X, y = rng.normal(size=(80, 4)).astype('float32'), rng.normal(size=80)
        model = make_forest(n_estimators=9, max_depth=3, max_features=2, random_state=7, n_jobs=1).fit(X, y)
        for tree, indices in zip(model.estimators_, model.estimators_samples_):
            self.assertEqual(len(indices), len(y))
            self.assertLess(len(np.unique(indices)), len(y))
            leaves = tree.apply(X[indices])
            for leaf in np.unique(leaves):
                self.assertAlmostEqual(tree.tree_.value[leaf, 0, 0], y[indices][leaves == leaf].mean())
        assert_allclose(model.predict(X), np.mean([tree.predict(X) for tree in model.estimators_], axis=0))
        parallel = make_forest(n_estimators=9, max_depth=3, max_features=2, random_state=7, n_jobs=2).fit(X, y)
        parallel.n_jobs = 1
        assert_array_equal(model.predict(X), parallel.predict(X))

    def test_realized_depth_and_cpu_allocation(self):
        model = make_forest(n_estimators=5, max_depth=6, max_features=1, random_state=9, n_jobs=1)
        model.fit(np.ones((20, 2)), np.arange(20.))
        self.assertEqual(diagnostics(model)['average_tree_depth'], 0.)
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '3', 'OMP_NUM_THREADS': '8'}):
            self.assertEqual(cpu_budget(), 3)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cpu_budget(), 1)


class RFWorkflowTests(unittest.TestCase):
    setUp = fixtures.BatchWorkflowTests.setUp

    def configure(self):
        self.cfg.RUN.MODELS = load_config('gkx_rf_v1').RUN.MODELS
        self.cfg.RUN.MODELS.RF.N_TREES = 7
        self.cfg.RUN.MODELS.RF.DEPTH_GRID = [1, 2]
        self.cfg.RUN.MODELS.RF.MAX_FEATURES_GRID = [1, 3, 5]
        self.cfg.RUN.FITTING = 'dense'
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004

    def test_selection_checkpoints_resume_and_reports(self):
        self.configure()
        full = create_run(self.cfg, self.panel.info)
        staged = create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()), patch.dict(os.environ, {'OMP_NUM_THREADS': '2'}):
            result = run_parquet(self.panel, self.cfg, checkpoint_dir=full)
            run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
            resumed = run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            with patch('core.dense_rf_run.make_forest', side_effect=AssertionError('must skip')):
                run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
            output = make_report(full)
        assert_array_equal(pd.read_csv(result.predictions_path).prediction,
                           pd.read_csv(resumed.predictions_path).prediction)
        self.assertTrue((output / 'figure3_complexity.png').is_file())
        records = read_checkpoints(full, self.cfg)
        table, complexity, curves = report_tables(records, self.cfg)
        self.assertTrue(table.complete_horizon.all())
        self.assertEqual(len(curves), 12)
        self.assertTrue((complexity.average_tree_depth <= complexity.max_depth).all())
        for (year, _), record in records.items():
            tuning = record['tuning']
            X, y, _, _ = load_window(self.panel, 2000, year - 2)
            V, vy, _, _ = load_window(self.panel, year - 1, year - 1)
            scores = []
            for candidate in tuning['validation_results']:
                forest = make_forest(**candidate['params'], n_jobs=1).fit(X.astype('float32'), y)
                score = np.mean((vy - forest.predict(V.astype('float32')))**2)
                self.assertAlmostEqual(candidate['validation_loss'], score)
                scores.append(score)
            self.assertEqual(tuning['best_params'], tuning['validation_results'][np.argmin(scores)]['params'])
            best = make_forest(**tuning['best_params'], n_jobs=1).fit(X.astype('float32'), y)
            T, _, _, _ = load_window(self.panel, year, year)
            assert_allclose(best.predict(T.astype('float32')), record['predictions']['prediction'], atol=1e-15)
        records[2003, 'RF']['tuning']['average_tree_depth'] = 99
        with self.assertRaisesRegex(ValueError, 'depth diagnostics'):
            report_tables(records, self.cfg)

    def test_invalid_grid_rejected_before_loading(self):
        self.configure()
        self.cfg.RUN.MODELS.RF.MAX_FEATURES_GRID = [920]
        folder = create_run(self.cfg, self.panel.info)
        with patch('core.dense_rf_run.load_window', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'max_features'):
                run_parquet(self.panel, self.cfg, checkpoint_dir=folder)


if __name__ == '__main__':
    unittest.main()
