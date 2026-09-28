from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from core.batch_run import run_parquet
from core.data_cleaning.partition_gkx import partition_gkx
from core.data_query.parquet_dataset import PanelBatch, ParquetPanel
from core.models import LinearRegressor, PCRRegressor, PLSRegressor
from core.models.batched import reduce_training, fit_components, fit_huber
from core.replication_report import report_tables
from core.run import run
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


class ArrayWindow:
    """Small reference fixture only; production windows always read Parquet."""
    def __init__(self, X, y, batch_size):
        self.X, self.y, self.n_rows, self.batch_size = X, y, len(y), batch_size
        self.panel = SimpleNamespace(features=list(range(X.shape[1])))

    def batches(self):
        for start in range(0, self.n_rows, self.batch_size):
            end = start + self.batch_size
            yield PanelBatch(self.X[start:end], self.y[start:end], None, None)


class BatchModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(519)
        self.X = rng.normal(size=(107, 5)) * [1, 2, 3, 4, 5] + 2
        self.X = np.column_stack((self.X, self.X[:, 0], np.full(107, .1)))
        self.y = .03 + self.X[:, 0] * .04 + rng.normal(scale=.02, size=107)
        self.y[::15] += .7

    def test_components_match_dense_across_batch_sizes_and_transforms(self):
        for intercept in (True, False):
            for standardize in (True, False):
                for size in (3, 31, 200):
                    with self.subTest(intercept=intercept, standardize=standardize, size=size), redirect_stdout(StringIO()):
                        window = ArrayWindow(self.X, self.y, size)
                        reduction = reduce_training(window, fit_intercept=intercept, standardize=standardize)
                        for name, cls in (("PCR", PCRRegressor), ("PLS", PLSRegressor)):
                            for k in (0, 1, 3, 5):
                                params = {"n_components": k, "fit_intercept": intercept, "standardize": standardize}
                                if name == "PLS":
                                    params["tol"] = 1e-12
                                dense = cls(**params).fit(self.X, self.y)
                                batched = fit_components(name, params, reduction)
                                assert_allclose(batched.predict(self.X), dense.predict(self.X), atol=2e-12, rtol=1e-10)
                                assert_allclose(batched.predict(self.X + .2), dense.predict(self.X + .2), atol=2e-12)
                                self.assertEqual(batched.n_components_, dense.n_components_)

    def test_huber_fixed_and_quantile_match_dense_with_rank_deficiency(self):
        for intercept in (True, False):
            for standardize in (True, False):
                for quantile in (None, .999):
                    with self.subTest(intercept=intercept, standardize=standardize, quantile=quantile), redirect_stdout(StringIO()):
                        window = ArrayWindow(self.X, self.y, 19)
                        reduction = reduce_training(window, fit_intercept=intercept, standardize=standardize)
                        params = {"loss": "huber", "solver": "svd_apg", "huber_quantile": quantile,
                                  "huber_delta": .08 if quantile is None else None, "tol": 1e-10,
                                  "fit_intercept": intercept, "standardize": standardize}
                        dense = LinearRegressor(**params).fit(self.X, self.y)
                        batched = fit_huber(params, reduction, window)
                        self.assertTrue(batched.converged_)
                        self.assertAlmostEqual(batched.huber_delta_, dense.huber_delta_, places=12)
                        self.assertAlmostEqual(batched.objective_, dense.objective_, places=12)
                        assert_allclose(batched.predict(self.X), dense.predict(self.X), atol=1e-7)

    def test_small_n_large_p_and_numerical_rank_cutoff(self):
        rng = np.random.default_rng(902)
        X, y = rng.normal(size=(8, 15)), rng.normal(size=8)
        with redirect_stdout(StringIO()):
            reduction = reduce_training(ArrayWindow(X, y, 2))
        for name, cls in (("PCR", PCRRegressor), ("PLS", PLSRegressor)):
            params = {"n_components": 6, "fit_intercept": True, "standardize": False}
            if name == "PLS":
                params["tol"] = 1e-12
            assert_allclose(fit_components(name, params, reduction).predict(X), cls(**params).fit(X, y).predict(X), atol=1e-11)
        with self.assertRaises(ValueError):
            fit_components("PCR", {"n_components": 8}, reduction)
        X = np.column_stack((np.arange(1000.), np.arange(1000.) * 1e-14))
        with redirect_stdout(StringIO()):
            reduction = reduce_training(ArrayWindow(X, np.sin(X[:, 0]), 13))
        self.assertEqual(reduction.rank, 1)
        with self.assertRaisesRegex(ValueError, "numerical rank"):
            fit_components("PCR", {"n_components": 2}, reduction)

    def test_constant_target_and_zero_design(self):
        X, y = np.full((35, 3), .1), np.full(35, .1)
        with redirect_stdout(StringIO()):
            reduction = reduce_training(ArrayWindow(X, y, 9))
        model = fit_components("PLS", {"n_components": 3, "tol": 1e-12}, reduction)
        self.assertEqual(model.n_components_, 0)
        assert_allclose(model.predict(X), y)
        for intercept in (True, False):
            window = ArrayWindow(np.zeros_like(X), np.arange(len(X), dtype=float), 11)
            with redirect_stdout(StringIO()):
                reduction = reduce_training(window, fit_intercept=intercept)
                model = fit_huber({"loss": "huber", "solver": "svd_apg", "huber_delta": 1.,
                                   "fit_intercept": intercept}, reduction, window)
            self.assertTrue(model.converged_)
            assert_allclose(model.predict(window.X), 17. if intercept else 0.)


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        rng = np.random.default_rng(75)
        months = pd.period_range("2000-01", "2005-12", freq="M").repeat(3)
        self.X = rng.normal(size=(len(months), 5))
        self.y = .02 + .1 * self.X[:, 0] + rng.normal(scale=.03, size=len(months))
        self.y[::21] += .5
        self.months = months
        frame = pd.DataFrame(self.X, columns=[f"x{i}" for i in range(5)])
        frame["excess_return"] = self.y
        frame["return_month"], frame["feature_month"] = months.astype(str), (months - 1).astype(str)
        frame["permno"] = np.tile(["a", "b", "c"], len(months) // 3)
        frame.to_csv(source / "panel.csv", index=False)
        metadata = {"schema_version": 1, "dataset_name": "fixture", "return_units": "decimal",
                    "id_columns": ["permno"], "feature_month_column": "feature_month", "return_month_column": "return_month",
                    "target_column": "excess_return", "feature_columns": list(frame.columns[:5]),
                    "audit_columns": [], "n_rows": len(frame),
                    "panel_sha256": hashlib.sha256((source / "panel.csv").read_bytes()).hexdigest()}
        (source / "metadata.json").write_text(json.dumps(metadata))
        with redirect_stdout(StringIO()):
            output = partition_gkx(source_dir=source, output_dir=self.root / "fixture_parquet", chunksize=29)
        self.panel = ParquetPanel(output, batch_size=11)
        self.cfg = load_config("gkx_table1_figure3_v4")
        self.cfg.DATA.DATASET = self.panel.info["name"]
        self.cfg.DATA.BATCH_SIZE = 11
        self.cfg.RUN.OUTPUT_DIR = str(self.root / "runs")
        self.cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
        self.cfg.RUN.SPLIT.FIRST_TEST_YEAR = 2003
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2005
        self.cfg.RUN.SPLIT.VALIDATION_YEARS = 1
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = [0, 1, 2, 3]
        self.cfg.RUN.MODELS.PLS.COMPONENT_GRID = [0, 1, 2, 3]

    def test_reader_projection_windows_and_values(self):
        with patch("pyarrow.parquet.read_table", side_effect=AssertionError("Full read forbidden")):
            batches = list(self.panel.window(2001, 2002).batches())
        self.assertTrue(all(len(b.y) <= 11 and b.X.dtype == np.float64 for b in batches))
        selected = (self.months.year >= 2001) & (self.months.year <= 2002)
        assert_array_equal(np.concatenate([b.X for b in batches]), self.X[selected])
        assert_array_equal(np.concatenate([b.row_position for b in batches]), np.flatnonzero(selected))
        self.assertTrue(all(b.X is None for b in self.panel.window(2001, 2002).batches(features=False)))
        read_years = {int(path.parent.name.split("=")[1]) for path in self.panel._verified}
        self.assertEqual(read_years, {2001, 2002})

    def test_corrupt_part_and_incomplete_manifest_are_rejected(self):
        part = next(p for p in self.panel.metadata["parts"] if p["return_year"] == 2001)
        with (self.panel.folder / part["path"]).open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            list(self.panel.window(2001, 2001).batches())
        manifest = self.panel.metadata.copy()
        manifest["status"] = "incomplete"
        (self.panel.folder / "metadata.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "complete"):
            ParquetPanel(self.panel.folder)

    def test_dense_staged_and_uninterrupted_results_agree(self):
        full = create_run(self.cfg, self.panel.info)
        staged = create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()):
            dense = run(self.X, self.y, self.months, self.cfg)
            complete = run_parquet(self.panel, self.cfg, checkpoint_dir=full)
            run_parquet(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
            original_reduce = reduce_training
            with patch("core.batch_run.reduce_training", wraps=original_reduce) as fitting:
                resumed = run_parquet(self.panel, self.cfg, checkpoint_dir=staged)
                self.assertEqual(fitting.call_count, 2)
        pd.testing.assert_frame_equal(complete.annual_metrics, resumed.annual_metrics)
        pd.testing.assert_frame_equal(complete.summary, resumed.summary)
        pd.testing.assert_frame_equal(pd.read_csv(complete.predictions_path), pd.read_csv(resumed.predictions_path))
        actual = pd.read_csv(complete.predictions_path)
        assert_allclose(actual.prediction, dense.predictions.prediction, atol=2e-7)
        assert_allclose(complete.annual_metrics.best_validation_loss, dense.annual_metrics.best_validation_loss, atol=1e-9)
        assert_allclose(complete.summary.r2_zero, dense.summary.r2_zero, atol=1e-7)
        full_records, lazy = read_checkpoints(full, self.cfg), read_checkpoints(full, self.cfg, lazy=True)
        expected_tables = report_tables(full_records, self.cfg)
        with patch("core.replication_report.pd.concat", side_effect=AssertionError("No full panel concat")):
            for expected, actual in zip(expected_tables, report_tables(lazy, self.cfg)):
                pd.testing.assert_frame_equal(expected, actual)
        for tuning in dense.tuning_history:
            self.assertEqual(tuning["best_params"], lazy[tuning["test_year"], tuning["specification"]]["tuning"]["best_params"])

    def test_complete_array_pls_matches_original_and_resumes(self):
        from core.dense_parquet_run import run_parquet as dense_run

        self.cfg.RUN.FITTING = "dense"
        self.cfg.RUN.MODELS.ENABLED = ["PLS"]
        full = create_run(self.cfg, self.panel.info)
        staged = create_run(self.cfg, self.panel.info)
        original_fit = PLSRegressor.fit
        fits = []

        def fitted(model, X, y):
            fits.append((len(y), model.n_components))
            return original_fit(model, X, y)

        with redirect_stdout(StringIO()), patch("core.batch_run.reduce_training", side_effect=AssertionError("No QR fitting")):
            reference = run(self.X, self.y, self.months, self.cfg)
            complete = dense_run(self.panel, self.cfg, checkpoint_dir=full)
            dense_run(self.panel, self.cfg, checkpoint_dir=staged, stop_after_year=2003)
            with patch.object(PLSRegressor, "fit", new=fitted):
                resumed = dense_run(self.panel, self.cfg, checkpoint_dir=staged)
        self.assertEqual(fits, [(n, k) for n in (108, 144) for k in range(4)])
        pd.testing.assert_frame_equal(complete.annual_metrics, resumed.annual_metrics)
        pd.testing.assert_frame_equal(pd.read_csv(complete.predictions_path), pd.read_csv(resumed.predictions_path))
        assert_allclose(pd.read_csv(complete.predictions_path).prediction,
                        reference.predictions.prediction, atol=1e-14, rtol=1e-12)
        assert_allclose(complete.summary.r2_zero, reference.summary.r2_zero, atol=1e-14)
        saved = read_checkpoints(full, self.cfg)
        for tuning in reference.tuning_history:
            actual = saved[tuning["test_year"], "PLS"]["tuning"]
            from core.figure4_report import restore_predictor
            predict, names, _ = restore_predictor(actual)
            record = saved[tuning["test_year"], "PLS"]
            positions = np.asarray(record["predictions"]["row_position"])
            assert_allclose(predict(self.X[positions]), record["predictions"]["prediction"], atol=1e-14)
            self.assertEqual(names, list(self.panel.features))
            self.assertEqual(actual["best_params"], tuning["best_params"])
            for a, b in zip(actual["validation_results"], tuning["validation_results"]):
                self.assertAlmostEqual(a["validation_loss"], b["validation_loss"], places=14)
                self.assertEqual(a["n_components"], b["n_components"])

    def test_failure_keeps_completed_models_and_resume_skips_them(self):
        folder = create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()), patch("core.batch_run.fit_huber", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                run_parquet(self.panel, self.cfg, checkpoint_dir=folder, stop_after_year=2003)
        self.assertEqual(set(read_checkpoints(folder, self.cfg)), {(2003, "PCR"), (2003, "PLS")})
        with redirect_stdout(StringIO()), patch("core.batch_run.fit_components", side_effect=AssertionError("Already saved")):
            run_parquet(self.panel, self.cfg, checkpoint_dir=folder, stop_after_year=2003)
        self.assertEqual(len(read_checkpoints(folder, self.cfg)), 3)

    def test_main_dispatches_to_batch_reader_without_csv(self):
        from main import main
        with patch("main.load_config", return_value=self.cfg), patch("main.load_dataset", side_effect=AssertionError("CSV path")), \
                patch.dict("core.data_query.datasets.DATASET_PATHS", {self.cfg.DATA.DATASET: self.panel.folder}), \
                patch("core.replication_report.make_report") as report, redirect_stdout(StringIO()):
            result = main("fixture", stop_after_year=2003)
        self.assertEqual(len(result.annual_metrics), 3)
        report.assert_called_once()

    def test_complete_array_pcr_matches_original_and_resumes_with_memory_toggle(self):
        from main import main
        from utils.memory_report import MemoryReport
        self.cfg.RUN.FITTING = "dense"
        self.cfg.RUN.MODELS.ENABLED = ["PCR"]
        original_fit = PCRRegressor.fit
        fitted_shapes = []

        def fit(model, X, y):
            fitted_shapes.append(X.shape)
            return original_fit(model, X, y)

        with redirect_stdout(StringIO()):
            reference = run(self.X, self.y, self.months, self.cfg)
        with patch("main.load_config", return_value=self.cfg), \
                patch("main.load_dataset", side_effect=AssertionError("No CSV loading")), \
                patch.dict("core.data_query.datasets.DATASET_PATHS", {self.cfg.DATA.DATASET: self.panel.folder}), \
                patch("core.models.batched.reduce_training", side_effect=AssertionError("No QR fit")), \
                patch.object(PCRRegressor, "fit", new=fit), redirect_stdout(StringIO()):
            with patch("utils.memory_report.process_memory", side_effect=AssertionError("Disabled")):
                first = main("fixture", stop_after_year=2003)
            folder = next(Path(self.cfg.RUN.OUTPUT_DIR).iterdir())
            self.assertEqual(len(first.annual_metrics), 1)
            self.assertFalse(list(folder.glob("memory_*.json")))
            counters = {"resident_bytes": 10, "peak_resident_bytes": 100, "private_committed_bytes": 20}
            with patch("utils.memory_report.process_memory", return_value=counters):
                resumed = main(resume=folder, memory=True)
            report = json.loads(next(folder.glob("memory_*.json")).read_text())
            self.assertEqual(report["peak_resident_gib"], 100 / 1024**3)
            self.assertEqual(report["stages"][-1]["stage"], "Final measurement")
            complete = main("fixture")
        self.assertEqual(fitted_shapes[:3], [(72, 5), (108, 5), (144, 5)])
        self.assertEqual(len(fitted_shapes), 6)  # Three staged + three uninterrupted fits.
        actual = pd.read_csv(resumed.predictions_path)
        assert_allclose(actual.prediction, reference.predictions.prediction, atol=1e-14)
        assert_allclose(resumed.annual_metrics.best_validation_loss,
                        reference.annual_metrics.best_validation_loss, atol=1e-14)
        pd.testing.assert_frame_equal(resumed.summary, complete.summary)
        pd.testing.assert_frame_equal(actual, pd.read_csv(complete.predictions_path))
        for tuning in reference.tuning_history:
            saved = read_checkpoints(folder, self.cfg)[tuning["test_year"], "PCR"]["tuning"]
            self.assertEqual(saved["best_params"], tuning["best_params"])
            from core.figure4_report import restore_predictor
            predict, names, _ = restore_predictor(saved)
            rows = actual.loc[actual.test_year.eq(tuning["test_year"])]
            assert_allclose(predict(self.X[rows.row_position.to_numpy()]), rows.prediction, atol=1e-14)
            self.assertEqual(names, list(self.panel.features))
        output = folder / "table1_figure3"
        self.assertTrue((output / "figure3_complexity.png").is_file())
        self.assertTrue(pd.read_csv(output / "table1_all.csv").complete_horizon.all())
        self.assertEqual(set(pd.read_csv(output / "complexity.csv").model), {"PCR"})

    def test_complete_array_failure_preserves_prior_year_and_memory_failure_is_nonfatal(self):
        from core.dense_parquet_run import run_parquet as dense_run
        from utils.memory_report import MemoryReport
        self.cfg.RUN.FITTING = "dense"
        self.cfg.RUN.MODELS.ENABLED = ["PCR"]
        folder = create_run(self.cfg, self.panel.info)
        with redirect_stdout(StringIO()):
            dense_run(self.panel, self.cfg, checkpoint_dir=folder, stop_after_year=2003)
            with patch.object(PCRRegressor, "fit", side_effect=MemoryError("simulated")):
                with self.assertRaises(MemoryError):
                    dense_run(self.panel, self.cfg, checkpoint_dir=folder)
            monitor = MemoryReport(enabled=True)
            monitor.folder = folder
            with patch("utils.memory_report.process_memory", side_effect=OSError("unavailable")):
                monitor.snapshot("Final measurement")
        self.assertEqual(set(read_checkpoints(folder, self.cfg)), {(2003, "PCR")})
        self.assertEqual(pd.read_csv(folder / "status.csv").status.tolist(), ["complete", "pending", "pending"])


if __name__ == "__main__":
    unittest.main()
