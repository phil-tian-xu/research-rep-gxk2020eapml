"""Independent zero-out, reconstruction and end-to-end reporting checks."""

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from core.data_cleaning.partition_gkx import partition_gkx
from core.data_query.parquet_dataset import ParquetPanel
from core.evaluation.importance import zero_out_importance
from core.figure4_importance import characteristic_groups, streaming_zero_out
from core.figure4_report import make_report, restore_predictor, summarize_and_plot
from core.run_state import dataset_identity, save_checkpoint
from utils.general_utils import load_config


class StreamingImportanceTests(unittest.TestCase):
    def test_grouped_linear_shortcut_matches_literal_zero_out_across_batches(self):
        rng = np.random.default_rng(19)
        X = pd.DataFrame(rng.normal(size=(23, 4)), columns=["a", "a__macro_dp", "b", "sic2_10"])
        beta = np.array([.3, -.2, 0., .1])
        y = rng.normal(size=len(X))
        predict = lambda x: np.asarray(x) @ beta + .15
        groups = characteristic_groups(X.columns)
        reference = zero_out_importance(predict, X, y,
            groups={k: X.columns[v].tolist() for k, v in groups.items()})
        for size in (1, 7, 23):
            batches = ((X.iloc[i:i+size].to_numpy(), y[i:i+size]) for i in range(0, len(X), size))
            actual, diagnostics = streaming_zero_out(predict, batches, n_features=4,
                                                      groups=groups, coefficients=beta)
            assert_allclose(actual.importance, reference, atol=1e-14)
            self.assertEqual(diagnostics["n_train"], 23)
        self.assertEqual(actual.set_index("predictor").loc["b", "importance"], 0)

    def test_nonlinear_predictor_uses_same_evaluator(self):
        X = np.array([[1., 2.], [3., 1.], [-1., 4.]])
        predict = lambda x: x[:, 0] * x[:, 1] + x[:, 1] ** 2
        y = np.array([2., 5., 1.])
        actual, _ = streaming_zero_out(predict, [(X, y)], n_features=2, groups={"a": [0], "b": [1]})
        expected = []
        for j in range(2):
            altered = X.copy()
            altered[:, j] = 0
            expected.append((np.sum((y-predict(altered))**2)-np.sum((y-predict(X))**2))/(y@y))
        assert_allclose(actual.importance, expected)

    def test_rejects_bad_groups_zero_benchmark_and_missing_model_state(self):
        with self.assertRaises(ValueError):
            characteristic_groups(["a__macro_dp"])
        with self.assertRaises(ValueError):
            streaming_zero_out(lambda x: x[:, 0], [(np.ones((2, 1)), np.zeros(2))],
                               n_features=1, groups={"a": [0]})
        with self.assertRaises(ValueError):
            restore_predictor({"specification": "PCR", "best_params": {"n_components": 5}})
        with self.assertRaises(NotImplementedError):
            restore_predictor({"specification": "RF"})

    def test_negative_scores_preserved_and_annual_means_precede_normalization(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            annual = pd.DataFrame({"model": ["TEST"]*4, "year": [2000, 2000, 2001, 2001],
                                   "predictor": ["a", "b"]*2, "importance": [1., 1., 8., 0.]})
            summary = summarize_and_plot(annual, output, "test").set_index("predictor")
            self.assertAlmostEqual(summary.loc["a", "relative_importance"], .9)
            annual.loc[annual.predictor.eq("b"), "importance"] = -1.
            summary = summarize_and_plot(annual, output, "test")
            self.assertTrue(summary.relative_importance.isna().all())
            self.assertTrue((summary.mean_importance < 0).any())


class Figure4IntegrationTests(unittest.TestCase):
    def test_saved_fit_reporting_and_prediction_mismatch_rejection(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            rng = np.random.default_rng(4)
            months = pd.period_range("2000-01", "2003-12", freq="M").repeat(2)
            names = ["a", "a__macro_dp", "b"]
            X = rng.normal(size=(len(months), 3))
            beta, intercept = np.array([.2, .1, -.1]), .03
            y = X @ beta + intercept + rng.normal(scale=.01, size=len(X))
            frame = pd.DataFrame(X, columns=names)
            frame["excess_return"] = y
            frame["return_month"] = months.astype(str)
            frame["feature_month"] = (months-1).astype(str)
            frame["permno"] = np.tile(["a", "b"], len(months)//2)
            frame.to_csv(source / "panel.csv", index=False)
            metadata = dict(schema_version=1, dataset_name="fixture", return_units="decimal",
                id_columns=["permno"], feature_month_column="feature_month", return_month_column="return_month",
                target_column="excess_return", feature_columns=names, audit_columns=[], n_rows=len(frame),
                panel_sha256=hashlib.sha256((source / "panel.csv").read_bytes()).hexdigest())
            (source / "metadata.json").write_text(json.dumps(metadata))
            data = root / "panel"
            partition_gkx(source_dir=source, output_dir=data, chunksize=17)
            panel = ParquetPanel(data)
            cfg = load_config("gkx_enet_huber_v1")
            cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
            cfg.RUN.SPLIT.FIRST_TEST_YEAR = cfg.RUN.SPLIT.LAST_TEST_YEAR = 2003
            cfg.RUN.SPLIT.VALIDATION_YEARS = 1
            run = root / "experiment/annual/years/2003/run"
            run.mkdir(parents=True)
            import yaml
            (run / "resolved_config.yaml").write_text(cfg.dump())
            (run / "manifest.json").write_text(json.dumps(dict(schema_version=1,
                config=yaml.safe_load(cfg.dump()), dataset=dataset_identity(panel.info))))
            test = months.year == 2003
            tuning = dict(specification="ENET_HUBER", test_year=2003, feature_names=names,
                          coefficients=beta.tolist(), intercept=intercept)
            predictions = pd.DataFrame(dict(specification="ENET_HUBER", test_year=2003,
                row_position=np.flatnonzero(test), return_month=months[test].astype(str),
                observed=y[test], prediction=X[test]@beta+intercept))
            metrics = dict(specification="ENET_HUBER", test_year=2003, n_test=int(test.sum()),
                           n_train=int((months.year <= 2001).sum()), r2_zero=.1)
            save_checkpoint(run, metrics, predictions, tuning)
            output = make_report([root / "experiment"], data, root / "output", [2003], batch_size=7)
            result = pd.read_csv(output / "annual_importance.csv")
            train = months.year <= 2001
            reference = zero_out_importance(lambda x: np.asarray(x)@beta+intercept,
                pd.DataFrame(X[train], columns=names), y[train], groups={"a": names[:2], "b": ["b"]})
            assert_allclose(result.importance, reference, atol=1e-12)
            self.assertEqual(json.loads((output / "methodology.json").read_text())["status"], "complete")
            self.assertTrue((output / "enet_huber.png").is_file())
            checkpoint = run / "years/2003/ENET_HUBER.json"
            envelope = json.loads(checkpoint.read_text())
            record = json.loads(envelope["payload"])
            record["tuning"]["intercept"] += 1
            payload = json.dumps(record)
            checkpoint.write_text(json.dumps(dict(payload=payload, sha256=hashlib.sha256(payload.encode()).hexdigest())))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                make_report([root / "experiment"], data, root / "output", [2003])


if __name__ == "__main__":
    unittest.main()
