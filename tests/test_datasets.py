from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from core.data_query.datasets import DATASET_PATHS, load_dataset
from core.data_query.generate_experimental_data import generate_panel, save_generated_dataset
from core.run import run, save_run
from main import main
from utils.general_utils import load_config


class SavedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "linear"
        self.name = "synthetic_linear_v1"
        self.settings = {"start_month": "2000-01", "end_month": "2004-12",
                         "stocks_per_month": 2, "coefficients": [.1, -.05, .01]}
        save_generated_dataset(self.folder, dataset_name=self.name, **self.settings)
        mapping = patch.dict(DATASET_PATHS, {self.name: self.folder})
        mapping.start()
        self.addCleanup(mapping.stop)
        self.cfg = load_config("model_demo")
        self.cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
        self.cfg.RUN.SPLIT.FIRST_TEST_YEAR = 2003
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004
        self.cfg.RUN.SPLIT.VALIDATION_YEARS = 1
        self.cfg.RUN.MODELS.ENABLED = ["OLS"]

    def rewrite_panel(self, panel):
        path = self.folder / "panel.csv"
        panel.to_csv(path, index=False)
        metadata_path = self.folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["n_rows"] = len(panel)
        metadata["panel_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_round_trip_keeps_values_dates_and_excludes_identifiers(self):
        expected, _ = generate_panel(**self.settings)
        X, y, months, info = load_dataset(self.name)
        self.assertEqual(list(X.columns), ["x0", "x1", "x2"])
        assert_allclose(X, expected[X.columns], rtol=1e-15, atol=1e-16)
        assert_allclose(y, expected.excess_return, rtol=1e-15, atol=1e-16)
        assert_array_equal(months, pd.PeriodIndex(expected.return_month, freq="M"))
        self.assertEqual(info["name"], self.name)
        self.assertEqual(info["metadata"]["generation"]["seed"], 2020)

    def test_feature_order_is_declared_by_metadata(self):
        path = self.folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["feature_columns"] = ["x2", "x0", "x1"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        X, _, _, _ = load_dataset(self.name)
        self.assertEqual(list(X.columns), ["x2", "x0", "x1"])

    def test_overwrite_refused_and_missing_data_never_generated(self):
        before = (self.folder / "panel.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            save_generated_dataset(self.folder, dataset_name=self.name, seed=999, **self.settings)
        self.assertEqual(before, (self.folder / "panel.csv").read_bytes())
        missing = Path(self.temporary.name) / "not_generated"
        with patch.dict(DATASET_PATHS, {self.name: missing}), self.assertRaises(FileNotFoundError):
            load_dataset(self.name)
        self.assertFalse(missing.exists())
        with self.assertRaises(ValueError):
            load_dataset("not_a_dataset")

    def test_changed_file_rejected_by_hash(self):
        with (self.folder / "panel.csv").open("a") as output:
            output.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_dataset(self.name)

    def test_duplicate_assets_dates_and_nonfinite_values_rejected(self):
        original = pd.read_csv(self.folder / "panel.csv")
        changed = original.copy()
        changed.iloc[1] = changed.iloc[0]
        self.rewrite_panel(changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_dataset(self.name)
        changed = original.copy()
        changed.loc[0, "feature_month"] = changed.loc[0, "return_month"]
        self.rewrite_panel(changed)
        with self.assertRaisesRegex(ValueError, "one month"):
            load_dataset(self.name)
        changed = original.copy()
        changed.loc[0, "x0"] = np.nan
        self.rewrite_panel(changed)
        with self.assertRaises(ValueError):
            load_dataset(self.name)

    def test_models_do_not_change_saved_dataset_and_provenance_is_saved(self):
        before = {name: (self.folder / name).read_bytes() for name in ("panel.csv", "metadata.json")}
        with patch("main.load_config", return_value=self.cfg), redirect_stdout(io.StringIO()):
            first = main()
            self.cfg.RUN.MODELS.STANDARDIZE = True
            second = main()
        for name, contents in before.items():
            self.assertEqual(contents, (self.folder / name).read_bytes())
        self.assertEqual(first.dataset_info, second.dataset_info)
        run_dir = save_run(first, Path(self.temporary.name) / "runs")
        self.assertEqual(json.loads((run_dir / "dataset.json").read_text()), first.dataset_info)
        X, y, months, info = load_dataset(self.name)
        self.cfg.DATA.DATASET = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            run(X, y, months, self.cfg, dataset_info=info)

    def test_same_entry_point_runs_a_non_synthetic_dataset_name_and_saves(self):
        # Reuse toy observations to check routing, not real GKX replication results.
        metadata_path = self.folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["dataset_name"] = "gkx"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.cfg.DATA.DATASET = "gkx"
        output_dir = Path(self.temporary.name) / "shared_runs"
        self.cfg.RUN.OUTPUT_DIR = str(output_dir)
        with patch.dict(DATASET_PATHS, {"gkx": self.folder}), \
                patch("main.load_config", return_value=self.cfg), redirect_stdout(io.StringIO()):
            run_result = main("prepared_panel_test")
        self.assertEqual(run_result.dataset_info["name"], "gkx")
        self.assertEqual(list(run_result.summary.specification), ["OLS"])
        saved_dirs = list(output_dir.iterdir())
        self.assertEqual(len(saved_dirs), 1)
        saved_info = json.loads((saved_dirs[0] / "dataset.json").read_text())
        self.assertEqual(saved_info["name"], "gkx")
        saved_predictions = pd.read_csv(saved_dirs[0] / "predictions.csv")
        assert_allclose(saved_predictions.prediction, run_result.predictions.prediction)

    def test_nonlinear_signal_and_sample_extension_are_reproducible(self):
        linear, _ = generate_panel(**self.settings)
        nonlinear, settings = generate_panel(kind="nonlinear", **self.settings)
        assert_array_equal(linear[["x0", "x1", "x2"]], nonlinear[["x0", "x1", "x2"]])
        extra = .02 * (linear.x0**2 - 1) + .015 * linear.x1 * linear.x2
        assert_allclose(nonlinear.excess_return - linear.excess_return, extra, atol=1e-15)
        self.assertEqual(settings["kind"], "nonlinear")
        extended, _ = generate_panel(**{**self.settings, "end_month": "2005-12"})
        assert_array_equal(linear, extended.iloc[:len(linear)])

    def test_nonlinear_yaml_only_switches_dataset(self):
        linear, nonlinear = load_config("model_demo"), load_config("model_demo_nonlinear")
        self.assertEqual(nonlinear.DATA.DATASET, "synthetic_nonlinear_v1")
        self.assertEqual(linear.RUN, nonlinear.RUN)


if __name__ == "__main__":
    unittest.main()
