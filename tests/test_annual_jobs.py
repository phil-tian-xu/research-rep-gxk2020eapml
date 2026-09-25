from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from core.annual_jobs import prepare, run_task, collect, load_plan, annual_config
from core.data_cleaning.partition_gkx import partition_gkx
from core.data_query.datasets import DATASET_PATHS
from core.data_query.parquet_dataset import ParquetPanel
from core.run_state import load_run_config, read_checkpoints
from main import main
from utils.general_utils import load_config


class AnnualJobsTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        silence = redirect_stdout(StringIO())
        silence.__enter__()
        self.addCleanup(silence.__exit__, None, None, None)
        source = self.root / "source"
        source.mkdir()
        months = pd.period_range("2000-01", "2004-12", freq="M").repeat(2)
        rng = np.random.default_rng(75)
        X = rng.normal(size=(len(months), 3))
        frame = pd.DataFrame(X, columns=["x0", "x1", "x2"])
        frame["excess_return"] = X[:, 0] * .1 + rng.normal(scale=.05, size=len(X))
        frame["return_month"] = months.astype(str)
        frame["feature_month"] = (months - 1).astype(str)
        frame["permno"] = np.tile(["a", "b"], len(months) // 2)
        frame.to_csv(source / "panel.csv", index=False)
        metadata = {"schema_version": 1, "dataset_name": "fixture", "return_units": "decimal",
                    "id_columns": ["permno"], "feature_month_column": "feature_month",
                    "return_month_column": "return_month", "target_column": "excess_return",
                    "feature_columns": ["x0", "x1", "x2"], "audit_columns": [], "n_rows": len(frame),
                    "panel_sha256": hashlib.sha256((source / "panel.csv").read_bytes()).hexdigest()}
        (source / "metadata.json").write_text(json.dumps(metadata))
        panel = partition_gkx(source_dir=source, output_dir=self.root / "panel", chunksize=19)
        self.cfg = load_config("gkx_pcr_full_v5")
        self.cfg.DATA.DATASET = ParquetPanel(panel).info["name"]
        self.cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
        self.cfg.RUN.SPLIT.FIRST_TEST_YEAR = 2003
        self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2004
        self.cfg.RUN.SPLIT.VALIDATION_YEARS = 1
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = [0, 1, 2]
        self.cfg.RUN.REPORT_TABLE1_FIGURE3 = False
        mapping = patch.dict(DATASET_PATHS, {self.cfg.DATA.DATASET: panel})
        mapping.start()
        self.addCleanup(mapping.stop)
        self.experiment = self.root / "experiment"
        with patch("core.annual_jobs.load_config", return_value=self.cfg):
            prepare("fixture", 2003, 2004, self.experiment)

    def run_both(self):
        for index in (1, 0):  # Out-of-order completion is intentional.
            run_task(self.experiment, index)

    def test_isolated_years_match_serial_predictions_tuning_and_pooled_report(self):
        self.run_both()
        self.cfg.RUN.OUTPUT_DIR = str(self.root / "serial")
        with patch("main.load_config", return_value=self.cfg):
            main("fixture")
        serial_dir, = (self.root / "serial").iterdir()
        serial = read_checkpoints(serial_dir, load_run_config(serial_dir))
        folder, plan = load_plan(self.experiment)
        for year in (2003, 2004):
            cfg = annual_config(plan, folder, year)
            self.assertEqual(cfg.RUN.SPLIT.FIRST_TRAIN_YEAR, 2000)
            self.assertEqual(cfg.RUN.SPLIT.FIRST_TEST_YEAR, year)
            self.assertEqual(cfg.RUN.SPLIT.LAST_TEST_YEAR, year)
            run_dir, = (folder / "years" / str(year)).glob("*/manifest.json")
            records = read_checkpoints(run_dir.parent, cfg)
            self.assertEqual(set(records), {(year, "PCR")})
            actual, expected = records[year, "PCR"], serial[year, "PCR"]
            self.assertEqual(actual["tuning"], expected["tuning"])
            assert_allclose(actual["predictions"]["prediction"], expected["predictions"]["prediction"], atol=0, rtol=0)
        with patch("core.models.dimension_reduction.PCRRegressor.fit", side_effect=AssertionError("refit")):
            run_task(folder, 0)  # Repeated submission resumes and skips the completed fit.
        output = collect(folder)
        pooled = pd.read_csv(output / "table1_all.csv").iloc[0]
        observed = np.concatenate([r["predictions"]["observed"] for r in serial.values()])
        prediction = np.concatenate([r["predictions"]["prediction"] for r in serial.values()])
        self.assertAlmostEqual(pooled.r2_zero, 1 - np.sum((observed-prediction)**2) / np.sum(observed**2))
        self.assertEqual(list(pd.read_csv(output / "complexity.csv").test_year), [2003, 2004])
        self.assertTrue((output / "figure3_complexity.png").is_file())
        self.assertTrue((output / "source_identity.json").is_file())

    def test_missing_year_is_reported_and_duplicate_year_is_rejected(self):
        run_task(self.experiment, 0)
        with self.assertRaisesRegex(ValueError, "Missing/incomplete"):
            collect(self.experiment)
        status = pd.read_csv(self.experiment / "task_summary.csv")
        self.assertEqual(list(status.status), ["complete", "missing"])
        run_dir, = (self.experiment / "years" / "2003").glob("*/manifest.json")
        shutil.copytree(run_dir.parent, run_dir.parent.parent / "duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            collect(self.experiment)

    def test_source_and_generated_config_changes_are_rejected(self):
        with patch("core.annual_jobs.runtime_identity", return_value={}):
            with self.assertRaisesRegex(ValueError, "Execution code"):
                run_task(self.experiment, 0)
        config_path = self.experiment / "configs" / "2003.yaml"
        config_path.write_text(config_path.read_text().replace("FIRST_TRAIN_YEAR: 2000", "FIRST_TRAIN_YEAR: 2001"))
        with self.assertRaisesRegex(ValueError, "configuration was changed"):
            run_task(self.experiment, 0)

    def test_mixed_dataset_or_runtime_is_rejected(self):
        self.run_both()
        manifest_path, = (self.experiment / "years" / "2004").glob("*/manifest.json")
        manifest = json.loads(manifest_path.read_text())
        original = json.loads(manifest_path.read_text())
        manifest["dataset"]["panel_sha256"] = "different"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Dataset identity"):
            collect(self.experiment)
        original["runtime"]["python"] = "different"
        manifest_path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "Execution identity"):
            collect(self.experiment)

    def test_bounds_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "Task index"):
            run_task(self.experiment, -1)
        with patch("core.annual_jobs.load_config", return_value=self.cfg):
            with self.assertRaises(FileExistsError):
                prepare("fixture", 2003, 2004, self.experiment)
            with self.assertRaisesRegex(ValueError, "horizon"):
                prepare("fixture", 2002, 2004, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
