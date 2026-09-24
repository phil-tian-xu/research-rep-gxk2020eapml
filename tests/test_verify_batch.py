from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from core.data_cleaning.partition_gkx import partition_gkx
from core.data_examine.verify_batch import (check_reference, compare_records, main,
                                          memory_summary, run_worker, watch_memory)
from core.data_query.parquet_dataset import ParquetPanel
from core.run import run
from core.run_state import create_run, read_checkpoints
from utils.general_utils import load_config


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        rng = np.random.default_rng(74)
        dates = pd.period_range("2000-01", "2004-12", freq="M")
        X = rng.normal(size=(len(dates), 3))
        y = X[:, 0] * .03 + rng.normal(scale=.03, size=len(dates))
        y[::13] += .5
        frame = pd.DataFrame(X, columns=["a", "b", "c"])
        frame["return_month"], frame["feature_month"] = dates.astype(str), (dates - 1).astype(str)
        frame["excess_return"], frame["permno"] = y, "1"
        frame.to_csv(source / "panel.csv", index=False)
        metadata = {"schema_version": 1, "dataset_name": "fixture", "return_units": "decimal",
                    "id_columns": ["permno"], "feature_month_column": "feature_month", "return_month_column": "return_month",
                    "target_column": "excess_return", "feature_columns": ["a", "b", "c"], "audit_columns": [],
                    "n_rows": len(frame), "panel_sha256": hashlib.sha256((source / "panel.csv").read_bytes()).hexdigest()}
        metadata_path = source / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        self.cfg = load_config("gkx_table1_figure3_v4")
        self.cfg.DATA.DATASET, self.cfg.DATA.FORMAT = "fixture", "csv"
        self.cfg.RUN.SPLIT.FIRST_TRAIN_YEAR, self.cfg.RUN.SPLIT.VALIDATION_YEARS = 2000, 1
        self.cfg.RUN.SPLIT.FIRST_TEST_YEAR, self.cfg.RUN.SPLIT.LAST_TEST_YEAR = 2003, 2004
        self.cfg.RUN.MODELS.PCR.COMPONENT_GRID = [0, 1, 2]
        self.cfg.RUN.MODELS.PLS.COMPONENT_GRID = [0, 1, 2]
        self.cfg.RUN.REPORT_TABLE1_FIGURE3 = False
        self.cfg.RUN.OUTPUT_DIR = str(self.root / "reference")
        info = {"name": "fixture", "panel_sha256": metadata["panel_sha256"],
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
        self.reference = create_run(self.cfg, info)
        with redirect_stdout(StringIO()):
            run(X, y, dates, self.cfg, dataset_info=info, checkpoint_dir=self.reference)
            self.parquet = partition_gkx(source_dir=source, output_dir=self.root / "parquet", chunksize=17)
        self.records = read_checkpoints(self.reference, self.cfg)

    def compare_command(self, *extra):
        return ["compare", "--reference-run", str(self.reference), "--parquet-dir", str(self.parquet),
                "--output-dir", str(self.root / "verification"), "--batch-size", "11", *extra]

    def test_comparison_detects_different_predictions_parameters_scores_and_rows(self):
        reference = self.records[2003, "PCR"]
        self.assertTrue(all(row["passed"] for row in compare_records(reference, reference, atol=1e-7, rtol=1e-6)))
        for field in ("prediction", "row_position"):
            actual = deepcopy(reference)
            actual["predictions"][field][0] += 1
            self.assertFalse(all(row["passed"] for row in compare_records(reference, actual, atol=1e-7, rtol=1e-6)))
        for mutate in (lambda r: r["tuning"]["best_params"].update(n_components=99),
                       lambda r: r["tuning"]["validation_results"][0].update(validation_loss=99)):
            actual = deepcopy(reference)
            mutate(actual)
            self.assertFalse(all(row["passed"] for row in compare_records(reference, actual, atol=1e-7, rtol=1e-6)))

    def test_comparison_aligns_predictions_and_huber_threshold_is_checked(self):
        reference = self.records[2003, "OLS_HUBER"]
        actual = deepcopy(reference)
        actual["predictions"] = {key: value[::-1] for key, value in actual["predictions"].items()}
        self.assertTrue(all(r["passed"] for r in compare_records(reference, actual, atol=1e-7, rtol=1e-6)))
        actual["tuning"]["validation_results"][0]["huber_delta"] += .1
        self.assertFalse(next(r["passed"] for r in compare_records(reference, actual, atol=1e-7, rtol=1e-6)
                              if r["field"] == "huber_thresholds"))

    def test_reference_hash_mismatch_is_rejected_before_fitting(self):
        panel = ParquetPanel(self.parquet)
        check_reference(panel, self.reference, self.cfg)
        panel.metadata["source_panel_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source hashes"):
            check_reference(panel, self.reference, self.cfg)

    def test_compare_command_runs_fresh_workers_and_matches_saved_results(self):
        # This exercises the real subprocess entry point and the production batch runner.
        with redirect_stdout(StringIO()):
            code = main(self.compare_command())
        self.assertEqual(code, 0)
        report_path = next((self.root / "verification").glob("*/report.json"))
        report = json.loads(report_path.read_text())
        self.assertEqual(report["years"], [2003, 2004])
        self.assertTrue(report["complete_reference_horizon"])
        self.assertTrue(report["correctness_passed"])
        self.assertTrue(report["largest_window_selected"])
        self.assertEqual(len(report["results"]), 2)
        if os.name == "nt":
            self.assertTrue(report["memory_target_passed"])
            self.assertTrue(all(r["peak_resident_gib"] > 0 for r in report["results"]))

    def test_memory_mode_selects_first_and_largest_and_exceeded_target_fails(self):
        cfg = self.cfg.clone()
        cfg.DATA.DATASET, cfg.DATA.FORMAT = "fixture_parquet", "parquet"
        calls = []

        def fake_worker(command, **kwargs):
            path = Path(command[-1])
            job = json.loads(path.read_text())
            calls.append(job)
            measurement = {"test_year": job["year"], "status": "complete", "peak_resident_gib": 9.,
                           "memory_target_passed": False}
            path.with_name("measurement.json").write_text(json.dumps(measurement))
            return type("Completed", (), {"returncode": 0})()

        with patch("core.data_examine.verify_batch.load_config", return_value=cfg), \
                patch("core.data_examine.verify_batch.subprocess.run", side_effect=fake_worker), redirect_stdout(StringIO()):
            code = main(["memory", "--parquet-dir", str(self.parquet), "--output-dir", str(self.root / "memory")])
        self.assertEqual(code, 1)
        self.assertEqual([job["year"] for job in calls], [2003, 2004])
        self.assertTrue(all(job["config"]["RUN"]["SPLIT"]["FIRST_TRAIN_YEAR"] == 2000 for job in calls))
        self.assertEqual(calls[-1]["config"]["RUN"]["SPLIT"]["LAST_TEST_YEAR"], 2004)

    def test_worker_failure_preserves_memory_measurement(self):
        cfg = self.cfg.clone()
        cfg.DATA.DATASET, cfg.DATA.FORMAT = "fixture_parquet", "parquet"
        cfg.DATA.BATCH_SIZE = 11
        cfg.RUN.OUTPUT_DIR = str(self.root / "failed_run")
        import yaml
        job = self.root / "job.json"
        job.write_text(json.dumps({"config": yaml.safe_load(cfg.dump()), "year": 2003,
                                   "parquet_dir": str(self.parquet), "memory_target_gib": 8.}))
        counters = {"resident_bytes": 1, "peak_resident_bytes": 9 * 1024**3, "private_committed_bytes": 2}
        with patch("core.batch_run.run_parquet", side_effect=MemoryError("test failure")), \
                patch("core.data_examine.verify_batch.process_memory", return_value=counters), \
                patch("core.data_examine.verify_batch.traceback.print_exc"), redirect_stdout(StringIO()):
            self.assertEqual(run_worker(job), 1)
        report = json.loads(job.with_name("measurement.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["memory_target_passed"])
        self.assertEqual(report["peak_resident_gib"], 9.)


class MemorySummaryTests(unittest.TestCase):
    def report(self):
        return {"years": [1987, 1988, 2016], "memory_target_gib": 8.,
                "largest_training_test_years": [2016], "memory_target_passed": False,
                "status": "failed", "correctness_passed": False,
                "results": [{"test_year": 1987, "status": "complete", "peak_resident_gib": .391},
                            {"test_year": 1988, "status": "complete", "peak_resident_gib": .387}]}

    def test_early_correctness_failure_is_incomplete_not_ram_exceedance(self):
        summary = memory_summary(self.report())
        self.assertEqual(summary["peak_resident_gib"], .391)
        self.assertEqual(summary["memory_status"], "incomplete")
        self.assertIsNone(summary["memory_target_passed"])
        self.assertFalse(summary["largest_window_completed"])
        self.assertEqual(summary["pending_or_failed_years"], [2016])

    def test_exceedance_is_reported_even_if_fit_failed(self):
        report = self.report()
        report["results"].append({"test_year": 2016, "status": "failed", "peak_resident_gib": 9.})
        summary = memory_summary(report)
        self.assertEqual(summary["peak_resident_gib"], 9.)
        self.assertEqual(summary["memory_status"], "exceeded")
        self.assertFalse(summary["memory_target_passed"])
        self.assertFalse(summary["largest_window_completed"])

    def test_complete_measurements_pass_and_missing_measurements_are_unknown(self):
        report = self.report()
        report["results"].append({"test_year": 2016, "status": "complete", "peak_resident_gib": .6})
        self.assertTrue(memory_summary(report)["memory_target_passed"])
        self.assertTrue(memory_summary(report)["largest_window_completed"])
        report["results"][-1]["peak_resident_gib"] = None
        self.assertEqual(memory_summary(report)["memory_status"], "unavailable")
        self.assertIsNone(memory_summary(report)["memory_target_passed"])

    def test_summary_command_reads_old_report_without_refitting_or_overwriting(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report()))
            original = path.read_bytes()
            output = StringIO()
            with patch("core.data_examine.verify_batch.verify", side_effect=AssertionError("No fits")), redirect_stdout(output):
                self.assertEqual(main(["summary", "--run-dir", directory]), 0)
            self.assertIn("MAXIMUM OBSERVED PYTHON RAM: 0.391 GiB", output.getvalue())
            self.assertIn("INCOMPLETE", output.getvalue())
            self.assertEqual(path.read_bytes(), original)

    def test_live_monitor_publishes_os_peak_and_current_ram_separately(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory_latest.json"
            stop = Mock()
            stop.wait.side_effect = [False, True]
            counters = {"resident_bytes": 1024**3, "peak_resident_bytes": 2 * 1024**3,
                        "private_committed_bytes": 3 * 1024**3}
            output = StringIO()
            with patch("core.data_examine.verify_batch.process_memory", return_value=counters), redirect_stdout(output):
                watch_memory(stop, path, 2016)
            self.assertEqual(json.loads(path.read_text())["peak_resident_gib"], 2.)
            self.assertIn("current=1.000 GiB; MAX SO FAR=2.000 GiB", output.getvalue())


if __name__ == "__main__":
    unittest.main()
