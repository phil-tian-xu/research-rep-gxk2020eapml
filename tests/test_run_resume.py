from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from core.data_query.datasets import DATASET_PATHS, load_dataset
from core.data_query.generate_experimental_data import save_generated_dataset
from core.model_selection import tune_hyperparameters
from core.run import run
from core.run_state import (atomic_write, create_run, load_run_config, read_checkpoints,
                            run_lock, runtime_identity)
from main import main
from utils.general_utils import load_config


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = load_config("gkx_all_models_workflow_v2")
        self.cfg.RUN.OUTPUT_DIR = str(self.root / "runs")
        data = self.root / "panel"
        save_generated_dataset(data, dataset_name=self.cfg.DATA.DATASET,
                               start_month="1957-03", end_month="1991-12",
                               stocks_per_month=2, coefficients=[.1, -.05, .01, .02])
        mapping = patch.dict(DATASET_PATHS, {self.cfg.DATA.DATASET: data})
        mapping.start()
        self.addCleanup(mapping.stop)
        self.X, self.y, self.months, self.info = load_dataset(self.cfg.DATA.DATASET)
        output = redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def execute(self, folder, **kwargs):
        return run(self.X, self.y, self.months, self.cfg, dataset_info=self.info,
                   checkpoint_dir=folder, **kwargs)

    def test_workflow_v2_three_stages_match_uninterrupted_and_skip_fits(self):
        baseline = run(self.X, self.y, self.months, self.cfg, dataset_info=self.info)
        first_output = StringIO()
        with patch("main.load_config", return_value=self.cfg), redirect_stdout(first_output):
            first = main("gkx_all_models_workflow_v2", stop_after_year=1988)
        folder, = (self.root / "runs").iterdir()
        self.assertIn("Run incomplete. To continue, please use --resume mode:", first_output.getvalue())
        self.assertIn(f"$runDir = '{folder.resolve()}'", first_output.getvalue())
        self.assertIn('python -B main.py --resume "$runDir"', first_output.getvalue())
        self.assertEqual(len(first.annual_metrics), 6)
        with patch("core.run.tune_hyperparameters", wraps=tune_hyperparameters) as tune:
            second = main(resume=folder, stop_after_year=1990)
            self.assertEqual(tune.call_count, 6)
            self.assertEqual({c.kwargs["test_year"] for c in tune.call_args_list}, {1989, 1990})
        self.assertEqual(len(second.annual_metrics), 12)
        final_output = StringIO()
        with patch("core.run.tune_hyperparameters", wraps=tune_hyperparameters) as tune, redirect_stdout(final_output):
            final = main(resume=folder)
            self.assertEqual(tune.call_count, 3)
        self.assertNotIn("Run incomplete", final_output.getvalue())
        for name in ("annual_metrics", "summary", "predictions"):
            assert_frame_equal(getattr(final, name), getattr(baseline, name), check_exact=True)
        self.assertEqual(final.tuning_history, baseline.tuning_history)
        saved = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        assert_frame_equal(saved, baseline.predictions, check_dtype=False, check_exact=True)
        self.assertTrue((pd.read_csv(folder / "status.csv").status == "complete").all())
        with patch("core.run.tune_hyperparameters", side_effect=AssertionError("refitted")):
            main(resume=folder)

    def test_failed_candidate_keeps_completed_model_and_resumes_partial_year(self):
        folder = create_run(self.cfg, self.info)

        def interrupt_pcr(*args, **kwargs):
            if kwargs["model_name"] == "PCR":
                raise RuntimeError("simulated fit failure")
            return tune_hyperparameters(*args, **kwargs)

        failure_output = StringIO()
        with patch("core.run.tune_hyperparameters", side_effect=interrupt_pcr), redirect_stdout(failure_output):
            with self.assertRaisesRegex(RuntimeError, "simulated fit failure"):
                self.execute(folder, stop_after_year=1987)
        self.assertIn("Run incomplete", failure_output.getvalue())
        self.assertIn(f"$runDir = '{folder.resolve()}'", failure_output.getvalue())
        self.assertEqual(list(read_checkpoints(folder, self.cfg)), [(1987, "OLS")])
        status = pd.read_csv(folder / "status.csv")
        self.assertEqual((status.status == "complete").sum(), 1)
        with patch("core.run.tune_hyperparameters", wraps=tune_hyperparameters) as tune:
            self.execute(folder, stop_after_year=1987)
            self.assertEqual([c.kwargs["model_name"] for c in tune.call_args_list], ["PCR", "PLS"])

    def test_atomic_failure_is_not_a_completed_checkpoint(self):
        folder = create_run(self.cfg, self.info)
        with patch("core.run_state.os.replace", side_effect=OSError("interrupted write")):
            with self.assertRaisesRegex(OSError, "interrupted write"):
                self.execute(folder, stop_after_year=1987)
        self.assertEqual(read_checkpoints(folder, self.cfg), {})
        # A hard kill can leave a temp file; it must not be mistaken for completion.
        abandoned = folder / "years" / "1987" / ".OLS.json.abandoned.tmp"
        abandoned.write_text("partial", encoding="utf-8")
        result = self.execute(folder, stop_after_year=1987)
        self.assertEqual(len(result.annual_metrics), 3)

    def test_incompatible_config_data_and_runtime_rejected_before_fitting(self):
        folder = create_run(self.cfg, self.info)
        with patch("core.run.tune_hyperparameters", side_effect=AssertionError("fitted")):
            self.cfg.RUN.MODELS.PLS.COMPONENT_GRID = [1, 2]
            with self.assertRaisesRegex(ValueError, "config differs"):
                self.execute(folder)
            self.cfg = load_run_config(folder)
            self.info = dict(self.info, panel_sha256="changed")
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                self.execute(folder)
            self.info = load_dataset(self.cfg.DATA.DATASET)[3]
            for field in ("packages", "source_sha256"):
                changed = copy.deepcopy(runtime_identity())
                changed[field]["changed"] = "different"
                with patch("core.run_state.runtime_identity", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "runtime differs"):
                        self.execute(folder)

    def test_corrupt_checkpoint_rejected_and_lock_prevents_two_writers(self):
        folder = create_run(self.cfg, self.info)
        with run_lock(folder):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                self.execute(folder)
        self.execute(folder, stop_after_year=1987)
        path = folder / "years" / "1987" / "OLS.json"
        record = json.loads(path.read_text())
        record["payload"] += " "
        atomic_write(path, json.dumps(record))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.execute(folder)

    def test_stop_year_validation_and_saved_config_edit(self):
        folder = create_run(self.cfg, self.info)
        for year in (1986, 1992, True):
            with self.assertRaises(ValueError):
                self.execute(folder, stop_after_year=year)
        path = folder / "resolved_config.yaml"
        path.write_text(path.read_text().replace("1991", "1992"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Saved configuration differs"):
            load_run_config(folder)


if __name__ == "__main__":
    unittest.main()
