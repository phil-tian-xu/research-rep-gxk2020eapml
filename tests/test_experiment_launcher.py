from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

import run_experiment as launcher
from core.annual_jobs import load_plan
from core.run_state import atomic_write


class ExperimentLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        panel = self.root / "panel"
        panel.mkdir()
        (panel / "metadata.json").write_text("{}")
        mapping = patch.dict(launcher.DATASET_PATHS, {name: panel for name in launcher.SAMPLES.values()})
        mapping.start()
        self.addCleanup(mapping.stop)
        output = redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def prepare(self, model="pcr", backend="--scrp", years="2009:2016", sample="full"):
        return launcher.main([model, backend, "--sample", sample, "--years", years,
                              "--root", str(self.root / "results"), "--dry-run"])

    def test_sample_years_and_defaults_are_recorded_without_new_repository_configs(self):
        folder = self.prepare("pls", "--local", "2016", "p05")
        record = launcher.read_launch(folder)
        _, plan = load_plan(folder / "annual")
        self.assertEqual(plan["config"]["RUN"]["MODELS"]["ENABLED"], ["PLS"])
        self.assertEqual(plan["config"]["DATA"]["DATASET"], launcher.SAMPLES["p05"])
        self.assertEqual(plan["years"], [2016])
        self.assertEqual(plan["config"]["RUN"]["SPLIT"]["FIRST_TRAIN_YEAR"], 1957)
        self.assertEqual(plan["config"]["RUN"]["MODELS"]["PLS"]["COMPONENT_GRID"], list(range(11)))
        self.assertEqual(record["resources"]["parallel"], 1)
        self.assertEqual(record["submissions"][0]["state"], "prepared")
        self.assertTrue((folder / "research.yaml").exists())
        self.assertEqual(len(list((folder / "annual/configs").glob("*.yaml"))), 1)

    def test_huber_commands_select_correct_model_and_full_sample(self):
        for command, model in (("enet", "ENET_HUBER"), ("ols3", "OLS3_HUBER")):
            folder = self.prepare(command, "--scrp", "2016", "full")
            _, plan = load_plan(folder / "annual")
            self.assertEqual(plan["config"]["RUN"]["MODELS"]["ENABLED"], [model])
            self.assertEqual(plan["config"]["DATA"]["DATASET"], "gkx_parquet")
            self.assertEqual(launcher.read_launch(folder)["model"], model)

    def test_sbatch_arguments_job_receipt_and_dependency(self):
        folder = self.prepare()
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "397999;cluster\n", "")) as run:
            launcher.main(["resume", str(folder), "--parallel", "4", "--cpus", "8",
                           "--mem", "128G", "--after", "397886"])
        command = run.call_args.args[0]
        for flag in ("--parsable", "--array=0,1,2,3,4,5,6,7%4", "--cpus-per-task=8",
                     "--mem=128G", "--dependency=afterany:397886"):
            self.assertIn(flag, command)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["cwd"], launcher.ROOT)
        record = launcher.read_launch(folder)
        self.assertEqual(record["submissions"][-1]["job_id"], "397999")
        self.assertEqual(record["submissions"][-1]["state"], "submitted")
        self.assertTrue(Path(command[-2]).exists())

    def test_submission_failure_is_recorded_and_can_be_retried(self):
        folder = self.prepare()
        error = subprocess.CalledProcessError(1, ["sbatch"], stderr="Invalid account")
        with patch("run_experiment.subprocess.run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main(["resume", str(folder)])
        self.assertEqual(launcher.read_launch(folder)["submissions"][-1]["state"], "submission_failed")
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "123\n", "")):
            launcher.main(["resume", str(folder)])
        self.assertEqual(launcher.read_launch(folder)["submissions"][-1]["job_id"], "123")

    def test_uncertain_submission_cannot_be_blindly_retried(self):
        folder = self.prepare()
        with patch("run_experiment.subprocess.run", side_effect=subprocess.TimeoutExpired("sbatch", 60)):
            with self.assertRaises(subprocess.TimeoutExpired):
                launcher.main(["resume", str(folder)])
        with self.assertRaisesRegex(ValueError, "uncertain"):
            launcher.main(["resume", str(folder)])

    def test_active_submission_blocks_retry_and_completed_years_are_omitted(self):
        folder = self.prepare()
        record = launcher.read_launch(folder)
        record["submissions"].append({"state": "submitted", "job_id": "123"})
        atomic_write(folder / "launch.json", json.dumps(record))
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "123\n", "")):
            with self.assertRaisesRegex(ValueError, "queued/running"):
                launcher.main(["resume", str(folder)])
        with patch("run_experiment.saved_years", return_value=[2009, 2010]), patch(
                "run_experiment.subprocess.run", side_effect=[subprocess.CompletedProcess([], 0, "", ""),
                                                              subprocess.CompletedProcess([], 0, "124", "")]) as run:
            launcher.main(["resume", str(folder)])
        self.assertIn("--array=2,3,4,5,6,7%4", run.call_args.args[0])
        self.assertEqual(launcher.read_launch(folder)["submissions"][-1]["job_id"], "124")

    def test_listing_persists_without_scheduler_and_marks_unavailable_status_unknown(self):
        one = self.prepare()
        self.prepare("pls", years="2016")
        rows = launcher.list_experiments(self.root / "results")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["model"] for row in rows}, {"PCR", "PLS"})
        self.assertEqual(rows[0]["saved"], "0/8")
        record = launcher.read_launch(one)
        record["submissions"].append({"state": "submitted", "job_id": "123"})
        atomic_write(one / "launch.json", json.dumps(record))
        with patch("run_experiment.subprocess.run", side_effect=FileNotFoundError("squeue")):
            rows = launcher.list_experiments(self.root / "results", refresh=True)
        self.assertEqual(rows[0]["scheduler_now"], "123=unknown")

    def test_scheduler_refresh_distinguishes_queue_and_accounting(self):
        responses = [subprocess.CompletedProcess([], 0, "123|RUNNING\n123|PENDING\n", ""),
                     subprocess.CompletedProcess([], 0, "124_0|COMPLETED|\n124_1|FAILED|\n", "")]
        with patch("run_experiment.subprocess.run", side_effect=responses):
            states = launcher.scheduler_states(["123", "124", "125"])
        self.assertEqual(states["123"], "PENDING:1,RUNNING:1")
        self.assertEqual(states["124"], "COMPLETED:1,FAILED:1")
        self.assertEqual(states["125"], "unknown")

    def test_local_execution_uses_separate_workers_and_collects_reports(self):
        folder = self.prepare(backend="--local", years="1987:1988", sample="p05")
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run, \
                patch("run_experiment.collect") as collect:
            launcher.main(["resume", str(folder), "--cpus", "2"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][-1], "1")
        self.assertEqual(run.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"], "2")
        collect.assert_called_once_with(folder / "annual")
        self.assertEqual(launcher.read_launch(folder)["submissions"][-1]["state"], "complete")

    def test_failed_local_worker_is_recorded_without_collecting(self):
        folder = self.prepare(backend="--local", years="1987", sample="p05")
        with patch("run_experiment.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["python"])), \
                patch("run_experiment.collect") as collect:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main(["resume", str(folder)])
        collect.assert_not_called()
        self.assertEqual(launcher.read_launch(folder)["submissions"][-1]["state"], "failed")

    def test_invalid_options_fail_before_creating_experiments(self):
        for options in (["--cpus", "0"], ["--parallel", "0"], ["--mem", "0G"], ["--time", "bad"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.main(["pcr", "--local", "--sample", "p05", "--years", "1987",
                               "--root", str(self.root / "invalid"), *options])
        self.assertFalse((self.root / "invalid").exists())

    def test_aihub_and_scrp_share_research_settings_but_not_deployment(self):
        for model in launcher.BASELINES:
            with self.subTest(model=model):
                scrp = self.prepare(model, years="1987", sample="p05")
                hub = self.prepare(model, backend="--aihub", years="1987", sample="p05")
                self.assertEqual((scrp / "research.yaml").read_bytes(), (hub / "research.yaml").read_bytes())
                _, scrp_plan = load_plan(scrp / "annual")
                _, hub_plan = load_plan(hub / "annual")
                self.assertEqual(scrp_plan["runtime"], hub_plan["runtime"])
                record = launcher.read_launch(hub)
                self.assertEqual(record["backend"], "aihub")
                self.assertEqual(record["resources"]["parallel"], 1)
                command = record["submissions"][-1]["sbatch_command"]
                self.assertIn("--array=0%1", command)
                self.assertEqual(Path(command[-2]).name, "run_annual_aihub.sh")
                self.assertFalse(any(flag.startswith(("--account=", "--partition=", "--qos=")) for flag in command))
                script = Path(command[-2]).read_bytes()
                self.assertNotIn(b"\r", script)
                self.assertNotIn(b"#SBATCH --gres", script)
                scrp_command = launcher.read_launch(scrp)["submissions"][-1]["sbatch_command"]
                for flag in ("--account=pg", "--partition=scrp", "--qos=c32g4", "--array=0%4"):
                    self.assertIn(flag, scrp_command)

    def test_aihub_resume_preserves_backend_overrides_and_dependency(self):
        folder = self.prepare(backend="--aihub")
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "321\n", "")) as run:
            launcher.main(["resume", str(folder), "--account", "test-account", "--partition", "test-partition",
                           "--qos", "test-qos", "--after", "123"])
        for flag in ("--account=test-account", "--partition=test-partition", "--qos=test-qos", "--dependency=afterany:123"):
            self.assertIn(flag, run.call_args.args[0])
        launcher.main(["resume", str(folder), "--dry-run"])
        record = launcher.read_launch(folder)
        self.assertEqual(record["backend"], "aihub")
        self.assertEqual(record["resources"]["account"], "test-account")
        self.assertEqual(record["resources"]["parallel"], 1)

    def test_aihub_limits_fail_before_preparation_and_on_resume(self):
        for options in (["--parallel", "2"], ["--mem", "245761M"], ["--cpus", "61"],
                        ["--time", "1-00:00:01"], ["--time", "00:00:00"], ["--time", "00:99:00"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.main(["pcr", "--aihub", "--sample", "full", "--years", "2016",
                               "--root", str(self.root / "invalid"), "--dry-run", *options])
        self.assertFalse((self.root / "invalid").exists())
        folder = self.prepare(backend="--aihub")
        with self.assertRaisesRegex(ValueError, "240G"):
            launcher.main(["resume", str(folder), "--parallel", "2", "--dry-run"])
        self.assertEqual(len(launcher.read_launch(folder)["submissions"]), 1)

    def test_aihub_root_lookup_listing_and_no_backend_migration(self):
        with patch.dict("os.environ", {}, clear=True), patch("run_experiment.shutil.which", return_value="sbatch"):
            self.assertEqual(launcher.results_root(aihub=True), launcher.ROOT / "results/aihub_experiments")
            self.assertNotEqual(launcher.results_root(aihub=True), launcher.results_root(scrp=True))
        folder = self.prepare(backend="--aihub")
        launcher.main(["resume", folder.name, "--aihub", "--root", str(folder.parent), "--dry-run"])
        rows = launcher.main(["list", "--aihub", "--root", str(folder.parent)])
        self.assertEqual(len(rows), 1)
        with patch("run_experiment.collect") as collect:
            launcher.main(["collect", folder.name, "--aihub", "--root", str(folder.parent)])
        collect.assert_called_once_with(folder / "annual")
        scrp = self.prepare()
        with self.assertRaisesRegex(ValueError, "another backend"):
            launcher.main(["resume", str(scrp), "--aihub", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
