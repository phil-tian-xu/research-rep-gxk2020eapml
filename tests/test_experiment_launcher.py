from contextlib import redirect_stdout
from io import StringIO
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest
from unittest.mock import patch

import run_experiment as launcher
from core.annual_jobs import load_plan
from core.run_state import atomic_write, run_lock


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
        self.assertEqual(plan["config"]["RUN"]["MODELS"]["PLS"]["COMPONENT_GRID"], list(range(30)))
        self.assertTrue(plan["config"]["RUN"]["MODELS"]["STANDARDIZE"])
        self.assertTrue(plan["config"]["RUN"]["MODELS"]["FIT_INTERCEPT"])
        self.assertEqual(record["baseline_config"], "gkx_pls_v2")
        self.assertEqual(record["resources"]["parallel"], 1)
        self.assertEqual(record["submissions"][0]["state"], "prepared")
        self.assertTrue((folder / "research.yaml").exists())
        self.assertEqual(len(list((folder / "annual/configs").glob("*.yaml"))), 1)

    def test_huber_commands_select_correct_model_and_full_sample(self):
        for command, model in (("enet", "ENET_HUBER"), ("ols3", "OLS3_HUBER"), ("glm", "GLM_HUBER"), ("rf", "RF")):
            folder = self.prepare(command, "--scrp", "2016", "full")
            _, plan = load_plan(folder / "annual")
            self.assertEqual(plan["config"]["RUN"]["MODELS"]["ENABLED"], [model])
            self.assertEqual(plan["config"]["DATA"]["DATASET"], "gkx_parquet")
            self.assertEqual(launcher.read_launch(folder)["model"], model)

    def test_explicit_dataset_changes_only_data_selection_and_survives_resume(self):
        baseline = self.prepare("nn3", "--local", "1987", "p05")
        panel = self.root / "compact"
        panel.mkdir()
        chars = [f"c{i}" for i in range(94)]
        macros = [f"macro_{m}" for m in ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar")]
        industries = [f"sic2_{i:02d}" for i in range(74)]
        (panel / "metadata.json").write_text(json.dumps(dict(dataset_name="gkx_compact_full",
            format="parquet", status="complete", feature_columns=chars + macros + industries,
            feature_specification=dict(kind="compact", characteristics=chars,
                                       macro_columns=macros, industry_columns=industries))))
        with patch.dict(launcher.DATASET_PATHS, {"gkx_compact_full": panel}):
            folder = launcher.main(["nn3", "--local", "--dataset", "gkx_compact_full", "--years", "1987",
                                   "--config", "gkx_nn3_v1", "--root", str(self.root / "results"), "--dry-run"])
            original = (folder / "research.yaml").read_bytes()
            launcher.main(["resume", str(folder), "--dry-run"])
        _, plan = load_plan(folder / "annual")
        _, reference = load_plan(baseline / "annual")
        self.assertEqual(plan["config"]["RUN"]["MODELS"], reference["config"]["RUN"]["MODELS"])
        self.assertEqual(plan["config"]["RUN"]["SPLIT"], reference["config"]["RUN"]["SPLIT"])
        self.assertEqual(plan["config"]["DATA"]["DATASET"], "gkx_compact_full")
        self.assertEqual(launcher.read_launch(folder)["dataset"], "gkx_compact_full")
        self.assertEqual((folder / "research.yaml").read_bytes(), original)

    def test_dataset_and_sample_are_mutually_exclusive_and_invalid_manifest_fails_early(self):
        with redirect_stdout(StringIO()), patch("sys.stderr", StringIO()):
            with self.assertRaises(SystemExit):
                launcher.parser().parse_args(["nn1", "--local", "--sample", "full", "--dataset", "gkx_parquet", "--years", "1987"])
        with self.assertRaisesRegex(ValueError, "complete Parquet"):
            launcher.main(["nn1", "--local", "--dataset", "gkx_parquet", "--years", "1987",
                           "--root", str(self.root / "invalid"), "--dry-run"])
        self.assertFalse((self.root / "invalid").exists())

    def test_squared_ols_commands_keep_existing_ols3_huber_mapping(self):
        for command, model in (("ols", "OLS"), ("ols3-mse", "OLS3"), ("ols3", "OLS3_HUBER")):
            folder = self.prepare(command, "--local", "1987", "p05")
            _, plan = load_plan(folder / "annual")
            self.assertEqual(plan["config"]["RUN"]["MODELS"]["ENABLED"], [model])

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

    def test_nn6_to_nn10_use_existing_settings_and_local_scrp_aihub_routes(self):
        reference = launcher.load_config("gkx_nn1_v1").RUN
        for depth in range(6, 11):
            for backend in ("--local", "--scrp", "--aihub"):
                with self.subTest(depth=depth, backend=backend):
                    folder = self.prepare(f"nn{depth}", backend, "1987", "p05")
                    record = launcher.read_launch(folder)
                    _, plan = load_plan(folder / "annual")
                    self.assertEqual(record["model"], f"NN{depth}")
                    self.assertEqual(record["baseline_config"], f"gkx_nn{depth}_v1")
                    self.assertEqual(plan["config"]["RUN"]["MODELS"]["NN"], dict(reference.MODELS.NN))
                    self.assertEqual(plan["config"]["RUN"]["EVALUATION"], dict(reference.EVALUATION))
                    self.assertEqual(record["resources"]["parallel"], 1)
                    self.assertEqual(record["resources"]["gpus"], 0 if backend == "--local" else 1)
                    command = record["submissions"][0].get("sbatch_command", [])
                    if backend == "--scrp":
                        self.assertIn("--gpus=a100:1", command)
                        self.assertEqual(Path(command[-2]).name, "run_nn_scrp.sh")
                    elif backend == "--aihub":
                        self.assertIn("--gres=gpu:1", command)

    def test_launcher_can_recognize_nn10_without_optional_torch(self):
        code = ('import sys; sys.modules["torch"] = None; '
                'import run_experiment as launcher; '
                'assert launcher.MODEL_NAMES["nn10"] == "NN10"; '
                'assert launcher.BASELINES["nn10"] == "gkx_nn10_v1"')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=launcher.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nn_gpu_requests_and_model_config_are_preserved(self):
        for backend, gpu_flag in (("--scrp", "--gpus=a100:1"), ("--aihub", "--gres=gpu:1")):
            folder = self.prepare("nn3", backend, "1987", "p05")
            record = launcher.read_launch(folder)
            _, plan = load_plan(folder / "annual")
            self.assertEqual(plan["config"]["RUN"]["MODELS"]["ENABLED"], ["NN3"])
            self.assertEqual(plan["config"]["RUN"]["MODELS"]["NN"]["ENSEMBLE_SIZE"], 10)
            self.assertEqual(record["resources"]["parallel"], 1)
            self.assertEqual(record["resources"]["gpus"], 1)
            self.assertIn(gpu_flag, record["submissions"][0]["sbatch_command"])
            command = record["submissions"][0]["sbatch_command"]
            if backend == "--scrp":
                self.assertIn("--partition=a100", command)
                self.assertEqual(record["resources"]["gpu_type"], "a100")
                self.assertEqual(Path(command[-2]).name, "run_nn_scrp.sh")
                self.assertIn("--constraint=a100", command)
            else:
                self.assertEqual(record["resources"]["gpu_type"], "any")
                self.assertEqual(Path(command[-2]).name, "run_annual_aihub.sh")
        with self.assertRaisesRegex(ValueError, "unused GPUs"):
            launcher.main(["pcr", "--scrp", "--sample", "p05", "--years", "1987", "--gpus", "1", "--dry-run"])

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

    def test_a100_pilot_resources_survive_resume_without_changing_nn_settings(self):
        folder = self.prepare("nn1", "--scrp", "1987", "p05")
        research = (folder / "research.yaml").read_bytes()
        launcher.main(["resume", str(folder), "--cpus", "4", "--mem", "8G",
                       "--time", "00:30:00", "--dry-run"])
        record = launcher.read_launch(folder)
        self.assertEqual((folder / "research.yaml").read_bytes(), research)
        self.assertEqual(record["resources"]["partition"], "a100")
        self.assertEqual(record["resources"]["gpus"], 1)
        command = record["submissions"][-1]["sbatch_command"]
        for flag in ("--gpus=a100:1", "--partition=a100", "--cpus-per-task=4",
                     "--mem=8G", "--time=00:30:00", "--array=0%1"):
            self.assertIn(flag, command)
        self.assertEqual(Path(command[-2]).name, "run_nn_scrp.sh")
        self.assertIn("--constraint=a100", command)
        self.assertEqual(record["submissions"][-1]["state"], "prepared")

    def test_four_3090_jobs_preserve_full_research_settings_and_resume_resources(self):
        baseline = self.prepare("nn1", "--scrp", "1987:2016", "full")
        folder = launcher.main(["nn1", "--scrp", "--sample", "full", "--years", "1987:2016",
                               "--gpu-type", "rtx3090", "--gpus", "1", "--parallel", "4",
                               "--cpus", "8", "--mem", "96G", "--time", "12:00:00",
                               "--root", str(self.root / "results"), "--dry-run"])
        research = (folder / "research.yaml").read_bytes()
        self.assertEqual(research, (baseline / "research.yaml").read_bytes())
        _, plan = load_plan(folder / "annual")
        self.assertEqual(plan["years"], list(range(1987, 2017)))
        self.assertEqual(plan["config"]["DATA"]["DATASET"], "gkx_parquet")
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "123\n", "")) as run:
            launcher.main(["resume", str(folder)])
        command = run.call_args.args[0]
        for flag in ("--gpus=rtx3090:1", "--partition=scrp", "--cpus-per-task=8",
                     "--mem=96G", "--time=12:00:00",
                     "--array=" + ",".join(map(str, range(30))) + "%4"):
            self.assertIn(flag, command)
        self.assertFalse(any(flag.startswith("--constraint=") for flag in command))
        self.assertNotIn("--gpus=a100:1", command)
        script = Path(command[-2]).read_bytes()
        for header in (b"#SBATCH --constraint", b"#SBATCH --gpus", b"#SBATCH --partition"):
            self.assertNotIn(header, script)
        self.assertNotIn(b"\r", script)
        record = launcher.read_launch(folder)
        self.assertEqual(record["resources"]["gpu_type"], "rtx3090")
        self.assertEqual(record["resources"]["parallel"], 4)
        self.assertEqual(record["resources"]["gpus"], 1)
        self.assertEqual(record["submissions"][-1]["job_id"], "123")
        self.assertEqual(research, (folder / "research.yaml").read_bytes())

    def test_other_gpu_types_and_aihub_typed_requests(self):
        for backend, gpu_type, flag in (("--scrp", "rtx3060", "--gpus=rtx3060:1"),
                                        ("--scrp", "any", "--gpus=1"),
                                        ("--aihub", "rtx3090", "--gres=gpu:rtx3090:1")):
            with self.subTest(backend=backend, gpu_type=gpu_type):
                folder = launcher.main(["nn3", backend, "--sample", "p05", "--years", "1987",
                                       "--gpu-type", gpu_type, "--root", str(self.root / "results"), "--dry-run"])
                command = launcher.read_launch(folder)["submissions"][-1]["sbatch_command"]
                self.assertIn(flag, command)
                self.assertFalse(any(flag.startswith("--constraint=") for flag in command))
                if backend == "--scrp":
                    self.assertIn("--partition=scrp", command)
                else:
                    self.assertFalse(any(flag.startswith("--partition=") for flag in command))

    def test_legacy_gpu_defaults_and_type_changes_select_the_correct_partition(self):
        folder = self.prepare("nn1", "--scrp", "1987", "p05")
        record = launcher.read_launch(folder)
        record["resources"].pop("gpu_type")  # Records prepared before GPU selection existed.
        atomic_write(folder / "launch.json", json.dumps(record))
        launcher.main(["resume", str(folder), "--dry-run"])
        command = launcher.read_launch(folder)["submissions"][-1]["sbatch_command"]
        self.assertIn("--gpus=a100:1", command)
        self.assertIn("--constraint=a100", command)
        for gpu_type, override, partition in (("rtx3090", [], "scrp"),
                                               ("rtx3060", ["--partition", "large"], "large"),
                                               ("a100", [], "a100")):
            with self.subTest(gpu_type=gpu_type):
                launcher.main(["resume", str(folder), "--gpu-type", gpu_type, *override, "--dry-run"])
                launcher.main(["resume", str(folder), "--dry-run"])
                command = launcher.read_launch(folder)["submissions"][-1]["sbatch_command"]
                self.assertIn(f"--partition={partition}", command)
                self.assertIn(f"--gpus={gpu_type}:1", command)
                self.assertEqual("--constraint=a100" in command, gpu_type == "a100")

    def test_gpu_type_and_count_errors_do_not_prepare_experiments(self):
        for model, backend, extra in (("pcr", "--scrp", ["--gpu-type", "rtx3090"]),
                                       ("nn1", "--local", ["--gpu-type", "rtx3090"]),
                                       ("nn1", "--scrp", ["--gpus", "4"])):
            with self.subTest(model=model, backend=backend), self.assertRaises(ValueError):
                launcher.main([model, backend, "--sample", "p05", "--years", "1987",
                               "--root", str(self.root / "invalid"), "--dry-run", *extra])
        self.assertFalse((self.root / "invalid").exists())

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
                parallel = 1 if model.startswith("nn") else 4
                partition = "a100" if model.startswith("nn") else "scrp"
                for flag in ("--account=pg", f"--partition={partition}", "--qos=c32g4", f"--array=0%{parallel}"):
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

    def test_remove_preserves_files_and_hides_only_selected_copy(self):
        original = self.prepare(backend="--local", years="1987")
        copy = original.with_name("friendly-copy")
        shutil.copytree(original, copy)
        (copy / "predictions.csv").write_text("observed,prediction\n1,2\n")
        before = {str(p.relative_to(copy)): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
        destination = launcher.main(["remove", str(copy), "--root", str(original.parent)])
        self.assertEqual(destination, (self.root / "results_archive" / "friendly-copy").resolve())
        self.assertFalse(copy.exists())
        self.assertTrue(original.exists())
        for name, data in before.items():
            self.assertEqual((destination / name).read_bytes(), data)
        self.assertEqual(len(launcher.list_experiments(original.parent)), 1)
        destination.rename(copy)  # Restoring the original path preserves the record.
        self.assertEqual(len(launcher.list_experiments(original.parent)), 2)

    def test_remove_checks_slurm_and_refuses_unavailable_or_active_queue(self):
        folder = self.prepare(years="1987")
        record = launcher.read_launch(folder)
        record["submissions"].append({"state": "submitted", "job_id": "123"})
        atomic_write(folder / "launch.json", json.dumps(record))
        with patch("run_experiment.subprocess.run", side_effect=FileNotFoundError("squeue")):
            with self.assertRaisesRegex(ValueError, "Cannot verify Slurm"):
                launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "123\n", "")):
            with self.assertRaisesRegex(ValueError, "queued/running"):
                launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        self.assertTrue(folder.exists())
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            destination = launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        self.assertTrue(destination.is_dir())

    def test_remove_refuses_collision_missing_and_outside_root(self):
        folder = self.prepare(years="1987")
        with self.assertRaisesRegex(ValueError, "not found"):
            launcher.main(["remove", "missing", "--root", str(folder.parent)])
        with self.assertRaisesRegex(ValueError, "direct experiment"):
            launcher.main(["remove", str(folder), "--root", str(self.root / "elsewhere")])
        destination = self.root / "results_archive" / folder.name
        destination.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        self.assertTrue(folder.exists())

    def test_remove_refuses_uncertain_submissions_and_invalid_records(self):
        folder = self.prepare(years="1987")
        record = launcher.read_launch(folder)
        record["submissions"].append({"state": "unknown"})
        atomic_write(folder / "launch.json", json.dumps(record))
        with self.assertRaisesRegex(ValueError, "uncertain"):
            launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        record["backend"] = "invalid"
        atomic_write(folder / "launch.json", json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Invalid launcher"):
            launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        self.assertTrue(folder.exists())

    def test_remove_refuses_held_worker_lock(self):
        folder = self.prepare(backend="--local", years="1987")
        year = folder / "annual/years/1987"
        year.mkdir(parents=True, exist_ok=True)
        with run_lock(year):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                launcher.main(["remove", folder.name, "--root", str(folder.parent)])
        self.assertTrue(folder.exists())

    def test_remove_aihub_uses_its_root_and_keeps_archive_separate(self):
        folder = self.prepare(backend="--aihub", years="1987")
        with patch.dict("os.environ", {}, clear=True), patch.object(launcher, "ROOT", self.root):
            hub_root = self.root / "results/aihub_experiments"
            hub_root.mkdir()
            moved = folder.rename(hub_root / folder.name)
            destination = launcher.main(["remove", moved.name, "--aihub"])
        self.assertEqual(destination, (hub_root.with_name("aihub_experiments_archive") / folder.name).resolve())
        self.assertTrue(destination.exists())


if __name__ == "__main__":
    unittest.main()
