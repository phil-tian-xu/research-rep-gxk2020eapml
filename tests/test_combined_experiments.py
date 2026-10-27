"""Split execution, integrity failures, pooled scores and downstream reports."""

import importlib.util
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from pandas.testing import assert_frame_equal

import test_annual_jobs as fixtures
import run_experiment as launcher
from core.annual_jobs import collect, prepare, run_task
from core.combined_experiments import (collect_combination, load_combination, resolve_annual_run,
                                       source_directory, source_file_hashes)
from core.figure4_report import make_report


class CombinedExperimentTests(unittest.TestCase):
    setUp = fixtures.AnnualJobsTests.setUp

    def source(self, name, first, last, backend="local", gpu="any"):
        folder = self.root / name
        folder.mkdir()
        with patch("core.annual_jobs.load_config", return_value=deepcopy(self.cfg)):
            prepare("fixture", first, last, folder / "annual")
        (folder / "launch.json").write_text(json.dumps(dict(schema_version=1, id=name,
            model=self.cfg.RUN.MODELS.ENABLED[0], sample="fixture", years=[first, last],
            backend=backend, resources=dict(parallel=2 if backend != "local" else 1, cpus=1,
                gpus=1 if backend != "local" else 0, gpu_type=gpu if backend != "local" else None,
                mem="8G", time="00:30:00", account="pg", partition="a100" if gpu == "a100" else "scrp", qos="c32g4"),
            submissions=[])))
        return folder

    def combine(self, left, right):
        return launcher.main(["combine", str(left), str(right), "--years", "2003:2004", "--root", str(self.root)])

    def completed_pair(self):
        left, right = self.source("left", 2003, 2003), self.source("right", 2004, 2004)
        group = self.combine(left, right)
        for parent in (right, left):
            run_task(parent / "annual", 0)
        return left, right, group

    def test_one_id_local_resume_collect_matches_unsplit_and_figure4(self):
        left, right = self.source("left", 2003, 2003), self.source("right", 2004, 2004)
        group = self.combine(left, right)
        status, = launcher.main(["status", str(group)])
        self.assertEqual(status["saved"], "0/2")

        def worker(command, **kwargs):
            self.assertIn("core.annual_jobs", command)
            run_task(Path(command[command.index("--experiment") + 1]),
                     int(command[command.index("--task-index") + 1]))
            return subprocess.CompletedProcess(command, 0)

        with patch("run_experiment.subprocess.run", side_effect=worker) as calls:
            launcher.main(["resume", str(group)])
        self.assertEqual(calls.call_count, 2)
        status, = launcher.main(["status", str(group)])
        self.assertEqual((status["saved"], status["reports"]), ("2/2", "ready"))
        output = launcher.main(["collect", str(group)])
        with patch("run_experiment.execute", side_effect=AssertionError("must skip")):
            launcher.main(["resume", str(group)])
        for index in (0, 1):
            run_task(self.experiment, index)
        reference = collect(self.experiment)
        for name in ("table1_all.csv", "complexity.csv", "tuning_curves.csv"):
            assert_frame_equal(pd.read_csv(output / name), pd.read_csv(reference / name))
        sources = pd.read_csv(output / "year_sources.csv")
        self.assertEqual(sources.test_year.tolist(), [2003, 2004])
        self.assertEqual(sources.source_experiment.tolist(), ["left", "right"])
        for year in (2003, 2004):
            self.assertTrue((resolve_annual_run(group, year) / "manifest.json").is_file())
        with patch("core.models.dimension_reduction.PCRRegressor.fit", side_effect=AssertionError("refit")):
            result = make_report([group], self.root / "panel", self.root / "reports", [2003, 2004], batch_size=7)
        self.assertEqual(pd.read_csv(result / "verification.csv").year.tolist(), [2003, 2004])
        from core.figure5_report import make_report as figure5
        figure5([result], self.root / "figure5")
        # Completed sources also support the direct multi-input collect interface.
        combined_again = launcher.main(["collect", str(left), str(right), "--years", "2003:2004",
                                        "--root", str(self.root / "groups")])
        assert_frame_equal(pd.read_csv(output / "table1_all.csv"), pd.read_csv(combined_again / "table1_all.csv"))
        with self.assertRaisesRegex(ValueError, "referenced"):
            launcher.remove_experiment(left.name, self.root)

    def test_registration_rejects_overlapping_missing_extra_or_incompatible_plans(self):
        left = self.source("left", 2003, 2003)
        for name, first, last in (("overlap", 2003, 2004), ("missing", 2003, 2003)):
            with self.assertRaisesRegex(ValueError, "year coverage"):
                self.combine(left, self.source(name, first, last))
        right = self.source("right", 2004, 2004)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            launcher.main(["combine", str(left), str(right), "--years", "2003", "--root", str(self.root)])
        plan_file = right / "annual/experiment.json"
        original = json.loads(plan_file.read_text())
        altered = json.loads(plan_file.read_text())
        altered["config"]["RUN"]["SPLIT"]["VALIDATION_YEARS"] = 2
        plan_file.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            self.combine(left, right)
        original["runtime"]["python"] = "different"
        plan_file.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "execution identities"):
            self.combine(left, right)

    def test_missing_duplicate_corrupt_and_edited_source_rejected_without_publishing(self):
        left, right = self.source("left", 2003, 2003), self.source("right", 2004, 2004)
        group = self.combine(left, right)
        run_task(left / "annual", 0)
        with self.assertRaisesRegex(ValueError, "Missing/incomplete"):
            collect_combination(group)
        self.assertFalse((group / "annual/combined").exists())
        run_task(right / "annual", 0)
        run = resolve_annual_run(right, 2004)
        shutil.copytree(run, run.parent / "duplicate")
        with self.assertRaisesRegex(ValueError, "found 2"):
            collect_combination(group)
        shutil.rmtree(run.parent / "duplicate")
        checkpoint = run / "years/2004/PCR.json"
        original = checkpoint.read_text()
        envelope = json.loads(original)
        envelope["sha256"] = "corrupt"
        checkpoint.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "checksum"):
            collect_combination(group)
        checkpoint.write_text(original)
        plan_file = right / "annual/experiment.json"
        plan = json.loads(plan_file.read_text())
        plan["config"]["RUN"]["SPLIT"]["FIRST_TRAIN_YEAR"] += 1
        plan_file.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "configurations differ|plan changed"):
            collect_combination(group)

    def test_dataset_runtime_feature_order_and_score_mismatches_rejected(self):
        left, right, group = self.completed_pair()
        run = resolve_annual_run(right, 2004)
        manifest_file = run / "manifest.json"
        original = manifest_file.read_text()
        for field, message in (("dataset", "Dataset identities"), ("runtime", "Execution identity")):
            manifest = json.loads(original)
            manifest[field]["changed"] = True
            manifest_file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, message):
                collect_combination(group)
        manifest_file.write_text(original)
        from core.run_state import atomic_write
        import hashlib
        checkpoint = run / "years/2004/PCR.json"
        original = checkpoint.read_text()
        for field in ("features", "score", "date", "count"):
            envelope = json.loads(original)
            record = json.loads(envelope["payload"])
            if field == "features":
                record["tuning"]["feature_names"].reverse()
            elif field == "score":
                record["metrics"]["test_loss"] += 1
            elif field == "date":
                record["predictions"]["return_month"][0] = "2005-01"
            else:
                record["metrics"]["n_test"] += 1
            payload = json.dumps(record)
            atomic_write(checkpoint, json.dumps(dict(payload=payload, sha256=hashlib.sha256(payload.encode()).hexdigest())))
            with self.assertRaisesRegex(ValueError, "Feature order|metrics disagree|panel/count/date"):
                collect_combination(group)
        checkpoint.write_text(original)

    def test_gpu_resume_keeps_two_resource_pools_skips_active_and_guards_unknown(self):
        self.cfg.RUN.MODELS = launcher.load_config("gkx_nn1_v1").RUN.MODELS
        left = self.source("early", 2003, 2003, "scrp", "rtx3090")
        right = self.source("late", 2004, 2004, "scrp", "a100")
        group = self.combine(left, right)

        def slurm(command, **kwargs):
            if command[0] == "squeue":
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 0, "1001\n" if "--gpus=rtx3090:1" in command else "1002\n", "")

        with patch("run_experiment.subprocess.run", side_effect=slurm) as calls:
            launcher.main(["resume", str(group)])
        submitted = [c.args[0] for c in calls.call_args_list if c.args[0][0] == "sbatch"]
        self.assertEqual(len(submitted), 2)
        self.assertIn("--gpus=rtx3090:1", submitted[0])
        self.assertIn("--gpus=a100:1", submitted[1])
        self.assertTrue(all("--array=0%2" in cmd for cmd in submitted))
        status, = launcher.main(["status", str(group)])
        self.assertEqual(status["job_ids"], "1001,1002")
        with patch("run_experiment.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "1001\n1002\n", "")) as calls:
            launcher.main(["resume", str(group)])
        self.assertTrue(all(c.args[0][0] == "squeue" for c in calls.call_args_list))
        with self.assertRaisesRegex(ValueError, "preserves each source"):
            launcher.main(["resume", str(group), "--parallel", "4"])
        record = launcher.read_launch(right)
        record["submissions"][-1]["state"] = "unknown"
        (right / "launch.json").write_text(json.dumps(record))
        with patch("run_experiment.subprocess.run", side_effect=slurm) as calls:
            with self.assertRaisesRegex(ValueError, "uncertain"):
                launcher.main(["resume", str(group)])
        self.assertFalse(any(c.args[0][0] == "sbatch" for c in calls.call_args_list))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional NN dependency")
    def test_nn_split_training_summary_and_pooling_match_unsplit(self):
        self.cfg.RUN.MODELS = launcher.load_config("gkx_nn1_v1").RUN.MODELS
        settings = self.cfg.RUN.MODELS.NN
        settings.DEVICE = "cpu"
        settings.LEARNING_RATE_GRID = [.001, .01]
        settings.L1_GRID = [.00001]
        settings.ENSEMBLE_SIZE = 2
        settings.MAX_EPOCHS = 3
        settings.PATIENCE = 2
        settings.BATCH_SIZE = 16
        left, right, group = self.completed_pair()
        output = collect_combination(group)
        summary = pd.read_csv(output / "nn_training_summary.csv")
        self.assertEqual(len(summary), 4)
        self.assertTrue((summary.parameter_update_l2 > 0).all())
        self.assertTrue((summary.nonzero_dense_weights > 0).all())
        self.assertTrue((summary.best_epoch <= summary.epochs_run).all())
        # One uninterrupted two-year annual plan, using the identical NN settings.
        full = self.source("full", 2003, 2004)
        for i in (0, 1):
            run_task(full / "annual", i)
        reference = collect(full / "annual")
        assert_frame_equal(pd.read_csv(output / "table1_all.csv"), pd.read_csv(reference / "table1_all.csv"))
        observed, predicted = [], []
        for parent, year in ((left, 2003), (right, 2004)):
            run = resolve_annual_run(parent, year)
            payload = json.loads(json.loads((run / f"years/{year}/NN1.json").read_text())["payload"])
            observed.extend(payload["predictions"]["observed"])
            predicted.extend(payload["predictions"]["prediction"])
        y, forecast = np.array(observed), np.array(predicted)
        self.assertAlmostEqual(pd.read_csv(output / "table1_all.csv").r2_zero.iloc[0], 1 - np.sum((y-forecast)**2)/(y@y))
        self.assertTrue(all((group / p).is_file() for p in pd.read_csv(output / "year_sources.csv").training_csv))


    def snapshot(self, folder):
        return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in folder.rglob("*") if p.is_file()}

    def test_all_results_copied_and_originals_unchanged_on_first_and_repeated_collect(self):
        left, right, group = self.completed_pair()
        for parent in (left, right):
            (parent / "research.yaml").write_text(self.cfg.dump())
            log = parent / "annual/logs/worker/run.log"
            log.parent.mkdir(parents=True)
            log.write_text("training finished\n")
            script = parent / "submissions/attempt/run.sh"
            script.parent.mkdir(parents=True)
            script.write_text("saved script\n")
        before = [self.snapshot(p) for p in (left, right)]
        output = collect_combination(group)
        published = self.snapshot(group)
        plan = load_combination(group)
        manifest = json.loads((group / "sources/manifest.json").read_text())
        for original, source, entry in zip((left, right), plan["sources"], manifest["sources"]):
            stored = source_directory(group, source)
            self.assertEqual(source_file_hashes(original), source_file_hashes(stored))
            self.assertEqual(entry["files"], source_file_hashes(stored))
            self.assertFalse(list(stored.rglob(".run.lock")))
            self.assertTrue((stored / "annual/logs/worker/run.log").is_file())
        table = pd.read_csv(output / "year_sources.csv")
        for row in table.itertuples():
            self.assertTrue((group / row.checkpoint).is_file())
            self.assertFalse(Path(row.checkpoint).is_absolute())
            self.assertTrue(Path(row.original_checkpoint).is_file())
        with patch("core.combined_experiments.shutil.copytree", side_effect=AssertionError("recopy")):
            self.assertEqual(collect_combination(group), output)
        self.assertEqual(published, self.snapshot(group))
        self.assertEqual(before, [self.snapshot(p) for p in (left, right)])

    def test_downloaded_folder_status_resume_collection_and_figures_need_no_originals(self):
        left, right, group = self.completed_pair()
        output = collect_combination(group)
        expected = self.snapshot(group)
        downloaded = self.root / "download" / group.name
        shutil.copytree(group, downloaded)
        # Only synthetic fixture sources are made unavailable; never real results.
        left.rename(self.root / "offline-left")
        right.rename(self.root / "offline-right")
        with patch("run_experiment.execute", side_effect=AssertionError("refit")), \
                patch("run_experiment.runtime_identity", side_effect=AssertionError("fit environment")):
            status, = launcher.main(["status", str(downloaded)])
            self.assertEqual((status["saved"], status["reports"]), ("2/2", "ready"))
            launcher.main(["resume", str(downloaded)])
            result = launcher.main(["collect", str(downloaded)])
        self.assertEqual(expected, self.snapshot(downloaded))
        assert_frame_equal(pd.read_csv(result / "table1_all.csv"), pd.read_csv(output / "table1_all.csv"))
        with patch("core.models.dimension_reduction.PCRRegressor.fit", side_effect=AssertionError("refit")):
            figure4 = make_report([downloaded], self.root / "panel", self.root / "reports", [2003, 2004], batch_size=7)
        self.assertEqual(pd.read_csv(figure4 / "verification.csv").year.tolist(), [2003, 2004])
        from core.figure5_report import make_report as figure5
        figure5([figure4], self.root / "figure5")

    def test_existing_published_combination_upgrades_without_changing_scores_or_provenance(self):
        left, right, group = self.completed_pair()
        output = collect_combination(group)
        identity = (output / "source_identity.json").read_bytes()
        plan = (group / "collection.json").read_bytes()
        reports = {p.name: p.read_bytes() for p in output.iterdir()
                   if p.name not in {"year_sources.csv", "comparison_notes.txt"}}
        # Reconstruct the previous linked layout, with an already-published report.
        shutil.rmtree(group / "sources")
        launch = launcher.read_launch(group)
        launch.pop("source_storage")
        (group / "launch.json").write_text(json.dumps(launch))
        legacy = json.loads(identity)["sources"]
        (output / "year_sources.csv").write_text(pd.DataFrame(legacy).to_csv(index=False))
        before = [self.snapshot(p) for p in (left, right)]
        with patch("core.models.dimension_reduction.PCRRegressor.fit", side_effect=AssertionError("refit")):
            launcher.main(["collect", str(group)])
        self.assertTrue((group / "sources/manifest.json").is_file())
        self.assertEqual(identity, (output / "source_identity.json").read_bytes())
        self.assertEqual(plan, (group / "collection.json").read_bytes())
        for name, content in reports.items():
            self.assertEqual(content, (output / name).read_bytes())
        self.assertEqual(before, [self.snapshot(p) for p in (left, right)])

    def test_failed_or_corrupt_copy_publishes_nothing_and_can_retry(self):
        left, right, group = self.completed_pair()
        before = [self.snapshot(p) for p in (left, right)]
        copytree = shutil.copytree

        def copying(src, dst, *args, **kwargs):
            result = copytree(src, dst, *args, **kwargs)
            if (Path(dst) / "launch.json").is_file():
                (Path(dst) / "launch.json").write_text("corrupt copy")
            return result

        for error in (OSError("out of space"), None):
            if error is not None:
                patcher = patch("core.combined_experiments.shutil.copytree", side_effect=error)
            else:
                patcher = patch("core.combined_experiments.shutil.copytree", side_effect=copying)
            with patcher, self.assertRaisesRegex((OSError, ValueError), "out of space|checksum mismatch"):
                collect_combination(group)
            self.assertFalse((group / "sources").exists())
            self.assertFalse((group / "annual/combined").exists())
            self.assertFalse(list(group.glob(".sources-*")))
            self.assertEqual(before, [self.snapshot(p) for p in (left, right)])
        collect_combination(group)

    def test_damaged_embedded_results_never_fall_back_or_submit_training(self):
        left, right, group = self.completed_pair()
        output = collect_combination(group)
        row = pd.read_csv(output / "year_sources.csv").iloc[0]
        checkpoint = group / row.checkpoint
        content = checkpoint.read_bytes()
        before = [self.snapshot(p) for p in (left, right)]
        checkpoint.write_bytes(content + b"\n")  # Valid JSON, but not the verified copy.
        with self.assertRaisesRegex(ValueError, "checksum/file inventory"):
            collect_combination(group)
        checkpoint.unlink()
        with patch("run_experiment.execute", side_effect=AssertionError("submission")):
            with self.assertRaisesRegex(ValueError, "Embedded source is incomplete"):
                launcher.main(["resume", str(group)])
        checkpoint.write_bytes(content)
        unexpected = checkpoint.parent / "unexpected.csv"
        unexpected.write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "checksum/file inventory"):
            collect_combination(group)
        unexpected.unlink()
        collect_combination(group)
        self.assertEqual(before, [self.snapshot(p) for p in (left, right)])

    def test_missing_entire_bundle_does_not_revert_to_originals(self):
        left, right, group = self.completed_pair()
        collect_combination(group)
        before = [self.snapshot(p) for p in (left, right)]
        (group / "sources").rename(group / "unavailable-sources")
        with patch("run_experiment.execute", side_effect=AssertionError("submission")):
            with self.assertRaisesRegex(ValueError, "source bundle is missing"):
                launcher.main(["resume", str(group)])
            with self.assertRaisesRegex(ValueError, "source bundle is missing"):
                collect_combination(group)
        self.assertEqual(before, [self.snapshot(p) for p in (left, right)])

    def test_combined_folder_inside_source_is_rejected_before_recursive_copy(self):
        left, right, group = self.completed_pair()
        nested = left / "nested-combined"
        shutil.copytree(group, nested)
        with self.assertRaisesRegex(ValueError, "outside every original source"):
            collect_combination(nested)
        self.assertFalse((nested / "sources").exists())
        self.assertFalse(list(nested.glob(".sources-*")))

    def test_busy_source_is_rejected_without_changing_originals(self):
        left, right, group = self.completed_pair()
        run = resolve_annual_run(left, 2003)
        from core.run_state import run_lock
        before = [self.snapshot(p) for p in (left, right)]
        with run_lock(run):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                collect_combination(group)
        self.assertFalse((group / "sources").exists())
        self.assertEqual(before, [self.snapshot(p) for p in (left, right)])

    def test_posix_cluster_provenance_is_portable_on_windows(self):
        left, right, group = self.completed_pair()
        output = collect_combination(group)
        expected = pd.read_csv(output / "table1_all.csv")
        mapping = {str(p): "/data/users/test/experiments/" + p.name for p in (left, right)}

        def foreign_paths(value):
            if isinstance(value, dict):
                return {k: foreign_paths(v) for k, v in value.items()}
            if isinstance(value, list):
                return [foreign_paths(v) for v in value]
            if isinstance(value, str):
                for original, foreign in mapping.items():
                    if value.startswith(original):
                        return foreign + value[len(original):].replace("\\", "/")
            return value

        bundle = json.loads((group / "sources/manifest.json").read_text())
        for entry in bundle["sources"]:
            parent = group / "sources" / entry["directory"]
            for path in parent.rglob("*"):
                if path.suffix == ".yaml":
                    path.write_text(yaml.safe_dump(foreign_paths(yaml.safe_load(path.read_text()))))
                elif path.name in {"experiment.json", "manifest.json", "launch.json"}:
                    path.write_text(json.dumps(foreign_paths(json.loads(path.read_text()))))
            entry["experiment"] = foreign_paths(entry["experiment"])
            entry["files"] = source_file_hashes(parent)
        plan = foreign_paths(json.loads((group / "collection.json").read_text()))
        (group / "collection.json").write_text(json.dumps(plan))
        bundle["collection_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        (group / "sources/manifest.json").write_text(json.dumps(bundle))
        identity = foreign_paths(json.loads((output / "source_identity.json").read_text()))
        (output / "source_identity.json").write_text(json.dumps(identity))
        launcher.main(["collect", str(group)])
        status, = launcher.main(["status", str(group)])
        self.assertEqual(status["saved"], "2/2")
        assert_frame_equal(expected, pd.read_csv(output / "table1_all.csv"))
        self.assertTrue(all((group / p).is_file() for p in pd.read_csv(output / "year_sources.csv").checkpoint))


if __name__ == "__main__":
    unittest.main()
