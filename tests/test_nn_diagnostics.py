"""Saved-fit diagnostics: independent arithmetic, collapse cases and source preservation."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import yaml

import run_experiment as launcher
from core.combined_experiments import combination_plan, source_file_hashes
from core.nn_diagnostics import diagnostic_tables, make_report, parameter_summary
from core.models.nn_architectures import ARCHITECTURES
from core.run_state import create_run, runtime_identity, save_checkpoint
from utils.general_utils import load_config


def saved_record(year=2003):
    seeds = [7, 8]
    tensors = []
    members = []
    for seed, weight in zip(seeds, (1e-9, .02)):
        tensors.append({"layers.0.weight": np.full((32, 2), weight).tolist(),
            "layers.0.bias": [0.] * 32, "layers.2.weight": [1.] * 32,
            "layers.2.bias": [0.] * 32, "layers.2.running_mean": [0.] * 32,
            "layers.2.running_var": [1.] * 32, "layers.2.num_batches_tracked": 4,
            "layers.3.weight": np.full((1, 32), weight).tolist(), "layers.3.bias": [0.]})
        history = [dict(epoch=i + 1, train_mse=t, validation_mse=v, online_half_mse=h,
                        online_l1_penalty=p, first_batch_gradient_l2=g,
                        train_prediction_std=.01, validation_prediction_std=.02)
                   for i, (t, v, h, p, g) in enumerate(zip(
                       (1., .4, .5), (1., .6, .8), (.7, .2, .25), (.1, .05, .03), (.3, .1, .05)))]
        members.append(dict(seed=seed, best_epoch=2, epochs_run=3, stop_reason="patience",
            best_validation_mse=.6, initial_train_mse=1.2, initial_validation_mse=1.3,
            parameter_update_l2=.1, dense_weight_l1=96 * weight,
            nonzero_dense_weights=96, dense_weight_count=96, history=history))
    candidates = [dict(params=dict(learning_rate=rate, l1=.0001), validation_loss=score,
                       prediction_mean=.01, prediction_std=.005, members=deepcopy(members))
                  for rate, score in ((.001, .4), (.01, .3))]
    tuning = dict(specification="NN1", test_year=year, hidden_sizes=[32], seeds=seeds,
        feature_names=["a", "b"], best_params=candidates[1]["params"], validation_results=candidates,
        nn_state=dict(version=1, dtype="float32", n_features=2, hidden_sizes=[32], feature_names=None, members=tensors),
        objective="mean_half_squared_error_plus_l1_dense_kernels", return_units="decimal",
        validation_baselines=dict(zero=.7, training_mean=.65))
    return dict(metrics=dict(specification="NN1", test_year=year, n_test=4, best_validation_loss=.3, r2_zero=0.),
        tuning=tuning, predictions=dict(specification=["NN1"] * 4, test_year=[year] * 4,
            row_position=[1, 2, 3, 4], return_month=[f"{year}-01"] * 2 + [f"{year}-02"] * 2,
            observed=[.1, -.1, .2, -.2], prediction=[.01, .01, .02, .02]))


class NNDiagnosticTablesTests(unittest.TestCase):
    def test_loss_decomposition_retained_epoch_and_ensemble_selection(self):
        tables = diagnostic_tables(saved_record())
        selected = tables["members"].loc[lambda x: x.selected]
        self.assertEqual(selected.candidate.tolist(), [2, 2])
        assert_allclose(selected.retained_train_mse, .4)
        assert_allclose(selected.final_epoch_train_mse, .5)
        assert_allclose(selected.retained_validation_mse, .6)
        self.assertEqual(tables["candidates"].loc[lambda x: x.selected, "ensemble_validation_mse"].item(), .3)
        epochs = tables["epochs"].loc[lambda x: x.selected & x.seed.eq(7)]
        assert_allclose(epochs.online_objective, [.8, .25, .28])
        self.assertEqual(epochs.loc[epochs.retained, "epoch"].item(), 2)
        self.assertTrue(selected.diagnostic_warnings.eq("").all())

    def test_tensor_counts_near_zero_and_bn_are_separate(self):
        table = parameter_summary(saved_record()["tuning"]["nn_state"], [7, 8])
        tiny = table.loc[table.seed.eq(7) & table.kind.eq("dense_kernel")]
        self.assertEqual(tiny["count"].sum(), 96)
        self.assertTrue(tiny.zero_fraction.eq(0).all())
        self.assertTrue(tiny["near_zero_fraction_1e-08"].eq(1).all())
        assert_allclose(tiny.l1_norm.sum(), 9.6e-8)
        self.assertTrue(table.loc[table.kind.eq("dense_bias"), "zero_fraction"].eq(1).all())
        self.assertFalse(table.loc[table.kind.str.startswith("bn_running"), "trainable"].any())

    def test_collapsed_weights_and_missing_updates_are_flags_not_exclusions(self):
        record = saved_record()
        for tensor in ("layers.0.weight", "layers.3.weight"):
            record["tuning"]["nn_state"]["members"][0][tensor] = np.zeros_like(
                record["tuning"]["nn_state"]["members"][0][tensor]).tolist()
        member = record["tuning"]["validation_results"][1]["members"][0]
        member.update(nonzero_dense_weights=0, dense_weight_l1=0., parameter_update_l2=0.)
        for epoch in member["history"]:
            epoch.update(train_prediction_std=0., validation_prediction_std=0., first_batch_gradient_l2=0.)
        tables = diagnostic_tables(record)
        retained = tables["members"].loc[lambda x: x.selected & x.seed.eq(7)].iloc[0]
        for warning in ("all_dense_weights_zero", "zero_parameter_update", "zero_dense_kernel:output",
                        "near_constant_train_prediction", "all_recorded_first_batch_gradients_zero"):
            self.assertIn(warning, retained.diagnostic_warnings)
        self.assertEqual(len(tables["members"].loc[lambda x: x.selected]), 2)

    def test_zero_output_kernel_detected_when_hidden_weights_remain_nonzero(self):
        record = saved_record()
        record["tuning"]["nn_state"]["members"][1]["layers.3.weight"] = [[0.] * 32]
        record["tuning"]["validation_results"][1]["members"][1].update(
            nonzero_dense_weights=64, dense_weight_l1=64 * .02)
        row = diagnostic_tables(record)["members"].loc[lambda x: x.selected & x.seed.eq(8)].iloc[0]
        self.assertIn("zero_dense_kernel:output", row.diagnostic_warnings)
        self.assertNotIn("all_dense_weights_zero", row.diagnostic_warnings)

    def test_zero_bn_scales_are_detected_without_zero_dense_weights(self):
        record = saved_record()
        record["tuning"]["nn_state"]["members"][0]["layers.2.weight"] = [0.] * 32
        row = diagnostic_tables(record)["members"].loc[lambda x: x.selected & x.seed.eq(7)].iloc[0]
        self.assertIn("zero_bn_scale:hidden_1", row.diagnostic_warnings)
        self.assertNotIn("all_dense_weights_zero", row.diagnostic_warnings)

    def test_nn6_to_nn10_layer_layouts_without_torch_or_fitting(self):
        for depth in range(6, 11):
            with self.subTest(depth=depth):
                widths = [2, *ARCHITECTURES[f"NN{depth}"], 1]
                member = {}
                expected_count = 0
                for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
                    member[f"layers.{3 * index}.weight"] = np.full((outputs, inputs), .02).tolist()
                    member[f"layers.{3 * index}.bias"] = [0.] * outputs
                    expected_count += inputs * outputs
                    if index < depth:
                        for suffix in ("weight", "bias", "running_mean", "running_var"):
                            value = 1. if suffix in {"weight", "running_var"} else 0.
                            member[f"layers.{3 * index + 2}.{suffix}"] = [value] * outputs
                        member[f"layers.{3 * index + 2}.num_batches_tracked"] = 2
                state = dict(version=1, dtype="float32", n_features=2,
                             hidden_sizes=widths[1:-1], members=[member])
                table = parameter_summary(state, [7])
                kernels = table.loc[table.kind.eq("dense_kernel")]
                self.assertEqual((len(kernels), kernels["count"].sum()), (depth + 1, expected_count))

    def test_monthly_constant_forecasts_not_hidden_by_time_variation(self):
        table = diagnostic_tables(saved_record())["prediction_variation"]
        self.assertFalse(table.iloc[0].near_constant)
        self.assertTrue(table.iloc[1:].constant.all())
        assert_allclose(table.iloc[0].prediction_std, .005)

    def test_invalid_state_history_and_selection_are_rejected(self):
        mutations = [lambda r: r["tuning"]["nn_state"]["members"][0].update({"layers.0.weight": [[0.]]}),
                     lambda r: r["tuning"]["validation_results"][0]["members"][0].update(best_epoch=3),
                     lambda r: r["tuning"].update(best_params=dict(learning_rate=.001, l1=.0001)),
                     lambda r: r["tuning"]["nn_state"]["members"][0].update({"layers.3.bias": [float("nan")]}),
                     lambda r: r["tuning"]["validation_results"][1]["members"][0].update(nonzero_dense_weights=0)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = saved_record()
                mutate(record)
                with self.assertRaises(ValueError):
                    diagnostic_tables(record)


class NNDiagnosticReportTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        quiet = redirect_stdout(StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def fitted(self, parent, year=2003):
        cfg = load_config("gkx_nn1_v1")
        cfg.DATA.DATASET = "fixture"
        cfg.RUN.SPLIT.FIRST_TEST_YEAR = cfg.RUN.SPLIT.LAST_TEST_YEAR = year
        cfg.RUN.OUTPUT_DIR = str(parent)
        run = create_run(cfg, dict(name="fixture", panel_sha256="a" * 64, metadata_sha256="b" * 64))
        record = saved_record(year)
        save_checkpoint(run, record["metrics"], pd.DataFrame(record["predictions"]), record["tuning"])
        return run, cfg

    def test_report_and_cli_preserve_source_bytes_rng_and_execution_identity(self):
        run, _ = self.fitted(self.root / "fits")
        before = source_file_hashes(run)
        runtime = runtime_identity()
        rng = np.random.get_state()
        with patch.dict(sys.modules, {"torch": None}):
            output = launcher.main(["diagnose-nn", str(run), "--years", "2003", "--candidate", "1",
                                    "--output-root", str(self.root / "reports")])
        self.assertEqual(source_file_hashes(run), before)
        self.assertEqual(runtime_identity(), runtime)
        self.assertEqual(np.random.get_state()[0], rng[0])
        assert_allclose(np.random.get_state()[1], rng[1])
        metadata = json.loads((output / "methodology.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        summary = pd.read_csv(output / "summary.csv").iloc[0]
        self.assertEqual((summary.selected_candidate, summary.plotted_candidate), (2, 1))
        self.assertEqual(summary.near_constant_test_months, 2)
        self.assertTrue((output / "index.html").is_file())
        self.assertIn("NN1 · 2003", (output / "2003/index.html").read_text(encoding="utf-8"))
        for name in ("learning_curves", "training_objective", "optimization", "prediction_variation", "hyperparameters", "parameters"):
            self.assertGreater((output / "2003" / f"{name}.png").stat().st_size, 1000)
        self.assertIn("first batch", metadata["gradient_scope"])

    def test_invalid_source_or_output_does_not_publish_a_complete_report(self):
        run, _ = self.fitted(self.root / "fits")
        with self.assertRaisesRegex(ValueError, "outside"):
            make_report(run, run / "diagnostics")
        with self.assertRaisesRegex(ValueError, "horizon"):
            make_report(run, self.root / "reports", [2004])
        checkpoint = run / "years/2003/NN1.json"
        envelope = json.loads(checkpoint.read_text())
        envelope["sha256"] = "bad"
        checkpoint.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "checksum"):
            make_report(run, self.root / "reports")
        self.assertFalse(list((self.root / "reports").glob("*/index.html")))

    def test_relocated_combined_bundle_and_frozen_inventory(self):
        sources = []
        for year in (2003, 2004):
            source = self.root / f"source-{year}"
            _, cfg = self.fitted(source / "annual/years" / str(year), year)
            (source / "annual/experiment.json").write_text(json.dumps(dict(schema_version=1, years=[year],
                config=yaml.safe_load(cfg.dump()), runtime={}, orchestrator_sha256="fixture")))
            sources.append(source)
        plan = combination_plan(sources, (2003, 2004))
        group = self.root / "relocated"
        group.mkdir()
        inventory = []
        for index, source in enumerate(sources):
            original = f"/unavailable/source-{index}"
            plan["sources"][index]["experiment"] = original
            destination = group / "sources" / f"{index:02d}"
            shutil.copytree(source, destination)
            inventory.append(dict(experiment=original, directory=f"{index:02d}", files=source_file_hashes(destination)))
        (group / "collection.json").write_text(json.dumps(plan))
        (group / "launch.json").write_text(json.dumps(dict(schema_version=1, source_storage="sources/manifest.json")))
        (group / "sources/manifest.json").write_text(json.dumps(dict(schema_version=2,
            collection_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), sources=inventory)))
        before = source_file_hashes(group)
        output = make_report(group, self.root / "reports", [2004])
        self.assertTrue((output / "2004/index.html").is_file())
        self.assertEqual(source_file_hashes(group), before)
        checkpoint, = list((group / "sources/01").glob("annual/years/2004/*/years/2004/NN1.json"))
        checkpoint.write_bytes(checkpoint.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "frozen bundle inventory"):
            make_report(group, self.root / "reports", [2004])


if __name__ == "__main__":
    unittest.main()
