"""Threshold rules, baseline equivalence, and output accounting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.data_examine.ols3_thresholds import (
    BASELINE, FEATURES, SETTINGS, VARIANTS, experiment, fit_variants,
)
from core.dense_linear_run import fit_candidates
from core.models import LinearRegressor


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.uniform(-1, 1, (240, 3))
        self.y = .03 + self.X @ np.array([.1, -.04, .07]) + rng.normal(0, .02, 240)
        self.y[0] += 1
        self.y[1] -= .8

    def test_thresholds_and_existing_baseline(self):
        models, records = fit_variants(self.X, self.y)
        records = {r["variant"]: r for r in records}
        ols = models["ols"]
        expected = [np.quantile(v, .999, method="linear") for v in
                    (abs(self.y - ols.predict(self.X)), abs(self.y), abs(self.y - self.y.mean()))]
        for name, delta in zip(VARIANTS, expected):
            self.assertEqual(records[name]["threshold"], delta)
            self.assertEqual(records[name]["training_tail_fraction"],
                             np.mean(abs(self.y - models[name].predict(self.X)) > delta))
        original = fit_candidates(self.X, self.y, LinearRegressor,
                                  [{**SETTINGS, "loss": "huber", "huber_quantile": .999}],
                                  label="[baseline test]")[0]
        np.testing.assert_allclose(models[BASELINE].predict(self.X), original.predict(self.X), atol=1e-14)
        np.testing.assert_allclose(ols.coef_, np.linalg.lstsq(
            np.column_stack([self.X, np.ones(len(self.X))]), self.y, rcond=None)[0][:3], atol=1e-14)

    def test_outputs_and_training_only_fit(self):
        # Evaluation returns differ sharply; thresholds must still use training only.
        _, reference = fit_variants(self.X, self.y)
        class Panel:
            metadata = {"year_counts": {str(y): 240 for y in range(1957, 1989)}}
            info = {"name": "synthetic", "panel_sha256": "test", "metadata_sha256": "test"}
        calls = []
        def load(panel, first, last, *, feature_columns):
            self.assertEqual(feature_columns, FEATURES)
            calls.append((first, last))
            y = self.y if first == 1957 else self.y + .5
            return self.X.copy(), y.copy(), np.arange(len(y)), np.full(len(y), f"{last}-01")
        with tempfile.TemporaryDirectory() as tmp, patch(
                "core.data_examine.ols3_thresholds.ParquetPanel", return_value=Panel()), patch(
                "core.data_examine.ols3_thresholds.load_window", side_effect=load):
            output = experiment("unused", [1987, 1988], tmp)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(calls, [(1957, 1974), (1975, 1986), (1987, 1987),
                                     (1957, 1975), (1976, 1987), (1988, 1988)])
            fits = json.loads((output / "annual/1987/fits.json").read_text())
            self.assertEqual([r["threshold"] for r in fits], [r["threshold"] for r in reference])
            table = pd.read_csv(output / "table1/comparison.csv").set_index("variant")
            for name in VARIANTS:
                observed, predicted = [], []
                for year in (1987, 1988):
                    with np.load(output / f"annual/{year}/test_predictions.npz", allow_pickle=False) as saved:
                        observed.extend(saved["observed"])
                        predicted.extend(saved[name])
                y, pred = np.array(observed), np.array(predicted)
                self.assertAlmostEqual(table.loc[name, "r2_percent"], 100 * (1 - np.sum((y-pred)**2) / (y@y)))
            self.assertTrue((output / "diagnostics/threshold.png").is_file())

    def test_failure_is_not_complete(self):
        class Panel:
            metadata = {"year_counts": {str(y): 1 for y in range(1957, 1988)}}
            info = {"name": "synthetic"}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "core.data_examine.ols3_thresholds.ParquetPanel", return_value=Panel()), patch(
                "core.data_examine.ols3_thresholds.load_window", side_effect=RuntimeError("failed load")):
            with self.assertRaisesRegex(RuntimeError, "failed load"):
                experiment("unused", [1987], tmp)
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_years"], [])


if __name__ == "__main__":
    unittest.main()
