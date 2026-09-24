from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.data_examine.pcr_memory import experiment, load_window
from core.data_query.parquet_dataset import PanelBatch
from core.models import PCRRegressor
from core.data_examine.pcr_full_memory import main as full_main
from utils.path_info import data_path


class PcrMemoryTests(unittest.TestCase):
    def test_full_entrypoint_selects_full_universe_and_separate_outputs(self):
        with redirect_stdout(StringIO()), patch("core.data_examine.pcr_full_memory.experiment") as run:
            full_main([])
        run.assert_called_once_with(data_path / "processed" / "gkx_parquet",
                                    test_year=1987, components=10,
                                    output_dir="results/pcr_full_memory")

    def setUp(self):
        rng = np.random.default_rng(72)
        self.X = rng.normal(size=(67, 4))
        self.y = .03 * self.X[:, 0] + rng.normal(scale=.01, size=67)
        self.calls = []

        def window(first, last):
            self.calls.append((first, last))
            start, end = {(2000, 2000): (0, 40), (2001, 2001): (40, 55), (2002, 2002): (55, 67)}[first, last]

            def batches():
                for left in range(start, end, 7):
                    right = min(left + 7, end)
                    yield PanelBatch(self.X[left:right], self.y[left:right], np.arange(left, right),
                                     np.full(right - left, f"{first}-01"))
            return SimpleNamespace(n_rows=end - start, batches=batches)

        self.panel = SimpleNamespace(window=window, features=list("abcd"),
            info={"name": "fixture", "panel_sha256": "panel", "metadata_sha256": "metadata"})

    def test_loader_returns_full_float64_arrays_in_original_order(self):
        X, y, positions, months = load_window(self.panel, 2000, 2000)
        np.testing.assert_array_equal(X, self.X[:40])
        np.testing.assert_array_equal(y, self.y[:40])
        np.testing.assert_array_equal(positions, np.arange(40))
        self.assertEqual(X.dtype, np.float64)
        self.assertTrue((months == "2000-01").all())

    def test_experiment_fits_original_model_once_on_complete_training_window(self):
        original_fit = PCRRegressor.fit
        fit_sizes = []

        def fit(model, X, y):
            fit_sizes.append(X.shape)
            np.testing.assert_array_equal(X, self.X[:40])
            np.testing.assert_array_equal(y, self.y[:40])
            return original_fit(model, X, y)

        counters = {"resident_bytes": 20, "peak_resident_bytes": 50, "private_committed_bytes": 100}
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch("core.data_examine.pcr_memory.ParquetPanel", return_value=self.panel), \
                patch("core.data_examine.pcr_memory.process_memory", return_value=counters), \
                patch.object(PCRRegressor, "fit", new=fit), \
                patch("core.models.batched.reduce_training", side_effect=AssertionError("No batch fit")):
            output = experiment(directory, test_year=2002, first_train_year=2000, validation_years=1,
                                components=2, output_dir=directory)
            report = json.loads((output / "report.json").read_text())
            actual = pd.read_csv(output / "test_predictions.csv")
        self.assertEqual(fit_sizes, [(40, 4)])
        self.assertEqual(self.calls, [(2000, 2000), (2001, 2001), (2002, 2002)])
        reference = PCRRegressor(2).fit(self.X[:40], self.y[:40])
        np.testing.assert_allclose(actual.prediction, reference.predict(self.X[55:]), atol=1e-15)
        self.assertEqual(report["windows"]["training"]["X_y_bytes"], (40 * 4 + 40) * 8)
        self.assertEqual(report["peak_resident_gib"], 50 / 1024**3)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["tuning"])

    def test_failed_fit_still_saves_memory_report(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch("core.data_examine.pcr_memory.ParquetPanel", return_value=self.panel), \
                patch.object(PCRRegressor, "fit", side_effect=MemoryError("test failure")):
            with self.assertRaises(MemoryError):
                experiment(directory, test_year=2002, first_train_year=2000, validation_years=1,
                           components=2, output_dir=directory)
            report = json.loads(next(Path(directory).glob("*/report.json")).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("peak_resident_gib", report)


if __name__ == "__main__":
    unittest.main()
