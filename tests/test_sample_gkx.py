from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from core.data_cleaning.sample_gkx import sample_gkx
from core.data_query.datasets import DATASET_PATHS, load_dataset


class SampleGkxTests(unittest.TestCase):
    def test_nested_samples_keep_exact_histories_and_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent"
            source.mkdir()
            rows = []
            for month in pd.period_range("2000-01", "2000-03", freq="M"):
                for permno in range(1, 201):
                    # Some stocks have gaps; others enter late or leave early.
                    if month.month == 2 and permno % 3 == 0 or month.month == 1 and permno % 5 == 0:
                        continue
                    rows.append([permno, str(month - 1), str(month), .01, .12345678901234567])
            panel = pd.DataFrame(rows, columns=["permno", "feature_month", "return_month", "excess_return", "x"])
            path = source / "panel.csv"
            panel.to_csv(path, index=False, float_format="%.17g")
            metadata = {"schema_version": 1, "dataset_name": "gkx", "id_columns": ["permno"],
                        "feature_month_column": "feature_month", "return_month_column": "return_month",
                        "target_column": "excess_return", "feature_columns": ["x"], "return_units": "decimal",
                        "n_rows": len(panel), "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (source / "metadata.json").write_text(json.dumps(metadata))
            with redirect_stdout(io.StringIO()):
                small = sample_gkx(.05, source_dir=source, output_root=root)
                large = sample_gkx(.10, source_dir=source, output_root=root)
                repeated = sample_gkx(.05, source_dir=source, output_root=root / "repeat")
            sets = []
            for folder in [small, large]:
                universe = pd.read_csv(folder / "universe_selection.csv")
                selected = set(universe.loc[universe.selected == 1, "permno"])
                sets.append(selected)
                original_lines = path.read_bytes().splitlines(keepends=True)
                expected = original_lines[0] + b"".join(line for line in original_lines[1:] if int(line.split(b",",1)[0]) in selected)
                self.assertEqual((folder / "panel.csv").read_bytes(), expected)
                name = folder.name
                with patch.dict(DATASET_PATHS, {name: folder}):
                    X, y, _, _ = load_dataset(name)
                self.assertEqual(len(X), int(universe.loc[universe.selected == 1, "n_rows"].sum()))
                self.assertEqual(len(y), len(X))
            self.assertTrue(sets[0] < sets[1])
            self.assertEqual((small / "panel.csv").read_bytes(), (repeated / "panel.csv").read_bytes())
            with self.assertRaises(FileExistsError):
                sample_gkx(.05, source_dir=source, output_root=root)
            with path.open("ab") as dest:
                dest.write(original_lines[-1])
            with self.assertRaisesRegex(ValueError, "hash/row count"):
                sample_gkx(.5, source_dir=source, output_root=root)
            self.assertFalse((root / "gkx_dev_p50_seed2020/metadata.json").exists())

    def test_invalid_probabilities(self):
        for value in [0, -1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sample_gkx(value)


if __name__ == "__main__":
    unittest.main()
