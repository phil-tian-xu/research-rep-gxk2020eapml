from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.data_cleaning.partition_gkx import partition_gkx


class PartitionGkxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        months = pd.PeriodIndex(["2000-12", "2000-12", "2001-01", "2001-12", "2002-01", "2002-02"], freq="M")
        self.frame = pd.DataFrame({
            "permno": ["00001", "00002", "00001", "00002", "00001", "00002"],
            "feature_month": (months - 1).astype(str), "return_month": months.astype(str),
            "excess_return": [.01, -.02, .03, -.04, .05, -.06],
            "x": [0., 0., 0., np.nextafter(1., 2.), -0., 1.2345678901234567],
            "sic2": pd.array([None, None, None, 12, 99, None], dtype="Int64"),
            "sic_missing": [1, 1, 1, 0, 0, 1],
        })
        self.frame.to_csv(self.source / "panel.csv", index=False)
        self.metadata = {
            "schema_version": 1, "dataset_name": "gkx", "return_units": "decimal",
            "id_columns": ["permno"], "feature_month_column": "feature_month",
            "return_month_column": "return_month", "target_column": "excess_return",
            "feature_columns": ["x"], "audit_columns": ["sic2", "sic_missing"],
            "n_rows": len(self.frame),
        }
        self.write_metadata()

    def write_metadata(self):
        self.metadata["panel_sha256"] = hashlib.sha256((self.source / "panel.csv").read_bytes()).hexdigest()
        (self.source / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def convert(self, name="output", chunksize=3):
        with redirect_stdout(io.StringIO()):
            return partition_gkx(source_dir=self.source, output_dir=self.root / name, chunksize=chunksize)

    def read_parts(self, output):
        manifest = json.loads((output / "metadata.json").read_text())
        tables = []
        for part in manifest["parts"]:
            path = output / part["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), part["sha256"])
            with pq.ParquetFile(path) as parquet:
                table = parquet.read()
                self.assertTrue(all(parquet.metadata.row_group(i).num_rows <= manifest["chunksize"]
                                    for i in range(parquet.num_row_groups)))
            self.assertEqual(table.schema.field("x").type, pa.float64())
            self.assertEqual(table.schema.field("excess_return").type, pa.float64())
            self.assertEqual(table.schema.field("sic2").type, pa.int64())
            frame = table.to_pandas()
            self.assertTrue(frame.return_month.str.startswith(str(part["return_year"])).all())
            self.assertLessEqual(len(frame), manifest["chunksize"])
            tables.append(frame)
        return pd.concat(tables).sort_values("source_row_position").reset_index(drop=True), manifest

    def test_year_boundaries_float64_nulls_positions_and_source_preservation(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        output = self.convert()
        actual, metadata = self.read_parts(output)
        self.assertEqual(metadata["year_counts"], {"2000": 2, "2001": 2, "2002": 2})
        self.assertEqual(len(metadata["parts"]), 4)
        self.assertEqual(actual.permno.tolist(), self.frame.permno.tolist())
        np.testing.assert_array_equal(actual.source_row_position, np.arange(len(self.frame)))
        # Check float bit patterns, including the sign of zero, not just rounded values.
        for column in ["x", "excess_return"]:
            np.testing.assert_array_equal(actual[column].to_numpy().view("uint64"),
                                          self.frame[column].to_numpy().view("uint64"))
        self.assertEqual(actual.sic2.isna().tolist(), self.frame.sic2.isna().tolist())
        self.assertEqual(actual.sic_missing.tolist(), self.frame.sic_missing.tolist())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual((output / "source_metadata.json").read_bytes(), before["metadata.json"])
        other, _ = self.read_parts(self.convert("other", chunksize=5))
        pd.testing.assert_frame_equal(actual, other)

    def test_existing_output_is_refused(self):
        self.convert()
        with self.assertRaises(FileExistsError):
            self.convert()

    def test_source_hash_mismatch_is_rejected(self):
        with (self.source / "panel.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.convert()
        self.assertFalse((self.root / "output").exists())

    def test_wrong_row_count_has_no_completion_manifest(self):
        self.metadata["n_rows"] += 1
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "row count"):
            self.convert()
        self.assertFalse((self.root / "output" / "metadata.json").exists())

    def test_invalid_month_is_rejected(self):
        self.frame.loc[3, "return_month"] = "2001-13"
        self.frame.to_csv(self.source / "panel.csv", index=False)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "Invalid return month"):
            self.convert()
        self.assertFalse((self.root / "output" / "metadata.json").exists())

    def test_failed_write_has_no_completion_manifest(self):
        with patch("core.data_cleaning.partition_gkx.pq.write_table", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.convert()
        self.assertFalse((self.root / "output" / "metadata.json").exists())

    def test_invalid_chunk_size(self):
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                self.convert(chunksize=value)


if __name__ == "__main__":
    unittest.main()
