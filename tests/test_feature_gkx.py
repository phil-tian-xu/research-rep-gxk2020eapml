from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.testing import assert_array_equal, assert_allclose

import test_prepare_gkx as fixtures
from core.data_cleaning.feature_gkx import prepare_features
from core.data_cleaning.partition_gkx import partition_gkx
from core.data_query.parquet_dataset import ParquetPanel, load_window
from core.data_query.datasets import validate_feature_dataset


class FeaturePanelTests(unittest.TestCase):
    def setUp(self):
        fixtures.PreparationTests.setUp(self)
        fixtures.PreparationTests.build(self)
        with redirect_stdout(StringIO()):
            self.source = partition_gkx(source_dir=self.out, output_dir=self.out.parent / "parquet", chunksize=3)

    def build(self, name="compact", *, expanded=False, batch_size=2):
        output = self.out.parent / name
        with redirect_stdout(StringIO()):
            prepare_features(source_dir=self.source, raw_dir=self.raw, output_dir=output,
                             dataset_name=name, expanded=expanded, batch_size=batch_size)
        return ParquetPanel(output)

    def test_compact_and_expanded_preserve_source_and_match_known_lagged_products(self):
        original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob("*") if p.is_file()}
        compact, expanded = self.build(), self.build("expanded", expanded=True)
        C, y, positions, months = load_window(compact, 2000, 2000)
        E, ey, ep, em = load_window(expanded, 2000, 2000)
        source = ParquetPanel(self.source)
        S, sy, sp, sm = load_window(source, 2000, 2000)
        self.assertEqual(len(compact.features), 94 + 8 + 2)
        self.assertEqual(len(expanded.features), 94 + 752 + 2)
        self.assertEqual(expanded.features, source.features)
        self.assertFalse(any(c.startswith("macro_") for c in expanded.features))
        assert_array_equal(E, S)
        assert_array_equal(E[:, :94], C[:, :94])
        for left, right in ((y, sy), (ey, sy), (positions, sp), (ep, sp), (months, sm), (em, sm)):
            assert_array_equal(left, right)
        assert_array_equal(C[:, :94], S[:, :94])
        # January returns use December 1999 macros; February uses January 2000.
        assert_allclose(C[:, compact.features.index("macro_dp")], [-np.log(10)]*2 + [-np.log(20)]*2)
        assert_allclose(C[:, compact.features.index("macro_bm")], [.2, .2, .3, .3])
        products = [c for c in expanded.features if "__macro_" in c]
        assert_array_equal(E[:, [expanded.features.index(c) for c in products]],
                           S[:, [source.features.index(c) for c in products]])
        for part in compact.metadata["parts"]:
            with pq.ParquetFile(compact.folder / part["path"]) as f:
                frame = f.read().to_pandas()
            self.assertIn("sic_missing", frame)
            self.assertNotIn("sic_missing", compact.features)
        self.assertEqual(original, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original})

    def test_batch_boundaries_leave_model_values_unchanged(self):
        first, second = self.build("first", expanded=True, batch_size=1), self.build("second", expanded=True, batch_size=5)
        for a, b in zip(load_window(first, 2000, 2000), load_window(second, 2000, 2000)):
            assert_array_equal(a, b)

    def test_expanded_preserves_paper_inputs_without_raw_macro_files(self):
        (self.raw / "macro_predictor.csv").unlink()
        (self.raw / "risk_free_rate_fama_french.csv").unlink()
        expanded = self.build("paper", expanded=True)
        source = ParquetPanel(self.source)
        self.assertEqual(expanded.features, source.features)
        self.assertNotIn("macro_source", expanded.metadata)
        for actual, expected in zip(load_window(expanded, 2000, 2000), load_window(source, 2000, 2000)):
            assert_array_equal(actual, expected)

    def test_missing_macro_has_no_completion_manifest_and_existing_output_is_preserved(self):
        self.build()
        before = (self.out.parent / "compact" / "metadata.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, (self.out.parent / "compact" / "metadata.json").read_bytes())
        path = self.raw / "macro_predictor.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "macro"):
            self.build("missing")
        self.assertFalse((self.out.parent / "missing" / "metadata.json").exists())

    def test_corrupt_source_is_rejected(self):
        metadata = json.loads((self.source / "metadata.json").read_text())
        path = self.source / metadata["parts"][0]["path"]
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build("corrupt")
        self.assertFalse((self.out.parent / "corrupt" / "metadata.json").exists())

    def test_stale_928_expanded_and_wrong_compact_manifests_are_rejected(self):
        compact = self.build()
        expanded = self.build("paper", expanded=True)
        compact_metadata = {**compact.metadata, "dataset_name": "gkx_compact_p05"}
        expanded_metadata = {**expanded.metadata, "dataset_name": "gkx_expanded_p05"}
        validate_feature_dataset(compact_metadata)
        validate_feature_dataset(expanded_metadata)
        # The superseded expanded definition included eight standalone macros.
        stale = {**expanded_metadata,
                 "feature_columns": expanded.features + compact.metadata["feature_specification"]["macro_columns"]}
        with self.assertRaisesRegex(ValueError, "feature definition"):
            validate_feature_dataset(stale)
        with self.assertRaisesRegex(ValueError, "feature definition"):
            validate_feature_dataset({**compact_metadata, "feature_columns": compact.features[::-1]})
        with self.assertRaisesRegex(ValueError, "feature specification"):
            validate_feature_dataset({**compact_metadata, "feature_specification": None})
        # Names from the original workflow have no new metadata requirement.
        validate_feature_dataset({"dataset_name": "gkx_parquet"})

    def test_loader_rejects_bad_compact_before_reading_parts(self):
        compact = self.build("gkx_compact_p05")
        path = compact.folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["feature_columns"] = metadata["feature_columns"][:-1]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "feature definition"):
            ParquetPanel(compact.folder)


if __name__ == "__main__":
    unittest.main()
