from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from core.raw_data_examinine.data_check_sic import inspect_sic


class SicInspectionTests(unittest.TestCase):
    def test_missing_patterns_across_chunks_and_output_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            pd.DataFrame({"permno": [1, 2, 3, 1, 2, 3],
                          "DATE": [20000131]*3 + [20000229]*3,
                          "sic2": [None, None, 1, None, 37, 99]}).to_csv(source, index=False)
            with redirect_stdout(io.StringIO()):
                info = inspect_sic(source, root / "out", chunk_size=2)
                inspect_sic(source, root / "other", chunk_size=5)
            self.assertEqual(info["n_permnos"], 3)
            self.assertEqual(info["n_missing_sic_rows"], 3)
            universe = pd.read_csv(root / "out/permno_sic_summary.csv", index_col="permno")
            self.assertEqual(universe.sic_status.tolist(), ["always_missing", "sometimes_missing", "never_missing"])
            self.assertEqual(universe.n_distinct_sic.tolist(), [0, 1, 2])
            self.assertEqual(universe.loc[2, "first_valid_date"], 20000229)
            keys = pd.read_csv(root / "out/missing_sic_rows.csv")
            self.assertEqual(list(keys.itertuples(index=False, name=None)),
                             [(1, 20000131), (2, 20000131), (1, 20000229)])
            codes = pd.read_csv(root / "out/sic_values.csv")
            self.assertEqual(codes.sic2.tolist(), [1, 37, 99])
            for name in ["permno_sic_summary.csv", "sic_values.csv", "missing_sic_rows.csv"]:
                self.assertEqual((root / "out" / name).read_bytes(), (root / "other" / name).read_bytes())
            with self.assertRaises(FileExistsError):
                inspect_sic(source, root / "out")

    def test_all_missing_and_empty_missing_list(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, value in [("missing", None), ("known", 10)]:
                source = root / f"{label}.csv"
                pd.DataFrame({"permno": [1], "DATE": [20000131], "sic2": [value]}).to_csv(source, index=False)
                with redirect_stdout(io.StringIO()):
                    inspect_sic(source, root / label)
                self.assertEqual(len(pd.read_csv(root / label / "missing_sic_rows.csv")), int(value is None))
                self.assertEqual(len(pd.read_csv(root / label / "sic_values.csv")), int(value is not None))


if __name__ == "__main__":
    unittest.main()
