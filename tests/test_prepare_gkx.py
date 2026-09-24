from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from core.data_cleaning.prepare_gkx import prepare, rank_characteristics, read_macros_and_rf
from core.data_query.datasets import DATASET_PATHS, load_dataset


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "raw"
        self.raw.mkdir()
        self.out = Path(self.temp.name) / "processed"
        self.chars = [f"char{i}" for i in range(94)]
        rows = []
        for date in [20000131, 20000229]:
            for permno in [1, 2, 3]:
                row = {"permno": permno, "DATE": date, "sic2": 10 if permno == 1 else 20}
                row.update({c: float(permno) for c in self.chars})
                row["char93"] = np.nan
                rows.append(row)
        pd.DataFrame(rows).to_csv(self.raw / "datashare.csv", index=False)
        pd.DataFrame({"yyyymm": [199912, 200001, 200002], "Index": [10., 20., 40.],
                      "D12": [1., 1., 1.], "E12": [2., 2., 2.], "b/m": [.2, .3, .4],
                      "tbl": [.01, .02, .03], "AAA": [.03]*3, "BAA": [.05]*3,
                      "lty": [.04]*3, "ntis": [.1]*3, "svar": [.01]*3}).to_csv(
                          self.raw / "macro_predictor.csv", index=False)
        pd.DataFrame({"": [200001, 200002], "RF": [.2, .3]}).to_csv(
            self.raw / "risk_free_rate_fama_french.csv", index=False)
        pd.DataFrame({"permno": [1, 2, 3, 1, 2],
                      "date": ["2000-01-28"]*3 + ["2000-02-29"]*2,
                      "ret": [.1, "A", .3, .2, .4]}).to_csv(self.raw / "monthly_raw.csv", index=False)

    def build(self, chunksize=2):
        with redirect_stdout(io.StringIO()):
            return prepare(self.raw, self.out, "2000-01", "2000-02", chunksize)

    def test_round_trip_alignment_interactions_and_target_exclusion(self):
        self.build()
        with patch.dict(DATASET_PATHS, {"gkx": self.out}):
            X, y, months, info = load_dataset("gkx")
        assert_allclose(y, [.098, .298, .197, .397])
        assert_allclose(X.char0, [-1, 1, -1, 0])
        assert_allclose(X.char93, 0)
        assert_allclose(X.char0__macro_dp, [-np.log(.1), np.log(.1), -np.log(.05), 0])
        assert_allclose(X.sic2_10, [1, 0, 1, 0])
        self.assertEqual(X.shape, (4, 94*9 + 2))
        self.assertEqual(list(months.astype(str)), ["2000-01"]*2 + ["2000-02"]*2)
        panel = pd.read_csv(self.out / "panel.csv")
        self.assertEqual(panel.feature_month.tolist(), ["1999-12"]*2 + ["2000-01"]*2)
        report = pd.read_csv(self.out / "preparation_report.csv")
        self.assertEqual(report.unmatched_crsp_rows.tolist(), [0, 1])
        self.assertEqual(report.matched_missing_returns.tolist(), [1, 0])
        self.assertEqual(info["metadata"]["n_rows"], 4)

    def test_chunk_boundaries_do_not_change_panel(self):
        self.build(1)
        first = (self.out / "panel.csv").read_bytes()
        self.out = self.out.parent / "second"
        self.build(5)
        self.assertEqual(first, (self.out / "panel.csv").read_bytes())

    def test_rank_ties_imputation_constant_and_singleton(self):
        frame = pd.DataFrame({"a": [1., 2., 2., np.nan, 5.], "b": [np.nan]*5, "c": [7.]*5})
        ranked = rank_characteristics(frame)
        assert_allclose(ranked.a, [-1, 0, 0, 0, 1])
        assert_allclose(ranked[["b", "c"]], 0)
        assert_allclose(rank_characteristics(frame.iloc[:1]), 0)

    def test_duplicate_crsp_and_missing_macro_fail(self):
        path = self.raw / "monthly_raw.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate CRSP"):
            self.build()
        frame.to_csv(path, index=False)
        path = self.raw / "macro_predictor.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "macro predictors"):
            self.build()

    def test_unsorted_gkx_leaves_no_completion_metadata(self):
        path = self.raw / "datashare.csv"
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "ordered"):
            self.build()
        self.assertFalse((self.out / "metadata.json").exists())

    def test_no_overwrite(self):
        self.build()
        digest = json.loads((self.out / "metadata.json").read_text())["panel_sha256"]
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(json.loads((self.out / "metadata.json").read_text())["panel_sha256"], digest)

    def test_missing_sic_retained_without_new_predictor_or_temporal_fill(self):
        path = self.raw / "datashare.csv"
        frame = pd.read_csv(path)
        frame.loc[[0, 1], "sic2"] = np.nan
        frame.to_csv(path, index=False)
        self.build()
        panel = pd.read_csv(self.out / "panel.csv")
        self.assertEqual(panel.sic_missing.tolist(), [1, 0, 0, 0])
        self.assertTrue(pd.isna(panel.loc[0, "sic2"]))
        self.assertEqual(panel.loc[2, "sic2"], 10)
        with patch.dict(DATASET_PATHS, {"gkx": self.out}):
            X, y, _, _ = load_dataset("gkx")
        self.assertNotIn("sic_missing", X.columns)
        self.assertNotIn("sic2", X.columns)
        self.assertEqual(X.shape, (4, 848))
        assert_allclose(X.loc[0, ["sic2_10", "sic2_20"]], 0)
        assert_allclose(X.char0, [-1, 1, -1, 0])
        assert_allclose(y, [.098, .298, .197, .397])
        report = pd.read_csv(self.out / "preparation_report.csv")
        self.assertEqual(report.missing_sic_input_rows.tolist(), [2, 0])
        self.assertEqual(report.missing_sic_output_rows.tolist(), [1, 0])

    def test_all_missing_sic_and_invalid_observed_codes(self):
        path = self.raw / "datashare.csv"
        frame = pd.read_csv(path)
        frame["sic2"] = np.nan
        frame.to_csv(path, index=False)
        self.build()
        with patch.dict(DATASET_PATHS, {"gkx": self.out}):
            X, _, _, _ = load_dataset("gkx")
        self.assertEqual(X.shape, (4, 846))
        for number, value in enumerate([100, -1, 37.5]):
            frame.loc[0, "sic2"] = value
            frame.to_csv(path, index=False)
            self.out = self.out.parent / f"invalid_{number}"
            with self.assertRaisesRegex(ValueError, "Invalid nonmissing sic2"):
                self.build()

    def test_macro_thousands_separators_and_numeric_validation(self):
        path = self.raw / "macro_predictor.csv"
        frame = pd.read_csv(path)
        frame["Index"] = ["999.50", "1,049.34", "6,845.50"]
        frame.to_csv(path, index=False)
        macro, rf = read_macros_and_rf(self.raw)
        assert_allclose(macro.dp, -np.log([999.50, 1049.34, 6845.50]))
        assert_allclose(macro.ep, np.log(2) - np.log([999.50, 1049.34, 6845.50]))
        assert_allclose(rf, [.002, .003])
        frame["tbl"] = ["0.01", "not-a-number", "0.03"]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "invalid numeric values in 'tbl'"):
            read_macros_and_rf(self.raw)


if __name__ == "__main__":
    unittest.main()
