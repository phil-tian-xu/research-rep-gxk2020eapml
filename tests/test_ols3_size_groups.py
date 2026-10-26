"""Size-group membership, pooled scoring, and saved-fit/window-check integration."""

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import yaml

from core.data_cleaning.partition_gkx import partition_gkx
from core.data_examine.ols3_size_groups import FEATURES, make_report, read_sizes, size_group_summary
from core.data_query.parquet_dataset import ParquetPanel, load_window
from core.dense_linear_run import diagnostics, fit_candidates
from core.evaluation.prediction_metrics import zero_benchmark_r2
from core.models import LinearRegressor
from core.run import configured_specifications
from core.run_state import dataset_identity, save_checkpoint
from utils.general_utils import load_config


class SizeGroupTests(unittest.TestCase):
    def test_monthly_membership_and_invalid_sizes_retained_in_all(self):
        frame = pd.DataFrame(dict(stock_id=[10, 20, 30, 40, 50, 60] * 2,
            return_month=pd.PeriodIndex(["2000-01"] * 6 + ["2000-02"] * 6, freq="M"),
            row_position=np.arange(12), market_equity=[10., 20., 30., 40., 0., np.nan,
                                                      40., 30., 20., 10., -1., np.nan],
            observed=np.arange(1., 13.), prediction=[0., 2., 0., 4., 0., 0., 7., 0., 9., 0., 0., 0.]))
        summary, audit, invalid = size_group_summary(frame, 1)
        summary = summary.set_index("sample")
        # January's largest is 40; February's largest is 10. Both forecasts are exact.
        self.assertEqual(summary.loc["top", "r2_zero"], 1.)
        self.assertEqual(summary.loc["bottom", "benchmark_sse"], 1.**2 + 10.**2)
        self.assertEqual(summary.loc["bottom", "r2_zero"], 0.)
        self.assertEqual(summary.loc["all", "n_obs"], 12)
        self.assertAlmostEqual(summary.loc["all", "r2_zero"],
            1 - np.sum((frame.observed-frame.prediction)**2) / np.sum(frame.observed**2))
        self.assertEqual(audit.n_size_eligible.tolist(), [4, 4])
        self.assertEqual(audit.n_missing_size.tolist(), [1, 1])
        self.assertEqual(audit.n_nonpositive_size.tolist(), [1, 1])
        self.assertEqual(len(invalid), 4)
        shuffled, _, _ = size_group_summary(frame.sample(frac=1, random_state=1), 1)
        assert_allclose(shuffled.set_index("sample").r2_zero, summary.r2_zero)

    def test_numeric_ties_and_coverage_failures(self):
        frame = pd.DataFrame(dict(stock_id=[100, 20, 30, 40], return_month=["2000-01"]*4,
            row_position=range(4), market_equity=[50., 50., 10., 10.],
            observed=[1., 2., 3., 4.], prediction=[0., 2., 3., 0.]))
        summary, _, _ = size_group_summary(frame, 1)
        self.assertEqual(summary.set_index("sample").loc["top", "r2_zero"], 1.)
        self.assertEqual(summary.set_index("sample").loc["bottom", "r2_zero"], 1.)
        with self.assertRaisesRegex(ValueError, "twice"):
            size_group_summary(frame, 3)
        frame["market_equity"] = 1.
        with self.assertRaisesRegex(ValueError, "overlapping"):
            size_group_summary(frame, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            size_group_summary(pd.concat([frame, frame.iloc[:1]]), 1)

    def test_raw_size_ranking_uses_full_cross_section_and_handles_chunk_boundaries(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "raw.csv"
            raw = pd.DataFrame(dict(permno=[1, 2, 3, 4]*12,
                DATE=np.repeat([int(f"2000{m:02d}28") for m in range(1, 13)], 4),
                mvel1=[10., np.nan, 30., 40.]*12))
            raw.to_csv(path, index=False)
            sizes = read_sizes(path, [2000], chunksize=3)
            assert_allclose(sizes.ranked_size.iloc[:4], [-1., 0., 0., 1.])
            self.assertTrue(pd.isna(sizes.market_equity.iloc[1]))
            pd.concat([raw, raw.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_sizes(path, [2000])


class SizeReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "source"
        source.mkdir()
        rng = np.random.default_rng(70)
        months = pd.period_range("2000-01", "2003-12", freq="M").repeat(6)
        ids = np.tile(np.arange(100, 106), 48)
        values = np.tile([10., 20., 30., 40., 50., 60.], 48)
        for i in range(1, 48, 2):
            values[i*6:(i+1)*6] = values[i*6:(i+1)*6][::-1]
        values[np.flatnonzero(months == pd.Period("2003-01"))[:2]] = [0., np.nan]
        raw = pd.DataFrame(dict(permno=ids, DATE=months.to_timestamp(how="end").strftime("%Y%m%d"), mvel1=values))
        self.raw_path = self.root / "raw.csv"
        raw.to_csv(self.raw_path, index=False)
        filled = pd.Series(values).fillna(pd.Series(values).groupby(months).transform("median"))
        ranks = filled.groupby(months).rank()
        X = np.column_stack([2*(ranks-1)/5-1, rng.uniform(-1, 1, (len(months), 2))])
        y = .02 + X @ np.array([.03, -.01, .02]) + rng.normal(scale=.01, size=len(X))
        y[::19] += .2
        frame = pd.DataFrame(X, columns=FEATURES)
        frame["permno"] = ids
        frame["feature_month"] = (months-1).astype(str)
        frame["return_month"] = months.astype(str)
        frame["excess_return"] = y
        # A raw-size row with no target must still participate in predictor ranking.
        keep = ~((months == pd.Period("2003-03")) & (ids == 105))
        frame = frame.loc[keep].reset_index(drop=True)
        frame.to_csv(source / "panel.csv", index=False)
        metadata = dict(schema_version=1, dataset_name="fixture", return_units="decimal",
            id_columns=["permno"], feature_month_column="feature_month", return_month_column="return_month",
            target_column="excess_return", feature_columns=FEATURES, audit_columns=[], n_rows=len(frame),
            panel_sha256=hashlib.sha256((source / "panel.csv").read_bytes()).hexdigest())
        (source / "metadata.json").write_text(json.dumps(metadata))
        self.data = self.root / "parquet"
        with redirect_stdout(StringIO()):
            partition_gkx(source_dir=source, output_dir=self.data, chunksize=17)
        panel = ParquetPanel(self.data)
        self.experiment = self.root / "experiment"
        self.checkpoints = []
        cfg = load_config("gkx_ols3_huber_v1")
        cfg.DATA.DATASET = "fixture_parquet"
        cfg.RUN.SPLIT.FIRST_TRAIN_YEAR = 2000
        cfg.RUN.SPLIT.VALIDATION_YEARS = 1
        cfg.RUN.SPLIT.FIRST_TEST_YEAR = cfg.RUN.SPLIT.LAST_TEST_YEAR = 2002
        self.panel_frame = frame
        self.cfg = cfg
        for year in [2002, 2003]:
            cfg.RUN.SPLIT.FIRST_TEST_YEAR = cfg.RUN.SPLIT.LAST_TEST_YEAR = year
            run = self.experiment / f"annual/years/{year}/fixture"
            run.mkdir(parents=True)
            (run / "resolved_config.yaml").write_text(cfg.dump())
            (run / "manifest.json").write_text(json.dumps(dict(schema_version=1,
                config=yaml.safe_load(cfg.dump()), dataset=dataset_identity(panel.info))))
            years = pd.PeriodIndex(frame.return_month, freq="M").year
            train, validation, test = years <= year-2, years == year-1, years == year
            _, factory, candidates = configured_specifications(cfg.RUN.MODELS)[0]
            with redirect_stdout(StringIO()):
                model = fit_candidates(frame.loc[train, FEATURES].to_numpy(), frame.excess_return[train].to_numpy(),
                                       factory, candidates, label="fixture")[0]
            observed = frame.excess_return[test].to_numpy()
            predicted = model.predict(frame.loc[test, FEATURES].to_numpy())
            validation_loss = float(np.mean((model.predict(frame.loc[validation, FEATURES].to_numpy())
                                              - frame.excess_return[validation].to_numpy())**2))
            predictions = pd.DataFrame(dict(specification="OLS3_HUBER", test_year=year,
                row_position=np.flatnonzero(test), return_month=frame.return_month[test].to_numpy(),
                observed=observed, prediction=predicted))
            tuning = dict(specification="OLS3_HUBER", test_year=year, feature_names=FEATURES,
                coefficients=model.coef_.tolist(), intercept=model.intercept_, best_params=candidates[0],
                validation_results=[dict(params=candidates[0], validation_loss=validation_loss, **diagnostics(model))],
                **diagnostics(model))
            metrics = dict(specification="OLS3_HUBER", test_year=year, n_test=int(test.sum()),
                n_train=int(train.sum()), n_validation=int(validation.sum()), test_loss_name="MSE",
                test_loss=float(np.mean((observed-predicted)**2)), r2_zero=zero_benchmark_r2(observed, predicted),
                best_validation_loss=validation_loss)
            save_checkpoint(run, metrics, predictions, tuning)
            self.checkpoints.append(run / f"years/{year}/OLS3_HUBER.json")

    def report(self, years=(2002, 2003), **kwargs):
        with redirect_stdout(StringIO()):
            return make_report(self.experiment, self.data, self.raw_path, self.root / "outputs", years,
                               size_count=2, **kwargs)

    def test_default_never_fits_and_preserves_all_observations_and_pooled_sums(self):
        originals = [path.read_bytes() for path in self.checkpoints]
        with patch("core.data_examine.ols3_size_groups.fit_candidates", side_effect=AssertionError("No fitting")):
            output = self.report()
        table = pd.read_csv(output / "comparison.csv").set_index("sample")
        annual = pd.read_csv(output / "annual_metrics.csv")
        for sample in ["all", "top", "bottom"]:
            rows = annual.loc[annual["sample"].eq(sample)]
            self.assertAlmostEqual(table.loc[sample, "r2_zero"], 1-rows.sse.sum()/rows.benchmark_sse.sum())
        self.assertEqual(table.loc["all", "n_obs"], 143)
        self.assertEqual(table.loc["top", "n_obs"], 48)
        self.assertTrue(table.paper_r2_percent.isna().all())
        self.assertEqual(len(pd.read_csv(output / "invalid_sizes.csv")), 2)
        self.assertEqual(json.loads((output / "methodology.json").read_text())["status"], "complete")
        self.assertEqual(originals, [path.read_bytes() for path in self.checkpoints])

    def test_optional_refit_uses_only_pretest_years_and_identical_groups(self):
        with patch("core.data_examine.ols3_size_groups.load_window", wraps=load_window) as loader:
            output = self.report(refit_train_validation=True)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[1:3], (2000, 2002))
        fits = json.loads((output / "fits.json").read_text())
        self.assertEqual(fits[1]["n_train"], 144)
        self.assertEqual(fits[1]["last_train_year"], 2001)
        self.assertEqual(fits[-1]["n_train"], 216)
        self.assertEqual(fits[-1]["last_train_year"], 2002)
        frame = self.panel_frame
        params = json.loads(json.loads(self.checkpoints[-1].read_text())["payload"])["tuning"]["best_params"]
        for fitted in [fits[1], fits[-1]]:
            mask = frame.return_month.str[:4].astype(int) < fitted["test_year"]
            reference = LinearRegressor(**params).fit(frame.loc[mask, FEATURES].to_numpy(), frame.excess_return[mask].to_numpy())
            assert_allclose(fitted["coefficients"], reference.coef_, atol=1e-12)
            self.assertAlmostEqual(fitted["intercept"], reference.intercept_, places=12)
        table = pd.read_csv(output / "comparison.csv")
        first = table.loc[table.variant.eq("saved_training_only")].set_index("sample")
        second = table.loc[table.variant.eq("refit_train_validation")].set_index("sample")
        assert_allclose(first[["n_obs", "benchmark_sse"]], second[["n_obs", "benchmark_sse"]])

    def test_rejects_corrupt_checkpoint_and_wrong_characteristic_source(self):
        original = self.checkpoints[0].read_text()
        envelope = json.loads(original)
        envelope["sha256"] = "bad"
        self.checkpoints[0].write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.report()
        self.checkpoints[0].write_text(original)
        raw = pd.read_csv(self.raw_path)
        rows = raw.index[raw.DATE.eq(20020131)][:2]
        raw.loc[rows, "mvel1"] = raw.loc[rows, "mvel1"].iloc[::-1].to_numpy()
        raw.to_csv(self.raw_path, index=False)
        with self.assertRaisesRegex(ValueError, "Raw size ranks differ"):
            self.report()


if __name__ == "__main__":
    unittest.main()
