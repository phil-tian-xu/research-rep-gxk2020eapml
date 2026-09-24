# GKX development samples

Run independently from model training:

```powershell
python -B -m core.data_cleaning.sample_gkx
```

The single function `sample_gkx(probability, seed=2020, ...)` is used for both
default probabilities, 0.05 and 0.10. Outputs are
`data/processed/gkx_dev_p05_seed2020/` and
`data/processed/gkx_dev_p10_seed2020/`. Both names are registered in
`core/data_query/datasets.py`; select either using `DATA.DATASET` in a run config.
Other probabilities/seeds are supported by CLI options but require their own
loader path registration. Existing output directories are not overwritten.

## Selection

For each PERMNO, compute SHA-256 of the ASCII string `seed:PERMNO` and select
when the resulting integer is below `floor(probability * 2**256)`. This gives a
stable pseudorandom selection independent of row order, history length,
survival, returns, and other stock characteristics. The same seed makes 5%
nested in 10%. Proportions are approximate, not fixed quotas.

Retain all parent observations for each selected PERMNO, including its original
entry/exit and gaps. Preserve the complete parent date range and all 920 feature
columns. Do not rank, impute, standardize, or form interactions again. The
original CSV header and selected lines are copied byte-for-byte in parent order.
The reader targets the single-line numeric CSV format produced by prepare_gkx.

The sampler streams the parent panel once per probability, checks the parent
SHA-256 and row count, and writes completion metadata only after validation.
Failed runs can leave partial files without metadata. Memory scales with the
number of PERMNOs; it does not load the wide panel into memory.

## Reports and limitations

Each output includes `panel.csv`, loader-compatible `metadata.json`, and:

- `universe_selection.csv`: every parent PERMNO, selection flag, history length,
  first/last observed month, and full-month coverage flag.
- `monthly_sampling.csv`: parent/sample row counts and realized fraction per month.
- `sampling_summary.json`: nominal and realized fractions, seed, method, and
  full-coverage versus partial-coverage proportions.

Full coverage refers to observed rows in the processed parent, not proof of
continuous listing. It assumes the parent's unique security-month keys. All
selected histories are preserved, but size, industry, and history proportions
are preserved only in expectation. Do not search over seeds to improve returns
or model performance. These are development datasets, not replacements for
full-panel replication or full-universe portfolio statistics.

The model loader still uses dense float64 arrays; memory will exceed the CSV
reader's footprint during loading/training. The parent remains unchanged.

## Generated samples, 2026-09-22

Parent: 29,885 PERMNOs, 3,740,946 rows, 718 months (1957-03 through 2016-12).

| Dataset | PERMNOs | Rows | Parent row fraction | CSV bytes |
| --- | ---: | ---: | ---: | ---: |
| gkx_dev_p05_seed2020 | 1,496 | 182,367 | 4.87% | 2,539,858,337 |
| gkx_dev_p10_seed2020 | 3,041 | 374,996 | 10.02% | 5,212,290,782 |

Monthly retained fractions range from 4.19% to 5.72% and from 9.05% to 10.98%,
respectively. Full-month coverage applies to 120 parent PERMNOs, 9 selected in
5%, and 17 selected in 10%. This illustrates that random sampling preserves
history proportions approximately, not exactly. No selection was conditioned
on future coverage. The smaller PERMNO set is a strict subset of the larger;
manifest history counts reconcile with sample rows and monthly reports.

Verification: all 91 tests passed, including deterministic/nested selection,
unchanged row bytes and gaps, loader compatibility, and rejection of altered
parents. Both generated real datasets passed the existing full dataset loader
(including SHA-256, date alignment, unique keys, and finite model-input checks).
