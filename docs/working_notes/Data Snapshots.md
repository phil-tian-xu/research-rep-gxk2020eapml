# Source-data snapshots

The browsing extracts in `data/processed/data_snapshot_20261021/` contain:

- `characteristics_94_100stocks_2015_2016.csv`: January 2015 through December
  2016, 100 PERMNOs and 2,231 available stock-month observations. Contains the
  original 94 characteristics plus `permno`, `DATE`, and `sic2`.
- `goyal_welch_2024_2025.csv`: 24 monthly observations and all original columns
  from the latest two complete calendar years in the local macro source.

Reproduce the CSV extracts with
`python -B -m core.data_query.create_data_snapshot --output-dir NEW_DIRECTORY`.
The script refuses existing destinations and selects the latest complete macro
years from the source file. With a newer macro vintage, those years may change.

Sampling uses `random.Random(2020).sample` on sorted January-2015 PERMNOs.
Membership is fixed for the 24-month window, with no future survival or
nonmissing-characteristic requirement. Available months per sampled security
range from 2 to 24. No absent rows or values are filled.

These CSVs filter `data/raw/datashare.csv` and `data/raw/macro_predictor.csv`
directly, preserving field text, column order, dates and missing values.
Only row selection, row sorting and LF line endings are applied. Source and
output SHA-256 hashes, selected PERMNOs and coverage counts are in `metadata.json`.

GKX `DATE` retains its original return-month label. Macro `yyyymm` retains its
observation-month label. The two extracts cover different periods and are for
browsing; they are not joined or prepared model inputs. The optional two-sheet
Excel copy converts numeric fields to numbers and missing fields to empty cells;
the CSVs retain the exact original field text.
