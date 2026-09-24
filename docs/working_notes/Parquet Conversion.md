# Prepared GKX CSV to yearly Parquet

Run from the repository root, in the project environment:

```powershell
python -B -m core.data_cleaning.partition_gkx
```

Defaults: source `data/processed/gkx/panel.csv` and its `metadata.json`;
destination `data/processed/gkx_parquet`; chunk size 10,000 rows. PyArrow is
declared in `pyproject.toml` and was installed in the local project environment.

To select another prepared dataset or reduce the chunk size:

```powershell
python -B -m core.data_cleaning.partition_gkx --source-dir data/processed/gkx_dev_p05_seed2020 --output-dir data/processed/gkx_dev_p05_parquet --chunksize 5000
```

## Conversion

The converter first verifies the source CSV's SHA-256 against its existing
metadata using a streaming disk pass. This can take time before the first
conversion progress line. It then uses `pd.read_csv(..., chunksize=...)` with
explicit types and `float_precision="round_trip"`. No list of data chunks or
full-panel concatenation is created. Parsing, Arrow conversion, and read-back
buffers still consume additional memory beyond the current chunk itself.

Each incoming chunk is divided by **realized return year** and written under
`return_year=YYYY/part-NNNNNN.parquet`. A chunk crossing year boundaries produces
one part for each represented year. A year may span many parts. Each part has
at most `chunksize` rows; row groups have at most `min(chunksize, 10000)` rows.
Compression is Snappy. One file is written and checked at a time.

All source columns are preserved in their original order. Features and the
target are explicitly float64, even when early chunks contain only zeros or
integers. Identifiers and month labels remain strings. The known audit columns
`sic2` and `sic_missing` use nullable int64; any other declared audit columns
are retained as strings. No ranking, imputation, return adjustment, filtering,
or model fitting occurs. Nonfinite features/targets and missing identifiers or
month labels are rejected rather than silently repaired or dropped.

`source_row_position` is an added zero-based identifier for the original CSV
row order. It is audit information, never a predictor. Parquet dataset readers
need not return files/rows in original order; use this field for alignment.
`return_year` is encoded in directory names, not duplicated in the file columns.

## Integrity and interrupted runs

Every part is read back and compared with its input Arrow table before it is
published. Per-file SHA-256 hashes and row counts are recorded. A copy of the
source metadata is kept as `source_metadata.json`. Source size/modification time
and metadata bytes are checked again before completion; converted row counts
must equal the source metadata count.

The final `metadata.json` is published only after all checks pass. It contains
`format="parquet"`, `status="complete"`, source hashes, column roles/types,
year counts, and the ordered part list. This is a new format manifest, not a
CSV-loader-compatible metadata file. A directory without this completion
manifest is incomplete, even if it contains some valid Parquet files.

Existing destination directories are refused, including incomplete ones.
Choose a new destination for a retry; conversion resume is not implemented.
The converter never overwrites or deletes the source CSV or source metadata.

## Scope and verification

The converter only changes storage. The subsequent batch workflow is now
implemented: v4 selects `gkx_parquet` with `DATA.FORMAT: parquet` and reads
10,000 rows at a time. See [Batch Parquet Workflow](Batch%20Parquet%20Workflow.md).
Other configs retain the existing CSV workflow. Loading/concatenating all
Parquet parts manually can still exhaust memory.

Seven focused tests pass, covering cross-year chunks, stable schemas, float64
bit patterns, missing audits, original row positions, chunk-size independence,
unchanged source files, and refusal/incomplete-output behavior on failures.
A real-data smoke check converted the first 10,000 rows in a temporary fixture
with 4,000-row chunks and verified all parts. Peak Windows process resident RAM
was about 0.34 GiB. This is a small-sample measurement, not a full-conversion
memory guarantee. Phil subsequently completed the full 3,740,946-row conversion;
the completed manifest is used by the batch workflow.
