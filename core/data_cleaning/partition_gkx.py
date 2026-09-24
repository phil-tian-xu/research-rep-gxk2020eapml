"""Convert a prepared GKX panel.csv to yearly Parquet parts in bounded chunks.

Run from the repository root: python -B -m core.data_cleaning.partition_gkx
The CSV is read-only. This standalone conversion does not change main.py's loader.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.path_info import data_path


def partition_gkx(*, source_dir=data_path / "processed" / "gkx",
                  output_dir=data_path / "processed" / "gkx_parquet", chunksize=10_000):
    """Preserve all source columns; append zero-based source_row_position.

    Predictors and target use float64 even if a chunk contains only integers.
    Dates/identifiers retain their text; SIC audits use nullable integers.
    Every part is read back and compared before completion metadata is published.
    An interrupted/failed conversion leaves an incomplete directory, never a
    complete manifest. Existing output directories are refused, not resumed.
    """
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    panel_path, metadata_path = source_dir / "panel.csv", source_dir / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    parent = json.loads(metadata_bytes)
    if parent.get("schema_version") != 1 or parent.get("return_units") != "decimal":
        raise ValueError("Expected prepared panel schema_version=1 and decimal returns.")
    features, ids = parent["feature_columns"], parent["id_columns"]
    target, month = parent["target_column"], parent["return_month_column"]
    feature_month = parent["feature_month_column"]
    audits = parent.get("audit_columns", [])
    expected = ids + [feature_month, month, target] + features + audits
    with panel_path.open(encoding="utf-8", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if (not features or not ids or len(set(expected)) != len(expected) or
            len(set(columns)) != len(columns) or set(columns) != set(expected) or
            {"source_row_position", "return_year"}.intersection(columns)):
        raise ValueError("CSV columns must match unique metadata roles, with no reserved column names.")

    # Explicit types prevent chunk-dependent inference, including all-missing SIC.
    numeric = [target] + features
    dtypes = {name: "string" for name in columns}
    dtypes.update({name: "float64" for name in numeric})
    for name in audits:
        if name in {"sic2", "sic_missing"}:
            dtypes[name] = "Int64"
    arrow_types = {"string": pa.string(), "float64": pa.float64(), "Int64": pa.int64()}
    schema = pa.schema([(name, arrow_types[dtypes[name]]) for name in columns] +
                       [("source_row_position", pa.int64())])

    initial_stat = panel_path.stat()
    print(f"Source: {panel_path}\nOutput: {output}\nChunk size: {chunksize:,}", flush=True)
    print("Verifying source SHA-256 (streaming disk pass, not loading the panel)...", flush=True)
    with panel_path.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_digest != parent["panel_sha256"]:
        raise ValueError("Source panel SHA-256 differs from metadata.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_metadata.json").write_bytes(metadata_bytes)
    total, year_counts, parts = 0, {}, []
    # This is an iterator, not a list of chunks. Never concatenate the whole panel.
    with pd.read_csv(panel_path, dtype=dtypes, chunksize=chunksize,
                     float_precision="round_trip") as reader:
        for chunk_number, chunk in enumerate(reader, 1):
            if chunk.empty:
                continue
            if chunk[ids + [month, feature_month]].isna().any().any():
                raise ValueError(f"Missing identifiers/months in chunk {chunk_number}.")
            if not chunk[month].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
                raise ValueError(f"Invalid return month in chunk {chunk_number}; expected YYYY-MM.")
            if not np.isfinite(chunk[numeric].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"Nonfinite model values in chunk {chunk_number}.")
            chunk["source_row_position"] = np.arange(total, total + len(chunk), dtype=np.int64)
            years = chunk[month].str[:4].astype(int)
            # One incoming chunk may cross one or more year boundaries.
            for year, positions in chunk.groupby(years, sort=True).indices.items():
                part = chunk.iloc[positions]
                partition = output / f"return_year={year}"
                partition.mkdir(exist_ok=True)
                path = partition / f"part-{chunk_number:06d}.parquet"
                temporary = path.with_suffix(".parquet.tmp")
                table = pa.Table.from_pandas(part, schema=schema, preserve_index=False, safe=True)
                pq.write_table(table, temporary, compression="snappy", row_group_size=min(chunksize, 10_000))
                with pq.ParquetFile(temporary) as saved:
                    restored = saved.read(use_threads=False)
                if not table.equals(restored, check_metadata=False):
                    raise ValueError(f"Parquet read-back differs from input: {temporary}")
                temporary.replace(path)
                with path.open("rb") as stream:
                    part_digest = hashlib.file_digest(stream, "sha256").hexdigest()
                parts.append({"path": path.relative_to(output).as_posix(), "return_year": int(year),
                              "n_rows": len(part), "sha256": part_digest})
                year_counts[str(year)] = year_counts.get(str(year), 0) + len(part)
                del part, table, restored
            total += len(chunk)
            print(f"Chunk {chunk_number}: {total:,}/{parent['n_rows']:,} rows; "
                  f"{len(parts)} parts written and verified.", flush=True)
            del chunk

    final_stat = panel_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source CSV changed during conversion; output has no completion metadata.")
    if metadata_path.read_bytes() != metadata_bytes:
        raise ValueError("Source metadata changed during conversion.")
    if total == 0 or total != parent["n_rows"]:
        raise ValueError("Source row count mismatch; output has no completion metadata.")
    manifest = {
        "schema_version": 1, "format": "parquet", "status": "complete",
        "dataset_name": parent["dataset_name"] + "_parquet",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(panel_path), "source_panel_sha256": source_digest,
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "n_rows": total, "year_counts": year_counts, "parts": parts,
        "partition_column": "return_year", "row_position_column": "source_row_position",
        "columns": schema.names, "dtypes": {field.name: str(field.type) for field in schema},
        "feature_columns": features, "id_columns": ids, "target_column": target,
        "return_month_column": month, "feature_month_column": feature_month,
        "audit_columns": audits, "return_units": parent["return_units"],
        "chunksize": chunksize, "compression": "snappy",
        "pandas_version": pd.__version__, "pyarrow_version": pa.__version__,
    }
    temporary_manifest = output / "metadata.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(output / "metadata.json")
    print(f"Complete: {total:,} rows in {len(parts)} parts, {len(year_counts)} years.\n{output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=data_path / "processed" / "gkx")
    parser.add_argument("--output-dir", type=Path, default=data_path / "processed" / "gkx_parquet")
    parser.add_argument("--chunksize", type=int, default=10_000)
    args = parser.parse_args()
    partition_gkx(source_dir=args.source_dir, output_dir=args.output_dir, chunksize=args.chunksize)
