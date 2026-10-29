"""Prepare compact or expanded predictors from an existing GKX Parquet panel.

python -B -m core.data_cleaning.feature_gkx --dataset gkx_compact_full
Stock characteristics, targets, securities and row positions are preserved.
Macro observations join on feature_month; no additional stock-characteristic lag,
ranking, imputation or scaling is applied. Existing destinations are refused.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.data_cleaning.prepare_gkx import MACROS, read_macros_and_rf
from core.data_query.datasets import DATASET_PATHS
from core.data_query.parquet_dataset import ParquetPanel
from core.run_state import atomic_write
from utils.path_info import data_path


FEATURE_DATASETS = ("gkx_compact_full", "gkx_compact_p05", "gkx_expanded_full", "gkx_expanded_p05")


def prepare_features(*, source_dir, raw_dir, output_dir, dataset_name, expanded=False, batch_size=10000):
    """Stream unchanged source observations into a new, verified Parquet dataset.

    Compact inputs are 94 characteristics + 8 macros + observed SIC dummies.
    Expanded inputs copy the paper's original characteristics, 752 macro products
    and SIC dummies, in their original order. No standalone macros are added.
    """
    source, raw, output = Path(source_dir).resolve(), Path(raw_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output directories must be separate.")
    panel = ParquetPanel(source, batch_size=batch_size)
    metadata_bytes = (source / "metadata.json").read_bytes()
    parent = panel.metadata
    chars = [c for c in panel.features if "__macro_" not in c and not c.startswith("sic2_")]
    industries = [c for c in panel.features if c.startswith("sic2_")]
    interactions = [f"{c}__macro_{m}" for m in MACROS for c in chars]
    if len(chars) != 94 or set(panel.features) != set(chars + interactions + industries):
        raise ValueError("Source must declare the original 94 characteristics, macro products and SIC dummies.")
    macro_names = [f"macro_{m}" for m in MACROS]
    features = list(panel.features) if expanded else chars + macro_names + industries
    carry = [c for c in parent["columns"] if c not in panel.features]
    if len(set(carry + features)) != len(carry + features):
        raise ValueError("Output column names collide.")
    macro_path = raw / "macro_predictor.csv"
    macro_bytes = None if expanded else macro_path.read_bytes()
    macros = None if expanded else read_macros_and_rf(raw)[0]
    month, feature_month, target, position = (parent[k] for k in
        ("return_month_column", "feature_month_column", "target_column", "row_position_column"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_metadata.json").write_bytes(metadata_bytes)
    parts, year_counts, previous = [], {}, {}
    total = 0
    schema = None
    for part_number, part in enumerate(parent["parts"]):
        path = panel.verify_part(part)
        count = 0
        year = part["return_year"]
        with pq.ParquetFile(path) as parquet:
            selected = carry + (features if expanded else chars + industries)
            numeric = (features if expanded else chars + industries) + [target]
            if any(parquet.schema_arrow.field(c).type != pa.float64() for c in numeric):
                raise ValueError("Source model columns must use float64.")
            current_schema = pa.schema([parquet.schema_arrow.field(c) for c in carry] +
                [pa.field(c, pa.float64()) for c in features])
            if schema is not None and current_schema != schema:
                raise ValueError("Source part schemas differ.")
            schema = current_schema
            for batch_number, batch in enumerate(parquet.iter_batches(
                    batch_size=batch_size, columns=selected, use_threads=False)):
                frame = batch.to_pandas(use_threads=False)
                dates = pd.PeriodIndex(frame[month], freq="M")
                formation = pd.PeriodIndex(frame[feature_month], freq="M")
                positions = frame[position].to_numpy()
                if (dates.isna().any() or formation.isna().any() or not (formation + 1 == dates).all() or
                        not (dates.year == year).all() or frame[parent["id_columns"]].isna().any().any() or
                        positions[0] <= previous.get(year, -1) or np.any(np.diff(positions) <= 0) or
                        positions[-1] >= parent["n_rows"]):
                    raise ValueError("Source identifiers, month alignment or row positions are invalid.")
                previous[year] = int(positions[-1])
                if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
                    raise ValueError("Nonfinite source model value.")
                if expanded:
                    prepared = frame[carry + features]
                else:
                    macro_values = macros.reindex(formation).to_numpy(dtype=np.float64)
                    if not np.isfinite(macro_values).all():
                        raise ValueError("Missing/nonfinite macro observation.")
                    prepared = pd.concat([frame[carry + chars],
                        pd.DataFrame(macro_values, columns=macro_names), frame[industries]], axis=1)
                table = pa.Table.from_pandas(prepared, schema=schema, preserve_index=False, safe=True)
                partition = output / f"return_year={year}"
                partition.mkdir(exist_ok=True)
                destination = partition / f"part-{part_number:06d}-{batch_number:06d}.parquet"
                temporary = destination.with_suffix(".parquet.tmp")
                pq.write_table(table, temporary, compression="snappy", row_group_size=batch_size)
                with pq.ParquetFile(temporary) as saved:
                    restored = saved.read(use_threads=False)
                if not table.equals(restored, check_metadata=False):
                    raise ValueError("Prepared Parquet read-back differs from input.")
                temporary.replace(destination)
                with destination.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                parts.append(dict(path=destination.relative_to(output).as_posix(), return_year=year,
                                  n_rows=len(frame), sha256=digest))
                count += len(frame)
                total += len(frame)
                year_counts[str(year)] = year_counts.get(str(year), 0) + len(frame)
        if count != part["n_rows"]:
            raise ValueError("Source part row count differs from metadata.")
        panel.verify_part(part)  # Refuse a source changed while it was being read.
        print(f"Prepared {total:,}/{parent['n_rows']:,} rows; {len(features)} predictors", flush=True)
    if ((source / "metadata.json").read_bytes() != metadata_bytes or
            (not expanded and macro_path.read_bytes() != macro_bytes)):
        raise ValueError("Source metadata or macro data changed during preparation.")
    if not total or total != parent["n_rows"] or year_counts != parent["year_counts"]:
        raise ValueError("Prepared sample differs from the source sample.")
    manifest = {**parent, "dataset_name": dataset_name, "parts": parts,
        "created_utc": datetime.now(timezone.utc).isoformat(), "feature_columns": features,
        "columns": schema.names, "dtypes": {field.name: str(field.type) for field in schema},
        "chunksize": batch_size, "compression": "snappy",
        "feature_specification": dict(kind="expanded" if expanded else "compact", characteristics=chars,
            macro_columns=[] if expanded else macro_names, industry_columns=industries, scaling="none",
            macro_products="copied unchanged from source" if expanded else "not included",
            macro_alignment="macro observation month = feature_month = return_month - 1"),
        "parent_dataset": dict(name=parent["dataset_name"], directory=str(source),
            metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest()),
        "pandas_version": pd.__version__, "pyarrow_version": pa.__version__}
    if not expanded:
        manifest["macro_source"] = dict(path=str(macro_path), sha256=hashlib.sha256(macro_bytes).hexdigest())
    atomic_write(output / "metadata.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=FEATURE_DATASETS, required=True)
    parser.add_argument("--raw-dir", type=Path, default=data_path / "raw")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    source_name = "gkx_dev_p05_seed2020_parquet" if args.dataset.endswith("_p05") else "gkx_parquet"
    prepare_features(source_dir=DATASET_PATHS[source_name], raw_dir=args.raw_dir,
        output_dir=DATASET_PATHS[args.dataset], dataset_name=args.dataset,
        expanded=args.dataset.startswith("gkx_expanded_"), batch_size=args.batch_size)
