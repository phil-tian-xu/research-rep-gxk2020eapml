"""Extract small, untransformed source-data snapshots for browsing.

Run: python -B -m core.data_query.create_data_snapshot
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random

from utils.path_info import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=data_path / "processed" / "data_snapshot_20261021")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {args.output_dir}")

    characteristic_source = data_path / "raw" / "datashare.csv"
    macro_source = data_path / "raw" / "macro_predictor.csv"
    source_stats = {path: path.stat() for path in (characteristic_source, macro_source)}
    january_rows, selected_rows = [], []
    selected = None
    raw_digest = hashlib.sha256()
    previous_month = b""
    with characteristic_source.open("rb") as source:
        header = next(source)
        raw_digest.update(header)
        columns = next(csv.reader([header.decode("utf-8-sig").strip()]))
        if columns[:2] != ["permno", "DATE"] or len(columns) != 97 or "sic2" not in columns:
            raise ValueError("Expected permno, DATE, 94 characteristics, and sic2.")
        for line in source:
            raw_digest.update(line)
            permno, date, _ = line.split(b",", 2)
            month = date[:6]
            if month < previous_month:
                raise ValueError("The characteristic source must be sorted by month.")
            previous_month = month
            if month == b"201501":
                january_rows.append(line)
            elif b"201502" <= month <= b"201612":
                if selected is None:
                    universe = sorted({int(row.split(b",", 1)[0]) for row in january_rows})
                    selected = set(random.Random(2020).sample(universe, 100))
                    selected_rows.extend(row for row in january_rows
                                         if int(row.split(b",", 1)[0]) in selected)
                if int(permno) in selected:
                    selected_rows.append(line)
    if selected is None:
        raise ValueError("The requested characteristic months are unavailable.")
    selected_rows.sort(key=lambda row: (row.split(b",", 2)[1], int(row.split(b",", 1)[0])))
    characteristic_bytes = header.rstrip(b"\r\n") + b"\n" + b"".join(
        row.rstrip(b"\r\n") + b"\n" for row in selected_rows)
    characteristic_rows = list(csv.DictReader(io.StringIO(characteristic_bytes.decode("utf-8-sig"))))
    keys = {(row["permno"], row["DATE"][:6]) for row in characteristic_rows}
    stock_counts = Counter(int(row["permno"]) for row in characteristic_rows)
    month_counts = Counter(row["DATE"][:6] for row in characteristic_rows)
    if len(keys) != len(characteristic_rows) or set(stock_counts) != selected or len(month_counts) != 24:
        raise ValueError("Unexpected duplicate keys, stock count, or month coverage.")

    macro_bytes = macro_source.read_bytes()
    macro_lines = macro_bytes.splitlines(keepends=True)
    macro_columns = next(csv.reader([macro_lines[0].decode("utf-8-sig").strip()]))
    if macro_columns[0] != "yyyymm":
        raise ValueError("Expected yyyymm as the first macro column.")
    all_macro_rows = list(csv.DictReader(io.StringIO(macro_bytes.decode("utf-8-sig"))))
    source_months = [int(row["yyyymm"]) for row in all_macro_rows]
    latest_month = max(source_months)
    last_year = latest_month // 100 if latest_month % 100 == 12 else latest_month // 100 - 1
    macro_months = {year * 100 + month for year in (last_year - 1, last_year)
                    for month in range(1, 13)}
    macro_selected = [(month, line) for month, line in zip(source_months, macro_lines[1:], strict=True)
                      if month in macro_months]
    if len(macro_selected) != 24 or {month for month, _ in macro_selected} != macro_months:
        raise ValueError("The latest two complete macro years must contain 24 unique months.")
    macro_selected.sort()
    macro_output_bytes = macro_lines[0].rstrip(b"\r\n") + b"\n" + b"".join(
        line.rstrip(b"\r\n") + b"\n" for _, line in macro_selected)

    for path, before in source_stats.items():
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during extraction: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "characteristics_94_100stocks_2015_2016.csv": characteristic_bytes,
        f"goyal_welch_{last_year - 1}_{last_year}.csv": macro_output_bytes,
    }
    output_records = {}
    for name, content in outputs.items():
        destination = args.output_dir / name
        destination.write_bytes(content)
        if destination.read_bytes() != content:
            raise RuntimeError(f"Output read-back mismatch: {destination}")
        output_records[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Source-data browsing snapshot; the two tables cover different periods and are not merged.",
        "generator": "python -B -m core.data_query.create_data_snapshot",
        "transformations": "Date/stock filtering and row sorting only; source field text and missing values preserved. LF newlines.",
        "characteristics": {
            "months": ["2015-01", "2016-12"], "rows": len(characteristic_rows),
            "stocks": len(selected), "characteristic_count": 94,
            "identifier_columns": ["permno", "DATE", "sic2"],
            "date_meaning": "Original GKX DATE label (return month); no extra lag applied.",
            "sampling": "random.Random(2020).sample(sorted January-2015 PERMNOs, 100); fixed membership; no survival or nonmissing-value requirement.",
            "january_2015_universe": len(universe), "selected_permnos": sorted(selected),
            "rows_per_month": dict(sorted(month_counts.items())),
            "months_per_stock": dict(sorted(stock_counts.items())),
        },
        "macro": {"years": [last_year - 1, last_year], "rows": 24,
                  "columns": macro_columns, "source_latest_month": latest_month,
                  "date_meaning": "Original Goyal-Welch observation month, yyyymm; no lag or predictor construction."},
        "sources": [
            {"path": str(characteristic_source), "bytes": source_stats[characteristic_source].st_size,
             "sha256": raw_digest.hexdigest()},
            {"path": str(macro_source), "bytes": len(macro_bytes),
             "sha256": hashlib.sha256(macro_bytes).hexdigest()},
        ],
        "outputs": output_records,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(args.output_dir), "stock_rows": len(characteristic_rows),
                      "stocks": len(selected), "months_per_stock_min": min(stock_counts.values()),
                      "macro_years": metadata["macro"]["years"], "macro_rows": 24}, indent=2))


if __name__ == "__main__":
    main()
