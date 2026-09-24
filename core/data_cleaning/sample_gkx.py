"""Sample a fixed PERMNO universe and copy its complete processed histories.

Run: python -B -m core.data_cleaning.sample_gkx
Creates gkx_dev_p05_seed2020 and gkx_dev_p10_seed2020 by default.
The same seed makes the lower-probability sample a subset of the larger one.
"""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from utils.path_info import data_path


def sample_gkx(probability, *, seed=2020, source_dir=data_path / "processed" / "gkx",
               output_root=data_path / "processed"):
    """Select once per PERMNO using SHA-256(seed:PERMNO); preserve row bytes.

    This reader targets the numeric, single-line CSV emitted by prepare_gkx.
    It validates the leading identifier/date columns and verifies the complete
    parent's hash before publishing metadata. Memory scales with stock count.
    """
    probability = Decimal(str(probability))
    if not probability.is_finite() or not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1].")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    label = format(probability * 100, "f").rstrip("0").rstrip(".") if probability * 100 % 1 else str(int(probability * 100))
    label = label.zfill(2).replace(".", "p")
    name = f"gkx_dev_p{label}_seed{seed}"
    source_dir, output = Path(source_dir), Path(output_root) / name
    parent_bytes = (source_dir / "metadata.json").read_bytes()
    parent = json.loads(parent_bytes)
    if parent["id_columns"] != ["permno"]:
        raise ValueError("Expected a PERMNO-indexed parent panel.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    numerator, denominator = probability.as_integer_ratio()
    threshold = numerator * (1 << 256) // denominator
    universe, monthly = {}, {}
    total = kept = 0
    output.mkdir(parents=True, exist_ok=False)
    with (source_dir / "panel.csv").open("rb") as source, (output / "panel.csv").open("wb") as target:
        header = source.readline()
        columns = next(csv.reader([header.decode("utf-8").strip()]))
        if columns[:3] != ["permno", parent["feature_month_column"], parent["return_month_column"]]:
            raise ValueError("Expected leading permno, feature_month, return_month columns.")
        source_hash.update(header)
        output_hash.update(header)
        target.write(header)
        previous_month = None
        for line in source:
            source_hash.update(line)
            fields = line.split(b",", 3)
            if len(fields) != 4:
                raise ValueError("Expected one numeric panel observation per CSV line.")
            permno = int(fields[0])
            month = fields[2].decode("ascii")
            if previous_month is not None and month < previous_month:
                raise ValueError("Parent panel must be ordered by return_month.")
            previous_month = month
            if permno not in universe:
                score = int.from_bytes(hashlib.sha256(f"{seed}:{permno}".encode("ascii")).digest(), "big")
                universe[permno] = {"selected": score < threshold, "n_rows": 0,
                                    "first_month": month, "last_month": month}
            stock = universe[permno]
            stock["n_rows"] += 1
            stock["last_month"] = month
            counts = monthly.setdefault(month, [0, 0])
            counts[0] += 1
            total += 1
            if stock["selected"]:
                target.write(line)
                output_hash.update(line)
                counts[1] += 1
                kept += 1
            if total % 500_000 == 0:
                print(f"{name}: scanned {total:,} rows; retained {kept:,}", flush=True)
    if source_hash.hexdigest() != parent["panel_sha256"] or total != parent["n_rows"]:
        raise ValueError("Parent hash/row count mismatch; output has no completion metadata.")
    if not kept or any(counts[1] == 0 for counts in monthly.values()):
        raise ValueError("Sample is empty in at least one month; output has no completion metadata.")

    selected_count = sum(stock["selected"] for stock in universe.values())
    full_count = sum(stock["n_rows"] == len(monthly) for stock in universe.values())
    selected_full = sum(stock["selected"] and stock["n_rows"] == len(monthly) for stock in universe.values())
    with (output / "universe_selection.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(["permno", "selected", "n_rows", "first_month", "last_month", "full_month_coverage"])
        for permno, stock in sorted(universe.items()):
            writer.writerow([permno, int(stock["selected"]), stock["n_rows"], stock["first_month"],
                             stock["last_month"], int(stock["n_rows"] == len(monthly))])
    with (output / "monthly_sampling.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(["return_month", "parent_rows", "sample_rows", "row_fraction"])
        for month, counts in sorted(monthly.items()):
            writer.writerow([month, *counts, counts[1] / counts[0]])
    sampling = {
        "probability": float(probability), "seed": seed,
        "method": "SHA256 of ASCII seed:integer_PERMNO interpreted as unsigned big-endian integer < floor(probability*2**256)",
        "unit": "PERMNO; retain every available parent row for selected securities",
        "parent_rows": total, "sample_rows": kept, "row_fraction": kept / total,
        "parent_permnos": len(universe), "sample_permnos": selected_count,
        "permno_fraction": selected_count / len(universe), "n_months": len(monthly),
        "parent_full_coverage_permnos": full_count, "sample_full_coverage_permnos": selected_full,
        "parent_full_coverage_fraction": full_count / len(universe),
        "sample_full_coverage_fraction": selected_full / selected_count,
        "full_coverage_definition": "Number of parent observations equals number of parent months; parent security-month uniqueness required",
        "transformations": "None; header and selected data rows copied byte-for-byte, retaining original order, ranks, interactions and gaps",
    }
    (output / "sampling_summary.json").write_text(json.dumps(sampling, indent=2) + "\n", encoding="utf-8")
    metadata = dict(parent)
    metadata.update(dataset_name=name, dataset_type="empirical_development_subsample",
                    created_utc=datetime.now(timezone.utc).isoformat(), n_rows=kept,
                    panel_sha256=output_hash.hexdigest(), sampling=sampling,
                    parent_dataset={"directory": str(source_dir.resolve()), "dataset_name": parent["dataset_name"],
                                    "panel_sha256": parent["panel_sha256"],
                                    "metadata_sha256": hashlib.sha256(parent_bytes).hexdigest()})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{name}: {selected_count:,}/{len(universe):,} PERMNOs; {kept:,}/{total:,} rows ({kept/total:.2%})", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", nargs="+", type=float, default=[.05, .10])
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--source-dir", type=Path, default=data_path / "processed" / "gkx")
    parser.add_argument("--output-root", type=Path, default=data_path / "processed")
    args = parser.parse_args()
    for probability in args.probabilities:
        sample_gkx(probability, seed=args.seed, source_dir=args.source_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
