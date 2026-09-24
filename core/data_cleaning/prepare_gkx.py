"""Build the saved GKX panel independently of the model workflow.

Run: python -B -m core.data_cleaning.prepare_gkx
Raw inputs are never changed. An existing output directory is never overwritten.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import data_path


MACROS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
SOURCE_FILES = ["datashare.csv", "macro_predictor.csv",
                "risk_free_rate_fama_french.csv", "monthly_raw.csv"]


def read_macros_and_rf(raw):
    # Spreadsheet exports quote index levels such as "1,049.34".
    macro = pd.read_csv(raw / "macro_predictor.csv", dtype={"yyyymm": str}, thousands=",")
    macro.index = pd.to_datetime(macro.pop("yyyymm"), format="%Y%m", errors="raise").dt.to_period("M")
    if macro.index.has_duplicates or macro.index.isna().any():
        raise ValueError("Macro months must be unique and nonmissing.")
    for col in ["Index", "D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis", "svar"]:
        try:
            macro[col] = pd.to_numeric(macro[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"macro_predictor.csv: invalid numeric values in {col!r}.") from exc
    for col in ["Index", "D12", "E12"]:
        if (macro[col].dropna() <= 0).any():
            raise ValueError(f"Nonpositive {col}: cannot construct log ratios.")
    predictors = pd.DataFrame({
        "dp": np.log(macro.D12) - np.log(macro.Index),
        "ep": np.log(macro.E12) - np.log(macro.Index),
        "bm": macro["b/m"], "ntis": macro.ntis, "tbl": macro.tbl,
        "tms": macro.lty - macro.tbl, "dfy": macro.BAA - macro.AAA,
        "svar": macro.svar,
    })
    rf = pd.read_csv(raw / "risk_free_rate_fama_french.csv", dtype=str)
    months = pd.to_datetime(rf.iloc[:, 0], format="%Y%m", errors="raise").dt.to_period("M")
    risk_free = pd.Series(pd.to_numeric(rf["RF"], errors="raise").to_numpy() / 100,
                          index=months, name="rf")
    if risk_free.index.has_duplicates or risk_free.index.isna().any():
        raise ValueError("RF months must be unique and nonmissing.")
    return predictors, risk_free


def monthly_characteristics(path, start, end, chunksize):
    """Stream complete months; require nondecreasing source dates by month."""
    pending = None
    previous = None
    with pd.read_csv(path, chunksize=chunksize) as reader:
        for chunk in reader:
            chunk["return_month"] = pd.to_datetime(chunk.DATE.astype(str), format="%Y%m%d",
                                                   errors="raise").dt.to_period("M")
            months = chunk.return_month
            if months.isna().any() or not months.is_monotonic_increasing or (
                previous is not None and months.iloc[0] < previous
            ):
                raise ValueError("datashare.csv must be ordered by nonmissing month; no rows were silently reordered.")
            previous = months.iloc[-1]
            chunk = chunk.loc[months.between(start, end)]
            if not chunk.empty:
                combined = pd.concat([pending, chunk], ignore_index=True) if pending is not None else chunk
                last_month = combined.return_month.iloc[-1]
                for month, frame in combined.loc[combined.return_month != last_month].groupby("return_month", sort=True):
                    yield month, frame
                pending = combined.loc[combined.return_month == last_month].copy()
            if previous > end:
                break
    if pending is not None:
        yield pending.return_month.iloc[0], pending


def rank_characteristics(frame):
    """Median-impute then average-rank; map ordinal ranks by 2*(rank-1)/(n-1)-1.

    Entirely missing and constant cross-sections map to zero. These conventions
    are explicit replication choices, not a claimed exact author implementation.
    """
    values = frame.apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite firm characteristic.")
    filled = values.fillna(values.median()).fillna(0.0)
    if len(filled) == 1:
        return filled * 0.0
    return 2 * (filled.rank(method="average") - 1) / (len(filled) - 1) - 1


def prepare(raw_dir=data_path / "raw", output_dir=data_path / "processed" / "gkx",
            start_month="1957-03", end_month="2016-12", chunksize=100_000):
    raw, output = Path(raw_dir), Path(output_dir)
    start, end = pd.Period(start_month, "M"), pd.Period(end_month, "M")
    if start > end or chunksize < 1:
        raise ValueError("Require start <= end and positive chunksize.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_info = {}
    for name in SOURCE_FILES:
        path = raw / name
        stat = path.stat()
        source_info[name] = {"path": str(path.resolve()), "bytes": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
    with (raw / "datashare.csv").open(newline="", encoding="utf-8") as source:
        header = next(csv.reader(source))
    chars = [col for col in header if col not in {"permno", "DATE", "sic2"}]
    if len(header) != len(set(header)) or len(chars) != 94 or not {"permno", "DATE", "sic2"}.issubset(header):
        raise ValueError("Expected identifiers, sic2, and exactly 94 unique characteristics.")
    macros, rf = read_macros_and_rf(raw)
    expected_months = pd.period_range(start, end, freq="M")
    if not np.isfinite(macros.reindex(expected_months - 1).to_numpy()).all():
        raise ValueError("Missing or nonfinite macro predictors in the required feature months.")
    if not np.isfinite(rf.reindex(expected_months).to_numpy()).all():
        raise ValueError("Missing or nonfinite RF in the required return months.")

    # Cheap first pass establishes one fixed industry vocabulary for every month.
    industries = set()
    with pd.read_csv(raw / "datashare.csv", usecols=["DATE", "sic2"], chunksize=chunksize) as reader:
        for chunk in reader:
            dates = pd.to_datetime(chunk.DATE.astype(str), format="%Y%m%d").dt.to_period("M")
            sic = pd.to_numeric(chunk.loc[dates.between(start, end), "sic2"], errors="raise").dropna()
            if not (sic.between(0, 99) & (sic == np.floor(sic))).all():
                raise ValueError("Invalid nonmissing sic2 in the selected sample.")
            industries.update(sic.astype(int))
    industries = sorted(industries)

    # Only the narrow target table is held in memory, never the expanded panel.
    parts = []
    for chunk in pd.read_csv(raw / "monthly_raw.csv", usecols=["permno", "date", "ret"],
                             dtype={"ret": str}, chunksize=chunksize):
        chunk["return_month"] = pd.to_datetime(chunk.pop("date"), errors="raise").dt.to_period("M")
        chunk = chunk.loc[chunk.return_month.between(start, end)].copy()
        # Legacy CRSP nonnumeric return missing codes; unexpected strings fail.
        missing = chunk.ret.isin(["A", "B", "C", ".", ".A", ".B", ".C"])
        chunk["ret"] = pd.to_numeric(chunk.ret.mask(missing), errors="raise")
        parts.append(chunk)
    returns = pd.concat(parts, ignore_index=True)
    if returns.permno.isna().any() or returns.duplicated(["permno", "return_month"]).any():
        raise ValueError("Missing CRSP identifiers or duplicate CRSP security-month keys.")
    if np.isinf(returns.ret).any() or (returns.ret.dropna() < -1).any():
        raise ValueError("Invalid CRSP simple returns.")
    returns = returns.set_index(["return_month", "permno"]).sort_index()
    del parts
    features = chars + [f"{char}__macro_{macro}" for macro in MACROS for char in chars]
    features += [f"sic2_{code:02d}" for code in industries]
    if len(features) != len(set(features)):
        raise ValueError("Feature name collision.")
    output.mkdir(parents=True, exist_ok=False)
    panel_path = output / "panel.csv"
    reports, total, seen = [], 0, []
    for month, frame in monthly_characteristics(raw / "datashare.csv", start, end, chunksize):
        seen.append(month)
        frame = frame.sort_values("permno").reset_index(drop=True)
        if frame.permno.isna().any() or frame.permno.duplicated().any():
            raise ValueError(f"Missing/duplicate GKX security identifiers: {month}")
        if not (frame.permno == np.floor(frame.permno)).all():
            raise ValueError(f"Noninteger PERMNO: {month}")
        ranked = rank_characteristics(frame[chars])
        macro = macros.loc[month - 1]
        blocks = [ranked]
        for name in MACROS:
            blocks.append((ranked * macro[name]).rename(columns=lambda col: f"{col}__macro_{name}"))
        sic = pd.to_numeric(frame.sic2, errors="raise")
        sic_missing = sic.isna()
        blocks.append(pd.DataFrame({f"sic2_{code:02d}": (sic == code).fillna(False).astype(int)
                                   for code in industries}))
        panel = pd.concat(blocks, axis=1)
        # Audit columns are deliberately excluded from metadata's feature_columns.
        panel["sic2"] = sic.astype("Int64")
        panel["sic_missing"] = sic_missing.astype(int)
        keys = pd.MultiIndex.from_arrays([[month] * len(frame), frame.permno], names=["return_month", "permno"])
        target = returns.ret.reindex(keys)
        matched = keys.isin(returns.index)
        valid = target.notna().to_numpy()
        panel.insert(0, "excess_return", target.to_numpy() - rf.loc[month])
        panel.insert(0, "return_month", str(month))
        panel.insert(0, "feature_month", str(month - 1))
        panel.insert(0, "permno", frame.permno.astype("int64"))
        panel = panel.loc[valid]
        if not np.isfinite(panel[["excess_return", *features]].to_numpy()).all():
            raise ValueError(f"Nonfinite final model inputs: {month}")
        panel.to_csv(panel_path, mode="a", header=not panel_path.exists(), index=False,
                     float_format="%.17g", lineterminator="\n")
        total += len(panel)
        reports.append({"return_month": str(month), "gkx_rows": len(frame),
                        "unmatched_crsp_rows": int((~matched).sum()),
                        "matched_missing_returns": int((matched & ~valid).sum()),
                        "excluded_rows": int((~valid).sum()), "output_rows": len(panel),
                        "missing_sic_input_rows": int(sic_missing.sum()),
                        "missing_sic_output_rows": int((sic_missing & valid).sum()),
                        "missing_characteristic_cells": int(frame[chars].isna().sum().sum()),
                        "all_missing_characteristics": ";".join(frame[chars].columns[frame[chars].isna().all()])})
        pd.DataFrame(reports).to_csv(output / "preparation_report.csv", index=False)
        print(f"{month}: {len(panel):,} rows saved; {(~valid).sum():,} excluded", flush=True)
    if seen != list(expected_months) or total == 0:
        raise ValueError("Missing GKX months or empty output. Partial output has no completion metadata.")
    for name, info in source_info.items():
        stat = (raw / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (info["bytes"], info["mtime_ns"]):
            raise ValueError(f"Source changed during preparation: {name}")
    with panel_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "schema_version": 1, "dataset_name": "gkx", "dataset_type": "empirical",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "id_columns": ["permno"], "feature_month_column": "feature_month",
        "return_month_column": "return_month", "target_column": "excess_return",
        "feature_columns": features, "return_units": "decimal", "n_rows": total,
        "panel_sha256": digest, "sources": source_info,
        "audit_columns": ["sic2", "sic_missing"],
        "preparation": {
            "start_month": str(start), "end_month": str(end), "industry_codes": industries,
            "n_characteristics": len(chars), "n_features": len(features),
            "alignment": "GKX DATE = return month; macro observation month = return month - 1; RF = return month",
            "target": "CRSP ret - Fama French RF/100; no additional delisting adjustment",
            "universe": "All GKX rows in sample; missing targets excluded AFTER characteristic ranking",
            "imputation": "Monthly raw-characteristic median before ranking; entirely missing columns mapped to zero",
            "ranking": "Average tied ranks; 2*(rank-1)/(n-1)-1; singleton and constant cross-sections zero",
            "macro_formulas": {"dp": "ln(D12)-ln(Index)", "ep": "ln(E12)-ln(Index)",
                               "bm": "b/m", "ntis": "ntis", "tbl": "tbl", "tms": "lty-tbl",
                               "dfy": "BAA-AAA", "svar": "svar"},
            "industry_vocabulary": "Sorted observed SIC codes across selected GKX sample; all dummies retained",
            "missing_sic": "Retain rows; all industry dummies zero; preserve NA sic2 and sic_missing flag outside model features; no temporal filling",
            "standardization": "None; fitted scaling belongs in model training",
            "pandas_version": pd.__version__, "numpy_version": np.__version__,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=data_path / "raw")
    parser.add_argument("--output-dir", type=Path, default=data_path / "processed" / "gkx")
    parser.add_argument("--start-month", default="1957-03")
    parser.add_argument("--end-month", default="2016-12")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    print(f"Prepared: {prepare(**vars(args))}")


if __name__ == "__main__":
    main()
