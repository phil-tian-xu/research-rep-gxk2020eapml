"""Inspect SIC coverage in the entire GKX file without changing any values.

Run: python -B -m core.raw_data_examinine.data_check_sic
Outputs CSVs under data/sic_inspection; existing directories are not overwritten.
Dates are original GKX DATE labels, not shifted forecast formation dates.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from utils.path_info import data_path


def inspect_sic(input_path, output_dir, chunk_size=100_000):
    input_path, output_dir = Path(input_path), Path(output_dir)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    missing_path = output_dir / "missing_sic_rows.csv"
    pd.DataFrame(columns=["permno", "DATE"]).to_csv(missing_path, index=False)
    summaries = []
    code_counts = Counter()
    code_permnos, permno_codes = defaultdict(set), defaultdict(set)
    with pd.read_csv(input_path, usecols=["permno", "DATE", "sic2"],
                     dtype={"permno": "Int64", "DATE": "string", "sic2": "string"},
                     chunksize=chunk_size) as reader:
        for chunk in reader:
            if chunk[["permno", "DATE"]].isna().any().any():
                raise ValueError("Missing PERMNO or DATE in source.")
            pd.to_datetime(chunk.DATE, format="%Y%m%d", errors="raise")
            sic = pd.to_numeric(chunk.sic2, errors="raise")
            observed = sic.dropna()
            if not (observed.between(0, 99) & (observed % 1 == 0)).all():
                raise ValueError("Nonmissing SIC must be an integer between 0 and 99.")
            chunk["sic2"] = sic.astype("Int64")
            missing = chunk.sic2.isna()
            chunk.loc[missing, ["permno", "DATE"]].to_csv(
                missing_path, mode="a", index=False, header=False)
            chunk["missing_date"] = chunk.DATE.where(missing)
            chunk["valid_date"] = chunk.DATE.where(~missing)
            summaries.append(chunk.groupby("permno").agg(
                n_rows=("DATE", "size"), n_valid_sic=("sic2", "count"),
                first_date=("DATE", "min"), last_date=("DATE", "max"),
                first_missing_date=("missing_date", "min"), last_missing_date=("missing_date", "max"),
                first_valid_date=("valid_date", "min"), last_valid_date=("valid_date", "max")))
            code_counts.update(chunk.sic2.dropna().astype(int).tolist())
            pairs = chunk.loc[~missing, ["permno", "sic2"]].drop_duplicates()
            for permno, code in pairs.itertuples(index=False, name=None):
                code_permnos[int(code)].add(int(permno))
                permno_codes[int(permno)].add(int(code))

    if not summaries:
        raise ValueError("Source has no observations.")
    universe = pd.concat(summaries).groupby(level=0).agg({
        "n_rows": "sum", "n_valid_sic": "sum", "first_date": "min", "last_date": "max",
        "first_missing_date": "min", "last_missing_date": "max",
        "first_valid_date": "min", "last_valid_date": "max"})
    universe["n_missing_sic"] = universe.n_rows - universe.n_valid_sic
    universe["missing_fraction"] = universe.n_missing_sic / universe.n_rows
    universe["sic_status"] = "sometimes_missing"
    universe.loc[universe.n_missing_sic == 0, "sic_status"] = "never_missing"
    universe.loc[universe.n_valid_sic == 0, "sic_status"] = "always_missing"
    universe["n_distinct_sic"] = [len(permno_codes[p]) for p in universe.index]
    universe["observed_sic_codes"] = [";".join(f"{c:02d}" for c in sorted(permno_codes[p]))
                                      for p in universe.index]
    universe.to_csv(output_dir / "permno_sic_summary.csv", index_label="permno")
    pd.DataFrame([{"sic2": code, "sic2_label": f"{code:02d}", "n_rows": count,
                   "n_permnos": len(code_permnos[code])} for code, count in sorted(code_counts.items())],
                 columns=["sic2", "sic2_label", "n_rows", "n_permnos"]).to_csv(
                     output_dir / "sic_values.csv", index=False)
    summary = {
        "input_path": str(input_path.resolve()), "scope": "entire source file; no date or stock filters",
        "first_date": universe.first_date.min(), "last_date": universe.last_date.max(),
        "n_rows": int(universe.n_rows.sum()), "n_permnos": len(universe),
        "n_observed_sic_codes": len(code_counts), "n_missing_sic_rows": int(universe.n_missing_sic.sum()),
        "n_permnos_always_missing": int((universe.sic_status == "always_missing").sum()),
        "n_permnos_sometimes_missing": int((universe.sic_status == "sometimes_missing").sum()),
        "n_permnos_never_missing": int((universe.sic_status == "never_missing").sum()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"CSV reports: {output_dir.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=data_path / "raw" / "datashare.csv")
    parser.add_argument("--output-dir", type=Path, default=data_path / "sic_inspection")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    inspect_sic(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
