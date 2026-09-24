"""Report date ranges and monthly distinct stock counts for Xiu and Green."""

from pathlib import Path

import pandas as pd


def read_time_coverage(path, chunk_size=100_000):
    """Read identifiers and dates in chunks, deduplicating stocks by month."""
    min_date = None
    max_date = None
    monthly_stocks = {}

    with pd.read_csv(
        path,
        usecols=["permno", "DATE"],
        dtype={"permno": "Int64", "DATE": "string"},
        chunksize=chunk_size,
    ) as reader:
        for chunk in reader:
            if chunk.empty:
                continue
            if chunk[["permno", "DATE"]].isna().any().any():
                raise ValueError(f"Missing PERMNO or DATE in {path}")

            dates = pd.to_datetime(chunk["DATE"], format="%Y%m%d", errors="raise")
            chunk_min = dates.min()
            chunk_max = dates.max()
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

            chunk["month"] = dates.dt.to_period("M")
            for month, group in chunk.groupby("month"):
                monthly_stocks.setdefault(month, set()).update(group["permno"].unique())

    counts = pd.Series(
        {month: len(stocks) for month, stocks in monthly_stocks.items()},
        dtype="int64",
    )
    return min_date, max_date, counts


def main(green_target_file, xiu_target_file):
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    output_dir = root / "results" / "time_coverage"

    date_ranges = []
    monthly_counts = {}
    for dataset, filename in {
        "xiu": xiu_target_file,
        "green": green_target_file,
    }.items():
        min_date, max_date, counts = read_time_coverage(raw_dir / filename)
        date_ranges.append({
            "dataset": dataset,
            "min_date": min_date,
            "max_date": max_date,
        })
        monthly_counts[f"n_{dataset}"] = counts

    coverage = pd.DataFrame(date_ranges)
    monthly = pd.DataFrame(monthly_counts).sort_index()
    if not monthly.empty:
        # Include every calendar month across the combined date range.
        months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
        monthly = monthly.reindex(months)
    monthly = monthly.fillna(0).astype("int64")
    monthly.index.name = "month"

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "date_coverage.csv", index=False, date_format="%Y-%m-%d")
    monthly.to_csv(output_dir / "monthly_stock_counts.csv")

    print("Date coverage:")
    print(coverage.to_string(index=False))
    print("\nMonthly distinct stock counts:")
    print(monthly.to_string())
    print(f"\nSaved tables to {output_dir}")


if __name__ == "__main__":
    green_target_file = "green_raw_v1_19570101_20211231.csv"  # green_raw_v1_19570101_20211231, green_raw_v1_20000101_20091231, RPSdata_RFS
    xiu_target_file = "datashare.csv"

    main(green_target_file, xiu_target_file)
