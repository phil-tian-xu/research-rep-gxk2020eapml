"""Compare Xiu (datashare) and Green (RPS) stock coverage."""

from pathlib import Path

import pandas as pd


def read_stock_sets(path, chunk_size=100_000):
    """Read only identifiers/dates; keep distinct PERMNOs for each month."""
    monthly = {}
    with pd.read_csv(
        path,
        usecols=["permno", "DATE"],
        dtype={"permno": "Int64", "DATE": "string"},
        chunksize=chunk_size,
    ) as reader:
        for chunk in reader:
            if chunk[["permno", "DATE"]].isna().any().any():
                raise ValueError(f"Missing PERMNO or DATE in {path}")
            chunk["month"] = pd.to_datetime(
                chunk["DATE"], format="%Y%m%d", errors="raise"
            ).dt.strftime("%Y-%m")
            for month, group in chunk.groupby("month"):
                monthly.setdefault(month, set()).update(group["permno"].unique())
    return monthly


def compare_stocks(xiu, green):
    """Percentages are on a 0–100 scale; empty denominators yield NaN."""
    both = len(xiu & green)
    return {
        "n_xiu": len(xiu),
        "n_green": len(green),
        "n_both": both,
        "n_only_xiu": len(xiu - green),
        "n_only_green": len(green - xiu),
        "pct_xiu_in_green": 100 * both / len(xiu) if xiu else float("nan"),
        "pct_green_in_xiu": 100 * both / len(green) if green else float("nan"),
    }


def main(green_target_file, xiu_target_file):
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    output_dir = root / "results" / "stock_coverage"

    xiu_monthly = read_stock_sets(raw_dir / xiu_target_file)
    green_monthly = read_stock_sets(raw_dir / green_target_file)
    xiu = set().union(*xiu_monthly.values())
    green = set().union(*green_monthly.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    # Overall membership uses each dataset's entire available sample.
    for name, stocks in {
        "xiu_permnos": xiu,
        "green_permnos": green,
        "both_permnos": xiu & green,
        "only_xiu_permnos": xiu - green,
        "only_green_permnos": green - xiu,
    }.items():
        pd.DataFrame({"permno": sorted(stocks)}).to_csv(
            output_dir / f"{name}.csv", index=False
        )

    overall = pd.DataFrame([compare_stocks(xiu, green)])
    overall.to_csv(output_dir / "overall_coverage.csv", index=False)

    # Compare distinct stocks within the overlapping calendar period.
    overlap_rows = []

    if xiu_monthly and green_monthly:
        overlap_start = max(min(xiu_monthly), min(green_monthly))
        overlap_end = min(max(xiu_monthly), max(green_monthly))

        if overlap_start <= overlap_end:
            xiu_overlap = set().union(*(
                stocks
                for month, stocks in xiu_monthly.items()
                if overlap_start <= month <= overlap_end
            ))
            green_overlap = set().union(*(
                stocks
                for month, stocks in green_monthly.items()
                if overlap_start <= month <= overlap_end
            ))

            overlap_rows.append({
                "start_month": overlap_start,
                "end_month": overlap_end,
                **compare_stocks(xiu_overlap, green_overlap),
            })

    overlap = pd.DataFrame(
        overlap_rows,
        columns=["start_month", "end_month", *overall.columns],
    )
    overlap.to_csv(output_dir / "overlap_coverage.csv", index=False)

    print("\nOverlapping-period distinct PERMNO coverage:")
    if overlap.empty:
        print("No overlapping period, or an input dataset is empty.")
    else:
        print(overlap.to_string(index=False))


    # Include all observed months, including months absent from one dataset.
    rows = []
    for month in sorted(xiu_monthly.keys() | green_monthly.keys()):
        rows.append({
            "month": month,
            **compare_stocks(
                xiu_monthly.get(month, set()),
                green_monthly.get(month, set()),
            ),
        })
    monthly = pd.DataFrame(rows, columns=["month", *overall.columns])
    monthly.to_csv(output_dir / "monthly_coverage.csv", index=False)

    print("Overall distinct PERMNO coverage:")
    print(overall.to_string(index=False))
    print(f"Saved PERMNO lists and coverage tables to {output_dir}")


if __name__ == "__main__":
    green_target_file = "green_raw_v1_19570101_20211231.csv"  # green_raw_v1_19570101_20211231, green_raw_v1_20000101_20091231, RPSdata_RFS
    xiu_target_file = "datashare.csv"

    main(green_target_file, xiu_target_file)

