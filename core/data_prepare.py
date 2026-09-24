"""Query legacy CRSP by year, appending to one CSV per table for the full sample.

Run from the repository root: python -m core.data_prepare
Install project dependencies first: uv sync
Set WRDS_USERNAME and configure the external pgpass file (see connect_wrds).
Configure the fields, universe, and dates below. None means all historical
CRSP securities; no filter is applied for survival or current listing status.
"""

import json
from datetime import date, datetime, timezone

import pandas as pd

from core.data_query.wrds_data_query import (
    add_crsp_price_fields,
    complete_crsp_months,
    connect_wrds,
    query_crsp_stock_data,
    query_crsp_universe,
    validate_crsp_frequency,
)
from utils.path_info import data_path
# Universe: a PERMNO list, or None for ALL securities (including delisted ones).
# Keep the small test default until ready for the full extraction.
# PERMNOS = [10107, 14593]
PERMNOS = None

# Native frequencies: "monthly" (default) or "daily". Weekly/annual requests
# raise guidance to download native data and compound in a later processing step.
FREQUENCY = "monthly"
# Choose raw fields. permno and date are always added automatically.
# See wrds_data_query.CRSP_MONTHLY_FIELDS / CRSP_DAILY_FIELDS for definitions.
FIELDS = [
    "permco", "ret", "retx", "prc", "shrout", "vol",
    "cfacpr", "cfacshr", "bidlo", "askhi",
]
# Dates refer to realized-return periods, not feature periods. Monthly queries
# require whole months; daily queries allow any inclusive date range.
START_DATE = "1954-01-01"
END_DATE = "2026-12-31"
INCLUDE_DAILY_AMOUNT = True
# Raw extracts stay sparse. The intermediate panel adds NA rows for every
# universe security/month, including before listing and after delisting.
# Applies only to monthly data; daily extracts retain observed trading dates.
COMPLETE_MONTH_GRID = True


def main():
    validate_crsp_frequency(FREQUENCY)  # Fail before login or creating files.
    start, end = date.fromisoformat(START_DATE), date.fromisoformat(END_DATE)
    if start > end:
        raise ValueError("START_DATE must not be after END_DATE.")
    if FREQUENCY == "monthly" and (start.day != 1 or not pd.Timestamp(end).is_month_end):
        raise ValueError("Use an ordered range of whole calendar months.")
    complete_month_grid = COMPLETE_MONTH_GRID and FREQUENCY == "monthly"
    if COMPLETE_MONTH_GRID and FREQUENCY == "daily":
        print("COMPLETE_MONTH_GRID applies only to monthly data; daily output retains observed rows.")
    stock_table = "msf" if FREQUENCY == "monthly" else "dsf"
    delist_table = "msedelist" if FREQUENCY == "monthly" else "dsedelist"
    raw_name, panel_name = f"{FREQUENCY}_raw", f"{FREQUENCY}_panel"
    retrieved = datetime.now(timezone.utc)
    run_id = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = data_path / "raw" / "crsp" / run_id
    derived_dir = data_path / "intermediate" / "crsp" / run_id
    metadata = {
        "retrieved_utc": retrieved.isoformat(),
        "source_tables": [f"crsp.{stock_table}", f"crsp.{delist_table}"]
        + (["crsp.dsf"] if INCLUDE_DAILY_AMOUNT and FREQUENCY == "monthly" else []),
        "frequency": FREQUENCY,
        "format": "legacy; no automatic CIZ fallback",
        "permnos": PERMNOS,
        "universe": f"all historical PERMNOs in crsp.{stock_table}" if PERMNOS is None else "specified PERMNOs",
        "fields": list(dict.fromkeys(["permno", "date", *FIELDS])),
        "complete_month_grid": complete_month_grid,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "yearly_row_counts": {},
        "status": "incomplete",
        "amount_definition": "Positive daily closing price times nonnegative daily volume; proxy only; summed by month only for monthly output",
        "return_treatment": "Unchanged; no delisting adjustment, RF subtraction, or lead shift",
    }
    observed_permnos = set()
    db = connect_wrds()
    try:
        # Query the historical catalog once, not once per year. It can include
        # securities with no observations anywhere in the requested date range.
        universe = (
            query_crsp_universe(db, frequency=FREQUENCY)["permno"].astype(int).tolist()
            if PERMNOS is None else sorted(set(PERMNOS))
        )
        if not universe:
            raise ValueError("Empty universe; use None explicitly for all CRSP securities.")
        output_dir.mkdir(parents=True, exist_ok=False)
        derived_dir.mkdir(parents=True, exist_ok=False)
        pd.DataFrame({"permno": universe}).to_csv(output_dir / "universe.csv", index=False)
        metadata_path = output_dir / "query_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        for year in range(start.year, end.year + 1):
            # Write each header once, even if the first year's table is empty.
            first_year = year == start.year
            write_mode = "w" if first_year else "a"
            window_start = max(start, date(year, 1, 1)).isoformat()
            window_end = min(end, date(year, 12, 31)).isoformat()
            tables = query_crsp_stock_data(
                db, PERMNOS, window_start, window_end,
                include_daily_amount=INCLUDE_DAILY_AMOUNT, fields=FIELDS,
                frequency=FREQUENCY,
            )
            for name, frame in tables.items():
                frame.to_csv(
                    output_dir / f"{name}.csv", index=False,
                    mode=write_mode, header=first_year,
                )
            observed_permnos.update(tables[raw_name]["permno"])
            derived = add_crsp_price_fields(tables[raw_name], frequency=FREQUENCY)
            if complete_month_grid:
                derived = complete_crsp_months(derived, universe, window_start, window_end)
            else:
                derived["has_crsp_row"] = True
            derived.to_csv(
                derived_dir / f"{panel_name}.csv", index=False,
                mode=write_mode, header=first_year,
            )
            metadata["yearly_row_counts"][str(year)] = {
                name: len(frame) for name, frame in tables.items()
            }
            metadata["yearly_row_counts"][str(year)][panel_name] = len(derived)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"{year}: {len(tables[raw_name]):,} observed rows; {len(derived):,} panel rows")

        missing = sorted(set(universe) - observed_permnos)
        pd.DataFrame({"permno": missing}).to_csv(output_dir / "permnos_without_rows.csv", index=False)
        metadata["status"] = "complete"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"PERMNOs with no {FREQUENCY} observations in this date range: {len(missing):,}")
    finally:
        db.close()
    print(f"Raw extract: {output_dir}")
    print(f"{FREQUENCY.capitalize()} panel: {derived_dir}")


if __name__ == "__main__":
    main()
