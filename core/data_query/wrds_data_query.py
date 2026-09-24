"""Selected-universe or all-security queries against legacy CRSP on WRDS.

Requires pandas and wrds. These queries intentionally use crsp.msf/dsf,
not CIZ: do not substitute CIZ tables without reviewing their definitions.
No stock filters, return imputation, RF subtraction, or target shifting here.
"""

import os
from datetime import date
from numbers import Integral

import pandas as pd
import wrds


CRSP_MONTHLY_FIELDS = {
    "permno": "Permanent security identifier (not company identifier).",
    "permco": "Permanent company identifier; a company can have multiple PERMNOs.",
    "date": "CRSP monthly observation date; use calendar month for joins.",
    "ret": "Holding-period total return including distributions, decimal units.",
    "retx": "Holding-period return excluding dividends, decimal units.",
    "prc": "Unadjusted month-end price in USD; negative flags a bid/ask average.",
    "shrout": "Unadjusted shares outstanding, in thousands of shares.",
    "vol": "Legacy monthly trading volume, in hundreds of shares.",
    "cfacpr": "Cumulative factor for price adjustment: abs(prc) / cfacpr.",
    "cfacshr": "Cumulative factor for share adjustment: shrout * cfacshr.",
    "bidlo": "Lowest daily price or bid/ask average during the month.",
    "askhi": "Highest daily price or bid/ask average during the month.",
}


CRSP_DAILY_FIELDS = {
    **CRSP_MONTHLY_FIELDS,
    "date": "CRSP daily observation date.",
    "prc": "Unadjusted daily closing price in USD; negative flags a bid/ask average.",
    "vol": "Legacy daily trading volume, in shares.",
    "bidlo": "Daily low price, or bid quotation when a low price is unavailable.",
    "askhi": "Daily high price, or ask quotation when a high price is unavailable.",
}


def validate_crsp_frequency(frequency):
    """Accept native stock-file frequencies; leave resampling to processing."""
    if frequency in {"weekly", "annual", "annually", "yearly"}:
        raise ValueError(
            "Legacy CRSP stock files queried here support daily and monthly data. "
            "For weekly returns, download daily data; for annual returns, download "
            "monthly or daily data and compound later as prod(1 + ret) - 1. "
            "Raw data preparation does not compound returns."
        )
    if frequency not in {"monthly", "daily"}:
        raise ValueError("frequency must be 'monthly' or 'daily'.")


def query_crsp_universe(db, frequency="monthly"):
    """All historical PERMNOs, including securities inactive in our window.

    This is a catalog for extraction/padding, not an investable universe at each
    date. Do not treat synthetic missing rows as eligible stock observations.
    """
    validate_crsp_frequency(frequency)
    table = "msf" if frequency == "monthly" else "dsf"
    return db.raw_sql(f"SELECT DISTINCT permno FROM crsp.{table} ORDER BY permno")


def connect_wrds(username=None):
    r"""Use WRDS_USERNAME and PostgreSQL's external password file.

    Windows: %APPDATA%/postgresql/pgpass.conf
    Entry: wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD
    Escape literal backslashes as \\ and colons as \: in the actual file.
    Alternatively, log in once using wrds.Connection() and accept its offer
    to create the password file. Duo/MFA can still be required by WRDS.
    Never put passwords in this module or a tracked project file.
    """
    username = username or os.environ.get("WRDS_USERNAME")
    if not username:
        raise ValueError("Set WRDS_USERNAME or pass your WRDS username explicitly.")
    return wrds.Connection(wrds_username=username)


def query_crsp_stock_data(
    db, permnos=None, start_date=None, end_date=None,
    include_daily_amount=False, fields=None, frequency="monthly",
):
    """Return native daily/monthly data, delistings, and optional trading proxies.

    Dates are inclusive ISO strings; monthly requests require whole months.
    permnos=None queries all securities, without a current-listing filter.
    An empty list is an error, never an implicit all-stock query. fields selects
    stock columns; permno and date are always included. This returns one date
    window in memory; the entry point batches long samples by calendar year.

    Raw msf fields: ret/retx are decimal returns with/without dividends; signed
    prc is dollars (negative means a bid/ask average); shrout is thousands of
    shares; vol is legacy monthly volume in hundreds of shares. cfacpr/cfacshr
    are cumulative adjustment factors. Missing values remain missing.

    The optional amount is SUM(daily closing price * daily share volume),
    NOT transaction-level dollar turnover. Negative daily prc observations are
    quote-based and excluded. Coverage counts accompany partial-month sums;
    no daily observations (including early historical gaps) are not filled.
    Daily volume is already in shares. Daily proxies are not aggregated.
    """
    validate_crsp_frequency(frequency)
    stock_table = "msf" if frequency == "monthly" else "dsf"
    delist_table = "msedelist" if frequency == "monthly" else "dsedelist"
    allowed_fields = CRSP_MONTHLY_FIELDS if frequency == "monthly" else CRSP_DAILY_FIELDS
    fields = list(allowed_fields) if fields is None else list(fields)
    unknown = set(fields) - allowed_fields.keys()
    if unknown:
        raise ValueError(f"Unsupported {frequency} fields: {sorted(unknown)}")
    fields = list(dict.fromkeys(["permno", "date", *fields]))
    if permnos is not None:
        permnos = list(permnos)
        if not permnos or any(
            isinstance(p, bool) or not isinstance(p, Integral) or p <= 0 for p in permnos
        ):
            raise ValueError("permnos must be None or a nonempty list of positive integers.")
    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date are required.")
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date.")
    if frequency == "monthly" and (start.day != 1 or not pd.Timestamp(end).is_month_end):
        raise ValueError("Use an ordered range from a month's first to last day.")
    params = {
        "start_date": start,
        "end_date": end,
    }
    universe_filter = ""
    if permnos is not None:
        params["permnos"] = tuple(sorted({int(p) for p in permnos}))
        universe_filter = "AND permno IN %(permnos)s"
    stock = db.raw_sql(
        f"""
        SELECT {', '.join(fields)}
        FROM crsp.{stock_table}
        WHERE date BETWEEN %(start_date)s AND %(end_date)s
          {universe_filter}
        ORDER BY permno, date
        """,
        params=params, date_cols=["date"],
    )
    stock["date"] = pd.to_datetime(stock["date"])
    periods = stock.assign(period=stock["date"].dt.to_period("M" if frequency == "monthly" else "D"))
    if periods.duplicated(["permno", "period"]).any():
        raise ValueError(f"Unexpected duplicate PERMNO-period observations in {frequency} CRSP data.")

    # Keep delistings separate: retain events even if the monthly stock row is
    # missing. Combining ret and dlret is a subsequent, explicit research choice.
    delistings = db.raw_sql(
        f"""
        SELECT permno, dlstdt, dlstcd, dlret, dlretx, dlprc
        FROM crsp.{delist_table}
        WHERE dlstdt BETWEEN %(start_date)s AND %(end_date)s
          {universe_filter}
        ORDER BY permno, dlstdt
        """,
        params=params, date_cols=["dlstdt"],
    )
    result = {f"{frequency}_raw": stock, "delistings_raw": delistings}
    if include_daily_amount and frequency == "daily":
        result["daily_trading_proxy"] = db.raw_sql(
            f"""
            SELECT permno, date,
                   CASE WHEN vol >= 0 THEN vol END AS volume_shares,
                   CASE WHEN prc > 0 AND vol >= 0
                        THEN prc::double precision * vol::double precision
                   END AS dollar_volume_close_proxy
            FROM crsp.dsf
            WHERE date BETWEEN %(start_date)s AND %(end_date)s
              {universe_filter}
            ORDER BY permno, date
            """,
            params=params, date_cols=["date"],
        )
    elif include_daily_amount:
        # Aggregate on WRDS rather than downloading the daily stock panel.
        result["monthly_trading_proxy"] = db.raw_sql(
            f"""
            SELECT permno, date_trunc('month', date)::date AS month,
                   count(*) AS daily_rows,
                   count(*) FILTER (WHERE vol >= 0) AS volume_days,
                   sum(CASE WHEN vol >= 0 THEN vol END) AS daily_volume_sum_shares,
                   count(*) FILTER (WHERE prc > 0 AND vol >= 0) AS amount_days,
                   count(*) FILTER (WHERE prc < 0) AS quote_price_days,
                   sum(CASE WHEN prc > 0 AND vol >= 0
                            THEN prc::double precision * vol::double precision
                       END) AS dollar_volume_close_proxy
            FROM crsp.dsf
            WHERE date BETWEEN %(start_date)s AND %(end_date)s
              {universe_filter}
            GROUP BY permno, date_trunc('month', date)
            ORDER BY permno, month
            """,
            params=params, date_cols=["month"],
        )
    return result


def add_crsp_price_fields(stock, frequency="monthly"):
    """Derive inspection fields without modifying the raw query result.

    abs(prc)/cfacpr adjusts for splits and related capital changes, not cash
    dividend reinvestment. Use CRSP ret for total returns; never adjust ret again.
    Cumulative factors use the database's reference basis, which can reflect
    later corporate actions: adjusted levels are not point-in-time predictors.
    Monthly volume is not split-adjusted here because intra-month splits make
    an end-of-month factor insufficient for exact monthly volume adjustment.
    """
    validate_crsp_frequency(frequency)
    data = stock.copy()
    data["month"] = data["date"].dt.to_period("M").astype(str)
    # Only derive fields whose inputs were selected; never silently query extras.
    if "prc" in data:
        data["price_is_quote"] = data["prc"].lt(0).where(data["prc"].notna()).astype("boolean")
        data["price_abs"] = data["prc"].abs()
        if "cfacpr" in data:
            data["price_adjusted"] = data["price_abs"] / data["cfacpr"].where(data["cfacpr"] > 0)
    if "shrout" in data:
        data["shares_outstanding"] = data["shrout"].where(data["shrout"] >= 0) * 1000
        if "cfacshr" in data:
            data["shares_adjusted"] = data["shares_outstanding"] * data["cfacshr"].where(data["cfacshr"] > 0)
        if "prc" in data:
            data["market_cap_usd"] = data["price_abs"] * data["shares_outstanding"]
    if "vol" in data:
        data["volume_shares"] = data["vol"].where(data["vol"] >= 0) * (100 if frequency == "monthly" else 1)
    return data


def complete_crsp_months(monthly, permnos, start_date, end_date):
    """Pad a date window with NA; preserve every observed value, even at delisting.

    `month` is the grid key. `date` remains the raw observation date (NaT on
    synthetic rows). has_crsp_row distinguishes no row from an observed missing
    return. Missingness alone does not establish listing/delisting status.
    Call on one year at a time to limit memory for the all-stock universe.
    """
    data = monthly.copy()
    data["month"] = data["date"].dt.to_period("M").astype(str)
    if data.duplicated(["permno", "month"]).any():
        raise ValueError("Cannot pad duplicate PERMNO-month observations.")
    universe = sorted(set(permnos))
    months = pd.period_range(start_date, end_date, freq="M").astype(str)
    if not set(data["permno"]).issubset(universe) or not data["month"].isin(months).all():
        raise ValueError("Padding universe/date range would discard observed rows.")
    data["has_crsp_row"] = True
    index = pd.MultiIndex.from_product([universe, months], names=["permno", "month"])
    data = data.set_index(["permno", "month"]).reindex(index).reset_index()
    data["has_crsp_row"] = data["has_crsp_row"].eq(True)
    return data
