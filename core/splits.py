"""Annual GKX windows indexed by the month of the realized target return."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.model_inputs import positive_integer


@dataclass(frozen=True)
class AnnualSplit:
    test_year: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def annual_splits(
    return_month, *, first_test_year=1987, last_test_year=2016,
    first_train_year=1957, validation_years=12,
):
    """Yield positional indices, preserving input row order and whole-year windows.

    Default first split: train 1957-1974, validation 1975-1986, test 1987.
    Training expands; validation rolls; test is one year. Dates label y, so
    December-t features predicting January-(t+1) returns belong to year t+1.
    Callers construct correctly lagged X before using this function. This
    helper does not verify release dates or annual/overlapping target horizons.
    """
    for name, value in (("first_test_year", first_test_year), ("last_test_year", last_test_year),
                        ("first_train_year", first_train_year), ("validation_years", validation_years)):
        positive_integer(value, name)
    if last_test_year < first_test_year or first_train_year >= first_test_year - validation_years:
        raise ValueError("Year settings must leave nonempty training, validation, and test windows.")
    raw = np.asarray(return_month)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("return_month must be a nonempty one-dimensional sequence.")
    if np.issubdtype(raw.dtype, np.number):
        raise ValueError("Use monthly periods or dates, not integer YYYYMM/nanosecond labels.")
    if isinstance(return_month, pd.PeriodIndex) or isinstance(getattr(return_month, "dtype", None), pd.PeriodDtype):
        months = pd.PeriodIndex(return_month)
        if months.freqstr != "M":
            raise ValueError("Period dates must have monthly frequency.")
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(return_month, errors="raise"))
        if dates.tz is not None:
            raise ValueError("Use timezone-naive financial month labels.")
        months = dates.to_period("M")
    if months.isna().any():
        raise ValueError("return_month must not contain missing dates.")
    years = months.year.to_numpy()
    for year in range(first_test_year, last_test_year + 1):
        validation_start = year - validation_years
        train = np.flatnonzero((years >= first_train_year) & (years < validation_start))
        validation = np.flatnonzero((years >= validation_start) & (years < year))
        test = np.flatnonzero(years == year)
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"The {year} split has an empty train, validation, or test sample.")
        yield AnnualSplit(year, train, validation, test)
