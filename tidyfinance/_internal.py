"""Internal utility functions for tidyfinance."""

import numpy as np
import pandas as pd
import re


def _parse_date(d: str, is_end: bool = False) -> pd.Timestamp:
    """
    Parse a date-like string into a normalized 'pd.Timestamp'.

    Parameters
    ----------
    d : str or None
        Date string in one of two supported formats: 'YYYY-MM-DD'
        (any pandas-parseable date) or 'YYYYMM' (year-month, six
        digits). 'None' returns 'None'.
    is_end : bool, default False
        Only relevant for the 'YYYYMM' form. When 'True', shift the
        parsed timestamp to the last day of that month; when 'False',
        the timestamp is the first day of the month.

    Returns
    -------
    pd.Timestamp or None
        Normalized timestamp at midnight (time component stripped),
        or 'None' if 'd' was 'None'.
    """
    if d is None:
        return None
    d = str(d)
    if len(d) == 6 and d.isdigit():  # YYYYMM
        ts = pd.to_datetime(d, format="%Y%m")
        if is_end:
            # Move to last day of the month
            ts = ts + pd.offsets.MonthEnd(0)
        return ts.normalize()
    return pd.to_datetime(d).normalize()


def _validate_dates(
    start_date: str = None,
    end_date: str = None,
    use_default_range: bool = False,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Validate and process start and end dates.

    Parameters
    ----------
    start_date : str, optional
        The start date in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date in "YYYY-MM-DD" format.
    use_default_range : bool, optional
        Whether to use a default date range if no dates are provided.

    Returns
    -------
    tuple
        A tuple containing the validated start and end dates.
    """
    if start_date is None and end_date is None:
        if use_default_range:
            today = pd.Timestamp.today().normalize()
            start_date = today - pd.DateOffset(years=2)
            end_date = today - pd.DateOffset(years=1)
            print(
                "No start_date or end_date provided. Using the range "
                f"{start_date.date()} to {end_date.date()} to avoid "
                "downloading large amounts of data."
            )
            return start_date.date(), end_date.date()
        else:
            print("No start_date or end_date provided. " "Returning the full dataset.")
            return None, None

    start_date = _parse_date(start_date, is_end=False) if start_date else None
    end_date = _parse_date(end_date, is_end=True) if end_date else None

    # If only one date is provided, fill the other sensibly
    if start_date is None:
        # If only end_date is given, leave start open
        return None, end_date
    if end_date is None:
        # If only start_date is given, leave end open
        return start_date, None

    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")
    return start_date, end_date


def _return_datetime(dates):
    """
    Coerce a date-like series to 'datetime64[ns]' via string round-trip.

    Parameters
    ----------
    dates : pd.Series or pd.Index
        Series whose entries can be cast to string and parsed by
        'pd.to_datetime' (e.g., 'PeriodIndex', date objects,
        date strings).

    Returns
    -------
    pd.DatetimeIndex
        Parsed timestamps with no time-of-day component.
    """
    return pd.to_datetime(dates.astype(str))


def _transfrom_to_snake_case(column_name):
    """
    Convert a string to snake_case.

    Inserts underscores before CamelCase boundaries, lowercases all
    letters, replaces spaces and special characters with underscores,
    and collapses runs of underscores into one.

    Parameters
    ----------
    column_name : str
        Arbitrary identifier-like string.

    Returns
    -------
    str
        Snake_case version of 'column_name'.
    """
    column_name = re.sub(r"(?<!^)(?=[A-Z])", "_", column_name)
    column_name = column_name.replace(" ", "_").replace("-", "_").lower()
    column_name = "".join(c if c.isalnum() or c == "_" else "_" for c in column_name)

    while "__" in column_name:
        column_name = column_name.replace("__", "_")

    return column_name.strip("_")


def _get_random_user_agent():
    """
    Retrieve a random User-Agent string.

    Returns
    -------
    str
        One entry sampled uniformly at random from a fixed list of
        modern browser User-Agent strings. Used to vary headers in
        requests that would otherwise be rejected by some endpoints.
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:116.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.141 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_7_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36 Edg/116.0.1938.69",
    ]
    return str(np.random.choice(user_agents))


def _to_offset(x):
    """
    Normalize a lag specification to a pandas time offset.

    Parameters
    ----------
    x : int, pd.Timedelta, or pd.DateOffset
        Lag value. Integers are interpreted as a number of days.
        'pd.Timedelta' and 'pd.DateOffset' instances are returned
        unchanged.

    Returns
    -------
    pd.Timedelta or pd.DateOffset
        An offset object suitable for date arithmetic.

    Raises
    ------
    TypeError
        If 'x' is not one of the supported types (booleans are
        rejected even though they subclass 'int').
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return pd.Timedelta(days=x)
    if isinstance(x, (pd.Timedelta, pd.tseries.offsets.BaseOffset)):
        return x
    raise TypeError(
        f"lag/max_lag must be int, pd.Timedelta, or pd.DateOffset; "
        f"got {type(x).__name__}."
    )


def _check_new_col(data: pd.DataFrame, names) -> None:
    """
    Guard against overwriting user columns with internal helpers.

    Raises 'ValueError' if any of 'names' already exist in
    'data.columns'. Used before adding temporary columns like
    '_upper', '_lower', '_src_date' inside lagging functions, so
    user data is never silently clobbered.

    Parameters
    ----------
    data : pd.DataFrame
        Input frame whose columns are checked.
    names : str or iterable of str
        Column name(s) the caller intends to introduce.

    Raises
    ------
    ValueError
        If any name in 'names' is already a column of 'data'.
    """
    if isinstance(names, str):
        names = [names]
    existing = [n for n in names if n in data.columns]
    if existing:
        raise ValueError(
            f"Cannot proceed: column(s) {existing} would be created by "
            "this operation but already exist in the input. Rename or "
            "drop them first."
        )


def _validate_column_name(value, arg: str, description: str) -> None:
    """
    Validate that 'value' is a single string usable as a column name.

    Parameters
    ----------
    value : Any
        Value to check.
    arg : str
        Name of the argument being validated, used in the error
        message.
    description : str
        Short description of what the column represents, used in
        the error message (e.g., 'date', 'sorting').

    Raises
    ------
    ValueError
        If 'value' is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"'{arg}' must be a string indicating the column name "
            f"for the {description} variable."
        )


def _validate_flag(value, arg: str, message: str | None = None) -> None:
    """
    Validate that 'value' is a Python bool.

    Parameters
    ----------
    value : Any
        Value to check. Numeric 0 or 1 are rejected; only actual
        'bool' instances pass.
    arg : str
        Name of the argument being validated, used in the default
        error message when 'message' is not supplied.
    message : str, optional
        Custom error message. If 'None' (the default), a generic
        "'<arg>' must be a single boolean." message is raised.

    Raises
    ------
    ValueError
        If 'value' is not a 'bool'.
    """
    if not isinstance(value, bool):
        if message is None:
            message = f"'{arg}' must be a single boolean."
        raise ValueError(message)


def _validate_optional_number(
    value,
    message: str,
    min: float = float("-inf"),
    max: float = float("inf"),
    min_strict: bool = False,
    max_strict: bool = False,
) -> None:
    """
    Validate that 'value' is None or a finite number within bounds.

    Accepts 'None' as a way to signal "not supplied". When 'value' is
    a number, it must be an 'int' or 'float' (booleans are rejected),
    must not be 'NaN', and must lie within the requested range.

    Parameters
    ----------
    value : None, int, or float
        Value to check.
    message : str
        Error message raised on any validation failure.
    min : float, default '-inf'
        Lower bound for 'value'.
    max : float, default '+inf'
        Upper bound for 'value'.
    min_strict : bool, default False
        If 'True', require 'value > min' rather than 'value >= min'.
    max_strict : bool, default False
        If 'True', require 'value < max' rather than 'value <= max'.

    Raises
    ------
    ValueError
        If 'value' is not 'None', not numeric, is 'NaN', or lies
        outside the requested bounds.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    if value != value:  # NaN check
        raise ValueError(message)
    lower_ok = (value > min) if min_strict else (value >= min)
    upper_ok = (value < max) if max_strict else (value <= max)
    if not (lower_ok and upper_ok):
        raise ValueError(message)
