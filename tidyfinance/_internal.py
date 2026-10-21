"""Internal utility functions for tidyfinance."""

import calendar
import datetime as dt
import os
import re
import sys
import warnings

import numpy as np
import polars as pl

_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


def _user_stacklevel() -> int:
    """Return the 'stacklevel' that attributes a warning to the user.

    Meant to be passed to 'warnings.warn' by the function that calls
    this helper. The warning is attributed to the first caller outside
    the tidyfinance package, however deep inside the package it is
    raised (e.g. through 'download_data' and the backend wrapper).
    Python shows 'DeprecationWarning' by default only when it is
    attributed to code in '__main__', so attributing it to a package
    frame would hide it.

    Returns
    -------
    int
        The 'stacklevel' for 'warnings.warn' in the calling function.
    """
    # Frame 0 is this helper, frame 1 the function that calls
    # warnings.warn (stacklevel 1), frame 2 its caller (stacklevel 2).
    frame = sys._getframe(2)
    stacklevel = 2
    while frame is not None:
        filename = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if not filename.startswith(_PACKAGE_DIR + os.sep):
            break
        frame = frame.f_back
        stacklevel += 1
    return stacklevel


def _deprecate_legacy_dataset(legacy: str, dataset: str) -> None:
    """Warn that a legacy identifier was passed as 'dataset'.

    Legacy identifiers such as 'factors_ff_3_monthly' are still accepted
    as values of 'dataset'. The warning names the value and the dataset
    that replaces it, rather than the deprecated 'type' argument, which
    the caller did not use.

    Parameters
    ----------
    legacy : str
        The legacy identifier passed as 'dataset'.
    dataset : str
        The dataset name that replaces it.

    Warns
    -----
    DeprecationWarning
        Always.
    """
    # Attribute the warning to the user's call, not to package code,
    # so Python does not hide it.
    warnings.warn(
        f"dataset={legacy!r} is deprecated. Use dataset={dataset!r} instead.",
        DeprecationWarning,
        stacklevel=_user_stacklevel(),
    )


def _shift_years(d: dt.date, years: int) -> dt.date:
    """Shift a date by whole years, clamping Feb 29 to Feb 28."""
    year = d.year + years
    day = min(d.day, calendar.monthrange(year, d.month)[1])
    return dt.date(year, d.month, day)


def _parse_date(d, is_end: bool = False) -> dt.date | None:
    """
    Parse a date-like input into a 'datetime.date'.

    Parameters
    ----------
    d : str, datetime.date, or None
        Date in one of two supported string formats: 'YYYY-MM-DD' (an
        ISO date) or 'YYYYMM' (year-month, six digits). Date and
        datetime objects are accepted directly. 'None' returns 'None'.
    is_end : bool, default False
        Only relevant for the 'YYYYMM' form. When 'True', shift the
        parsed date to the last day of that month; when 'False', the
        date is the first day of the month.

    Returns
    -------
    datetime.date or None
        Parsed calendar date, or 'None' if 'd' was 'None'.
    """
    if d is None:
        return None
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    d = str(d)
    if len(d) == 6 and d.isdigit():  # YYYYMM
        year, month = int(d[:4]), int(d[4:])
        if is_end:
            day = calendar.monthrange(year, month)[1]
        else:
            day = 1
        return dt.date(year, month, day)
    return dt.date.fromisoformat(d[:10])


def _validate_dates(
    start_date=None,
    end_date=None,
    use_default_range: bool = False,
) -> tuple[dt.date | None, dt.date | None]:
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
        A tuple containing the validated start and end dates as
        'datetime.date' objects (or 'None').
    """
    if start_date is None and end_date is None:
        if use_default_range:
            today = dt.date.today()
            start_date = _shift_years(today, -2)
            end_date = _shift_years(today, -1)
            print(
                "No start_date or end_date provided. Using the range "
                f"{start_date} to {end_date} to avoid "
                "downloading large amounts of data."
            )
            return start_date, end_date
        else:
            print(
                "No start_date or end_date provided. "
                "Returning the full dataset."
            )
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
    column_name = "".join(
        c if c.isalnum() or c == "_" else "_" for c in column_name
    )

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


# Units accepted in polars duration strings (subset relevant for lags).
_POLARS_OFFSET_RE = re.compile(r"^(\d+(d|w|mo|q|y))+$")


def _to_offset(x) -> str:
    """
    Normalize a lag specification to a polars offset string.

    Parameters
    ----------
    x : int, str, datetime.timedelta, pd.Timedelta, or pd.DateOffset
        Lag value. Integers are interpreted as a number of days.
        Strings must be positive polars duration strings composed of
        'd', 'w', 'mo', 'q', or 'y' units (e.g. '1mo', '3d', '1y2mo').
        'datetime.timedelta' / 'pd.Timedelta' values are converted to
        days (they must represent whole days), and calendar-based
        'pd.DateOffset' objects (with 'years', 'months', 'weeks', or
        'days' components) are translated to the equivalent duration
        string.

    Returns
    -------
    str
        A polars offset string suitable for
        'polars.Expr.dt.offset_by'.

    Raises
    ------
    TypeError
        If 'x' is not one of the supported types (booleans are
        rejected even though they subclass 'int').
    ValueError
        If a string is not a valid positive polars duration, a
        timedelta does not represent whole days, or a 'pd.DateOffset'
        uses unsupported components.
    """
    if isinstance(x, bool):
        raise TypeError(
            "lag/max_lag must be int, str, timedelta, or DateOffset; "
            "got bool."
        )
    if isinstance(x, int):
        if x < 0:
            raise ValueError("lag/max_lag must be non-negative.")
        return f"{x}d"
    if isinstance(x, str):
        if not _POLARS_OFFSET_RE.match(x):
            raise ValueError(
                f"Invalid offset string {x!r}. Use positive polars "
                "durations composed of 'd', 'w', 'mo', 'q', or 'y', "
                "e.g. '1mo' or '1y2mo'."
            )
        return x
    if isinstance(x, dt.timedelta):
        # Covers pd.Timedelta, which subclasses datetime.timedelta.
        if x.seconds or x.microseconds:
            raise ValueError(
                "Timedelta lags must represent whole days for "
                "calendar-date data."
            )
        if x.days < 0:
            raise ValueError("lag/max_lag must be non-negative.")
        return f"{x.days}d"

    # Calendar offsets (pd.DateOffset) without importing pandas up
    # front: detect by the characteristic 'kwds'/'n' attributes.
    if type(x).__name__ == "DateOffset" and hasattr(x, "kwds"):
        allowed = ("years", "months", "weeks", "days")
        unsupported = [k for k in x.kwds if k not in allowed]
        if unsupported:
            raise ValueError(
                "Unsupported DateOffset components for lags: "
                f"{unsupported}. Use years, months, weeks, or days."
            )
        n = getattr(x, "n", 1)
        if not x.kwds:
            # pd.DateOffset(k) without keyword components means k
            # calendar days in pandas.
            if n < 0:
                raise ValueError("lag/max_lag must be non-negative.")
            return f"{n}d"
        unit_map = {"years": "y", "months": "mo", "weeks": "w", "days": "d"}
        parts = []
        for k in allowed:
            value = x.kwds.get(k, 0) * n
            if value < 0:
                raise ValueError("lag/max_lag must be non-negative.")
            if value:
                parts.append(f"{value}{unit_map[k]}")
        return "".join(parts) if parts else "0d"

    raise TypeError(
        f"lag/max_lag must be int, str, timedelta, or DateOffset; "
        f"got {type(x).__name__}."
    )


def _negate_offset(offset: str) -> str:
    """Negate a positive polars offset string component-wise.

    For example '1y2mo' becomes '-1y-2mo', which shifts dates
    backwards when passed to 'polars.Expr.dt.offset_by'.
    """
    return re.sub(r"(\d+[a-z]+)", r"-\1", offset)


def _offset_end(start: dt.date, offset: str) -> dt.date:
    """Apply a polars offset string to a single date."""
    return pl.Series([start]).dt.offset_by(offset).item()


def _check_new_col(data: pl.DataFrame, names) -> None:
    """
    Guard against overwriting user columns with internal helpers.

    Raises 'ValueError' if any of 'names' already exist in
    'data.columns'. Used before adding temporary columns like
    '_upper', '_lower', '_src_date' inside lagging functions, so
    user data is never silently clobbered.

    Parameters
    ----------
    data : pl.DataFrame
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
