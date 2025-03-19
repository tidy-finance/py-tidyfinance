"""Internal utility functions for tidyfinance."""

import pandas as pd


def _assign_exchange(primaryexch):
    """
    Assign exchange for CRSP data.

    Parameters
    ----------
        primaryexch (str): A string of exchange letter.

    Returns
    -------
        str: Exchange name.
    """
    if primaryexch == "N":
        return "NYSE"
    elif primaryexch == "A":
        return "AMEX"
    elif primaryexch == "Q":
        return "NASDAQ"
    else:
        return "Other"


def _assign_industry(siccd):
    """
    Assign industry for CRSP data.

    Parameters
    ----------
        siccd (int): An integer to present the siccd.

    Returns
    -------
        str: Industry name.
    """
    if 1 <= siccd <= 999:
        return "Agriculture"
    elif 1000 <= siccd <= 1499:
        return "Mining"
    elif 1500 <= siccd <= 1799:
        return "Construction"
    elif 2000 <= siccd <= 3999:
        return "Manufacturing"
    elif 4000 <= siccd <= 4899:
        return "Transportation"
    elif 4900 <= siccd <= 4999:
        return "Utilities"
    elif 5000 <= siccd <= 5199:
        return "Wholesale"
    elif 5200 <= siccd <= 5999:
        return "Retail"
    elif 6000 <= siccd <= 6799:
        return "Finance"
    elif 7000 <= siccd <= 8999:
        return "Services"
    elif 9000 <= siccd <= 9999:
        return "Public"
    else:
        return "Missing"


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
    if start_date is None or end_date is None:
        if use_default_range:
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(years=2)
            print(
                "No start_date or end_date provided. Using the range "
                f"{start_date.date()} to {end_date.date()} to avoid "
                "downloading large amounts of data."
            )
            return start_date.date(), end_date.date()
        else:
            print(
                "No start_date or end_date provided. Returning the full dataset."
            )
            return None, None
    else:
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")
        return start_date, end_date


def _return_datetime(dates):
    """Return date without time and change period to timestamp."""
    dates = pd.Series(dates)
    if isinstance(dates.iloc[0], pd.Period):  # Check if 'Date' is a Period
        dates = dates.dt.to_timestamp(how="start").dt.date
    dates = pd.to_datetime(dates, errors="coerce")


def _transfrom_to_snake_case(column_name):
    """
    Convert a string to snake_case.

    - Converts uppercase letters to lowercase.
    - Replaces spaces and special characters with underscores.
    - Ensures no multiple underscores.
    """
    column_name = column_name.replace(" ", "_").replace("-", "_").lower()
    column_name = "".join(
        c if c.isalnum() or c == "_" else "_" for c in column_name
    )

    # Remove multiple underscores
    while "__" in column_name:
        column_name = column_name.replace("__", "_")

    # Remove leading/trailing underscores
    return column_name.strip("_")
