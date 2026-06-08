"""Main module for tidyfinance package."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS

from ._internal import (
    _to_offset,
    _check_new_col,
    _validate_column_name,
)


def add_lagged_columns(
    data: pd.DataFrame,
    cols: list[str] | str,
    lag,
    max_lag=None,
    by: list[str] | str | None = None,
    drop_na: bool = False,
    ff_adjustment: bool = False,
    date_col: str = "date",
    data_options: dict | None = None,
) -> pd.DataFrame:
    """Append lagged versions of specified columns to a DataFrame using
    a join-based approach.

    When lag == max_lag (the default), an equi-join is used: source
    dates are shifted forward by lag and matched exactly. When
    lag < max_lag, a window join is used: for each row, the most
    recent source value within [date - max_lag, date - lag] is
    selected.

    The combination of by and the date column must be unique in
    data. If by is None, dates alone must be unique.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to lag.
    cols : list of str or str
        Names of the columns to lag. Each column produces a new column
        suffixed with _lag.
    lag : int, pd.Timedelta, or pd.DateOffset
        Minimum lag (inclusive) to apply. An int is interpreted as
        days.
    max_lag : int, pd.Timedelta, or pd.DateOffset, optional
        Maximum lag (inclusive). Defaults to lag (exact lag).
    by : list of str or str, optional
        Grouping columns (e.g. a stock identifier). Lagged values are
        matched within groups. Defaults to None.
    drop_na : bool, optional
        If True, NaN values in the source columns are excluded
        before matching, so the lookup skips over missing observations.
        Applied independently per column. Defaults to False.
    ff_adjustment : bool, optional
        If True, only the last observation per year (within each
        group defined by by) is retained as a source for lagged
        values, following Fama-French conventions for annual
        accounting data. Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to "date".

    Returns
    -------
    pd.DataFrame
        DataFrame with the same rows as data and new columns
        appended, each suffixed with _lag. Unmatched rows receive
        NaN in the lagged columns.
    """
    if data_options is not None:
        date_col = data_options.get("date", date_col)

    if isinstance(cols, str):
        cols = [cols]
    if isinstance(by, str):
        by = [by]
    by_list = by or []

    lag_offset = _to_offset(lag)
    max_lag_offset = _to_offset(max_lag if max_lag is not None else lag)

    if date_col not in data.columns:
        raise ValueError(
            f"`data` must contain the date column `{date_col}`."
        )

    ref = pd.Timestamp("2020-01-01")
    lag_end = ref + lag_offset
    max_lag_end = ref + max_lag_offset
    if lag_end < ref or max_lag_end < lag_end:
        raise ValueError(
            "`lag` and `max_lag` must be non-negative and `max_lag` "
            "must be >= `lag`."
        )

    missing_cols = [c for c in cols if c not in data.columns]
    if missing_cols:
        raise ValueError(
            f"`data` is missing column(s): {missing_cols}."
        )

    if by_list:
        missing_by = [c for c in by_list if c not in data.columns]
        if missing_by:
            raise ValueError(
                f"`data` is missing grouping column(s): {missing_by}."
            )

    join_cols = by_list + [date_col]
    if data[join_cols].duplicated().any():
        raise ValueError(
            "The combination of `by` and date columns must be unique "
            "in `data`."
        )

    exact_lag = (lag_end == max_lag_end)
    result = data.copy()

    if not exact_lag:
        _check_new_col(result, ["_upper", "_lower"])
        result["_upper"] = result[date_col] - lag_offset
        result["_lower"] = result[date_col] - max_lag_offset

    for col in cols:
        lag_col_name = f"{col}_lag"
        if lag_col_name in result.columns:
            raise ValueError(
                f"Column `{lag_col_name}` already exists in `data`."
            )

        lagged = data[join_cols + [col]].copy()

        if drop_na:
            lagged = lagged.dropna(subset=[col])

        if ff_adjustment:
            grp_cols = by_list + ["_yr"]
            lagged = lagged.assign(_yr=lagged[date_col].dt.year)
            max_dates = lagged.groupby(grp_cols)[date_col].transform("max")
            lagged = lagged[lagged[date_col] == max_dates].drop(
                columns="_yr"
            )

        if exact_lag:
            lagged[date_col] = lagged[date_col] + lag_offset
            lagged = lagged.rename(columns={col: lag_col_name})
            result = result.merge(lagged, on=join_cols, how="left")
        else:
            result = _window_lag_join(
                result, lagged, by_list, date_col, col, lag_col_name
            )

    if not exact_lag:
        result = result.drop(columns=["_upper", "_lower"])

    return result


def _window_lag_join(
    result: pd.DataFrame,
    lagged: pd.DataFrame,
    by_list: list[str],
    date_col: str,
    col: str,
    lag_col_name: str,
) -> pd.DataFrame:
    """Backward window join used by add_lagged_columns for non-exact lags.

    For each row in result (which already carries _upper and _lower
    bounds from the caller), finds the most recent row in lagged whose
    date falls within the window [_lower, _upper] and copies its col
    value into a new column named lag_col_name. The match is performed
    by group when by_list is non-empty.

    Internally uses pd.merge_asof with direction="backward" on _upper
    to locate the closest source date at or before the upper bound,
    then filters out rows whose source date falls below the lower
    bound. The original row order of result is preserved.

    Parameters
    ----------
    result : pd.DataFrame
        Target frame. Must contain the columns in by_list plus
        _upper and _lower (window bounds). Must not contain a
        column named _orig_idx.
    lagged : pd.DataFrame
        Source frame. Must contain the columns in by_list plus
        date_col and col. Must not contain a column named _src_date.
    by_list : list of str
        Grouping columns shared by both frames. Pass an empty list
        for an ungrouped join.
    date_col : str
        Name of the date column in lagged.
    col : str
        Name of the source value column in lagged to copy.
    lag_col_name : str
        Name of the new column to add to result with the matched
        source values. Unmatched rows receive NaN.

    Returns
    -------
    pd.DataFrame
        result with lag_col_name appended, in the original row order.
        The helper columns _orig_idx and _src_date are removed before
        return; the caller is responsible for dropping _upper and
        _lower.
    """
    _check_new_col(result, "_orig_idx")
    _check_new_col(lagged, "_src_date")
    result = result.assign(_orig_idx=np.arange(len(result)))
    lagged = lagged.rename(
        columns={date_col: "_src_date", col: lag_col_name}
    )

    left_sorted = result.sort_values("_upper", kind="mergesort")
    right_sorted = lagged.sort_values("_src_date", kind="mergesort")

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="_upper",
        right_on="_src_date",
        by=by_list if by_list else None,
        direction="backward",
    )

    mask = merged["_src_date"].notna() & (
        merged["_src_date"] >= merged["_lower"]
    )
    merged.loc[~mask, lag_col_name] = np.nan

    merged = merged.sort_values("_orig_idx", kind="mergesort")
    merged = merged.drop(
        columns=["_orig_idx", "_src_date"]
    ).reset_index(drop=True)
    return merged


def join_lagged_values(
    original_data: pd.DataFrame,
    new_data: pd.DataFrame,
    id_keys: list[str] | str,
    min_lag,
    max_lag,
    ff_adjustment: bool = False,
    date_col: str = "date",
    data_options: dict | None = None,
) -> pd.DataFrame:
    """Join lagged values from new_data into original_data over
    a date range.

    Unlike :func:`add_lagged_columns`, this supports joining across
    DataFrames with different date grids (e.g. monthly source into
    quarterly target). All columns in new_data besides id_keys
    and the date column are lagged and joined under their original
    names.

    Parameters
    ----------
    original_data : pd.DataFrame
        Target panel data.
    new_data : pd.DataFrame
        Source variables to lag and merge.
    id_keys : list of str or str
        Identifier column(s) shared by both frames.
    min_lag, max_lag : int, pd.Timedelta, or pd.DateOffset
        Inclusive lag bounds.
    ff_adjustment : bool, optional
        If True, keeps only the last observation per identifier and
        year in new_data before lagging. Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to "date".

    Returns
    -------
    pd.DataFrame
        original_data with new columns from new_data appended.
    """
    if data_options is not None:
        date_col = data_options.get("date", date_col)

    if isinstance(id_keys, str):
        id_keys = [id_keys]
    if not isinstance(id_keys, list) or not all(
        isinstance(k, str) for k in id_keys
    ):
        raise ValueError("`id_keys` must be a string or list of strings.")

    min_lag_offset = _to_offset(min_lag)
    max_lag_offset = _to_offset(max_lag)

    if date_col not in original_data.columns:
        raise ValueError(
            f"`original_data` must contain the column `{date_col}`."
        )
    if date_col not in new_data.columns:
        raise ValueError(
            f"`new_data` must contain the column `{date_col}`."
        )

    missing_original = [
        k for k in id_keys if k not in original_data.columns
    ]
    if missing_original:
        raise ValueError(
            f"`original_data` is missing id column(s): "
            f"{missing_original}."
        )

    missing_new = [k for k in id_keys if k not in new_data.columns]
    if missing_new:
        raise ValueError(
            f"`new_data` is missing id column(s): {missing_new}."
        )

    new_column_names = [
        c for c in new_data.columns if c not in id_keys + [date_col]
    ]
    if not new_column_names:
        raise ValueError(
            f"`new_data` must contain columns besides {id_keys} and "
            f"`{date_col}`."
        )

    original_non_key = [
        c for c in original_data.columns
        if c not in id_keys + [date_col]
    ]
    duplicate_cols = [
        c for c in new_column_names if c in original_non_key
    ]
    if duplicate_cols:
        raise ValueError(
            f"Column(s) in `new_data` already exist in "
            f"`original_data`: {duplicate_cols}. Remove or rename them "
            "before joining."
        )

    new_data = new_data.copy()
    helper_cols = ["_lower", "_upper"]
    if ff_adjustment:
        helper_cols.append("_year")
    _check_new_col(new_data, helper_cols)
    new_data["_lower"] = new_data[date_col] + min_lag_offset
    new_data["_upper"] = new_data[date_col] + max_lag_offset
    if ff_adjustment:
        new_data["_year"] = new_data[date_col].dt.year

    result = original_data.copy()

    for col in new_column_names:
        select_cols = id_keys + [date_col, col, "_lower", "_upper"]
        if ff_adjustment:
            select_cols.append("_year")
        tmp = new_data[select_cols].copy()

        if ff_adjustment:
            grp_cols = id_keys + ["_year"]
            max_dates = tmp.groupby(grp_cols)[date_col].transform("max")
            tmp = tmp[tmp[date_col] == max_dates].drop(
                columns=[date_col, "_year"]
            )
        else:
            tmp = tmp.drop(columns=[date_col])

        _check_new_col(result, "_orig_idx")
        result = result.assign(_orig_idx=np.arange(len(result)))
        sort_keys_left = id_keys + [date_col]
        sort_keys_right = id_keys + ["_lower"]
        left_sorted = result.sort_values(
            sort_keys_left, kind="mergesort"
        )
        right_sorted = tmp.sort_values(
            sort_keys_right, kind="mergesort"
        )

        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on=date_col,
            right_on="_lower",
            by=id_keys if id_keys else None,
            direction="backward",
        )

        mask = merged["_lower"].notna() & (
            merged[date_col] <= merged["_upper"]
        )
        merged.loc[~mask, col] = np.nan

        merged = merged.sort_values("_orig_idx", kind="mergesort")
        merged = merged.drop(
            columns=["_orig_idx", "_lower", "_upper"]
        ).reset_index(drop=True)
        result = merged

    return result


def data_options(
    id: str = "permno",
    date: str = "date",
    exchange: str = "exchange",
    mktcap_lag: str = "mktcap_lag",
    ret_excess: str = "ret_excess",
    portfolio: str = "portfolio",
    siccd: str = "siccd",
    price: str = "prc_adj",
    listing_age: str = "listing_age",
    be: str = "be",
    earnings: str = "ib",
    **kwargs,
) -> dict:
    """Construct a tidyfinance data options dict.

    Maps the Tidy Finance naming conventions to the actual column names.

    Parameters
    ----------
    id : str, default "permno"
        Entity identifier column.
    date : str, default "date"
        Date column.
    exchange : str, default "exchange"
        Exchange column.
    mktcap_lag : str, default "mktcap_lag"
        Market capitalization lag column.
    ret_excess : str, default "ret_excess"
        Excess return column.
    portfolio : str, default "portfolio"
        Portfolio assignment column.
    siccd : str, default "siccd"
        SIC code column.
    price : str, default "prc_adj"
        Adjusted price column.
    listing_age : str, default "listing_age"
        Listing age column.
    be : str, default "be"
        Book equity column.
    earnings : str, default "ib"
        Earnings column (Compustat income before extraordinary items).
    **kwargs
        Any additional column mappings stored verbatim in the dict.

    Returns
    -------
    dict
        Mapping with at least the 11 standard column-name keys plus
        any extras provided via kwargs.
    """
    _validate_column_name(id, "id", "entity")
    _validate_column_name(date, "date", "date")
    _validate_column_name(exchange, "exchange", "exchange")
    _validate_column_name(
        mktcap_lag, "mktcap_lag", "market capitalization lag"
    )
    _validate_column_name(ret_excess, "ret_excess", "excess return")
    _validate_column_name(portfolio, "portfolio", "portfolio")
    _validate_column_name(
        siccd, "siccd", "Standard Industrial Classification code"
    )
    _validate_column_name(price, "price", "(adjusted) price")
    _validate_column_name(listing_age, "listing_age", "listing age")
    _validate_column_name(be, "be", "book equity")
    _validate_column_name(earnings, "earnings", "earnings")

    return {
        "id": id,
        "date": date,
        "exchange": exchange,
        "mktcap_lag": mktcap_lag,
        "ret_excess": ret_excess,
        "portfolio": portfolio,
        "siccd": siccd,
        "price": price,
        "listing_age": listing_age,
        "be": be,
        "earnings": earnings,
        **kwargs,
    }


def assign_portfolio(
    data: pd.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict = None,
    breakpoint_function=None,
) -> pd.Series:
    """
    Assign portfolio labels based on a sorting variable and breakpoints.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame containing the dataset for
        portfolio assignment.
    sorting_variable (str): Column name used for sorting and
        portfolio assignment.
    breakpoint_options (dict, optional): Named arguments passed
        to the breakpoint function.
    breakpoint_function (callable, optional): Function to compute breakpoints.
        Must return an ascending vector of breakpoints.

    Returns
    -------
    pd.Series: A Series of portfolio assignments.
    """
    if sorting_variable not in data.columns:
        raise ValueError(
            f"Sorting variable '{sorting_variable}' not found in data."
        )

    if len(data[sorting_variable].unique()) == 1:
        print(
            "Warning: The sorting variable is constant, assigning all to portfolio 1."
        )
        return pd.Series(1, index=data.index, dtype=int)

    if breakpoint_function is None:
        raise ValueError("A valid breakpoint function must be provided.")

    # Compute breakpoints
    breakpoints = breakpoint_function(
        data, sorting_variable, breakpoint_options
    )

    # Assign portfolios using pd.cut
    assigned_portfolios = pd.cut(
        data[sorting_variable],
        bins=breakpoints,
        labels=range(1, breakpoints.size),
        include_lowest=True,
        right=False,
    )

    return assigned_portfolios.astype(int)


def breakpoint_options(
    n_portfolios: int = None,
    percentiles: list = None,
    breakpoint_exchanges: str = None,
    smooth_bunching: bool = False,
    **kwargs,
) -> dict:
    """
    Create a dictionary of options for breakpoints in portfolio sorting.

    Parameters
    ----------
    n_portfolios (int, optional): Number of portfolios to create.
        Must be a positive integer.
    percentiles (list, optional): List of percentile thresholds
        (values between 0 and 1).
    breakpoint_exchanges (str, optional): Exchange for
        which the breakpoints apply.
    smooth_bunching (bool, optional): Whether smooth bunching
        should be applied. Default is False.
    **kwargs: Additional optional arguments.

    Returns
    -------
    dict: A dictionary containing breakpoint options.
    """
    # Validate n_portfolios
    if n_portfolios is not None:
        if not isinstance(n_portfolios, int) or n_portfolios <= 0:
            raise ValueError("n_portfolios must be a positive integer.")

    # Validate percentiles
    if percentiles is not None:
        if not all(
            isinstance(p, (int, float)) and 0 <= p <= 1 for p in percentiles
        ):
            raise ValueError(
                "percentiles must be a list of values between 0 and 1."
            )

    # Validate breakpoint_exchanges
    if breakpoint_exchanges is not None:
        if not isinstance(breakpoint_exchanges, str) or (
            not breakpoint_exchanges
        ):
            raise ValueError("breakpoint_exchanges must be a non-empty string.")

    # Validate smooth_bunching
    if not isinstance(smooth_bunching, bool):
        raise ValueError("smooth_bunching must be a boolean value.")

    options = {
        "n_portfolios": n_portfolios,
        "percentiles": percentiles,
        "breakpoint_exchanges": breakpoint_exchanges,
        "smooth_bunching": smooth_bunching,
        **kwargs,
    }
    return options


def compute_breakpoints(
    data: pd.DataFrame, sorting_variable: str, breakpoint_options: dict
) -> np.ndarray:
    """
    Compute breakpoints based on a sorting variable for portfolios.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame with the dataset for breakpoint computation.
    sorting_variable (str): Column name used to determine breakpoints.
    breakpoint_options (dict): Dictionary containing breakpoint parameters,
        including:
        - "n_portfolios" (int, optional): Number of equally sized portfolios
        - "percentiles" (list, optional): Custom percentiles for breakpoints
        - "breakpoint_exchanges" (list, optional):
                Exchanges to filter the data before computing breakpoints
        - "smooth_bunching" (bool, optional): To smooth edge breakpoints or not

    Returns
    -------
    np.ndarray: A sorted array of breakpoints.
    """
    if not isinstance(breakpoint_options, dict):
        raise ValueError("breakpoint_options must be a dictionary.")

    n_portfolios = breakpoint_options.get("n_portfolios")
    percentiles = breakpoint_options.get("percentiles")
    exchanges = breakpoint_options.get("breakpoint_exchanges")
    smooth_bunching = breakpoint_options.get("smooth_bunching", False)

    if n_portfolios is not None and percentiles is not None:
        raise ValueError(
            "Provide either 'n_portfolios' or 'percentiles', not both."
        )
    elif n_portfolios is None and percentiles is None:
        raise ValueError(
            "Either 'n_portfolios' or 'percentiles' must be specified."
        )

    if exchanges is not None:
        if "exchange" not in data.columns:
            raise ValueError(
                "Data must contain an 'exchange' column to use breakpoint_exchanges."
            )
        data = data.query(f"exchange in {exchanges}")

    if n_portfolios is not None:
        if n_portfolios <= 1:
            raise ValueError("n_portfolios must be greater than 1.")
        breakpoints = (
            data.get(sorting_variable)
            .quantile(
                np.linspace(0, 1, num=n_portfolios + 1), interpolation="linear"
            )
            .drop_duplicates()
        )
    else:
        breakpoints = (
            data.get(sorting_variable)
            .quantile([0] + percentiles + [1], interpolation="linear")
            .drop_duplicates()
        )
    try:
        breakpoints.iloc[0] = -np.inf
        breakpoints.iloc[-1] = np.inf
    except AttributeError:
        breakpoints.iloc[0] = -np.Inf
        breakpoints.iloc[-1] = np.Inf

    if smooth_bunching:
        if (breakpoints[0] == breakpoints[1]) and (
            breakpoints[-2] == breakpoints[-1]
        ):
            print(
                "Warning: Clustering at extreme breakpoints detected. "
                "Adjusting non-edge portfolios."
            )
            new_values = data[sorting_variable][
                (data[sorting_variable] > breakpoints[0])
                & (data[sorting_variable] < breakpoints[-1])
            ]
            new_probs = np.linspace(0, 1, len(breakpoints) - 2)
            breakpoints[1:-1] = np.quantile(new_values, new_probs)

    breakpoints.loc[1:] += 1e-20  # Ensure proper binning
    return breakpoints


# def compute_long_short_returns(
#     data: pd.DataFrame,
#     direction: str = "top_minus_bottom",
#     date_col: str = "date",
#     portfolio_col: str = "portfolio",
#     ret_excess_col: str = "ret_excess",
# ) -> pd.DataFrame:
#     """
#     Compute long-short returns based on portfolio returns.

#     Parameters
#     ----------
#     data (pd.DataFrame): DataFrame containing portfolio returns with columns
#         for portfolio ID, date, and return measurements.
#     direction (str, optional): Direction of calculation. "top_minus_bottom"
#         (default) or "bottom_minus_top".
#     date_col (str, optional): Column name indicating dates.
#     portfolio_col (str, optional): Column name indicating portfolio
#         identifiers.
#     ret_excess_col (str, optional): Column name prefix for excess return
#         measurements.

#     Returns
#     -------
#     pd.DataFrame: A DataFrame with computed long-short returns.
#     """
#     if direction not in ["top_minus_bottom", "bottom_minus_top"]:
#         raise ValueError(
#             "direction must be either 'top_minus_bottom' or'bottom_minus_top'"
#         )
#     data = data.copy()

#     # Identify top and bottom portfolios
#     grouped = data.groupby(date_col)
#     top_bottom = grouped.filter(lambda x: x[portfolio_col].nunique() >= 2)
#     top_bottom["portfolio"] = np.where(
#         top_bottom[portfolio_col] == top_bottom[portfolio_col].max(),
#         "top",
#         "bottom",
#     )

#     # Pivot data to get top and bottom returns
#     long_short_df = (
#         top_bottom.pivot_table(
#             index=[date_col], columns="portfolio", values=ret_excess_col
#         )
#         .dropna()
#         .reset_index()
#     )

#     # Compute long-short returns
#     long_short_df["long_short_return"] = (
#         long_short_df["top"] - long_short_df["bottom"]
#     ) * (-1 if direction == "bottom_minus_top" else 1)

#     return long_short_df[[date_col, "long_short_return"]]


# def compute_portfolio_returns(
#     sorting_data,
#     sorting_variables,
#     sorting_method,
#     rebalancing_month=None,
#     breakpoint_options_main=None,
#     breakpoint_options_secondary=None,
#     breakpoint_function_main=None,
#     breakpoint_function_secondary=None,
#     min_portfolio_size=0,
#     data_options=None,
# ):
#     """Compute portfolio returns based on sorting variables and methods.

#     Parameters:
#         sorting_data (pd.DataFrame): Data for portfolio assignment and return computation.
#         sorting_variables (list): List of variables for sorting.
#         sorting_method (str): Sorting method ('univariate' or 'bivariate').
#         rebalancing_month (int, optional): Month for annual rebalancing.
#         breakpoint_options_main (dict, optional): Options for main sorting variable.
#         breakpoint_options_secondary (dict, optional): Options for secondary sorting variable.
#         breakpoint_function_main (callable, optional): Function for main sorting.
#         breakpoint_function_secondary (callable, optional): Function for secondary sorting.
#         min_portfolio_size (int): Minimum portfolio size.
#         data_options (dict, optional): Additional data processing options.

#     Returns:
#         pd.DataFrame: DataFrame with computed portfolio returns.
#     """
#     pass


def create_summary_statistics(
    data: pd.DataFrame,
    variables: list,
    by: str = None,
    detail: bool = False,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Compute summary statistics for specified numeric variables in a DataFrame.

    Parameters
    ----------
    data (pd.DataFrame): A DataFrame containing the data to summarize.
    variables (list): List of column names to summarize
        (must be numeric or boolean).
    by (str, optional): A column name to group data before summarization.
        Default is None.
    detail (bool, optional): Whether to include detailed quantiles.
        Default is False.
    drop_na (bool, optional): Whether to drop missing values before
        summarizing. Default is False.

    Returns
    -------
    pd.DataFrame: A DataFrame containing summary statistics for
        each selected variable.
    """
    # Check that all specified variables are numeric or boolean
    non_numeric_vars = [
        var
        for var in variables
        if not np.issubdtype(data[var].dtype, np.number)
    ]
    if non_numeric_vars:
        raise ValueError(
            "The following columns are not numeric or boolean: "
            f"{', '.join(non_numeric_vars)}"
        )

    # Drop missing values if specified
    if drop_na:
        data = data.dropna(subset=variables)

    # Compute summary statistics using describe
    percentiles = (
        [0.5]
        if not detail
        else [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    if by:
        summary_df = (
            data.groupby(by)
            .describe(percentiles=percentiles)
            .get(variables)
            .reset_index()
            .rename(columns={"index": "variable"})
        )
    else:
        summary_df = (
            data.get(variables)
            .describe(percentiles=percentiles)
            .transpose()
            .reset_index()
            .rename(columns={"index": "variable"})
        )

    return summary_df.round(3)


def estimate_betas(
    data: pd.DataFrame,
    model: str,
    lookback: pd.Timedelta,
    min_obs: int = None,
    id_col: str = "permno",
) -> pd.DataFrame:
    """
    Estimate rolling betas using RollingOLS.from_formula from statsmodels.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame containing stock return data with date and
        stock identifier columns.
    model (str): A formula representing the regression model
        (e.g., 'ret_excess ~ mkt_excess').
    lookback (int): The period window size to estimate rolling betas.
    min_obs (int, optional): Minimum number of observations required for a
        valid estimate. Defaults to 80% of lookback.
    id_col (str, optional): Column name representing the stock identifier.
        Defaults to 'permno'.

    Returns
    -------
    pd.DataFrame: A DataFrame with estimated rolling betas for each stock
        and time period.
    """
    if min_obs is None:
        min_obs = int(lookback * 0.8)
    elif min_obs <= 0:
        raise ValueError("min_obs must be a positive integer.")

    results = []
    for stock_id, group in data.groupby(id_col):
        group = group.sort_values("date")

        rolling_model = RollingOLS.from_formula(
            formula=model,
            data=group,
            window=lookback,
            min_nobs=min_obs,
            missing="drop",
        ).fit()

        betas = rolling_model.params
        betas[id_col] = stock_id
        betas["date"] = group["date"].values
        results.append(betas)

    betas_df = pd.concat(results).reset_index()
    return betas_df


def estimate_fama_macbeth(
    data: pd.DataFrame,
    model: str,
    vcov: str = "newey-west",
    vcov_options: dict = {"maxlags": 6},
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Estimate Fama-MacBeth regressions by running cross-sectional regressions.

    Parameters
    ----------
    data (pd.DataFrame): A DataFrame containing the data for the regression.
    model (str): A formula representing the regression model.
    vcov (str): Type of standard errors to compute. Options are "iid" or
        "newey-west".
    vcov_options (dict, optional): Additional options for the Newey-West
        standard errors.
    date_col (str): Column name representing the time periods.

    Returns
    -------
    pd.DataFrame: A DataFrame containing estimated risk premia,
        standard errors, and t-statistics.
    """
    if vcov not in ["iid", "newey-west"]:
        raise ValueError("vcov must be either 'iid' or 'newey-west'.")

    if date_col not in data.columns:
        raise ValueError(f"The data must contain a {date_col} column.")

    # Run cross-sectional regressions
    cross_section_results = []
    for date, group in data.groupby(date_col):
        if len(group) <= len(model.split("~")[1].split("+")):
            continue

        model_fit = smf.ols(model, data=group).fit()
        params = model_fit.params.to_dict()
        params[date_col] = date
        cross_section_results.append(params)

    risk_premiums = pd.DataFrame(cross_section_results)

    # Compute time-series averages
    price_of_risk = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")["estimate"]
        .mean()
        .reset_index()
        .rename(columns={"estimate": "risk_premium"})
    )

    # Compute standard errors based on vcov choice
    def compute_t_statistic(x):
        model = smf.ols("estimate ~ 1", x)
        if vcov == "newey-west":
            fit = model.fit(cov_type="HAC", cov_kwds=vcov_options)
        else:
            fit = model.fit()
        return x["estimate"].mean() / fit.bse["Intercept"]

    price_of_risk_t_stat = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")
        .apply(compute_t_statistic, include_groups=False)
        .reset_index()
        .rename(columns={0: "t_statistic"})
    )

    result_df = price_of_risk.merge(price_of_risk_t_stat, on="factor").round(3)

    return result_df
