"""Main module for tidyfinance package."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS
import warnings
import re

from ._internal import (
    _to_offset,
    _check_new_col,
    _validate_column_name,
    _validate_flag,
    _validate_optional_number,
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
            f"'data' must contain the date column '{date_col}'."
        )

    ref = pd.Timestamp("2020-01-01")
    lag_end = ref + lag_offset
    max_lag_end = ref + max_lag_offset
    if lag_end < ref or max_lag_end < lag_end:
        raise ValueError(
            "'lag' and 'max_lag' must be non-negative and 'max_lag' "
            "must be >= 'lag'."
        )

    missing_cols = [c for c in cols if c not in data.columns]
    if missing_cols:
        raise ValueError(
            f"'data' is missing column(s): {missing_cols}."
        )

    if by_list:
        missing_by = [c for c in by_list if c not in data.columns]
        if missing_by:
            raise ValueError(
                f"'data' is missing grouping column(s): {missing_by}."
            )

    join_cols = by_list + [date_col]
    if data[join_cols].duplicated().any():
        raise ValueError(
            "The combination of 'by' and date columns must be unique "
            "in 'data'."
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
                f"Column '{lag_col_name}' already exists in 'data'."
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

    Unlike :func:'add_lagged_columns', this supports joining across
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
        raise ValueError("'id_keys' must be a string or list of strings.")

    min_lag_offset = _to_offset(min_lag)
    max_lag_offset = _to_offset(max_lag)

    if date_col not in original_data.columns:
        raise ValueError(
            f"'original_data' must contain the column '{date_col}'."
        )
    if date_col not in new_data.columns:
        raise ValueError(
            f"'new_data' must contain the column '{date_col}'."
        )

    missing_original = [
        k for k in id_keys if k not in original_data.columns
    ]
    if missing_original:
        raise ValueError(
            f"'original_data' is missing id column(s): "
            f"{missing_original}."
        )

    missing_new = [k for k in id_keys if k not in new_data.columns]
    if missing_new:
        raise ValueError(
            f"'new_data' is missing id column(s): {missing_new}."
        )

    new_column_names = [
        c for c in new_data.columns if c not in id_keys + [date_col]
    ]
    if not new_column_names:
        raise ValueError(
            f"'new_data' must contain columns besides {id_keys} and "
            f"'{date_col}'."
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
            f"Column(s) in 'new_data' already exist in "
            f"'original_data': {duplicate_cols}. Remove or rename them "
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
        # pd.merge_asof requires the merge key (date) sorted globally
        # in ascending order. The by= argument handles the grouping
        # itself, so we only sort by the merge key.
        left_sorted = result.sort_values(date_col, kind="mergesort")
        right_sorted = tmp.sort_values("_lower", kind="mergesort")

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
    data_options: dict = None,
) -> pd.Series:
    """Assign data points to portfolios based on a sorting variable.

    Users may pass a custom function to compute breakpoints. The
    function must take 'data' and 'sorting_variable' as the first two
    arguments, then 'breakpoint_options' and 'data_options'. The
    function must return an ascending sequence of breakpoints.
    Defaults to compute_breakpoints.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset for portfolio assignment.
    sorting_variable : str
        Column in 'data' used for sorting and portfolio assignment.
    breakpoint_options : dict, optional
        Named arguments passed to 'breakpoint_function'. Typically
        produced by breakpoint_options.
    breakpoint_function : callable, optional
        Function to compute breakpoints. Must return an ascending
        sequence. Defaults to compute_breakpoints.
    data_options : dict, optional
        Column-name mapping (see data_options). Passed through to
        'breakpoint_function'.

    Returns
    -------
    pd.Series
        Portfolio assignments as a float series.
    """
    if breakpoint_function is None:
        breakpoint_function = compute_breakpoints

    if sorting_variable not in data.columns:
        raise ValueError(
            f"Sorting variable '{sorting_variable}' not found in data."
        )

    x = data[sorting_variable]
    n = len(x)

    # Constant check: count of distinct non-NaN values.
    non_na = x.dropna().values
    if len(np.unique(non_na)) <= 1:
        warnings.warn(
            "The sorting variable is constant and only one portfolio "
            "is returned.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series([1.0] * n, index=data.index, dtype=float)

    breakpoints = breakpoint_function(
        data, sorting_variable, breakpoint_options, data_options
    )

    breakpoints = np.asarray(breakpoints, dtype=float)
    if np.any(pd.isna(breakpoints)):
        warnings.warn(
            "No portfolios were assigned due to missing breakpoints.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series([np.nan] * n, index=data.index, dtype=float)

    # Extend the outer breakpoint edges to +/- infinity so values
    # outside the original range still fall into a boundary
    # portfolio rather than NaN.
    extended_bins = breakpoints.copy()
    extended_bins[0] = -np.inf
    extended_bins[-1] = np.inf

    cut_result = pd.cut(
        x,
        bins=extended_bins,
        labels=range(1, len(breakpoints)),
        right=False,
    )
    result = cut_result.astype(float)

    # Cluster warning: number of populated portfolios differs from
    # the expected count (ties collapsed adjacent breakpoints).
    n_expected = len(breakpoints) - 1
    n_actual = result.dropna().nunique()
    if n_actual != n_expected:
        warnings.warn(
            "The number of portfolios differs from the specified "
            "parameter due to clusters in the sorting variable.",
            UserWarning,
            stacklevel=2,
        )
    return result


def breakpoint_options(
    n_portfolios: int = None,
    percentiles: list = None,
    breakpoints_exchanges=None,
    smooth_bunching: bool = False,
    breakpoints_min_size_threshold: float = None,
    **kwargs,
) -> dict:
    """
    Create a dictionary of options for breakpoints in portfolio sorting.

    Parameters
    ----------
    n_portfolios : int, optional
        Number of portfolios to create. Must be a positive integer.
    percentiles : list, optional
        Percentile thresholds for defining breakpoints. Each value
        must be between 0 and 1.
    breakpoints_exchanges : str, optional
        A non-empty string specifying the exchange from which to
        compute the breakpoints.
    smooth_bunching : bool, default False
        Whether smooth bunching should be applied.
    breakpoints_exchanges : str or list of str, optional
        Exchange (or exchanges) from which to compute the breakpoints.
        Accepts either a non-empty string or a non-empty list of
        strings.
    **kwargs
        Additional optional arguments, stored verbatim in the dict.

    Returns
    -------
    dict
        Dictionary containing breakpoint options.
    """
    # Validate n_portfolios
    if n_portfolios is not None:
        if (
            isinstance(n_portfolios, bool)
            or not isinstance(n_portfolios, (int, float))
            or n_portfolios <= 0
            or n_portfolios != int(n_portfolios)
        ):
            raise ValueError("n_portfolios must be a positive integer.")

    # Validate percentiles
    if percentiles is not None:
        try:
            valid = all(
                not isinstance(p, bool)
                and isinstance(p, (int, float))
                and 0 <= p <= 1
                for p in percentiles
            )
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(
                "percentiles must be a numeric vector with values "
                "between 0 and 1."
            )

    # Validate breakpoints_exchanges
    if breakpoints_exchanges is not None:
        if isinstance(breakpoints_exchanges, str):
            if not breakpoints_exchanges:
                raise ValueError(
                    "breakpoints_exchanges must be a non-empty string "
                    "or a non-empty list of strings."
                )
        elif isinstance(breakpoints_exchanges, (list, tuple)):
            if len(breakpoints_exchanges) == 0 or not all(
                isinstance(e, str) for e in breakpoints_exchanges
            ):
                raise ValueError(
                    "breakpoints_exchanges must be a non-empty string "
                    "or a non-empty list of strings."
                )
        else:
            raise ValueError(
                "breakpoints_exchanges must be a non-empty string "
                "or a non-empty list of strings."
            )

    # Validate smooth_bunching
    _validate_flag(
        smooth_bunching,
        "smooth_bunching",
        "smooth_bunching must be a single boolean value (True or False).",
    )

    # Validate breakpoints_min_size_threshold (None or number in (0, 1))
    _validate_optional_number(
        breakpoints_min_size_threshold,
        "breakpoints_min_size_threshold must be None or a single "
        "numeric value between 0 and 1 (exclusive).",
        min=0,
        max=1,
        min_strict=True,
        max_strict=True,
    )

    return {
        "n_portfolios": n_portfolios,
        "percentiles": percentiles,
        "breakpoints_exchanges": breakpoints_exchanges,
        "smooth_bunching": smooth_bunching,
        "breakpoints_min_size_threshold": breakpoints_min_size_threshold,
        **kwargs,
    }


def compute_breakpoints(
    data: pd.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict,
    data_options: dict = None,
) -> np.ndarray:
    """
    Compute breakpoints based on a sorting variable for portfolios.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the dataset for breakpoint computation.
    sorting_variable : str
        Column name used to determine breakpoints.
    breakpoint_options : dict
        Dictionary containing breakpoint parameters, including:

        - 'n_portfolios' (int, optional): Number of equally sized portfolios.
        - 'percentiles' (list, optional): Custom percentiles for breakpoints.
        - 'breakpoints_exchanges' (str or list, optional): Exchanges to
          filter the data before computing breakpoints.
        - 'smooth_bunching' (bool, optional): Whether to smooth edge
          breakpoints.
        - 'breakpoints_min_size_threshold' (float, optional): Quantile below
          which stocks are excluded from breakpoint computation.

    data_options : dict, optional
        Column-name mapping (see data_options). The 'exchange' key is used
        when breakpoints_exchanges is set, and 'mktcap_lag' is used when
        breakpoints_min_size_threshold is set. Defaults to a mapping with
        'exchange' -> 'exchange' and 'mktcap_lag' -> 'mktcap_lag' if None.

    Returns
    -------
    np.ndarray
        A sorted array of breakpoints.
    """
    if not isinstance(breakpoint_options, dict):
        raise ValueError(
            "Please provide a dictionary with breakpoint options."
        )

    n_portfolios = breakpoint_options.get("n_portfolios")
    percentiles = breakpoint_options.get("percentiles")
    breakpoints_exchanges = breakpoint_options.get("breakpoints_exchanges")
    smooth_bunching = breakpoint_options.get("smooth_bunching", False)
    breakpoints_min_size_threshold = breakpoint_options.get(
        "breakpoints_min_size_threshold"
    )

    if data_options is None:
        data_options = {"exchange": "exchange", "mktcap_lag": "mktcap_lag"}

    if n_portfolios is not None and percentiles is not None:
        raise ValueError(
            "Please provide either 'n_portfolios' or 'percentiles', "
            "not both."
        )
    if n_portfolios is None and percentiles is None:
        raise ValueError(
            "You must provide either 'n_portfolios' or 'percentiles'."
        )

    sorting_values = data[sorting_variable].values

    keep_mask = None
    if breakpoints_exchanges is not None:
        exchange_col = data_options["exchange"]
        if exchange_col not in data.columns:
            raise ValueError(
                f"Please provide the column '{exchange_col}' when "
                "filtering using 'breakpoints_exchanges'."
            )
        exchanges_list = (
            [breakpoints_exchanges]
            if isinstance(breakpoints_exchanges, str)
            else list(breakpoints_exchanges)
        )
        keep_mask = data[exchange_col].isin(exchanges_list).values
        sorting_values = sorting_values[keep_mask]

    if breakpoints_min_size_threshold is not None:
        mktcap_col = data_options["mktcap_lag"]
        if mktcap_col not in data.columns:
            raise ValueError(
                f"Column '{mktcap_col}' is required when using "
                "'breakpoints_min_size_threshold'."
            )
        if keep_mask is not None:
            mktcap_ref = data[mktcap_col].values[keep_mask]
        else:
            mktcap_ref = data[mktcap_col].values
        mktcap_ref_clean = mktcap_ref[~pd.isna(mktcap_ref)]
        size_cutoff = np.quantile(
            mktcap_ref_clean, breakpoints_min_size_threshold
        )
        all_mktcap = data[mktcap_col].values
        above_size = (~pd.isna(all_mktcap)) & (all_mktcap > size_cutoff)
        combined_mask = (
            keep_mask & above_size
            if keep_mask is not None
            else above_size
        )
        sorting_values = data[sorting_variable].values[combined_mask]

    if len(sorting_values) == 0:
        warnings.warn(
            "No breakpoints were calculated, likely due to an "
            "insufficient number of observations after filtering for "
            "breakpoint exchanges.",
            UserWarning,
            stacklevel=2,
        )
        return np.array([np.nan])

    if n_portfolios is not None:
        if n_portfolios <= 1:
            raise ValueError("'n_portfolios' must be larger than 1.")
        probs = np.linspace(0, 1, n_portfolios + 1)
    else:
        probs = np.concatenate([[0], np.asarray(percentiles), [1]])
        n_portfolios = len(probs) - 1

    sorting_values_clean = sorting_values[~pd.isna(sorting_values)]
    breakpoints = np.quantile(sorting_values_clean, probs)

    if smooth_bunching:
        both_edges = (
            breakpoints[0] == breakpoints[1]
            and breakpoints[n_portfolios - 1] == breakpoints[n_portfolios]
        )
        lower_edge = breakpoints[0] == breakpoints[1]
        upper_edge = (
            breakpoints[n_portfolios - 1] == breakpoints[n_portfolios]
        )

        if both_edges:
            if percentiles is not None:
                warnings.warn(
                    "'smooth_bunching' is True and equally-spaced "
                    "portfolios are returned for non-edge portfolios.",
                    UserWarning,
                    stacklevel=2,
                )
            mask = (sorting_values_clean > breakpoints[0]) & (
                sorting_values_clean < breakpoints[n_portfolios]
            )
            sorting_values_new = sorting_values_clean[mask]
            probs_new = np.linspace(0, 1, n_portfolios - 1)
            breakpoints_new = np.quantile(sorting_values_new, probs_new)
            breakpoints_new[-1] += 1e-15
            breakpoints = np.concatenate(
                [
                    [breakpoints[0]],
                    breakpoints_new,
                    [breakpoints[n_portfolios]],
                ]
            )
        elif lower_edge:
            if percentiles is not None:
                warnings.warn(
                    "'smooth_bunching' is True and equally-spaced "
                    "portfolios are returned for non-edge portfolios.",
                    UserWarning,
                    stacklevel=2,
                )
            sorting_values_new = sorting_values_clean[
                sorting_values_clean > breakpoints[0]
            ]
            probs_new = np.linspace(0, 1, n_portfolios)
            breakpoints_new = np.quantile(sorting_values_new, probs_new)
            breakpoints = np.concatenate(
                [[breakpoints[0]], breakpoints_new]
            )
        elif upper_edge:
            if percentiles is not None:
                warnings.warn(
                    "'smooth_bunching' is True and equally-spaced "
                    "portfolios are returned for non-edge portfolios.",
                    UserWarning,
                    stacklevel=2,
                )
            sorting_values_new = sorting_values_clean[
                sorting_values_clean < breakpoints[n_portfolios - 1]
            ]
            probs_new = np.linspace(0, 1, n_portfolios)
            breakpoints_new = np.quantile(sorting_values_new, probs_new)
            breakpoints_new[-1] += 1e-15
            breakpoints = np.concatenate(
                [breakpoints_new, [breakpoints[n_portfolios]]]
            )

    breakpoints[1:] += 1e-20
    return breakpoints


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


def filter_options(
    exclude_financials: bool = False,
    exclude_utilities: bool = False,
    min_stock_price: float = None,
    min_size_quantile: float = None,
    min_listing_age: float = None,
    exclude_negative_book_equity: bool = False,
    exclude_negative_earnings: bool = False,
    **kwargs,
) -> dict:
    """
    Create a dictionary of filter options for sample construction.

    These options control which observations are retained before
    portfolio sorting.

    Parameters
    ----------
    exclude_financials : bool, default False
        Whether to exclude financial firms (SIC codes 6000 to 6799).
    exclude_utilities : bool, default False
        Whether to exclude utility firms (SIC codes 4900 to 4999).
    min_stock_price : float, optional
        Minimum stock price required to include an observation. Must
        be strictly positive when provided. None disables the filter.
    min_size_quantile : float, optional
        Minimum cross-sectional size quantile (based on lagged market
        cap) required to include an observation. Must be strictly
        between 0 and 1 when provided. The cutoff is computed from
        NYSE stocks only; this requires an 'exchange' column in the
        data. None disables the filter.
    min_listing_age : float, optional
        Minimum number of months a stock must have been listed in
        CRSP. Must be non-negative when provided. None disables the
        filter.
    exclude_negative_book_equity : bool, default False
        Whether to exclude observations with non-positive book equity.
    exclude_negative_earnings : bool, default False
        Whether to exclude observations with non-positive earnings.
    **kwargs
        Additional optional arguments, stored verbatim in the dict.

    Returns
    -------
    dict
        Dictionary containing filter options.
    """
    _validate_flag(exclude_financials, "exclude_financials")
    _validate_flag(exclude_utilities, "exclude_utilities")
    _validate_flag(
        exclude_negative_book_equity, "exclude_negative_book_equity"
    )
    _validate_flag(exclude_negative_earnings, "exclude_negative_earnings")

    _validate_optional_number(
        min_stock_price,
        "min_stock_price must be a single positive numeric.",
        min=0,
        min_strict=True,
    )
    _validate_optional_number(
        min_size_quantile,
        "min_size_quantile must be a single numeric strictly between "
        "0 and 1.",
        min=0,
        max=1,
        min_strict=True,
        max_strict=True,
    )
    _validate_optional_number(
        min_listing_age,
        "min_listing_age must be a single non-negative numeric.",
        min=0,
    )

    return {
        "exclude_financials": exclude_financials,
        "exclude_utilities": exclude_utilities,
        "min_stock_price": min_stock_price,
        "min_size_quantile": min_size_quantile,
        "min_listing_age": min_listing_age,
        "exclude_negative_book_equity": exclude_negative_book_equity,
        "exclude_negative_earnings": exclude_negative_earnings,
        **kwargs,
    }


_FILTER_OPTIONS_KEYS = {
    "exclude_financials",
    "exclude_utilities",
    "min_stock_price",
    "min_size_quantile",
    "min_listing_age",
    "exclude_negative_book_equity",
    "exclude_negative_earnings",
}

_BREAKPOINT_OPTIONS_KEYS = {
    "n_portfolios",
    "percentiles",
    "breakpoints_exchanges",
    "smooth_bunching",
    "breakpoints_min_size_threshold",
}


def portfolio_sort_options(
    filter_options: dict = None,
    breakpoint_options_main: dict = None,
    breakpoint_options_secondary: dict = None,
    **kwargs,
) -> dict:
    """
    Create a dictionary of portfolio sort options.

    Bundles sample-construction filters and breakpoint specifications
    for use with portfolio-sort workflows.

    Parameters
    ----------
    filter_options : dict, optional
        Filter options dict produced by filter_options(). None (the
        default) applies no filters.
    breakpoint_options_main : dict
        Breakpoint options dict produced by breakpoint_options(),
        specifying breakpoints for the primary sorting variable.
        Required.
    breakpoint_options_secondary : dict, optional
        Breakpoint options dict produced by breakpoint_options() for
        the secondary sorting variable. None (the default) selects a
        univariate sort.
    **kwargs
        Additional optional arguments, stored verbatim in the dict.

    Returns
    -------
    dict
        Dictionary containing the bundled portfolio sort options.
    """
    if filter_options is not None and not (
        isinstance(filter_options, dict)
        and _FILTER_OPTIONS_KEYS.issubset(filter_options.keys())
    ):
        raise ValueError(
            "filter_options must be None or a dict produced by "
            "filter_options()."
        )

    if breakpoint_options_main is None:
        raise ValueError("breakpoint_options_main must be provided.")

    if not (
        isinstance(breakpoint_options_main, dict)
        and _BREAKPOINT_OPTIONS_KEYS.issubset(
            breakpoint_options_main.keys()
        )
    ):
        raise ValueError(
            "breakpoint_options_main must be a dict produced by "
            "breakpoint_options()."
        )

    if breakpoint_options_secondary is not None and not (
        isinstance(breakpoint_options_secondary, dict)
        and _BREAKPOINT_OPTIONS_KEYS.issubset(
            breakpoint_options_secondary.keys()
        )
    ):
        raise ValueError(
            "breakpoint_options_secondary must be None or a dict "
            "produced by breakpoint_options()."
        )

    return {
        "filter_options": filter_options,
        "breakpoint_options_main": breakpoint_options_main,
        "breakpoint_options_secondary": breakpoint_options_secondary,
        **kwargs,
    }


def _summarise_portfolio_returns(
    data: pd.DataFrame,
    group_cols: list,
    ret_col: str,
    w_col: str,
    w_capped_col: str,
    min_portfolio_size: int,
) -> pd.DataFrame:
    """Compute vw, ew, and vw_capped returns within groups.

    Groups with fewer than min_portfolio_size observations get NaN
    in all three return columns. Groups whose weight sum is zero get
    NaN in the corresponding weighted return.

    Parameters
    ----------
    data : pd.DataFrame
        Stock-level panel with the return and weight columns.
    group_cols : list
        Columns to group by (typically the portfolio and date columns).
    ret_col : str
        Excess-return column name.
    w_col : str
        Market-cap weight column name.
    w_capped_col : str
        Capped market-cap weight column name.
    min_portfolio_size : int
        Minimum observations per group; groups below this size get NaN.

    Returns
    -------
    pd.DataFrame
        One row per group with columns from group_cols plus
        ret_excess_vw, ret_excess_ew, and ret_excess_vw_capped.
    """
    work = data.copy()
    work["_rw"] = work[ret_col] * work[w_col]
    work["_rwc"] = work[ret_col] * work[w_capped_col]

    sums = work.groupby(group_cols, as_index=False).agg(
        _n=(ret_col, "size"),
        _r_sum=(ret_col, "sum"),
        _w_sum=(w_col, "sum"),
        _wc_sum=(w_capped_col, "sum"),
        _rw_sum=("_rw", "sum"),
        _rwc_sum=("_rwc", "sum"),
    )

    sums["ret_excess_ew"] = sums["_r_sum"] / sums["_n"]
    sums["ret_excess_vw"] = np.where(
        sums["_w_sum"] == 0, np.nan, sums["_rw_sum"] / sums["_w_sum"]
    )
    sums["ret_excess_vw_capped"] = np.where(
        sums["_wc_sum"] == 0,
        np.nan,
        sums["_rwc_sum"] / sums["_wc_sum"],
    )

    too_small = sums["_n"] < min_portfolio_size
    sums.loc[
        too_small,
        ["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"],
    ] = np.nan

    return sums[
        list(group_cols)
        + ["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"]
    ]


def _aggregate_bivariate_returns(
    portfolio_returns: pd.DataFrame,
    date_col: str,
    ret_col: str,
    w_col: str,
    w_capped_col: str,
    min_portfolio_size: int,
) -> pd.DataFrame:
    """Aggregate bivariate-sort returns across the secondary dimension.

    Computes cell-level returns over (portfolio_main,
    portfolio_secondary, date) without an occupancy threshold, then
    averages across the secondary buckets to obtain reported
    (portfolio_main, date) returns. min_portfolio_size is applied to
    the per-(portfolio_main, date) firm count.

    Parameters
    ----------
    portfolio_returns : pd.DataFrame
        Panel with columns portfolio_main, portfolio_secondary, the
        date column, and per-stock returns/weights.
    date_col, ret_col, w_col, w_capped_col : str
        Column names.
    min_portfolio_size : int
        Minimum firms per reported (portfolio_main, date) cross-section.
        Cross-sections below this size receive NaN.

    Returns
    -------
    pd.DataFrame
        Columns portfolio, the date column, and the three return columns.
    """
    n_per_main = (
        portfolio_returns.dropna(
            subset=["portfolio_main", "portfolio_secondary"]
        )
        .groupby(["portfolio_main", date_col], as_index=False)
        .size()
        .rename(columns={"size": "n_firms"})
    )

    cell_returns = _summarise_portfolio_returns(
        portfolio_returns,
        ["portfolio_main", "portfolio_secondary", date_col],
        ret_col,
        w_col,
        w_capped_col,
        min_portfolio_size=0,
    )

    avg_returns = (
        cell_returns.groupby(
            ["portfolio_main", date_col], as_index=False
        )
        .agg(
            ret_excess_vw=("ret_excess_vw", "mean"),
            ret_excess_ew=("ret_excess_ew", "mean"),
            ret_excess_vw_capped=("ret_excess_vw_capped", "mean"),
        )
        .rename(columns={"portfolio_main": "portfolio"})
    )

    n_renamed = n_per_main.rename(
        columns={"portfolio_main": "portfolio"}
    )
    result = avg_returns.merge(
        n_renamed, on=["portfolio", date_col], how="left"
    )

    insufficient = result["n_firms"].isna() | (
        result["n_firms"] < min_portfolio_size
    )
    for col in (
        "ret_excess_vw",
        "ret_excess_ew",
        "ret_excess_vw_capped",
    ):
        result.loc[result[col].isna() | insufficient, col] = np.nan

    return result.drop(columns="n_firms")


def _join_rebalanced_portfolios(
    data: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    date_col: str,
    id_col: str,
    rebalancing_month: int,
) -> pd.DataFrame:
    """Join annual portfolio assignments to all dates in rebalancing window.

    Each date in 'data' is matched to the most recent portfolio
    assignment in 'portfolio_data' for the same id, using the
    12-month window starting at the rebalancing month.

    Parameters
    ----------
    data : pd.DataFrame
        Full stock-level panel.
    portfolio_data : pd.DataFrame
        Portfolio assignments at rebalancing dates. Must contain
        id_col, date_col, and one or more portfolio columns.
    date_col, id_col : str
        Date and stock-identifier column names.
    rebalancing_month : int
        The month (1-12) when portfolios are rebalanced annually.

    Returns
    -------
    pd.DataFrame
        'data' with the portfolio columns joined from 'portfolio_data'
        (NaN where no rebalancing window matches).
    """
    def _window_year(d):
        return d.year if d.month >= rebalancing_month else d.year - 1

    data_w = data.copy()
    data_w["_window_year"] = data_w[date_col].apply(_window_year)

    pd_w = portfolio_data.copy()
    pd_w["_window_year"] = pd_w[date_col].dt.year
    pd_w = pd_w.drop(columns=date_col)

    merged = data_w.merge(
        pd_w, on=[id_col, "_window_year"], how="left"
    )
    return merged.drop(columns="_window_year")


def compute_portfolio_returns(
    data: pd.DataFrame,
    sorting_variables,
    sorting_method: str,
    rebalancing_month: int = None,
    breakpoint_options_main: dict = None,
    breakpoint_options_secondary: dict = None,
    breakpoint_function_main=None,
    breakpoint_function_secondary=None,
    min_portfolio_size: int = 1,
    cap_weight: float = 0.8,
    data_options: dict = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute univariate or bivariate portfolio returns.

    Sorts stocks into portfolios based on one or two sorting variables
    and computes equal-weighted, value-weighted, and capped
    value-weighted returns per portfolio-date cross-section. Supports
    periodic (per-date) or annual (single rebalancing month per year)
    rebalancing.

    Parameters
    ----------
    data : pd.DataFrame
        Stock-level panel. Must contain the id, date, and excess
        return columns (configurable via data_options), the sorting
        variable(s), and optionally a market-cap lag column.
    sorting_variables : str or list of str
        Column(s) to sort by. One for univariate, two for bivariate.
    sorting_method : str
        One of 'univariate', 'bivariate-dependent', or
        'bivariate-independent'.
    rebalancing_month : int, optional
        Month (1-12) to use for annual rebalancing. None means
        rebalance every period.
    breakpoint_options_main : dict
        Breakpoint options for the primary sorting variable (produced
        by breakpoint_options()). Required.
    breakpoint_options_secondary : dict, optional
        Breakpoint options for the secondary sorting variable.
        Required for bivariate sorts.
    breakpoint_function_main : callable, optional
        Function for computing primary breakpoints. Defaults to
        compute_breakpoints.
    breakpoint_function_secondary : callable, optional
        Function for computing secondary breakpoints. Defaults to
        compute_breakpoints.
    min_portfolio_size : int, default 1
        Minimum firms per (portfolio, date) cross-section.
        Cross-sections below this size receive NaN.
    cap_weight : float, default 0.8
        Quantile of market cap at which to cap weights for the capped
        value-weighted return. Must be in [0, 1].
    data_options : dict, optional
        Column-name mapping (see data_options). Defaults to
        data_options() if None.
    quiet : bool, default False
        If True, suppress informational warnings about missing
        observations.

    Returns
    -------
    pd.DataFrame
        Complete portfolio-date panel with columns: portfolio, date,
        and the return columns. If a market-cap lag column is present
        in the input, returns ret_excess_vw, ret_excess_ew, and
        ret_excess_vw_capped. Otherwise only ret_excess_ew.
    """
    _validate_flag(quiet, "quiet")

    if data_options is None:
        data_options = {
            "id": "permno",
            "date": "date",
            "exchange": "exchange",
            "mktcap_lag": "mktcap_lag",
            "ret_excess": "ret_excess",
            "portfolio": "portfolio",
            "siccd": "siccd",
            "price": "prc_adj",
            "listing_age": "listing_age",
            "be": "be",
            "earnings": "ib",
        }

    if breakpoint_function_main is None:
        breakpoint_function_main = compute_breakpoints
    if breakpoint_function_secondary is None:
        breakpoint_function_secondary = compute_breakpoints

    if isinstance(sorting_variables, str):
        sorting_variables = [sorting_variables]
    if not sorting_variables:
        raise ValueError("You must provide at least one sorting variable.")

    valid_methods = (
        "univariate",
        "bivariate-dependent",
        "bivariate-independent",
    )
    if sorting_method not in valid_methods:
        raise ValueError("Invalid sorting method.")

    if (
        sorting_method in ("bivariate-dependent", "bivariate-independent")
        and breakpoint_options_secondary is None
    ):
        warnings.warn(
            "No 'breakpoint_options_secondary' specified in bivariate "
            "sort.",
            UserWarning,
            stacklevel=2,
        )

    if (
        isinstance(cap_weight, bool)
        or not isinstance(cap_weight, (int, float))
        or pd.isna(cap_weight)
        or cap_weight < 0
        or cap_weight > 1
    ):
        raise ValueError(
            "'cap_weight' must be a single numeric value in [0, 1]."
        )

    id_col = data_options["id"]
    date_col = data_options["date"]
    ret_col = data_options["ret_excess"]
    w_col = data_options["mktcap_lag"]
    w_capped_col = w_col + "_capped"

    required_columns = list(sorting_variables) + [
        id_col,
        date_col,
        ret_col,
    ]
    missing_columns = [
        c for c in required_columns if c not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing columns: {', '.join(missing_columns)}."
        )

    mktcap_lag_missing = w_col not in data.columns
    data = data.copy()
    if mktcap_lag_missing:
        data[w_col] = 1

    all_dates = sorted(data[date_col].unique())

    data = data.dropna(subset=list(sorting_variables))

    _check_new_col(data, w_capped_col)
    cap_quantile = data.groupby(date_col)[w_col].transform(
        lambda x: x.quantile(cap_weight)
    )
    data[w_capped_col] = np.minimum(data[w_col], cap_quantile)

    missing_mcap = data[w_col].isna()
    data.loc[missing_mcap, w_col] = 0
    data.loc[missing_mcap, w_capped_col] = 0

    if len(data) == 0:
        if not quiet:
            warnings.warn(
                "Returning an empty panel: all observations were "
                "filtered out (insufficient observations on every date).",
                UserWarning,
                stacklevel=2,
            )
        cols = ["portfolio", date_col, "ret_excess_ew"]
        if not mktcap_lag_missing:
            cols = [
                "portfolio",
                date_col,
                "ret_excess_vw",
                "ret_excess_ew",
                "ret_excess_vw_capped",
            ]
        return pd.DataFrame({c: [] for c in cols})

    if rebalancing_month is not None and (
        rebalancing_month < 1 or rebalancing_month > 12
    ):
        raise ValueError("Invalid rebalancing_month.")

    if sorting_method == "univariate":
        if len(sorting_variables) > 1:
            raise ValueError(
                "Only provide one sorting variable for univariate sorts."
            )

        sv = sorting_variables[0]

        if rebalancing_month is None:
            def _assigner(g):
                return assign_portfolio(
                    g,
                    sorting_variable=sv,
                    breakpoint_options=breakpoint_options_main,
                    breakpoint_function=breakpoint_function_main,
                    data_options=data_options,
                )

            data["portfolio"] = data.groupby(
                date_col, group_keys=False
            ).apply(_assigner, include_groups=False)
            portfolio_returns = data
        else:
            filtered_data = data[
                data[date_col].dt.month == rebalancing_month
            ]
            if len(filtered_data) == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}. Check that the data contains "
                    "dates in the specified rebalancing month."
                )

            def _assigner(g):
                return assign_portfolio(
                    g,
                    sorting_variable=sv,
                    breakpoint_options=breakpoint_options_main,
                    breakpoint_function=breakpoint_function_main,
                    data_options=data_options,
                )

            portfolio_data = filtered_data.copy()
            portfolio_data["portfolio"] = portfolio_data.groupby(
                date_col, group_keys=False
            ).apply(_assigner, include_groups=False)
            portfolio_data = portfolio_data[
                [id_col, date_col, "portfolio"]
            ]
            portfolio_returns = _join_rebalanced_portfolios(
                data,
                portfolio_data,
                date_col,
                id_col,
                rebalancing_month,
            )

        portfolio_returns = _summarise_portfolio_returns(
            portfolio_returns,
            ["portfolio", date_col],
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
        )

    elif sorting_method == "bivariate-dependent":
        if len(sorting_variables) != 2:
            raise ValueError(
                "Provide two sorting variables for bivariate sorts."
            )
        sv_main, sv_sec = sorting_variables[0], sorting_variables[1]

        def _assign_main(g):
            return assign_portfolio(
                g,
                sorting_variable=sv_main,
                breakpoint_options=breakpoint_options_main,
                breakpoint_function=breakpoint_function_main,
                data_options=data_options,
            )

        def _assign_sec(g):
            return assign_portfolio(
                g,
                sorting_variable=sv_sec,
                breakpoint_options=breakpoint_options_secondary,
                breakpoint_function=breakpoint_function_secondary,
                data_options=data_options,
            )

        if rebalancing_month is None:
            data["portfolio_secondary"] = data.groupby(
                date_col, group_keys=False
            ).apply(_assign_sec, include_groups=False)
            data["portfolio_main"] = data.groupby(
                [date_col, "portfolio_secondary"], group_keys=False
            ).apply(_assign_main, include_groups=False)
            portfolio_returns = data
        else:
            filtered_data = data[
                data[date_col].dt.month == rebalancing_month
            ]
            if len(filtered_data) == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = filtered_data.copy()
            portfolio_data["portfolio_secondary"] = (
                portfolio_data.groupby(date_col, group_keys=False).apply(
                    _assign_sec, include_groups=False
                )
            )
            portfolio_data["portfolio_main"] = portfolio_data.groupby(
                [date_col, "portfolio_secondary"], group_keys=False
            ).apply(_assign_main, include_groups=False)
            portfolio_data = portfolio_data[
                [
                    id_col,
                    date_col,
                    "portfolio_main",
                    "portfolio_secondary",
                ]
            ]
            portfolio_returns = _join_rebalanced_portfolios(
                data,
                portfolio_data,
                date_col,
                id_col,
                rebalancing_month,
            )

        portfolio_returns = _aggregate_bivariate_returns(
            portfolio_returns,
            date_col,
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
        )

    else:  # bivariate-independent
        if len(sorting_variables) != 2:
            raise ValueError(
                "Provide two sorting variables for bivariate sorts."
            )
        sv_main, sv_sec = sorting_variables[0], sorting_variables[1]

        def _assign_main(g):
            return assign_portfolio(
                g,
                sorting_variable=sv_main,
                breakpoint_options=breakpoint_options_main,
                breakpoint_function=breakpoint_function_main,
                data_options=data_options,
            )

        def _assign_sec(g):
            return assign_portfolio(
                g,
                sorting_variable=sv_sec,
                breakpoint_options=breakpoint_options_secondary,
                breakpoint_function=breakpoint_function_secondary,
                data_options=data_options,
            )

        if rebalancing_month is None:
            data["portfolio_secondary"] = data.groupby(
                date_col, group_keys=False
            ).apply(_assign_sec, include_groups=False)
            data["portfolio_main"] = data.groupby(
                date_col, group_keys=False
            ).apply(_assign_main, include_groups=False)
            portfolio_returns = data
        else:
            filtered_data = data[
                data[date_col].dt.month == rebalancing_month
            ]
            if len(filtered_data) == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = filtered_data.copy()
            portfolio_data["portfolio_secondary"] = (
                portfolio_data.groupby(date_col, group_keys=False).apply(
                    _assign_sec, include_groups=False
                )
            )
            portfolio_data["portfolio_main"] = portfolio_data.groupby(
                date_col, group_keys=False
            ).apply(_assign_main, include_groups=False)
            portfolio_data = portfolio_data[
                [
                    id_col,
                    date_col,
                    "portfolio_main",
                    "portfolio_secondary",
                ]
            ]
            portfolio_returns = _join_rebalanced_portfolios(
                data,
                portfolio_data,
                date_col,
                id_col,
                rebalancing_month,
            )

        portfolio_returns = _aggregate_bivariate_returns(
            portfolio_returns,
            date_col,
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
        )

    portfolio_returns = portfolio_returns[
        portfolio_returns["portfolio"].notna()
    ]

    if rebalancing_month is not None:
        matching_dates = [
            d
            for d in all_dates
            if pd.Timestamp(d).month == rebalancing_month
        ]
        if not matching_dates:
            raise ValueError(
                f"No dates in data match for rebalancing month = "
                f"{rebalancing_month}."
            )
        first_rebalancing_date = min(matching_dates)
        all_dates = [d for d in all_dates if d >= first_rebalancing_date]

    all_portfolios = (
        portfolio_returns["portfolio"].dropna().unique().tolist()
    )
    complete_panel = pd.MultiIndex.from_product(
        [all_portfolios, all_dates], names=["portfolio", date_col]
    ).to_frame(index=False)

    return_cols = (
        ["ret_excess_ew"]
        if mktcap_lag_missing
        else [
            "ret_excess_vw",
            "ret_excess_ew",
            "ret_excess_vw_capped",
        ]
    )

    portfolio_returns = complete_panel.merge(
        portfolio_returns, on=["portfolio", date_col], how="left"
    )[["portfolio", date_col] + return_cols]

    n_missing = portfolio_returns["ret_excess_ew"].isna().sum()
    if not quiet and n_missing > 0:
        warnings.warn(
            f"Returning a complete panel with {n_missing} missing "
            "values in factor returns due to insufficient observations "
            f"(fewer than {min_portfolio_size} firms per "
            "(portfolio, date) cross-section).",
            UserWarning,
            stacklevel=2,
        )

    return portfolio_returns


def compute_long_short_returns(
    data: pd.DataFrame,
    direction: str = "top_minus_bottom",
    data_options: dict = None,
) -> pd.DataFrame:
    """
    Compute long-short returns from a portfolio-return panel.

    For each date, identifies the lowest-numbered ('bottom') and
    highest-numbered ('top') portfolios and computes the long-short
    return as (top - bottom) or (bottom - top) depending on
    'direction'. Dates with only one portfolio receive NaN.

    Parameters
    ----------
    data : pd.DataFrame
        Portfolio-return panel. Must contain the date, portfolio,
        and one or more return columns whose names contain the
        excess-return key from data_options.
    direction : str, default 'top_minus_bottom'
        Either 'top_minus_bottom' (long top, short bottom) or
        'bottom_minus_top' (long bottom, short top).
    data_options : dict, optional
        Column-name mapping (see data_options). Uses defaults
        {'date': 'date', 'ret_excess': 'ret_excess',
        'portfolio': 'portfolio'} when None.

    Returns
    -------
    pd.DataFrame
        One row per date with one column per return measurement
        holding the long-short return.
    """
    if data_options is None:
        data_options = {
            "date": "date",
            "ret_excess": "ret_excess",
            "portfolio": "portfolio",
        }

    date_col = data_options["date"]
    ret_excess_col = data_options["ret_excess"]
    portfolio_col = data_options["portfolio"]

    ret_cols = [c for c in data.columns if ret_excess_col in c]

    work = data.copy()
    work["_min_p"] = work.groupby(date_col)[portfolio_col].transform("min")
    work["_max_p"] = work.groupby(date_col)[portfolio_col].transform("max")

    single_portfolio = work["_min_p"] == work["_max_p"]
    is_bottom = work[portfolio_col] == work["_min_p"]
    is_top = work[portfolio_col] == work["_max_p"]
    work["_leg"] = pd.Series(index=work.index, dtype="object")
    work.loc[is_bottom & ~single_portfolio, "_leg"] = "bottom"
    work.loc[is_top & ~single_portfolio, "_leg"] = "top"

    legs_only = work[work["_leg"].isin(["bottom", "top"])].copy()

    all_dates = (
        pd.Series(data[date_col].unique(), name=date_col)
        .sort_values()
        .reset_index(drop=True)
    )
    result = pd.DataFrame({date_col: all_dates})

    for ret_col in ret_cols:
        pivoted = legs_only.pivot_table(
            index=date_col,
            columns="_leg",
            values=ret_col,
            aggfunc="first",
        )
        for leg in ("bottom", "top"):
            if leg not in pivoted.columns:
                pivoted[leg] = np.nan

        if direction == "bottom_minus_top":
            ls = pivoted["bottom"] - pivoted["top"]
        else:
            ls = pivoted["top"] - pivoted["bottom"]

        ls_df = ls.rename(ret_col).reset_index()
        result = result.merge(ls_df, on=date_col, how="left")

    return result


def compute_rolling_value(
    data: pd.DataFrame,
    f,
    period: str = "month",
    periods: int = 12,
    min_obs: int = None,
    data_options: dict = None,
) -> np.ndarray:
    """Apply a summary function over rolling calendar-period windows.

    For each row at date d, takes the window of all rows whose date
    falls in the same or one of the previous (periods - 1) calendar
    periods. Drops NaN rows; if fewer than min_obs rows remain, the
    result is NaN, otherwise applies f to the window data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with a datetime column named per data_options.
    f : callable
        Function applied to each window. Receives a DataFrame slice
        (complete cases) and must return a single scalar.
    period : str, default 'month'
        Calendar period unit: 'month', 'quarter', or 'year'.
    periods : int, default 12
        Number of periods in the rolling window.
    min_obs : int, optional
        Minimum non-missing rows required per window. Defaults to
        periods.
    data_options : dict, optional
        Column-name mapping (see data_options). Uses {'date': 'date'}
        if None.

    Returns
    -------
    np.ndarray
        Numeric vector aligned with the rows of data.
    """
    if data_options is None:
        data_options = {"date": "date"}
    if min_obs is None:
        min_obs = periods

    date_col = data_options.get("date")

    if not isinstance(date_col, str):
        raise ValueError(
            "'date' in data_options must be a single non-missing string."
        )

    if date_col not in data.columns:
        raise ValueError(
            f"'data' must contain a '{date_col}' column."
        )

    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        raise ValueError(
            f"The '{date_col}' column must be of datetime dtype."
        )

    if not isinstance(period, str):
        raise ValueError("'period' must be a single string.")

    period_freq_map = {"month": "M", "quarter": "Q", "year": "Y"}
    if period not in period_freq_map:
        raise ValueError(
            "'period' must be one of 'month', 'quarter', 'year'."
        )

    buckets = data[date_col].dt.to_period(period_freq_map[period])

    n = len(data)
    result = np.full(n, np.nan)

    for i in range(n):
        anchor_bucket = buckets.iloc[i]
        start_bucket = anchor_bucket - (periods - 1)
        in_window = (buckets >= start_bucket) & (
            buckets <= anchor_bucket
        )
        window_data = data[in_window].dropna()
        if len(window_data) >= min_obs:
            result[i] = f(window_data)

    return result


def _require_column(data: pd.DataFrame, col: str, arg: str, info: str = None
) -> None:
    """Raise ValueError if col is not in data.columns."""
    if col not in data.columns:
        if info is None:
            info = f"Set {arg} to the correct column name."
        raise ValueError(
            f"Column '{col}' not found in 'data'. {info}"
        )


def _filter_with_log(
    data: pd.DataFrame, condition, label: str, quiet: bool
) -> pd.DataFrame:
    """Apply a boolean filter and optionally warn about dropped rows."""
    n_before = len(data)
    data = data[condition]
    n_dropped = n_before - len(data)
    if not quiet and n_dropped > 0:
        warnings.warn(
            f"Filter '{label}': removed {n_dropped} observation(s).",
            UserWarning,
            stacklevel=2,
        )
    return data


def filter_sorting_data(
    data: pd.DataFrame,
    filter_options: dict = None,
    data_options: dict = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Apply sample-construction filters to a stock-level panel.

    Filters are applied in a fixed order: financials exclusion,
    utilities exclusion, minimum stock price, minimum size quantile,
    minimum listing age, positive book equity, positive earnings.
    An informational warning is emitted for each filter that removes
    at least one row, unless 'quiet' is True.

    Parameters
    ----------
    data : pd.DataFrame
        Stock-level panel to filter.
    filter_options : dict, optional
        Filter options dict produced by filter_options(). None
        applies no filters (uses the filter_options() defaults).
    data_options : dict, optional
        Column-name mapping (see data_options). Uses the
        data_options() defaults if None.
    quiet : bool, default False
        If True, suppress per-filter warnings.

    Returns
    -------
    pd.DataFrame
        Filtered data with the same columns as input.
    """
    if not isinstance(quiet, bool):
        raise ValueError("'quiet' must be a single boolean.")

    if filter_options is None:
        filter_options = {
            "exclude_financials": False,
            "exclude_utilities": False,
            "min_stock_price": None,
            "min_size_quantile": None,
            "min_listing_age": None,
            "exclude_negative_book_equity": False,
            "exclude_negative_earnings": False,
        }

    if data_options is None:
        data_options = {
            "id": "permno",
            "date": "date",
            "exchange": "exchange",
            "mktcap_lag": "mktcap_lag",
            "ret_excess": "ret_excess",
            "portfolio": "portfolio",
            "siccd": "siccd",
            "price": "prc_adj",
            "listing_age": "listing_age",
            "be": "be",
            "earnings": "ib",
        }

    # exclude_financials / exclude_utilities (share the SIC column)
    if filter_options.get("exclude_financials") or filter_options.get(
        "exclude_utilities"
    ):
        col_siccd = data_options["siccd"]
        _require_column(data, col_siccd, "data_options['siccd']")

        if filter_options.get("exclude_financials"):
            keep = data[col_siccd].isna() | ~(
                (data[col_siccd] >= 6000) & (data[col_siccd] <= 6799)
            )
            data = _filter_with_log(
                data, keep, "exclude_financials", quiet
            )

        if filter_options.get("exclude_utilities"):
            keep = data[col_siccd].isna() | ~(
                (data[col_siccd] >= 4900) & (data[col_siccd] <= 4999)
            )
            data = _filter_with_log(
                data, keep, "exclude_utilities", quiet
            )

    # min_stock_price
    if filter_options.get("min_stock_price") is not None:
        col_price = data_options["price"]
        _require_column(data, col_price, "data_options['price']")
        keep = data[col_price].notna() & (
            data[col_price] >= filter_options["min_stock_price"]
        )
        data = _filter_with_log(data, keep, "min_stock_price", quiet)

    # min_size_quantile
    if filter_options.get("min_size_quantile") is not None:
        col_mktcap_lag = data_options["mktcap_lag"]
        col_date = data_options["date"]
        col_exchange = data_options["exchange"]
        _require_column(
            data, col_mktcap_lag, "data_options['mktcap_lag']"
        )
        _require_column(data, col_date, "data_options['date']")
        _require_column(
            data,
            col_exchange,
            "data_options['exchange']",
            info=(
                "The size quantile cutoff is computed from NYSE stocks. "
                "Set data_options['exchange'] to the correct column name."
            ),
        )

        n_before = len(data)
        size_threshold = filter_options["min_size_quantile"]
        nyse_data = data[data[col_exchange] == "NYSE"]
        size_cutoffs = (
            nyse_data.groupby(col_date)[col_mktcap_lag]
            .quantile(size_threshold)
            .reset_index(name="_size_cutoff")
        )

        dates_in_data = set(data[col_date].unique())
        dates_with_cutoff = set(size_cutoffs[col_date].unique())
        dates_missing_cutoff = dates_in_data - dates_with_cutoff
        if dates_missing_cutoff:
            warnings.warn(
                f"Filter 'min_size_quantile': "
                f"{len(dates_missing_cutoff)} date(s) dropped because "
                "no NYSE stocks are available to compute the size "
                "quantile cutoff.",
                UserWarning,
                stacklevel=2,
            )

        data = data.merge(size_cutoffs, on=col_date, how="inner")
        data = data[
            data[col_mktcap_lag].notna()
            & (data[col_mktcap_lag] >= data["_size_cutoff"])
        ]
        data = data.drop(columns="_size_cutoff")
        n_dropped = n_before - len(data)
        if not quiet and n_dropped > 0:
            warnings.warn(
                f"Filter 'min_size_quantile': removed {n_dropped} "
                "observation(s).",
                UserWarning,
                stacklevel=2,
            )

    # min_listing_age
    if filter_options.get("min_listing_age") is not None:
        col_listing_age = data_options["listing_age"]
        _require_column(
            data, col_listing_age, "data_options['listing_age']"
        )
        keep = data[col_listing_age].notna() & (
            data[col_listing_age] >= filter_options["min_listing_age"]
        )
        data = _filter_with_log(data, keep, "min_listing_age", quiet)

    # exclude_negative_book_equity
    if filter_options.get("exclude_negative_book_equity"):
        col_be = data_options["be"]
        _require_column(data, col_be, "data_options['be']")
        keep = data[col_be].notna() & (data[col_be] > 0)
        data = _filter_with_log(
            data, keep, "exclude_negative_book_equity", quiet
        )

    # exclude_negative_earnings
    if filter_options.get("exclude_negative_earnings"):
        col_earn = data_options["earnings"]
        _require_column(data, col_earn, "data_options['earnings']")
        keep = data[col_earn].notna() & (data[col_earn] > 0)
        data = _filter_with_log(
            data, keep, "exclude_negative_earnings", quiet
        )

    return data


def implement_portfolio_sort(
    data: pd.DataFrame,
    sorting_variables,
    sorting_method: str,
    portfolio_sort_options: dict,
    rebalancing_month: int = None,
    breakpoint_function_main=None,
    breakpoint_function_secondary=None,
    min_portfolio_size: int = 1,
    cap_weight: float = 0.8,
    data_options: dict = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper that filters a panel and computes portfolio returns.

    Equivalent to calling filter_sorting_data() followed by
    compute_portfolio_returns() with the filter and breakpoint
    specifications bundled in 'portfolio_sort_options'.

    Parameters
    ----------
    data : pd.DataFrame
        Stock-level panel.
    sorting_variables : str or list of str
        Column(s) to sort on.
    sorting_method : str
        'univariate', 'bivariate-dependent', or 'bivariate-independent'.
    portfolio_sort_options : dict
        Dict produced by portfolio_sort_options(). Bundles filter and
        breakpoint specifications.
    rebalancing_month : int, optional
        Month (1-12) for annual rebalancing. None means rebalance
        every period.
    breakpoint_function_main : callable, optional
        Function for primary breakpoints. Defaults to compute_breakpoints.
    breakpoint_function_secondary : callable, optional
        Function for secondary breakpoints. Defaults to
        compute_breakpoints.
    min_portfolio_size : int, default 1
        Minimum firms per reported cross-section.
    cap_weight : float, default 0.8
        Quantile for capping value weights.
    data_options : dict, optional
        Column-name mapping. Uses defaults if None.
    quiet : bool, default False
        If True, suppress informational warnings.

    Returns
    -------
    pd.DataFrame
        Portfolio returns panel as produced by compute_portfolio_returns().
    """
    if not isinstance(quiet, bool):
        raise ValueError("'quiet' must be a single boolean.")

    _data_options_keys = {
        "id",
        "date",
        "exchange",
        "mktcap_lag",
        "ret_excess",
        "portfolio",
        "siccd",
        "price",
        "listing_age",
        "be",
        "earnings",
    }
    if data_options is not None and not (
        isinstance(data_options, dict)
        and _data_options_keys.issubset(data_options.keys())
    ):
        raise ValueError(
            "'data_options' must be None or a dict produced by "
            "data_options()."
        )

    _portfolio_sort_options_keys = {
        "filter_options",
        "breakpoint_options_main",
        "breakpoint_options_secondary",
    }
    if not (
        isinstance(portfolio_sort_options, dict)
        and _portfolio_sort_options_keys.issubset(
            portfolio_sort_options.keys()
        )
    ):
        raise ValueError(
            "'portfolio_sort_options' must be a dict produced by "
            "portfolio_sort_options()."
        )

    filter_opts = portfolio_sort_options["filter_options"]
    breakpoint_options_main = portfolio_sort_options[
        "breakpoint_options_main"
    ]
    breakpoint_options_secondary = portfolio_sort_options[
        "breakpoint_options_secondary"
    ]

    data = filter_sorting_data(
        data,
        filter_options=filter_opts,
        data_options=data_options,
        quiet=quiet,
    )

    return compute_portfolio_returns(
        data,
        sorting_variables=sorting_variables,
        sorting_method=sorting_method,
        rebalancing_month=rebalancing_month,
        breakpoint_options_main=breakpoint_options_main,
        breakpoint_options_secondary=breakpoint_options_secondary,
        breakpoint_function_main=breakpoint_function_main,
        breakpoint_function_secondary=breakpoint_function_secondary,
        min_portfolio_size=min_portfolio_size,
        cap_weight=cap_weight,
        data_options=data_options,
        quiet=quiet,
    )


def estimate_model(
    data: pd.DataFrame,
    model: str,
    min_obs: int = 1,
    output="coefficients",
):
    """Estimate a linear model from a formula string.

    Parses the formula, validates that independent variables exist in
    the data, fits an OLS regression on the complete-case subset, and
    returns the requested output(s).

    Parameters
    ----------
    data : pd.DataFrame
        Data frame containing the dependent variable and one or more
        independent variables referenced in the formula.
    model : str
        Formula string like 'y ~ x1 + x2 + x3' or 'y ~ x1 - 1' for
        no-intercept models.
    min_obs : int, default 1
        Minimum number of complete-case observations required to fit
        the model. Cross-sections below this threshold return NaN.
    output : str or list of str, default 'coefficients'
        One or more of 'coefficients', 'tstats', 'residuals'. If a
        single string is given, the corresponding object is returned
        directly. If a list is given, a dict is returned.

    Returns
    -------
    pd.DataFrame, np.ndarray, or dict
        Coefficients or tstats as a 1-row DataFrame with model term
        names as columns. Residuals as a numeric vector aligned with
        input rows (NaN for rows with missing data or insufficient
        observations). If multiple outputs requested, a dict.
    """
    if isinstance(output, str):
        output_list = [output]
        return_multiple = False
    else:
        output_list = list(output)
        return_multiple = len(output_list) > 1

    valid_outputs = ("coefficients", "tstats", "residuals")
    invalid = [o for o in output_list if o not in valid_outputs]
    if invalid:
        raise ValueError(
            f"'output' must contain one or more of "
            f"{list(valid_outputs)}, not {invalid}."
        )

    if "~" not in model:
        raise ValueError("'model' must contain '~'.")
    parts = model.split("~", 1)
    dep_var = parts[0].strip()
    rhs = parts[1].strip()
    tokens = re.split(r"[\s+]+", rhs)
    independent_vars = [
        t for t in tokens if t and t not in ("-", "1")
    ]

    if "intercept" in independent_vars:
        raise ValueError(
            "None of the columns in 'model' may be called 'intercept'. "
            "Please rename the column and try again."
        )

    missing_vars = [
        v for v in independent_vars if v not in data.columns
    ]
    if missing_vars:
        raise ValueError(
            "The following independent variables are missing in the "
            f"data: {', '.join(missing_vars)}."
        )

    model_vars = [dep_var] + independent_vars
    complete = data[model_vars].notna().all(axis=1)
    n_complete = int(complete.sum())

    insufficient = (n_complete < min_obs) or (
        n_complete <= len(independent_vars)
    )

    fit = None
    if not insufficient:
        try:
            fit = smf.ols(model, data=data[complete]).fit()
        except Exception:
            insufficient = True

    def to_df(series):
        renamed = series.rename({"Intercept": "intercept"})
        return pd.DataFrame(
            [renamed.values], columns=list(renamed.index)
        )

    def na_df():
        if len(independent_vars) == 0:
            return np.nan
        return pd.DataFrame(
            [[np.nan] * len(independent_vars)],
            columns=independent_vars,
        )

    result = {}

    if "coefficients" in output_list:
        if insufficient:
            result["coefficients"] = na_df()
        else:
            result["coefficients"] = to_df(fit.params)

    if "tstats" in output_list:
        if insufficient:
            result["tstats"] = na_df()
        else:
            result["tstats"] = to_df(fit.tvalues)

    if "residuals" in output_list:
        if insufficient:
            result["residuals"] = np.full(len(data), np.nan)
        else:
            resid = np.full(len(data), np.nan)
            resid[complete.values] = fit.resid.values
            result["residuals"] = resid

    if return_multiple:
        return result
    return result[output_list[0]]
