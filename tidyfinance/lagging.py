"""Lagging and rolling-window functions for tidyfinance."""

import numpy as np
import pandas as pd

from ._internal import _check_new_col, _to_offset


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
    """Append lagged columns to a data frame via a join-based approach.

    When 'lag == max_lag' (the default), an equi-join is used: source
    dates are shifted forward by 'lag' and matched exactly. When
    'lag < max_lag', an inequality join is used: for each row, the most
    recent source value within the window '[date - max_lag, date - lag]'
    is selected.

    The combination of 'by' and the date column must be unique in 'data'.
    If 'by' is None, dates alone must be unique.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to lag.
    cols : list of str or str
        Names of the columns to lag. Each column produces a new column
        suffixed with '_lag'.
    lag : int, pd.Timedelta, or pd.DateOffset
        Minimum lag (inclusive) to apply, e.g. 'pd.DateOffset(months=1)'.
        An int is interpreted as days.
    max_lag : int, pd.Timedelta, or pd.DateOffset, optional
        Maximum lag (inclusive) to apply. Defaults to 'lag' (exact lag).
    by : list of str or str, optional
        Grouping column(s) (e.g. a stock identifier). Lagged values are
        matched within groups. Defaults to None.
    drop_na : bool, optional
        If True, NaN values in the source columns are excluded before
        matching, so the lookup skips over missing observations. Applied
        independently per column. Defaults to False.
    ff_adjustment : bool, optional
        If True, only the last observation per year (within each group
        defined by 'by') is retained as a source for lagged values,
        following Fama-French conventions for annual accounting data.
        Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to 'date'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element is
        used to specify the date column. Uses the 'data_options' default
        when None: 'date' -> 'date'.

    Returns
    -------
    pd.DataFrame
        Data frame with the same rows as 'data' and new columns
        appended, each suffixed with '_lag'. Unmatched rows receive NaN
        in the lagged columns.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import add_lagged_columns
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', periods=10, freq='MS')
    data = pd.DataFrame({
        'permno': [1] * 10 + [2] * 10,
        'date': list(dates) * 2,
        'size': rng.uniform(100, 200, 20),
        'bm': rng.uniform(0.5, 1.5, 20),
    })
    # Exact lag: each row gets the value from exactly 2 months earlier
    add_lagged_columns(
        data,
        cols=['size', 'bm'],
        lag=pd.DateOffset(months=2),
        by='permno',
    )
    # Window lag: most recent value from 2 to 4 months earlier
    add_lagged_columns(
        data,
        cols='size',
        lag=pd.DateOffset(months=2),
        max_lag=pd.DateOffset(months=4),
        by='permno',
    )
    ```
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
        raise ValueError(f"'data' must contain the date column '{date_col}'.")

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
        raise ValueError(f"'data' is missing column(s): {missing_cols}.")

    if by_list:
        missing_by = [c for c in by_list if c not in data.columns]
        if missing_by:
            raise ValueError(f"'data' is missing grouping column(s): {missing_by}.")

    join_cols = by_list + [date_col]
    if data[join_cols].duplicated().any():
        raise ValueError(
            "The combination of 'by' and date columns must be unique in 'data'."
        )

    exact_lag = lag_end == max_lag_end
    result = data.copy()

    if not exact_lag:
        _check_new_col(result, ["_upper", "_lower"])
        result["_upper"] = result[date_col] - lag_offset
        result["_lower"] = result[date_col] - max_lag_offset

    for col in cols:
        lag_col_name = f"{col}_lag"
        if lag_col_name in result.columns:
            raise ValueError(f"Column '{lag_col_name}' already exists in 'data'.")

        lagged = data[join_cols + [col]].copy()

        if drop_na:
            lagged = lagged.dropna(subset=[col])

        if ff_adjustment:
            grp_cols = by_list + ["_yr"]
            lagged = lagged.assign(_yr=lagged[date_col].dt.year)
            max_dates = lagged.groupby(grp_cols)[date_col].transform("max")
            lagged = lagged[lagged[date_col] == max_dates].drop(columns="_yr")

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
    lagged = lagged.rename(columns={date_col: "_src_date", col: lag_col_name})

    left_sorted = result.sort_values("_upper", kind="mergesort")
    right_sorted = lagged.sort_values("_src_date", kind="mergesort")

    # Align datetime precision (e.g. ms vs us) for merge_asof.
    left_sorted["_upper"] = pd.to_datetime(left_sorted["_upper"]).astype(
        "datetime64[ns]"
    )
    right_sorted["_src_date"] = pd.to_datetime(right_sorted["_src_date"]).astype(
        "datetime64[ns]"
    )

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="_upper",
        right_on="_src_date",
        by=by_list if by_list else None,
        direction="backward",
    )

    mask = merged["_src_date"].notna() & (merged["_src_date"] >= merged["_lower"])
    merged.loc[~mask, lag_col_name] = np.nan

    merged = merged.sort_values("_orig_idx", kind="mergesort")
    merged = merged.drop(columns=["_orig_idx", "_src_date"]).reset_index(drop=True)
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
    """Join lagged values of variables over a date range.

    Joins lagged values of selected variables from one data frame
    ('new_data') into another ('original_data'), based on date ranges
    defined by 'min_lag' and 'max_lag'. Unlike 'add_lagged_columns',
    this function supports joining across data frames with different
    date grids (e.g. monthly source data into quarterly target data).
    All columns in 'new_data' besides 'id_keys' and the date column are
    lagged and joined under their original names.

    Parameters
    ----------
    original_data : pd.DataFrame
        Target panel data.
    new_data : pd.DataFrame
        Source variables to lag and merge. All columns besides
        'id_keys' and the date column will be lagged and joined.
    id_keys : list of str or str
        Identifier column(s) shared by both frames.
    min_lag : int, pd.Timedelta, or pd.DateOffset
        Lower lag bound (inclusive).
    max_lag : int, pd.Timedelta, or pd.DateOffset
        Upper lag bound (inclusive).
    ff_adjustment : bool, optional
        If True, keeps only the last observation per identifier and
        year in 'new_data' before lagging (Fama-French convention).
        Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to 'date'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element is
        used to identify the date column. Uses the 'data_options'
        default when None: 'date' -> 'date'.

    Returns
    -------
    pd.DataFrame
        'original_data' with all columns from 'new_data' appended as
        lagged values (keeping their original names).

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import join_lagged_values
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=6, freq='MS')
    df1 = pd.DataFrame({
        'id': [1] * 6 + [2] * 6,
        'date': list(dates) * 2,
    })
    df2 = df1.copy()
    df2['x'] = rng.standard_normal(len(df2))
    join_lagged_values(
        original_data=df1,
        new_data=df2,
        id_keys='id',
        min_lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
    )
    ```
    """
    if data_options is not None:
        date_col = data_options.get("date", date_col)

    if isinstance(id_keys, str):
        id_keys = [id_keys]
    if not isinstance(id_keys, list) or not all(isinstance(k, str) for k in id_keys):
        raise ValueError("'id_keys' must be a string or list of strings.")

    min_lag_offset = _to_offset(min_lag)
    max_lag_offset = _to_offset(max_lag)

    if date_col not in original_data.columns:
        raise ValueError(f"'original_data' must contain the column '{date_col}'.")
    if date_col not in new_data.columns:
        raise ValueError(f"'new_data' must contain the column '{date_col}'.")

    missing_original = [k for k in id_keys if k not in original_data.columns]
    if missing_original:
        raise ValueError(
            f"'original_data' is missing id column(s): {missing_original}."
        )

    missing_new = [k for k in id_keys if k not in new_data.columns]
    if missing_new:
        raise ValueError(f"'new_data' is missing id column(s): {missing_new}.")

    new_column_names = [c for c in new_data.columns if c not in id_keys + [date_col]]
    if not new_column_names:
        raise ValueError(
            f"'new_data' must contain columns besides {id_keys} and " f"'{date_col}'."
        )

    original_non_key = [
        c for c in original_data.columns if c not in id_keys + [date_col]
    ]
    duplicate_cols = [c for c in new_column_names if c in original_non_key]
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
            tmp = tmp[tmp[date_col] == max_dates].drop(columns=[date_col, "_year"])
        else:
            tmp = tmp.drop(columns=[date_col])

        _check_new_col(result, "_orig_idx")
        result = result.assign(_orig_idx=np.arange(len(result)))
        # pd.merge_asof requires the merge key (date) sorted globally
        # in ascending order. The by= argument handles the grouping
        # itself, so we only sort by the merge key.
        left_sorted = result.sort_values(date_col, kind="mergesort")
        right_sorted = tmp.sort_values("_lower", kind="mergesort")

        # Align datetime precision (e.g. ms vs us) for merge_asof.
        left_sorted[date_col] = pd.to_datetime(left_sorted[date_col]).astype(
            "datetime64[ns]"
        )
        right_sorted["_lower"] = pd.to_datetime(right_sorted["_lower"]).astype(
            "datetime64[ns]"
        )

        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on=date_col,
            right_on="_lower",
            by=id_keys if id_keys else None,
            direction="backward",
        )

        mask = merged["_lower"].notna() & (merged[date_col] <= merged["_upper"])
        merged.loc[~mask, col] = np.nan

        merged = merged.sort_values("_orig_idx", kind="mergesort")
        merged = merged.drop(columns=["_orig_idx", "_lower", "_upper"]).reset_index(
            drop=True
        )
        result = merged

    return result


def compute_rolling_value(
    data: pd.DataFrame,
    f,
    period: str = "month",
    periods: int = 12,
    min_obs: int = None,
    data_options: dict = None,
) -> np.ndarray:
    """Compute a rolling value by period.

    Applies an arbitrary summary function over rolling time-period
    windows. Each window spans 'periods' units of 'period' (e.g., 12
    months). Before calling 'f', rows with any missing values are
    dropped from the window; if fewer than 'min_obs' rows remain, the
    result is NaN instead.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame with a datetime column named according to
        'data_options[date]' (default 'date').
    f : callable
        Function applied to each window. Receives a data frame slice
        (complete cases only) and must return a single scalar value.
    period : str, default 'month'
        Calendar period unit for the rolling windows. One of 'month',
        'quarter', or 'year'.
    periods : int, default 12
        Number of periods to include in the rolling window.
    min_obs : int, optional
        Minimum number of non-missing rows required per window.
        Defaults to 'periods'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element
        is used to specify the date column. Uses the 'data_options'
        default when None: 'date' -> 'date'.

    Returns
    -------
    np.ndarray
        Numeric vector aligned with the rows of 'data'.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import compute_rolling_value
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=24, freq='MS'),
        'value': rng.standard_normal(24),
    })
    df['rolling_sd'] = compute_rolling_value(
        df,
        f=lambda x: x['value'].std(ddof=1),
        period='month',
        periods=4,
        min_obs=2,
    )
    ```
    """
    if data_options is None:
        data_options = {"date": "date"}
    if min_obs is None:
        min_obs = periods

    date_col = data_options.get("date")

    if not isinstance(date_col, str):
        raise ValueError("'date' in data_options must be a single non-missing string.")

    if date_col not in data.columns:
        raise ValueError(f"'data' must contain a '{date_col}' column.")

    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        raise ValueError(f"The '{date_col}' column must be of datetime dtype.")

    if not isinstance(period, str):
        raise ValueError("'period' must be a single string.")

    period_freq_map = {"month": "M", "quarter": "Q", "year": "Y"}
    if period not in period_freq_map:
        raise ValueError("'period' must be one of 'month', 'quarter', 'year'.")

    buckets = data[date_col].dt.to_period(period_freq_map[period])

    n = len(data)
    result = np.full(n, np.nan)

    for i in range(n):
        anchor_bucket = buckets.iloc[i]
        start_bucket = anchor_bucket - (periods - 1)
        in_window = (buckets >= start_bucket) & (buckets <= anchor_bucket)
        window_data = data[in_window].dropna()
        if len(window_data) >= min_obs:
            result[i] = f(window_data)

    return result
