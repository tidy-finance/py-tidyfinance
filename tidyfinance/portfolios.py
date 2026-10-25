"""Portfolio sorting and return functions for tidyfinance."""

import inspect
import numbers
import warnings

import numpy as np
import polars as pl

from ._internal import (
    _check_new_col,
    _quantile,
    _user_stacklevel,
    _validate_column_name,
    _validate_flag,
    _validate_optional_number,
    _warn_near_option_names,
)


def data_options(
    id: str = "permno",
    date: str = "date",
    exchange: str = "exchange",
    mktcap_lag: str = "mktcap_lag",
    ret_excess: str = "ret_excess",
    portfolio: str = "portfolio",
    siccd: str = "siccd",
    price: str = "prc_lag",
    listing_age: str = "listing_age",
    be: str = "be",
    earnings: str = "ib",
    **kwargs,
) -> dict:
    """Create data options for tidyfinance functions.

    Creates a dict of data options used by tidyfinance-related
    functions. These options map the specific data variables to the
    Tidy Finance naming conventions, allowing functions to flexibly
    work with different datasets by specifying the relevant column
    names. Additional options can be passed through '**kwargs'.

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
    price : str, default "prc_lag"
        Price column used by the 'min_stock_price' screen of
        'filter_sorting_data'. The default is the nominal price at the
        end of the previous month, as returned by
        'download_data("WRDS", "crsp_monthly")'.
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
        Mapping with at least the 11 standard column-name keys plus any
        extras provided via '**kwargs'.

    Examples
    --------
    ```python
    from tidyfinance import data_options
    data_options(id='permno', date='date', exchange='exchange')
    ```
    """
    _validate_column_name(id, "id", "entity")
    _validate_column_name(date, "date", "date")
    _validate_column_name(exchange, "exchange", "exchange")
    _validate_column_name(mktcap_lag, "mktcap_lag", "market capitalization lag")
    _validate_column_name(ret_excess, "ret_excess", "excess return")
    _validate_column_name(portfolio, "portfolio", "portfolio")
    _validate_column_name(
        siccd, "siccd", "Standard Industrial Classification code"
    )
    _validate_column_name(price, "price", "price")
    _validate_column_name(listing_age, "listing_age", "listing age")
    _validate_column_name(be, "be", "book equity")
    _validate_column_name(earnings, "earnings", "earnings")

    _warn_near_option_names(
        kwargs,
        [
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
        ],
        "data_options",
    )

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
    data: pl.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict = None,
    breakpoint_function=None,
    data_options: dict = None,
) -> pl.Series:
    """Assign data points to portfolios based on a sorting variable.

    Users may pass a custom function to compute breakpoints. The
    function must take 'data' and 'sorting_variable' as the first two
    arguments, then 'breakpoint_options' and 'data_options'. The
    function must return an ascending sequence of breakpoints.
    Defaults to compute_breakpoints.

    Parameters
    ----------
    data : pl.DataFrame
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
    pl.Series
        Portfolio assignments as an Int64 series aligned with the
        input rows. Each entry is the 1-indexed portfolio number;
        values outside the breakpoint range fall into the boundary
        portfolios. Missing inputs are returned as null.

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    from tidyfinance import assign_portfolio, breakpoint_options
    rng = np.random.default_rng(42)
    data = pl.DataFrame({
        'exchange': rng.choice(['NYSE', 'NASDAQ'], 100),
        'market_cap': rng.uniform(1, 100, 100),
    })
    # Quintile portfolios on market_cap with NYSE breakpoints
    assign_portfolio(
        data,
        sorting_variable='market_cap',
        breakpoint_options=breakpoint_options(
            n_portfolios=5,
            breakpoints_exchanges='NYSE',
        ),
    )
    ```
    """
    if breakpoint_function is None:
        breakpoint_function = compute_breakpoints

    if sorting_variable not in data.columns:
        raise ValueError(
            f"Sorting variable '{sorting_variable}' not found in data."
        )

    x = data.get_column(sorting_variable).cast(pl.Float64).fill_nan(None)
    n = len(x)

    # Constant check: count of distinct non-missing values.
    non_na = x.drop_nulls().to_numpy()
    if len(np.unique(non_na)) <= 1:
        warnings.warn(
            "The sorting variable is constant and only one portfolio "
            "is returned.",
            UserWarning,
            stacklevel=2,
        )
        return pl.Series("portfolio", [1] * n, dtype=pl.Int64)

    breakpoints = breakpoint_function(
        data, sorting_variable, breakpoint_options, data_options
    )

    breakpoints = np.asarray(breakpoints, dtype=float)
    if np.any(np.isnan(breakpoints)):
        warnings.warn(
            "No portfolios were assigned due to missing breakpoints.",
            UserWarning,
            stacklevel=2,
        )
        return pl.Series("portfolio", [None] * n, dtype=pl.Int64)

    # The outer breakpoint edges are treated as -/+ infinity so values
    # outside the original range still fall into a boundary portfolio.
    # Intervals are closed on the left and open on the right, so the
    # portfolio index is the count of interior breakpoints <= x plus 1.
    xa = x.to_numpy()
    assignments = np.searchsorted(breakpoints[1:-1], xa, side="right") + 1
    result = pl.Series("portfolio", assignments, dtype=pl.Int64)
    result = pl.select(
        pl.when(x.is_null()).then(None).otherwise(result).alias("portfolio")
    ).to_series()

    # Cluster warning: number of populated portfolios differs from
    # the expected count (ties collapsed adjacent breakpoints).
    n_expected = len(breakpoints) - 1
    n_actual = result.drop_nulls().n_unique()
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
    """Create breakpoint options for portfolio sorting.

    Generates a structured dict of options for defining breakpoints in
    portfolio sorting. It includes parameters for the number of
    portfolios, percentile thresholds, exchange-specific breakpoints,
    and smooth bunching, along with additional optional parameters.

    Parameters
    ----------
    n_portfolios : int, optional
        Number of portfolios to create. Must be a positive integer. If
        not provided, defaults to None.
    percentiles : list of float, optional
        Percentile thresholds for defining breakpoints. Each value must
        be between 0 and 1. If not provided, defaults to None.
    breakpoints_exchanges : str or list of str, optional
        Non-empty exchange (or list of exchanges) from which to compute
        the breakpoints. If not provided, defaults to None.
    smooth_bunching : bool, default False
        Indicates whether smooth bunching should be applied.
    breakpoints_min_size_threshold : float, optional
        When set to a value between 0 and 1, stocks with market
        capitalization below this quantile are excluded from breakpoint
        computation. The quantile is computed among
        'breakpoints_exchanges' stocks if specified, otherwise among
        all stocks. Requires a market capitalization column in the data
        (see 'data_options'). Defaults to None (no size filtering).
    **kwargs
        Additional optional arguments. These will be captured in the
        resulting structure as part of the dict.

    Returns
    -------
    dict
        Dictionary containing the provided breakpoint options,
        including any additional arguments passed via '**kwargs'.

    Examples
    --------
    ```python
    from tidyfinance import breakpoint_options
    # Quintile portfolios with NYSE breakpoints
    breakpoint_options(
        n_portfolios=5,
        breakpoints_exchanges='NYSE',
    )
    # Custom percentile thresholds (mutually exclusive with n_portfolios)
    breakpoint_options(
        percentiles=[0.3, 0.7],
        breakpoints_exchanges='NYSE',
    )
    # Exclude the smallest 20% of NYSE stocks from breakpoint computation
    breakpoint_options(
        n_portfolios=10,
        breakpoints_exchanges='NYSE',
        breakpoints_min_size_threshold=0.2,
    )
    ```
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

    _warn_near_option_names(
        kwargs,
        [
            "n_portfolios",
            "percentiles",
            "breakpoints_exchanges",
            "smooth_bunching",
            "breakpoints_min_size_threshold",
        ],
        "breakpoint_options",
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
    data: pl.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict,
    data_options: dict = None,
) -> np.ndarray:
    """Compute breakpoints based on a sorting variable.

    Computes breakpoints based on a specified sorting variable. It can
    optionally filter the data by exchanges or lagged size quantiles
    before computing the breakpoints. The function requires either the
    number of portfolios to be created or specific percentiles for the
    breakpoints, but not both. The function also optionally handles
    cases where the sorting variable clusters on the edges, by
    assigning all extreme values to the edges and attempting to compute
    equally populated breakpoints with the remaining values.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame with the dataset for breakpoint computation.
    sorting_variable : str
        Column name in 'data' to be used for determining breakpoints.
    breakpoint_options : dict
        Named dict of 'breakpoint_options' for the breakpoints. The
        accepted entries include:

        - 'n_portfolios' (int, optional): Number of equally sized
          portfolios to create. Mutually exclusive with 'percentiles'.
        - 'percentiles' (list of float, optional): Percentiles defining
          the breakpoints of the portfolios. Mutually exclusive with
          'n_portfolios'.
        - 'breakpoints_exchanges' (str or list of str, optional):
          Exchange names to filter the data before computing
          breakpoints. Exchanges must be stored in the column given by
          'data_options' (defaults to 'exchange'). If None, no
          filtering is applied.
        - 'smooth_bunching' (bool, optional): Whether to attempt
          smoothing non-extreme portfolios if the sorting variable
          bunches on the extremes (True) or not (False, the default).
          In some cases, smoothing will not result in equal-sized
          portfolios off the edges due to multiple clusters. If
          sufficiently large bunching is detected, 'percentiles' is
          ignored and equally-spaced portfolios are returned for these
          cases with a warning.
        - 'breakpoints_min_size_threshold' (float, optional): Value
          between 0 and 1 (exclusive). When set, stocks with market
          capitalization below this quantile are excluded from
          breakpoint computation. The quantile is computed among
          'breakpoints_exchanges' stocks if specified, otherwise among
          all stocks. Requires a market capitalization column in the
          data (column name determined by 'data_options').

    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'exchange' key is
        used to specify the exchange column, and 'mktcap_lag' is used
        to specify the market capitalization column. Uses the
        'data_options' default when None: 'exchange' -> 'exchange' and
        'mktcap_lag' -> 'mktcap_lag'.

    Returns
    -------
    np.ndarray
        Sorted array of breakpoints of the desired length.

    Notes
    -----
    This function raises a ValueError if both 'n_portfolios' and
    'percentiles' are provided or missing simultaneously.

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    from tidyfinance import compute_breakpoints, breakpoint_options
    rng = np.random.default_rng(42)
    data = pl.DataFrame({
        'id': range(1, 101),
        'exchange': rng.choice(['NYSE', 'NASDAQ'], 100),
        'market_cap': range(1, 101),
    })
    compute_breakpoints(
        data, 'market_cap', breakpoint_options(n_portfolios=5)
    )
    compute_breakpoints(
        data,
        'market_cap',
        breakpoint_options(
            percentiles=[0.2, 0.4, 0.6, 0.8],
            breakpoints_exchanges=['NYSE'],
        ),
    )
    ```
    """
    if not isinstance(breakpoint_options, dict):
        raise ValueError("Please provide a dictionary with breakpoint options.")

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
            "Please provide either 'n_portfolios' or 'percentiles', not both."
        )
    if n_portfolios is None and percentiles is None:
        raise ValueError(
            "You must provide either 'n_portfolios' or 'percentiles'."
        )

    sorting_all = (
        data.get_column(sorting_variable).cast(pl.Float64).to_numpy()
    )
    sorting_values = sorting_all

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
        keep_mask = (
            data.get_column(exchange_col)
            .is_in(exchanges_list)
            .fill_null(False)
            .to_numpy()
        )
        sorting_values = sorting_all[keep_mask]

    # An absolute minimum market cap is set internally by dependent
    # bivariate sorts in compute_portfolio_returns(), which compute it
    # once per date from the full cross-section instead of within each
    # secondary portfolio.
    min_mktcap = breakpoint_options.get("breakpoints_min_mktcap")

    if breakpoints_min_size_threshold is not None or min_mktcap is not None:
        if min_mktcap is None:
            min_mktcap = _compute_min_mktcap(
                data,
                breakpoints_min_size_threshold,
                breakpoints_exchanges,
                data_options,
            )
        mktcap_col = data_options["mktcap_lag"]
        if mktcap_col not in data.columns:
            raise ValueError(
                f"Column '{mktcap_col}' is required when using "
                "'breakpoints_min_size_threshold'."
            )
        all_mktcap = (
            data.get_column(mktcap_col).cast(pl.Float64).to_numpy()
        )
        above_size = (~np.isnan(all_mktcap)) & (all_mktcap > min_mktcap)
        combined_mask = (
            keep_mask & above_size if keep_mask is not None else above_size
        )
        sorting_values = sorting_all[combined_mask]

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

    sorting_values_clean = sorting_values[~np.isnan(sorting_values)]
    breakpoints = _quantile(sorting_values_clean, probs)

    if smooth_bunching:
        both_edges = (
            breakpoints[0] == breakpoints[1]
            and breakpoints[n_portfolios - 1] == breakpoints[n_portfolios]
        )
        lower_edge = breakpoints[0] == breakpoints[1]
        upper_edge = breakpoints[n_portfolios - 1] == breakpoints[n_portfolios]

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
            breakpoints_new = _quantile(sorting_values_new, probs_new)
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
            breakpoints_new = _quantile(sorting_values_new, probs_new)
            breakpoints = np.concatenate([[breakpoints[0]], breakpoints_new])
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
            breakpoints_new = _quantile(sorting_values_new, probs_new)
            breakpoints_new[-1] += 1e-15
            breakpoints = np.concatenate(
                [breakpoints_new, [breakpoints[n_portfolios]]]
            )

    breakpoints[1:] += 1e-20
    return breakpoints


def _compute_min_mktcap(
    data: pl.DataFrame,
    breakpoints_min_size_threshold: float,
    breakpoints_exchanges,
    data_options: dict,
) -> float:
    """The minimum market cap of 'breakpoints_min_size_threshold'.

    The 'breakpoints_min_size_threshold' quantile of the market caps in
    'data', among the 'breakpoints_exchanges' stocks if specified.
    Returns NaN if no market cap is available, so no stock passes.
    """
    mktcap_col = data_options["mktcap_lag"]
    if mktcap_col not in data.columns:
        raise ValueError(
            f"Column '{mktcap_col}' is required when using "
            "'breakpoints_min_size_threshold'."
        )
    mktcap_ref = data.get_column(mktcap_col).cast(pl.Float64).to_numpy()
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
        keep = (
            data.get_column(exchange_col)
            .is_in(exchanges_list)
            .fill_null(False)
            .to_numpy()
        )
        mktcap_ref = mktcap_ref[keep]
    return _quantile(mktcap_ref, breakpoints_min_size_threshold)


def _main_min_mktcap(
    data: pl.DataFrame,
    breakpoint_options: dict,
    breakpoint_function,
    data_options: dict,
) -> float | None:
    """Minimum market cap of the main breakpoints in dependent sorts.

    In dependent bivariate sorts, the main breakpoints are computed
    within each secondary portfolio. The minimum market cap of
    'breakpoints_min_size_threshold' must nevertheless come from the
    full cross-section of a date, so it is computed here, before
    grouping by the secondary portfolio. Only the default
    'compute_breakpoints' uses it; custom breakpoint functions receive
    their options unchanged.

    Returns None if no minimum applies.
    """
    threshold = (breakpoint_options or {}).get(
        "breakpoints_min_size_threshold"
    )
    if (
        threshold is None
        or inspect.unwrap(breakpoint_function) is not compute_breakpoints
    ):
        return None
    return _compute_min_mktcap(
        data,
        threshold,
        breakpoint_options.get("breakpoints_exchanges"),
        data_options,
    )


def _with_min_mktcap(breakpoint_options: dict, min_mktcap) -> dict:
    """Add an absolute minimum market cap to breakpoint options.

    Returns the options unchanged if 'min_mktcap' is None or NaN.
    """
    if min_mktcap is None or np.isnan(min_mktcap):
        return breakpoint_options
    return {**breakpoint_options, "breakpoints_min_mktcap": min_mktcap}


def filter_options(
    exclude_financials: bool = False,
    exclude_utilities: bool = False,
    min_stock_price: float = None,
    min_size_quantile: float = None,
    min_listing_age: float = None,
    exclude_negative_book_equity: bool = False,
    exclude_negative_earnings: bool = False,
    timing: str = "formation",
    **kwargs,
) -> dict:
    """Create filter options for sample construction.

    Creates a dict of filter options used for sample construction in
    tidyfinance-related functions. These options control which
    observations are retained before portfolio sorting.

    Parameters
    ----------
    exclude_financials : bool, default False
        Whether to exclude financial firms (SIC codes 6000 to 6799).
    exclude_utilities : bool, default False
        Whether to exclude utility firms (SIC codes 4900 to 4999).
    min_stock_price : float, optional
        Minimum stock price required to include an observation. Must
        be strictly positive when provided. None (the default) applies
        no price filter.
    min_size_quantile : float, optional
        Minimum cross-sectional size quantile (based on lagged market
        cap) required to include an observation. Must be strictly
        between 0 and 1 when provided. The cutoff is computed from
        NYSE stocks only; this requires an 'exchange' column in the
        data (as mapped via 'data_options'). None (the default)
        applies no size quantile filter.
    min_listing_age : float, optional
        Minimum number of months a stock must have been listed in
        CRSP. Must be non-negative when provided. None (the default)
        applies no listing age filter.
    exclude_negative_book_equity : bool, default False
        Whether to exclude observations with non-positive book equity.
    exclude_negative_earnings : bool, default False
        Whether to exclude observations with non-positive earnings.
    timing : {'formation', 'every_period'}, default 'formation'
        When the screens apply if portfolios are rebalanced once a year
        ('rebalancing_month' in 'filter_sorting_data' or
        'implement_portfolio_sort'). 'formation' evaluates the screens
        in the formation month only and keeps every stock that passes
        them for the whole holding period (ex-ante screens).
        'every_period' evaluates them in every month, so a stock that
        fails a screen in a holding month loses that month's return
        (ex-post screens). Ex-post screens use information that is not
        available when the portfolios are formed, so the resulting
        factor returns can contain look-ahead bias; a warning is issued
        when they are used. With monthly rebalancing, every month is a
        formation month and both options are identical.
    **kwargs
        Additional optional arguments, stored verbatim in the dict.

    Returns
    -------
    dict
        Dict containing the specified filter options.

    Examples
    --------
    ```python
    from tidyfinance import filter_options
    filter_options(
        exclude_financials=True,
        exclude_utilities=True,
        min_stock_price=1,
        min_listing_age=12,
    )
    # Apply the screens in every month (ex-post), not only at formation
    filter_options(min_stock_price=5, timing='every_period')
    ```
    """
    _validate_flag(exclude_financials, "exclude_financials")
    _validate_flag(exclude_utilities, "exclude_utilities")
    _validate_flag(exclude_negative_book_equity, "exclude_negative_book_equity")
    _validate_flag(exclude_negative_earnings, "exclude_negative_earnings")

    if not isinstance(timing, str) or timing not in (
        "formation",
        "every_period",
    ):
        raise ValueError(
            "timing must be either 'formation' or 'every_period'."
        )

    _validate_optional_number(
        min_stock_price,
        "min_stock_price must be a single positive numeric.",
        min=0,
        min_strict=True,
    )
    _validate_optional_number(
        min_size_quantile,
        "min_size_quantile must be a single numeric strictly between 0 and 1.",
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

    _warn_near_option_names(
        kwargs,
        [
            "exclude_financials",
            "exclude_utilities",
            "min_stock_price",
            "min_size_quantile",
            "min_listing_age",
            "exclude_negative_book_equity",
            "exclude_negative_earnings",
            "timing",
        ],
        "filter_options",
    )

    return {
        "exclude_financials": exclude_financials,
        "exclude_utilities": exclude_utilities,
        "min_stock_price": min_stock_price,
        "min_size_quantile": min_size_quantile,
        "min_listing_age": min_listing_age,
        "exclude_negative_book_equity": exclude_negative_book_equity,
        "exclude_negative_earnings": exclude_negative_earnings,
        "timing": timing,
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
    """Create portfolio sort options.

    Creates a dict of options that bundles sample construction filters
    and breakpoint specifications for use with
    'implement_portfolio_sort'.

    Parameters
    ----------
    filter_options : dict, optional
        Dict produced by 'filter_options', or None (the default, which
        applies no filters). The accepted entries include:

        - 'exclude_financials' (bool): Whether to exclude financial
          firms (SIC codes 6000 to 6799). Defaults to False.
        - 'exclude_utilities' (bool): Whether to exclude utility firms
          (SIC codes 4900 to 4999). Defaults to False.
        - 'min_stock_price' (float, optional): Minimum stock price
          required to include an observation. None (the default)
          applies no price filter.
        - 'min_size_quantile' (float, optional): Minimum cross-sectional
          size quantile (based on lagged market cap) required to
          include an observation. None (the default) applies no size
          quantile filter.
        - 'min_listing_age' (float, optional): Minimum number of months
          a stock must have been listed in CRSP. None (the default)
          applies no listing age filter.
        - 'exclude_negative_book_equity' (bool): Whether to exclude
          observations with non-positive book equity. Defaults to False.
        - 'exclude_negative_earnings' (bool): Whether to exclude
          observations with non-positive earnings. Defaults to False.
        - 'timing' (str): Either 'formation' (the default) to apply the
          screens only in the formation month when portfolios are
          rebalanced once a year, or 'every_period' to apply them in
          every month, which can introduce look-ahead bias.

    breakpoint_options_main : dict, optional
        Dict produced by 'breakpoint_options', specifying breakpoints
        for the primary sorting variable, or None (the default) when
        no primary breakpoints are required. The accepted entries
        include:

        - 'n_portfolios' (int, optional): Number of equally sized
          portfolios. Mutually exclusive with 'percentiles'.
        - 'percentiles' (list of float, optional): Percentiles for
          defining the breakpoints. Mutually exclusive with
          'n_portfolios'.
        - 'breakpoints_exchanges' (str or list of str, optional):
          Exchange names to filter the data before computing
          breakpoints. If None, no filtering is applied.
        - 'smooth_bunching' (bool, optional): Whether to attempt
          smoothing non-extreme portfolios if the sorting variable
          bunches on the extremes.
        - 'breakpoints_min_size_threshold' (float, optional): Value
          between 0 and 1 (exclusive) below which stocks are excluded
          from breakpoint computation.

    breakpoint_options_secondary : dict, optional
        Dict produced by 'breakpoint_options', specifying breakpoints
        for the secondary sorting variable, or None (the default) for
        univariate sorts. The accepted entries are the same as for
        'breakpoint_options_main'.
    **kwargs
        Additional optional arguments, stored verbatim in the dict.

    Returns
    -------
    dict
        Dict containing the specified options.

    Examples
    --------
    ```python
    from tidyfinance import (
        portfolio_sort_options,
        filter_options,
        breakpoint_options,
    )
    portfolio_sort_options(
        filter_options=filter_options(exclude_financials=True),
        breakpoint_options_main=breakpoint_options(n_portfolios=10),
    )
    ```
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
        and _BREAKPOINT_OPTIONS_KEYS.issubset(breakpoint_options_main.keys())
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
    data: pl.DataFrame,
    group_cols: list,
    ret_col: str,
    w_col: str,
    w_capped_col: str,
    min_portfolio_size: int,
) -> pl.DataFrame:
    """Compute vw, ew, and vw_capped returns within groups.

    Groups with fewer than min_portfolio_size observations or with a
    missing return get null in all three return columns. Groups whose
    weight sum is zero get null in the corresponding weighted return.
    Rows with a null value in any group column are excluded.

    Parameters
    ----------
    data : pl.DataFrame
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
        Minimum observations per group; groups below this size get
        null.

    Returns
    -------
    pl.DataFrame
        One row per group with columns from group_cols plus
        ret_excess_vw, ret_excess_ew, and ret_excess_vw_capped.
    """
    work = data.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in group_cols])
    )

    sums = work.group_by(group_cols, maintain_order=True).agg(
        _n=pl.len(),
        _n_ret_null=pl.col(ret_col).is_null().sum(),
        _r_sum=pl.col(ret_col).sum(),
        _w_sum=pl.col(w_col).sum(),
        _wc_sum=pl.col(w_capped_col).sum(),
        _rw_sum=(pl.col(ret_col) * pl.col(w_col)).sum(),
        _rwc_sum=(pl.col(ret_col) * pl.col(w_capped_col)).sum(),
    )

    sums = sums.with_columns(
        ret_excess_ew=pl.col("_r_sum") / pl.col("_n"),
        ret_excess_vw=pl.when(pl.col("_w_sum") == 0)
        .then(None)
        .otherwise(pl.col("_rw_sum") / pl.col("_w_sum")),
        ret_excess_vw_capped=pl.when(pl.col("_wc_sum") == 0)
        .then(None)
        .otherwise(pl.col("_rwc_sum") / pl.col("_wc_sum")),
    )

    # A missing return makes the portfolio return missing rather than
    # being treated as zero
    no_return = (pl.col("_n") < min_portfolio_size) | (
        pl.col("_n_ret_null") > 0
    )
    sums = sums.with_columns(
        [
            pl.when(no_return).then(None).otherwise(pl.col(c)).alias(c)
            for c in (
                "ret_excess_vw",
                "ret_excess_ew",
                "ret_excess_vw_capped",
            )
        ]
    )

    return sums.select(
        list(group_cols)
        + ["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"]
    )


def _aggregate_bivariate_returns(
    portfolio_returns: pl.DataFrame,
    date_col: str,
    ret_col: str,
    w_col: str,
    w_capped_col: str,
    min_portfolio_size: int,
    by: list | None = None,
) -> pl.DataFrame:
    """Aggregate bivariate-sort returns across the secondary dimension.

    Computes cell-level returns over (portfolio_main,
    portfolio_secondary, date) without an occupancy threshold, then
    averages across the secondary buckets to obtain reported
    (portfolio_main, date) returns. min_portfolio_size is applied to
    the per-(portfolio_main, date) firm count.

    Parameters
    ----------
    portfolio_returns : pl.DataFrame
        Panel with columns portfolio_main, portfolio_secondary, the
        date column, and per-stock returns/weights.
    date_col, ret_col, w_col, w_capped_col : str
        Column names.
    min_portfolio_size : int
        Minimum firms per reported (portfolio_main, date) cross-section.
        Cross-sections below this size receive null.
    by : list of str, optional
        Additional grouping columns, e.g. the cohort of staggered
        portfolios.

    Returns
    -------
    pl.DataFrame
        Columns portfolio, the date column, the 'by' columns, and the
        three return columns.
    """
    by = list(by or [])

    n_per_main = (
        portfolio_returns.filter(
            pl.col("portfolio_main").is_not_null()
            & pl.col("portfolio_secondary").is_not_null()
        )
        .group_by(["portfolio_main", date_col] + by, maintain_order=True)
        .agg(n_firms=pl.len())
    )

    cell_returns = _summarise_portfolio_returns(
        portfolio_returns,
        ["portfolio_main", "portfolio_secondary", date_col] + by,
        ret_col,
        w_col,
        w_capped_col,
        min_portfolio_size=0,
    )

    avg_returns = (
        cell_returns.group_by(
            ["portfolio_main", date_col] + by, maintain_order=True
        )
        .agg(
            ret_excess_vw=pl.col("ret_excess_vw").mean(),
            ret_excess_ew=pl.col("ret_excess_ew").mean(),
            ret_excess_vw_capped=pl.col("ret_excess_vw_capped").mean(),
        )
        .rename({"portfolio_main": "portfolio"})
    )

    result = avg_returns.join(
        n_per_main.rename({"portfolio_main": "portfolio"}),
        on=["portfolio", date_col] + by,
        how="left",
        maintain_order="left",
    )

    insufficient = pl.col("n_firms").is_null() | (
        pl.col("n_firms") < min_portfolio_size
    )
    result = result.with_columns(
        [
            pl.when(insufficient).then(None).otherwise(pl.col(c)).alias(c)
            for c in (
                "ret_excess_vw",
                "ret_excess_ew",
                "ret_excess_vw_capped",
            )
        ]
    )

    return result.drop("n_firms")


def _compute_staggered_returns(
    assignments: pl.DataFrame,
    holding_data: pl.DataFrame,
    sorting_method: str,
    holding_period: int,
    skip: int,
    dates: pl.Series,
    id_col: str,
    date_col: str,
    ret_col: str,
    w_col: str,
    w_capped_col: str,
    min_portfolio_size: int,
) -> pl.DataFrame:
    """Compute returns of staggered (overlapping) portfolios.

    Holds each cohort of portfolios formed on a date for
    'holding_period' dates, starting 'skip' dates after formation. The
    return of a portfolio on a date is the equal-weighted average of
    the returns of the cohorts held on that date (Jegadeesh and Titman,
    1993). Each cohort's return is computed from 'holding_data', so a
    stock stays in its cohort even if it lacks a sorting variable or
    fails a screen in a holding month.

    Parameters
    ----------
    assignments : pl.DataFrame
        Portfolio assignments with the id and date columns and
        'portfolio' (univariate) or 'portfolio_main' and
        'portfolio_secondary' (bivariate).
    holding_data : pl.DataFrame
        The full panel with returns and (capped) weights.
    sorting_method, holding_period, skip, min_portfolio_size
        See 'compute_portfolio_returns'.
    dates : pl.Series
        The sorted dates of the panel. Holding dates are counted on
        this grid, so it must not have gaps.
    id_col, date_col, ret_col, w_col, w_capped_col : str
        Column names.

    Returns
    -------
    pl.DataFrame
        Columns portfolio, the date column, and the three return
        columns.
    """
    assignments = _check_new_col(
        assignments, ["_cohort", "_index", "_offset"], "compute_portfolio_returns"
    )
    grid = (
        pl.DataFrame({date_col: dates})
        .with_row_index("_index")
        .with_columns(pl.col("_index").cast(pl.Int64))
    )
    offsets = pl.DataFrame(
        {"_offset": list(range(skip, skip + holding_period))},
        schema={"_offset": pl.Int64},
    )

    # Expand each assignment to its holding dates on the date grid
    held = (
        assignments.join(grid, on=date_col, how="inner")
        .rename({date_col: "_cohort"})
        .join(offsets, how="cross")
        .with_columns((pl.col("_index") + pl.col("_offset")).alias("_index"))
        .drop("_offset")
        .join(grid, on="_index", how="inner")
        .drop("_index")
    )

    cohort_returns = held.join(
        holding_data.select([id_col, date_col, ret_col, w_col, w_capped_col]),
        on=[id_col, date_col],
        how="inner",
    )

    if sorting_method == "univariate":
        cohort_returns = _summarise_portfolio_returns(
            cohort_returns,
            ["portfolio", "_cohort", date_col],
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
        )
    else:
        cohort_returns = _aggregate_bivariate_returns(
            cohort_returns,
            date_col,
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
            by=["_cohort"],
        )

    return cohort_returns.group_by(
        ["portfolio", date_col], maintain_order=True
    ).agg(
        pl.col("ret_excess_vw").mean(),
        pl.col("ret_excess_ew").mean(),
        pl.col("ret_excess_vw_capped").mean(),
    )


def _join_rebalanced_portfolios(
    data: pl.DataFrame,
    portfolio_data: pl.DataFrame,
    date_col: str,
    id_col: str,
    rebalancing_month: int,
) -> pl.DataFrame:
    """Join annual portfolio assignments to all dates in rebalancing window.

    Each date in 'data' is matched to the most recent portfolio
    assignment in 'portfolio_data' for the same id, using the
    12-month window starting at the rebalancing month.

    Parameters
    ----------
    data : pl.DataFrame
        Full stock-level panel.
    portfolio_data : pl.DataFrame
        Portfolio assignments at rebalancing dates. Must contain
        id_col, date_col, and one or more portfolio columns.
    date_col, id_col : str
        Date and stock-identifier column names.
    rebalancing_month : int
        The month (1-12) when portfolios are rebalanced annually.

    Returns
    -------
    pl.DataFrame
        'data' with the portfolio columns joined from 'portfolio_data'
        (null where no rebalancing window matches).
    """
    window_year = (
        pl.when(pl.col(date_col).dt.month() >= rebalancing_month)
        .then(pl.col(date_col).dt.year())
        .otherwise(pl.col(date_col).dt.year() - 1)
        .alias("_window_year")
    )

    data_w = data.with_columns(window_year)
    pd_w = portfolio_data.with_columns(
        pl.col(date_col).dt.year().alias("_window_year")
    ).drop(date_col)

    merged = data_w.join(
        pd_w, on=[id_col, "_window_year"], how="left", maintain_order="left"
    )
    return merged.drop("_window_year")


def _cap_market_cap(
    data: pl.DataFrame,
    w_col: str,
    w_capped_col: str,
    date_col: str,
    exchange_col: str,
    cap_weight: float,
    cap_weight_exchanges,
    quiet: bool,
    cap_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Cap market capitalization per date.

    Caps 'w_col' at the 'cap_weight' percentile of the stocks listed on
    'cap_weight_exchanges' on each date, or of all stocks if
    'cap_weight_exchanges' is None, the exchange column is missing, or
    no stock with market capitalization is listed on these exchanges on
    a date. The result is stored in 'w_capped_col'.

    The percentile is computed from 'cap_data' (default: 'data'), e.g.
    the formation universe when the weights of held stocks are capped.
    Missing market caps in 'cap_data' must still be null, not zero.
    Dates of 'data' without rows in 'cap_data' are capped from 'data'
    itself.
    """
    if cap_data is None:
        cap_data = data

    if (
        cap_weight_exchanges is not None
        and exchange_col not in cap_data.columns
    ):
        if not quiet:
            warnings.warn(
                f"Column '{exchange_col}' not found: capping market "
                "capitalization at the percentile of all stocks. Add the "
                "exchange column or set 'cap_weight_exchanges=None' to "
                "silence this message.",
                UserWarning,
                stacklevel=_user_stacklevel(),
            )
        cap_weight_exchanges = None

    exchanges = None
    if cap_weight_exchanges is not None:
        exchanges = (
            [cap_weight_exchanges]
            if isinstance(cap_weight_exchanges, str)
            else list(cap_weight_exchanges)
        )

    def compute_caps(frame: pl.DataFrame) -> pl.DataFrame:
        if exchanges is None or exchange_col not in frame.columns:
            reference = pl.lit(True)
        else:
            reference = pl.col(exchange_col).is_in(exchanges).fill_null(False)
        frame = frame.select(
            pl.col(date_col),
            pl.col(w_col).cast(pl.Float64),
            reference.alias("_reference"),
        )
        rows = []
        for (date,), part in frame.group_by(date_col, maintain_order=True):
            w = part.get_column(w_col).to_numpy()
            is_reference = part.get_column("_reference").to_numpy()
            has_reference = bool((is_reference & ~np.isnan(w)).any())
            cap = _quantile(
                w[is_reference] if has_reference else w, cap_weight
            )
            rows.append((date, has_reference, None if np.isnan(cap) else cap))
        return pl.DataFrame(
            rows,
            schema={
                date_col: frame.schema[date_col],
                "_has_reference": pl.Boolean,
                "_cap": pl.Float64,
            },
            orient="row",
        )

    caps = compute_caps(cap_data)

    n_fallback = caps.filter(~pl.col("_has_reference")).height
    if not quiet and cap_weight_exchanges is not None and n_fallback > 0:
        names = ", ".join(f"'{e}'" for e in exchanges)
        dates = "that date" if n_fallback == 1 else "these dates"
        warnings.warn(
            f"No {names} stocks with market capitalization on "
            f"{n_fallback} date{'' if n_fallback == 1 else 's'}: capping "
            f"at the percentile of all stocks on {dates}.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    # Dates of data without rows in cap_data are capped from data itself
    missing_dates = data.join(caps.select(date_col), on=date_col, how="anti")
    if missing_dates.height > 0:
        caps = pl.concat([caps, compute_caps(missing_dates)])

    caps = caps.select([date_col, "_cap"])

    data = _check_new_col(data, "_cap", "compute_portfolio_returns")
    return (
        data.join(caps, on=date_col, how="left", maintain_order="left")
        .with_columns(
            pl.min_horizontal(pl.col(w_col), pl.col("_cap")).alias(
                w_capped_col
            )
        )
        .drop("_cap")
    )


def compute_portfolio_returns(
    data: pl.DataFrame,
    sorting_variables,
    sorting_method: str,
    rebalancing_month: int = None,
    holding_period: int = 1,
    skip: int = 0,
    breakpoint_options_main: dict = None,
    breakpoint_options_secondary: dict = None,
    breakpoint_function_main=None,
    breakpoint_function_secondary=None,
    min_portfolio_size: int = 1,
    cap_weight: float = 0.8,
    cap_weight_exchanges="NYSE",
    data_options: dict = None,
    quiet: bool = False,
) -> pl.DataFrame:
    """Compute portfolio returns.

    Computes individual portfolio returns based on specified sorting
    variables and sorting methods. The portfolios can be rebalanced
    every period or on an annual frequency by specifying a rebalancing
    month, which is only applicable at a monthly return frequency. The
    function supports univariate and bivariate sorts, with the latter
    supporting dependent and independent sorting methods.

    Holding periods and skips are counted on the grid of dates in
    'data', which must therefore not have gaps (e.g., one row per
    stock-month in a monthly panel). With staggered rebalancing
    ('holding_period > 1' or 'skip > 0'), the returned panel starts on
    the first date on which all 'holding_period' cohorts are held.

    The function checks for consistency in the provided arguments. For
    univariate sorts, a single sorting variable and a corresponding
    number of portfolios must be provided. For bivariate sorts, two
    sorting variables and two corresponding numbers of portfolios (or
    percentiles) are required. The sorting method determines how
    portfolios are assigned and how returns are computed. The function
    handles missing and extreme values appropriately based on the
    specified sorting method and rebalancing frequency.

    Parameters
    ----------
    data : pl.DataFrame
        Stock-level panel. Must contain the id, date, and excess
        return columns (configurable via 'data_options'), the sorting
        variable(s), and optionally a market-cap lag column for
        value-weighted returns.
    sorting_variables : str or list of str
        Column name(s) in 'data' to use for sorting and portfolio
        assignment. For univariate sorts, provide a single variable.
        For bivariate sorts, provide two variables, where the first
        string refers to the main variable and the second string
        refers to the secondary ('control') variable.
    sorting_method : {'univariate', 'bivariate-dependent',
                      'bivariate-independent'}
        Sorting method to use. For bivariate sorts, the portfolio
        returns are averaged over the controlling sorting variable
        (i.e., the second sorting variable), and only portfolio
        returns for the main sorting variable are returned.
    rebalancing_month : int, optional
        Integer between 1 and 12 specifying the month in which to form
        portfolios that are held constant for one year. For example,
        setting it to 7 creates portfolios in July that are held
        constant until June of the following year. The default None
        corresponds to periodic rebalancing.
    holding_period : int, default 1
        Number of periods for which portfolios are held. With the
        default 1, portfolios are rebalanced every period. With
        'holding_period=K > 1', portfolios are formed every period and
        each cohort is held for K periods, so that K overlapping
        cohorts are held at any time (staggered rebalancing, as in
        Jegadeesh and Titman, 1993). The return of a portfolio in a
        period is the equal-weighted average of the returns of the
        cohorts held in that period; within a cohort, returns are
        weighted as in the 'ret_excess_*' columns, using the weights of
        the holding period. A stock stays in its cohort for the whole
        holding period even if its sorting variables are missing in a
        later period. Cannot be combined with 'rebalancing_month'.
    skip : int, default 0
        Number of periods between portfolio formation and the first
        holding period. The default 0 holds a cohort from the period in
        which it is formed, consistent with sorting variables that are
        already lagged (i.e., known at the start of the period).
        Jegadeesh and Titman (1993), e.g., also report strategies that
        skip one period. Cannot be combined with 'rebalancing_month'.
    breakpoint_options_main : dict
        Named dict of 'breakpoint_options' passed to
        'breakpoint_function_main' for the main sorting variable.
        Required.
    breakpoint_options_secondary : dict, optional
        Named dict of 'breakpoint_options' passed to
        'breakpoint_function_secondary'. Required for bivariate sorts.
    breakpoint_function_main : callable, optional
        Function to compute breakpoints for the main sorting variable.
        Defaults to 'compute_breakpoints'.
    breakpoint_function_secondary : callable, optional
        Function to compute breakpoints for the secondary sorting
        variable. Defaults to 'compute_breakpoints'.
    min_portfolio_size : int, default 1
        Minimum number of firms required in the reported portfolio
        cross-section on a given date. For univariate sorts that is
        firms per portfolio-date; for bivariate sorts that is firms
        per main-portfolio-date summed across the secondary buckets.
        Cross-sections below the threshold have their returns set to
        null. A typical value is 5 (the Fama-French convention). Set
        to 0 to deactivate the check entirely.
    cap_weight : float, default 0.8
        Quantile of the cross-sectional 'mktcap_lag' distribution at
        which market capitalizations are capped per date when computing
        the capped value-weighted excess return ('ret_excess_vw_capped').
        Must be in [0, 1].
    cap_weight_exchanges : str or list of str, optional
        Exchange(s) whose stocks determine the cap: market
        capitalization is capped at the 'cap_weight' percentile of the
        stocks listed on these exchanges on each date. Defaults to
        'NYSE', following Jensen, Kelly, and Pedersen (2023), who cap
        at the 80th percentile of NYSE stocks. None uses all stocks.
        If the exchange column (mapped via 'data_options') is missing,
        or no stock is listed on these exchanges on a date, the
        percentile of all stocks is used and a warning says so.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'id', 'date',
        'ret_excess', and 'mktcap_lag' elements are used, and
        'exchange' with 'cap_weight_exchanges'. Uses 'data_options'
        defaults when None.
    quiet : bool, default False
        If True, suppress informational warnings about missing
        observations in the output panel and about the fallback of
        'cap_weight_exchanges'.

    Returns
    -------
    pl.DataFrame
        Data frame with computed portfolio returns as a complete panel
        (all portfolio-date combinations), containing:

        - 'portfolio': Portfolio identifier.
        - date column (as in 'data_options'): Date of the portfolio
          return.
        - 'ret_excess_vw': Value-weighted excess return (only if 'data'
          contains the market-cap lag column). Null if insufficient
          observations.
        - 'ret_excess_ew': Equal-weighted excess return. Null if
          insufficient observations.
        - 'ret_excess_vw_capped': Capped value-weighted excess return
          (only if 'data' contains the market-cap lag column). Weights
          are computed using market capitalization capped at the
          'cap_weight' percentile per date (see
          'cap_weight_exchanges'). Null if insufficient observations.

    Notes
    -----
    Ensure that 'data' contains all required columns: the specified
    sorting variables and excess returns (see options and defaults set
    in 'data_options'). A ValueError is raised if any required column
    is missing.

    In dependent bivariate sorts, the minimum market cap of
    'breakpoints_min_size_threshold' in 'breakpoint_options_main' is
    computed once per date from the full cross-section, not within
    each secondary portfolio. This applies to the default
    'compute_breakpoints'; custom breakpoint functions receive their
    options unchanged.

    References
    ----------
    Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and
    selling losers: Implications for stock market efficiency. Journal
    of Finance, 48(1), 65-91.
    https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

    Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is there a
    replication crisis in finance? Journal of Finance, 78(5),
    2465-2518. https://doi.org/10.1111/jofi.13249

    Examples
    --------
    ```python
    import datetime as dt
    import numpy as np
    import polars as pl
    from tidyfinance import (
        compute_portfolio_returns,
        breakpoint_options,
    )
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        dt.date(2020, 1, 1), dt.date(2028, 4, 1), '1mo', eager=True
    )
    data = pl.DataFrame({
        'permno': range(1, 501),
        'date': dates.to_numpy().repeat(5),
        'mktcap_lag': rng.uniform(100, 1000, 500),
        'ret_excess': rng.standard_normal(500),
        'size': rng.uniform(50, 150, 500),
    })
    # Univariate sorting with periodic rebalancing
    compute_portfolio_returns(
        data, 'size', 'univariate',
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
    )
    # Bivariate dependent sorting with annual rebalancing
    compute_portfolio_returns(
        data, ['size', 'mktcap_lag'], 'bivariate-dependent', 7,
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
        breakpoint_options_secondary=breakpoint_options(n_portfolios=3),
    )
    ```
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
            "price": "prc_lag",
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
            "No 'breakpoint_options_secondary' specified in bivariate sort.",
            UserWarning,
            stacklevel=2,
        )

    if (
        isinstance(cap_weight, bool)
        or not isinstance(cap_weight, (int, float))
        or np.isnan(cap_weight)
        or cap_weight < 0
        or cap_weight > 1
    ):
        raise ValueError(
            "'cap_weight' must be a single numeric value in [0, 1]."
        )

    if cap_weight_exchanges is not None and not (
        (isinstance(cap_weight_exchanges, str) and cap_weight_exchanges)
        or (
            isinstance(cap_weight_exchanges, (list, tuple))
            and len(cap_weight_exchanges) > 0
            and all(
                isinstance(e, str) and e for e in cap_weight_exchanges
            )
        )
    ):
        raise ValueError(
            "'cap_weight_exchanges' must be None, a string, or a list of "
            "strings."
        )

    def _is_whole_number(x) -> bool:
        return (
            not isinstance(x, bool)
            and isinstance(x, numbers.Real)
            and not np.isnan(x)
            and x == round(x)
        )

    if not _is_whole_number(holding_period) or holding_period < 1:
        raise ValueError("'holding_period' must be a single positive integer.")
    if not _is_whole_number(skip) or skip < 0:
        raise ValueError("'skip' must be a single non-negative integer.")
    holding_period = int(holding_period)
    skip = int(skip)
    staggered = holding_period > 1 or skip > 0
    if staggered and rebalancing_month is not None:
        raise ValueError(
            "'holding_period' and 'skip' cannot be combined with "
            "'rebalancing_month'."
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
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}.")

    mktcap_lag_missing = w_col not in data.columns
    if mktcap_lag_missing:
        data = data.with_columns(pl.lit(1.0).alias(w_col))

    all_dates = data.get_column(date_col).unique().sort()

    # Treat float NaN as missing everywhere downstream.
    float_cols = [
        c
        for c in set(list(sorting_variables) + [ret_col, w_col])
        if data.schema[c] in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        data = data.with_columns(
            [pl.col(c).fill_nan(None) for c in float_cols]
        )

    portfolio_cols = (
        ["portfolio"]
        if sorting_method == "univariate"
        else ["portfolio_secondary", "portfolio_main"]
    )
    data = _check_new_col(
        data,
        [w_capped_col]
        + [c for c in portfolio_cols if c not in sorting_variables],
        "compute_portfolio_returns",
    )

    # With staggered rebalancing, held stocks keep their returns even
    # without sorting variables, so the full panel is kept for the
    # holding periods.
    holding_data = data if staggered else None

    data = data.drop_nulls(subset=list(sorting_variables))

    # Compute capped market capitalization per date before replacing
    # nulls, so that the quantile is not distorted by zero-filled values.
    exchange_col = data_options.get("exchange", "exchange")
    cap_exchanges = None if mktcap_lag_missing else cap_weight_exchanges
    data = _cap_market_cap(
        data,
        w_col,
        w_capped_col,
        date_col,
        exchange_col,
        cap_weight,
        cap_exchanges,
        quiet,
    )

    def _zero_fill_missing_mcap(df: pl.DataFrame) -> pl.DataFrame:
        """Set missing market caps to 0, so these observations do not
        contribute to the value-weighted returns."""
        missing_mcap = pl.col(w_col).is_null()
        return df.with_columns(
            pl.when(missing_mcap)
            .then(0.0)
            .otherwise(pl.col(w_col))
            .alias(w_col),
            pl.when(missing_mcap)
            .then(0.0)
            .otherwise(pl.col(w_capped_col))
            .alias(w_capped_col),
        )

    if staggered:
        # Cap the weights of held stocks with the cap of the formation
        # universe. This runs before the formation data is zero-filled,
        # so missing market caps do not enter the percentile as zeros.
        holding_data = _zero_fill_missing_mcap(
            _cap_market_cap(
                holding_data,
                w_col,
                w_capped_col,
                date_col,
                exchange_col,
                cap_weight,
                cap_exchanges,
                quiet=True,
                cap_data=data,
            )
        )

    data = _zero_fill_missing_mcap(data)

    if data.height == 0:
        if not quiet:
            warnings.warn(
                "Returning an empty panel: all observations were "
                "filtered out (insufficient observations on every date).",
                UserWarning,
                stacklevel=2,
            )
        schema = {"portfolio": pl.Int64, date_col: data.schema[date_col]}
        if mktcap_lag_missing:
            schema["ret_excess_ew"] = pl.Float64
        else:
            schema["ret_excess_vw"] = pl.Float64
            schema["ret_excess_ew"] = pl.Float64
            schema["ret_excess_vw_capped"] = pl.Float64
        return pl.DataFrame(schema=schema)

    if rebalancing_month is not None and (
        rebalancing_month < 1 or rebalancing_month > 12
    ):
        raise ValueError("Invalid rebalancing_month.")

    def _assign_grouped(df, keys, sv, bp_opts, bp_fn, out_col):
        """Assign portfolios within each group defined by 'keys'."""
        parts = []
        for part in df.partition_by(keys, maintain_order=True):
            key_values = part.select(keys).row(0)
            if any(v is None for v in key_values):
                parts.append(
                    part.with_columns(
                        pl.lit(None, dtype=pl.Int64).alias(out_col)
                    )
                )
                continue
            assignments = assign_portfolio(
                part.drop(keys),
                sorting_variable=sv,
                breakpoint_options=bp_opts,
                breakpoint_function=bp_fn,
                data_options=data_options,
            )
            parts.append(part.with_columns(assignments.alias(out_col)))
        return pl.concat(parts)

    def _assign_dependent_main(df, sv, bp_opts, bp_fn):
        """Assign main portfolios within the secondary portfolios of
        each date, with the minimum market cap of the main breakpoints
        computed once per date from the full cross-section."""
        parts = []
        for part in df.partition_by(date_col, maintain_order=True):
            min_mktcap = _main_min_mktcap(part, bp_opts, bp_fn, data_options)
            parts.append(
                _assign_grouped(
                    part,
                    [date_col, "portfolio_secondary"],
                    sv,
                    _with_min_mktcap(bp_opts, min_mktcap),
                    bp_fn,
                    "portfolio_main",
                )
            )
        return pl.concat(parts)

    def _assign_formation_portfolios(df):
        """Assign portfolios on every date of 'df': 'portfolio' for
        univariate sorts, 'portfolio_main' and 'portfolio_secondary'
        for bivariate sorts."""
        if sorting_method == "univariate":
            return _assign_grouped(
                df,
                [date_col],
                sorting_variables[0],
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio",
            ).select([id_col, date_col, "portfolio"])

        df = _assign_grouped(
            df,
            [date_col],
            sorting_variables[1],
            breakpoint_options_secondary,
            breakpoint_function_secondary,
            "portfolio_secondary",
        )
        if sorting_method == "bivariate-dependent":
            df = _assign_dependent_main(
                df,
                sorting_variables[0],
                breakpoint_options_main,
                breakpoint_function_main,
            )
        else:
            df = _assign_grouped(
                df,
                [date_col],
                sorting_variables[0],
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio_main",
            )
        return df.select(
            [id_col, date_col, "portfolio_main", "portfolio_secondary"]
        )

    if staggered:
        if sorting_method == "univariate" and len(sorting_variables) > 1:
            raise ValueError(
                "Only provide one sorting variable for univariate sorts."
            )
        if sorting_method != "univariate" and len(sorting_variables) != 2:
            raise ValueError(
                "Provide two sorting variables for bivariate sorts."
            )

        dates = all_dates
        portfolio_returns = _compute_staggered_returns(
            _assign_formation_portfolios(data),
            holding_data,
            sorting_method,
            holding_period,
            skip,
            dates,
            id_col,
            date_col,
            ret_col,
            w_col,
            w_capped_col,
            min_portfolio_size,
        )

        # The panel starts once all holding_period cohorts are held
        first_formation = dates.to_list().index(
            data.get_column(date_col).min()
        )
        first_full = first_formation + skip + holding_period - 1
        all_dates = dates.slice(first_full)

    elif sorting_method == "univariate":
        if len(sorting_variables) > 1:
            raise ValueError(
                "Only provide one sorting variable for univariate sorts."
            )

        sv = sorting_variables[0]

        if rebalancing_month is None:
            portfolio_returns = _assign_grouped(
                data,
                [date_col],
                sv,
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio",
            )
        else:
            filtered_data = data.filter(
                pl.col(date_col).dt.month() == rebalancing_month
            )
            if filtered_data.height == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}. Check that the data contains "
                    "dates in the specified rebalancing month."
                )

            portfolio_data = _assign_grouped(
                filtered_data,
                [date_col],
                sv,
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio",
            ).select([id_col, date_col, "portfolio"])
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

        if rebalancing_month is None:
            data = _assign_grouped(
                data,
                [date_col],
                sv_sec,
                breakpoint_options_secondary,
                breakpoint_function_secondary,
                "portfolio_secondary",
            )
            portfolio_returns = _assign_dependent_main(
                data,
                sv_main,
                breakpoint_options_main,
                breakpoint_function_main,
            )
        else:
            filtered_data = data.filter(
                pl.col(date_col).dt.month() == rebalancing_month
            )
            if filtered_data.height == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = _assign_grouped(
                filtered_data,
                [date_col],
                sv_sec,
                breakpoint_options_secondary,
                breakpoint_function_secondary,
                "portfolio_secondary",
            )
            portfolio_data = _assign_dependent_main(
                portfolio_data,
                sv_main,
                breakpoint_options_main,
                breakpoint_function_main,
            ).select(
                [
                    id_col,
                    date_col,
                    "portfolio_main",
                    "portfolio_secondary",
                ]
            )
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

        if rebalancing_month is None:
            data = _assign_grouped(
                data,
                [date_col],
                sv_sec,
                breakpoint_options_secondary,
                breakpoint_function_secondary,
                "portfolio_secondary",
            )
            portfolio_returns = _assign_grouped(
                data,
                [date_col],
                sv_main,
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio_main",
            )
        else:
            filtered_data = data.filter(
                pl.col(date_col).dt.month() == rebalancing_month
            )
            if filtered_data.height == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = _assign_grouped(
                filtered_data,
                [date_col],
                sv_sec,
                breakpoint_options_secondary,
                breakpoint_function_secondary,
                "portfolio_secondary",
            )
            portfolio_data = _assign_grouped(
                portfolio_data,
                [date_col],
                sv_main,
                breakpoint_options_main,
                breakpoint_function_main,
                "portfolio_main",
            ).select(
                [
                    id_col,
                    date_col,
                    "portfolio_main",
                    "portfolio_secondary",
                ]
            )
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

    portfolio_returns = portfolio_returns.filter(
        pl.col("portfolio").is_not_null()
    )

    if rebalancing_month is not None:
        matching_dates = all_dates.filter(
            all_dates.dt.month() == rebalancing_month
        )
        if matching_dates.is_empty():
            raise ValueError(
                f"No dates in data match for rebalancing month = "
                f"{rebalancing_month}."
            )
        first_rebalancing_date = matching_dates.min()
        all_dates = all_dates.filter(all_dates >= first_rebalancing_date)

    all_portfolios = (
        portfolio_returns.get_column("portfolio").unique().drop_nulls().sort()
    )
    complete_panel = pl.DataFrame({"portfolio": all_portfolios}).join(
        pl.DataFrame({date_col: all_dates}),
        how="cross",
        maintain_order="left_right",
    )

    return_cols = (
        ["ret_excess_ew"]
        if mktcap_lag_missing
        else [
            "ret_excess_vw",
            "ret_excess_ew",
            "ret_excess_vw_capped",
        ]
    )

    portfolio_returns = complete_panel.join(
        portfolio_returns,
        on=["portfolio", date_col],
        how="left",
        maintain_order="left",
    ).select(["portfolio", date_col] + return_cols)

    n_missing = portfolio_returns.get_column("ret_excess_ew").null_count()
    if not quiet and n_missing > 0:
        warnings.warn(
            f"Returning a complete panel with {n_missing} missing "
            "portfolio returns. A portfolio return is missing on a date "
            "when the portfolio has no assigned stocks, has fewer stocks "
            f"than 'min_portfolio_size' ({min_portfolio_size}), or has a "
            "stock with a missing return.",
            UserWarning,
            stacklevel=2,
        )

    return portfolio_returns


def compute_long_short_returns(
    data: pl.DataFrame,
    direction: str = "top_minus_bottom",
    data_options: dict = None,
) -> pl.DataFrame:
    """Compute long-short returns.

    Calculates long-short returns based on the returns of portfolios.
    The long-short return is computed as the difference between the
    returns of the 'top' and 'bottom' portfolios. The direction of the
    calculation can be adjusted based on whether the return from the
    'bottom' portfolio is subtracted from or added to the return from
    the 'top' portfolio.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing portfolio returns. Must include columns
        for the portfolio identifier and date (as specified in
        'data_options') and one or more of the return columns that
        'compute_portfolio_returns' produces ('ret_excess_vw',
        'ret_excess_ew', 'ret_excess_vw_capped'). Columns whose name
        contains data_options['ret_excess'] are also used.
    direction : str, default 'top_minus_bottom'
        Direction of the long-short return calculation. Must be either
        'top_minus_bottom' or 'bottom_minus_top'. If set to
        'bottom_minus_top', the return is computed as (bottom - top).
        Any other value raises a 'ValueError'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element
        specifies the date column, the 'ret_excess' element specifies
        the excess return column, and 'portfolio' specifies the
        assigned portfolio. Uses the 'data_options' defaults when
        None: 'date' -> 'date', 'ret_excess' -> 'ret_excess', and
        'portfolio' -> 'portfolio'.

    Returns
    -------
    pl.DataFrame
        Data frame with columns for date and the computed long-short
        returns. The data frame is arranged by date and pivoted to
        have return measurement types as columns with their
        corresponding long-short returns.

    Examples
    --------
    ```python
    import datetime as dt
    import numpy as np
    import polars as pl
    from tidyfinance import (
        compute_portfolio_returns,
        compute_long_short_returns,
        breakpoint_options,
    )
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        dt.date(2020, 1, 1), dt.date(2028, 4, 1), '1mo', eager=True
    )
    data = pl.DataFrame({
        'permno': range(1, 101),
        'date': dates,
        'mktcap_lag': rng.uniform(100, 1000, 100),
        'ret_excess': rng.standard_normal(100),
        'size': rng.uniform(50, 150, 100),
    })
    portfolio_returns = compute_portfolio_returns(
        data, 'size', 'univariate',
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
    )
    compute_long_short_returns(portfolio_returns)
    ```
    """
    valid_directions = ("top_minus_bottom", "bottom_minus_top")
    if not isinstance(direction, str) or direction not in valid_directions:
        raise ValueError(
            "'direction' must be one of 'top_minus_bottom' and "
            "'bottom_minus_top'."
        )

    if data_options is None:
        data_options = {
            "date": "date",
            "ret_excess": "ret_excess",
            "portfolio": "portfolio",
        }

    date_col = data_options["date"]
    ret_excess_col = data_options["ret_excess"]
    portfolio_col = data_options["portfolio"]

    # The return columns are those that 'compute_portfolio_returns'
    # produces, which it names the same way regardless of
    # data_options['ret_excess']. For backward compatibility also accept
    # any column whose name contains the configured 'ret_excess' name.
    # Columns are kept in data order.
    canonical = ("ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped")
    ret_cols = [
        c for c in data.columns if c in canonical or ret_excess_col in c
    ]
    if not ret_cols:
        raise ValueError(
            "No portfolio return columns found in 'data'. Expected one or "
            "more of 'ret_excess_vw', 'ret_excess_ew', and "
            "'ret_excess_vw_capped' (as returned by "
            "'compute_portfolio_returns'), or a column whose name "
            f"contains '{ret_excess_col}'."
        )

    work = _check_new_col(
        data, ["_min_p", "_max_p", "_leg"], "compute_long_short_returns"
    ).with_columns(
        _min_p=pl.col(portfolio_col).min().over(date_col),
        _max_p=pl.col(portfolio_col).max().over(date_col),
    )

    single_portfolio = pl.col("_min_p") == pl.col("_max_p")
    work = work.with_columns(
        pl.when(single_portfolio)
        .then(None)
        .when(pl.col(portfolio_col) == pl.col("_min_p"))
        .then(pl.lit("bottom"))
        .when(pl.col(portfolio_col) == pl.col("_max_p"))
        .then(pl.lit("top"))
        .otherwise(None)
        .alias("_leg")
    )

    legs_only = work.filter(pl.col("_leg").is_in(["bottom", "top"]))

    all_dates = data.get_column(date_col).unique().sort()
    result = pl.DataFrame({date_col: all_dates})

    for ret_col in ret_cols:
        legs = (
            legs_only.group_by([date_col, "_leg"], maintain_order=True)
            .agg(pl.col(ret_col).drop_nulls().first())
            .pivot("_leg", index=date_col, values=ret_col)
        )
        for leg in ("bottom", "top"):
            if leg not in legs.columns:
                legs = legs.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(leg)
                )

        if direction == "bottom_minus_top":
            ls = pl.col("bottom") - pl.col("top")
        else:
            ls = pl.col("top") - pl.col("bottom")

        ls_df = legs.select(pl.col(date_col), ls.alias(ret_col))
        result = result.join(
            ls_df, on=date_col, how="left", maintain_order="left"
        )

    return result


def _require_column(
    data: pl.DataFrame, col: str, arg: str, info: str = None
) -> None:
    """Raise ValueError if col is not in data.columns."""
    if col not in data.columns:
        if info is None:
            info = f"Set {arg} to the correct column name."
        raise ValueError(f"Column '{col}' not found in 'data'. {info}")


def _filter_with_log(
    data: pl.DataFrame, condition, label: str, quiet: bool
) -> pl.DataFrame:
    """Apply a boolean filter and optionally warn about dropped rows."""
    n_before = data.height
    data = data.filter(condition)
    n_dropped = n_before - data.height
    if not quiet and n_dropped > 0:
        warnings.warn(
            f"Filter '{label}': removed {n_dropped} observation(s).",
            UserWarning,
            stacklevel=2,
        )
    return data


def filter_sorting_data(
    data: pl.DataFrame,
    filter_options: dict = None,
    data_options: dict = None,
    rebalancing_month: int = None,
    quiet: bool = False,
) -> pl.DataFrame:
    """Filter sorting data.

    Applies sample construction filters to a data frame before
    portfolio sorting. Filters are applied in a fixed order:
    financials exclusion, utilities exclusion, minimum stock price,
    minimum size quantile, minimum listing age, positive book equity,
    and positive earnings. An informational warning is emitted for
    each filter that actually removes at least one observation.

    Without 'rebalancing_month', every month is a formation month and
    each row is screened on its own values. The screens are then free
    of look-ahead bias only if the screened columns are known at the
    start of the month: the default price column 'prc_lag' and
    'mktcap_lag' are lagged by one month, and accounting variables
    such as book equity and earnings should be lagged as well (e.g.,
    with 'join_lagged_values').

    With 'rebalancing_month', portfolios are formed once a year and
    held for twelve months. By default
    ('filter_options(timing="formation")'), each screen is evaluated in
    the formation month only, and a stock that passes all screens keeps
    all its rows until the next formation month, even if it fails a
    screen in a holding month. Rows of stocks that fail a screen at
    formation, and rows before the first formation month, are removed.
    With 'timing="every_period"', the screens are evaluated in every
    month (ex-post) and a warning is issued, because dropping
    holding-month returns based on information from those months can
    bias factor returns.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing the stock-level panel data to be
        filtered.
    filter_options : dict, optional
        Dict produced by 'filter_options'. If None (the default), the
        defaults from 'filter_options' are used (i.e., no filters are
        applied). The accepted entries include:

        - 'exclude_financials' (bool): Whether to exclude financial
          firms (SIC codes 6000 to 6799). Defaults to False.
        - 'exclude_utilities' (bool): Whether to exclude utility firms
          (SIC codes 4900 to 4999). Defaults to False.
        - 'min_stock_price' (float, optional): Minimum stock price
          required to include an observation. None (the default)
          applies no price filter.
        - 'min_size_quantile' (float, optional): Minimum cross-sectional
          size quantile (based on lagged market cap) required to
          include an observation. None (the default) applies no size
          quantile filter. The cutoff is computed from NYSE stocks
          only; the 'exchange' column (mapped via 'data_options') must
          be present or a ValueError is raised.
        - 'min_listing_age' (float, optional): Minimum number of months
          a stock must have been listed in CRSP. None (the default)
          applies no listing age filter.
        - 'exclude_negative_book_equity' (bool): Whether to exclude
          observations with non-positive book equity. Defaults to False.
        - 'exclude_negative_earnings' (bool): Whether to exclude
          observations with non-positive earnings. Defaults to False.
        - 'timing' (str): Either 'formation' (the default) to apply the
          screens only in the formation month when portfolios are
          rebalanced once a year, or 'every_period' to apply them in
          every month, which can introduce look-ahead bias.

    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'siccd' element
        specifies the SIC code column, 'price' specifies the price
        column of the 'min_stock_price' screen, 'mktcap_lag' specifies
        the market capitalization column, 'date' specifies the date
        column, 'listing_age' specifies the listing age column, 'be'
        specifies the book equity column, and 'earnings' specifies the
        earnings column. With 'rebalancing_month', the 'id' element
        (default 'permno') is used to keep a stock across its holding
        period. Uses the 'data_options' defaults when None.
    rebalancing_month : int, optional
        Month (1 to 12) in which portfolios are formed when they are
        rebalanced once a year (see 'compute_portfolio_returns'). None
        (the default) means monthly rebalancing. Together with the
        'timing' element of 'filter_options', it determines when the
        screens apply.
    quiet : bool, default False
        Whether informational messages should be suppressed.

    Returns
    -------
    pl.DataFrame
        Filtered data frame, preserving the class and structure of the
        input.

    Examples
    --------
    ```python
    import datetime as dt
    import polars as pl
    from tidyfinance import filter_sorting_data, filter_options
    data = pl.DataFrame({
        'permno': range(1, 6),
        'date': [dt.date(2020, 1, 1)] * 5,
        'siccd': [6100, 2000, 4950, 3000, 6500],
        'prc_lag':[5.0, 0.5, 15.0, 20.0, 10.0],
    })
    filter_sorting_data(
        data,
        filter_options=filter_options(
            exclude_financials=True,
            min_stock_price=1,
        ),
    )
    # Annual rebalancing in July: the price screen is evaluated in July
    # only, so stock 1 is kept in August although its price falls below 1
    data_annual = pl.DataFrame({
        'permno': [1, 1, 2, 2],
        'date': [dt.date(2020, 7, 1), dt.date(2020, 8, 1)] * 2,
        'prc_lag': [2.0, 0.5, 0.5, 2.0],
    })
    filter_sorting_data(
        data_annual,
        filter_options=filter_options(min_stock_price=1),
        rebalancing_month=7,
    )
    ```
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
            "price": "prc_lag",
            "listing_age": "listing_age",
            "be": "be",
            "earnings": "ib",
        }

    if rebalancing_month is not None and (
        isinstance(rebalancing_month, bool)
        or not isinstance(rebalancing_month, numbers.Real)
        or rebalancing_month not in range(1, 13)
    ):
        raise ValueError(
            "'rebalancing_month' must be a single integer between 1 and 12."
        )

    if rebalancing_month is None or not _has_active_screens(filter_options):
        return _apply_screens(data, filter_options, data_options, quiet)

    if filter_options.get("timing", "formation") == "every_period":
        _warn_every_period_screens()
        return _apply_screens(data, filter_options, data_options, quiet)

    return _screen_at_formation(
        data, filter_options, data_options, int(rebalancing_month), quiet
    )


def _has_active_screens(filter_options: dict) -> bool:
    """Whether any screen in 'filter_options' is active."""
    return (
        bool(filter_options.get("exclude_financials"))
        or bool(filter_options.get("exclude_utilities"))
        or filter_options.get("min_stock_price") is not None
        or filter_options.get("min_size_quantile") is not None
        or filter_options.get("min_listing_age") is not None
        or bool(filter_options.get("exclude_negative_book_equity"))
        or bool(filter_options.get("exclude_negative_earnings"))
    )


def _warn_every_period_screens() -> None:
    """Warn that screens applied in every period can bias factor
    returns."""
    warnings.warn(
        "Screens are applied in every month, including holding months. "
        "A stock that fails a screen after portfolio formation loses "
        "those months' returns, so factor returns can contain "
        "look-ahead bias. Use filter_options(timing='formation') to "
        "apply the screens only when portfolios are formed.",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _screen_at_formation(
    data: pl.DataFrame,
    filter_options: dict,
    data_options: dict,
    rebalancing_month: int,
    quiet: bool,
) -> pl.DataFrame:
    """Apply the screens in the formation month and keep the stocks
    that pass for the holding period.

    Portfolios formed in 'rebalancing_month' of year y are held until
    the month before 'rebalancing_month' of year y + 1.
    """
    col_id = data_options["id"]
    col_date = data_options["date"]
    _require_column(data, col_id, "data_options['id']")
    _require_column(data, col_date, "data_options['date']")
    data = _check_new_col(data, ["_cohort", "_keep"], "filter_sorting_data")

    year = pl.col(col_date).dt.year()
    month = pl.col(col_date).dt.month()
    is_formation = month == rebalancing_month

    passed = _apply_screens(
        data.filter(is_formation), filter_options, data_options, quiet
    )
    passed_keys = (
        passed.select(pl.col(col_id), year.alias("_cohort"))
        .unique()
        .with_columns(pl.lit(True).alias("_keep"))
    )

    data = (
        data.with_columns(
            (
                year
                - pl.when(month < rebalancing_month).then(1).otherwise(0)
            ).alias("_cohort")
        )
        .join(
            passed_keys,
            on=[col_id, "_cohort"],
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col("_keep").fill_null(False))
    )

    n_holding_dropped = data.filter(
        ~pl.col("_keep") & ~is_formation
    ).height
    if not quiet and n_holding_dropped > 0:
        warnings.warn(
            f"Screens applied in formation month {rebalancing_month}: "
            f"removed {n_holding_dropped} holding-month observation(s) "
            "of stocks that failed a screen at formation or precede the "
            "first formation month.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    return data.filter(pl.col("_keep")).drop(["_cohort", "_keep"])


def _apply_screens(
    data: pl.DataFrame,
    filter_options: dict,
    data_options: dict,
    quiet: bool,
) -> pl.DataFrame:
    """Apply the screens of 'filter_options' to every row of 'data'."""
    # Treat float NaN as missing in the filter columns (polars-native
    # inputs can carry NaN, which would otherwise pass the filters and
    # shift quantile cutoffs).
    nan_cols = list(
        dict.fromkeys(
            data_options[key]
            for key in ("mktcap_lag", "price", "be", "earnings")
            if data_options.get(key) in data.columns
            and data.schema[data_options[key]] in (pl.Float32, pl.Float64)
        )
    )
    if nan_cols:
        data = data.with_columns(
            [pl.col(c).fill_nan(None) for c in nan_cols]
        )

    # exclude_financials / exclude_utilities (share the SIC column)
    if filter_options.get("exclude_financials") or filter_options.get(
        "exclude_utilities"
    ):
        col_siccd = data_options["siccd"]
        _require_column(data, col_siccd, "data_options['siccd']")

        if filter_options.get("exclude_financials"):
            keep = pl.col(col_siccd).is_null() | ~(
                (pl.col(col_siccd) >= 6000) & (pl.col(col_siccd) <= 6799)
            )
            data = _filter_with_log(data, keep, "exclude_financials", quiet)

        if filter_options.get("exclude_utilities"):
            keep = pl.col(col_siccd).is_null() | ~(
                (pl.col(col_siccd) >= 4900) & (pl.col(col_siccd) <= 4999)
            )
            data = _filter_with_log(data, keep, "exclude_utilities", quiet)

    # min_stock_price
    if filter_options.get("min_stock_price") is not None:
        col_price = data_options["price"]
        _require_column(data, col_price, "data_options['price']")
        keep = pl.col(col_price).is_not_null() & (
            pl.col(col_price) >= filter_options["min_stock_price"]
        )
        data = _filter_with_log(data, keep, "min_stock_price", quiet)

    # min_size_quantile
    if filter_options.get("min_size_quantile") is not None:
        col_mktcap_lag = data_options["mktcap_lag"]
        col_date = data_options["date"]
        col_exchange = data_options["exchange"]
        _require_column(data, col_mktcap_lag, "data_options['mktcap_lag']")
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

        n_before = data.height
        size_threshold = filter_options["min_size_quantile"]
        nyse_data = data.filter(pl.col(col_exchange) == "NYSE")
        cutoff_dates, cutoff_values = [], []
        for (date,), part in nyse_data.group_by(col_date, maintain_order=True):
            cutoff = _quantile(
                part.get_column(col_mktcap_lag).cast(pl.Float64).to_numpy(),
                size_threshold,
            )
            cutoff_dates.append(date)
            cutoff_values.append(None if np.isnan(cutoff) else cutoff)
        size_cutoffs = pl.DataFrame(
            {col_date: cutoff_dates, "_size_cutoff": cutoff_values},
            schema={
                col_date: data.schema[col_date],
                "_size_cutoff": pl.Float64,
            },
        )

        dates_in_data = set(data.get_column(col_date).unique().to_list())
        dates_with_cutoff = set(
            size_cutoffs.get_column(col_date).unique().to_list()
        )
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

        data = _check_new_col(data, "_size_cutoff", "filter_sorting_data")
        data = data.join(
            size_cutoffs, on=col_date, how="inner", maintain_order="left"
        )
        data = data.filter(
            pl.col(col_mktcap_lag).is_not_null()
            & (pl.col(col_mktcap_lag) >= pl.col("_size_cutoff"))
        )
        data = data.drop("_size_cutoff")
        n_dropped = n_before - data.height
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
        _require_column(data, col_listing_age, "data_options['listing_age']")
        keep = pl.col(col_listing_age).is_not_null() & (
            pl.col(col_listing_age) >= filter_options["min_listing_age"]
        )
        data = _filter_with_log(data, keep, "min_listing_age", quiet)

    # exclude_negative_book_equity
    if filter_options.get("exclude_negative_book_equity"):
        col_be = data_options["be"]
        _require_column(data, col_be, "data_options['be']")
        keep = pl.col(col_be).is_not_null() & (pl.col(col_be) > 0)
        data = _filter_with_log(
            data, keep, "exclude_negative_book_equity", quiet
        )

    # exclude_negative_earnings
    if filter_options.get("exclude_negative_earnings"):
        col_earn = data_options["earnings"]
        _require_column(data, col_earn, "data_options['earnings']")
        keep = pl.col(col_earn).is_not_null() & (pl.col(col_earn) > 0)
        data = _filter_with_log(data, keep, "exclude_negative_earnings", quiet)

    return data


def _screen_staggered_formation(
    data: pl.DataFrame,
    sorting_variables,
    filter_options: dict,
    data_options: dict | None,
    quiet: bool,
) -> pl.DataFrame:
    """Apply screens to the formation of staggered portfolios.

    With staggered rebalancing, every month is a formation month, but
    stocks are held for several months. With 'timing="formation"', a
    stock that fails a screen is not added to the cohort formed in that
    month, but stays in the cohorts formed earlier: its sorting
    variables are set to null in that month instead of removing the
    row. With 'timing="every_period"', failing rows are removed, so
    earlier cohorts lose the stock's return, and a warning is issued.
    """
    if data_options is None:
        data_options = {
            "id": "permno",
            "date": "date",
            "exchange": "exchange",
            "mktcap_lag": "mktcap_lag",
            "ret_excess": "ret_excess",
            "portfolio": "portfolio",
            "siccd": "siccd",
            "price": "prc_lag",
            "listing_age": "listing_age",
            "be": "be",
            "earnings": "ib",
        }

    passed = _apply_screens(data, filter_options, data_options, quiet)

    if filter_options.get("timing", "formation") == "every_period":
        _warn_every_period_screens()
        return passed

    id_col = data_options["id"]
    date_col = data_options["date"]
    _require_column(data, id_col, "data_options['id']")
    _require_column(data, date_col, "data_options['date']")
    data = _check_new_col(data, "_passed", "implement_portfolio_sort")

    if isinstance(sorting_variables, str):
        sorting_variables = [sorting_variables]

    passed_keys = (
        passed.select([id_col, date_col])
        .unique()
        .with_columns(pl.lit(True).alias("_passed"))
    )
    data = data.join(
        passed_keys, on=[id_col, date_col], how="left", maintain_order="left"
    ).with_columns(pl.col("_passed").fill_null(False))

    return data.with_columns(
        [
            pl.when(pl.col("_passed"))
            .then(pl.col(sv))
            .otherwise(None)
            .alias(sv)
            for sv in sorting_variables
        ]
    ).drop("_passed")


def implement_portfolio_sort(
    data: pl.DataFrame,
    sorting_variables,
    sorting_method: str,
    portfolio_sort_options: dict,
    rebalancing_month: int = None,
    holding_period: int = 1,
    skip: int = 0,
    breakpoint_function_main=None,
    breakpoint_function_secondary=None,
    min_portfolio_size: int = 1,
    cap_weight: float = 0.8,
    cap_weight_exchanges="NYSE",
    data_options: dict = None,
    quiet: bool = False,
) -> pl.DataFrame:
    """Implement a portfolio sort.

    A convenience wrapper that combines sample construction filtering
    and portfolio return computation into a single call. Equivalent to
    calling 'filter_sorting_data' followed by
    'compute_portfolio_returns' with the filter and breakpoint
    specifications bundled in 'portfolio_sort_options'.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing the stock-level panel data.
    sorting_variables : str or list of str
        One or two column names to sort portfolios on.
    sorting_method : str
        Sorting method to use. One of 'univariate',
        'bivariate-dependent', or 'bivariate-independent'.
    portfolio_sort_options : dict
        Dict produced by 'portfolio_sort_options', bundling filter and
        breakpoint specifications. The accepted entries include
        'filter_options', 'breakpoint_options_main', and
        'breakpoint_options_secondary'.
    rebalancing_month : int, optional
        Month in which portfolios are rebalanced annually. None (the
        default) means monthly rebalancing. Also determines the
        formation month in which the screens of 'filter_options' are
        applied (see 'filter_sorting_data').
    holding_period, skip : int, default 1 and 0
        Staggered rebalancing: portfolios are formed every period and
        each cohort is held for 'holding_period' periods, starting
        'skip' periods after formation. See 'compute_portfolio_returns'.
        With the default 'filter_options(timing="formation")', the
        screens only decide which stocks enter the cohort formed in a
        period; a stock that fails a screen later stays in its earlier
        cohorts until their holding period ends.
    breakpoint_function_main : callable, optional
        Function used to compute breakpoints for the main sorting
        variable. Defaults to 'compute_breakpoints'.
    breakpoint_function_secondary : callable, optional
        Function used to compute breakpoints for the secondary sorting
        variable. Defaults to 'compute_breakpoints'.
    min_portfolio_size : int, default 1
        Minimum number of firms in the reported portfolio
        cross-section on a given date. For univariate sorts that is
        firms per portfolio-date; for bivariate sorts that is firms
        per main-portfolio-date summed across the secondary buckets.
        Cross-sections below the threshold have their returns set to
        null. Set to 0 to deactivate the check.
    cap_weight : float, default 0.8
        Quantile of the cross-sectional 'mktcap_lag' distribution at
        which market capitalizations are capped per date when computing
        the capped value-weighted excess return ('ret_excess_vw_capped').
        Must be in [0, 1].
    cap_weight_exchanges : str or list of str, optional
        Exchange(s) whose stocks determine the cap, or None for all
        stocks. Defaults to 'NYSE'. See 'compute_portfolio_returns'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). All elements are
        forwarded to 'filter_sorting_data' and
        'compute_portfolio_returns'. Uses the 'data_options' defaults
        when None.
    quiet : bool, default False
        Whether informational messages should be suppressed.

    Returns
    -------
    pl.DataFrame
        Data frame of portfolio returns as returned by
        'compute_portfolio_returns'.

    Examples
    --------
    ```python
    import datetime as dt
    import numpy as np
    import polars as pl
    from tidyfinance import (
        implement_portfolio_sort,
        portfolio_sort_options,
        filter_options,
        breakpoint_options,
    )
    rng = np.random.default_rng(123)
    dates = pl.date_range(
        dt.date(2020, 1, 1), dt.date(2028, 4, 1), '1mo', eager=True
    )
    data = pl.DataFrame({
        'permno': range(1, 501),
        'date': dates.to_numpy().repeat(5),
        'mktcap_lag': rng.uniform(100, 1000, 500),
        'ret_excess': rng.standard_normal(500),
        'prc_lag':rng.uniform(0.5, 50, 500),
        'size': rng.uniform(50, 150, 500),
    })
    implement_portfolio_sort(
        data=data,
        sorting_variables='size',
        sorting_method='univariate',
        portfolio_sort_options=portfolio_sort_options(
            filter_options=filter_options(min_stock_price=1),
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
        ),
    )
    ```
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
            "'data_options' must be None or a dict produced by data_options()."
        )

    _portfolio_sort_options_keys = {
        "filter_options",
        "breakpoint_options_main",
        "breakpoint_options_secondary",
    }
    if not (
        isinstance(portfolio_sort_options, dict)
        and _portfolio_sort_options_keys.issubset(portfolio_sort_options.keys())
    ):
        raise ValueError(
            "'portfolio_sort_options' must be a dict produced by "
            "portfolio_sort_options()."
        )

    filter_opts = portfolio_sort_options["filter_options"]
    breakpoint_options_main = portfolio_sort_options["breakpoint_options_main"]
    breakpoint_options_secondary = portfolio_sort_options[
        "breakpoint_options_secondary"
    ]

    staggered = holding_period != 1 or skip != 0
    if (
        staggered
        and filter_opts is not None
        and _has_active_screens(filter_opts)
    ):
        data = _screen_staggered_formation(
            data,
            sorting_variables=sorting_variables,
            filter_options=filter_opts,
            data_options=data_options,
            quiet=quiet,
        )
    else:
        data = filter_sorting_data(
            data,
            filter_options=filter_opts,
            data_options=data_options,
            rebalancing_month=rebalancing_month,
            quiet=quiet,
        )

    return compute_portfolio_returns(
        data,
        sorting_variables=sorting_variables,
        sorting_method=sorting_method,
        rebalancing_month=rebalancing_month,
        holding_period=holding_period,
        skip=skip,
        breakpoint_options_main=breakpoint_options_main,
        breakpoint_options_secondary=breakpoint_options_secondary,
        breakpoint_function_main=breakpoint_function_main,
        breakpoint_function_secondary=breakpoint_function_secondary,
        min_portfolio_size=min_portfolio_size,
        cap_weight=cap_weight,
        cap_weight_exchanges=cap_weight_exchanges,
        data_options=data_options,
        quiet=quiet,
    )
