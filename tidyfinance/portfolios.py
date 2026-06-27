"""Portfolio sorting and return functions for tidyfinance."""

import warnings

import numpy as np
import pandas as pd

from ._internal import (_check_new_col, _validate_column_name, _validate_flag,
                        _validate_optional_number)


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
        Portfolio assignments as a float series. Each entry is the
        1-indexed portfolio number; values outside the breakpoint
        range fall into the boundary portfolios. NaN inputs are
        returned as NaN.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import assign_portfolio, breakpoint_options
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
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
    data : pd.DataFrame
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
    import pandas as pd
    from tidyfinance import compute_breakpoints, breakpoint_options
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
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
            keep_mask & above_size if keep_mask is not None else above_size
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
            breakpoints_new = np.quantile(sorting_values_new, probs_new)
            breakpoints_new[-1] += 1e-15
            breakpoints = np.concatenate(
                [breakpoints_new, [breakpoints[n_portfolios]]]
            )

    breakpoints[1:] += 1e-20
    return breakpoints


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
    ```
    """
    _validate_flag(exclude_financials, "exclude_financials")
    _validate_flag(exclude_utilities, "exclude_utilities")
    _validate_flag(exclude_negative_book_equity, "exclude_negative_book_equity")
    _validate_flag(exclude_negative_earnings, "exclude_negative_earnings")

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
        cell_returns.groupby(["portfolio_main", date_col], as_index=False)
        .agg(
            ret_excess_vw=("ret_excess_vw", "mean"),
            ret_excess_ew=("ret_excess_ew", "mean"),
            ret_excess_vw_capped=("ret_excess_vw_capped", "mean"),
        )
        .rename(columns={"portfolio_main": "portfolio"})
    )

    n_renamed = n_per_main.rename(columns={"portfolio_main": "portfolio"})
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

    merged = data_w.merge(pd_w, on=[id_col, "_window_year"], how="left")
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
    """Compute portfolio returns.

    Computes individual portfolio returns based on specified sorting
    variables and sorting methods. The portfolios can be rebalanced
    every period or on an annual frequency by specifying a rebalancing
    month, which is only applicable at a monthly return frequency. The
    function supports univariate and bivariate sorts, with the latter
    supporting dependent and independent sorting methods.

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
    data : pd.DataFrame
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
        NaN. A typical value is 5 (the Fama-French convention). Set to
        0 to deactivate the check entirely.
    cap_weight : float, default 0.8
        Quantile of the cross-sectional 'mktcap_lag' distribution at
        which market capitalizations are capped per date when computing
        the capped value-weighted excess return ('ret_excess_vw_capped').
        Must be in [0, 1].
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'id', 'date',
        'ret_excess', and 'mktcap_lag' elements are used. Uses
        'data_options' defaults when None.
    quiet : bool, default False
        If True, suppress informational warnings about missing
        observations in the output panel.

    Returns
    -------
    pd.DataFrame
        Data frame with computed portfolio returns as a complete panel
        (all portfolio-date combinations), containing:

        - 'portfolio': Portfolio identifier.
        - date column (as in 'data_options'): Date of the portfolio
          return.
        - 'ret_excess_vw': Value-weighted excess return (only if 'data'
          contains the market-cap lag column). NaN if insufficient
          observations.
        - 'ret_excess_ew': Equal-weighted excess return. NaN if
          insufficient observations.
        - 'ret_excess_vw_capped': Capped value-weighted excess return
          (only if 'data' contains the market-cap lag column). Weights
          are computed using market capitalization capped at the
          'cap_weight' percentile per date. NaN if insufficient
          observations.

    Notes
    -----
    Ensure that 'data' contains all required columns: the specified
    sorting variables and excess returns (see options and defaults set
    in 'data_options'). A ValueError is raised if any required column
    is missing.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import (
        compute_portfolio_returns,
        breakpoint_options,
    )
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='MS')
    data = pd.DataFrame({
        'permno': range(1, 501),
        'date': np.repeat(dates, 5),
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
            "No 'breakpoint_options_secondary' specified in bivariate sort.",
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
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}.")

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

            data["portfolio"] = data.groupby(date_col, group_keys=False).apply(
                _assigner, include_groups=False
            )
            portfolio_returns = data
        else:
            filtered_data = data[data[date_col].dt.month == rebalancing_month]
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
            portfolio_data = portfolio_data[[id_col, date_col, "portfolio"]]
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
            filtered_data = data[data[date_col].dt.month == rebalancing_month]
            if len(filtered_data) == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = filtered_data.copy()
            portfolio_data["portfolio_secondary"] = portfolio_data.groupby(
                date_col, group_keys=False
            ).apply(_assign_sec, include_groups=False)
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
            filtered_data = data[data[date_col].dt.month == rebalancing_month]
            if len(filtered_data) == 0:
                raise ValueError(
                    f"No observations match 'rebalancing_month' = "
                    f"{rebalancing_month}."
                )
            portfolio_data = filtered_data.copy()
            portfolio_data["portfolio_secondary"] = portfolio_data.groupby(
                date_col, group_keys=False
            ).apply(_assign_sec, include_groups=False)
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
            d for d in all_dates if pd.Timestamp(d).month == rebalancing_month
        ]
        if not matching_dates:
            raise ValueError(
                f"No dates in data match for rebalancing month = "
                f"{rebalancing_month}."
            )
        first_rebalancing_date = min(matching_dates)
        all_dates = [d for d in all_dates if d >= first_rebalancing_date]

    all_portfolios = portfolio_returns["portfolio"].dropna().unique().tolist()
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
    """Compute long-short returns.

    Calculates long-short returns based on the returns of portfolios.
    The long-short return is computed as the difference between the
    returns of the 'top' and 'bottom' portfolios. The direction of the
    calculation can be adjusted based on whether the return from the
    'bottom' portfolio is subtracted from or added to the return from
    the 'top' portfolio.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame containing portfolio returns. Must include columns
        for the portfolio identifier, date, and return measurements
        (as specified in 'data_options').
    direction : str, default 'top_minus_bottom'
        Direction of the long-short return calculation. Must be either
        'top_minus_bottom' or 'bottom_minus_top'. If set to
        'bottom_minus_top', the return is computed as (bottom - top).
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element
        specifies the date column, the 'ret_excess' element specifies
        the excess return column, and 'portfolio' specifies the
        assigned portfolio. Uses the 'data_options' defaults when
        None: 'date' -> 'date', 'ret_excess' -> 'ret_excess', and
        'portfolio' -> 'portfolio'.

    Returns
    -------
    pd.DataFrame
        Data frame with columns for date and the computed long-short
        returns. The data frame is arranged by date and pivoted to
        have return measurement types as columns with their
        corresponding long-short returns.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import (
        compute_portfolio_returns,
        compute_long_short_returns,
        breakpoint_options,
    )
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='MS')
    data = pd.DataFrame({
        'permno': range(1, 101),
        'date': np.repeat(dates, 1),
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


def _require_column(
    data: pd.DataFrame, col: str, arg: str, info: str = None
) -> None:
    """Raise ValueError if col is not in data.columns."""
    if col not in data.columns:
        if info is None:
            info = f"Set {arg} to the correct column name."
        raise ValueError(f"Column '{col}' not found in 'data'. {info}")


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
    """Filter sorting data.

    Applies sample construction filters to a data frame before
    portfolio sorting. Filters are applied in a fixed order:
    financials exclusion, utilities exclusion, minimum stock price,
    minimum size quantile, minimum listing age, positive book equity,
    and positive earnings. An informational warning is emitted for
    each filter that actually removes at least one observation.

    Parameters
    ----------
    data : pd.DataFrame
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

    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'siccd' element
        specifies the SIC code column, 'price' specifies the (adjusted)
        price column, 'mktcap_lag' specifies the market capitalization
        column, 'date' specifies the date column, 'listing_age'
        specifies the listing age column, 'be' specifies the book
        equity column, and 'earnings' specifies the earnings column.
        Uses the 'data_options' defaults when None.
    quiet : bool, default False
        Whether informational messages should be suppressed.

    Returns
    -------
    pd.DataFrame
        Filtered data frame, preserving the class and structure of the
        input.

    Examples
    --------
    ```python
    import pandas as pd
    from tidyfinance import filter_sorting_data, filter_options
    data = pd.DataFrame({
        'permno': range(1, 6),
        'date': pd.Timestamp('2020-01-01'),
        'siccd': [6100, 2000, 4950, 3000, 6500],
        'prc_adj': [5, 0.5, 15, 20, 10],
    })
    filter_sorting_data(
        data,
        filter_options=filter_options(
            exclude_financials=True,
            min_stock_price=1,
        ),
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
            data = _filter_with_log(data, keep, "exclude_financials", quiet)

        if filter_options.get("exclude_utilities"):
            keep = data[col_siccd].isna() | ~(
                (data[col_siccd] >= 4900) & (data[col_siccd] <= 4999)
            )
            data = _filter_with_log(data, keep, "exclude_utilities", quiet)

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
        _require_column(data, col_listing_age, "data_options['listing_age']")
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
        data = _filter_with_log(data, keep, "exclude_negative_earnings", quiet)

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
    """Implement a portfolio sort.

    A convenience wrapper that combines sample construction filtering
    and portfolio return computation into a single call. Equivalent to
    calling 'filter_sorting_data' followed by
    'compute_portfolio_returns' with the filter and breakpoint
    specifications bundled in 'portfolio_sort_options'.

    Parameters
    ----------
    data : pd.DataFrame
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
        default) means monthly rebalancing.
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
        NaN. Set to 0 to deactivate the check.
    cap_weight : float, default 0.8
        Quantile of the cross-sectional 'mktcap_lag' distribution at
        which market capitalizations are capped per date when computing
        the capped value-weighted excess return ('ret_excess_vw_capped').
        Must be in [0, 1].
    data_options : dict, optional
        Column-name mapping (see 'data_options'). All elements are
        forwarded to 'filter_sorting_data' and
        'compute_portfolio_returns'. Uses the 'data_options' defaults
        when None.
    quiet : bool, default False
        Whether informational messages should be suppressed.

    Returns
    -------
    pd.DataFrame
        Data frame of portfolio returns as returned by
        'compute_portfolio_returns'.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import (
        implement_portfolio_sort,
        portfolio_sort_options,
        filter_options,
        breakpoint_options,
    )
    rng = np.random.default_rng(123)
    dates = pd.date_range('2020-01-01', periods=100, freq='MS')
    data = pd.DataFrame({
        'permno': range(1, 501),
        'date': np.repeat(dates, 5),
        'mktcap_lag': rng.uniform(100, 1000, 500),
        'ret_excess': rng.standard_normal(500),
        'prc_adj': rng.uniform(0.5, 50, 500),
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
