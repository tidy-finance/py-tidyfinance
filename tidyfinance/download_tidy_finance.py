"""Tidy Finance hosted data downloads for tidyfinance."""

import io
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import polars as pl
from curl_cffi import requests

from ._internal import _get_random_user_agent, _parse_date, _validate_dates

_SUPPORTED_DATASETS_HF = (
    "high_frequency_sp500",
    "factor_library",
    "factor_library_grid",
)

_FACTOR_LIBRARY_DEFAULTS = {
    "min_size_quantile": 0.2,
    "exclude_financials": False,
    "exclude_utilities": False,
    "exclude_negative_earnings": False,
    "sorting_variable_lag": "6m",
    "rebalancing": "monthly",
    "n_portfolios_main": 10,
    "sorting_method": "univariate",
    "n_portfolios_secondary": None,
    "breakpoints_exchanges": "NYSE",
    "breakpoints_min_size_threshold": None,
    "weighting_scheme": "VW",
}

_FACTOR_LIBRARY_SUPPORTED_FILTERS = (
    "sorting_variable",
    *_FACTOR_LIBRARY_DEFAULTS.keys(),
)

_FACTOR_LIBRARY_URL = (
    "https://huggingface.co/datasets/tidy-finance/factor-library/resolve/main"
)

_hf_session = requests.Session()


def _read_parquet_url(url: str) -> pl.DataFrame:
    """
    Fetch a parquet file over HTTPS and parse it with polars.

    Single-shot seam used wherever the pandas implementation called
    'pd.read_parquet(url)' directly. Tests patch this function.

    Parameters
    ----------
    url : str
        Fully qualified URL of the parquet file.

    Returns
    -------
    pl.DataFrame
        The parsed parquet content.
    """
    response = requests.get(
        url,
        headers={"User-Agent": _get_random_user_agent()},
        impersonate="chrome",
        timeout=120,
    )
    response.raise_for_status()
    return pl.read_parquet(io.BytesIO(response.content))


def _cast_date_column(data: pl.DataFrame, column: str = "date") -> pl.DataFrame:
    """
    Normalize a calendar-date column to 'pl.Date'.

    Handles columns stored as 'pl.Date', (tz-aware) 'pl.Datetime', or
    ISO-formatted strings so date comparisons work regardless of how
    the parquet file encoded the column.

    Parameters
    ----------
    data : pl.DataFrame
        Frame containing 'column'.
    column : str, default 'date'
        Name of the column to cast.

    Returns
    -------
    pl.DataFrame
        Frame with 'column' cast to 'pl.Date'.
    """
    dtype = data.schema[column]
    if dtype == pl.Date:
        return data
    if dtype == pl.String:
        return data.with_columns(pl.col(column).str.to_date())
    return data.with_columns(pl.col(column).cast(pl.Date))


def _download_data_risk_free(
    start_date: str = None,
    end_date: str = None,
    frequency: str = "monthly",
) -> pl.DataFrame:
    """
    Download risk-free rate data.

    Downloads pre-processed risk-free rate data from the
    'tidy-finance/risk-free' dataset on Hugging Face. The dataset is
    updated monthly via a scheduled GitHub Actions workflow that
    splices the 3-Month Treasury Bill Secondary Market Rate (pre-2001)
    with the 4-Week Treasury Bill Secondary Market Rate (from 2001
    onwards) sourced from FRED. For monthly data, the monthly TB3MS
    series is spliced with the daily DTB4WK series aggregated to
    month-end. For daily data, the daily DTB3 series is spliced with
    the daily DTB4WK series, both at the business-day frequency
    provided by FRED.

    Both series are quoted as annualised bank discount rates on a
    360-day basis. Given an annualised discount rate 'd' and a T-bill
    with 'n' days to maturity, the holding-period return is
    'HPR = d * n / 360 / (1 - d * n / 360)', which is then converted
    to the target period length via '(1 + HPR) ** (target / source) - 1'.

    The series are spliced at 2001-07-01. Pre-2001, TB3MS (monthly) or
    DTB3 (daily) is used (3-month T-bill, n = 90). Monthly conversion
    uses exponent '1 / 3'; daily conversion uses exponent '1 / 63'
    (approximate trading days per quarter). From 2001 onwards, DTB4WK
    (4-week T-bill, n = 28) is used. For monthly data, the last
    non-missing observation per calendar month is taken and the
    exponent is '365 / (28 * 12)'. For daily data, observations are
    used as-is and the exponent is '1 / 20' (approximate trading days
    per 4-week period).

    Business-day gaps in the daily series (e.g. holidays) are handled
    by forward-filling the most recent available rate. Monthly data
    starts in 1934-01-01 (TB3MS). Daily data starts in 1954-01-04
    because of the availability of DTB3.

    Parameters
    ----------
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    frequency : str, optional
        A character string, either 'monthly' (default) or 'daily',
        specifying the frequency of the returned data. Daily data
        starts in 1954-01-04 because of availability of DTB3, while
        monthly data starts in 1934-01-01.

    Returns
    -------
    pl.DataFrame
        A data frame with two columns:

        - 'date': the date of the observation.
        - 'risk_free': the risk-free rate for the period.

    Examples
    --------
    ```python
    from tidyfinance import download_data_risk_free
    download_data_risk_free('2020-01-01', '2020-12-31')
    download_data_risk_free(
        '2020-01-01', '2020-12-31', frequency='daily'
    )
    ```
    """
    if frequency not in ("monthly", "daily"):
        raise ValueError("frequency must be 'monthly' or 'daily'.")

    start_date, end_date = _validate_dates(start_date, end_date)

    url = (
        "https://huggingface.co/datasets/tidy-finance/risk-free/"
        f"resolve/main/risk_free_{frequency}.parquet"
    )

    try:
        risk_free_data = _read_parquet_url(url)

    except Exception as e:
        raise RuntimeError(
            "Failed to download risk-free rate data from HuggingFace. "
            f"URL attempted: {url}"
        ) from e

    risk_free_data = _cast_date_column(risk_free_data)

    # Filter each bound independently so an open-ended range works.
    if start_date is not None:
        risk_free_data = risk_free_data.filter(pl.col("date") >= start_date)
    if end_date is not None:
        risk_free_data = risk_free_data.filter(pl.col("date") <= end_date)

    return risk_free_data


# %% hugging face functions for tidy finance data


def _get_available_huggingface_files(
    organization: str, dataset: str
) -> pl.DataFrame:
    """
    List parquet files in a Hugging Face dataset.

    Queries the Hugging Face Datasets API and returns a data frame of
    files with a '.parquet' suffix. The function follows pagination
    links returned in the response 'Link' header and returns the path
    and size for each file. Requires internet access and the dataset
    to be publicly accessible or accessible with appropriate
    authentication.

    Parameters
    ----------
    organization : str
        Hugging Face organization or user name.
    dataset : str
        Dataset name under the organization.

    Returns
    -------
    pl.DataFrame
        A data frame with columns 'path' (str) and 'size' (int).

    Examples
    --------
    ```python
    from tidyfinance.download_tidy_finance import (
        _get_available_huggingface_files,
    )
    _get_available_huggingface_files('voigtstefan', 'sp500')
    ```
    """
    api_url = (
        f"https://huggingface.co/api/datasets/{organization}/{dataset}"
        "/tree/main?recursive=1"
    )
    paths = []
    sizes = []
    next_url = api_url

    while next_url:
        resp = requests.get(next_url, timeout=30)
        resp.raise_for_status()
        for entry in resp.json():
            is_file = entry.get("type") == "file"
            is_parquet = entry["path"].endswith(".parquet")
            if is_file and is_parquet:
                paths.append(entry["path"])
                sizes.append(entry.get("size"))
        link = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        next_url = match.group(1) if match else None

    return pl.DataFrame(
        {"path": paths, "size": sizes},
        schema={"path": pl.String, "size": pl.Int64},
    )


def _download_factor_library_grid() -> pl.DataFrame:
    """
    Download the factor library grid from Hugging Face.

    Returns the 'tidy-finance/factor-library-grid' dataset, which
    describes every portfolio construction available in the factor
    library (one row per construction, identified by 'id'). Use the
    returned data frame to discover which combinations of
    'sorting_variable', 'weighting_scheme', 'rebalancing', and other
    columns exist before requesting their returns with
    '_download_factor_library_ids'.

    Returns
    -------
    pl.DataFrame
        A data frame with one row per portfolio construction in the
        factor library, including the integer 'id' column used by
        '_download_factor_library_ids'.

    Raises
    ------
    ValueError
        If no parquet files are found in the
        'tidy-finance/factor-library-grid' repository.

    Examples
    --------
    ```python
    from tidyfinance.download_tidy_finance import (
        _download_factor_library_grid,
    )
    _download_factor_library_grid()
    ```
    """
    available = _get_available_huggingface_files(
        "tidy-finance", "factor-library-grid"
    )
    if available.is_empty() or "path" not in available.columns:
        raise ValueError(
            "No parquet files were found in the Hugging Face dataset repo "
            "'tidy-finance/factor-library-grid'."
        )
    grid_path = available["path"][0]
    grid_url = (
        "https://huggingface.co/datasets/tidy-finance"
        f"/factor-library-grid/resolve/main/{grid_path}"
    )
    return _read_parquet_url(grid_url)


def _filter_factor_library_grid(fill_all: bool = False, **filters) -> list:
    """
    Filter the factor-library grid and return matching portfolio IDs.

    Downloads the 'tidy-finance/factor-library-grid' dataset from Hugging
    Face and filters it by the provided column-value pairs. Columns not
    explicitly specified are held at sensible defaults when 'fill_all=False'.

    Parameters
    ----------
    fill_all : bool, optional
        If 'False' (default), columns absent from 'filters' are set to
        their defaults before filtering. If 'True', only the explicitly
        provided filters are applied and all other columns are left
        unrestricted.
    **filters : dict
        Named arguments of the form 'column=value' used to filter the
        grid. Each value may be a scalar or a list/tuple to match multiple
        levels. Passing 'None' for a column removes that filter entirely,
        returning all values for that column (e.g.,
        'min_size_quantile=None' includes all size groups). Supported
        columns and their defaults are:

        - 'sorting_variable': no default. When omitted, all sorting
          variables are returned (subject to the remaining defaults).
        - 'min_size_quantile': 0.2
        - 'exclude_financials': False
        - 'exclude_utilities': False
        - 'exclude_negative_earnings': False
        - 'sorting_variable_lag': "6m"
        - 'rebalancing': "monthly"
        - 'n_portfolios_main': 10
        - 'sorting_method': "univariate"
        - 'n_portfolios_secondary': None
        - 'breakpoints_exchanges': "NYSE"
        - 'breakpoints_min_size_threshold': None
        - 'weighting_scheme': "VW"

    Returns
    -------
    list
        Integer portfolio IDs matching the specified criteria. Returns an
        empty list when no rows satisfy the filters.

    Raises
    ------
    ValueError
        If an unrecognised filter name is provided.
    """
    unsupported = set(filters) - set(_FACTOR_LIBRARY_SUPPORTED_FILTERS)
    if unsupported:
        raise ValueError(
            f"Unsupported filter name(s): {sorted(unsupported)}. "
            f"Supported filters: {list(_FACTOR_LIBRARY_SUPPORTED_FILTERS)}."
        )

    if filters.get("sorting_method", "univariate") != "univariate":
        if filters.get("n_portfolios_secondary") is None:
            raise ValueError(
                "When sorting_method is not 'univariate', "
                "n_portfolios_secondary must be provided."
            )

    # A filter the caller explicitly sets to None is removed entirely so
    # all values for that column are returned, mirroring purrr::compact()
    # on NULL in the R implementation. These columns are recorded so the
    # defaults below do not reintroduce a filter for them. Default None
    # values (e.g. n_portfolios_secondary) are applied afterwards and
    # instead match rows where the column is null.
    unrestricted = {col for col, value in filters.items() if value is None}
    filters = {col: v for col, v in filters.items() if v is not None}

    if not fill_all:
        for col, default in _FACTOR_LIBRARY_DEFAULTS.items():
            if col not in filters and col not in unrestricted:
                filters[col] = default

    grid = _download_factor_library_grid()

    for col, value in filters.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if values == [None]:
            grid = grid.filter(pl.col(col).is_null())
        else:
            grid = grid.filter(pl.col(col).is_in(values))
    return grid["id"].to_list()


def _fetch_parquet_url(url, retries=5, backoff=2.0):
    """
    Fetch a parquet file via a reused curl_cffi session with retry.

    Returns None without retrying when the file does not exist (HTTP
    404). Other failures are retried with exponential backoff and raise
    a ConnectionError once all 'retries' attempts have failed.
    """
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_err = None
    for attempt in range(retries):
        try:
            r = _hf_session.get(url, headers=headers, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return pl.read_parquet(io.BytesIO(r.content))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    raise ConnectionError(
        f"Failed to download {url} after {retries} attempts: {last_err}"
    ) from last_err


def _factor_library_file(portfolio_id: int) -> str:
    """
    Name the factor library file that holds a portfolio ID.

    The returns are cut into files of 1,000 consecutive IDs named after
    the range they cover, so the file follows from the ID alone.

    Parameters
    ----------
    portfolio_id : int
        Portfolio ID from the factor library grid.

    Returns
    -------
    str
        The file name, e.g. 'id_0001001-0002000.parquet' for ID 1500.
    """
    id_first = (portfolio_id - 1) // 1000 * 1000 + 1
    return f"id_{id_first:07d}-{id_first + 999:07d}.parquet"


def _download_factor_library_ids(ids: list) -> pl.DataFrame:
    """
    Download factor library returns for a vector of portfolio IDs.

    Given a vector of portfolio IDs from the
    'tidy-finance/factor-library-grid' Hugging Face dataset, downloads
    the corresponding return data from the
    'tidy-finance/factor-library' dataset on Hugging Face. The returns
    are stored in files of 1,000 consecutive IDs named after the range
    they cover (e.g., 'id_0000001-0001000.parquet'), so the function
    downloads only the files that hold the requested IDs, each once.
    The grid metadata is joined onto the result.

    Use this function when you already know the portfolio IDs you want
    (for example, from a previous call to '_download_data_huggingface'
    with 'dataset' set to 'factor_library'). To resolve IDs from
    filter criteria (sorting variable, weighting scheme, breakpoints,
    etc.) and download in a single call, use
    '_download_data_huggingface' instead.

    Raises an error if 'ids' is empty or if none of the requested IDs
    exist in the grid. IDs whose portfolio sort produced no portfolios
    have no returns; they are absent from the result, with a warning.
    Returns are stored in single precision and returned as 64-bit
    floats, and months without a valid long-short return are stored
    as 0.

    Parameters
    ----------
    ids : list of int
        Portfolio IDs to download. IDs correspond to rows of the
        'tidy-finance/factor-library-grid' dataset.

    Returns
    -------
    pl.DataFrame
        A data frame with the columns 'id', 'date', and 'ret' (the
        monthly long-short excess return) and the grid metadata
        columns for the requested IDs.

    Raises
    ------
    ValueError
        If 'ids' is empty, or if none of the requested IDs exist in the
        factor library grid.

    Warns
    -----
    UserWarning
        If some of the requested IDs have no returns in the factor
        library.

    Examples
    --------
    ```python
    from tidyfinance.download_tidy_finance import (
        _download_factor_library_ids,
    )
    _download_factor_library_ids([1, 2, 3])
    ```
    """
    if not ids:
        raise ValueError(
            "No portfolio IDs provided. "
            "Check that your filter criteria match at least one portfolio."
        )

    id_grid = _download_factor_library_grid().filter(pl.col("id").is_in(ids))
    if id_grid.is_empty():
        raise ValueError(
            "None of the requested portfolio IDs exist in the factor library "
            "grid. Check that the provided 'ids' are valid and exist in the "
            "factor library grid."
        )

    requested = id_grid["id"].to_list()
    files = list(dict.fromkeys(map(_factor_library_file, requested)))

    def _fetch_returns(file):
        data = _fetch_parquet_url(f"{_FACTOR_LIBRARY_URL}/{file}")
        # No file exists for a range in which no sort produced portfolios.
        if data is None:
            return None
        # Each file holds 1,000 series; keep only the requested ones.
        return data.filter(pl.col("id").is_in(requested))

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [f for f in ex.map(_fetch_returns, files) if f is not None]

    if frames:
        returns = pl.concat(frames, how="diagonal_relaxed")
    else:
        returns = pl.DataFrame(
            schema={
                "id": id_grid.schema["id"],
                "date": pl.Date,
                "ret": pl.Float32,
            }
        )
    # Returns are stored in single precision; widen them to the double
    # precision the R package returns.
    returns = returns.with_columns(pl.col("ret").cast(pl.Float64))

    no_returns = (
        id_grid.select("id")
        .join(returns.select("id").unique(), on="id", how="anti")
        .sort("id")["id"]
    )
    if not no_returns.is_empty():
        shown = ", ".join(map(str, no_returns.head(5).to_list()))
        if no_returns.len() > 5:
            shown += f", ... ({no_returns.len():,} IDs)"
        warnings.warn(
            "Portfolio IDs without returns in the factor library, because "
            "their portfolio sort produced no portfolios, are absent from "
            f"the result: {shown}.",
            stacklevel=2,
        )

    return returns.join(id_grid, on="id", how="inner", maintain_order="left")


def _download_data_huggingface_factor_library(
    fill_all: bool = False,
    ids: list = None,
    start_date: str = None,
    end_date: str = None,
    **filters,
) -> pl.DataFrame:
    """
    Download factor library data from Hugging Face.

    Thin wrapper that resolves portfolio IDs from the grid (or uses an
    explicit ids list) and then downloads the corresponding return
    data.

    Parameters
    ----------
    fill_all : bool, optional
        Forwarded to _filter_factor_library_grid. When True, columns
        not specified in filters are left unrestricted rather than set
        to their defaults. Defaults to False.
    ids : list, optional
        Explicit portfolio IDs to download. When provided, the grid
        filter is skipped. Cannot be combined with filters.
    start_date : str, optional
        Inclusive lower bound for the returned data's date column,
        in YYYY-MM-DD format. Defaults to None (no lower bound).
    end_date : str, optional
        Inclusive upper bound for the returned data's date column,
        in YYYY-MM-DD format. Defaults to None (no upper bound).
    **filters : dict
        Named filter arguments forwarded to
        _filter_factor_library_grid.

    Returns
    -------
    pl.DataFrame
        Portfolio returns with grid metadata columns appended.

    Raises
    ------
    ValueError
        If both ids and filter arguments are supplied.
    """
    if ids is not None and filters:
        raise ValueError("'ids' cannot be combined with filter arguments.")

    if ids is not None:
        result = _download_factor_library_ids(ids)
    else:
        resolved_ids = _filter_factor_library_grid(fill_all=fill_all, **filters)
        result = _download_factor_library_ids(resolved_ids)

    # Normalize the date column to pl.Date so comparisons work
    # regardless of whether the parquet stored dates as date,
    # datetime, tz-aware datetime, or string.
    if start_date is not None or end_date is not None:
        result = _cast_date_column(result)

    if start_date is not None:
        result = result.filter(pl.col("date") >= _parse_date(start_date))
    if end_date is not None:
        result = result.filter(pl.col("date") <= _parse_date(end_date))

    return result


def _download_data_huggingface(
    dataset: str = None,
    start_date: str | date = None,
    end_date: str | date = None,
    type: str = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Download data from a Hugging Face dataset.

    Downloads data from a supported Hugging Face dataset. For
    'high_frequency_sp500', parquet files are filtered by date range
    and row-bound. For 'factor_library', portfolio characteristics are
    selected via '_filter_factor_library_grid', the matching return
    data is downloaded, and the result is filtered to 'start_date' and
    'end_date' when both are supplied. For 'factor_library_grid', the
    grid itself is returned via '_download_factor_library_grid'.

    For 'dataset' set to 'factor_library', the defaults below reflect
    one common portfolio construction choice but may not suit every
    research question; always verify that the selected combination
    matches the intended design. Supported filter columns and their
    defaults are:

    - 'sorting_variable': optional. The firm characteristic used to
      sort stocks into portfolios, named like the Open Source Asset
      Pricing signals (e.g., 'size' for market equity, 'bm' for
      book-to-market). See the 'factor_library_grid' dataset for all
      values. No default is applied; when omitted, all sorting
      variables are returned (subject to the remaining defaults).
    - 'min_size_quantile' (defaults to 0.2): fraction of the smallest
      stocks (by market cap) excluded from the portfolio universe; 0.2
      drops the bottom 20%.
    - 'exclude_financials' (defaults to False): whether to drop
      financial-sector stocks (SIC 6000-6799) from the universe.
    - 'exclude_utilities' (defaults to False): whether to drop
      utility-sector stocks (SIC 4900-4999) from the universe.
    - 'exclude_negative_earnings' (defaults to False): whether to drop
      firms with negative earnings before sorting.
    - 'sorting_variable_lag' (defaults to '6m'): lag applied to the
      sorting variable before portfolio assignment: '1m' (the timing
      of Open Source Asset Pricing), '3m', '6m', or 'ff'
      (Fama-French).
    - 'rebalancing' (defaults to 'monthly'): how frequently portfolios
      are reformed; 'monthly' or 'annual'.
    - 'n_portfolios_main' (defaults to 10): number of quantile groups
      (e.g., 10 for decile portfolios).
    - 'sorting_method' (defaults to 'univariate'): whether portfolios
      are formed on a single sort ('univariate') or on a double sort
      with size as the second variable ('bivariate-dependent' or
      'bivariate-independent').
    - 'n_portfolios_secondary' (defaults to None): number of size
      groups for the secondary sort. Required when 'sorting_method'
      is not 'univariate'.
    - 'breakpoints_exchanges' (defaults to 'NYSE'): exchange(s) used
      to compute breakpoints; 'NYSE' uses only NYSE-listed stocks to
      define quantile cutoffs (the conventional Fama-French approach).
    - 'breakpoints_min_size_threshold' (defaults to None): minimum
      size quantile of the stocks that set the main breakpoints (e.g.,
      0.2). None means no minimum-size screen is applied.
    - 'weighting_scheme' (defaults to 'VW'): return weighting within
      portfolios; 'VW' for value-weighted, 'EW' for equal-weighted, or
      'capped VW' for value-weighted with capped weights.

    Passing 'None' for any filter column removes that filter entirely,
    returning all values for that column (e.g., 'min_size_quantile=None'
    includes all size groups).

    Parameters
    ----------
    dataset : str
        The dataset to download. Supported values are
        'high_frequency_sp500', 'factor_library', and
        'factor_library_grid'.
    start_date : str or date, optional
        Start date (inclusive) in 'YYYY-MM-DD' format. Used for
        'high_frequency_sp500' to filter parquet files by date, and
        forwarded to 'factor_library' as a date-range lower bound. For
        'high_frequency_sp500' defaults to the available sample's start
        ('2007-06-27') when not supplied; for 'factor_library', None
        returns the full history.
    end_date : str or date, optional
        End date (inclusive) in 'YYYY-MM-DD' format. Used for
        'high_frequency_sp500' to filter parquet files by date, and
        forwarded to 'factor_library' as a date-range upper bound. For
        'high_frequency_sp500' defaults to the available sample's end
        ('2007-07-27') when not supplied; for 'factor_library', None
        returns the full history.
    type : str, optional
        Deprecated. Use 'dataset' instead. If provided, emits a
        DeprecationWarning and strips any leading 'hf_' prefix.
    **kwargs : dict
        For 'dataset' set to 'factor_library': either named arguments
        used to filter the portfolio grid, or 'ids=<vector>' to bypass
        the grid filter and download specific portfolios directly via
        '_download_factor_library_ids'. Filter arguments take the form
        'column=value', where 'value' may be a list or tuple to match
        multiple levels. Optionally pass 'fill_all=True' to leave
        unspecified columns unrestricted (default False, i.e.,
        unspecified columns are fixed at the defaults listed above).
        Passing None for any parameter removes that filter entirely,
        returning all values for that column. Passing an unrecognised
        column name raises a 'ValueError'. 'ids' cannot be combined
        with filter arguments. Ignored when 'dataset' is not
        'factor_library'.

    Returns
    -------
    pl.DataFrame
        A data frame with the downloaded data. For
        'high_frequency_sp500', contains 5-second aggregated order-book
        snapshots filtered to the requested date range. For
        'factor_library', contains the columns 'id', 'date', and 'ret'
        joined with the full grid metadata for the matched portfolio
        IDs.

    Raises
    ------
    ValueError
        If 'dataset' is None, unsupported, or if invalid filter names
        are passed for the factor library.

    Examples
    --------
    ```python
    from tidyfinance.download_tidy_finance import (
        _download_data_huggingface,
    )
    _download_data_huggingface(
        'high_frequency_sp500', '2007-07-26', '2007-07-27'
    )
    _download_data_huggingface(
        'factor_library',
        sorting_variable='high52',
        rebalancing='annual',
    )
    _download_data_huggingface(
        'factor_library', sorting_variable='assetgrowth', fill_all=True
    )
    _download_data_huggingface(
        'factor_library',
        sorting_variable='size',
        start_date='2000-01-01',
        end_date='2020-12-31',
    )
    _download_data_huggingface('factor_library', ids=[1, 2, 3])
    ```
    """
    if type is not None:
        warnings.warn(
            "'type' is deprecated; use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = type[len("hf_") :] if type.startswith("hf_") else type

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    if dataset.startswith("hf_"):
        warnings.warn(
            "Passing a dataset name with the 'hf_' prefix is deprecated. "
            "Use 'dataset' without the prefix instead "
            "(e.g. high_frequency_sp500 instead of hf_high_frequency_sp500).",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = dataset[len("hf_") :]

    if dataset not in _SUPPORTED_DATASETS_HF:
        raise ValueError(
            f"Unsupported Hugging Face dataset: '{dataset}'. "
            f"Supported datasets: {_SUPPORTED_DATASETS_HF}."
        )

    if dataset == "high_frequency_sp500":
        organization = "voigtstefan"
        dataset_name = "sp500"

        # 'high_frequency_sp500' is hosted as a one-month sample on
        # HuggingFace. If the caller does not supply a window, default
        # to the full available range so the function returns sensible
        # data rather than an empty frame.
        if start_date is None:
            start_date = "2007-06-27"
        if end_date is None:
            end_date = "2007-07-27"

        available_files = _get_available_huggingface_files(
            organization, dataset_name
        )
        available_files = available_files.with_columns(
            date=pl.col("path")
            .str.extract(r"date=(\d{4}-\d{2}-\d{2})", 1)
            .str.to_date()
        )

        # The requested window is a contiguous daily range, so a bounds
        # check is equivalent to membership in the expanded date list and
        # avoids polars' ambiguous 'is_in' against a Series of the same
        # dtype (deprecated in polars >= 1.31).
        files_to_download = available_files.filter(
            pl.col("date").is_between(
                _parse_date(start_date), _parse_date(end_date)
            )
        )

        if files_to_download.is_empty():
            return pl.DataFrame()

        return pl.concat(
            [
                _read_parquet_url(
                    f"https://huggingface.co/datasets/{organization}"
                    f"/{dataset_name}/resolve/main/{path}"
                )
                for path in files_to_download["path"]
            ],
            how="diagonal_relaxed",
        )

    if dataset == "factor_library_grid":
        return _download_factor_library_grid()

    if dataset == "factor_library":
        return _download_data_huggingface_factor_library(
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

    raise ValueError(f"Unsupported dataset: '{dataset}'.")
