"""Tidy Finance hosted data downloads for tidyfinance."""

import io
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import polars as pl
from curl_cffi import requests

from ._internal import (
    _deprecate_legacy_dataset,
    _get_random_user_agent,
    _inform,
    _parse_date,
    _user_stacklevel,
    _validate_dates,
)

_SUPPORTED_DATASETS_HF = (
    "high_frequency_sp500",
    "factor_library",
    "factor_library_costs",
    "factor_library_grid",
    "factor_library_persistence",
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

_hf_session = requests.Session()


def _parse_parquet(content: bytes) -> pl.DataFrame:
    """
    Parse the bytes of a parquet file into a polars data frame.

    The file is read through pyarrow ('use_pyarrow=True'). Files written
    with nanoparquet, such as the factor library, store single-precision
    columns as 'float' but declare them as 'double' in the embedded
    Arrow schema. The native polars reader takes the type from that
    schema and fails on the mismatch; pyarrow reads the stored type.

    Parameters
    ----------
    content : bytes
        Content of the parquet file.

    Returns
    -------
    pl.DataFrame
        The parsed parquet content.
    """
    return pl.read_parquet(io.BytesIO(content), use_pyarrow=True)


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
    return _parse_parquet(response.content)


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


# Session cache for the factor library grid, one grid per revision. The
# grid has more than four million rows, so it is downloaded at most once
# per session and revision by '_download_factor_library_grid'.
_factor_library_cache: dict = {}

_FACTOR_LIBRARY_RETURNS_SCHEMA = {
    "id": pl.Int32,
    "date": pl.Date,
    "ret": pl.Float64,
}

_FACTOR_LIBRARY_COSTS_SCHEMA = {
    "id": pl.Int32,
    "date": pl.Date,
    "turnover": pl.Float64,
    "cost": pl.Float64,
    "microcap_share": pl.Float64,
}

# Hugging Face allows anonymous users 3,000 files per five minutes and IP
# address. Downloads run 16 at a time and 8 per second on average, also
# across calls, which keeps them below that limit.
_FACTOR_LIBRARY_MAX_ACTIVE = 16
_FACTOR_LIBRARY_MAX_TRIES = 3


class _Throttle:
    """Token bucket that limits requests, shared across threads and calls.

    Holds up to 'capacity' tokens and refills them over 'fill_time_s'
    seconds. Each request takes one token and waits while none is left.
    """

    def __init__(self, capacity: int, fill_time_s: float):
        self.capacity = capacity
        self.rate = capacity / fill_time_s
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity,
                    self.tokens + (now - self.updated) * self.rate,
                )
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                delay = (1 - self.tokens) / self.rate
            time.sleep(delay)


_hf_throttle = _Throttle(capacity=16, fill_time_s=2)


class _RateLimited(Exception):
    """Hugging Face refused a file with HTTP 429."""

    def __init__(self, seconds: int):
        super().__init__(f"Rate limited; resets in {seconds} seconds.")
        self.seconds = seconds


class _RevisionNotFound(Exception):
    """The requested revision does not exist in the dataset."""


def _rate_limit_reset(response) -> int:
    """Return the seconds until the Hugging Face rate limit resets.

    Reads 't=<seconds>' from the 'RateLimit' header. Without the header,
    a full window of five minutes resets the limit. One more second
    avoids a retry just before the reset.
    """
    header = response.headers.get("RateLimit", "") or ""
    match = re.search(r"t=([0-9]+)", header)
    seconds = int(match.group(1)) if match else 300
    return seconds + 1


def _wait_for_rate_limit(seconds: int) -> None:
    """Say how long the download waits for the rate limit, then wait."""
    _inform(
        "Hugging Face limits the downloads per five minutes. Waiting "
        f"{seconds} second{'s' if seconds != 1 else ''} for the limit to "
        "reset."
    )
    time.sleep(seconds)


def _check_revision(revision, arg: str) -> None:
    """Raise unless 'revision' is a single non-empty string."""
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            f"'{arg}' must be a single string: a commit hash, a branch, "
            "or a tag."
        )


def _factor_library_url(dataset: str, revision: str, path: str) -> str:
    """Return the URL of a file of a factor library dataset at a revision.

    The Hub expects the slashes of a branch such as 'refs/pr/1' encoded.
    """
    return (
        f"https://huggingface.co/datasets/tidy-finance/{dataset}/resolve/"
        f"{revision.replace('/', '%2F')}/{path}"
    )


def _fetch_parquet_url(url, retries=5, backoff=2.0):
    """
    Fetch a parquet file via a reused curl_cffi session with retry.

    Returns None without retrying when the file does not exist (HTTP
    404). Raises '_RevisionNotFound' when the revision in the URL does
    not exist, and '_RateLimited' without retrying when Hugging Face
    refuses the file because of its rate limit (HTTP 429), so the
    caller can wait for the limit to reset. Other download failures are
    retried with exponential backoff and raise a ConnectionError once
    all 'retries' attempts have failed. The file is parsed once, after
    the download, so a file that cannot be read raises right away
    instead of being downloaded again. Every request waits for the
    shared download throttle.
    """
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_err = None
    for attempt in range(retries):
        try:
            _hf_throttle.wait()
            r = _hf_session.get(url, headers=headers, timeout=120)
            if r.status_code == 404:
                if r.headers.get("X-Error-Code") == "RevisionNotFound":
                    raise _RevisionNotFound(url)
                return None
            if r.status_code == 429:
                raise _RateLimited(_rate_limit_reset(r))
            r.raise_for_status()
            content = r.content
            break
        except (_RevisionNotFound, _RateLimited):
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    else:
        raise ConnectionError(
            f"Failed to download {url} after {retries} attempts: {last_err}"
        ) from last_err
    return _parse_parquet(content)


def _read_factor_library_files(
    dataset: str, revision: str, paths: list
) -> list:
    """Download files of a factor library dataset at a revision.

    Downloads up to 16 files at once. Files that Hugging Face refuses
    because of its rate limit are downloaded again after the limit
    resets, twice at most.

    Parameters
    ----------
    dataset : str
        'factor-library' or 'factor-library-grid'.
    revision : str
        Commit hash, branch, or tag of the dataset.
    paths : list of str
        Paths of the files within the dataset.

    Returns
    -------
    list
        One data frame per path, in the order of 'paths', or None for a
        file that does not exist.

    Raises
    ------
    ValueError
        If the dataset has no such revision.
    RuntimeError
        If the rate limit still refuses a file after three attempts.
    """
    urls = [_factor_library_url(dataset, revision, p) for p in paths]
    results: list = [None] * len(urls)
    pending = list(range(len(urls)))

    def fetch(i):
        try:
            return i, _fetch_parquet_url(urls[i]), None
        except _RateLimited as e:
            return i, None, e

    for attempt in range(1, _FACTOR_LIBRARY_MAX_TRIES + 1):
        refused = []
        try:
            with ThreadPoolExecutor(
                max_workers=_FACTOR_LIBRARY_MAX_ACTIVE
            ) as ex:
                for i, data, limited in ex.map(fetch, pending):
                    if limited is not None:
                        refused.append((i, limited))
                    else:
                        results[i] = data
        except _RevisionNotFound:
            raise ValueError(
                f"Revision '{revision}' is not in 'tidy-finance/{dataset}'."
            ) from None
        if not refused:
            break
        pending = [i for i, _ in refused]
        seconds = refused[0][1].seconds
        if attempt == _FACTOR_LIBRARY_MAX_TRIES:
            raise RuntimeError(
                "Hugging Face refused the download because of its rate "
                "limit. Anonymous users may download 3,000 files per five "
                "minutes and IP address, which includes the downloads of "
                "other programs from the same address. The limit resets "
                f"in {seconds} second{'s' if seconds != 1 else ''}. Try "
                "again then."
            ) from refused[0][1]
        _wait_for_rate_limit(seconds)

    return results


def _read_factor_library_file(
    dataset: str, revision: str, path: str
) -> pl.DataFrame:
    """Download one file of a factor library dataset; raise if missing."""
    data = _read_factor_library_files(dataset, revision, [path])[0]
    if data is None:
        raise ValueError(
            f"'{path}' is not in revision '{revision}' of "
            f"'tidy-finance/{dataset}'. Check that the revision is a "
            "commit of this dataset."
        )
    return data


def _download_factor_library_grid(
    refresh: bool = False, grid_revision: str = "main"
) -> pl.DataFrame:
    """
    Download the factor library grid from Hugging Face.

    Returns the grid of the 'tidy-finance/factor-library-grid' dataset,
    which describes every portfolio construction available in the
    factor library (one row per construction, identified by 'id'). Use
    the returned data frame to discover which combinations of
    'sorting_variable', 'weighting_scheme', 'rebalancing', and other
    columns exist before requesting their returns with
    'download_factor_library_ids'. The grid is read by name from
    'portfolio_sort_grid.parquet', without listing the files of the
    dataset.

    The grid is downloaded once per Python session and revision and
    cached in memory, so repeated calls (including those made by
    'download_factor_library_ids' and
    'download_data("Tidy Finance", "factor_library", ...)') reuse it.
    Set 'refresh=True' to download it again.

    Parameters
    ----------
    refresh : bool, default False
        If True, the grid is downloaded again and replaces the copy
        cached in the current session.
    grid_revision : str, default 'main'
        Commit hash, branch, or tag of the Hugging Face dataset
        'tidy-finance/factor-library-grid' to read the grid from. Pin a
        commit to reproduce results with a fixed release of the factor
        library.

    Returns
    -------
    pl.DataFrame
        A data frame with one row per portfolio construction in the
        factor library, including the integer 'id' column used by
        'download_factor_library_ids'.

    Raises
    ------
    ValueError
        If 'refresh' is not True or False, if 'grid_revision' is not a
        single string, or if the revision does not exist or holds no
        grid.

    Examples
    --------
    ```python
    from tidyfinance import download_factor_library_grid
    grid = download_factor_library_grid()
    grid = download_factor_library_grid(refresh=True)
    ```
    """
    if not isinstance(refresh, bool):
        raise ValueError("'refresh' must be True or False.")
    _check_revision(grid_revision, "grid_revision")

    grids = _factor_library_cache.setdefault("grids", {})
    if refresh or grid_revision not in grids:
        # The repo also holds one slice of the grid per sorting variable
        # and the list of sorting variables, so the grid is read by its
        # file name instead of from the file listing of the repo.
        # Assigned only after a successful download, so a failed
        # download is not cached.
        grids[grid_revision] = _read_factor_library_file(
            "factor-library-grid",
            grid_revision,
            "portfolio_sort_grid.parquet",
        )
    return grids[grid_revision]


def _filter_factor_library_grid(
    fill_all: bool = False, grid_revision: str = "main", **filters
) -> list:
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
    grid_revision : str, default 'main'
        Commit hash, branch, or tag of the grid dataset to filter.
    **filters : dict
        Named arguments of the form 'column=value' used to filter the
        grid. Each value may be a scalar or a list/tuple to match multiple
        levels. Passing 'None' for a column removes that filter entirely,
        returning all values for that column (e.g.,
        'min_size_quantile=None' includes all size groups). Passing
        '[None]' selects the missing level instead, e.g.
        'min_size_quantile=[None]' for the portfolios without a size
        screen. Supported columns and their defaults are:

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
    # all values for that column are returned. These columns are recorded
    # so the defaults below do not reintroduce a filter for them. Default None
    # values (e.g. n_portfolios_secondary) are applied afterwards and
    # instead match rows where the column is null.
    unrestricted = {col for col, value in filters.items() if value is None}
    filters = {col: v for col, v in filters.items() if v is not None}

    if not fill_all:
        for col, default in _FACTOR_LIBRARY_DEFAULTS.items():
            if col not in filters and col not in unrestricted:
                filters[col] = default

    grid = _download_factor_library_grid(grid_revision=grid_revision)
    for col, value in filters.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if values == [None]:
            grid = grid.filter(pl.col(col).is_null())
        else:
            grid = grid.filter(pl.col(col).is_in(values))

    return grid["id"].to_list()


def _factor_library_file(portfolio_id: int) -> str:
    """
    Return the name of the factor library file that holds an ID.

    The returns are stored in files of 1,000 consecutive IDs named
    after the range they cover, e.g. 'id_0000001-0001000.parquet' for
    IDs 1 to 1,000. The costs are stored in files of the same names in
    the 'costs/' folder.

    Parameters
    ----------
    portfolio_id : int
        A portfolio ID of the factor library grid.

    Returns
    -------
    str
        The file name.
    """
    id_first = (portfolio_id - 1) // 1000 * 1000 + 1
    return f"id_{id_first:07d}-{id_first + 999:07d}.parquet"


def _download_factor_library_table(
    ids: list, table: str, revision: str, grid_revision: str
) -> pl.DataFrame:
    """Download the returns or costs of portfolio IDs, joined with the grid.

    Parameters
    ----------
    ids : list of int
        Portfolio IDs.
    table : {'returns', 'costs'}
        The table to download.
    revision : str
        Revision of 'tidy-finance/factor-library'.
    grid_revision : str
        Revision of 'tidy-finance/factor-library-grid'.
    """
    _check_revision(revision, "revision")
    if not ids:
        raise ValueError(
            "No portfolio IDs provided. "
            "Check that your filter criteria match at least one portfolio."
        )

    id_grid = _download_factor_library_grid(grid_revision=grid_revision).filter(
        pl.col("id").is_in(ids)
    )
    if id_grid.is_empty():
        raise ValueError(
            "None of the requested portfolio IDs exist in the factor library "
            "grid. Check that the provided 'ids' are valid and exist in the "
            "factor library grid."
        )

    requested = id_grid["id"].to_list()
    files = list(dict.fromkeys(map(_factor_library_file, requested)))
    if table == "costs":
        files = [f"costs/{f}" for f in files]

    # The library has no file for a range of IDs in which no portfolio
    # sort produced portfolios, so such a file is missing.
    frames = [
        # Each file holds 1,000 series; keep only the requested ones.
        f.filter(pl.col("id").is_in(requested))
        for f in _read_factor_library_files("factor-library", revision, files)
        if f is not None
    ]
    schema = (
        _FACTOR_LIBRARY_RETURNS_SCHEMA
        if table == "returns"
        else _FACTOR_LIBRARY_COSTS_SCHEMA
    )
    if frames:
        data = pl.concat(frames, how="diagonal_relaxed")
    else:
        data = pl.DataFrame(schema={**schema, "id": id_grid.schema["id"]})
    # Values are stored in single precision; return them in double
    # precision.
    data = data.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c, t in schema.items()
            if t == pl.Float64 and c in data.columns
        ]
    )

    no_data = (
        id_grid.select("id")
        .join(data.select("id").unique(), on="id", how="anti")
        .sort("id")["id"]
    )
    if not no_data.is_empty():
        shown = ", ".join(map(str, no_data.head(5).to_list()))
        if no_data.len() > 5:
            shown += f", ... ({no_data.len():,} IDs)"
        warnings.warn(
            f"Portfolio IDs without {table} in the factor library, because "
            "their portfolio sort produced no portfolios, are absent from "
            f"the result: {shown}.",
            stacklevel=_user_stacklevel(),
        )

    return data.join(id_grid, on="id", how="inner", maintain_order="left")


def _download_factor_library_ids(
    ids: list, revision: str = "main", grid_revision: str = "main"
) -> pl.DataFrame:
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
    (for example, from a previous call to 'download_data' with
    'dataset' set to 'factor_library'). To resolve IDs from filter
    criteria (sorting variable, weighting scheme, breakpoints, etc.)
    and download in a single call, use 'download_data_huggingface'
    instead.

    Raises an error if 'ids' is empty or if none of the requested IDs
    exist in the grid. IDs whose portfolio sort produced no portfolios
    have no returns; they are absent from the result, with a warning.
    Returns are stored in single precision and returned as 64-bit
    floats, and months without a valid long-short return are stored
    as 0.

    Up to 16 files are downloaded at once and 8 per second on average,
    also across calls, which keeps the download below the limit of
    Hugging Face for anonymous users of 3,000 files per five minutes.
    When Hugging Face still refuses a file, the download waits until
    the limit resets and tries again, twice at most.

    Parameters
    ----------
    ids : list of int
        Portfolio IDs to download. IDs correspond to rows of the
        'tidy-finance/factor-library-grid' dataset.
    revision : str, default 'main'
        Commit hash, branch, or tag of the Hugging Face dataset
        'tidy-finance/factor-library', which holds the returns.
    grid_revision : str, default 'main'
        Commit hash, branch, or tag of the Hugging Face dataset
        'tidy-finance/factor-library-grid', which holds the grid.

    Returns
    -------
    pl.DataFrame
        A data frame with the columns 'id', 'date', and 'ret' (the
        monthly long-short excess return) and the grid metadata
        columns for the requested IDs.

    Raises
    ------
    ValueError
        If 'ids' is empty, if none of the requested IDs exist in the
        factor library grid, or if a revision does not exist.
    RuntimeError
        If the rate limit of Hugging Face refuses a file three times.

    Warns
    -----
    UserWarning
        If some of the requested IDs have no returns in the factor
        library.

    Examples
    --------
    ```python
    from tidyfinance import download_factor_library_ids
    returns = download_factor_library_ids([1, 2, 3])
    ```
    """
    return _download_factor_library_table(
        ids, "returns", revision, grid_revision
    )


def _download_factor_library_persistence(
    sorting_variable=None,
    grid_revision: str = "main",
    start_date=None,
    end_date=None,
) -> pl.DataFrame:
    """Download how up to date each sorting variable of the library is.

    Reads 'sorting_variable_persistence.parquet' of the Hugging Face
    dataset 'tidy-finance/factor-library-grid', which describes for each
    sorting variable how up to date it is under each lag and
    rebalancing frequency.

    Parameters
    ----------
    sorting_variable : str or list of str, optional
        Sorting variables to keep. All are returned when None.
    grid_revision : str, default 'main'
        Commit hash, branch, or tag of the grid dataset.
    start_date, end_date : str or datetime.date, optional
        Inclusive date range to keep. The full history is returned when
        both are None.

    Raises
    ------
    ValueError
        If a sorting variable is unknown or the revision does not exist.
    """
    _check_revision(grid_revision, "grid_revision")
    start_date, end_date = _validate_dates(start_date, end_date)

    persistence = _read_factor_library_file(
        "factor-library-grid",
        grid_revision,
        "sorting_variable_persistence.parquet",
    )

    if sorting_variable is not None:
        selected = (
            [sorting_variable]
            if isinstance(sorting_variable, str)
            else list(sorting_variable)
        )
        known = set(persistence["sorting_variable"].to_list())
        unknown = [v for v in selected if v not in known]
        if unknown:
            raise ValueError(
                f"Unknown sorting variable(s): {unknown}. See "
                "'download_factor_library_grid' for all values."
            )
        persistence = persistence.filter(
            pl.col("sorting_variable").is_in(selected)
        )

    if start_date is not None and end_date is not None:
        persistence = _cast_date_column(persistence).filter(
            pl.col("date").is_between(start_date, end_date)
        )

    return persistence


def _download_data_huggingface_factor_library(
    fill_all: bool = False,
    ids: list = None,
    start_date: str = None,
    end_date: str = None,
    table: str = "returns",
    revision: str = "main",
    grid_revision: str = "main",
    **filters,
) -> pl.DataFrame:
    """
    Download factor library data from Hugging Face.

    Thin wrapper that resolves portfolio IDs from the grid (or uses an
    explicit ids list) and then downloads the corresponding returns or
    costs.

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
    table : {'returns', 'costs'}, default 'returns'
        The table of the factor library to download.
    revision : str, default 'main'
        Revision of 'tidy-finance/factor-library'.
    grid_revision : str, default 'main'
        Revision of 'tidy-finance/factor-library-grid'.
    **filters : dict
        Named filter arguments forwarded to
        _filter_factor_library_grid.

    Returns
    -------
    pl.DataFrame
        Portfolio returns or costs with grid metadata columns appended.

    Raises
    ------
    ValueError
        If both ids and filter arguments are supplied.
    """
    if ids is not None and filters:
        raise ValueError("'ids' cannot be combined with filter arguments.")

    if ids is None:
        ids = _filter_factor_library_grid(
            fill_all=fill_all, grid_revision=grid_revision, **filters
        )
    result = _download_factor_library_table(ids, table, revision, grid_revision)

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
    'end_date' when both are supplied. 'factor_library_costs' downloads
    the turnover, the trading cost, and the microcap share of each
    portfolio from January 1994, with the same filters and 'ids' as
    'factor_library' and the grid joined in the same way. For
    'factor_library_grid', the grid itself is returned via
    'download_factor_library_grid'. 'factor_library_persistence'
    describes how up to date each sorting variable is under each lag
    and rebalancing frequency, optionally for some 'sorting_variable's
    only.

    The factor library downloads can be pinned to a release:
    'revision' selects the commit of the Hugging Face dataset
    'tidy-finance/factor-library', which holds the returns and the
    costs, and 'grid_revision' the commit of
    'tidy-finance/factor-library-grid', which holds the grid and the
    persistence. Both default to 'main'.

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
    includes all size groups). Passing '[None]' selects the missing
    level instead, e.g. 'min_size_quantile=[None]' for the portfolios
    without a size screen.

    Parameters
    ----------
    dataset : str
        The dataset to download. Supported values are
        'high_frequency_sp500', 'factor_library',
        'factor_library_costs', 'factor_library_grid', and
        'factor_library_persistence'.
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
        For 'dataset' set to 'factor_library' or
        'factor_library_costs': either named arguments
        used to filter the portfolio grid, or 'ids=<vector>' to bypass
        the grid filter and download specific portfolios directly via
        '_download_factor_library_ids'. Filter arguments take the form
        'column=value', where 'value' may be a list or tuple to match
        multiple levels. Optionally pass 'fill_all=True' to leave
        unspecified columns unrestricted (default False, i.e.,
        unspecified columns are fixed at the defaults listed above).
        Passing None for any parameter removes that filter entirely,
        returning all values for that column, and '[None]' selects its
        missing level. Passing an unrecognised
        column name raises a 'ValueError'. 'ids' cannot be combined
        with filter arguments. 'revision' and 'grid_revision' pin the
        releases of the returns (or costs) and of the grid. For
        'dataset' set to 'factor_library_grid', 'refresh=True' downloads
        the grid again instead of returning the copy cached in the
        current session, and 'grid_revision' pins it. For
        'factor_library_persistence', 'sorting_variable' and
        'grid_revision' are accepted. 'revision' does not apply to the
        grid or the persistence and raises a 'ValueError'. Ignored for
        'high_frequency_sp500'.

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
        replacement = dataset[len("hf_") :]
        _deprecate_legacy_dataset(dataset, replacement)
        dataset = replacement

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

    # Both are in tidy-finance/factor-library-grid, so 'revision', which
    # pins tidy-finance/factor-library, would pin nothing.
    if (
        dataset in ("factor_library_grid", "factor_library_persistence")
        and "revision" in kwargs
    ):
        raise ValueError(
            f"'revision' does not apply to '{dataset}'. 'revision' pins "
            "the returns and the costs. Pin the grid and the persistence "
            "with 'grid_revision'."
        )

    if dataset == "factor_library_grid":
        return _download_factor_library_grid(**kwargs)

    if dataset == "factor_library_persistence":
        return _download_factor_library_persistence(
            start_date=start_date, end_date=end_date, **kwargs
        )

    if dataset in ("factor_library", "factor_library_costs"):
        return _download_data_huggingface_factor_library(
            start_date=start_date,
            end_date=end_date,
            table="returns" if dataset == "factor_library" else "costs",
            **kwargs,
        )

    raise ValueError(f"Unsupported dataset: '{dataset}'.")
