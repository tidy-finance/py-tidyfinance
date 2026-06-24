"""Data download and retrieval module for tidyfinance."""

# %% libraries
import io
import os
import re
import tempfile
import time
import warnings
import zipfile
from datetime import date
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from curl_cffi import requests
from sqlalchemy import text

from ._internal import (
    _assign_exchange,
    _assign_industry,
    _format_cusips,
    _get_random_user_agent,
    _transfrom_to_snake_case,
    _validate_dates,
)
from ._pseudo import _simulate_pseudo_data
from .supported_datasets import (
    _check_supported_dataset_ff,
    _check_supported_dataset_q,
    _check_supported_dataset_wrds,
    _check_supported_dataset_wrds_crsp,
    _check_supported_domain,
    _determine_frequency_ff,
    _determine_frequency_q,
    _is_breakpoints_ff,
    _is_legacy_type,
    _is_legacy_type_ff,
    _is_legacy_type_q,
    _is_legacy_type_wrds,
    _parse_type_to_domain_dataset,
    _resolve_domain_alias,
)
from .utilities import (
    _process_additional_columns,
    disconnect_connection,
    get_wrds_connection,
    list_supported_indexes,
    process_trace_data,
)

# %% constant

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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

_hf_session = requests.Session()

# %% functions


def get_available_famafrench_datasets():
    """
    Get the list of datasets available from the Fama/French data library.

    Returns
    -------
    list
        A list of valid dataset names for use with download_data_factors_ff.
    """
    ff_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    ff_url_prefix = "ftp/"
    ff_url_suffix = "_CSV.zip"
    try:
        from lxml.html import document_fromstring
    except Exception:
        raise ImportError(
            "Please install lxml if you want to use the "
            "get_datasets_famafrench function"
        )

    response = requests.get(f"{ff_url}data_library.html")
    root = document_fromstring(response.content)

    datasets = [
        e.attrib["href"] for e in root.findall(".//a") if "href" in e.attrib
    ]
    datasets = [
        dataset_i
        for dataset_i in datasets
        if dataset_i.startswith(ff_url_prefix)
        and dataset_i.endswith(ff_url_suffix)
    ]
    datasets_list = list(
        map(lambda x: x[len(ff_url_prefix) : -len(ff_url_suffix)], datasets)
    )
    return datasets_list


def download_data(
    domain: str = None,
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download and process data based on domain and dataset.

    Downloads and processes data based on the specified domain (e.g.,
    Fama-French factors, Global Q factors, or macro predictors), dataset,
    and date range. The function checks whether the specified domain is
    supported and then delegates to the appropriate function for
    downloading and processing the data.

    Parameters
    ----------
    domain : str
        The domain of the dataset to download, given as one of the
        canonical names returned by 'list_supported_datasets()':
        'Fama-French', 'Global Q', 'Goyal-Welch', 'WRDS', 'Pseudo Data',
        'Index Constituents', 'FRED', 'Stock Prices',
        'Open Source Asset Pricing', 'Tidy Finance'. The previous
        short names (e.g. 'famafrench', 'wrds', 'pseudo') are still
        accepted but deprecated and will be removed in a future release.
    dataset : str, optional
        The specific dataset to download within the domain.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        or a subset is returned, depending on the dataset type.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        or a subset is returned, depending on the dataset type.
    type : str, optional
        Deprecated. Use 'domain' and 'dataset' instead. If provided, a
        DeprecationWarning is emitted and the legacy type is
        translated to a ('domain', 'dataset') pair via
        'list_supported_datasets'.
    **kwargs
        Additional arguments passed to specific download functions
        depending on 'domain'. For instance, if 'domain' is
        'constituents', arguments are passed to
        '_download_data_constituents'. If 'domain' is 'tidyfinance' and
        'dataset' is 'factor_library', arguments are either filter
        inputs (e.g., 'sorting_variable', 'rebalancing', 'fill_all') or
        an explicit 'ids' vector that bypasses the grid filter and
        downloads the specified portfolios directly via
        '_download_factor_library_ids'; see
        '_download_data_huggingface' for details.

    Returns
    -------
    pd.DataFrame
        A data frame with processed data, including dates and the
        relevant financial metrics, filtered by the specified date
        range.

    Examples
    --------
    >>> from tidyfinance import download_data
    >>> download_data(
    ...     'Fama-French',
    ...     'Fama/French 5 Factors (2x3) [Daily]',
    ...     '2000-01-01',
    ...     '2020-12-31',
    ... )
    >>> download_data(
    ...     'Goyal-Welch', 'monthly', '2000-01-01', '2020-12-31'
    ... )
    >>> download_data('Index Constituents', index='DAX')
    >>> download_data('FRED', series=['GDP', 'CPIAUCNS'])
    >>> download_data('Stock Prices', symbols=['AAPL', 'MSFT'])
    >>> download_data(
    ...     'Tidy Finance', 'risk_free', '2020-01-01', '2020-12-31'
    ... )
    >>> download_data(
    ...     'Tidy Finance',
    ...     'high_frequency_sp500',
    ...     '2007-07-26',
    ...     '2007-07-27',
    ... )
    >>> download_data(
    ...     'Tidy Finance',
    ...     'factor_library',
    ...     sorting_variable='52w',
    ...     rebalancing='annual',
    ... )
    >>> download_data('Tidy Finance', 'factor_library', ids=[1, 2, 3])
    >>> download_data('Tidy Finance', 'factor_library_grid')
    """
    if type is not None:
        warnings.warn(
            "'type' is deprecated; use 'domain' and 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain, dataset = _parse_type_to_domain_dataset(type)

    if domain is not None and _is_legacy_type(domain):
        warnings.warn(
            "Passing a legacy 'type' string as 'domain' is deprecated; "
            "use 'domain' and 'dataset' instead. "
            "See list_supported_datasets() for the mapping.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain, dataset = _parse_type_to_domain_dataset(domain)

    if domain is None:
        raise ValueError("Argument 'domain' is required.")

    domain = _resolve_domain_alias(domain)

    _check_supported_domain(domain)

    if domain == "Fama-French":
        processed_data = _download_data_factors_ff(
            dataset=dataset, start_date=start_date, end_date=end_date
        )
    elif domain == "Global Q":
        processed_data = _download_data_factors_q(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Goyal-Welch":
        processed_data = _download_data_macro_predictors(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "WRDS":
        processed_data = _download_data_wrds(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Index Constituents":
        processed_data = _download_data_constituents(dataset=dataset, **kwargs)
    elif domain == "FRED":
        processed_data = _download_data_fred(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Stock Prices":
        processed_data = _download_data_stock_prices(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Open Source Asset Pricing":
        processed_data = _download_data_osap(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Tidy Finance":
        if dataset == "risk_free":
            processed_data = _download_data_risk_free(
                start_date=start_date, end_date=end_date, **kwargs
            )
        else:
            processed_data = _download_data_huggingface(
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
    elif domain == "Pseudo Data":
        processed_data = _simulate_pseudo_data(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    else:
        raise ValueError("Unsupported domain.")
    return processed_data


def _famafrench_downloader(file_url, start_date=None, end_date=None):
    """Download a Kenneth French data ZIP and parse its first table.

    Parameters
    ----------
    file_url : str
        Path relative to the Kenneth French data library base URL,
        matching the 'file_url' column of '_FF_DATASETS' (e.g.,
        'ftp/F-F_Research_Data_Factors_CSV.zip').
    start_date : str, optional
        Filter the parsed table to dates >= 'start_date'.
    end_date : str, optional
        Filter the parsed table to dates <= 'end_date'.

    Returns
    -------
    pd.DataFrame
        The first parsed table in the archive, indexed by date. Factor
        files return columns such as 'Mkt-RF', 'SMB', 'HML', 'RF';
        breakpoint files return one column per percentile bin plus
        diagnostic columns ('<=0', '>0', 'Count') depending on the
        file. Returns 'None' implicitly if no table could be parsed.
        The Fama-French archives often contain several tables
        (value-weighted, equal-weighted, etc.); only the first is
        returned. Download the source ZIP directly if you need the
        others.
    """
    ff_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    datatset_url = ff_url + file_url
    resp = requests.get(datatset_url, impersonate="chrome")
    resp.raise_for_status()

    # decode
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        data_raw = zf.read(name).decode("latin1")

    # Derive the stem (e.g., "ME_Breakpoints") for breakpoint column-
    # naming logic by stripping the ftp/ prefix and _CSV.zip suffix.
    stem = file_url
    if stem.startswith("ftp/"):
        stem = stem[len("ftp/") :]
    if stem.endswith("_CSV.zip"):
        stem = stem[: -len("_CSV.zip")]

    # Breakpoint dataset cases
    params = {"index_col": 0}
    if stem.endswith("_Breakpoints"):
        if "-" in stem:
            cols = ["<=0", ">0"]
        else:
            cols = ["Count"]
        r = list(range(0, 105, 5))
        params["names"] = ["Date"] + cols + list(zip(r, r[1:]))
        params["skiprows"] = 1 if stem != "Prior_2-12_Breakpoints" else 3

    doc_chunks, tables = [], []
    for chunk in data_raw.split(2 * "\r\n"):
        if len(chunk) < 800:
            doc_chunks.append(chunk.replace("\r\n", " ").strip())
        else:
            tables.append(chunk)

    for i, src in enumerate(tables):
        match = re.search(r"^\s*,", src, re.M)  # the table starts there
        start = 0 if not match else match.start()

        # raw to dataframe
        try:
            df = pd.read_csv(io.StringIO("Date" + src[start:]), **params)
        except pd.errors.ParserError as e:
            warnings.warn(str(e), UserWarning, stacklevel=2)
            continue

        # get index as datetime
        try:
            idx_name = df.index.name

            s = df.index.astype(str)
            if (s.str.len() == 8).all():
                fmt, add, freq = "%Y%m%d", "", "D"
                s_dt = [f"{i}{add}" for i in s]
                dt = pd.to_datetime(s_dt, format=fmt, errors="coerce")
            elif (s.str.len() == 6).all():
                fmt, add, freq = "%Y%m%d", "01", "M"
                s_dt = [f"{i}01" for i in s]
                dt = pd.to_datetime(s_dt, format="%Y%m%d", errors="coerce")
            elif (s.str.len() == 4).all():
                fmt, add, freq = "%Y%m%d", "0101", "A-DEC"
                s_dt = [f"{i}0101" for i in s]
                dt = pd.to_datetime(s_dt, format="%Y%m%d", errors="coerce")
                dt = dt.to_period(freq).to_timestamp(how="end")
            else:
                raise ValueError("Unrecognized integer date format in index.")
            df = df[~dt.isna()].copy()
            df.index = dt[~dt.isna()]
            df.index.name = idx_name.strip().lower()
        except Exception:
            pass

        # start and end dates
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df


def _download_data_factors_ff(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
) -> pd.DataFrame:
    """
    Download and process Fama-French factor data.

    Downloads and processes Fama-French factor data based on the
    specified dataset name and date range. The data is downloaded
    directly from Kenneth French's data library and processed into a
    structured format, including date conversion, scaling of factor
    values, and filtering by the specified date range.

    If there are multiple tables in the raw Fama-French data (e.g.,
    value-weighted and equal-weighted returns), the function only
    returns the first table because these are the most popular. Download
    the source ZIP archive directly if you need less commonly used
    tables.

    Parameters
    ----------
    dataset : str
        The name of the Fama-French dataset to download (e.g.,
        'Fama/French 3 Factors').
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, the value is
        translated to a dataset name via 'list_supported_datasets' and
        a DeprecationWarning is emitted.

    Returns
    -------
    pd.DataFrame
        A data frame with processed factor data, including the date,
        risk-free rate, market excess return, and other factors,
        filtered by the specified date range.

    References
    ----------
    Fama, E. F., and French, K. R. (1993). Common risk factors in the
    returns on stocks and bonds. Journal of Financial Economics,
    33(1), 3-56. https://doi.org/10.1016/0304-405X(93)90023-5

    Fama, E. F., and French, K. R. (2015). A five-factor asset pricing
    model. Journal of Financial Economics, 116(1), 1-22.
    https://doi.org/10.1016/j.jfineco.2014.10.010

    Carhart, M. M. (1997). On persistence in mutual fund performance.
    Journal of Finance, 52(1), 57-82.
    https://doi.org/10.1111/j.1540-6261.1997.tb03808.x

    Examples
    --------
    >>> from tidyfinance import download_data_factors_ff
    >>> download_data_factors_ff(
    ...     'Fama/French 3 Factors', '2000-01-01', '2020-12-31'
    ... )
    >>> download_data_factors_ff(
    ...     '10 Industry Portfolios', '2000-01-01', '2020-12-31'
    ... )
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _, dataset = _parse_type_to_domain_dataset(type)

    if dataset is not None and _is_legacy_type_ff(dataset):
        warnings.warn(
            "Passing a legacy type as 'dataset' is deprecated. "
            "Use the dataset_name from list_supported_datasets("
            "domain='Fama-French').",
            DeprecationWarning,
            stacklevel=2,
        )
        _, dataset = _parse_type_to_domain_dataset(dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    file_url = _check_supported_dataset_ff(dataset)

    start_date, end_date = _validate_dates(start_date, end_date)

    frequency = _determine_frequency_ff(dataset)
    if frequency not in ("daily", "weekly", "monthly"):
        raise ValueError(
            "This dataset has neither daily, weekly, nor monthly frequency."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            raw_downloaded = _famafrench_downloader(
                file_url, start_date=start_date, end_date=end_date
            )
            if not _is_breakpoints_ff(dataset):
                raw_downloaded = raw_downloaded.div(100)
            raw_data = (
                raw_downloaded.reset_index()
                .rename(
                    columns=lambda x: (
                        x.lower()
                        .replace("-rf", "_excess")
                        .replace("rf", "risk_free")
                        if isinstance(x, str)
                        else x
                    )
                )
                .apply(
                    lambda x: x.replace([-99.99, -999], np.nan)
                    if x.name != "date"
                    else x
                )
            )
            raw_data = raw_data[
                ["date"] + [col for col in raw_data.columns if col != "date"]
            ].reset_index(drop=True)
            return raw_data
    except Exception as e:
        warnings.warn(
            f"Returning an empty dataset due to download failure: {e}",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()


def _download_data_factors_q(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    url: str = "https://global-q.org/uploads/1/2/2/6/122679606/",
) -> pd.DataFrame:
    """
    Download and process Global Q factor data.

    Downloads and processes Global Q factor data based on the specified
    dataset, date range, and source URL. The processing includes date
    conversion, renaming variables to a standardized format, scaling
    factor values, and filtering by the specified date range.

    Parameters
    ----------
    dataset : str
        The name of the dataset to download. Recognized prefixes are
        'q5_factors_daily', 'q5_factors_weekly', 'q5_factors_weekly_w2w',
        'q5_factors_monthly', 'q5_factors_quarterly', and
        'q5_factors_annual'. Pass an explicit year suffix to pin the
        vintage (e.g., 'q5_factors_daily_2024'). If only the bare
        prefix is supplied, the 2024 vintage is appended as a default;
        any other dataset without a recognized prefix raises
        'ValueError'.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, the value is
        translated to a dataset name via 'list_supported_datasets' and
        a DeprecationWarning is emitted.
    url : str, optional
        The base URL from which to download the dataset files.

    Returns
    -------
    pd.DataFrame
        A data frame with processed factor data, including the date,
        risk-free rate, market excess return, and other factors,
        filtered by the specified date range.

    Examples
    --------
    >>> from tidyfinance import download_data_factors_q
    >>> download_data_factors_q(
    ...     'q5_factors_daily_2024', '2020-01-01', '2020-12-31'
    ... )
    >>> download_data_factors_q('q5_factors_annual_2024')
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _, dataset = _parse_type_to_domain_dataset(type)

    if dataset is not None and _is_legacy_type_q(dataset):
        warnings.warn(
            "Passing a legacy type as 'dataset' is deprecated. "
            "Use the dataset_name from list_supported_datasets("
            "domain='Global Q').",
            DeprecationWarning,
            stacklevel=2,
        )
        _, dataset = _parse_type_to_domain_dataset(dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    start_date, end_date = _validate_dates(start_date, end_date)

    valid_prefixes = [
        "q5_factors_daily_",
        "q5_factors_weekly_w2w_",
        "q5_factors_weekly_",
        "q5_factors_monthly_",
        "q5_factors_quarterly_",
        "q5_factors_annual_",
    ]
    base_prefixes = [p.rstrip("_") for p in valid_prefixes]
    if any(dataset == p for p in base_prefixes):
        dataset = f"{dataset}_2024"
    elif not any(dataset.startswith(p) for p in valid_prefixes):
        raise ValueError(
            f"Unsupported dataset: '{dataset}'. Dataset name"
            " must include the year, e.g. 'q5_factors_daily_2024'."
        )
    try:
        raw_data = (
            pd.read_csv(
                f"{url}{dataset}.csv", engine="python", on_bad_lines="skip"
            )
            .rename(columns=lambda x: x.lower().replace("r_", ""))
            .rename(columns={"f": "risk_free", "mkt": "mkt_excess"})
        )
    except Exception as e:
        raise ValueError(
            f"Could not download or parse dataset '{dataset}': {e}"
        ) from e

    if "monthly" in dataset:
        raw_data = raw_data.assign(
            date=pd.to_datetime(
                dict(year=raw_data.year, month=raw_data.month, day=1)
            )
        ).drop(columns=["year", "month"])
    if "weekly" in dataset:
        raw_data = raw_data.assign(
            date=pd.to_datetime(
                dict(
                    year=raw_data.year,
                    month=raw_data.month,
                    day=raw_data.day,
                )
            )
        ).drop(columns=["year", "month", "day"])
    if "annual" in dataset:
        raw_data = raw_data.assign(
            date=lambda x: pd.to_datetime(x["year"].astype(str) + "-01-01")
        ).drop(columns=["year"])

    raw_data = raw_data.assign(date=lambda x: pd.to_datetime(x["date"])).apply(
        lambda x: x.div(100) if x.name != "date" else x
    )

    if start_date and end_date:
        raw_data = raw_data.query("@start_date <= date <= @end_date")

    raw_data = raw_data[
        ["date"] + [col for col in raw_data.columns if col != "date"]
    ].reset_index(drop=True)
    return raw_data


def _download_data_macro_predictors(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    sheet_id: str = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG",
) -> pd.DataFrame:
    """
    Download and process macro predictor data.

    Downloads and processes macroeconomic predictor data based on the
    specified dataset (monthly, quarterly, or annual), date range, and
    source URL. The function downloads the data from a Google Sheets
    export link and processes the raw data into a structured format,
    calculating additional financial metrics and filtering by the
    specified date range.

    Parameters
    ----------
    dataset : str
        The dataset to download. Accepts 'monthly', 'quarterly', or
        'annual'.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'macro_predictors_' prefix is stripped (e.g.,
        'macro_predictors_monthly' becomes 'monthly') and a
        DeprecationWarning is emitted.
    sheet_id : str, optional
        The Google Sheets ID from which to download the dataset, with
        the default '1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG'.

    Returns
    -------
    pd.DataFrame
        A data frame with processed data, filtered by the specified
        date range and including financial metrics.

    References
    ----------
    Welch, I., and Goyal, A. (2008). A comprehensive look at the
    empirical performance of equity premium prediction. Review of
    Financial Studies, 21(4), 1455-1508.
    https://doi.org/10.1093/rfs/hhm014

    Examples
    --------
    >>> from tidyfinance import download_data_macro_predictors
    >>> download_data_macro_predictors('monthly')
    >>> download_data_macro_predictors(
    ...     'quarterly', '2000-01-01', '2020-12-31'
    ... )
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^macro_predictors_", "", type)

    if dataset is not None and dataset.startswith("macro_predictors_"):
        warnings.warn(
            "Passing 'macro_predictors_'-prefixed dataset names is "
            "deprecated. Use 'monthly' instead of "
            "'macro_predictors_monthly'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^macro_predictors_", "", dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    start_date, end_date = _validate_dates(start_date, end_date)

    if dataset in ["monthly", "quarterly", "annual"]:
        try:
            macro_sheet_url = (
                "https://docs.google.com/spreadsheets/d/"
                f"{sheet_id}/gviz/tq?tqx=out:csv&sheet="
                f"{dataset.capitalize()}"
            )
            raw_data = pd.read_csv(macro_sheet_url)
        except Exception as e:
            warnings.warn(
                f"Returning an empty dataset due to download failure: {e}",
                UserWarning,
                stacklevel=2,
            )
            return pd.DataFrame()
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. "
            "Use 'monthly', 'quarterly', or 'annual'."
        )

    if dataset == "monthly":
        raw_data = raw_data.assign(
            date=lambda x: pd.to_datetime(x["yyyymm"], format="%Y%m")
        ).drop(columns=["yyyymm"])
    if dataset == "quarterly":
        raw_data = raw_data.assign(
            date=lambda x: pd.to_datetime(
                x["yyyyq"].astype(str).str[:4]
                + "-"
                + (x["yyyyq"].astype(str).str[4].astype(int) * 3 - 2).astype(
                    str
                )
                + "-01"
            )
        ).drop(columns=["yyyyq"])
    if dataset == "annual":
        raw_data = raw_data.assign(
            date=lambda x: pd.to_datetime(x["yyyy"].astype(str) + "-01-01")
        ).drop(columns=["yyyy"])

    raw_data = raw_data.apply(
        lambda x: (
            pd.to_numeric(x.astype(str).str.replace(",", ""), errors="coerce")
            if x.dtype == "object" or pd.api.types.is_string_dtype(x)
            else x
        )
    )
    raw_data = raw_data.apply(
        lambda x: (
            pd.to_numeric(x.astype(str).str.replace(",", ""), errors="coerce")
            if pd.api.types.is_string_dtype(x) or x.dtype == "object"
            else x
        )
    ).assign(
        IndexDiv=lambda df: df["Index"] + df["D12"],
        logret=lambda df: (
            df["IndexDiv"]
            .apply(lambda x: np.nan if pd.isna(x) else np.log(x))
            .diff()
        ),
        rp_div=lambda df: df["logret"].shift(-1) - df["Rfree"],
        log_d12=lambda df: df["D12"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
        ),
        log_e12=lambda df: df["E12"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
        ),
        dp=lambda df: (
            df["log_d12"]
            - df["Index"].apply(lambda x: np.nan if pd.isna(x) else np.log(x))
        ),
        dy=lambda df: (
            df["log_d12"]
            - df["Index"]
            .shift(1)
            .apply(lambda x: np.nan if pd.isna(x) else np.log(x))
        ),
        ep=lambda df: (
            df["log_e12"]
            - df["Index"].apply(lambda x: np.nan if pd.isna(x) else np.log(x))
        ),
        de=lambda df: df["log_d12"] - df["log_e12"],
        tms=lambda df: df["lty"] - df["tbl"],
        dfy=lambda df: df["BAA"] - df["AAA"],
    )

    raw_data = raw_data[
        [
            "date",
            "rp_div",
            "dp",
            "dy",
            "ep",
            "de",
            "svar",
            "b/m",
            "ntis",
            "tbl",
            "lty",
            "ltr",
            "tms",
            "dfy",
            "infl",
        ]
    ]
    raw_data = raw_data.rename(
        columns={col: col.replace("/", "") for col in raw_data.columns}
    ).dropna()

    if start_date and end_date:
        raw_data = raw_data.query("@start_date <= date <= @end_date")

    return raw_data


def _download_data_constituents(
    index: str = None, dataset: str = None, **kwargs
) -> pd.DataFrame:
    """
    Download constituent data for a given stock index.

    Downloads and processes the constituent data for a specified
    financial index. The data is fetched from a remote CSV file,
    filtered, and cleaned to provide relevant information about
    constituents. The function retrieves the URL of the CSV file for
    the specified index from ETF sites, sends an HTTP GET request to
    download the file, and processes the CSV to extract equity
    constituents.

    Parameters
    ----------
    index : str
        Name of the financial index for which to download constituent
        data. Must be one of the supported indexes listed by
        'list_supported_indexes'.
    dataset : str, optional
        Convenience alias accepted from the unified 'download_data'
        dispatcher. Forwarded to 'index' with a UserWarning when
        'index' is not supplied directly.
    **kwargs
        Additional keyword arguments are accepted and silently
        ignored. They exist so that calls routed through
        'download_data' (which forwards arguments such as
        'start_date' and 'end_date' to every downloader) do not fail
        with a TypeError, even when those arguments are not meaningful
        for the constituents endpoint.

    Returns
    -------
    pd.DataFrame
        A data frame with five columns:

        - 'symbol': the ticker symbol of the equity constituent.
        - 'name': the name of the equity constituent.
        - 'location': the location where the company is based.
        - 'exchange': the exchange where the equity is traded.
        - 'currency': the currency in which the equity is traded,
          derived from the exchange.

        The data frame is filtered to exclude non-equity entries,
        blacklisted symbols, empty names, and any entries containing
        the index name or 'CASH'.

    Examples
    --------
    >>> from tidyfinance import download_data_constituents
    >>> download_data_constituents('DAX')
    """
    if dataset is not None and index is None:
        warnings.warn(
            "The 'dataset' argument is not valid for "
            "domain='Index Constituents'. Use 'index' instead, e.g. "
            "download_data(domain='Index Constituents', index='DAX').",
            UserWarning,
            stacklevel=2,
        )
        index = dataset

    symbol_blacklist = {"", "-", "USD", "GXU4", "EUR", "MARGIN_EUR", "MLIFT"}
    supported_indexes = list_supported_indexes()

    if index not in supported_indexes["index"].values:
        raise ValueError(
            "The index '{index}' is not supported. "
            f"Supported indexes: {', '.join(supported_indexes['index'])}"
        )

    url = supported_indexes.loc[
        supported_indexes["index"] == index, "url"
    ].values[0]
    skip_rows = supported_indexes.loc[
        supported_indexes["index"] == index, "skip"
    ].values[0]
    headers = {"User-Agent": _get_random_user_agent()}

    try:
        response = requests.get(url, impersonate="chrome120", headers=headers)
    except Exception:
        response = requests.get(url, impersonate="chrome120")

    if response.status_code != 200:
        raise ValueError(
            f"Failed to download data for index {index}. "
            "Please check the index name or try again later."
        )

    df = pd.read_csv(io.StringIO(response.text), skiprows=skip_rows)

    if "Anlageklasse" in df.columns:
        df = df[df["Anlageklasse"] == "Aktien"][
            ["Emittententicker", "Name", "Standort", "Börse"]
        ]
        df.columns = ["symbol", "name", "location", "exchange"]
    elif "Asset Class" in df.columns:
        df = df[df["Asset Class"] == "Equity"][
            ["Ticker", "Name", "Location", "Exchange"]
        ]
        df.columns = ["symbol", "name", "location", "exchange"]
    else:
        raise ValueError("Unknown column format in downloaded data.")

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[~df["symbol"].isin(symbol_blacklist)]
    df = df[df["name"] != ""]
    df = df[~df["name"].str.contains(index, case=False, na=False)]
    df = df[~df["name"].str.contains("CASH", case=False, na=False)]
    index_no_space = re.sub(r"\s+", "", index).lower()
    df = df[~df["name"].str.lower().str.contains(index_no_space, na=False)]

    df.loc[df["name"] == "NATIONAL BANK OF CANADA", "symbol"] = "NA"
    df["symbol"] = df["symbol"].str.replace(" ", "-").str.replace("/", "-")

    exchange_suffixes = {
        "Xetra": ".DE",
        "Deutsche Börse AG": ".DE",
        "Boerse Berlin": ".BE",
        "Borsa Italiana": ".MI",
        "Nyse Euronext - Euronext Paris": ".PA",
        "Euronext Amsterdam": ".AS",
        "Nasdaq Omx Helsinki Ltd.": ".HE",
        "Singapore Exchange": ".SI",
        "Asx - All Markets": ".AX",
        "London Stock Exchange": ".L",
        "SIX Swiss Exchange": ".SW",
        "Tel Aviv Stock Exchange": ".TA",
        "Tokyo Stock Exchange": ".T",
        "Hong Kong Stock Exchange": ".HK",
        "Toronto Stock Exchange": ".TO",
        "Euronext Brussels": ".BR",
        "Euronext Lisbon": ".LS",
        "Bovespa": ".SA",
        "Mexican Stock Exchange": ".MX",
        "Stockholm Stock Exchange": ".ST",
        "Oslo Stock Exchange": ".OL",
        "Johannesburg Stock Exchange": ".J",
        "Korea Exchange": ".KS",
        "Shanghai Stock Exchange": ".SS",
        "Shenzhen Stock Exchange": ".SZ",
    }
    df["symbol"] = df.apply(
        lambda row: row["symbol"] + exchange_suffixes.get(row["exchange"], ""),
        axis=1,
    )
    df["symbol"] = df["symbol"].str.replace("..", ".", regex=False)

    currency_map = {
        "Xetra": "EUR",
        "Deutsche Börse AG": "EUR",
        "Boerse Berlin": "EUR",
        "Borsa Italiana": "EUR",
        "Nyse Euronext - Euronext Paris": "EUR",
        "Euronext Amsterdam": "EUR",
        "Nasdaq Omx Helsinki Ltd.": "EUR",
        "Euronext Brussels": "EUR",
        "Euronext Lisbon": "EUR",
        "Singapore Exchange": "SGD",
        "Asx - All Markets": "AUD",
        "London Stock Exchange": "GBP",
        "SIX Swiss Exchange": "CHF",
        "Tel Aviv Stock Exchange": "ILS",
        "Tokyo Stock Exchange": "JPY",
        "Hong Kong Stock Exchange": "HKD",
        "Toronto Stock Exchange": "CAD",
        "Bovespa": "BRL",
        "Mexican Stock Exchange": "MXN",
        "Stockholm Stock Exchange": "SEK",
        "Oslo Stock Exchange": "NOK",
        "Johannesburg Stock Exchange": "ZAR",
        "Korea Exchange": "KRW",
        "Shanghai Stock Exchange": "CNY",
        "Shenzhen Stock Exchange": "CNY",
    }
    df["currency"] = df["exchange"].map(currency_map).fillna("USD")

    return df


def _download_data_fred(
    series: str | list,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download and process data from FRED.

    Downloads a specified data series from the Federal Reserve
    Economic Data (FRED) website, processes the data, and returns it
    as a data frame. The function constructs the URL based on the
    provided FRED series ID, performs an HTTP GET request to download
    the data in CSV format, and processes it to a tidy data frame
    format. The resulting data frame includes the date, value, and the
    series ID.

    Parameters
    ----------
    series : str or list of str
        A character vector specifying the FRED series ID(s) to
        download (e.g., 'GDP', ['GDP', 'UNRATE']).
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.

    Returns
    -------
    pd.DataFrame
        A data frame containing the processed data with three columns:

        - 'date': the date corresponding to the data point.
        - 'series': the FRED series ID corresponding to the data.
        - 'value': the value of the data series at that date.

    Examples
    --------
    >>> from tidyfinance import download_data_fred
    >>> download_data_fred('CPIAUCNS')
    >>> download_data_fred(['GDP', 'CPIAUCNS'], '2010-01-01', '2010-12-31')
    """
    if isinstance(series, str):
        series = [series]
    start_date, end_date = _validate_dates(start_date, end_date)
    fred_data = []
    for s in series:
        urls = [
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}",
            f"https://fred.stlouisfed.org/series/{s}/downloaddata/{s}.csv",
        ]
        headers = {
            "User-Agent": _get_random_user_agent(),
            "Accept": "text/csv,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        success = False
        for url in urls:
            for attempt in range(3):
                try:
                    response = requests.get(
                        url,
                        headers=headers,
                        impersonate="chrome110",
                        timeout=30,
                    )
                    response.raise_for_status()
                except Exception:
                    if attempt < 2:
                        time.sleep(1 * (attempt + 1))
                    continue

                try:
                    raw_data = (
                        pd.read_csv(pd.io.common.StringIO(response.text))
                        .rename(columns=lambda c: c.strip().lower())
                        .assign(
                            date=lambda x: pd.to_datetime(
                                x[x.columns[x.columns.str.contains("date")][0]]
                            ),
                            value=lambda x: pd.to_numeric(
                                x[
                                    x.columns[~x.columns.str.contains("date")][
                                        0
                                    ]
                                ],
                                errors="coerce",
                            ),
                            series=s,
                        )
                        .get(["date", "series", "value"])
                    )
                    fred_data.append(raw_data)
                    success = True
                    break
                except Exception:
                    break

            if success:
                break

        if not success:
            warnings.warn(
                f"Failed to retrieve data for series {s}",
                UserWarning,
                stacklevel=2,
            )
            fred_data.append(pd.DataFrame(columns=["date", "series", "value"]))

    fred_data = pd.concat(fred_data, ignore_index=True)
    if start_date and end_date:
        fred_data = fred_data.query(
            "@start_date <= date <= @end_date"
        ).reset_index(drop=True)
    return fred_data


def _download_data_stock_prices(
    symbols: str | list,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Downloads historical stock data from Yahoo Finance for the given
    symbols and date range.

    Parameters
    ----------
    symbols : str or list of str
        A character vector of stock symbols to download data for. At
        least one symbol must be provided.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a one-year subset
        of the dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a one-year subset
        of the dataset is returned.

    Returns
    -------
    pd.DataFrame
        A data frame containing the downloaded stock data with columns
        'symbol', 'date', 'volume', 'open', 'low', 'high', 'close', and
        'adjusted_close'.

    Examples
    --------
    >>> from tidyfinance import download_data_stock_prices
    >>> download_data_stock_prices(['AAPL', 'MSFT'])
    >>> download_data_stock_prices('GOOGL', '2021-01-01', '2022-01-01')
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    elif not isinstance(symbols, list) or not all(
        isinstance(sym, str) for sym in symbols
    ):
        raise ValueError("symbols must be a list of stock symbols (strings).")

    start_date, end_date = _validate_dates(
        start_date, end_date, use_default_range=True
    )

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    all_data = []

    for symbol in symbols:
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={start_timestamp}&period2={end_timestamp}"
            "&interval=1d"
        )

        headers = {"User-Agent": _get_random_user_agent()}
        try:
            response = requests.get(
                url, impersonate="chrome120", headers=headers
            )
        except Exception:
            response = requests.get(url, impersonate="chrome120")

        if response.status_code == 200:
            raw_data = response.json().get("chart", {}).get("result", [])

            if (not raw_data) or ("timestamp" not in raw_data[0]):
                warnings.warn(
                    f"No data found for {symbol}.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            timestamps = raw_data[0]["timestamp"]
            indicators = raw_data[0]["indicators"]["quote"][0]
            adjusted_close = raw_data[0]["indicators"]["adjclose"][0][
                "adjclose"
            ]

            df_symbol = pd.DataFrame().assign(
                date=pd.to_datetime(
                    pd.to_datetime(timestamps, utc=True, unit="s").date
                ),
                symbol=symbol,
                volume=indicators.get("volume"),
                open=indicators.get("open"),
                low=indicators.get("low"),
                high=indicators.get("high"),
                close=indicators.get("close"),
                adjusted_close=adjusted_close,
            )

            # Ensure symbol and date are the first columns
            cols = list(df_symbol.columns)
            remaining = [c for c in cols if c not in ["symbol", "date"]]
            ordered_cols = ["symbol", "date"] + remaining
            df_symbol = df_symbol[ordered_cols]

            all_data.append(df_symbol)

        else:
            warnings.warn(
                f"Failed to retrieve data for symbol {symbol} "
                f"(Status code: {response.status_code})",
                UserWarning,
                stacklevel=2,
            )
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        cols = list(df_all.columns)
        remaining = [c for c in cols if c not in ["symbol", "date"]]
        ordered_cols = ["symbol", "date"] + remaining
        df_all = df_all[ordered_cols]
    else:
        df_all = pd.DataFrame()
    return df_all


def _download_data_osap(
    start_date: str = None,
    end_date: str = None,
    sheet_id: str = "1JyhcF5PRKHcputlioxlu5j5GyLo4JYyY",
) -> pd.DataFrame:
    """
    Download and process Open Source Asset Pricing data.

    Downloads the data from the Open Source Asset Pricing project at
    https://www.openassetpricing.com/data/ from Google Sheets using a
    specified sheet ID, processes the data by converting column names
    to snake_case, and optionally filters the data based on a provided
    date range.

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
    sheet_id : str, optional
        A character string representing the Google Sheet ID from which
        to download the data. Default is
        '1JyhcF5PRKHcputlioxlu5j5GyLo4JYyY'.

    Returns
    -------
    pd.DataFrame
        A data frame containing the processed data. The column names
        are converted to snake_case, and the data is filtered by the
        specified date range if 'start_date' and 'end_date' are
        provided.

    Examples
    --------
    >>> from tidyfinance import download_data_osap
    >>> osap = download_data_osap(
    ...     start_date='2020-01-01', end_date='2020-06-30'
    ... )
    """
    start_date, end_date = _validate_dates(start_date, end_date)

    # Google Drive direct download link
    url = f"https://drive.google.com/uc?export=download&id={sheet_id}"

    try:
        raw_data = pd.read_csv(url)
    except Exception:
        warnings.warn(
            "Returning an empty dataset due to download failure.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    if raw_data.empty:
        warnings.warn(
            "Returning an empty dataset due to download failure.",
            UserWarning,
            stacklevel=2,
        )
        return raw_data

    if "date" in raw_data.columns:
        raw_data["date"] = pd.to_datetime(raw_data["date"], errors="coerce")

    raw_data.columns = [
        _transfrom_to_snake_case(col) for col in raw_data.columns
    ]

    if start_date and end_date:
        raw_data = raw_data.query("@start_date <= date <= @end_date")

    return raw_data


def _download_data_risk_free(
    start_date: str = None,
    end_date: str = None,
    frequency: str = "monthly",
) -> pd.DataFrame:
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
    pd.DataFrame
        A data frame with two columns:

        - 'date': the date of the observation.
        - 'risk_free': the risk-free rate for the period.

    Examples
    --------
    >>> from tidyfinance import download_data_risk_free
    >>> download_data_risk_free('2020-01-01', '2020-12-31')
    >>> download_data_risk_free(
    ...     '2020-01-01', '2020-12-31', frequency='daily'
    ... )
    """
    if frequency not in ("monthly", "daily"):
        raise ValueError("frequency must be 'monthly' or 'daily'.")

    start_date, end_date = _validate_dates(start_date, end_date)

    url = (
        "https://huggingface.co/datasets/tidy-finance/risk-free/"
        f"resolve/main/risk_free_{frequency}.parquet"
    )

    try:
        risk_free_data = pd.read_parquet(url)

    except Exception as e:
        raise RuntimeError(
            "Failed to download risk-free rate data from HuggingFace. "
            f"URL attempted: {url}"
        ) from e

    risk_free_data["date"] = pd.to_datetime(risk_free_data["date"])

    if start_date is not None:
        risk_free_data = risk_free_data[
            (risk_free_data["date"] >= start_date)
            & (risk_free_data["date"] <= end_date)
        ].reset_index(drop=True)

    return risk_free_data


def _download_data_wrds(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download data from WRDS.

    Acts as a wrapper to download data from various WRDS datasets
    including CRSP, Compustat, and CCM links based on the specified
    dataset. It is designed to handle different datasets by
    redirecting to the appropriate specific data download function.

    Parameters
    ----------
    dataset : str
        A string specifying the dataset to download. Supported values
        are 'crsp_monthly' and 'crsp_daily' for CRSP data,
        'compustat_annual' and 'compustat_quarterly' for Compustat
        data, 'ccm_links' for CCM links data, 'fisd' for FISD data, or
        'trace_enhanced' for TRACE data.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped and a DeprecationWarning is emitted.
    **kwargs
        Additional arguments passed to specific download functions
        depending on the 'dataset'.

    Returns
    -------
    pd.DataFrame
        A data frame containing the requested data, with the structure
        and contents depending on the specified 'dataset'.

    Examples
    --------
    >>> from tidyfinance import download_data_wrds
    >>> crsp_monthly = download_data_wrds(
    ...     'crsp_monthly', '2020-01-01', '2020-12-31'
    ... )
    >>> compustat_annual = download_data_wrds(
    ...     'compustat_annual', '2020-01-01', '2020-12-31'
    ... )
    >>> ccm_links = download_data_wrds('ccm_links')
    >>> fisd = download_data_wrds('fisd')
    >>> trace_enhanced = download_data_wrds(
    ...     'trace_enhanced', cusips=['00101JAH9']
    ... )
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if dataset is not None and _is_legacy_type_wrds(dataset):
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'crsp_monthly' instead of 'wrds_crsp_monthly'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_wrds(dataset)

    if dataset.startswith("crsp"):
        return _download_data_wrds_crsp(dataset, start_date, end_date, **kwargs)
    elif dataset.startswith("compustat"):
        return _download_data_wrds_compustat(
            dataset, start_date, end_date, **kwargs
        )
    elif dataset == "ccm_links":
        return _download_data_wrds_ccm_links(**kwargs)
    elif dataset == "fisd":
        return _download_data_wrds_fisd(**kwargs)
    elif dataset == "trace_enhanced":
        return _download_data_wrds_trace_enhanced(
            start_date=start_date, end_date=end_date, **kwargs
        )
    else:
        raise ValueError(f"Unsupported WRDS dataset: {dataset!r}.")


def _download_data_wrds_crsp(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    batch_size: int = 500,
    version: str = "v2",
    additional_columns: list = None,
    add_ccm_links: bool = False,
    adjust_volume: bool = False,
) -> pd.DataFrame:
    """
    Download data from WRDS CRSP.

    Downloads and processes stock return data from the CRSP database
    for a specified period. Users can choose between monthly and daily
    datasets. The function also adjusts returns for delisting and
    calculates market capitalization and excess returns over the
    risk-free rate.

    Parameters
    ----------
    dataset : str
        A string specifying the CRSP dataset to download:
        'crsp_monthly' or 'crsp_daily'.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped and a DeprecationWarning is emitted.
    batch_size : int, optional
        An optional integer specifying the batch size for processing
        daily data, with a default of 500.
    version : str, optional
        An optional character specifying which CRSP version to use.
        'v2' (the default) uses the updated second version of CRSP,
        and 'v1' downloads the legacy version of CRSP.
    additional_columns : list of str, optional
        Extra column names from the underlying CRSP source table to
        return alongside the standard output. For 'crsp_monthly' the
        source is 'crsp.msf_v2' (or 'crsp.msf' joined with
        'crsp.msenames' under 'version="v1"'); for 'crsp_daily' the
        source is 'crsp.dsf_v2' (or 'crsp.dsf' for 'v1'). Pass any
        column from those tables (e.g., 'mthvol', 'mthvolflg' for
        monthly; 'dlyvol', 'dlyfacprc' for daily). When
        'adjust_volume=True' for 'crsp_daily', this list must include
        the columns the adjustment needs ('dlyprc', 'dlyvol',
        'dlyfacprc', 'primaryexch' for 'v2'; 'prc', 'vol', 'cfacpr',
        'exchcd' for 'v1'); a 'ValueError' is raised otherwise.
    add_ccm_links : bool, optional
        A boolean indicating whether CRSP-Compustat links should be
        added automatically using '_download_data_wrds_ccm_links'.
        Defaults to False.
    adjust_volume : bool, optional
        A boolean indicating whether daily CRSP trading volume data
        should be adjusted according to Gao and Ritter (2010).
        Defaults to False. Note that cumulative price adjustment
        factors are computed from the data in memory; results may be
        incorrect if 'start_date' excludes early observations for some
        permnos.

    Returns
    -------
    pd.DataFrame
        A data frame containing CRSP stock returns, adjusted for
        delistings, along with calculated market capitalization and
        excess returns over the risk-free rate. The structure of the
        returned data frame depends on the selected dataset.

    References
    ----------
    Gao, X., and Ritter, J. R. (2010). The marketing of seasoned
    equity offerings. Journal of Financial Economics, 97(1), 33-52.
    https://doi.org/10.1016/j.jfineco.2010.03.007

    Examples
    --------
    >>> from tidyfinance import download_data_wrds_crsp
    >>> crsp_monthly = download_data_wrds_crsp(
    ...     'crsp_monthly', '2020-11-01', '2020-12-31'
    ... )
    >>> crsp_daily = download_data_wrds_crsp(
    ...     'crsp_daily', '2020-12-01', '2020-12-31'
    ... )
    >>> download_data_wrds_crsp(
    ...     'crsp_monthly',
    ...     '2020-11-01',
    ...     '2020-12-31',
    ...     additional_columns=['mthvol', 'mthvolflg'],
    ... )
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if dataset is not None and _is_legacy_type_wrds(dataset):
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'crsp_monthly' instead of 'wrds_crsp_monthly'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_wrds_crsp(dataset)

    start_date, end_date = _validate_dates(start_date, end_date)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be an integer larger than 0.")

    if version not in ["v1", "v2"]:
        raise ValueError("version must be 'v1' or 'v2'.")

    if version == "v1" and pd.Timestamp(end_date) > pd.Timestamp(
        "2024-12-31"
    ):
        raise ValueError(
            "end_date must not be later than December 2024 for "
            "version='v1'. CRSP discontinued the legacy version at the "
            "end of 2024. Use version='v2' for more recent data."
        )

    additional_columns_list = additional_columns or []

    if adjust_volume and dataset != "crsp_daily":
        raise ValueError("adjust_volume is only supported for 'crsp_daily'.")

    if adjust_volume:
        if version == "v1":
            required = {"prc", "vol", "cfacpr", "exchcd"}
            cols_str = "prc, vol, cfacpr, and exchcd"
        else:
            required = {"dlyprc", "dlyvol", "dlyfacprc", "primaryexch"}
            cols_str = "dlyprc, dlyvol, primaryexch, and dlyfacprc"
        if not required.issubset(set(additional_columns_list)):
            raise ValueError(
                f"{cols_str} must be contained in additional_columns "
                "for adjust_volume=True."
            )

    wrds_connection = get_wrds_connection()
    additional_columns_sql = (
        ", ".join(additional_columns_list) if additional_columns_list else ""
    )

    crsp_data = pd.DataFrame()
    try:
        if "crsp_monthly" in dataset:
            if version == "v1":
                # Query 1: msf joined with msenames (shrcd 10/11)
                additional_cols_select = (
                    ", " + ", ".join(
                        f"msf.{c}" for c in additional_columns_list
                    )
                    if additional_columns_list
                    else ""
                )
                msf_query = text(f"""
                    SELECT msf.permno, msf.date, msf.ret, msf.shrout,
                           msf.altprc, msf.cfacpr,
                           msn.exchcd, msn.siccd
                           {additional_cols_select}
                    FROM crsp.msf AS msf
                    INNER JOIN crsp.msenames AS msn
                      ON msf.permno = msn.permno
                      AND msf.date BETWEEN msn.namedt AND msn.nameendt
                      AND msn.shrcd IN (10, 11)
                    WHERE msf.date BETWEEN '{start_date}'
                          AND '{end_date}'
                """)
                msf_data = pd.read_sql_query(
                    msf_query,
                    con=wrds_connection,
                    parse_dates={"date"},
                )

                # Query 2: msedelist (delisting events)
                msedelist_query = text(
                    "SELECT permno, dlstdt, dlret, dlstcd "
                    "FROM crsp.msedelist"
                )
                msedelist = pd.read_sql_query(
                    msedelist_query,
                    con=wrds_connection,
                    parse_dates={"dlstdt"},
                )

                # Query 3: first_crsp_date per permno
                first_date_query = text(
                    "SELECT permno, MIN(namedt) AS first_crsp_date "
                    "FROM crsp.msenames GROUP BY permno"
                )
                first_crsp_date = pd.read_sql_query(
                    first_date_query,
                    con=wrds_connection,
                    parse_dates={"first_crsp_date"},
                )

                disconnect_connection(wrds_connection)

                # calculation_date + month-floored date
                crsp_monthly = msf_data.assign(
                    calculation_date=lambda x: pd.to_datetime(x["date"]),
                    date=lambda x: pd.to_datetime(x["date"])
                    .dt.to_period("M")
                    .dt.start_time,
                    shrout=lambda x: x["shrout"] * 1000,
                )

                # Join delisting on (permno, month-floored dlstdt)
                if len(msedelist) > 0:
                    msedelist = msedelist.assign(
                        date=lambda x: pd.to_datetime(x["dlstdt"])
                        .dt.to_period("M")
                        .dt.start_time
                    )[["permno", "date", "dlret", "dlstcd"]]
                    crsp_monthly = crsp_monthly.merge(
                        msedelist, on=["permno", "date"], how="left"
                    )
                else:
                    crsp_monthly["dlret"] = np.nan
                    crsp_monthly["dlstcd"] = np.nan

                # listing_age (months elapsed, clipped at 0)
                crsp_monthly = crsp_monthly.merge(
                    first_crsp_date, on="permno", how="left"
                ).assign(
                    listing_age=lambda df: (
                        (df["date"].dt.year - df["first_crsp_date"].dt.year)
                        * 12
                        + (df["date"].dt.month - df["first_crsp_date"].dt.month)
                        - (
                            df["date"].dt.day
                            < df["first_crsp_date"].dt.day
                        ).astype(int)
                    ).clip(lower=0)
                ).drop(columns="first_crsp_date")

                # mktcap (millions); zero -> NaN
                crsp_monthly["mktcap"] = (
                    (crsp_monthly["shrout"] * crsp_monthly["altprc"]).abs()
                    / 1e6
                )
                crsp_monthly.loc[
                    crsp_monthly["mktcap"] == 0, "mktcap"
                ] = np.nan

                # mktcap_lag via self-join shifted by 1 month
                mktcap_lag_df = crsp_monthly.assign(
                    date=lambda x: x["date"] + pd.DateOffset(months=1)
                )[["permno", "date", "mktcap"]].rename(
                    columns={"mktcap": "mktcap_lag"}
                )
                crsp_monthly = crsp_monthly.merge(
                    mktcap_lag_df, on=["permno", "date"], how="left"
                )

                # exchange via exchcd (numeric, v1 codes)
                exchange_map = {
                    1: "NYSE", 31: "NYSE",
                    2: "AMEX", 32: "AMEX",
                    3: "NASDAQ", 33: "NASDAQ",
                }
                crsp_monthly["exchange"] = (
                    crsp_monthly["exchcd"]
                    .map(exchange_map)
                    .fillna("Other")
                )

                # industry from SIC code
                crsp_monthly["industry"] = (
                    crsp_monthly["siccd"]
                    .fillna(-1)
                    .astype(int)
                    .apply(_assign_industry)
                )

                # ret_adj from delisting code (Shumway-style)
                def _compute_ret_adj_v1(row):
                    if pd.isna(row["dlstcd"]):
                        return row["ret"]
                    if not pd.isna(row["dlret"]):
                        return row["dlret"]
                    code = row["dlstcd"]
                    if code in (500, 520, 580, 584) or (
                        551 <= code <= 574
                    ):
                        return -0.30
                    if code == 100:
                        return row["ret"]
                    return -1.0

                crsp_monthly["ret_adj"] = crsp_monthly.apply(
                    _compute_ret_adj_v1, axis=1
                )
                crsp_monthly = crsp_monthly.drop(
                    columns=["dlret", "dlstcd"]
                )

                # prc_adj = |altprc nullified-at-zero| / cfacpr
                prc_zeroed = crsp_monthly["altprc"].replace(0, np.nan)
                crsp_monthly["prc_adj"] = (
                    prc_zeroed.abs() / crsp_monthly["cfacpr"]
                ).replace([np.inf, -np.inf], np.nan)

                # Merge risk-free and compute excess return
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date, end_date=end_date
                )
                crsp_monthly = (
                    crsp_monthly.merge(
                        risk_free_monthly, how="left", on="date"
                    )
                    .assign(
                        ret_excess=lambda x: x["ret_adj"]
                        - x["risk_free"]
                    )
                    .drop(columns="risk_free")
                    .dropna(subset=["ret_excess", "mktcap"])
                )

                processed_data = crsp_monthly
            elif version == "v2":
                crsp_query = f"""
                    SELECT msf.permno,
                        date_trunc('month', msf.mthcaldt)::date AS date,
                        msf.mthcaldt AS calculation_date,
                        msf.mthret AS ret, msf.shrout,
                        msf.mthprc AS prc,
                        ssih.primaryexch, ssih.siccd,
                        fcd.first_crsp_date
                        {", " + additional_columns_sql if additional_columns_sql else ""}
                    FROM crsp.msf_v2 AS msf
                    INNER JOIN crsp.stksecurityinfohist AS ssih
                    ON msf.permno = ssih.permno AND
                        ssih.secinfostartdt <= msf.mthcaldt AND
                        msf.mthcaldt <= ssih.secinfoenddt
                    LEFT JOIN (
                        SELECT permno,
                            MIN(secinfostartdt) AS first_crsp_date
                        FROM crsp.stksecurityinfohist
                        GROUP BY permno
                    ) AS fcd ON msf.permno = fcd.permno
                    WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}'
                    AND ssih.sharetype = 'NS'
                    AND ssih.securitytype = 'EQTY'
                    AND ssih.securitysubtype = 'COM'
                    AND ssih.usincflg = 'Y'
                    AND ssih.issuertype in ('ACOR', 'CORP')
                    AND ssih.primaryexch in ('N', 'A', 'Q')
                    AND ssih.conditionaltype in ('RW', 'NW')
                    AND ssih.tradingstatusflg = 'A'
                    """

                crsp_monthly = (
                    pd.read_sql_query(
                        sql=crsp_query,
                        con=wrds_connection,
                        dtype={"permno": int, "siccd": int},
                        parse_dates={
                            "date",
                            "calculation_date",
                            "first_crsp_date",
                        },
                    )
                    .assign(shrout=lambda x: x["shrout"] * 1000)
                    # listing_age is assigned before mktcap to keep the
                    # documented column order:
                    # ..., siccd, listing_age, mktcap, mktcap_lag, ...
                    .assign(
                        listing_age=lambda df: (
                            (df["date"].dt.year - df["first_crsp_date"].dt.year)
                            * 12
                            + (
                                df["date"].dt.month
                                - df["first_crsp_date"].dt.month
                            )
                            - (
                                df["date"].dt.day < df["first_crsp_date"].dt.day
                            ).astype(int)
                        ).clip(lower=0)
                    )
                    .assign(mktcap=lambda x: x["shrout"] * x["prc"] / 1000000)
                    .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
                    .drop(columns=["first_crsp_date"])
                )

                mktcap_lag = crsp_monthly.assign(
                    date=lambda x: x["date"] + pd.DateOffset(months=1),
                    mktcap_lag=lambda x: x["mktcap"],
                ).get(["permno", "date", "mktcap_lag"])

                crsp_monthly = crsp_monthly.merge(
                    mktcap_lag, how="left", on=["permno", "date"]
                ).assign(
                    exchange=lambda x: x["primaryexch"].apply(_assign_exchange),
                    industry=lambda x: x["siccd"].apply(_assign_industry),
                )
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date,
                    end_date=end_date,
                )
                crsp_monthly = (
                    crsp_monthly.merge(risk_free_monthly, how="left", on="date")
                    .assign(ret_excess=lambda x: x["ret"] - x["risk_free"])
                    .drop(columns=["risk_free"])
                    .dropna(subset=["ret_excess", "mktcap"])
                )
                processed_data = crsp_monthly
        elif "crsp_daily" in dataset:
            if version == "v1":
                # Distinct permnos from dsf within the date range
                permnos_query = text(
                    f"SELECT DISTINCT permno FROM crsp.dsf "
                    f"WHERE date BETWEEN '{start_date}' "
                    f"AND '{end_date}'"
                )
                permnos = pd.read_sql(
                    permnos_query,
                    con=wrds_connection,
                    dtype={"permno": int},
                )
                permnos = list(permnos["permno"].astype(str))

                if len(permnos) > 0:
                    batches = int(
                        np.ceil(len(permnos) / batch_size)
                    )
                    risk_free_daily = _download_data_risk_free(
                        start_date=start_date,
                        end_date=end_date,
                        frequency="daily",
                    )

                    for j in range(1, batches + 1):
                        permno_batch = permnos[
                            ((j - 1) * batch_size):(
                                min(j * batch_size, len(permnos))
                            )
                        ]
                        permno_batch_str = ", ".join(
                            f"'{p}'" for p in permno_batch
                        )
                        permno_in = f"({permno_batch_str})"

                        dsf_query = text(f"""
                            SELECT dsf.permno, dsf.date, dsf.ret
                                {", " + additional_columns_sql if additional_columns_sql else ""}
                            FROM crsp.dsf AS dsf
                            INNER JOIN crsp.msenames AS msn
                              ON dsf.permno = msn.permno
                              AND dsf.date BETWEEN msn.namedt
                                  AND msn.nameendt
                              AND msn.shrcd IN (10, 11)
                            WHERE dsf.permno IN {permno_in}
                            AND dsf.date BETWEEN '{start_date}'
                                AND '{end_date}'
                        """)
                        crsp_daily_sub = pd.read_sql_query(
                            dsf_query,
                            con=wrds_connection,
                            parse_dates={"date"},
                        ).dropna(subset=["permno", "date", "ret"])

                        if crsp_daily_sub.empty:
                            continue

                        # Per-batch msedelist
                        msedelist_query = text(
                            "SELECT permno, dlstdt, dlret "
                            "FROM crsp.msedelist "
                            f"WHERE permno IN {permno_in}"
                        )
                        msedelist_sub = pd.read_sql_query(
                            msedelist_query,
                            con=wrds_connection,
                            parse_dates={"dlstdt"},
                        ).dropna()

                        # Merge dsf with msedelist on (permno, date=dlstdt)
                        if not msedelist_sub.empty:
                            crsp_daily_sub = crsp_daily_sub.merge(
                                msedelist_sub.rename(
                                    columns={"dlstdt": "date"}
                                ),
                                on=["permno", "date"],
                                how="left",
                            )

                            # Bind delisting-only rows that don't
                            # match any dsf date
                            matched_dates = crsp_daily_sub[
                                crsp_daily_sub["dlret"].notna()
                            ][["permno", "date"]].drop_duplicates()
                            unmatched = msedelist_sub.merge(
                                matched_dates.rename(
                                    columns={"date": "dlstdt"}
                                ),
                                on=["permno", "dlstdt"],
                                how="left",
                                indicator=True,
                            ).query("_merge == 'left_only'").drop(
                                columns="_merge"
                            )
                            if not unmatched.empty:
                                unmatched = unmatched.rename(
                                    columns={"dlstdt": "date"}
                                )
                                for col in crsp_daily_sub.columns:
                                    if col not in unmatched.columns:
                                        unmatched[col] = np.nan
                                unmatched = unmatched[
                                    crsp_daily_sub.columns
                                ]
                                crsp_daily_sub = pd.concat(
                                    [crsp_daily_sub, unmatched],
                                    ignore_index=True,
                                )

                            # ret = dlret if not NaN
                            crsp_daily_sub["ret"] = np.where(
                                crsp_daily_sub["dlret"].notna(),
                                crsp_daily_sub["dlret"],
                                crsp_daily_sub["ret"],
                            )
                            crsp_daily_sub = crsp_daily_sub.drop(
                                columns="dlret"
                            )

                            # Filter date <= permno's last delisting
                            # date (or end_date if no delisting)
                            permno_dlstdt = (
                                msedelist_sub.groupby("permno")[
                                    "dlstdt"
                                ]
                                .max()
                                .reset_index()
                                .rename(
                                    columns={
                                        "dlstdt": "_permno_dlstdt"
                                    }
                                )
                            )
                            crsp_daily_sub = crsp_daily_sub.merge(
                                permno_dlstdt,
                                on="permno",
                                how="left",
                            )
                            crsp_daily_sub["_permno_dlstdt"] = (
                                crsp_daily_sub["_permno_dlstdt"]
                                .fillna(pd.Timestamp(end_date))
                            )
                            crsp_daily_sub = crsp_daily_sub[
                                crsp_daily_sub["date"]
                                <= crsp_daily_sub["_permno_dlstdt"]
                            ].drop(columns="_permno_dlstdt")

                        # Merge risk_free, compute ret_excess
                        crsp_daily_sub = (
                            crsp_daily_sub.merge(
                                risk_free_daily,
                                on="date",
                                how="left",
                            )
                            .assign(
                                ret_excess=lambda x: x["ret"]
                                - x["risk_free"]
                            )
                            .drop(columns="risk_free")
                        )

                        crsp_data = pd.concat(
                            [crsp_data, crsp_daily_sub]
                        )

                # Gao-Ritter volume adjustment for NASDAQ (v1: exchcd==3)
                if adjust_volume and not crsp_data.empty:
                    gr_date_1 = pd.Timestamp("2001-02-01")
                    gr_date_2 = pd.Timestamp("2002-01-01")
                    gr_date_3 = pd.Timestamp("2004-01-01")

                    crsp_data = (
                        crsp_data.sort_values(["permno", "date"])
                        .assign(
                            vol=lambda df: df["vol"].replace(
                                -99, np.nan
                            ),
                            prc=lambda df: df["prc"].replace(
                                0, np.nan
                            ),
                        )
                        .assign(
                            prc_adj=lambda df: (
                                df["prc"].abs() / df["cfacpr"]
                            ).replace([np.inf, -np.inf], np.nan)
                        )
                        .assign(
                            vol_adj=lambda df: np.select(
                                [
                                    (df["exchcd"] == 3)
                                    & (df["date"] < gr_date_1),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_1)
                                    & (df["date"] < gr_date_2),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_2)
                                    & (df["date"] < gr_date_3),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_3),
                                ],
                                [
                                    df["vol"] / 2.0,
                                    df["vol"] / 1.8,
                                    df["vol"] / 1.6,
                                    df["vol"] / 1.0,
                                ],
                                default=df["vol"],
                            )
                        )
                    )

                processed_data = crsp_data
            elif version == "v2":
                permnos = pd.read_sql(
                    sql="SELECT DISTINCT permno FROM crsp.stksecurityinfohist",
                    con=wrds_connection,
                    dtype={"permno": int},
                )
                permnos = list(permnos["permno"].astype(str))
                batches = np.ceil(len(permnos) / batch_size).astype(int)

                risk_free_daily = _download_data_risk_free(
                    start_date=start_date,
                    end_date=end_date,
                    frequency="daily",
                )

                for j in range(1, batches + 1):
                    permno_batch = permnos[
                        ((j - 1) * batch_size) : (
                            min(j * batch_size, len(permnos))
                        )
                    ]
                    permno_batch_formatted = ", ".join(
                        f"'{permno}'" for permno in permno_batch
                    )
                    permno_string = f"({permno_batch_formatted})"

                    crsp_daily_sub_query = f"""
                        SELECT dsf.permno, dlycaldt AS date, dlyret AS ret
                            {", " + additional_columns_sql if additional_columns_sql else ""}
                        FROM crsp.dsf_v2 AS dsf
                        INNER JOIN crsp.stksecurityinfohist AS ssih
                        ON dsf.permno = ssih.permno AND
                            ssih.secinfostartdt <= dsf.dlycaldt AND
                            dsf.dlycaldt <= ssih.secinfoenddt
                        WHERE dsf.permno IN {permno_string}
                        AND dlycaldt BETWEEN '{start_date}' AND '{end_date}'
                        AND ssih.sharetype = 'NS'
                        AND ssih.securitytype = 'EQTY'
                        AND ssih.securitysubtype = 'COM'
                        AND ssih.usincflg = 'Y'
                        AND ssih.issuertype in ('ACOR', 'CORP')
                        AND ssih.primaryexch in ('N', 'A', 'Q')
                        AND ssih.conditionaltype in ('RW', 'NW')
                        AND ssih.tradingstatusflg = 'A'
                        """

                    crsp_daily_sub = pd.read_sql_query(
                        sql=crsp_daily_sub_query,
                        con=wrds_connection,
                        dtype={"permno": int},
                        parse_dates={"date"},
                    ).dropna(subset=["permno", "date", "ret"])

                    if not crsp_daily_sub.empty:
                        crsp_daily_sub = (
                            crsp_daily_sub.merge(
                                risk_free_daily,
                                on="date",
                                how="left",
                            )
                            .assign(
                                ret_excess=lambda x: x["ret"] - x["risk_free"]
                            )
                            .drop(columns=["risk_free"])
                        )

                    print(
                        f"Batch {j} out of {batches} done "
                        f"({(j / batches) * 100:.2f}%)\n"
                    )

                    crsp_data = pd.concat([crsp_data, crsp_daily_sub])

                if adjust_volume and not crsp_data.empty:
                    gr_date_1 = pd.Timestamp("2001-02-01")
                    gr_date_2 = pd.Timestamp("2002-01-01")
                    gr_date_3 = pd.Timestamp("2004-01-01")

                    crsp_data = (
                        crsp_data.sort_values(["permno", "date"])
                        .assign(
                            cfacpr=lambda df: df.groupby("permno")[
                                "dlyfacprc"
                            ].cumprod(),
                            vol=lambda df: df["dlyvol"].replace(-99, np.nan),
                            prc=lambda df: df["dlyprc"].replace(0, np.nan),
                        )
                        .assign(
                            prc_adj=lambda df: (
                                df["prc"].abs() / df["cfacpr"]
                            ).replace([np.inf, -np.inf], np.nan)
                        )
                        .assign(
                            vol_adj=lambda df: np.select(
                                [
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] < gr_date_1),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_1)
                                    & (df["date"] < gr_date_2),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_2)
                                    & (df["date"] < gr_date_3),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_3),
                                ],
                                [
                                    df["vol"] / 2.0,
                                    df["vol"] / 1.8,
                                    df["vol"] / 1.6,
                                    df["vol"] / 1.0,
                                ],
                                default=df["vol"],
                            )
                        )
                        .drop(columns=["dlyvol", "dlyprc", "dlyfacprc"])
                    )

                processed_data = crsp_data
        else:
            raise ValueError(
                "Invalid dataset. Use 'crsp_monthly' or 'crsp_daily'."
            )
    finally:
        disconnect_connection(wrds_connection)

    if add_ccm_links:
        ccm_links = _download_data_wrds_ccm_links()
        merged = processed_data[["permno", "date"]].merge(
            ccm_links, on="permno", how="inner"
        )
        valid_links = merged.query(
            "gvkey.notna() and linkdt <= date <= linkenddt"
        )[["permno", "gvkey", "date"]]
        processed_data = processed_data.merge(
            valid_links, on=["permno", "date"], how="left"
        )

    return processed_data


def _download_data_wrds_ccm_links(
    linktype: list[str] = ["LU", "LC"], linkprim: list[str] = ["P", "C"]
) -> pd.DataFrame:
    """
    Download data from the WRDS CRSP/Compustat Merged (CCM) links database.

    Parameters
    ----------
    linktype : list of str, optional
        A list of strings indicating the types of links to download.
        Defaults to ["LU", "LC"].
    linkprim : list of str, optional
        A list of strings indicating the primacy of the links.
        Defaults to ["P", "C"].

    Returns
    -------
    pd.DataFrame
        A data frame containing columns permno, gvkey, linkprim, linkdt, and
        linkenddt (missing end dates replaced with today's date).
    """
    conn = get_wrds_connection()

    query = f"""
        SELECT lpermno AS permno, gvkey, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ({",".join(f"'{lt}'" for lt in linktype)})
        AND linkprim IN ({",".join(f"'{lp}'" for lp in linkprim)})
    """

    ccm_links = pd.read_sql(query, conn)

    ccm_links["linkenddt"] = ccm_links["linkenddt"].fillna(pd.Timestamp.today())

    disconnect_connection(conn)

    return ccm_links


def _download_data_wrds_compustat(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    additional_columns: list = None,
    only_usd: bool = False,
    only_us: bool = None,
) -> pd.DataFrame:
    """
    Download data from WRDS Compustat.

    Downloads financial data from the WRDS Compustat database for a
    given dataset, start date, and end date. The function filters the
    data according to industry format, data format, and consolidation
    level, and returns the most current data for each reporting
    period. Additionally, the annual data also includes the calculated
    book equity ('be'), operating profitability ('op'), and investment
    ('inv') for each company following Fama and French (1993, 2015),
    as well as income before extraordinary items ('ib').

    Parameters
    ----------
    dataset : str
        The dataset to download ('compustat_annual' or
        'compustat_quarterly').
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped (e.g., 'wrds_compustat_annual'
        becomes 'compustat_annual') and a DeprecationWarning is
        emitted.
    additional_columns : list of str, optional
        Additional columns from the Compustat table as a list of
        strings.
    only_usd : bool, optional
        A boolean indicating whether only USD-denominated shares
        should be returned (i.e., excluding Canadian firms). Defaults
        to False.
    only_us : bool, optional
        Deprecated. Use 'only_usd' instead. If supplied, the value is
        forwarded to 'only_usd' and a DeprecationWarning is emitted.

    Returns
    -------
    pd.DataFrame
        A data frame with financial data for the specified period,
        including variables for book equity ('be'), operating
        profitability ('op'), investment ('inv'), and others.

    References
    ----------
    Fama, E. F., and French, K. R. (1993). Common risk factors in the
    returns on stocks and bonds. Journal of Financial Economics,
    33(1), 3-56. https://doi.org/10.1016/0304-405X(93)90023-5

    Fama, E. F., and French, K. R. (2015). A five-factor asset pricing
    model. Journal of Financial Economics, 116(1), 1-22.
    https://doi.org/10.1016/j.jfineco.2014.10.010

    Examples
    --------
    >>> from tidyfinance import download_data_wrds_compustat
    >>> download_data_wrds_compustat(
    ...     'compustat_annual', '2020-01-01', '2020-12-31'
    ... )
    >>> download_data_wrds_compustat(
    ...     'compustat_quarterly', '2020-01-01', '2020-12-31'
    ... )
    >>> download_data_wrds_compustat(
    ...     'compustat_annual', additional_columns=['aodo', 'aldo']
    ... )
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if only_us is not None:
        warnings.warn(
            "The 'only_us' argument is deprecated. Use 'only_usd' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        only_usd = only_us

    if dataset is not None and dataset.startswith("wrds_"):
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'compustat_annual' instead of 'wrds_compustat_annual'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    start_date, end_date = _validate_dates(start_date, end_date)

    if dataset not in ["compustat_annual", "compustat_quarterly"]:
        raise ValueError(
            (
                "Invalid dataset specified. "
                "Use 'compustat_annual' or 'compustat_quarterly'."
            )
        )

    wrds_connection = get_wrds_connection()
    _base_annual_cols = {
        "gvkey", "datadate", "seq", "ceq", "at", "lt", "txditc", "txdb",
        "itcb", "pstkrv", "pstkl", "pstk", "capx", "oancf", "sale",
        "cogs", "xint", "xsga", "ib", "curcd",
    }
    _base_quarterly_cols = {
        "gvkey", "datadate", "rdq", "fqtr", "fyearq", "atq", "ceqq", "curcdq",
    }
    extra_annual = [
        c for c in (additional_columns or [])
        if c != "curcd" and c not in _base_annual_cols
    ]
    extra_quarterly = [
        c for c in (additional_columns or [])
        if c != "curcdq" and c not in _base_quarterly_cols
    ]
    additional_columns_annual = ", ".join(extra_annual) if extra_annual else ""
    additional_columns_quarterly = (
        ", ".join(extra_quarterly) if extra_quarterly else ""
    )

    if "compustat_annual" in dataset:
        query = text(f"""
            SELECT gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb,
                pstkrv, pstkl, pstk, capx, oancf, sale, cogs, xint, xsga,
                ib, curcd
                {", " + additional_columns_annual if additional_columns_annual else ""}
            FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """)

        compustat = pd.read_sql(query, wrds_connection)
        disconnect_connection(wrds_connection)

        # Compute Book Equity (be)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            compustat = compustat.assign(
                be=lambda x: (
                    x["seq"]
                    .combine_first(x["ceq"] + x["pstk"])
                    .combine_first(x["at"] - x["lt"])
                    + x["txditc"].combine_first(x["txdb"] + x["itcb"]).fillna(0)
                    - x["pstkrv"]
                    .combine_first(x["pstkl"])
                    .combine_first(x["pstk"])
                    .fillna(0)
                )
            ).assign(
                be=lambda x: x["be"].apply(lambda y: np.nan if y <= 0 else y)
            )
        # Compute Operating Profitability (op)
        compustat = compustat.assign(
            op=lambda df: (
                (
                    df["sale"]
                    - df[["cogs", "xsga", "xint"]].fillna(0).sum(axis=1)
                )
                / df["be"]
            )
        )
        # Keep the latest report per company per year
        compustat = (
            compustat.assign(
                year=lambda x: pd.DatetimeIndex(x["datadate"]).year
            )
            .sort_values("datadate")
            .groupby(["gvkey", "year"])
            .tail(1)
            .reset_index()
        )
        # Compute Investment (inv)
        compustat_lag = (
            compustat.get(["gvkey", "year", "at"])
            .assign(year=lambda x: x["year"] + 1)
            .rename(columns={"at": "at_lag"})
        )

        compustat = (
            compustat.merge(compustat_lag, how="left", on=["gvkey", "year"])
            .assign(inv=lambda x: x["at"] / x["at_lag"] - 1)
            .assign(inv=lambda x: np.where(x["at_lag"] <= 0, np.nan, x["inv"]))
        )

        if only_usd:
            compustat = compustat[compustat["curcd"] == "USD"]

        processed_data = compustat.assign(
            date=lambda df: (
                pd.to_datetime(df["datadate"]).dt.to_period("M").dt.start_time
            )
        ).drop(columns=["year", "at_lag", "curcd"])

    elif "compustat_quarterly" in dataset:
        query = text(f"""
            SELECT gvkey, datadate, rdq, fqtr, fyearq, atq, ceqq, curcdq
                {", " + additional_columns_quarterly if additional_columns_quarterly else ""}
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """)

        compustat = pd.read_sql(query, wrds_connection)
        disconnect_connection(wrds_connection)

        # Ensure necessary columns are not missing
        compustat = (
            compustat.dropna(subset=["gvkey", "datadate", "fyearq", "fqtr"])
            .assign(
                date=lambda df: (
                    pd.to_datetime(df["datadate"])
                    .dt.to_period("M")
                    .dt.start_time
                )
            )
            .sort_values("datadate", ascending=False, kind="stable")
            .drop_duplicates(subset=["gvkey", "fyearq", "fqtr"], keep="first")
            .sort_values(
                ["gvkey", "date", "rdq"], na_position="last", kind="stable"
            )
            .drop_duplicates(subset=["gvkey", "date"], keep="first")
            .query("rdq.isna() or date < rdq")
        )

        if only_usd:
            compustat = compustat[compustat["curcdq"] == "USD"]

        processed_data = compustat.get(
            ["gvkey", "date", "datadate", "atq", "ceqq"] + extra_quarterly
        )

    return processed_data


def _download_data_wrds_fisd(additional_columns: list = None) -> pd.DataFrame:
    """
    Download filtered FISD data from WRDS.

    Establishes a connection to the WRDS database to download a
    filtered subset of the FISD (Fixed Income Securities Database).
    The function filters the 'fisd_mergedissue' and
    'fisd_mergedissuer' tables based on several criteria related to
    the securities, such as security level, bond type, coupon type,
    and others, focusing on specific attributes that denote the nature
    of the securities. It finally returns a data frame with selected
    fields from the 'fisd_mergedissue' table after joining it with
    issuer information from the 'fisd_mergedissuer' table for issuers
    domiciled in the USA.

    Parameters
    ----------
    additional_columns : list of str, optional
        Additional columns from the FISD table as a list of strings.

    Returns
    -------
    pd.DataFrame
        A data frame containing a subset of FISD data with fields
        related to the bond's characteristics and issuer information.
        This includes complete CUSIP, maturity date, offering amount,
        offering date, dated date, interest frequency, coupon, last
        interest date, issue ID, issuer ID, and SIC code of the
        issuer.

    Examples
    --------
    >>> from tidyfinance import download_data_wrds_fisd
    >>> fisd = download_data_wrds_fisd()
    >>> fisd_extended = download_data_wrds_fisd(
    ...     additional_columns=['asset_backed', 'defeased']
    ... )
    """
    wrds_connection = get_wrds_connection()

    if additional_columns:
        if not all(
            re.match(r"^[a-z_][a-z0-9_]*$", col) for col in additional_columns
        ):
            raise ValueError("Column names must be valid SQL identifiers.")

    additional_columns_str = _process_additional_columns(additional_columns)

    fisd_query = (
        "SELECT complete_cusip, maturity, offering_amt, offering_date, "
        "dated_date, interest_frequency, coupon, last_interest_date, "
        f"issue_id, issuer_id{additional_columns_str} "
        "FROM fisd.fisd_mergedissue "
        "WHERE security_level = 'SEN' "
        "AND (slob = 'N' OR slob IS NULL) "
        "AND security_pledge IS NULL "
        "AND (asset_backed = 'N' OR asset_backed IS NULL) "
        "AND (defeased = 'N' OR defeased IS NULL) "
        "AND defeased_date IS NULL "
        "AND bond_type IN ('CDEB', 'CMTN', 'CMTZ', 'CZ', 'USBN') "
        "AND (pay_in_kind != 'Y' OR pay_in_kind IS NULL) "
        "AND pay_in_kind_exp_date IS NULL "
        "AND (yankee = 'N' OR yankee IS NULL) "
        "AND (canadian = 'N' OR canadian IS NULL) "
        "AND foreign_currency = 'N' "
        "AND coupon_type IN ('F', 'Z') "
        "AND fix_frequency IS NULL "
        "AND coupon_change_indicator = 'N' "
        "AND interest_frequency IN ('0', '1', '2', '4', '12') "
        "AND rule_144a = 'N' "
        "AND (private_placement = 'N' OR private_placement IS NULL) "
        "AND defaulted = 'N' "
        "AND filing_date IS NULL "
        "AND settlement IS NULL "
        "AND convertible = 'N' "
        "AND exchange IS NULL "
        "AND (putable = 'N' OR putable IS NULL) "
        "AND (unit_deal = 'N' OR unit_deal IS NULL) "
        "AND (exchangeable = 'N' OR exchangeable IS NULL) "
        "AND perpetual = 'N' "
        "AND (preferred_security = 'N' OR preferred_security IS NULL)"
    )

    fisd = pd.read_sql_query(
        sql=fisd_query,
        con=wrds_connection,
        dtype={
            "complete_cusip": str,
            "interest_frequency": str,
            "issue_id": int,
            "issuer_id": int,
        },
        parse_dates={
            "maturity",
            "offering_date",
            "dated_date",
            "last_interest_date",
        },
    )

    fisd_issuer_query = (
        "SELECT issuer_id, sic_code, country_domicile "
        "FROM fisd.fisd_mergedissuer"
    )

    fisd_issuer = pd.read_sql_query(
        sql=fisd_issuer_query,
        con=wrds_connection,
        dtype={"issuer_id": int, "sic_code": str, "country_domicile": str},
    )

    fisd = (
        fisd.merge(fisd_issuer, how="inner", on="issuer_id")
        .query("country_domicile == 'USA'")
        .drop(columns="country_domicile")
    )

    disconnect_connection(wrds_connection)

    return fisd


def _download_data_wrds_trace_enhanced(
    cusips: list, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Download Enhanced TRACE data from WRDS.

    Establishes a connection to the WRDS database to download the
    specified CUSIPs trade messages from the Trade Reporting and
    Compliance Engine (TRACE). The trade data is cleaned as suggested
    by Dick-Nielsen (2009, 2014).

    Parameters
    ----------
    cusips : list of str
        A character vector specifying the 9-digit CUSIPs to download.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.

    Returns
    -------
    pd.DataFrame
        A data frame containing the cleaned trade messages from TRACE
        for the selected CUSIPs over the time window specified. Output
        variables include identifying information (i.e., CUSIP, trade
        date/time) and trade-specific information (i.e., price/yield,
        volume, counterparty, and reporting side).

    References
    ----------
    Dick-Nielsen, J. (2009). Liquidity biases in TRACE. Journal of
    Fixed Income, 19(2), 43-55.
    https://doi.org/10.3905/jfi.2009.19.2.043

    Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working
    Paper. https://doi.org/10.2139/ssrn.2337908

    Examples
    --------
    >>> from tidyfinance import download_data_wrds_trace_enhanced
    >>> download_data_wrds_trace_enhanced(
    ...     ['00101JAH9'], '2019-01-01', '2021-12-31'
    ... )
    """
    if not all(isinstance(cusip, str) and len(cusip) == 9 for cusip in cusips):
        raise ValueError("All CUSIPs must be 9-character strings.")

    wrds_connection = get_wrds_connection()

    cusip_string = _format_cusips(cusips)

    query = (
        "SELECT cusip_id, bond_sym_id, trd_exctn_dt, "
        "trd_exctn_tm, days_to_sttl_ct, lckd_in_ind, "
        "wis_fl, sale_cndtn_cd, msg_seq_nb, "
        "trc_st, trd_rpt_dt, trd_rpt_tm, "
        "entrd_vol_qt, rptd_pr, yld_pt, "
        "asof_cd, orig_msg_seq_nb, rpt_side_cd, "
        "cntra_mp_id, stlmnt_dt, spcl_trd_fl "
        "FROM trace.trace_enhanced "
        f"WHERE cusip_id IN {cusip_string} "
    )

    if start_date and end_date:
        query += f"AND trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'"

    trace_enhanced_raw = pd.read_sql(
        query,
        wrds_connection,
        parse_dates={"trd_exctn_dt", "trd_rpt_dt", "stlmnt_dt"},
    )
    disconnect_connection(wrds_connection)

    trace_enhanced = process_trace_data(trace_enhanced_raw)

    return trace_enhanced


# %% hugging face functions for tidy finance data

def _get_available_huggingface_files(
    organization: str, dataset: str
) -> pd.DataFrame:
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
    pd.DataFrame
        A data frame with columns 'path' (str) and 'size' (int).

    Examples
    --------
    >>> from tidyfinance.data_download import _get_available_huggingface_files
    >>> _get_available_huggingface_files('voigtstefan', 'sp500')
    """
    api_url = (
        f"https://huggingface.co/api/datasets/{organization}/{dataset}"
        "/tree/main?recursive=1"
    )
    rows = []
    next_url = api_url

    while next_url:
        resp = requests.get(next_url, timeout=30)
        resp.raise_for_status()
        for entry in resp.json():
            is_file = entry.get("type") == "file"
            is_parquet = entry["path"].endswith(".parquet")
            if is_file and is_parquet:
                rows.append({"path": entry["path"], "size": entry.get("size")})
        link = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        next_url = match.group(1) if match else None

    return pd.DataFrame(rows, columns=["path", "size"])


def _download_factor_library_grid() -> pd.DataFrame:
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
    pd.DataFrame
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
    >>> from tidyfinance.data_download import _download_factor_library_grid
    >>> _download_factor_library_grid()
    """
    available = _get_available_huggingface_files(
        "tidy-finance", "factor-library-grid"
    )
    if available.empty or "path" not in available.columns:
        raise ValueError(
            "No parquet files were found in the Hugging Face dataset repo "
            "'tidy-finance/factor-library-grid'."
        )
    grid_path = available["path"].iloc[0]
    grid_url = (
        "https://huggingface.co/datasets/tidy-finance"
        f"/factor-library-grid/resolve/main/{grid_path}"
    )
    return pd.read_parquet(grid_url)


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
        levels. Supported columns and their defaults are:

        - 'sorting_variable': no default, required.
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

    if "sorting_variable" not in filters:
        raise ValueError("'sorting_variable' is required in filters.")

    if filters.get("sorting_method", "univariate") != "univariate":
        if filters.get("n_portfolios_secondary") is None:
            raise ValueError(
                "When sorting_method is not 'univariate', "
                "n_portfolios_secondary must be provided."
            )

    if not fill_all:
        filters = {**_FACTOR_LIBRARY_DEFAULTS, **filters}

    grid = _download_factor_library_grid().assign(
        sorting_variable=lambda x: x["sorting_variable"].str.replace(
            r"^sv_", "", regex=True
        )
    )

    for col, value in filters.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if values == [None]:
            grid = grid.loc[grid[col].isna()]
        else:
            grid = grid.loc[grid[col].isin(values)]
    return grid["id"].tolist()


def _fetch_parquet_url(url, retries=5, backoff=2.0):
    """Fetch a parquet file via a reused curl_cffi session with retry."""
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_err = None
    for attempt in range(retries):
        try:
            r = _hf_session.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return pd.read_parquet(io.BytesIO(r.content))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise ConnectionError(
        f"Failed to download {url} after {retries} attempts: {last_err}"
    ) from last_err


def _download_factor_library_ids(ids: list) -> pd.DataFrame:
    """
    Download factor library returns for a vector of portfolio IDs.

    Given a vector of portfolio IDs from the
    'tidy-finance/factor-library-grid' Hugging Face dataset, downloads
    the corresponding return data from the
    'tidy-finance/factor-library' dataset on Hugging Face. The
    function identifies the unique combinations of 'sorting_variable',
    'sorting_variable_lag', 'sorting_method', and 'n_portfolios_main'
    for the requested IDs, downloads one parquet file per combination
    in full, and then inner-joins to retain only the requested IDs.
    The grid metadata is joined back onto the result.

    Use this function when you already know the portfolio IDs you want
    (for example, from a previous call to '_download_data_huggingface'
    with 'dataset' set to 'factor_library'). To resolve IDs from
    filter criteria (sorting variable, weighting scheme, breakpoints,
    etc.) and download in a single call, use
    '_download_data_huggingface' instead.

    Raises an error if 'ids' is empty or contains IDs that cannot be
    matched to a parquet file, listing the affected IDs and their key
    columns.

    Parameters
    ----------
    ids : list of int
        Portfolio IDs to download. IDs correspond to rows of the
        'tidy-finance/factor-library-grid' dataset.

    Returns
    -------
    pd.DataFrame
        A data frame of portfolio returns with the grid metadata
        columns for the requested IDs appended.

    Raises
    ------
    ValueError
        If 'ids' is empty, or if any ID cannot be matched to a parquet
        file in the factor library.

    Examples
    --------
    >>> from tidyfinance.data_download import _download_factor_library_ids
    >>> _download_factor_library_ids([1, 2, 3])
    """
    if not ids:
        raise ValueError(
            "No portfolio IDs provided. "
            "Check that your filter criteria match at least one portfolio."
        )

    organization = "tidy-finance"
    dataset_name = "factor-library"

    path_pattern = re.compile(
        r"sorting_variable=([^/]+)/sorting_variable_lag=([^/]+)/"
    )
    available_files = _get_available_huggingface_files(
        organization, dataset_name
    )

    def _extract_keys(path: str) -> tuple:
        m = path_pattern.search(path)
        return (m.group(1), m.group(2)) if m else (None, None)

    available_files[["sorting_variable", "sorting_variable_lag"]] = (
        pd.DataFrame(
            available_files["path"].apply(_extract_keys).tolist(),
            index=available_files.index,
        )
    )

    grid = _download_factor_library_grid().assign(
        sorting_variable=lambda x: x["sorting_variable"].str.replace(
            r"^sv_", "", regex=True
        )
    )

    id_grid = grid.loc[grid["id"].isin(ids)].merge(
        available_files[["sorting_variable", "sorting_variable_lag", "path"]],
        on=["sorting_variable", "sorting_variable_lag"],
        how="left",
    )

    missing = id_grid.loc[id_grid["path"].isna()]
    if not missing.empty:
        missing_keys = [
            f"id={row.id} ({row.sorting_variable} / {row.sorting_variable_lag})"
            for row in missing.itertuples()
        ]
        raise ValueError(
            f"No parquet file found for {len(missing_keys)} portfolio ID(s): "
            f"{missing_keys}. Check that the sorting_variable and "
            "sorting_variable_lag values exist in the factor library."
        )

    unique_paths = id_grid["path"].dropna().unique()

    def _make_url(p):
        return (
            f"https://huggingface.co/datasets/{organization}"
            f"/{dataset_name}/resolve/main/{p}"
        )

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(_fetch_parquet_url,
                             [_make_url(p) for p in unique_paths])
                      )

    returns = pd.concat(frames, ignore_index=True)

    meta_cols = [c for c in id_grid.columns if c not in ("path", "size")]
    return returns.merge(
        id_grid[meta_cols].drop_duplicates("id"),
        on="id",
        how="inner",
    )


def _download_data_huggingface_factor_library(
    fill_all: bool = False,
    ids: list = None,
    start_date: str = None,
    end_date: str = None,
    **filters,
) -> pd.DataFrame:
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
    pd.DataFrame
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

    # Normalize the date column to proper datetime64 so comparisons
    # work regardless of whether the parquet stored dates as
    # datetime.date, pd.Timestamp, or tz-aware datetime.
    if start_date is not None or end_date is not None:
        result = result.copy()
        result["date"] = pd.to_datetime(result["date"])
        if getattr(result["date"].dt, "tz", None) is not None:
            result["date"] = result["date"].dt.tz_localize(None)

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        if getattr(start_ts, "tz", None) is not None:
            start_ts = start_ts.tz_localize(None)
        result = result[result["date"] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        if getattr(end_ts, "tz", None) is not None:
            end_ts = end_ts.tz_localize(None)
        result = result[result["date"] <= end_ts]
    if start_date is not None or end_date is not None:
        result = result.reset_index(drop=True)

    return result


def _download_data_huggingface(
    dataset: str = None,
    start_date: str | date = None,
    end_date: str | date = None,
    type: str = None,
    **kwargs,
) -> pd.DataFrame:
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

    - 'sorting_variable': required. The firm characteristic used to
      sort stocks into portfolios (e.g., 'me' for market equity, 'bm'
      for book-to-market). No default is applied.
    - 'min_size_quantile' (defaults to 0.2): fraction of the smallest
      stocks (by market cap) excluded from the portfolio universe; 0.2
      drops the bottom 20%.
    - 'exclude_financials' (defaults to False): whether to drop
      financial-sector stocks (SIC 6000-6999) from the universe.
    - 'exclude_utilities' (defaults to False): whether to drop
      utility-sector stocks (SIC 4900-4999) from the universe.
    - 'exclude_negative_earnings' (defaults to False): whether to drop
      firms with negative earnings before sorting.
    - 'sorting_variable_lag' (defaults to '6m'): lag applied to the
      sorting variable before portfolio assignment (e.g., '6m' = a
      six-month lag).
    - 'rebalancing' (defaults to 'monthly'): how frequently portfolios
      are reformed; 'monthly' or 'annual'.
    - 'n_portfolios_main' (defaults to 10): number of quantile groups
      (e.g., 10 for decile portfolios).
    - 'sorting_method' (defaults to 'univariate'): whether portfolios
      are formed on a single sort ('univariate') or a sequential
      double sort ('sequential').
    - 'n_portfolios_secondary' (defaults to None): number of groups
      for the secondary sort variable. Required when 'sorting_method'
      is not 'univariate'.
    - 'breakpoints_exchanges' (defaults to 'NYSE'): exchange(s) used
      to compute breakpoints; 'NYSE' uses only NYSE-listed stocks to
      define quantile cutoffs (the conventional Fama-French approach).
    - 'breakpoints_min_size_threshold' (defaults to None): minimum
      market-cap threshold (in USD) applied when computing
      breakpoints. None means no minimum-size screen is applied.
    - 'weighting_scheme' (defaults to 'VW'): return weighting within
      portfolios; 'VW' for value-weighted or 'EW' for equal-weighted.

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
    pd.DataFrame
        A data frame with the downloaded data. For
        'high_frequency_sp500', contains 5-second aggregated order-book
        snapshots filtered to the requested date range. For
        'factor_library', contains portfolio return data joined with
        the full grid metadata for the matched portfolio IDs.

    Raises
    ------
    ValueError
        If 'dataset' is None, unsupported, or if invalid filter names
        are passed for the factor library.

    Examples
    --------
    >>> from tidyfinance.data_download import _download_data_huggingface
    >>> _download_data_huggingface(
    ...     'high_frequency_sp500', '2007-07-26', '2007-07-27'
    ... )
    >>> _download_data_huggingface(
    ...     'factor_library',
    ...     sorting_variable='52w',
    ...     rebalancing='annual',
    ... )
    >>> _download_data_huggingface(
    ...     'factor_library', sorting_variable='ag', fill_all=True
    ... )
    >>> _download_data_huggingface(
    ...     'factor_library',
    ...     sorting_variable='me',
    ...     start_date='2000-01-01',
    ...     end_date='2020-12-31',
    ... )
    >>> _download_data_huggingface('factor_library', ids=[1, 2, 3])
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

        date_pattern = re.compile(r"date=(\d{4}-\d{2}-\d{2})")
        available_files = _get_available_huggingface_files(
            organization, dataset_name
        )
        available_files["date"] = pd.to_datetime(
            available_files["path"].str.extract(date_pattern, expand=False),
            format="%Y-%m-%d",
        ).dt.date

        requested_dates = set(
            pd.date_range(
                start=str(start_date), end=str(end_date), freq="D"
            ).date
        )
        files_to_download = available_files.loc[
            available_files["date"].isin(requested_dates)
        ]

        if files_to_download.empty:
            return pd.DataFrame()

        return pd.concat(
            [
                pd.read_parquet(
                    f"https://huggingface.co/datasets/{organization}"
                    f"/{dataset_name}/resolve/main/{path}"
                )
                for path in files_to_download["path"]
            ],
            ignore_index=True,
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
