"""Open-source data downloads for tidyfinance."""

import io
import re
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from curl_cffi import requests

from ._internal import (
    _get_random_user_agent,
    _transfrom_to_snake_case,
    _validate_dates,
)
from .supported_datasets import (
    _check_supported_dataset_ff,
    _determine_frequency_ff,
    _is_breakpoints_ff,
    _is_legacy_type_ff,
    _is_legacy_type_q,
    _parse_type_to_domain_dataset,
)
from .utilities import list_supported_indexes

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
    ```python
    from tidyfinance import download_data_factors_ff
    download_data_factors_ff(
        'Fama/French 3 Factors', '2000-01-01', '2020-12-31'
    )
    download_data_factors_ff(
        '10 Industry Portfolios', '2000-01-01', '2020-12-31'
    )
    ```
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
                    lambda x: (
                        x.replace([-99.99, -999], np.nan)
                        if x.name != "date"
                        else x
                    )
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
    ```python
    from tidyfinance import download_data_factors_q
    download_data_factors_q(
        'q5_factors_daily_2024', '2020-01-01', '2020-12-31'
    )
    download_data_factors_q('q5_factors_annual_2024')
    ```
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
    ```python
    from tidyfinance import download_data_macro_predictors
    download_data_macro_predictors('monthly')
    download_data_macro_predictors(
        'quarterly', '2000-01-01', '2020-12-31'
    )
    ```
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

    return raw_data.reset_index(drop=True)


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
    ```python
    from tidyfinance import download_data_constituents
    download_data_constituents('DAX')
    ```
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
    ```python
    from tidyfinance import download_data_fred
    download_data_fred('CPIAUCNS')
    download_data_fred(['GDP', 'CPIAUCNS'], '2010-01-01', '2010-12-31')
    ```
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
    ```python
    from tidyfinance import download_data_stock_prices
    download_data_stock_prices(['AAPL', 'MSFT'])
    download_data_stock_prices('GOOGL', '2021-01-01', '2022-01-01')
    ```
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
    to snake_case, aligning the date to the beginning of the month,
    scaling the percentage long-short returns to numeric values, and
    optionally filters the data based on a provided date range.

    The dataset contains monthly long-short returns of the predictor
    portfolios. Every column other than ``date`` is a return expressed
    in percent, so all of them are divided by 100 to convert them into
    plain numeric (decimal) returns.

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
        are converted to snake_case, the ``date`` column is aligned to
        the beginning of the month, all predictor columns (long-short
        returns in percent) are divided by 100 to obtain plain numeric
        (decimal) returns, and the data is filtered by the specified
        date range if 'start_date' and 'end_date' are provided.

    Examples
    --------
    ```python
    from tidyfinance import download_data_osap
    osap = download_data_osap(
        start_date='2020-01-01', end_date='2020-06-30'
    )
    ```
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
        raw_data["date"] = (
            pd.to_datetime(raw_data["date"], errors="coerce")
            .dt.to_period("M")
            .dt.start_time
        )

    raw_data.columns = [
        _transfrom_to_snake_case(col) for col in raw_data.columns
    ]

    # All columns except the date are long-short returns in percent, so
    # scale them to plain numeric (decimal) returns.
    return_columns = [col for col in raw_data.columns if col != "date"]
    raw_data[return_columns] = raw_data[return_columns] / 100

    if start_date and end_date:
        raw_data = raw_data.query("@start_date <= date <= @end_date")

    return raw_data.reset_index(drop=True)


def _download_data_pastor_stambaugh(
    start_date: str = None,
    end_date: str = None,
    url: str = (
        "https://faculty.chicagobooth.edu/-/media/faculty/lubos-pastor/"
        "data/liq_data_1962_2025.txt"
    ),
) -> pd.DataFrame:
    """
    Download and process Pastor-Stambaugh liquidity factors.

    Downloads and processes the liquidity factor data of Pastor and
    Stambaugh (2003) from
    `Pastor's data library
    <https://faculty.chicagobooth.edu/lubos-pastor/data>`_. The source
    is a whitespace-delimited text file whose header lines start with
    a percent sign. The function reads the three liquidity series,
    aligns the monthly date to the beginning of the month, and
    optionally filters the data based on a provided date range.

    The series are already expressed as plain numeric (decimal) values
    in the source data, so no rescaling is applied. The traded
    liquidity factor is only available from 1968 onward; earlier
    observations are coded as -99 in the source file and are returned
    as NaN.

    Parameters
    ----------
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    url : str, optional
        The URL of the liquidity data file. Because the file name
        embeds the last year of data, the default points to the most
        recent file known at release time; override it when a newer
        file becomes available.

    Returns
    -------
    pd.DataFrame
        A data frame with the columns 'date' (aligned to the
        beginning of the month), 'agg_liq' (levels of aggregate
        liquidity), 'innov_liq' (innovations in aggregate liquidity,
        the non-traded liquidity factor), and 'traded_liq' (the traded
        liquidity factor LIQ_V), filtered by the specified date range
        if 'start_date' and 'end_date' are provided.

    References
    ----------
    Pastor, L., and Stambaugh, R. F. (2003). Liquidity risk and
    expected stock returns. Journal of Political Economy, 111(3),
    642-685. https://doi.org/10.1086/374184

    Examples
    --------
    ```python
    from tidyfinance import download_data_pastor_stambaugh
    pastor_stambaugh = download_data_pastor_stambaugh(
        start_date='2020-01-01', end_date='2020-12-31'
    )
    ```
    """
    start_date, end_date = _validate_dates(start_date, end_date)

    try:
        raw_data = pd.read_csv(
            url,
            sep=r"\s+",
            comment="%",
            header=None,
            names=["month", "agg_liq", "innov_liq", "traded_liq"],
        )
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

    # The traded factor is coded -99 before it becomes available (1968).
    processed_data = raw_data.assign(
        date=lambda x: pd.to_datetime(x["month"].astype(str), format="%Y%m")
    )
    liquidity_columns = ["agg_liq", "innov_liq", "traded_liq"]
    processed_data[liquidity_columns] = processed_data[
        liquidity_columns
    ].replace(-99, np.nan)
    processed_data = processed_data[["date", *liquidity_columns]]

    if start_date and end_date:
        processed_data = processed_data.query(
            "@start_date <= date <= @end_date"
        )

    return processed_data.reset_index(drop=True)


def _download_data_stambaugh_yuan(
    dataset: str = "monthly",
    start_date: str = None,
    end_date: str = None,
    url: str = "https://finance.wharton.upenn.edu/~stambaug/",
) -> pd.DataFrame:
    """
    Download and process Stambaugh-Yuan mispricing factors.

    Downloads and processes the mispricing factor data of Stambaugh
    and Yuan (2017) from
    `Stambaugh's data library
    <https://finance.wharton.upenn.edu/~stambaug/>`_. The four-factor
    model (M4) combines the market and size factors with two
    mispricing factors, 'mgmt' (management) and 'perf' (performance).
    The function downloads the requested frequency, aligns the date,
    renames the columns to the package conventions, and optionally
    filters the data based on a provided date range.

    Returns are already expressed as plain numeric (decimal) values in
    the source data, so no rescaling is applied. The source files
    currently end in December 2016; a requested date range that lies
    entirely outside the available data emits a warning and returns
    an empty data frame.

    Parameters
    ----------
    dataset : str, default 'monthly'
        The data frequency to download, either 'monthly' or 'daily'.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.
    url : str, optional
        The base URL from which to download the dataset files. The
        file name ('M4.csv' or 'M4d.csv') is appended based on
        'dataset'.

    Returns
    -------
    pd.DataFrame
        A data frame with the columns 'date' (aligned to the
        beginning of the month for monthly data), 'mkt_excess' (the
        market excess return), 'smb' (size), 'mgmt' (the management
        mispricing factor), 'perf' (the performance mispricing
        factor), and 'risk_free' (the risk-free rate). All returns are
        plain numeric (decimal) values, filtered by the specified
        date range if 'start_date' and 'end_date' are provided.

    References
    ----------
    Stambaugh, R. F., and Yuan, Y. (2017). Mispricing factors. Review
    of Financial Studies, 30(4), 1270-1315.
    https://doi.org/10.1093/rfs/hhw107

    Examples
    --------
    ```python
    from tidyfinance import download_data_stambaugh_yuan
    download_data_stambaugh_yuan(
        start_date='2015-01-01', end_date='2016-12-31'
    )
    download_data_stambaugh_yuan(
        dataset='daily', start_date='2016-01-01', end_date='2016-12-31'
    )
    ```
    """
    if dataset not in ("monthly", "daily"):
        raise ValueError("'dataset' must be 'monthly' or 'daily'.")

    start_date, end_date = _validate_dates(start_date, end_date)

    file = "M4d.csv" if dataset == "daily" else "M4.csv"

    try:
        raw_data = pd.read_csv(f"{url}{file}")
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

    # The monthly file keys rows by YYYYMM, the daily file by YYYYMMDD.
    if dataset == "daily":
        processed_data = raw_data.assign(
            date=lambda x: pd.to_datetime(
                x["DATE"].astype(str), format="%Y%m%d"
            )
        )
    else:
        processed_data = raw_data.assign(
            date=lambda x: pd.to_datetime(
                x["YYYYMM"].astype(str), format="%Y%m"
            )
        )

    processed_data = processed_data.rename(
        columns={
            "MKTRF": "mkt_excess",
            "SMB": "smb",
            "MGMT": "mgmt",
            "PERF": "perf",
            "RF": "risk_free",
        }
    )[["date", "mkt_excess", "smb", "mgmt", "perf", "risk_free"]]

    if start_date and end_date:
        filtered_data = processed_data.query("@start_date <= date <= @end_date")

        if filtered_data.empty:
            available_start = processed_data["date"].min().date()
            available_end = processed_data["date"].max().date()
            warnings.warn(
                "The requested date range lies outside the available "
                "Stambaugh-Yuan data. Available data range: "
                f"{available_start} to {available_end}.",
                UserWarning,
                stacklevel=2,
            )

        processed_data = filtered_data

    return processed_data.reset_index(drop=True)


_JKP_BASE_URL = "https://jkpfactors-data.s3.amazonaws.com"


def _fetch_jkp_availability(max_tries: int = 3) -> dict:
    """
    Fetch the Global Factor Data availability manifest.

    Downloads and parses the JSON availability manifest that lists the
    regions, factors, and frequency restrictions offered by
    `Global Factor Data <https://jkpfactors.com/data>`_.

    Parameters
    ----------
    max_tries : int, default 3
        Number of download attempts before giving up.

    Returns
    -------
    dict
        The parsed manifest, including the 'factors', 'portfolios',
        and 'industry' keys (each mapping region codes to the
        available selectors) and 'factors_monthly_only' (region codes
        mapped to factors available at monthly frequency only).

    Raises
    ------
    Exception
        If the manifest could not be downloaded after 'max_tries'
        attempts.
    """
    url = f"{_JKP_BASE_URL}/public/availability.json"
    headers = {"User-Agent": _get_random_user_agent()}
    last_error = None
    for _ in range(max_tries):
        try:
            response = requests.get(
                url, headers=headers, timeout=60, impersonate="chrome120"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
    raise last_error


def _validate_jkp_selection(
    availability: dict,
    dataset: str,
    region: str,
    selector: str,
    frequency: str,
) -> None:
    """
    Validate a Global Factor Data selection against the manifest.

    Parameters
    ----------
    availability : dict
        The manifest returned by '_fetch_jkp_availability'.
    dataset : str
        One of 'factors', 'portfolios', or 'industry'.
    region : str
        The region or country code to validate.
    selector : str
        The factor code, theme, or industry classification to
        validate.
    frequency : str
        Either 'monthly' or 'daily'.

    Raises
    ------
    ValueError
        If 'region' is not available for 'dataset', if 'selector' is
        not available for 'region', or if 'selector' is only
        available at monthly frequency but 'frequency' is 'daily'.
    """
    regions = list(availability.get(dataset, {}).keys())
    if region not in regions:
        raise ValueError(
            f"Unsupported region: {region!r} for dataset {dataset!r}. "
            "Use list_supported_jkp_factors(dataset="
            f"{dataset!r}) to see valid regions."
        )

    available = availability[dataset][region]
    if selector not in available:
        raise ValueError(
            f"Unsupported selection {selector!r} for region {region!r} "
            f"in dataset {dataset!r}. Use list_supported_jkp_factors("
            f"{region!r}, {dataset!r}) to see valid values."
        )

    if dataset == "factors":
        monthly_only = availability.get("factors_monthly_only", {}).get(
            region, []
        )
        if frequency == "daily" and selector in monthly_only:
            raise ValueError(
                f"{selector!r} is only available at monthly frequency "
                f"for region {region!r}. Set frequency='monthly'."
            )


def _build_jkp_url(
    dataset: str, region: str, selector: str, frequency: str, weighting: str
) -> str:
    """
    Build the S3 download URL for a Global Factor Data archive.

    Parameters
    ----------
    dataset : str
        One of 'factors', 'portfolios', or 'industry'.
    region : str
        The region or country code.
    selector : str
        The factor code, theme, or industry classification.
    frequency : str
        Either 'monthly' or 'daily'. Ignored for 'industry', which is
        only published at monthly frequency.
    weighting : str
        The portfolio weighting scheme.

    Returns
    -------
    str
        The URL of the zipped CSV archive.
    """

    def b(x: str) -> str:
        # The S3 object keys wrap each selector in literal square
        # brackets, which are URL-encoded as %5B and %5D.
        return f"%5B{x}%5D"

    if dataset == "factors":
        return (
            f"{_JKP_BASE_URL}/public/"
            f"{b(region)}_{b(selector)}_{b(frequency)}_{b(weighting)}.zip"
        )
    elif dataset == "portfolios":
        return (
            f"{_JKP_BASE_URL}/public/portfolios/"
            f"{b(region)}_{b(selector)}_{b(frequency)}_{b(weighting)}.zip"
        )
    else:
        # The industry dataset is only published at monthly frequency.
        return (
            f"{_JKP_BASE_URL}/public/industry/"
            f"{b(region)}_{b(selector)}_{b('monthly')}_{b(weighting)}.zip"
        )


def _build_jkp_reference_url(dataset: str, frequency: str) -> str:
    """
    Build the S3 download URL for a Global Factor Data reference file.

    Parameters
    ----------
    dataset : str
        Either 'nyse_cutoffs' or 'return_cutoffs'.
    frequency : str
        Either 'monthly' or 'daily'. Selects between the monthly and
        daily 'return_cutoffs' file; ignored for 'nyse_cutoffs'.

    Returns
    -------
    str
        The URL of the plain CSV reference file.
    """
    base_url = f"{_JKP_BASE_URL}/public/other"

    if dataset == "nyse_cutoffs":
        return f"{base_url}/nyse_cutoffs.csv"
    elif frequency == "daily":
        return f"{base_url}/return_cutoffs_daily.csv"
    else:
        return f"{base_url}/return_cutoffs.csv"


def _download_jkp_file(url: str, max_tries: int = 5) -> pd.DataFrame:
    """
    Download and read a zipped Global Factor Data CSV file.

    Parameters
    ----------
    url : str
        URL of the zipped CSV archive.
    max_tries : int, default 5
        Number of download attempts before giving up.

    Returns
    -------
    pd.DataFrame
        The first CSV file found in the archive.

    Raises
    ------
    Exception
        If the archive could not be downloaded after 'max_tries'
        attempts.
    ValueError
        If no CSV file is found in the downloaded archive.
    """
    headers = {"User-Agent": _get_random_user_agent()}
    last_error = None
    response = None
    for _ in range(max_tries):
        try:
            response = requests.get(
                url, headers=headers, timeout=180, impersonate="chrome120"
            )
            response.raise_for_status()
            break
        except Exception as e:
            last_error = e
            response = None
    if response is None:
        raise last_error

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found in the downloaded archive.")
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f, na_values=["na"])


def _download_jkp_csv(url: str, max_tries: int = 5) -> pd.DataFrame:
    """
    Download and read a plain Global Factor Data CSV file.

    Parameters
    ----------
    url : str
        URL of the plain CSV file.
    max_tries : int, default 5
        Number of download attempts before giving up.

    Returns
    -------
    pd.DataFrame
        The parsed CSV file.

    Raises
    ------
    Exception
        If the file could not be downloaded after 'max_tries' attempts.
    """
    headers = {"User-Agent": _get_random_user_agent()}
    last_error = None
    for _ in range(max_tries):
        try:
            response = requests.get(
                url, headers=headers, timeout=180, impersonate="chrome120"
            )
            response.raise_for_status()
            return pd.read_csv(io.StringIO(response.text), na_values=["na"])
        except Exception as e:
            last_error = e
    raise last_error


def _process_jkp_data(
    data: pd.DataFrame,
    date_col: str,
    frequency: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Normalize dates and filter a Global Factor Data table.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with a date-like column named 'date_col'.
    date_col : str
        Name of the date column in 'data'. Renamed to 'date' if it
        differs.
    frequency : str
        Either 'monthly' or 'daily'. Monthly dates are aligned to the
        beginning of the month.
    start_date, end_date : Timestamp or None
        Filter bounds. No filtering is applied if either is None.

    Returns
    -------
    pd.DataFrame
        The processed data, filtered by the specified date range if
        both 'start_date' and 'end_date' are provided.
    """
    data = data.copy()
    if date_col != "date":
        data = data.rename(columns={date_col: "date"})

    data["date"] = pd.to_datetime(data["date"])

    if frequency == "monthly":
        data["date"] = data["date"].dt.to_period("M").dt.start_time

    if start_date is not None and end_date is not None:
        data = data.query("@start_date <= date <= @end_date")

    return data.reset_index(drop=True)


def _download_data_jkp(
    dataset: str = "factors",
    region: str = "usa",
    factors: str = "all_factors",
    classification: str = "gics",
    frequency: str = "monthly",
    weighting: str = "vw_cap",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download and process Global Factor Data.

    Downloads and processes data from
    `Global Factor Data <https://jkpfactors.com/data>`_, the public data
    library accompanying Jensen, Kelly, and Pedersen (2023). The data
    are stored as zipped CSV files (and a few plain CSV reference
    files) in a public AWS S3 bucket. For the factor, portfolio, and
    industry products the function validates the requested selection
    against the library's live availability manifest, then downloads
    the matching archive, unzips it, aligns monthly dates to the
    beginning of the month, and optionally filters by a date range.

    Returns are already expressed as plain numeric (decimal) values in
    the source data, so no rescaling is applied. The data are licensed
    under CC BY-NC 4.0 (non-commercial use).

    Parameters
    ----------
    dataset : str, default 'factors'
        The Global Factor Data product to download, one of:
        'factors' (characteristic-managed portfolio returns),
        'portfolios' (the underlying low/middle/high portfolios that
        make up each long-short factor), 'industry' (industry
        returns), 'nyse_cutoffs' (NYSE size breakpoints), or
        'return_cutoffs' (return winsorization cutoffs).
    region : str, default 'usa'
        The region or country to download, using the codes from the
        availability manifest (e.g., 'usa', 'world', 'developed',
        'emerging', or an ISO-3 country code such as 'jpn'). Ignored
        for the reference datasets 'nyse_cutoffs' and
        'return_cutoffs'. Call 'list_supported_jkp_factors' to see the
        available regions.
    factors : str, default 'all_factors'
        The factor content for the 'factors' and 'portfolios'
        datasets. For 'factors': 'mkt' (the market factor),
        'all_factors' (all factors), 'all_themes' (all themes), a
        single theme (e.g., 'value', 'momentum'), or a single factor
        code (e.g., 'be_me', 'ret_12_1'). For 'portfolios': a single
        factor code. Call 'list_supported_jkp_factors(region, dataset)'
        to see the values available for a region.
    classification : str, default 'gics'
        The industry classification for the 'industry' dataset,
        either 'gics' or 'ff49' (Fama-French 49 industries).
    frequency : str, default 'monthly'
        The data frequency, either 'monthly' or 'daily'. The
        'industry' dataset is only available at monthly frequency.
        For 'return_cutoffs', the frequency selects the monthly or
        daily cutoff file.
    weighting : str, default 'vw_cap'
        The portfolio weighting scheme: 'vw_cap' (capped
        value-weighted), 'vw' (value-weighted), or 'ew'
        (equal-weighted). Ignored for the reference datasets.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        is returned.

    Returns
    -------
    pd.DataFrame
        A data frame with the processed data. The 'date' column is
        aligned to the beginning of the month for monthly data, and
        all returns are plain numeric (decimal) values. The remaining
        columns depend on 'dataset': the 'factors' data carry
        'location', 'name', 'freq', 'weighting', 'direction',
        'n_stocks', 'n_stocks_min', and 'ret'; the 'portfolios' data
        add a 'pf' portfolio identifier; the 'industry' data carry the
        classification code alongside 'ret'; and the reference
        datasets carry breakpoint or cutoff columns.

    References
    ----------
    Jensen, T. I., Kelly, B. T., and Pedersen, L. H. (2023). Is there a
    replication crisis in finance? Journal of Finance, 78(5),
    2465-2518. https://doi.org/10.1111/jofi.13249

    Examples
    --------
    ```python
    from tidyfinance import download_data_jkp
    download_data_jkp(
        region='usa', factors='mkt',
        start_date='2000-01-01', end_date='2020-12-31',
    )
    download_data_jkp(
        dataset='portfolios', region='usa', factors='be_me',
        start_date='2000-01-01', end_date='2020-12-31',
    )
    download_data_jkp(dataset='industry', region='usa', classification='gics')
    download_data_jkp(dataset='nyse_cutoffs')
    ```
    """
    supported_datasets = (
        "factors",
        "portfolios",
        "industry",
        "nyse_cutoffs",
        "return_cutoffs",
    )
    if dataset not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. "
            f"Supported datasets: {', '.join(supported_datasets)}."
        )

    if frequency not in ("monthly", "daily"):
        raise ValueError("'frequency' must be 'monthly' or 'daily'.")

    start_date, end_date = _validate_dates(start_date, end_date)

    # Reference datasets are plain CSV files that need neither manifest
    # validation, weighting, nor a region selection.
    if dataset in ("nyse_cutoffs", "return_cutoffs"):
        url = _build_jkp_reference_url(dataset, frequency)
        try:
            raw_data = _download_jkp_csv(url)
        except Exception:
            raw_data = pd.DataFrame()

        if raw_data.empty:
            warnings.warn(
                "Returning an empty dataset due to a download or "
                "parsing failure.",
                UserWarning,
                stacklevel=2,
            )
            return raw_data

        return _process_jkp_data(
            raw_data,
            date_col="eom",
            frequency="monthly",
            start_date=start_date,
            end_date=end_date,
        )

    if weighting not in ("vw_cap", "vw", "ew"):
        raise ValueError("'weighting' must be one of 'vw_cap', 'vw', 'ew'.")

    if dataset == "industry" and frequency == "daily":
        raise ValueError(
            "The 'industry' dataset is only available at monthly "
            "frequency. Set frequency='monthly'."
        )

    try:
        availability = _fetch_jkp_availability()
    except Exception:
        warnings.warn(
            "Returning an empty dataset due to download failure.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    selector = classification if dataset == "industry" else factors
    _validate_jkp_selection(availability, dataset, region, selector, frequency)

    url = _build_jkp_url(dataset, region, selector, frequency, weighting)

    try:
        raw_data = _download_jkp_file(url)
    except Exception:
        raw_data = pd.DataFrame()

    if raw_data.empty:
        warnings.warn(
            "Returning an empty dataset due to a download or parsing failure.",
            UserWarning,
            stacklevel=2,
        )
        return raw_data

    data_frequency = "monthly" if dataset == "industry" else frequency
    processed_data = _process_jkp_data(
        raw_data,
        date_col="date",
        frequency=data_frequency,
        start_date=start_date,
        end_date=end_date,
    )

    if dataset == "portfolios" and "pf" in processed_data.columns:
        processed_data["pf"] = processed_data["pf"].astype(int)

    return processed_data
