"""Main module for tidyfinance package."""

import os
import yaml
import pandas as pd
import numpy as np
import requests
import webbrowser
import pandas_datareader as pdr
from sqlalchemy import create_engine


def add_lag_columns(
    data: pd.DataFrame,
    cols: list[str],
    by: str | None = None,
    lag: int = 0,
    max_lag: int | None = None,
    drop_na: bool = False,
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns to a Pandas DataFrame.

    Parameters
    ----------
        data (pd.DataFrame): The input DataFrame.
        cols (list[str]): List of column names to lag.
        by (str | None): Optional column to group by. Default is None.
        lag (int): Number of periods to lag. Must be non-negative.
        max_lag (int | None): Maximum lag period. Defaults to `lag` if None.
        drop_na (bool): Whether to drop rows with missing values in lagged
        columns. Default is False.
        date_col (str): The name of the date column. Default is "date".

    Returns
    -------
        pd.DataFrame: DataFrame with lagged columns appended.
    """
    if lag < 0 or (max_lag is not None and max_lag < lag):
        raise ValueError("`lag` must be non-negative, "
                         "and `max_lag` must be greater than or "
                         "equal to `lag`.")

    if max_lag is None:
        max_lag = lag

    # Ensure the date column is available
    if date_col not in data.columns:
        raise ValueError(f"Date column `{date_col}` not found in DataFrame.")

    result = data.copy()
    for col in cols:
        if col not in data.columns:
            raise ValueError(f"Column `{col}` not found in the DataFrame.")

        for index_lag in range(lag, max_lag + 1):
            lag_col_name = f"{col}_lag_{index_lag}"

            if by:
                result[lag_col_name] = result.groupby(by)[col].shift(index_lag)
            else:
                result[lag_col_name] = result[col].shift(index_lag)

            if drop_na:
                result = result.dropna(subset=[lag_col_name])

    return result


def assign_portfolio(data, sorting_variable, breakpoint_options=None, breakpoint_function=None, data_options=None):
    """Assign data points to portfolios based on a sorting variable.

    Parameters:
        data (pd.DataFrame): Data for portfolio assignment.
        sorting_variable (str): Column name used for sorting.
        breakpoint_options (dict, optional): Arguments for the breakpoint function.
        breakpoint_function (callable, optional): Function to compute breakpoints.
        data_options (dict, optional): Additional data processing options.

    Returns:
        pd.Series: Portfolio assignments for each row.
    """
    #     breakpoints = (data
    #                    .query("exchange == 'NYSE'")
    #                    .get(sorting_variable)
    #                    .quantile(percentiles, interpolation="linear")
    #                    )
    # breakpoints.iloc[0] = -np.Inf
    # breakpoints.iloc[breakpoints.size-1] = np.Inf
    
    # assigned_portfolios = pd.cut(
    #   data[sorting_variable],
    #   bins=breakpoints,
    #   labels=pd.Series(range(1, breakpoints.size)),
    #   include_lowest=True,
    #   right=False
    # )
    pass
    # return assigned_portfolios


def breakpoint_options(n_portfolios=None, percentiles=None, breakpoint_exchanges=None, smooth_bunching=False, **kwargs):
    """Create structured options for defining breakpoints.

    Parameters:
        n_portfolios (int, optional): Number of portfolios to create.
        percentiles (list, optional): Percentile thresholds for breakpoints.
        breakpoint_exchanges (list, optional): Exchanges for which breakpoints apply.
        smooth_bunching (bool): Whether to apply smooth bunching.
        kwargs: Additional optional parameters.

    Returns:
        dict: Breakpoint options.
    """
    pass

def compute_breakpoints(
    data: pd.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict,
    data_options: dict = None
) -> np.ndarray:
    """Compute breakpoints based on a sorting variable.

    Parameters:
        data (pd.DataFrame): Data for breakpoint computation.
        sorting_variable (str): Column name for sorting.
        breakpoint_options (dict): Options for breakpoints.
        data_options (dict, optional): Additional data processing options.

    Returns:
        list: Computed breakpoints.
    """
    pass


def compute_long_short_returns(data, direction="top_minus_bottom", data_options=None):
    """Calculate long-short returns based on portfolio returns.

    Parameters:
        data (pd.DataFrame): Data containing portfolio returns.
        direction (str): Calculation direction ('top_minus_bottom' or 'bottom_minus_top').
        data_options (dict, optional): Additional data processing options.

    Returns:
        pd.DataFrame: DataFrame with computed long-short returns.
    """
    pass


def compute_portfolio_returns(sorting_data, sorting_variables, sorting_method, rebalancing_month=None, breakpoint_options_main=None, breakpoint_options_secondary=None, breakpoint_function_main=None, breakpoint_function_secondary=None, min_portfolio_size=0, data_options=None):
    """Compute portfolio returns based on sorting variables and methods.

    Parameters:
        sorting_data (pd.DataFrame): Data for portfolio assignment and return computation.
        sorting_variables (list): List of variables for sorting.
        sorting_method (str): Sorting method ('univariate' or 'bivariate').
        rebalancing_month (int, optional): Month for annual rebalancing.
        breakpoint_options_main (dict, optional): Options for main sorting variable.
        breakpoint_options_secondary (dict, optional): Options for secondary sorting variable.
        breakpoint_function_main (callable, optional): Function for main sorting.
        breakpoint_function_secondary (callable, optional): Function for secondary sorting.
        min_portfolio_size (int): Minimum portfolio size.
        data_options (dict, optional): Additional data processing options.

    Returns:
        pd.DataFrame: DataFrame with computed portfolio returns.
    """
    pass


def create_summary_statistics(data, *args, by=None, detail=False, drop_na=False):
    """Create summary statistics for specified variables.

    Parameters:
        data (pd.DataFrame): Data containing variables to summarize.
        *args: Variables to summarize.
        by (str, optional): Grouping variable.
        detail (bool): Whether to include detailed statistics.
        drop_na (bool): Whether to drop missing values.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    pass


def create_wrds_dummy_database(
    path: str,
    url: str = ("https://github.com/tidy-finance/website/raw/main/blog/"
                "tidy-finance-dummy-data/data/tidy_finance.sqlite")
) -> None:
    """
    Download the WRDS dummy database from the Tidy Finance GitHub repository.

    It saves it to the specified path. If the file already exists,
    the user is prompted before it is replaced.

    Parameters
    ----------
        path (str): The file path where the SQLite database should be saved.
        url (str, optional): The URL where the SQLite database is stored.

    Returns
    -------
        None: Side effect - downloads a file to the specified path.
    """
    if not path:
        raise ValueError("Please provide a file path for the SQLite database. "
                         "We recommend 'data/tidy_finance.sqlite'.")

    if os.path.exists(path):
        response = input("The database file already exists at this path. Do "
                         "you want to replace it? (Y/n): ")
        if response.strip().lower() != "y":
            print("Operation aborted by the user.")
            return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                print(chunk)
                file.write(chunk)
        print(f"Downloaded WRDS dummy database to {path}.")
    except requests.RequestException as e:
        print(f"Error downloading the WRDS dummy database: {e}")


def data_options(id="permno", date="date", exchange="exchange", mktcap_lag="mktcap_lag", ret_excess="ret_excess", portfolio="portfolio", **kwargs):
    """Create a dictionary of data options for analysis.

    Parameters:
        id (str): Identifier variable name.
        date (str): Date variable name.
        exchange (str): Exchange variable name.
        mktcap_lag (str): Market capitalization lag variable.
        ret_excess (str): Excess return variable.
        portfolio (str): Portfolio variable.
        kwargs: Additional options.

    Returns:
        dict: Data options.
    """
    pass


def disconnection_connection(con):
    """Disconnect a database connection.

    Parameters:
        con (Any): Database connection object.

    Returns:
        bool: True if disconnection was successful, False otherwise.
    """
    pass


def download_data(
    data_set: str,
    start_date: str = None,
    end_date: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Download and process data based on the specified type.

    Parameters
    ----------
    data_set : str
        The type of dataset to download, indicating either factor data or
        macroeconomic predictors  (e.g., Fama-French factors, Global Q factors,
                                   or macro predictors).
    start_date : str, optional
        The start date for filtering the data, in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date for filtering the data, in "YYYY-MM-DD" format.
    **kwargs : dict
        Additional arguments passed to specific download functions depending
        on the `type`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed data, including dates and relevant financial
        metrics, filtered by the specified date range.
    """
    if "factors" in data_set:
        processed_data = download_data_factors(
            data_set, start_date, end_date, **kwargs
            )
    elif "macro_predictors" in data_set:
        processed_data = download_data_macro_predictors(
            data_set, start_date, end_date, **kwargs
            )
    elif "wrds" in data_set:
        processed_data = download_data_wrds(
            data_set, start_date, end_date, **kwargs
            )
    elif "constituents" in data_set:
        processed_data = download_data_constituents(**kwargs)
    elif "fred" in data_set:
        processed_data = download_data_fred(
            start_date=start_date, end_date=end_date, **kwargs
            )
    elif "stock_prices" in data_set:
        processed_data = download_data_stock_prices(
            start_date=start_date, end_date=end_date, **kwargs
            )
    elif "osap" in data_set:
        processed_data = download_data_osap(start_date, end_date, **kwargs)
    else:
        raise ValueError("Unsupported data type.")
    return processed_data


def download_data_factors(
    data_set: str,
    start_date: str = None,
    end_date: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Download and process factor data for the specified type and date range.

    Parameters
    ----------
    data_set : str
        The type of dataset to download, indicating factor model and frequency.
    start_date : str, optional
        The start date for filtering the data, in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date for filtering the data, in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed factor data, including dates,
        risk-free rates, market excess returns, and other factors,
        filtered by the specified date range.
    """
    if "factors_ff" in data_set:
        return download_data_factors_ff(data_set, start_date, end_date)
    elif "factors_q" in data_set:
        return download_data_factors_q(data_set, start_date, end_date)
    else:
        raise ValueError("Unsupported factor data type.")


def download_data_factors_ff(
    data_set: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """Download and process Fama-French factor data."""
    start_date, end_date = _validate_dates(start_date, end_date)
    all_data_sets = pdr.famafrench.get_available_datasets()
    if data_set in all_data_sets:
        try:
            raw_data = (pdr.famafrench.FamaFrenchReader(
                data_set, start=start_date, end=end_date).read()[0]
                .div(100)
                .reset_index()
                .rename(columns=lambda x:
                        x.lower()
                        .replace("-rf", "_excess")
                        .replace("rf", "risk_free")
                        )
                .assign(date=lambda x: _return_datetime(x['date']))
                .apply(lambda x: x.replace([-99.99, -999], pd.NA)
                       if x.name != 'date' else x
                       )
                )
            raw_data = raw_data[
                ["date"] + [col for col in raw_data.columns if col != "date"]
                ].reset_index(drop=True)
            return raw_data
        except ValueError:
            raise ValueError("Unsupported factor data type.")
    else:
        raise ValueError("Returning an empty data set due to download failure")
        print(f"{data_set} is not in list of available data sets. "
              " Returns empty DataFrame. Choose a dataset from:")
        print("")
        print(all_data_sets)
        return pd.DataFrame()


def download_data_factors_q(
    data_set: str,
    start_date: str = None,
    end_date: str = None,
    url: str = "https://global-q.org/uploads/1/2/2/6/122679606/"
) -> pd.DataFrame:
    """
    Download and process Global Q factor data.

    Parameters
    ----------
    data_set : str
        The type of dataset to download (e.g., "factors_q5_daily",
                                         "factors_q5_monthly").
    start_date : str, optional
        The start date for filtering the data, in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date for filtering the data, in "YYYY-MM-DD" format.
    url : str, optional
        The base URL from which to download the dataset files.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed factor data, including the date,
        risk-free rate, market excess return, and other factors.
    """
    start_date, end_date = _validate_dates(start_date, end_date)
    ref_year = pd.Timestamp.today().year - 1
    all_data_sets = [f"q5_factors_daily_{ref_year}",
                     f"q5_factors_weekly_{ref_year}",
                     f"q5_factors_weekly_w2w_{ref_year}",
                     f"q5_factors_monthly_{ref_year}",
                     f"q5_factors_quarterly_{ref_year}",
                     f"q5_factors_annual_{ref_year}"
                     ]
    if data_set in all_data_sets:
        raw_data = (pd.read_csv(f"{url}{data_set}.csv")
                    .rename(columns=lambda x: x.lower().replace("r_", ""))
                    .rename(columns={"f": "risk_free", "mkt": "mkt_excess"})
                    )
        if "monthly" in data_set:
            raw_data = (raw_data.assign(date=lambda x: pd.to_datetime(
                x["year"].astype(str) + "-" + x["month"].astype(str)+"-01")
                )
                .drop(columns=["year", "month"])
                )
        if "annual" in data_set:
            raw_data = (raw_data.assign(date=lambda x: pd.to_datetime(
                x["year"].astype(str) + "-01-01")
                )
                .drop(columns=["year"])
                )
        raw_data = (raw_data
                    .assign(date=lambda x: pd.to_datetime(x["date"]))
                    .apply(lambda x: x.div(100) if x.name != "date" else x)
                    )
        if start_date and end_date:
            raw_data = raw_data.query('@start_date <= date <= @end_date')
        raw_data = raw_data[
            ["date"] + [col for col in raw_data.columns if col != "date"]
            ].reset_index(drop=True)
        return raw_data
    else:
        raise ValueError("Returning an empty data set due to download "
                         "failure.")
        print(f"{data_set} might not be in list of available data sets: "
              " Also check the provided URL. Choose a dataset from:")
        print("")
        print(all_data_sets)
        return pd.DataFrame()


def download_data_macro_predictors(
    data_set: str,
    start_date: str = None,
    end_date: str = None,
    sheet_id: str = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG"
) -> pd.DataFrame:
    """
    Download and process macroeconomic predictor data.

    Parameters
    ----------
    data_set : str
        The type of dataset to download ("Monthly", "Quarterly", "Annual")
    start_date : str, optional
        The start date for filtering the data, in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date for filtering the data, in "YYYY-MM-DD" format.
    sheet_id : str, optional
        The Google Sheets ID from which to download the dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed data, including financial metrics, filtered
        by the specified date range.
    """
    start_date, end_date = _validate_dates(start_date, end_date)

    if data_set in ["Monthly", "Quarterly", "Annual"]:
        try:
            macro_sheet_url = ("https://docs.google.com/spreadsheets/d/"
                               f"{sheet_id}/gviz/tq?tqx=out:csv&sheet="
                               f"{data_set}"
                               )
            raw_data = pd.read_csv(macro_sheet_url)
        except Exception:
            print("Expected an empty DataFrame due to download failure.")
            return pd.DataFrame()
    else:
        raise ValueError("Unsupported macro predictor type.")
        return pd.DataFrame()

    if data_set == "Monthly":
        raw_data = (raw_data
                    .assign(date=lambda x: pd.to_datetime(x["yyyymm"],
                                                          format="%Y%m")
                            )
                    .drop(columns=['yyyymm'])
                    )
    if data_set == "Quarterly":
        raw_data = (raw_data
                    .assign(date=lambda x: pd.to_datetime(
                        x["yyyyq"].astype(str).str[:4]
                        + "-" + (x["yyyyq"].astype(str).str[4].astype(int)
                                 * 3 - 2).astype(str)
                        + "-01")
                        )
                    .drop(columns=['yyyyq'])
                    )
    if data_set == "Annual":
        raw_data = (raw_data
                    .assign(date=lambda x:
                            pd.to_datetime(x["yyyy"].astype(str) + "-01-01")
                            )
                    .drop(columns=['yyyy'])
                    )

    raw_data = raw_data.apply(
        lambda x: pd.to_numeric(x.astype(str).str.replace(",", ""),
                                errors='coerce') if x.dtype == "object" else x)
    raw_data = raw_data.assign(
        IndexDiv=lambda df: df["Index"] + df["D12"],
        logret=lambda df: df["IndexDiv"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
            ).diff(),
        rp_div=lambda df: df["logret"].shift(-1) - df["Rfree"],
        log_d12=lambda df: df["D12"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
            ),
        log_e12=lambda df: df["E12"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)),
        dp=lambda df: df["log_d12"] - df["Index"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)),
        dy=lambda df: df["log_d12"] - df["Index"].shift(1).apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
            ),
        ep=lambda df: df["log_e12"] - df["Index"].apply(
            lambda x: np.nan if pd.isna(x) else np.log(x)
            ),
        de=lambda df: df["log_d12"] - df["log_e12"],
        tms=lambda df: df["lty"] - df["tbl"],
        dfy=lambda df: df["BAA"] - df["AAA"]
    )

    raw_data = raw_data[[
        "date", "rp_div", "dp", "dy", "ep", "de", "svar", "b/m", "ntis",
        "tbl", "lty", "ltr", "tms", "dfy", "infl"
        ]]
    raw_data = (raw_data
                .rename(columns={col: col.replace("/", "")
                                 for col in raw_data.columns}
                        )
                .dropna()
                )

    if start_date and end_date:
        raw_data = raw_data.query('@start_date <= date <= @end_date')

    return raw_data


def download_data_fred(
    series: str | list,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download and process data from FRED.

    Parameters
    ----------
    series : str or list
        A list of FRED series IDs to download.
    start_date : str, optional
        The start date for filtering the data, in "YYYY-MM-DD" format.
    end_date : str, optional
        The end date for filtering the data, in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed data, including the date, value,
        and series ID, filtered by the specified date range.
    """
    if isinstance(series, str):
        series = [series]

    start_date, end_date = _validate_dates(start_date, end_date)
    fred_data = []

    for s in series:
        url = f"https://fred.stlouisfed.org/series/{s}/downloaddata/{s}.csv"
        headers = {"User-Agent": get_random_user_agent()}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            raw_data = (pd.read_csv(pd.io.common.StringIO(response.text))
                        .rename(columns=lambda x: x.lower())
                        .assign(date=lambda x: pd.to_datetime(x["date"]),
                                value=lambda x: pd.to_numeric(x["value"],
                                                              errors='coerce'),
                                series=s
                                )
                        )

            fred_data.append(raw_data)
        except requests.RequestException as e:
            print(f"Failed to retrieve data for series {s}: {e}")
            fred_data.append(pd.DataFrame(columns=["date", "value", "series"]))

    fred_data = pd.concat(fred_data, ignore_index=True)

    if start_date and end_date:
        fred_data = fred_data.query('@start_date <= date <= @end_date')

    return fred_data


def download_data_stock_prices(
    symbols: str | list,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    symbols : list
        A list of stock symbols to download data for.
        At least one symbol must be provided.
    start_date : str, optional
        Start date in "YYYY-MM-DD" format. Defaults to "2000-01-01".
    end_date : str, optional
        End date in "YYYY-MM-DD" format. Defaults to today's date.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns: symbol, date, volume, open, low,
        high, close, adjusted_close.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    elif not isinstance(symbols, list) or not all(isinstance(sym, str) for sym in symbols):
        raise ValueError("symbols must be a list of stock symbols (strings).")

    start_date, end_date = _validate_dates(start_date, end_date)

    if start_date is None:
        start_date = pd.Timestamp.today() - pd.DateOffset(years=2)
    if end_date is None:
        end_date = pd.Timestamp.today()

    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).timestamp())

    all_data = []

    for symbol in symbols:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"

        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            raw_data = response.json().get("chart", {}).get("result", [])

            if (not raw_data) or ('timestamp' not in raw_data[0]):
                print(f"Warning: No data found for {symbol}.")
                continue

            timestamps = raw_data[0]["timestamp"]
            indicators = raw_data[0]["indicators"]["quote"][0]
            adjusted_close = (raw_data[0]["indicators"]["adjclose"][0]
                              ["adjclose"]
                              )

            processed_data_symbol = (
                pd.DataFrame()
                .assign(date=pd.to_datetime(pd.to_datetime(timestamps,
                                                           utc=True,
                                                           unit="s").date),
                        symbol=symbol,
                        volume=indicators.get("volume"),
                        open=indicators.get("open"),
                        low=indicators.get("low"),
                        high=indicators.get("high"),
                        close=indicators.get("close"),
                        adjusted_close=adjusted_close
                        )
                .dropna()
                )

            all_data.append(processed_data_symbol)

        else:
            print(f"Failed to retrieve data for {symbol} (Status code: "
                  f"{response.status_code})")

    all_data = pd.concat(all_data,
                         ignore_index=True) if all_data else pd.DataFrame()
    return all_data


def download_data_osap(
    start_date: str = None,
    end_date: str = None,
    sheet_id: str = "1JyhcF5PRKHcputlioxlu5j5GyLo4JYyY"
) -> pd.DataFrame:
    """
    Download and process Open Source Asset Pricing (OSAP) data.

    Parameters
    ----------
    start_date : str, optional
        Start date in "YYYY-MM-DD" format. If None, full dataset is returned.
    end_date : str, optional
        End date in "YYYY-MM-DD" format. If None, full dataset is returned.
    sheet_id : str, optional
        Google Sheet ID from which to download the dataset.
        Default is "1JyhcF5PRKHcputlioxlu5j5GyLo4JYyY".

    Returns
    -------
    pd.DataFrame
        Processed dataset with snake_case column names,
        filtered by date range if provided.
    """
    start_date, end_date = _validate_dates(start_date, end_date)

    # Google Drive direct download link
    url = f"https://drive.google.com/uc?export=download&id={sheet_id}"

    try:
        raw_data = pd.read_csv(url)
    except Exception:
        print("Returning an empty dataset due to download failure.")
        return pd.DataFrame()

    if raw_data.empty:
        print("Returning an empty dataset due to download failure.")
        return raw_data

    # Convert date column to datetime format
    if "date" in raw_data.columns:
        raw_data["date"] = pd.to_datetime(raw_data["date"], errors="coerce")

    # Convert column names to snake_case
    raw_data.columns = [_transfrom_to_snake_case(col)
                        for col in raw_data.columns]

    # Filter data based on date range
    if start_date and end_date:
        raw_data = raw_data.query('@start_date <= date <= @end_date')

    return raw_data


def download_data_wrds(
    data_type: str,
    start_date: str = None,
    end_date: str = None,
    **kwargs
) -> dict:
    """
    Download data from WRDS based on the specified type.

    Parameters
    ----------
    data_type (str): Type of data to download
        (e.g., "wrds_crsp", "wrds_compustat").
    start_date (str, optional): Start date in "YYYY-MM-DD" format.
    end_date (str, optional): End date in "YYYY-MM-DD" format.
    **kwargs: Additional parameters specific to the dataset type.

    Returns
    -------
        dict: A dictionary representing the downloaded data.
    """
    if "wrds_crsp" in data_type:
        return download_data_wrds_crsp(
            data_type, start_date, end_date, **kwargs
            )
    elif "wrds_compustat" in data_type:
        return download_data_wrds_compustat(
            data_type, start_date, end_date, **kwargs
            )
    elif "wrds_ccm_links" in data_type:
        return download_data_wrds_ccm_links(**kwargs)
    elif "wrds_fisd" in data_type:
        return download_data_wrds_fisd(**kwargs)
    elif "wrds_trace_enhanced" in data_type:
        return download_data_wrds_trace_enhanced(
            start_date, end_date, **kwargs
            )
    else:
        raise ValueError("Unsupported data type.")
    return {}

# Placeholder functions for actual WRDS data retrieval
def download_data_wrds_crsp(
    data_type: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> dict:
    return {"type": data_type, "start_date": start_date, "end_date": end_date, "data": "CRSP data"}

def download_data_wrds_compustat(data_type: str, start_date: str, end_date: str, **kwargs) -> dict:
    return {"type": data_type, "start_date": start_date, "end_date": end_date, "data": "Compustat data"}

def download_data_wrds_ccm_links(**kwargs) -> dict:
    return {"type": "wrds_ccm_links", "data": "CCM Links data"}

def download_data_wrds_fisd(**kwargs) -> dict:
    return {"type": "wrds_fisd", "data": "FISD data"}

def download_data_wrds_trace_enhanced(start_date: str, end_date: str, **kwargs) -> dict:
    return {"type": "wrds_trace_enhanced", "start_date": start_date, "end_date": end_date, "data": "TRACE Enhanced data"}



def estimate_betas(data, model, lookback, min_obs=None, use_furrr=False, data_options=None):
    """Estimate rolling betas for a specified model.

    Parameters:
        data (pd.DataFrame): Data containing identifiers and model variables.
        model (str): Formula for the model (e.g., 'ret_excess ~ mkt_excess').
        lookback (int): Lookback period for rolling estimation.
        min_obs (int, optional): Minimum observations for estimation.
        use_furrr (bool): Whether to use parallel processing.
        data_options (dict, optional): Additional data options.

    Returns:
        pd.DataFrame: Estimated betas.
    """
    pass


def estimate_fama_macbeth(data, model, vcov="newey-west", vcov_options=None, data_options=None):
    """Estimate Fama-MacBeth regressions.

    Parameters:
        data (pd.DataFrame): Data for cross-sectional regressions.
        model (str): Formula for the regression model.
        vcov (str): Type of standard errors ('iid' or 'newey-west').
        vcov_options (dict, optional): Options for covariance matrix estimation.
        data_options (dict, optional): Additional data options.

    Returns:
        pd.DataFrame: Regression results with risk premiums and t-statistics.
    """
    pass


def estimate_model(data, model, min_obs=1):
    """Estimate coefficients of a linear model.

    Parameters:
        data (pd.DataFrame): Data for model estimation.
        model (str): Formula for the model (e.g., 'ret_excess ~ mkt_excess').
        min_obs (int): Minimum observations for estimation.

    Returns:
        pd.DataFrame: Model coefficients.
    """
    pass


def get_random_user_agent():
    """Retrieve a random user agent string.

    Returns
    -------
        str: A random user agent string.
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:116.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.141 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_7_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36 Edg/116.0.1938.69"
        ]
    return str(np.random.choice(user_agents))


def get_wrds_connection(config_path: str = "config.yaml") -> object:
    """
    Establish a connection to Wharton Research Data Services (WRDS) database.

    The function retrieves WRDS credentials from environment variables or
    a config.yaml file  and connects to the WRDS PostgreSQL database using
    SQLAlchemy.

    Parameters
    ----------
        config_path (str): Path to the configuration file.
        Default is "config.yaml".

    Returns
    -------
        object: A connection object to interact with the WRDS database.
    """
    wrds_user, wrds_password = load_wrds_credentials(config_path)

    engine = create_engine((f"postgresql://{wrds_user}:{wrds_password}"
                            "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
                            ),
                           connect_args={"sslmode": "require"}
                           )
    return engine.connect()


def disconnect_wrds_connection(
    connection: object
) -> bool:
    """Close the WRDS database connection.

    Parameters
    ----------
        connection (object): The active database connection to be closed.

    Returns
    -------
        bool: True if disconnection was successful, False otherwise.
    """
    try:
        connection.close()
        return True
    except Exception:
        return False


def list_supported_indexes(
) -> pd.DataFrame:
    """
    Return a DataFrame containing information on supported financial indexes.

    Each index is associated with a URL pointing to a CSV file containing
    the holdings of the index and a `skip` value indicating the number of
    lines to skip when reading the CSV.

    Returns
    -------
        pd.DataFrame: A DataFrame with three columns:
            - index (str): The name of the financial index
            (e.g., "DAX", "S&P 500").
            - url (str): The URL to the CSV file containing holdings data.
            - skip (int): The number of lines to skip when reading CSV file.
    """
    data = [
        ("DAX", "https://www.ishares.com/de/privatanleger/de/produkte/251464/ishares-dax-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=DAXEX_holdings&dataType=fund", 2),
        ("EURO STOXX 50", "https://www.ishares.com/de/privatanleger/de/produkte/251783/ishares-euro-stoxx-50-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXW1_holdings&dataType=fund", 2),
        ("Dow Jones Industrial Average", "https://www.ishares.com/de/privatanleger/de/produkte/251770/ishares-dow-jones-industrial-average-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXI3_holdings&dataType=fund", 2),
        ("Russell 1000", "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239707/ishares-russell-1000-etf/1495092304805.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund", 9),
        ("Russell 2000", "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239710/ishares-russell-2000-etf/1495092304805.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund", 9),
        ("Russell 3000", "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239714/ishares-russell-3000-etf/1495092304805.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund", 9),
        ("S&P 100", "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239723/ishares-sp-100-etf/1495092304805.ajax?fileType=csv&fileName=OEF_holdings&dataType=fund", 9),
        ("S&P 500", "https://www.ishares.com/de/privatanleger/de/produkte/253743/ishares-sp-500-b-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=SXR8_holdings&dataType=fund", 2),
        ("Nasdaq 100", "https://www.ishares.com/de/privatanleger/de/produkte/251896/ishares-nasdaq100-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXXT_holdings&dataType=fund", 2),
        ("FTSE 100", "https://www.ishares.com/de/privatanleger/de/produkte/251795/ishares-ftse-100-ucits-etf-inc-fund/1478358465952.ajax?fileType=csv&fileName=IUSZ_holdings&dataType=fund", 2),
        ("MSCI World", "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=EUNL_holdings&dataType=fund", 2),
        ("Nikkei 225", "https://www.ishares.com/ch/professionelle-anleger/de/produkte/253742/ishares-nikkei-225-ucits-etf/1495092304805.ajax?fileType=csv&fileName=CSNKY_holdings&dataType=fund", 2),
        ("TOPIX", "https://www.blackrock.com/jp/individual-en/en/products/279438/fund/1480664184455.ajax?fileType=csv&fileName=1475_holdings&dataType=fund", 2)
    ]
    return pd.DataFrame(data, columns=["index", "url", "skip"])


def list_supported_types(domain=None, as_vector=False):
    """List all supported dataset types.

    Parameters:
        domain (list, optional): Filter for specific domains.
        as_vector (bool): Whether to return as a list instead of DataFrame.

    Returns:
        Union[pd.DataFrame, list]: Supported dataset types.
    """
    pass


def list_tidy_finance_chapters(
) -> list:
    """
    Return a list of available chapters in the Tidy Finance resource.

    Returns
    -------
        list: A list where each element is the name of a chapter available in
        the Tidy Finance resource.
    """
    return [
        "setting-up-your-environment",
        "introduction-to-tidy-finance",
        "accessing-and-managing-financial-data",
        "wrds-crsp-and-compustat",
        "trace-and-fisd",
        "other-data-providers",
        "beta-estimation",
        "univariate-portfolio-sorts",
        "size-sorts-and-p-hacking",
        "value-and-bivariate-sorts",
        "replicating-fama-and-french-factors",
        "fama-macbeth-regressions",
        "fixed-effects-and-clustered-standard-errors",
        "difference-in-differences",
        "factor-selection-via-machine-learning",
        "option-pricing-via-machine-learning",
        "parametric-portfolio-policies",
        "constrained-optimization-and-backtesting",
        "wrds-dummy-data",
        "cover-and-logo-design",
        "clean-enhanced-trace-with-r",
        "proofs",
        "changelog"
    ]


def load_wrds_credentials(
    config_path: str = "config.yaml"
) -> tuple:
    """
    Load WRDS credentials from a config.yaml file if env variables are not set.

    Parameters
    ----------
        config_path (str): Path to the configuration file.
        Default is "config.yaml".

    Returns
    -------
        tuple: A tuple containing (wrds_user (str), wrds_password (str)).
    """
    wrds_user: str = os.getenv("WRDS_USER")
    wrds_password: str = os.getenv("WRDS_PASSWORD")

    if not wrds_user or not wrds_password:
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                wrds_user = config.get("WRDS", {}).get("USER", "")
                wrds_password = config.get("WRDS", {}).get("PASSWORD", "")

    if not wrds_user or not wrds_password:
        raise ValueError("WRDS credentials not found. Please set 'WRDS_USER' "
                         "and 'WRDS_PASSWORD' as environment variables or "
                         "in config.yaml.")

    return wrds_user, wrds_password


def open_tidy_finance_website(
    chapter: str = None
) -> None:
    """Open the Tidy Finance website or a specific chapter in a browser.

    Parameters
    ----------
        chapter (str, optional): Name of the chapter to open. Defaults to None.

    Returns
    -------
        None
    """
    base_url = "https://www.tidy-finance.org/python/"

    if chapter:
        tidy_finance_chapters = list_tidy_finance_chapters()
        if chapter in tidy_finance_chapters:
            final_url = f"{base_url}{chapter}.html"
        else:
            final_url = base_url
    else:
        final_url = base_url

    webbrowser.open(final_url)


def set_wrds_credentials() -> None:
    """Set WRDS credentials in the environment.

    Prompts the user for WRDS credentials and stores them in a YAML
    configuration file.

    The user can choose to store the credentials in the project directory or
    the home directory. If credentials already exist, the user is prompted for
    confirmation before overwriting them. Additionally, the user is given an
    option to add the configuration file to .gitignore.

    Returns
    -------
        - Saves the WRDS credentials in a `config.yaml` file
        - Optionally adds `config.yaml` to `.gitignore`
    """
    wrds_user = input("Enter your WRDS username: ")
    wrds_password = input("Enter your WRDS password: ")
    location_choice = (input("Where do you want to store the config.yaml "
                             "file? Enter 'project' for project directory or "
                             "'home' for home directory: ")
                       .strip().lower()
                       )

    if location_choice == "project":
        config_path = os.path.join(os.getcwd(), "config.yaml")
        gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    elif location_choice == "home":
        config_path = os.path.join(os.path.expanduser("~"), "config.yaml")
        gitignore_path = os.path.join(os.path.expanduser("~"), ".gitignore")
    else:
        print("Invalid choice. Please start again and enter "
              "'project' or 'home'.")
        return

    config: dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}

    if "WRDS" in config and "USER" in config["WRDS"] and "PASSWORD" in config["WRDS"]:
        overwrite_choice = (input("Credentials already exist. Do you want to "
                                  "overwrite them? Enter 'yes' or 'no': ")
                            .strip().lower()
                            )
        if overwrite_choice != "yes":
            print("Aborted. Credentials already exist and are not "
                  "overwritten.")
            return

    if os.path.exists(gitignore_path):
        add_gitignore = (input("Do you want to add config.yaml to .gitignore? "
                               "It is highly recommended! "
                               "Enter 'yes' or 'no': ")
                         .strip().lower()
                         )
        if add_gitignore == "yes":
            with open(gitignore_path, "r") as file:
                gitignore_lines = file.readlines()
            if "config.yaml\n" not in gitignore_lines:
                with open(gitignore_path, "a") as file:
                    file.write("config.yaml\n")
                print("config.yaml added to .gitignore.")
        elif add_gitignore == "no":
            print("config.yaml NOT added to .gitignore.")
        else:
            print("Invalid choice. Please start again "
                  "and enter 'yes' or 'no'.")
            return

    config["WRDS"] = {"USER": wrds_user, "PASSWORD": wrds_password}

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    print("WRDS credentials have been set and saved in config.yaml in your "
          f"{location_choice} directory.")


def trim(x, cut):
    """Trim a numeric vector by removing extreme values.

    Parameters:
        x (pd.Series): Numeric vector to trim.
        cut (float): Proportion to trim from both ends.

    Returns:
        pd.Series: Trimmed vector.
    """
    pass


def _winsorize(
    x: np.ndarray,
    cut: float
) -> np.ndarray:
    """Winsorize a numeric vector by replacing extreme values.

    Parameters
    ----------
        x (pd.Series): Numeric vector to winsorize.
        cut (float): Proportion to replace at both ends.

    Returns
    -------
        pd.Series: Winsorized vector.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        #raise ValueError("x must be an numpy array")

    if not (0 <= cut <= 0.5):
        raise ValueError("'cut' must be inside [0, 0.5].")

    if x.size == 0:
        return x
    
    x = np.array(x)  # Convert input to numpy array if not already
    lb, ub = np.nanquantile(x, [cut, 1 - cut])  # Compute quantiles
    x = np.clip(x, lb, ub)  # Winsorize values
    return x


def _validate_dates(
    start_date: str = None,
    end_date: str = None,
    use_default_range: bool = False
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
            print("No start_date or end_date provided. Using the range "
                  f"{start_date.date()} to {end_date.date()} to avoid "
                  "downloading large amounts of data.")
            return start_date.date(), end_date.date()
        else:
            print("No start_date or end_date provided. Returning the full "
                  "dataset.")
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
        dates = dates.dt.to_timestamp(how='start').dt.date
    dates = pd.to_datetime(dates, errors='coerce')


def _transfrom_to_snake_case(column_name):
    """
    Convert a string to snake_case.

    - Converts uppercase letters to lowercase.
    - Replaces spaces and special characters with underscores.
    - Ensures no multiple underscores.
    """
    column_name = column_name.replace(" ", "_").replace("-", "_").lower()
    column_name = "".join(c if c.isalnum() or c == "_" else "_"
                          for c in column_name)

    # Remove multiple underscores
    while "__" in column_name:
        column_name = column_name.replace("__", "_")

    # Remove leading/trailing underscores
    return column_name.strip("_")







