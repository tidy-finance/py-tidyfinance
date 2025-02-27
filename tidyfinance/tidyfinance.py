"""Main module for tidyfinance package."""

import pandas as pd
import requests
import pandas_datareader as pdr


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
    pass


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


def compute_breakpoints(data, sorting_variable, breakpoint_options, data_options=None):
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


def create_wrds_dummy_database(path, url=None):
    """Download and save the WRDS dummy database.

    Parameters:
        path (str): File path for the SQLite database.
        url (str, optional): URL of the WRDS dummy database.

    Returns:
        None
    """
    pass


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
    type : str
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
    type : str
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
    type : str
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
        raise ValueError("Returning an empty data set due to download failure.")
        print(f"{data_set} might not be in list of available data sets: "
              " Also check the provided URL. Choose a dataset from:")
        print("")
        print(all_data_sets)
        return pd.DataFrame()


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

    Returns:
        str: A random user agent string.
    """
    pass


def get_wrds_connection():
    """Establish a connection to the WRDS database.

    Returns:
        Any: Database connection object.
    """
    pass


def list_supported_indexes():
    """List supported financial indexes.

    Returns:
        pd.DataFrame: DataFrame with supported indexes and their URLs.
    """
    pass


def list_supported_types(domain=None, as_vector=False):
    """List all supported dataset types.

    Parameters:
        domain (list, optional): Filter for specific domains.
        as_vector (bool): Whether to return as a list instead of DataFrame.

    Returns:
        Union[pd.DataFrame, list]: Supported dataset types.
    """
    pass


def list_tidy_finance_chapters():
    """List available chapters in the Tidy Finance resource.

    Returns:
        list: Names of chapters in Tidy Finance.
    """
    pass


def open_tidy_finance_website(chapter=None):
    """Open the Tidy Finance website or a specific chapter in a browser.

    Parameters:
        chapter (str, optional): Name of the chapter to open. Defaults to None.

    Returns:
        None
    """
    pass


def set_wrds_credentials():
    """Set WRDS credentials in the environment.

    Returns:
        None
    """
    pass


def trim(x, cut):
    """Trim a numeric vector by removing extreme values.

    Parameters:
        x (pd.Series): Numeric vector to trim.
        cut (float): Proportion to trim from both ends.

    Returns:
        pd.Series: Trimmed vector.
    """
    pass


def winsorize(x, cut):
    """Winsorize a numeric vector by replacing extreme values.

    Parameters:
        x (pd.Series): Numeric vector to winsorize.
        cut (float): Proportion to replace at both ends.

    Returns:
        pd.Series: Winsorized vector.
    """
    pass


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
        return dates.dt.to_timestamp(how='start').dt.date
    else:
        return pd.to_datetime(dates, errors='coerce').dt.date







