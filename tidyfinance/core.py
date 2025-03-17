"""Main module for tidyfinance package."""

import os

import pandas as pd
import numpy as np
import requests


from ._internal import (_trim,
                        _winsorize,
                        _validate_dates,
                        _return_datetime,
                        _transfrom_to_snake_case,
                        _assign_exchange,
                        _assign_industry
                        )


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


def list_supported_types(domain=None, as_vector=False):
    """List all supported dataset types.

    Parameters:
        domain (list, optional): Filter for specific domains.
        as_vector (bool): Whether to return as a list instead of DataFrame.

    Returns:
        Union[pd.DataFrame, list]: Supported dataset types.
    """
    pass
