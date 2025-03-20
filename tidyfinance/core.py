"""Main module for tidyfinance package."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS


def add_lag_columns(
    data: pd.DataFrame,
    cols: list[str],
    by: str | None = None,
    lag: int = 0,
    max_lag: int | None = None,
    drop_na: bool = False,
    date_col: str = "date",
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
        raise ValueError(
            "`lag` must be non-negative, "
            "and `max_lag` must be greater than or "
            "equal to `lag`."
        )

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


def assign_portfolio(
    data: pd.DataFrame,
    sorting_variable: str,
    breakpoint_options: dict = None,
    breakpoint_function=None,
) -> pd.Series:
    """
    Assign portfolio labels based on a sorting variable and breakpoints.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame containing the dataset for
        portfolio assignment.
    sorting_variable (str): Column name used for sorting and
        portfolio assignment.
    breakpoint_options (dict, optional): Named arguments passed
        to the breakpoint function.
    breakpoint_function (callable, optional): Function to compute breakpoints.
        Must return an ascending vector of breakpoints.

    Returns
    -------
    pd.Series: A Series of portfolio assignments.
    """
    if sorting_variable not in data.columns:
        raise ValueError(
            f"Sorting variable '{sorting_variable}' not found in data."
        )

    if len(data[sorting_variable].unique()) == 1:
        print(
            "Warning: The sorting variable is constant, assigning all to portfolio 1."
        )
        return pd.Series(1, index=data.index, dtype=int)

    if breakpoint_function is None:
        raise ValueError("A valid breakpoint function must be provided.")

    # Compute breakpoints
    breakpoints = breakpoint_function(
        data, sorting_variable, breakpoint_options
    )

    # Assign portfolios using pd.cut
    assigned_portfolios = pd.cut(
        data[sorting_variable],
        bins=breakpoints,
        labels=range(1, breakpoints.size),
        include_lowest=True,
        right=False,
    )

    return assigned_portfolios.astype(int)


def breakpoint_options(
    n_portfolios: int = None,
    percentiles: list = None,
    breakpoint_exchanges: str = None,
    smooth_bunching: bool = False,
    **kwargs,
) -> dict:
    """
    Create a dictionary of options for breakpoints in portfolio sorting.

    Parameters
    ----------
    n_portfolios (int, optional): Number of portfolios to create.
        Must be a positive integer.
    percentiles (list, optional): List of percentile thresholds
        (values between 0 and 1).
    breakpoint_exchanges (str, optional): Exchange for
        which the breakpoints apply.
    smooth_bunching (bool, optional): Whether smooth bunching
        should be applied. Default is False.
    **kwargs: Additional optional arguments.

    Returns
    -------
    dict: A dictionary containing breakpoint options.
    """
    # Validate n_portfolios
    if n_portfolios is not None:
        if not isinstance(n_portfolios, int) or n_portfolios <= 0:
            raise ValueError("n_portfolios must be a positive integer.")

    # Validate percentiles
    if percentiles is not None:
        if not all(
            isinstance(p, (int, float)) and 0 <= p <= 1 for p in percentiles
        ):
            raise ValueError(
                "percentiles must be a list of values between 0 and 1."
            )

    # Validate breakpoint_exchanges
    if breakpoint_exchanges is not None:
        if not isinstance(breakpoint_exchanges, str) or (
            not breakpoint_exchanges
        ):
            raise ValueError("breakpoint_exchanges must be a non-empty string.")

    # Validate smooth_bunching
    if not isinstance(smooth_bunching, bool):
        raise ValueError("smooth_bunching must be a boolean value.")

    options = {
        "n_portfolios": n_portfolios,
        "percentiles": percentiles,
        "breakpoint_exchanges": breakpoint_exchanges,
        "smooth_bunching": smooth_bunching,
        **kwargs,
    }
    return options


def compute_breakpoints(
    data: pd.DataFrame, sorting_variable: str, breakpoint_options: dict
) -> np.ndarray:
    """
    Compute breakpoints based on a sorting variable for portfolios.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame with the dataset for breakpoint computation.
    sorting_variable (str): Column name used to determine breakpoints.
    breakpoint_options (dict): Dictionary containing breakpoint parameters,
        including:
        - "n_portfolios" (int, optional): Number of equally sized portfolios
        - "percentiles" (list, optional): Custom percentiles for breakpoints
        - "breakpoint_exchanges" (list, optional):
                Exchanges to filter the data before computing breakpoints
        - "smooth_bunching" (bool, optional): To smooth edge breakpoints or not

    Returns
    -------
    np.ndarray: A sorted array of breakpoints.
    """
    if not isinstance(breakpoint_options, dict):
        raise ValueError("breakpoint_options must be a dictionary.")

    n_portfolios = breakpoint_options.get("n_portfolios")
    percentiles = breakpoint_options.get("percentiles")
    exchanges = breakpoint_options.get("breakpoint_exchanges")
    smooth_bunching = breakpoint_options.get("smooth_bunching", False)

    if n_portfolios is not None and percentiles is not None:
        raise ValueError(
            "Provide either 'n_portfolios' or 'percentiles', not both."
        )
    elif n_portfolios is None and percentiles is None:
        raise ValueError(
            "Either 'n_portfolios' or 'percentiles' must be specified."
        )

    if exchanges is not None:
        if "exchange" not in data.columns:
            raise ValueError(
                "Data must contain an 'exchange' column to use breakpoint_exchanges."
            )
        data = data.query(f"exchange in {exchanges}")

    if n_portfolios is not None:
        if n_portfolios <= 1:
            raise ValueError("n_portfolios must be greater than 1.")
        breakpoints = (
            data.get(sorting_variable)
            .quantile(
                np.linspace(0, 1, num=n_portfolios + 1), interpolation="linear"
            )
            .drop_duplicates()
        )
    else:
        breakpoints = (
            data.get(sorting_variable)
            .quantile([0] + percentiles + [1], interpolation="linear")
            .drop_duplicates()
        )
    try:
        breakpoints.iloc[0] = -np.inf
        breakpoints.iloc[-1] = np.inf
    except AttributeError:
        breakpoints.iloc[0] = -np.Inf
        breakpoints.iloc[-1] = np.Inf

    if smooth_bunching:
        if (breakpoints[0] == breakpoints[1]) and (
            breakpoints[-2] == breakpoints[-1]
        ):
            print(
                "Warning: Clustering at extreme breakpoints detected. "
                "Adjusting non-edge portfolios."
            )
            new_values = data[sorting_variable][
                (data[sorting_variable] > breakpoints[0])
                & (data[sorting_variable] < breakpoints[-1])
            ]
            new_probs = np.linspace(0, 1, len(breakpoints) - 2)
            breakpoints[1:-1] = np.quantile(new_values, new_probs)

    breakpoints.loc[1:] += 1e-20  # Ensure proper binning
    return breakpoints


# def compute_long_short_returns(
#     data: pd.DataFrame,
#     direction: str = "top_minus_bottom",
#     date_col: str = "date",
#     portfolio_col: str = "portfolio",
#     ret_excess_col: str = "ret_excess",
# ) -> pd.DataFrame:
#     """
#     Compute long-short returns based on portfolio returns.

#     Parameters
#     ----------
#     data (pd.DataFrame): DataFrame containing portfolio returns with columns
#         for portfolio ID, date, and return measurements.
#     direction (str, optional): Direction of calculation. "top_minus_bottom"
#         (default) or "bottom_minus_top".
#     date_col (str, optional): Column name indicating dates.
#     portfolio_col (str, optional): Column name indicating portfolio
#         identifiers.
#     ret_excess_col (str, optional): Column name prefix for excess return
#         measurements.

#     Returns
#     -------
#     pd.DataFrame: A DataFrame with computed long-short returns.
#     """
#     if direction not in ["top_minus_bottom", "bottom_minus_top"]:
#         raise ValueError(
#             "direction must be either 'top_minus_bottom' or'bottom_minus_top'"
#         )
#     data = data.copy()

#     # Identify top and bottom portfolios
#     grouped = data.groupby(date_col)
#     top_bottom = grouped.filter(lambda x: x[portfolio_col].nunique() >= 2)
#     top_bottom["portfolio"] = np.where(
#         top_bottom[portfolio_col] == top_bottom[portfolio_col].max(),
#         "top",
#         "bottom",
#     )

#     # Pivot data to get top and bottom returns
#     long_short_df = (
#         top_bottom.pivot_table(
#             index=[date_col], columns="portfolio", values=ret_excess_col
#         )
#         .dropna()
#         .reset_index()
#     )

#     # Compute long-short returns
#     long_short_df["long_short_return"] = (
#         long_short_df["top"] - long_short_df["bottom"]
#     ) * (-1 if direction == "bottom_minus_top" else 1)

#     return long_short_df[[date_col, "long_short_return"]]


# def compute_portfolio_returns(
#     sorting_data,
#     sorting_variables,
#     sorting_method,
#     rebalancing_month=None,
#     breakpoint_options_main=None,
#     breakpoint_options_secondary=None,
#     breakpoint_function_main=None,
#     breakpoint_function_secondary=None,
#     min_portfolio_size=0,
#     data_options=None,
# ):
#     """Compute portfolio returns based on sorting variables and methods.

#     Parameters:
#         sorting_data (pd.DataFrame): Data for portfolio assignment and return computation.
#         sorting_variables (list): List of variables for sorting.
#         sorting_method (str): Sorting method ('univariate' or 'bivariate').
#         rebalancing_month (int, optional): Month for annual rebalancing.
#         breakpoint_options_main (dict, optional): Options for main sorting variable.
#         breakpoint_options_secondary (dict, optional): Options for secondary sorting variable.
#         breakpoint_function_main (callable, optional): Function for main sorting.
#         breakpoint_function_secondary (callable, optional): Function for secondary sorting.
#         min_portfolio_size (int): Minimum portfolio size.
#         data_options (dict, optional): Additional data processing options.

#     Returns:
#         pd.DataFrame: DataFrame with computed portfolio returns.
#     """
#     pass


def create_summary_statistics(
    data: pd.DataFrame,
    variables: list,
    by: str = None,
    detail: bool = False,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Compute summary statistics for specified numeric variables in a DataFrame.

    Parameters
    ----------
    data (pd.DataFrame): A DataFrame containing the data to summarize.
    variables (list): List of column names to summarize
        (must be numeric or boolean).
    by (str, optional): A column name to group data before summarization.
        Default is None.
    detail (bool, optional): Whether to include detailed quantiles.
        Default is False.
    drop_na (bool, optional): Whether to drop missing values before
        summarizing. Default is False.

    Returns
    -------
    pd.DataFrame: A DataFrame containing summary statistics for
        each selected variable.
    """
    # Check that all specified variables are numeric or boolean
    non_numeric_vars = [
        var
        for var in variables
        if not np.issubdtype(data[var].dtype, np.number)
    ]
    if non_numeric_vars:
        raise ValueError(
            "The following columns are not numeric or boolean: "
            f"{', '.join(non_numeric_vars)}"
        )

    # Drop missing values if specified
    if drop_na:
        data = data.dropna(subset=variables)

    # Compute summary statistics using describe
    percentiles = (
        [0.5]
        if not detail
        else [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    if by:
        summary_df = (
            data.groupby(by)
            .describe(percentiles=percentiles)
            .get(variables)
            .reset_index()
            .rename(columns={"index": "variable"})
        )
    else:
        summary_df = (
            data.get(variables)
            .describe(percentiles=percentiles)
            .transpose()
            .reset_index()
            .rename(columns={"index": "variable"})
        )

    return summary_df.round(3)


def estimate_betas(
    data: pd.DataFrame,
    model: str,
    lookback: pd.Timedelta,
    min_obs: int = None,
    id_col: str = "permno",
) -> pd.DataFrame:
    """
    Estimate rolling betas using RollingOLS.from_formula from statsmodels.

    Parameters
    ----------
    data (pd.DataFrame): DataFrame containing stock return data with date and
        stock identifier columns.
    model (str): A formula representing the regression model
        (e.g., 'ret_excess ~ mkt_excess').
    lookback (int): The period window size to estimate rolling betas.
    min_obs (int, optional): Minimum number of observations required for a
        valid estimate. Defaults to 80% of lookback.
    id_col (str, optional): Column name representing the stock identifier.
        Defaults to 'permno'.

    Returns
    -------
    pd.DataFrame: A DataFrame with estimated rolling betas for each stock
        and time period.
    """
    if min_obs is None:
        min_obs = int(lookback * 0.8)
    elif min_obs <= 0:
        raise ValueError("min_obs must be a positive integer.")

    results = []
    for stock_id, group in data.groupby(id_col):
        group = group.sort_values("date")

        rolling_model = RollingOLS.from_formula(
            formula=model,
            data=group,
            window=lookback,
            min_nobs=min_obs,
            missing="drop",
        ).fit()

        betas = rolling_model.params
        betas[id_col] = stock_id
        betas["date"] = group["date"].values
        results.append(betas)

    betas_df = pd.concat(results).reset_index()
    return betas_df


def estimate_fama_macbeth(
    data: pd.DataFrame,
    model: str,
    vcov: str = "newey-west",
    vcov_options: dict = {"maxlags": 6},
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Estimate Fama-MacBeth regressions by running cross-sectional regressions.

    Parameters
    ----------
    data (pd.DataFrame): A DataFrame containing the data for the regression.
    model (str): A formula representing the regression model.
    vcov (str): Type of standard errors to compute. Options are "iid" or
        "newey-west".
    vcov_options (dict, optional): Additional options for the Newey-West
        standard errors.
    date_col (str): Column name representing the time periods.

    Returns
    -------
    pd.DataFrame: A DataFrame containing estimated risk premia,
        standard errors, and t-statistics.
    """
    if vcov not in ["iid", "newey-west"]:
        raise ValueError("vcov must be either 'iid' or 'newey-west'.")

    if date_col not in data.columns:
        raise ValueError(f"The data must contain a {date_col} column.")

    # Run cross-sectional regressions
    cross_section_results = []
    for date, group in data.groupby(date_col):
        if len(group) <= len(model.split("~")[1].split("+")):
            continue

        model_fit = smf.ols(model, data=group).fit()
        params = model_fit.params.to_dict()
        params[date_col] = date
        cross_section_results.append(params)

    risk_premiums = pd.DataFrame(cross_section_results)

    # Compute time-series averages
    price_of_risk = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")["estimate"]
        .mean()
        .reset_index()
        .rename(columns={"estimate": "risk_premium"})
    )

    # Compute standard errors based on vcov choice
    def compute_t_statistic(x):
        model = smf.ols("estimate ~ 1", x)
        if vcov == "newey-west":
            fit = model.fit(cov_type="HAC", cov_kwds=vcov_options)
        else:
            fit = model.fit()
        return x["estimate"].mean() / fit.bse["Intercept"]

    price_of_risk_t_stat = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")
        .apply(compute_t_statistic, include_groups=False)
        .reset_index()
        .rename(columns={0: "t_statistic"})
    )

    result_df = price_of_risk.merge(price_of_risk_t_stat, on="factor").round(3)

    return result_df
