from .backend import set_backend, get_backend, use_backend
from .core import (
    add_lagged_columns,
    assign_portfolio,
    breakpoint_options,
    compute_breakpoints,
    compute_portfolio_returns,
    portfolio_sort_options,
    compute_long_short_returns,
    compute_rolling_value,
    create_summary_statistics,
    data_options,
    estimate_betas,
    estimate_fama_macbeth,
    estimate_model,
    filter_options,
    join_lagged_values,
    filter_sorting_data,
    implement_portfolio_sort,
)
from .data_download import (
    _download_data_constituents,
    _download_data_factors_ff,
    _download_data_factors_q,
    _download_data_macro_predictors,
    _download_data_osap,
    _download_data_risk_free,
    _download_data_stock_prices,
    _download_data_wrds,
    download_data,
    get_available_famafrench_datasets,
)
from .utilities import (
    get_wrds_connection,
    list_supported_indexes,
    list_tidy_finance_chapters,
    open_tidy_finance_website,
    set_wrds_credentials,
    trim,
    winsorize,
)
from .supported_datasets import list_supported_datasets

# Wrap the public, data-bearing API at the boundary so it honors the
# active backend (see tidyfinance.set_backend). The wrapped functions
# accept polars or pandas input and return the active backend's data
# frame type. Internal calls between core functions are left untouched,
# since they reference the original (undecorated) functions in their
# defining modules.
add_lagged_columns = use_backend(add_lagged_columns)
assign_portfolio = use_backend(assign_portfolio)
compute_breakpoints = use_backend(compute_breakpoints)
compute_portfolio_returns = use_backend(compute_portfolio_returns)
compute_long_short_returns = use_backend(compute_long_short_returns)
compute_rolling_value = use_backend(compute_rolling_value)
create_summary_statistics = use_backend(create_summary_statistics)
estimate_betas = use_backend(estimate_betas)
estimate_fama_macbeth = use_backend(estimate_fama_macbeth)
estimate_model = use_backend(estimate_model)
join_lagged_values = use_backend(join_lagged_values)
filter_sorting_data = use_backend(filter_sorting_data)
implement_portfolio_sort = use_backend(implement_portfolio_sort)
download_data = use_backend(download_data)
list_supported_datasets = use_backend(list_supported_datasets)
list_supported_indexes = use_backend(list_supported_indexes)

__all__ = [
    "set_backend",
    "get_backend",
    "download_data",
    "get_available_famafrench_datasets",
    "_download_data_factors_ff",
    "_download_data_factors_q",
    "_download_data_macro_predictors",
    "_download_data_wrds",
    "_download_data_constituents",
    "_download_data_stock_prices",
    "_download_data_osap",
    "_download_data_risk_free",
    "list_supported_indexes",
    "list_tidy_finance_chapters",
    "open_tidy_finance_website",
    "trim",
    "winsorize",
    "assign_portfolio",
    "compute_breakpoints",
    "estimate_betas",
    "estimate_fama_macbeth",
    "estimate_model",
    "set_wrds_credentials",
    "create_summary_statistics",
    "add_lagged_columns",
    "compute_portfolio_returns",
    "compute_long_short_returns",
    "compute_rolling_value",
    "portfolio_sort_options",
    "data_options",
    "filter_options",
    "breakpoint_options",
    "join_lagged_values",
    "get_wrds_connection",
    "list_supported_datasets",
    "filter_sorting_data",
    "implement_portfolio_sort",
]
