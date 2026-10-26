"""Tests for the public API of tidyfinance."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tidyfinance  # noqa: E402

# The exported functions of the R package tidyfinance. Update this list
# together with the package when a function is added or removed there.
R_EXPORTS = {
    "add_lagged_columns",
    "assign_portfolio",
    "breakpoint_options",
    "compute_breakpoints",
    "compute_long_short_returns",
    "compute_portfolio_returns",
    "compute_rolling_value",
    "create_summary_statistics",
    "data_options",
    "disconnect_connection",
    "download_data",
    "download_data_constituents",
    "download_data_damodaran",
    "download_data_factors_ff",
    "download_data_factors_q",
    "download_data_fred",
    "download_data_fred_md",
    "download_data_huggingface",
    "download_data_jkp",
    "download_data_macro_predictors",
    "download_data_osap",
    "download_data_pastor_stambaugh",
    "download_data_pseudo_ccm_links",
    "download_data_pseudo_compustat",
    "download_data_pseudo_crsp",
    "download_data_risk_free",
    "download_data_stambaugh_yuan",
    "download_data_stock_prices",
    "download_data_wrds",
    "download_data_wrds_ccm_links",
    "download_data_wrds_compustat",
    "download_data_wrds_crsp",
    "download_data_wrds_fisd",
    "download_data_wrds_trace_enhanced",
    "download_factor_library_grid",
    "download_factor_library_ids",
    "estimate_betas",
    "estimate_fama_macbeth",
    "estimate_model",
    "filter_options",
    "filter_sorting_data",
    "get_available_huggingface_files",
    "get_wrds_connection",
    "implement_portfolio_sort",
    "join_lagged_values",
    "list_supported_datasets",
    "list_supported_indexes",
    "list_supported_jkp_factors",
    "list_supported_types",
    "list_tidy_finance_chapters",
    "open_tidy_finance_website",
    "portfolio_sort_options",
    "set_wrds_credentials",
    "trim",
    "winsorize",
}

# Functions only the Python package needs.
PYTHON_ONLY = {"get_backend", "set_backend"}

# Public functions that return no data frame, so they are not wrapped
# for the backend: options, connections, links, and vector helpers.
NOT_BACKEND_WRAPPED = {
    "breakpoint_options",
    "data_options",
    "disconnect_connection",
    "filter_options",
    "get_backend",
    "get_wrds_connection",
    "list_tidy_finance_chapters",
    "open_tidy_finance_website",
    "portfolio_sort_options",
    "set_backend",
    "set_wrds_credentials",
    "trim",
    "winsorize",
}


def test_public_api_matches_the_exports_of_the_r_package():
    """Test the public API is the R exports plus the backend functions."""
    public = set(tidyfinance.__all__)
    expected = R_EXPORTS | PYTHON_ONLY
    assert not (expected - public), (
        f"Missing from the public API: {sorted(expected - public)}"
    )
    assert not (public - expected), (
        f"Public, but not exported by the R package: "
        f"{sorted(public - expected)}"
    )


def is_backend_wrapped(fn) -> bool:
    """Whether 'fn' is the backend wrapper of tidyfinance.backend.

    Checks the code of the function rather than its identity, so the
    result does not depend on whether a test reloaded a module."""
    return fn.__code__.co_qualname == "_use_backend.<locals>.wrapper"


@pytest.mark.parametrize(
    "name", sorted((R_EXPORTS | PYTHON_ONLY) - NOT_BACKEND_WRAPPED)
)
def test_data_frame_functions_honor_the_backend(name):
    """Test every public function that returns data frames is wrapped
    for the backend."""
    assert is_backend_wrapped(getattr(tidyfinance, name)), (
        f"'{name}' is not wrapped for the backend; add it to "
        "_BACKEND_WRAPPED in tidyfinance/__init__.py."
    )


@pytest.mark.parametrize("name", sorted(NOT_BACKEND_WRAPPED))
def test_functions_without_data_frames_are_not_wrapped(name):
    """Test functions that return no data frame are not wrapped."""
    assert not is_backend_wrapped(getattr(tidyfinance, name))


if __name__ == "__main__":
    pytest.main([__file__])
