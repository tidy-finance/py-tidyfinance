"""Tests for implement_portfolio_sort."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import (  # noqa: E402
    breakpoint_options,
    data_options,
    filter_options,
    implement_portfolio_sort,
    portfolio_sort_options,
)


def make_data(n_stocks=30, n_months=6, seed=42):
    """Construct a stock-month panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = [(p, d) for d in dates for p in range(1, n_stocks + 1)]
    df = pd.DataFrame(rows, columns=["permno", "date"])
    n = len(df)
    df["ret_excess"] = rng.standard_normal(n)
    df["size"] = rng.uniform(50, 150, n)
    df["mktcap_lag"] = rng.uniform(100, 1000, n)
    return df


_pso = portfolio_sort_options(
    breakpoint_options_main=breakpoint_options(n_portfolios=3)
)


def test_errors_when_quiet_is_not_logical():
    """Test errors when quiet is not boolean."""
    with pytest.raises(ValueError, match="quiet"):
        implement_portfolio_sort(
            make_data(),
            sorting_variables="size",
            sorting_method="univariate",
            portfolio_sort_options=_pso,
            quiet="yes",
        )


def test_errors_when_quiet_has_length_greater_than_1():
    """Test errors when quiet is a list."""
    with pytest.raises(ValueError, match="quiet"):
        implement_portfolio_sort(
            make_data(),
            sorting_variables="size",
            sorting_method="univariate",
            portfolio_sort_options=_pso,
            quiet=[True, False],
        )


def test_errors_when_quiet_is_none():
    """Test errors when quiet is None (Python equivalent of R's NA)."""
    with pytest.raises(ValueError, match="quiet"):
        implement_portfolio_sort(
            make_data(),
            sorting_variables="size",
            sorting_method="univariate",
            portfolio_sort_options=_pso,
            quiet=None,
        )


def test_errors_when_data_options_is_non_none_with_wrong_shape():
    """Test errors when data_options is non-None with wrong shape."""
    with pytest.raises(ValueError, match="data_options"):
        implement_portfolio_sort(
            make_data(),
            sorting_variables="size",
            sorting_method="univariate",
            portfolio_sort_options=_pso,
            data_options={"date": "date"},
        )


def test_errors_when_portfolio_sort_options_has_wrong_shape():
    """Test errors when portfolio_sort_options has wrong shape."""
    with pytest.raises(ValueError, match="portfolio_sort_options"):
        implement_portfolio_sort(
            make_data(),
            sorting_variables="size",
            sorting_method="univariate",
            portfolio_sort_options={"breakpoint_options_main": "x"},
        )


def test_delegates_to_filter_and_compute_on_valid_inputs():
    """Test delegates to filter and compute on valid inputs."""
    data = make_data()
    pso = portfolio_sort_options(
        filter_options=filter_options(min_stock_price=None),
        breakpoint_options_main=breakpoint_options(n_portfolios=3),
    )
    result = implement_portfolio_sort(
        data,
        sorting_variables="size",
        sorting_method="univariate",
        portfolio_sort_options=pso,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "portfolio" in result.columns
    assert "date" in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
