"""Tests for portfolio_sort_options."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import (  # noqa: E402
    breakpoint_options,
    filter_options,
    portfolio_sort_options,
)


def test_portfolio_sort_options_returns_dict():
    """Test portfolio_sort_options returns a dict."""
    opts = portfolio_sort_options(
        breakpoint_options_main=breakpoint_options(n_portfolios=5)
    )
    assert isinstance(opts, dict)


def test_portfolio_sort_options_stores_all_provided_components():
    """Test portfolio_sort_options stores all provided components."""
    fo = filter_options(exclude_financials=True)
    bpm = breakpoint_options(n_portfolios=10)
    bps = breakpoint_options(percentiles=[0.3, 0.7])

    opts = portfolio_sort_options(
        filter_options=fo,
        breakpoint_options_main=bpm,
        breakpoint_options_secondary=bps,
    )

    assert opts["filter_options"] is fo
    assert opts["breakpoint_options_main"] is bpm
    assert opts["breakpoint_options_secondary"] is bps


def test_portfolio_sort_options_accepts_none_filter_options():
    """Test portfolio_sort_options accepts None filter_options."""
    portfolio_sort_options(
        filter_options=None,
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
    )


def test_portfolio_sort_options_accepts_none_breakpoint_options_secondary():
    """Test portfolio_sort_options accepts None breakpoint_options_secondary."""
    opts = portfolio_sort_options(
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
        breakpoint_options_secondary=None,
    )
    assert opts["breakpoint_options_secondary"] is None


def test_portfolio_sort_options_stores_extra_arguments_via_kwargs():
    """Test portfolio_sort_options stores extra arguments via kwargs."""
    opts = portfolio_sort_options(
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
        custom_option="value",
    )
    assert opts["custom_option"] == "value"


def test_portfolio_sort_options_errors_when_filter_options_wrong_shape():
    """Test portfolio_sort_options errors when filter_options has wrong shape."""
    with pytest.raises(ValueError, match="filter_options"):
        portfolio_sort_options(
            filter_options={"exclude_financials": True},
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
        )


def test_portfolio_sort_options_errors_when_breakpoint_options_main_wrong_shape():
    """Test portfolio_sort_options errors when breakpoint_options_main wrong shape."""
    with pytest.raises(ValueError, match="breakpoint_options_main"):
        portfolio_sort_options(
            breakpoint_options_main={"n_portfolios": 5},
        )


def test_portfolio_sort_options_errors_when_breakpoint_options_main_missing():
    """Test portfolio_sort_options errors when breakpoint_options_main is missing."""
    with pytest.raises(ValueError, match="breakpoint_options_main"):
        portfolio_sort_options()


def test_portfolio_sort_options_errors_when_breakpoint_options_secondary_wrong_shape():
    """Test portfolio_sort_options errors when breakpoint_options_secondary wrong shape."""
    with pytest.raises(ValueError, match="breakpoint_options_secondary"):
        portfolio_sort_options(
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
            breakpoint_options_secondary={"n_portfolios": 3},
        )


if __name__ == "__main__":
    pytest.main([__file__])
