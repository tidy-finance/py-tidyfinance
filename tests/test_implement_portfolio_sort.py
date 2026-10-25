"""Tests for implement_portfolio_sort."""

import datetime as dt
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import (
    breakpoint_options,
    filter_options,
    implement_portfolio_sort,
    portfolio_sort_options,
)  # noqa: E402


def make_data(n_stocks=30, n_months=6, seed=42):
    """Construct a stock-month panel for tests."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2020, i + 1, 1) for i in range(n_months)]
    permnos = [p for _ in dates for p in range(1, n_stocks + 1)]
    date_col = [d for d in dates for _ in range(n_stocks)]
    n = len(permnos)
    return pl.DataFrame(
        {
            "permno": permnos,
            "date": date_col,
            "ret_excess": rng.standard_normal(n),
            "size": rng.uniform(50, 150, n),
            "mktcap_lag": rng.uniform(100, 1000, n),
        }
    )


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


def test_errors_when_portfolio_sort_options_is_missing():
    """Test portfolio_sort_options is required although it has a default
    so that it can follow skip, as in R."""
    with pytest.raises(ValueError, match="portfolio_sort_options"):
        implement_portfolio_sort(make_data(), "size", "univariate")


def test_portfolio_sort_options_follows_skip_as_in_r():
    """Test the arguments are in the order of the R function."""
    import inspect

    params = list(inspect.signature(implement_portfolio_sort).parameters)
    assert params[3:7] == [
        "rebalancing_month",
        "holding_period",
        "skip",
        "portfolio_sort_options",
    ]


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
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0
    assert "portfolio" in result.columns
    assert "date" in result.columns


def test_implement_portfolio_sort_screens_at_formation_when_annual():
    """Test implement_portfolio_sort screens at formation when annual."""
    # July 2020 to June 2021
    dates = [
        dt.date(2020 + (6 + i) // 12, (6 + i) % 12 + 1, 1) for i in range(12)
    ]
    permnos = [p for p in range(1, 21) for _ in dates]
    data = pl.DataFrame(
        {
            "permno": permnos,
            "date": dates * 20,
            "ret_excess": [0.01] * len(permnos),
            "mktcap_lag": [100.0] * len(permnos),
            "size": [float(p) for p in permnos],
            "prc_lag": [10.0] * len(permnos),
        }
    )
    # Stock 1 falls below the price screen after formation and loses 50%
    # in August 2020.
    is_stock1 = pl.col("permno") == 1
    data = data.with_columns(
        prc_lag=pl.when(is_stock1 & (pl.col("date") > dt.date(2020, 7, 1)))
        .then(1.0)
        .otherwise(pl.col("prc_lag")),
        ret_excess=pl.when(is_stock1 & (pl.col("date") == dt.date(2020, 8, 1)))
        .then(-0.5)
        .otherwise(pl.col("ret_excess")),
    )

    options = portfolio_sort_options(
        filter_options=filter_options(min_stock_price=5),
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
    )
    formation = implement_portfolio_sort(
        data,
        sorting_variables="size",
        sorting_method="univariate",
        rebalancing_month=7,
        portfolio_sort_options=options,
        quiet=True,
    )
    aug_formation = formation.filter(
        (pl.col("portfolio") == 1) & (pl.col("date") == dt.date(2020, 8, 1))
    )["ret_excess_ew"][0]
    assert aug_formation == pytest.approx((9 * 0.01 - 0.5) / 10)

    options_ex_post = portfolio_sort_options(
        filter_options=filter_options(
            min_stock_price=5,
            timing="every_period",
        ),
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
    )
    with pytest.warns(UserWarning, match="look-ahead"):
        ex_post = implement_portfolio_sort(
            data,
            sorting_variables="size",
            sorting_method="univariate",
            rebalancing_month=7,
            portfolio_sort_options=options_ex_post,
            quiet=True,
        )
    aug_ex_post = ex_post.filter(
        (pl.col("portfolio") == 1) & (pl.col("date") == dt.date(2020, 8, 1))
    )["ret_excess_ew"][0]
    assert aug_ex_post == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__])
