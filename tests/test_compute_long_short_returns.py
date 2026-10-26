"""Tests for compute_long_short_returns."""

import datetime as dt
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import (  # noqa: E402
    breakpoint_options,
    compute_long_short_returns,
    compute_portfolio_returns,
    data_options,
)


def make_portfolio_panel(n_portfolios=5, n_months=12, seed=42):
    """Construct a portfolio-return panel for tests."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2020 + i // 12, i % 12 + 1, 1) for i in range(n_months)]
    portfolios = [p for _ in dates for p in range(1, n_portfolios + 1)]
    date_col = [d for d in dates for _ in range(n_portfolios)]
    n = len(portfolios)
    return pl.DataFrame(
        {
            "portfolio": portfolios,
            "date": date_col,
            "ret_excess_ew": rng.normal(0, 0.05, n),
            "ret_excess_vw": rng.normal(0, 0.05, n),
        }
    )


def test_compute_long_short_returns_returns_top_minus_bottom_by_default():
    """Test compute_long_short_returns returns top - bottom by default."""
    panel = make_portfolio_panel()
    out = compute_long_short_returns(panel)
    assert list(out.columns) == ["date", "ret_excess_ew", "ret_excess_vw"]
    assert len(out) == 12

    # Sanity-check one date
    d = panel["date"][0]
    rows = panel.filter(pl.col("date") == d)
    expected_ew = (
        rows.filter(pl.col("portfolio") == rows["portfolio"].max())[
            "ret_excess_ew"
        ][0]
        - rows.filter(pl.col("portfolio") == rows["portfolio"].min())[
            "ret_excess_ew"
        ][0]
    )
    actual_ew = out.filter(pl.col("date") == d)["ret_excess_ew"][0]
    assert abs(actual_ew - expected_ew) < 1e-12


def test_compute_long_short_returns_returns_na_when_only_one_portfolio():
    """Test compute_long_short_returns returns null when one portfolio."""
    panel = make_portfolio_panel(n_portfolios=1)
    out = compute_long_short_returns(panel)
    assert list(out.columns) == ["date", "ret_excess_ew", "ret_excess_vw"]
    assert len(out) == 12
    assert out["ret_excess_ew"].is_null().all()
    assert out["ret_excess_vw"].is_null().all()


def test_compute_long_short_returns_handles_a_single_per_date_long_leg():
    """Test compute_long_short_returns handles a single per-date long leg."""
    panel = make_portfolio_panel(n_portfolios=2, n_months=4)
    first_date = panel["date"][0]
    panel = panel.filter(
        ~((pl.col("date") == first_date) & (pl.col("portfolio") == 2))
    )
    out = compute_long_short_returns(panel)
    assert len(out) == 4
    assert out.filter(pl.col("date") == first_date)["ret_excess_ew"][0] is None
    other_dates = out.filter(pl.col("date") != first_date)
    assert other_dates["ret_excess_ew"].is_not_null().all()


if __name__ == "__main__":
    pytest.main([__file__])


def test_an_unknown_direction_is_an_error():
    data = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)] * 2,
            "portfolio": [1, 2],
            "ret_excess_vw": [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="'direction' must be one of"):
        compute_long_short_returns(data, direction="top-bottom")


def test_accepts_the_options_used_for_the_sort():
    data = pl.DataFrame(
        {
            "id": list(range(1, 11)) * 2,
            "date": [dt.date(2020, 1, 1)] * 10 + [dt.date(2020, 2, 1)] * 10,
            "s": [float(i) for i in range(1, 11)] * 2,
            "rx": [i / 100 for i in range(1, 11)] * 2,
            "mktcap_lag": [1.0] * 20,
        }
    )
    options = data_options(id="id", ret_excess="rx")
    portfolio_returns = compute_portfolio_returns(
        data,
        "s",
        "univariate",
        data_options=options,
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
        quiet=True,
    )
    out = compute_long_short_returns(portfolio_returns, data_options=options)
    assert "ret_excess_ew" in out.columns
    assert out["ret_excess_ew"].to_list() == pytest.approx([0.05, 0.05])


def test_data_without_portfolio_return_columns_is_an_error():
    data = pl.DataFrame(
        {"date": [dt.date(2020, 1, 1)] * 2, "portfolio": [1, 2]}
    )
    with pytest.raises(ValueError, match="No portfolio return columns"):
        compute_long_short_returns(data)
