"""Tests for compute_long_short_returns."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import compute_long_short_returns  # noqa: E402


def make_portfolio_panel(n_portfolios=5, n_months=12, seed=42):
    """Construct a portfolio-return panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_months, freq="MS")
    rows = [(p, d) for d in dates for p in range(1, n_portfolios + 1)]
    df = pd.DataFrame(rows, columns=["portfolio", "date"])
    n = len(df)
    df["ret_excess_ew"] = rng.normal(0, 0.05, n)
    df["ret_excess_vw"] = rng.normal(0, 0.05, n)
    return df


def test_compute_long_short_returns_returns_top_minus_bottom_by_default():
    """Test compute_long_short_returns returns top - bottom by default."""
    panel = make_portfolio_panel()
    out = compute_long_short_returns(panel)
    assert list(out.columns) == ["date", "ret_excess_ew", "ret_excess_vw"]
    assert len(out) == 12

    # Sanity-check one date
    d = panel["date"].iloc[0]
    rows = panel[panel["date"] == d]
    expected_ew = (
        rows.loc[
            rows["portfolio"] == rows["portfolio"].max(), "ret_excess_ew"
        ].iloc[0]
        - rows.loc[
            rows["portfolio"] == rows["portfolio"].min(), "ret_excess_ew"
        ].iloc[0]
    )
    actual_ew = out.loc[out["date"] == d, "ret_excess_ew"].iloc[0]
    assert abs(actual_ew - expected_ew) < 1e-12


def test_compute_long_short_returns_returns_na_when_only_one_portfolio():
    """Test compute_long_short_returns returns NaN when only one portfolio."""
    panel = make_portfolio_panel(n_portfolios=1)
    out = compute_long_short_returns(panel)
    assert list(out.columns) == ["date", "ret_excess_ew", "ret_excess_vw"]
    assert len(out) == 12
    assert out["ret_excess_ew"].isna().all()
    assert out["ret_excess_vw"].isna().all()


def test_compute_long_short_returns_handles_a_single_per_date_long_leg():
    """Test compute_long_short_returns handles a single per-date long leg."""
    panel = make_portfolio_panel(n_portfolios=2, n_months=4)
    first_date = panel["date"].iloc[0]
    panel = panel[~((panel["date"] == first_date) & (panel["portfolio"] == 2))]
    out = compute_long_short_returns(panel)
    assert len(out) == 4
    assert pd.isna(out.loc[out["date"] == first_date, "ret_excess_ew"].iloc[0])
    other_dates = out[out["date"] != first_date]
    assert other_dates["ret_excess_ew"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__])
