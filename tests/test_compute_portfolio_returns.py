"""Tests for compute_portfolio_returns."""

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
    compute_portfolio_returns,
)


# %% test data helper


def make_data(n_stocks=30, n_months=6, start="2020-01-01", seed=42, with_mktcap=True):
    """Construct a stock-month panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    df = pd.DataFrame(
        [
            (permno, date)
            for date in dates
            for permno in range(1, n_stocks + 1)
        ],
        columns=["permno", "date"],
    )
    n = len(df)
    df["ret_excess"] = rng.standard_normal(n)
    df["size"] = rng.uniform(50, 150, n)
    df["bm"] = rng.uniform(0.5, 2, n)
    if with_mktcap:
        df["mktcap_lag"] = rng.uniform(100, 1000, n)
    return df


bp3 = breakpoint_options(n_portfolios=3)
bp2 = breakpoint_options(n_portfolios=2)


# %% argument validation


def test_quiet_must_be_a_single_logical():
    """Test quiet must be a single boolean."""
    with pytest.raises(ValueError, match="quiet"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            quiet="yes",
        )


def test_sorting_variables_cannot_be_empty():
    """Test sorting_variables cannot be empty."""
    with pytest.raises(ValueError, match="sorting variable"):
        compute_portfolio_returns(
            make_data(),
            [],
            "univariate",
            breakpoint_options_main=bp3,
        )


def test_invalid_sorting_method_raises_error():
    """Test invalid sorting_method raises error."""
    with pytest.raises(ValueError, match="sorting method"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bad-method",
            breakpoint_options_main=bp3,
        )


def test_bivariate_without_secondary_breakpoints_warns():
    """Test bivariate without secondary breakpoints emits warning."""
    with pytest.warns(UserWarning, match="breakpoint_options_secondary"):
        try:
            compute_portfolio_returns(
                make_data(),
                ["size", "bm"],
                "bivariate-dependent",
                breakpoint_options_main=bp3,
                quiet=True,
            )
        except Exception:
            pass


def test_cap_weight_outside_0_1_raises_error():
    """Test cap_weight outside [0, 1] raises error."""
    with pytest.raises(ValueError, match="cap_weight"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            cap_weight=2,
        )


def test_missing_column_raises_error():
    """Test missing column raises error."""
    with pytest.raises(ValueError, match="Missing columns"):
        compute_portfolio_returns(
            make_data(),
            "no_such_col",
            "univariate",
            breakpoint_options_main=bp3,
        )


def test_rebalancing_month_outside_1_12_raises_error():
    """Test rebalancing_month outside 1-12 raises error."""
    with pytest.raises(ValueError, match="rebalancing_month"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            rebalancing_month=13,
            breakpoint_options_main=bp3,
        )


# %% univariate periodic


def test_univariate_periodic_without_mktcap_returns_ew_only():
    """Test univariate periodic without mktcap returns ew only."""
    result = compute_portfolio_returns(
        make_data(with_mktcap=False),
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert list(result.columns) == ["portfolio", "date", "ret_excess_ew"]
    assert len(result) == 3 * 6


def test_univariate_periodic_with_mktcap_returns_all_ret_cols():
    """Test univariate periodic with mktcap returns all return columns."""
    result = compute_portfolio_returns(
        make_data(),
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert list(result.columns) == [
        "portfolio",
        "date",
        "ret_excess_vw",
        "ret_excess_ew",
        "ret_excess_vw_capped",
    ]
    assert len(result) == 3 * 6


def test_all_na_mktcap_lag_gives_na_value_weighted_returns():
    """Test all-NaN mktcap_lag gives NaN value-weighted returns."""
    data = make_data()
    data["mktcap_lag"] = np.nan
    result = compute_portfolio_returns(
        data,
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert result["ret_excess_vw"].isna().all()


def test_all_na_sorting_var_emits_message_and_returns_0_rows():
    """Test all-NaN sorting variable emits warning and returns 0 rows."""
    data = make_data()
    data["size"] = np.nan
    with pytest.warns(UserWarning, match="empty panel"):
        result = compute_portfolio_returns(
            data,
            "size",
            "univariate",
            breakpoint_options_main=bp3,
        )
    assert len(result) == 0


def test_univariate_sort_with_two_variables_raises_error():
    """Test univariate sort with two variables raises error."""
    with pytest.raises(ValueError, match="one sorting variable"):
        compute_portfolio_returns(
            make_data(),
            ["size", "bm"],
            "univariate",
            breakpoint_options_main=bp3,
        )


# %% univariate annual


def test_univariate_annual_no_data_in_month_raises_early_error():
    """Test univariate annual no data in month raises early error."""
    with pytest.raises(ValueError, match="No observations match"):
        compute_portfolio_returns(
            make_data(n_months=6, start="2020-01-01"),
            "size",
            "univariate",
            rebalancing_month=12,
            breakpoint_options_main=bp3,
        )


def test_univariate_annual_rebalancing_produces_valid_output():
    """Test univariate annual rebalancing produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        "size",
        "univariate",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_min_portfolio_size_produces_nas_and_emits_message():
    """Test min_portfolio_size produces NaNs and emits warning."""
    with pytest.warns(UserWarning, match="missing"):
        result = compute_portfolio_returns(
            make_data(n_stocks=5),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            min_portfolio_size=10,
        )
    assert result["ret_excess_ew"].isna().any()


# %% bivariate-dependent


def test_bivariate_dependent_with_one_variable_raises_error():
    """Test bivariate-dependent with one variable raises error."""
    with pytest.raises(ValueError, match="two sorting variables"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bivariate-dependent",
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
        )


def test_bivariate_dependent_periodic_produces_valid_output():
    """Test bivariate-dependent periodic produces valid output."""
    result = compute_portfolio_returns(
        make_data(),
        ["size", "bm"],
        "bivariate-dependent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_bivariate_dependent_annual_produces_valid_output():
    """Test bivariate-dependent annual produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        ["size", "bm"],
        "bivariate-dependent",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_bivariate_dependent_annual_no_match_raises_early_error():
    """Test bivariate-dependent annual no-match raises early error."""
    with pytest.raises(ValueError, match="No observations match"):
        compute_portfolio_returns(
            make_data(n_months=6, start="2020-01-01"),
            ["size", "bm"],
            "bivariate-dependent",
            rebalancing_month=12,
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
            quiet=True,
        )


def test_all_na_mktcap_in_bivariate_coerces_nan_vw_to_na():
    """Test all-NaN mktcap in bivariate produces NaN vw returns."""
    data = make_data()
    data["mktcap_lag"] = np.nan
    result = compute_portfolio_returns(
        data,
        ["size", "bm"],
        "bivariate-dependent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert result["ret_excess_vw"].isna().all()


# %% bivariate-independent


def test_bivariate_independent_with_one_variable_raises_error():
    """Test bivariate-independent with one variable raises error."""
    with pytest.raises(ValueError, match="two sorting variables"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bivariate-independent",
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
        )


def test_bivariate_independent_periodic_produces_valid_output():
    """Test bivariate-independent periodic produces valid output."""
    result = compute_portfolio_returns(
        make_data(),
        ["size", "bm"],
        "bivariate-independent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_bivariate_independent_annual_produces_valid_output():
    """Test bivariate-independent annual produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        ["size", "bm"],
        "bivariate-independent",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
