"""Test script for tidyfinance package."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from tidyfinance.core import (
    add_lag_columns,
    breakpoint_options,
    compute_breakpoints,
    # compute_long_short_returns,
    create_summary_statistics,
    estimate_betas,
    estimate_fama_macbeth,
)


# Helper function to create test data
def create_test_data():
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2023-01-01", periods=10, freq="ME")
    data = {
        "permno": np.repeat([1, 2], 10),
        "date": np.tile(dates, 2),
        "bm": np.random.uniform(0.5, 1.5, 20),
        "size": np.random.uniform(100, 200, 20),
    }
    return pd.DataFrame(data)


# Tests
def test_add_lagged_columns():
    """Test that lagged columns are added correctly"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm", "size"], lag=3, by="permno")

    # Check if lagged columns exist
    assert "bm_lag_3" in result.columns
    assert "size_lag_3" in result.columns

    # Check if the number of rows is preserved
    assert len(result) == len(data)


def test_negative_lag():
    """Test that negative lag raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lag_columns(data, cols=["bm", "size"], lag=-1)


def test_invalid_max_lag():
    """Test that max_lag < lag raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lag_columns(data, cols=["bm", "size"], lag=3, max_lag=1)


def test_without_grouping():
    """Test function works without grouping"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm", "size"], lag=3)

    assert "bm_lag_3" in result.columns
    assert "size_lag_3" in result.columns
    assert len(result) == len(data)


def test_preserve_original_values():
    """Test that original column values are preserved"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm", "size"], lag=3, by="permno")

    # Convert to lists for comparison
    assert result.get("bm").to_list() == data.get("bm").to_list()
    assert result.get("size").to_list() == data.get("size").to_list()


def test_lag_values_correctness():
    """Test that lag values are correct"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm"], lag=1, by="permno")

    # For each permno group, check if lag values are correct
    for permno in [1, 2]:
        group_data = result.query("permno == @permno").sort_values("date")
        orig_values = group_data.get("bm").to_list()
        lag_values = group_data.get("bm_lag_1").to_list()

        # Check if lagged values match original values shifted by 1
        assert lag_values[1:] == orig_values[:-1]
        assert np.isnan(lag_values[0])  # First value should be NaN


def test_multiple_lags():
    """Test that multiple lags work correctly"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm"], lag=1, max_lag=3, by="permno")

    # Check if all lag columns exist
    assert all(f"bm_lag_{i}" in result.columns for i in range(1, 4))


def test_invalid_column():
    """Test that invalid column names raise error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lag_columns(data, cols=["invalid_column"], lag=1)


def test_invalid_date_column():
    """Test that invalid date column raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lag_columns(data, cols=["bm"], lag=1, date_col="invalid_date")


@pytest.fixture
def sample_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    permnos = [1, 2]
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(permnos)),
            "permno": np.repeat(permnos, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "mkt_excess": np.random.randn(len(dates) * len(permnos)),
        }
    )
    return data


def test_estimate_rolling_betas_basic(sample_data: pd.DataFrame) -> None:
    lookback = 30
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)
    assert not result.empty, "Result should not be empty"
    assert "mkt_excess" in result.columns, (
        "Output should include beta estimate for mkt_excess"
    )


def test_estimate_rolling_betas_min_obs(sample_data: pd.DataFrame) -> None:
    lookback = 30
    min_obs = 10
    result = estimate_betas(
        sample_data, "ret_excess ~ mkt_excess", lookback, min_obs=min_obs
    )
    assert result.shape[0] > 0, "Result should have valid estimates"
    assert result["mkt_excess"].isna().sum() > 0, (
        "Some estimates should be NaN due to min_obs constraint"
    )


def sample_data_fmb() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=12, freq="ME")
    permnos = range(50)
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(permnos)),
            "permno": np.repeat(permnos, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "beta": np.random.randn(len(dates) * len(permnos)),
            "bm": np.random.randn(len(dates) * len(permnos)),
            "log_mktcap": np.random.randn(len(dates) * len(permnos)),
        }
    )
    return data


def test_estimate_fama_macbeth_basic(sample_data: pd.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap"
    )
    assert not result.empty, "Result should not be empty"
    assert "risk_premium" in result.columns, (
        "Output should include risk premia estimates"
    )


def test_estimate_fama_macbeth_vcov(sample_data: pd.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap", vcov="iid"
    )
    assert "t_statistic" in result.columns, (
        "Output should include t-statistics based on vcov choice"
    )


def sample_data_summary() -> pd.DataFrame:
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=100),
            "x": np.random.randn(100),
            "y": np.random.randint(0, 100, size=100),
            "z": np.random.randint(0, 100, size=100),
        }
    )
    return data


def test_create_summary_statistics_basic(sample_data) -> None:
    result = create_summary_statistics(sample_data_summary(), ["x", "y"])
    assert not result.empty, "Result should not be empty"
    assert "mean" in result.columns, "Output should include mean calculation"


def test_create_summary_statistics_by_group(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], by="group"
    )
    assert "group" in result.columns, "Output should include group column"
    assert "mean" in result.columns.get_level_values(1), (
        "Output should include mean calculation"
    )


def test_create_summary_statistics_detail(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], detail=True
    )
    assert "1%" in result.columns, (
        "Detailed statistics should include 1st percentile"
    )
    assert "99%" in result.columns, (
        "Detailed statistics should include 99th percentile"
    )


def sample_data_ls() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=10, freq="ME")
    portfolios = [1, 2]
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(portfolios)),
            "portfolio": np.repeat(portfolios, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(portfolios)),
        }
    )
    return data


# def test_compute_long_short_returns_basic(sample_data) -> None:
#     result = compute_long_short_returns(sample_data_ls())
#     assert not result.empty, "Result should not be empty"
#     assert "long_short_return" in result.columns, (
#         "Output should include long-short return calculation"
#     )


# def test_compute_long_short_returns_direction(sample_data) -> None:
#     result_top_bottom = compute_long_short_returns(
#         sample_data_ls(), direction="top_minus_bottom"
#     )
#     result_bottom_top = compute_long_short_returns(
#         sample_data_ls(), direction="bottom_minus_top"
#     )
#     assert (
#         result_top_bottom["long_short_return"]
#         == -result_bottom_top["long_short_return"]
#     ).all(), "Reversing direction should invert returns"


def test_breakpoint_options_default():
    options = breakpoint_options()
    assert options["smooth_bunching"] is False, (
        "Default smooth_bunching should be False"
    )


def test_breakpoint_options_custom():
    options = breakpoint_options(
        n_portfolios=5,
        percentiles=[0.2, 0.4, 0.6, 0.8],
        breakpoint_exchanges="NYSE",
    )
    assert options["n_portfolios"] == 5, "Custom n_portfolios should be 5"
    assert options["breakpoint_exchanges"] == "NYSE", (
        "Custom exchange should be 'NYSE'"
    )


def test_breakpoint_options_invalid():
    with pytest.raises(ValueError):
        breakpoint_options(n_portfolios=-1)  # Invalid n_portfolios
    with pytest.raises(ValueError):
        breakpoint_options(percentiles=[-0.1, 1.2])  # Invalid percentiles


def sample_data_breakpoints() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": np.arange(1, 101),
            "exchange": np.random.choice(["NYSE", "NASDAQ"], 100),
            "market_cap": np.random.uniform(100, 1000, 100),
        }
    )


def test_compute_breakpoints_n_portfolios(
    sample_data=sample_data_breakpoints(),
):
    breakpoints = compute_breakpoints(
        sample_data, "market_cap", {"n_portfolios": 5}
    )
    assert len(breakpoints) >= 2, (
        "Breakpoints should include at least min/max boundaries"
    )


def test_compute_breakpoints_percentiles(sample_data=sample_data_breakpoints()):
    breakpoints = compute_breakpoints(
        sample_data, "market_cap", {"percentiles": [0.2, 0.4, 0.6, 0.8]}
    )
    assert len(breakpoints) >= 2, (
        "Breakpoints should include at least min/max boundaries"
    )


def test_compute_breakpoints_invalid_options(
    sample_data=sample_data_breakpoints(),
):
    with pytest.raises(ValueError):
        compute_breakpoints(
            sample_data,
            "market_cap",
            {"n_portfolios": 5, "percentiles": [0.2, 0.4]},
        )
    with pytest.raises(ValueError):
        compute_breakpoints(sample_data, "market_cap", {})


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
