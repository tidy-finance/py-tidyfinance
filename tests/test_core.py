"""Test script for tidyfinance package."""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
from tidyfinance.core import add_lag_columns, estimate_rolling_betas


# Helper function to create test data
def create_test_data():
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=10, freq='ME')
    data = {
        'permno': np.repeat([1, 2], 10),
        'date': np.tile(dates, 2),
        'bm': np.random.uniform(0.5, 1.5, 20),
        'size': np.random.uniform(100, 200, 20)
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
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    permnos = [1, 2]
    data = pd.DataFrame({
        'date': np.tile(dates, len(permnos)),
        'permno': np.repeat(permnos, len(dates)),
        'ret_excess': np.random.randn(len(dates) * len(permnos)),
        'mkt_excess': np.random.randn(len(dates) * len(permnos)),
    })
    return data


def test_estimate_rolling_betas_basic(sample_data: pd.DataFrame) -> None:
    lookback = 30
    result = estimate_rolling_betas(sample_data, "ret_excess ~ mkt_excess",
                                    lookback)
    assert not result.empty, "Result should not be empty"
    assert 'mkt_excess' in result.columns, "Output should include beta estimate for mkt_excess"


def test_estimate_rolling_betas_min_obs(sample_data: pd.DataFrame) -> None:
    lookback = 30
    min_obs = 10
    result = estimate_rolling_betas(sample_data, "ret_excess ~ mkt_excess",
                                    lookback,
                                    min_obs=min_obs)
    assert result.shape[0] > 0, "Result should have valid estimates"
    assert result['mkt_excess'].isna().sum() > 0, "Some estimates should be NaN due to min_obs constraint"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
