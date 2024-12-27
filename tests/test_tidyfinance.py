"""Test script for tidyfinance package."""

import polars as pl
import pandas as pd
import numpy as np
import pytest
from tidyfinance import add_lag_columns


# Helper function to create test data
def create_test_data():
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=10, freq='M')
    data = {
        'permno': np.repeat([1, 2], 10),
        'date': np.tile(dates, 2),
        'bm': np.random.uniform(0.5, 1.5, 20),
        'size': np.random.uniform(100, 200, 20)
    }
    return pl.DataFrame(data)


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
    assert result.get_column("bm").to_list() == data.get_column("bm").to_list()
    assert result.get_column("size").to_list() == data.get_column("size").to_list()


def test_lag_values_correctness():
    """Test that lag values are correct"""
    data = create_test_data()
    result = add_lag_columns(data, cols=["bm"], lag=1, by="permno")

    # For each permno group, check if lag values are correct
    for permno in [1, 2]:
        group_data = result.filter(pl.col("permno") == permno).sort("date")
        orig_values = group_data.get_column("bm").to_list()
        lag_values = group_data.get_column("bm_lag_1").to_list()

        # Check if lagged values match original values shifted by 1
        assert lag_values[1:] == orig_values[:-1]
        assert lag_values[0] is None  # First value should be None


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


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
