"""Test script for tidyfinance package."""

import pandas as pd
import numpy as np
import pytest
import sys
import os
import yaml

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from tidyfinance.tidyfinance import (add_lag_columns,
                                     download_data_factors,
                                     download_data_macro_predictors,
                                     download_data_fred,
                                     download_data_stock_prices,
                                     download_data_osap,
                                     create_wrds_dummy_database,
                                     download_data_wrds_compustat,
                                     _trim,
                                     _winsorize
                                     )


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


def test_download_data_factors_invalid_data_set():
    with pytest.raises(ValueError, match="Unsupported factor data type."):
        download_data_factors("invalid_data_set", start_date="2020-01-01",
                              end_date="2022-12-31")


def test_download_data_factors_ff_data_set():
    with pytest.raises(ValueError, match="Unsupported factor data type."):
        download_data_factors('factors_test', start_date="2020-01-01",
                              end_date="2022-12-01")


def test_download_data_factors_q_handles_broken_url():
    with pytest.raises(ValueError,
                       match=("Returning an empty data set due to download "
                              "failure."
                              )):
        download_data_factors("factors_q5_annual", start_date="2020-01-01",
                              end_date="2022-12-01", url="test")


def test_download_data_factors_q_handles_start_date_after_end_date():
    with pytest.raises(ValueError,
                       match="start_date cannot be after end_date"):
        download_data_factors("factors_q5_annual", start_date="2021-12-31",
                              end_date="2020-01-01")


def test_download_data_macro_predictors_invalid_type():
    with pytest.raises(ValueError, match="Unsupported macro predictor type."):
        download_data_macro_predictors("invalid_type")


def test_download_data_macro_predictors_invalid_url():
    df = download_data_macro_predictors("Monthly",
                                        sheet_id="invalid_sheet_id")
    assert df.empty, "Expected an empty DataFrame due to download failure."


def test_download_data_fred_empty():
    df = download_data_fred("INVALID_SERIES")
    assert df.empty


def test_download_data_fred_valid_structure():
    df = download_data_fred("GDP", "2020-01-01", "2020-12-31")
    assert set(df.columns) == {"date", "value", "series"}


def test_download_data_stock_prices_returns_dataframe():
    """Test that the function returns a DataFrame with correct columns."""
    df = download_data_stock_prices(["AAPL"], "2022-01-01", "2022-02-02")
    expected_columns = {"symbol", "date", "volume", "open", "low",
                        "high", "close", "adjusted_close"}
    assert isinstance(df, pd.DataFrame), "Function should return a DataFrame"
    assert not df.empty, "Returned DataFrame should not be empty"
    assert expected_columns.issubset(df.columns), "Missing expected columns"


def test_download_data_osap_returns_dataframe():
    """Test that the function returns a DataFrame with correct columns."""
    df = download_data_osap()
    assert isinstance(df, pd.DataFrame), "Function should return a DataFrame"
    assert not df.empty, "Returned DataFrame should not be empty"


def test_create_wrds_dummy_database(tmp_path):
    """Test that the function correctly downloads and saves a file."""
    test_path = tmp_path / "test_wrds_dummy.sqlite"
    create_wrds_dummy_database(str(test_path))
    assert test_path.exists(), "Database file was not created."
    assert test_path.stat().st_size > 0, "Database file is empty."


def test_create_wrds_dummy_database_invalid_path():
    """Test that the function raises an error when no path is given."""
    with pytest.raises(ValueError):
        create_wrds_dummy_database("")


def test_set_wrds_credentials(tmp_path):
    """Test function for set_wrds_credentials using a temporary directory."""
    test_config_path = tmp_path / "config.yaml"
    test_gitignore_path = tmp_path / ".gitignore"
    test_credentials = {
        "WRDS": {
            "USER": "test_user",
            "PASSWORD": "test_password"
        }
    }

    with open(test_config_path, "w") as file:
        yaml.safe_dump(test_credentials, file)
    assert test_config_path.exists()

    with open(test_config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    assert loaded_config["WRDS"]["USER"] == "test_user"
    assert loaded_config["WRDS"]["PASSWORD"] == "test_password"

    with open(test_gitignore_path, "w") as file:
        file.write("config.yaml\n")

    assert test_gitignore_path.exists()

    with open(test_gitignore_path, "r") as file:
        gitignore_content = file.readlines()

    assert "config.yaml\n" in gitignore_content


def test_winsorize_correct_adjustment():
    """Test that winsorize correctly adjusts extreme values."""
    np.random.seed(123)
    x = np.random.randn(100)
    cut = 0.05
    winsorized_x = _winsorize(x, cut)

    assert np.min(winsorized_x) == np.quantile(x, cut), "Lower bound not correctly applied"
    assert np.max(winsorized_x) == np.quantile(x, 1 - cut), "Upper bound not correctly applied"
    assert np.all(winsorized_x >= np.quantile(x, cut)), "Values below lower bound not adjusted"
    assert np.all(winsorized_x <= np.quantile(x, 1 - cut)), "Values above upper bound not adjusted"


def test_winsorize_handles_na():
    """Test that winsorize correctly handles NaN values."""
    x = np.array([np.nan, 1, 2, 3, 4, 5, np.nan])
    cut = 0.1
    winsorized_x = _winsorize(x, cut)

    assert len(winsorized_x) == len(x), "Output length should match input length"
    assert np.all(np.isnan(winsorized_x) == np.isnan(x)), "NaN values should remain unchanged"
    assert np.all(winsorized_x[~np.isnan(winsorized_x)] >= np.nanquantile(x, cut)), "Non-NaN values below lower bound not adjusted"
    assert np.all(winsorized_x[~np.isnan(winsorized_x)] <= np.nanquantile(x, 1 - cut)), "Non-NaN values above upper bound not adjusted"


def test_winsorize_edge_cases():
    """Test winsorize with edge cases (empty input and identical values)."""
    assert np.array_equal(_winsorize([], 0.1), np.array([])), ("Empty array should return empty array")
    x = np.full(10, 1.0)
    assert np.array_equal(_winsorize(x, 0.1), x), "Identical values should remain unchanged"


def test_trim_correct_removal():
    """Test that trim correctly removes extreme values."""
    np.random.seed(123)
    x = np.random.randn(100)
    cut = 0.05

    trimmed_x = _trim(x, cut)

    assert np.min(trimmed_x) >= np.quantile(x, cut), "Lower bound not correctly applied"
    assert np.max(trimmed_x) <= np.quantile(x, 1 - cut), "Upper bound not correctly applied"


def test_trim_handles_na():
    """Test that trim correctly handles NaN values."""
    x = np.array([np.nan, 1, 2, 3, 4, 5, np.nan])
    cut = 0.1

    trimmed_x = _trim(x, cut)

    assert not np.any(np.isnan(trimmed_x)), "NaN values should be removed"

def test_trim_edge_cases():
    """Test trim with edge cases such as empty input and identical values."""
    x = np.full(10, 1.0)
    assert np.array_equal(_trim(x, 0.1), x), "Identical values should remain unchanged"


def test_invalid_type_parameter():
    """Test that an invalid type parameter raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid type specified. Use 'compustat_annual' or 'compustat_quarterly'."):
        download_data_wrds_compustat(dataset_type="invalid_type")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
