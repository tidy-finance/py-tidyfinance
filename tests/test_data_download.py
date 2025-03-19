"""Test script for tidyfinance package."""

import pandas as pd
import pytest
import sys
import os
import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    download_data_factors,
    download_data_macro_predictors,
    download_data_fred,
    download_data_stock_prices,
    download_data_osap,
    download_data_constituents,
    list_supported_indexes,
    create_wrds_dummy_database,
    download_data_wrds_compustat,
)


def test_download_data_factors_invalid_data_set():
    with pytest.raises(ValueError, match="Unsupported factor data type."):
        download_data_factors(
            "invalid_data_set", start_date="2020-01-01", end_date="2022-12-31"
        )


def test_download_data_factors_ff_data_set():
    with pytest.raises(ValueError, match="Unsupported factor data type."):
        download_data_factors(
            "factors_test", start_date="2020-01-01", end_date="2022-12-01"
        )


def test_download_data_factors_q_handles_broken_url():
    with pytest.raises(
        ValueError,
        match=("Returning an empty data set due to download failure."),
    ):
        download_data_factors(
            "factors_q5_annual",
            start_date="2020-01-01",
            end_date="2022-12-01",
            url="test",
        )


def test_download_data_factors_q_handles_start_date_after_end_date():
    with pytest.raises(ValueError, match="start_date cannot be after end_date"):
        download_data_factors(
            "factors_q5_annual", start_date="2021-12-31", end_date="2020-01-01"
        )


def test_download_data_macro_predictors_invalid_type():
    with pytest.raises(ValueError, match="Unsupported macro predictor type."):
        download_data_macro_predictors("invalid_type")


def test_download_data_macro_predictors_invalid_url():
    df = download_data_macro_predictors("Monthly", sheet_id="invalid_sheet_id")
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
    expected_columns = {
        "symbol",
        "date",
        "volume",
        "open",
        "low",
        "high",
        "close",
        "adjusted_close",
    }
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
        "WRDS": {"USER": "test_user", "PASSWORD": "test_password"}
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


def test_invalid_type_parameter():
    """Test that an invalid type parameter raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="Invalid type specified. Use 'compustat_annual' or 'compustat_quarterly'.",
    ):
        download_data_wrds_compustat(dataset_type="invalid_type")


def test_list_supported_indexes():
    """Test that the function returns a DataFrame with the expected columns."""
    df = list_supported_indexes()
    assert isinstance(df, pd.DataFrame)
    assert "index" in df.columns
    assert "url" in df.columns
    assert "skip" in df.columns


def test_download_data_constituents_invalid_index():
    """Test that an invalid index raises a ValueError."""
    with pytest.raises(ValueError):
        download_data_constituents("INVALID_INDEX")


def test_download_data_constituents_valid_index():
    """Test that valid index works."""
    df = download_data_constituents("DAX")
    assert isinstance(df, pd.DataFrame), "Function should return a DataFrame"
    assert not df.empty, "Returned DataFrame should not be empty"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
