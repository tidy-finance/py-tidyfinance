"""Test script for tidyfinance package."""

import os
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    create_wrds_dummy_database,
    _download_data_constituents,
    _download_data_factors_ff,
    # _download_data_factors_q,
    _download_data_fred,
    _download_data_macro_predictors,
    _download_data_osap,
    _download_data_stock_prices,
    _download_data_wrds_compustat,
    _download_data_huggingface,
    _download_factor_library_ids,
    _filter_factor_library_grid,
    download_data,
)  # noqa: E402


def test_download_data_factors_invalid_data_set():
    with pytest.raises(ValueError, match="Unsupported domain."):
        download_data(
            domain="invalid_data_set",
            dataset="invalid",
            start_date="2020-01-01",
            end_date="2022-12-31",
        )


def test_download_data_factors_ff_data_set():
    with pytest.raises(ValueError, match="Unsupported dataset."):
        download_data(
            domain="factors_ff",
            dataset="factors_test",
            start_date="2020-01-01",
            end_date="2022-12-01",
        )


def test_download_data_column_ordering():
    df = download_data(
        domain="stock_prices",
        symbols="AAPL",
        start_date="2000-01-01",
        end_date="2023-12-31"
    )
    expected_columns = ['symbol', 'date', 'volume', 'open', 'low', 'high',
                        'close', 'adjusted_close']
    assert list(df.columns) == expected_columns, ("Expected columns "
                                                  f"{expected_columns}, but "
                                                  f"got {list(df.columns)}"
                                                  )


def test_download_data_factors_q_handles_broken_url():
    with pytest.raises(
        ValueError,
        match=("No matching dataset found."),
    ):
        download_data(
            domain="factors_q",
            dataset="test",
            start_date="2020-01-01",
            end_date="2022-12-01",
            url="test",
        )


def test_download_data_factors_q_handles_start_date_after_end_date():
    with pytest.raises(ValueError,
                       match="start_date cannot be after end_date"):
        download_data(
            domain="factors_q",
            dataset="factors_q5_annual",
            start_date="2021-12-31",
            end_date="2020-01-01",
        )


def test_download_data_macro_predictors_invalid_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset."):
        _download_data_macro_predictors("invalid_dataset")


def test_download_data_macro_predictors_invalid_url():
    df = _download_data_macro_predictors("monthly",
                                         sheet_id="invalid_sheet_id")
    assert df.empty, "Expected an empty DataFrame due to download failure."


def test_download_data_fred_empty():
    df = _download_data_fred("INVALID_SERIES")
    assert df.empty


def test_download_data_fred_valid_structure():
    df = _download_data_fred("GDP", "2020-01-01", "2020-12-31")
    assert set(df.columns) == {"date", "value", "series"}


def test_download_data_stock_prices_returns_dataframe():
    """Test that the function returns a DataFrame with correct columns."""
    df = _download_data_stock_prices(["AAPL"], "2022-01-01", "2022-02-02")
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
    df = _download_data_osap()
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


def test_invalid_dataset_parameter():
    """Test that an invalid dataset parameter raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=("Invalid dataset specified. Use 'compustat_annual' "
               "or 'compustat_quarterly'."),
    ):
        _download_data_wrds_compustat(dataset="invalid")


def test_download_data_constituents_invalid_index():
    """Test that an invalid index raises a ValueError."""
    with pytest.raises(ValueError):
        _download_data_constituents("INVALID_INDEX")


def test_download_data_constituents_valid_index():
    """Test that valid index works."""
    df = _download_data_constituents("DAX")
    assert isinstance(df, pd.DataFrame), "Function should return a DataFrame"
    assert not df.empty, "Returned DataFrame should not be empty"


def test_download_data_factors_ff_valid():
    df = _download_data_factors_ff("F-F_Research_Data_5_Factors_2x3_daily")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_breakpoints_valid():
    df = _download_data_factors_ff(dataset="ME_Breakpoints",
                                   start_date='2010-02-01',
                                   end_date='2012-02-01')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_macro_predictors_valid():
    df = _download_data_macro_predictors("monthly")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_osap_valid():
    df = _download_data_osap()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_fred_valid():
    df = _download_data_fred(
        "GDP", start_date="2020-01-01", end_date="2020-12-31"
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_stock_prices_valid():
    df = _download_data_stock_prices(
        "AAPL", start_date="2020-01-01", end_date="2020-12-31"
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_huggingface_missing_dataset():
    with pytest.raises(ValueError, match="'dataset' is required"):
        _download_data_huggingface(dataset=None)


def test_download_data_huggingface_unsupported_dataset():
    with pytest.raises(ValueError, match="Unsupported Hugging Face dataset"):
        _download_data_huggingface(dataset="not_a_dataset")


def test_filter_factor_library_grid_unsupported_filter():
    with pytest.raises(ValueError, match="Unsupported filter name"):
        _filter_factor_library_grid(not_a_column="x")


def test_download_factor_library_ids_empty_ids():
    with pytest.raises(ValueError, match="No portfolio IDs provided"):
        _download_factor_library_ids([])


def test_download_data_huggingface_factor_library_valid():
    df = _download_data_huggingface(
        dataset="factor_library", sorting_variable="me"
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "id" in df.columns


def test_download_data_huggingface_high_frequency_sp500_valid():
    df = _download_data_huggingface(
        dataset="high_frequency_sp500",
        start_date="2007-07-26",
        end_date="2007-07-27",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_huggingface_high_frequency_sp500_empty_date_range():
    df = _download_data_huggingface(
        dataset="high_frequency_sp500",
        start_date="2000-01-01",
        end_date="2000-01-02",
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_download_data_tidyfinance_domain():
    df = download_data(
        domain="tidyfinance",
        dataset="high_frequency_sp500",
        start_date="2007-07-26",
        end_date="2007-07-27",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
