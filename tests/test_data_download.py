"""Test script for tidyfinance package."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch
import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    create_wrds_dummy_database,
    _download_data_constituents,
    _download_data_factors_ff,
    _download_data_fred,
    _download_data_macro_predictors,
    _download_data_osap,
    _download_data_risk_free,
    _download_data_stock_prices,
    _download_data_wrds_compustat,
    _download_data_wrds_crsp,
    _download_data_factors_q,
    download_data
)  # noqa: E402

from tidyfinance._internal import _parse_date, _validate_dates  # noqa: E402


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
        match="Unsupported dataset",
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


def test_download_data_factors_q_valid():
    df = _download_data_factors_q("q5_factors_monthly")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_download_data_constituents_dataset_kwarg_warns():
    with pytest.warns(UserWarning, match="'dataset' argument is not valid"):
        download_data(domain="constituents", dataset="DAX")


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


def test_download_data_wrds_crsp_v1_monthly_not_implemented():
    with patch("tidyfinance.data_download.get_wrds_connection"):
        with pytest.raises(NotImplementedError):
            _download_data_wrds_crsp(dataset="crsp_monthly", version="v1")


def test_download_data_wrds_crsp_v1_daily_not_implemented():
    with patch("tidyfinance.data_download.get_wrds_connection"):
        with pytest.raises(NotImplementedError):
            _download_data_wrds_crsp(dataset="crsp_daily", version="v1")


def test_download_data_wrds_crsp_adjust_volume_wrong_dataset():
    with pytest.raises(ValueError, match="adjust_volume is only supported"):
        _download_data_wrds_crsp(dataset="crsp_monthly", adjust_volume=True)


def test_download_data_wrds_crsp_adjust_volume_missing_columns():
    with pytest.raises(ValueError, match="dlyprc, dlyvol"):
        _download_data_wrds_crsp(dataset="crsp_daily", adjust_volume=True)


def test_download_data_wrds_crsp_monthly_no_prc_column():
    crsp_df = pd.DataFrame({
        "permno": [10001, 10001],
        "date": pd.to_datetime(["2019-12-01", "2020-01-01"]),
        "calculation_date": pd.to_datetime(["2019-12-31", "2020-01-31"]),
        "ret": [0.02, 0.01],
        "shrout": [1000, 1000],
        "prc": [49.0, 50.0],
        "primaryexch": ["N", "N"],
        "siccd": [3990, 3990],
        "first_crsp_date": pd.to_datetime(["2018-06-15", "2018-06-15"]),
    })
    factors_df = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-01", "2020-01-01"]),
        "mkt_excess": [0.003, 0.005],
        "smb": [0.001, 0.001],
        "hml": [0.002, 0.002],
        "risk_free": [0.0002, 0.0002],
    })
    with patch("tidyfinance.data_download.get_wrds_connection"):
        with patch("pandas.read_sql_query", return_value=crsp_df):
            with patch(
                "tidyfinance.data_download._download_data_factors_ff",
                return_value=factors_df,
            ):
                result = _download_data_wrds_crsp(
                    dataset="crsp_monthly",
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "altprc" not in result.columns


def test_download_data_risk_free_invalid_frequency():
    """Reject any frequency other than 'monthly' or 'daily'."""
    with pytest.raises(ValueError, match="frequency must be"):
        _download_data_risk_free(frequency="weekly")


def test_download_data_risk_free_monthly_url_and_shape():
    """Use the monthly parquet URL and return the expected columns."""
    mock_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "risk_free": [0.001, 0.0011, 0.0012],
    })
    with patch("pandas.read_parquet", return_value=mock_df) as mock_read:
        result = _download_data_risk_free(frequency="monthly")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "risk_free"]
    assert len(result) == 3

    called_url = mock_read.call_args[0][0]
    assert "tidy-finance/risk-free" in called_url
    assert called_url.endswith("risk_free_monthly.parquet")


def test_download_data_risk_free_date_filter():
    """start_date/end_date filter rows inclusively."""
    mock_df = pd.DataFrame({
        "date": pd.to_datetime([
            "2019-12-01", "2020-01-01", "2020-02-01",
            "2020-06-30", "2020-12-01",
        ]),
        "risk_free": [0.0009, 0.0010, 0.0011, 0.0013, 0.0015],
    })
    with patch("pandas.read_parquet", return_value=mock_df):
        result = _download_data_risk_free(
            start_date="2020-01-01",
            end_date="2020-06-30",
            frequency="monthly",
        )

    assert len(result) == 3
    assert result["date"].min() == pd.Timestamp("2020-01-01")
    assert result["date"].max() == pd.Timestamp("2020-06-30")


def test_download_data_risk_free_download_failure():
    """Wrap parquet download failures in a RuntimeError with URL info."""
    with patch(
        "pandas.read_parquet",
        side_effect=OSError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Failed to download"):
            _download_data_risk_free(frequency="monthly")


def test_parse_date_yyyy_mm_dd_returns_timestamp():
    """YYYY-MM-DD input must return a pd.Timestamp (not datetime.date)."""
    result = _parse_date("2020-01-15")
    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp("2020-01-15")


def test_parse_date_yyyymm_returns_timestamp():
    """YYYYMM input returns a pd.Timestamp at month-start (or month-end)."""
    start = _parse_date("202001", is_end=False)
    end = _parse_date("202001", is_end=True)
    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)
    assert start == pd.Timestamp("2020-01-01")
    assert end == pd.Timestamp("2020-01-31")


def test_parse_date_returns_same_type_for_both_formats():
    """Both date formats must return the same type (consistency guard)."""
    assert type(_parse_date("2020-01-15")) is type(_parse_date("202001"))


def test_validate_dates_returns_timestamps():
    """_validate_dates must propagate _parse_date's Timestamp return type."""
    start, end = _validate_dates("2020-01-01", "2020-12-31")
    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)


def test_validate_dates_returned_values_compare_against_datetime_series():
    """
    Regression guard: the returned dates must compare cleanly against a
    datetime Series. This was previously broken because
    _parse_date returned datetime.date, which pandas refuses to compare
    against datetime64[ns].
    """
    series = pd.to_datetime([
        "2019-12-01", "2020-01-15", "2020-06-30", "2021-01-01"
    ])
    start, end = _validate_dates("2020-01-01", "2020-12-31")
    mask = (series >= start) & (series <= end)
    assert mask.tolist() == [False, True, True, False]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
