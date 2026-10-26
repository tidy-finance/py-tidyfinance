"""Tests for download_data_risk_free."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_tidy_finance import (  # noqa: E402
    download_data_risk_free,
)


def test_invalid_frequency_aborts_with_informative_message():
    """Test invalid frequency aborts with informative message."""
    with pytest.raises(ValueError, match="monthly.*daily"):
        download_data_risk_free(frequency="weekly")


def test_download_failure_is_caught_and_re_thrown():
    """Test download failure is caught and re-thrown."""
    with patch(
        "tidyfinance.download_tidy_finance._read_parquet_url",
        side_effect=Exception("connection refused"),
    ):
        with pytest.raises(
            RuntimeError, match="Failed to download risk-free rate data"
        ):
            download_data_risk_free()


def test_full_dataset_returned_when_no_dates_are_supplied():
    """Test full dataset returned when no dates are supplied."""
    mock_data = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "risk_free": [0.001, 0.002],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._read_parquet_url",
        return_value=mock_data,
    ):
        result = download_data_risk_free()
    assert_frame_equal(result, mock_data)


def test_data_is_filtered_when_start_and_end_dates_are_supplied():
    """Test data is filtered when start and end dates are supplied."""
    mock_data = pl.DataFrame(
        {
            "date": [
                dt.date(2020, 1, 1),
                dt.date(2020, 2, 1),
                dt.date(2020, 3, 1),
            ],
            "risk_free": [0.001, 0.002, 0.003],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._read_parquet_url",
        return_value=mock_data,
    ):
        result = download_data_risk_free("2020-01-01", "2020-02-01")

    assert result.height == 2
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]


def test_data_is_filtered_with_open_ended_date_ranges():
    """A single date bound filters that side only instead of silently
    returning an empty frame."""
    mock_data = pl.DataFrame(
        {
            "date": [
                dt.date(2020, 1, 1),
                dt.date(2020, 2, 1),
                dt.date(2020, 3, 1),
            ],
            "risk_free": [0.001, 0.002, 0.003],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._read_parquet_url",
        return_value=mock_data,
    ):
        from_feb = download_data_risk_free(start_date="2020-02-01")
        until_feb = download_data_risk_free(end_date="2020-02-01")

    assert from_feb["date"].to_list() == [
        dt.date(2020, 2, 1),
        dt.date(2020, 3, 1),
    ]
    assert until_feb["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]


def test_datetime_date_column_is_cast_to_date():
    """Test a datetime-typed date column is normalized to pl.Date."""
    mock_data = pl.DataFrame(
        {
            "date": [
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 2, 1),
            ],
            "risk_free": [0.001, 0.002],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._read_parquet_url",
        return_value=mock_data,
    ):
        result = download_data_risk_free()
    assert result.schema["date"] == pl.Date


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
