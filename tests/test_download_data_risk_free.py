"""Tests for download_data_risk_free."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import _download_data_risk_free  # noqa: E402


def test_invalid_frequency_aborts_with_informative_message():
    """Test invalid frequency aborts with informative message."""
    with pytest.raises(ValueError, match="monthly.*daily"):
        _download_data_risk_free(frequency="weekly")


def test_download_failure_is_caught_and_re_thrown():
    """Test download failure is caught and re-thrown."""
    with patch(
        "tidyfinance.data_download.pd.read_parquet",
        side_effect=Exception("connection refused"),
    ):
        with pytest.raises(
            RuntimeError, match="Failed to download risk-free rate data"
        ):
            _download_data_risk_free()


def test_full_dataset_returned_when_no_dates_are_supplied():
    """Test full dataset returned when no dates are supplied."""
    mock_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "risk_free": [0.001, 0.002],
        }
    )
    with patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=mock_data
    ):
        result = _download_data_risk_free()
    pd.testing.assert_frame_equal(result, mock_data)


def test_data_is_filtered_when_start_and_end_dates_are_supplied():
    """Test data is filtered when start and end dates are supplied."""
    mock_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01"]
            ),
            "risk_free": [0.001, 0.002, 0.003],
        }
    )
    with patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=mock_data
    ):
        result = _download_data_risk_free("2020-01-01", "2020-02-01")

    assert len(result) == 2
    assert list(result["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
