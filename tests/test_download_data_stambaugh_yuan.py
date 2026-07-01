"""Tests for download_data_stambaugh_yuan."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import (  # noqa: E402
    _download_data_stambaugh_yuan,
)


def test_downloads_and_processes_monthly_mispricing_factors():
    raw = pd.DataFrame(
        {
            "YYYYMM": [196301, 196302],
            "MKTRF": [0.0493, -0.0238],
            "SMB": [0.0248, 0.0181],
            "MGMT": [0.0213, 0.0110],
            "PERF": [-0.0228, 0.0009],
            "RF": [0.0025, 0.0023],
        }
    )
    with patch(
        "tidyfinance.download_open_source.pd.read_csv", return_value=raw
    ):
        result = _download_data_stambaugh_yuan()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "date",
        "mkt_excess",
        "smb",
        "mgmt",
        "perf",
        "risk_free",
    ]
    # Dates are aligned to the beginning of the month.
    assert list(result["date"]) == [
        pd.Timestamp("1963-01-01"),
        pd.Timestamp("1963-02-01"),
    ]
    # Returns are already decimal and must not be rescaled.
    assert list(result["mkt_excess"]) == [0.0493, -0.0238]
    assert list(result["risk_free"]) == [0.0025, 0.0023]


def test_parses_daily_dates_from_the_date_column():
    captured_url = {}

    def fake_read_csv(url, *args, **kwargs):
        captured_url["url"] = url
        return pd.DataFrame(
            {
                "DATE": [19630102, 19630103],
                "MKTRF": [-0.0054, 0.0166],
                "SMB": [0.0094, 0.0053],
                "MGMT": [0.0062, 0.0050],
                "PERF": [-0.0073, -0.0215],
                "RF": [0.00011, 0.00011],
            }
        )

    with patch(
        "tidyfinance.download_open_source.pd.read_csv",
        side_effect=fake_read_csv,
    ):
        result = _download_data_stambaugh_yuan(dataset="daily")

    assert captured_url["url"].endswith("M4d.csv")
    assert list(result["date"]) == [
        pd.Timestamp("1963-01-02"),
        pd.Timestamp("1963-01-03"),
    ]


def test_filters_rows_when_both_dates_are_supplied():
    raw = pd.DataFrame(
        {
            "YYYYMM": [196301, 196302, 196303],
            "MKTRF": [1, 2, 3],
            "SMB": [1, 2, 3],
            "MGMT": [1, 2, 3],
            "PERF": [1, 2, 3],
            "RF": [1, 2, 3],
        }
    )
    with patch(
        "tidyfinance.download_open_source.pd.read_csv", return_value=raw
    ):
        result = _download_data_stambaugh_yuan(
            start_date="1963-02-01", end_date="1963-02-28"
        )

    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("1963-02-01")


def test_warns_when_the_requested_range_is_outside_the_available_data():
    raw = pd.DataFrame(
        {
            "YYYYMM": [196301, 196302],
            "MKTRF": [1, 2],
            "SMB": [1, 2],
            "MGMT": [1, 2],
            "PERF": [1, 2],
            "RF": [1, 2],
        }
    )
    with patch(
        "tidyfinance.download_open_source.pd.read_csv", return_value=raw
    ):
        with pytest.warns(UserWarning, match="outside the available"):
            result = _download_data_stambaugh_yuan(
                start_date="2020-01-01", end_date="2020-12-31"
            )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_aborts_on_unsupported_dataset():
    with pytest.raises(ValueError, match="dataset"):
        _download_data_stambaugh_yuan(dataset="weekly")


def test_returns_empty_dataframe_after_download_failure():
    with patch(
        "tidyfinance.download_open_source.pd.read_csv",
        side_effect=Exception("download failure"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = _download_data_stambaugh_yuan()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
