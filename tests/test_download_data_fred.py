"""Tests for download_data_fred."""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import _download_data_fred  # noqa: E402


def test_downloads_parses_and_filters_fred_data():
    """Test downloads, parses, and filters FRED data."""
    body = "observation_date,GDP\n2020-01-01,1\n2020-02-01,2\n"
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = body
    response_mock.raise_for_status = MagicMock()

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response_mock
    ):
        result = _download_data_fred(
            "GDP", start_date="2020-02-01", end_date="2020-02-01"
        )

    assert isinstance(result, pd.DataFrame)
    assert result["date"].iloc[0] == pd.Timestamp("2020-02-01")
    assert result["value"].iloc[0] == 2
    assert result["series"].iloc[0] == "GDP"


def test_returns_empty_data_when_fred_responds_with_non_200_status():
    """Test returns empty data when FRED responds with non-200 status."""
    response_mock = MagicMock()
    response_mock.status_code = 404
    response_mock.raise_for_status.side_effect = Exception("404 error")

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response_mock
    ):
        with pytest.warns(
            UserWarning, match="Failed to retrieve data for series GDP"
        ):
            result = _download_data_fred("GDP")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["date", "series", "value"]


def test_returns_empty_data_when_download_handling_returns_none():
    """Test returns empty data when download handling returns NULL."""
    with patch(
        "tidyfinance.download_open_source.requests.get",
        side_effect=Exception("connection failed"),
    ):
        with pytest.warns(
            UserWarning, match="Failed to retrieve data for series GDP"
        ):
            result = _download_data_fred("GDP")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["date", "series", "value"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
