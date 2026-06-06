"""Tests for download_data_osap."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance._internal import _transfrom_to_snake_case  # noqa: E402
from tidyfinance.data_download import _download_data_osap  # noqa: E402


def test_downloads_and_processes_all_rows():
    """Test downloads and processes all rows."""
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "LongName": [1, 2],
        }
    )
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        result = _download_data_osap(sheet_id="abc")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "long_name"]
    assert list(result["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]


def test_filters_rows_when_both_dates_are_supplied():
    """Test filters rows when both dates are supplied."""
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [1, 2, 3],
        }
    )
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        result = _download_data_osap(
            start_date="2020-02-01", end_date="2020-02-28"
        )

    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2020-02-01")
    assert result["value"].iloc[0] == 2


def test_returns_empty_dataframe_after_download_failure():
    """Test returns empty dataframe after download failure."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("download failure"),
    ):
        with pytest.warns(
            UserWarning, match="Returning an empty dataset"
        ):
            result = _download_data_osap()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_converts_names_to_snake_case():
    """Test converts names to snake case."""
    result = [
        _transfrom_to_snake_case(c)
        for c in ["LongName", "two words", "__Already___Odd__"]
    ]
    assert result == ["long_name", "two_words", "already_odd"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
