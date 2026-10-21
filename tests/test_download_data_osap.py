"""Tests for download_data_osap."""

import datetime as dt
import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance._internal import _transfrom_to_snake_case  # noqa: E402
from tidyfinance.download_open_source import _download_data_osap  # noqa: E402


def test_downloads_and_processes_all_rows():
    """Test downloads and processes all rows."""
    raw = pl.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29"],
            "LongName": [1, 2],
        }
    )
    with patch("tidyfinance.download_open_source._fetch_google_drive_csv", return_value=raw):
        result = _download_data_osap(sheet_id="abc")

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "long_name"]
    # Dates are aligned to the beginning of the month.
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]
    # Percentage returns are scaled to numeric (decimal) values.
    assert result["long_name"].to_list() == [0.01, 0.02]


def test_filters_rows_when_both_dates_are_supplied():
    """Test filters rows when both dates are supplied."""
    raw = pl.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "value": [1, 2, 3],
        }
    )
    with patch("tidyfinance.download_open_source._fetch_google_drive_csv", return_value=raw):
        result = _download_data_osap(
            start_date="2020-02-01", end_date="2020-02-28"
        )

    assert len(result) == 1
    assert result["date"][0] == dt.date(2020, 2, 1)
    assert result["value"][0] == 0.02


def test_returns_empty_dataframe_after_download_failure():
    """Test returns empty dataframe after download failure."""
    with patch(
        "tidyfinance.download_open_source._fetch_google_drive_csv",
        side_effect=Exception("download failure"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = _download_data_osap()

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


def _html_response(content_type, body):
    """Mock a Google Drive response that is an HTML page."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": content_type}
    response.text = body
    response.content = body.encode("utf-8")
    return response


def test_reports_an_exceeded_google_drive_quota():
    """Test the quota page is reported and not parsed as CSV."""
    response = _html_response(
        "text/html; charset=utf-8",
        "<html><head><title>Google Drive - Quota exceeded</title>"
        "</head></html>",
    )
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=response,
    ):
        with pytest.warns(UserWarning) as record:
            result = _download_data_osap()

    messages = [str(w.message) for w in record]
    assert any("download quota of the file is exceeded" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert result.shape == (0, 0)


def test_rejects_other_html_pages_instead_of_parsing_them():
    """Test other HTML pages are rejected instead of parsed as CSV."""
    response = _html_response(
        "text/html", "<html><body>Sign in</body></html>"
    )
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=response,
    ):
        with pytest.warns(UserWarning) as record:
            result = _download_data_osap()

    messages = [str(w.message) for w in record]
    assert any("HTML page instead of a CSV" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert result.width == 0


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
