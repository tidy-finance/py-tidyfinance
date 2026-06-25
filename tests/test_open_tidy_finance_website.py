"""Tests for open_tidy_finance_website."""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.utilities import open_tidy_finance_website  # noqa: E402

BASE_URL = "https://www.tidy-finance.org/python/"


def test_opens_base_url_when_chapter_is_none():
    """Test opens base URL and returns None when chapter is None."""
    with patch("tidyfinance.utilities.webbrowser.open") as mock_open:
        result = open_tidy_finance_website()

    assert result is None
    mock_open.assert_called_once_with(BASE_URL)


def test_opens_chapter_url_when_chapter_exists():
    """Test opens chapter URL when chapter exists in the chapter list."""
    with patch(
        "tidyfinance.utilities.list_tidy_finance_chapters",
        return_value=["beta-estimation"],
    ):
        with patch("tidyfinance.utilities.webbrowser.open") as mock_open:
            open_tidy_finance_website("beta-estimation")

    mock_open.assert_called_once_with(f"{BASE_URL}beta-estimation.html")


def test_falls_back_to_base_url_for_unknown_chapter():
    """Test falls back to base URL when chapter is not in the list."""
    with patch(
        "tidyfinance.utilities.list_tidy_finance_chapters",
        return_value=["beta-estimation"],
    ):
        with patch("tidyfinance.utilities.webbrowser.open") as mock_open:
            open_tidy_finance_website("unknown-chapter")

    mock_open.assert_called_once_with(BASE_URL)


if __name__ == "__main__":
    pytest.main([__file__])
