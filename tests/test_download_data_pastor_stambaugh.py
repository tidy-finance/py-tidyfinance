"""Tests for download_data_pastor_stambaugh."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import (  # noqa: E402
    download_data_pastor_stambaugh,
)


def test_downloads_and_processes_liquidity_factors():
    raw = pl.DataFrame(
        {
            "month": [196708, 196801],
            "agg_liq": [-0.01, 0.02],
            "innov_liq": [0.03, -0.04],
            "traded_liq": [-99.0, 0.05],
        }
    )
    with patch(
        "tidyfinance.download_open_source._fetch_whitespace_table",
        return_value=raw,
    ):
        result = download_data_pastor_stambaugh()

    assert isinstance(result, pl.DataFrame)
    assert result.columns == [
        "date",
        "agg_liq",
        "innov_liq",
        "traded_liq",
    ]
    # Dates are aligned to the beginning of the month.
    assert result["date"].to_list() == [
        dt.date(1967, 8, 1),
        dt.date(1968, 1, 1),
    ]
    # Returns are already decimal and must not be rescaled.
    assert result["agg_liq"].to_list() == [-0.01, 0.02]
    # The -99 sentinel for the pre-1968 traded factor becomes null.
    assert result["traded_liq"].is_null().to_list() == [True, False]
    assert result["traded_liq"][1] == 0.05


def test_filters_rows_when_both_dates_are_supplied():
    raw = pl.DataFrame(
        {
            "month": [202001, 202002, 202003],
            "agg_liq": [1, 2, 3],
            "innov_liq": [1, 2, 3],
            "traded_liq": [1, 2, 3],
        }
    )
    with patch(
        "tidyfinance.download_open_source._fetch_whitespace_table",
        return_value=raw,
    ):
        result = download_data_pastor_stambaugh(
            start_date="2020-02-01", end_date="2020-02-28"
        )

    assert len(result) == 1
    assert result["date"][0] == dt.date(2020, 2, 1)
    assert result["innov_liq"][0] == 2


def test_returns_empty_dataframe_after_download_failure():
    with patch(
        "tidyfinance.download_open_source._fetch_whitespace_table",
        side_effect=Exception("download failure"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = download_data_pastor_stambaugh()

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
