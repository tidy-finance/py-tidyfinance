"""Tests for download_data_pastor_stambaugh."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import (  # noqa: E402
    _download_data_pastor_stambaugh,
)


def test_downloads_and_processes_liquidity_factors():
    raw = pd.DataFrame(
        {
            "month": [196708, 196801],
            "agg_liq": [-0.01, 0.02],
            "innov_liq": [0.03, -0.04],
            "traded_liq": [-99, 0.05],
        }
    )
    with patch(
        "tidyfinance.download_open_source.pd.read_csv", return_value=raw
    ):
        result = _download_data_pastor_stambaugh()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "date",
        "agg_liq",
        "innov_liq",
        "traded_liq",
    ]
    # Dates are aligned to the beginning of the month.
    assert list(result["date"]) == [
        pd.Timestamp("1967-08-01"),
        pd.Timestamp("1968-01-01"),
    ]
    # Returns are already decimal and must not be rescaled.
    assert list(result["agg_liq"]) == [-0.01, 0.02]
    # The -99 sentinel for the pre-1968 traded factor becomes NaN.
    assert result["traded_liq"].isna().tolist() == [True, False]
    assert result["traded_liq"].iloc[1] == 0.05


def test_filters_rows_when_both_dates_are_supplied():
    raw = pd.DataFrame(
        {
            "month": [202001, 202002, 202003],
            "agg_liq": [1, 2, 3],
            "innov_liq": [1, 2, 3],
            "traded_liq": [1, 2, 3],
        }
    )
    with patch(
        "tidyfinance.download_open_source.pd.read_csv", return_value=raw
    ):
        result = _download_data_pastor_stambaugh(
            start_date="2020-02-01", end_date="2020-02-28"
        )

    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2020-02-01")
    assert result["innov_liq"].iloc[0] == 2


def test_returns_empty_dataframe_after_download_failure():
    with patch(
        "tidyfinance.download_open_source.pd.read_csv",
        side_effect=Exception("download failure"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = _download_data_pastor_stambaugh()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
