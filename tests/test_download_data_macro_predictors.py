"""Tests for download_data_macro_predictors."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (  # noqa: E402
    _download_data_macro_predictors,
)


def _macro_raw(date_col, values):
    return pd.DataFrame(
        {
            date_col: values,
            "Index": [100, 110, 120],
            "D12": [2, 2, 2],
            "E12": [5, 5, 5],
            "Rfree": [0.01, 0.01, 0.01],
            "svar": [1, 1, 1],
            "b/m": [0.4, 0.5, 0.6],
            "ntis": [0.1, 0.1, 0.1],
            "tbl": [0.02, 0.02, 0.02],
            "lty": [0.05, 0.05, 0.05],
            "ltr": [0.03, 0.03, 0.03],
            "BAA": [0.07, 0.07, 0.07],
            "AAA": [0.04, 0.04, 0.04],
            "infl": [0.02, 0.02, 0.02],
        }
    )


def test_dataset_is_required_and_must_be_supported():
    """Test dataset is required and must be supported."""
    with pytest.raises(ValueError):
        _download_data_macro_predictors()

    with pytest.raises(ValueError, match="Unsupported dataset"):
        _download_data_macro_predictors("daily")


def test_monthly_data_is_downloaded_and_processed():
    """Test monthly data is downloaded and processed."""
    raw = _macro_raw("yyyymm", ["202001", "202002", "202003"])
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        out = _download_data_macro_predictors("monthly")
    assert isinstance(out, pd.DataFrame)
    # The middle row survives the diff()/dropna pipeline
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2020-02-01")


def test_quarterly_data_is_downloaded_and_filtered():
    """Test quarterly data is downloaded and filtered."""
    raw = _macro_raw("yyyyq", ["20201", "20202", "20203"])
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        out = _download_data_macro_predictors(
            "quarterly",
            start_date="2020-04-01",
            end_date="2020-04-01",
        )
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2020-04-01")


def test_annual_data_is_downloaded_and_processed():
    """Test annual data is downloaded and processed."""
    raw = _macro_raw("yyyy", ["2019", "2020", "2021"])
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        out = _download_data_macro_predictors("annual")
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2020-01-01")


def test_empty_downloads_return_an_empty_dataframe():
    """Test empty downloads return an empty dataframe."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("download failed"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            out = _download_data_macro_predictors("monthly")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_explicit_deprecated_type_is_still_handled():
    """Test explicit deprecated type is still handled."""
    raw = _macro_raw("yyyymm", ["202001", "202002", "202003"])
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = _download_data_macro_predictors(
                type="macro_predictors_monthly"
            )
        assert out["date"].iloc[0] == pd.Timestamp("2020-02-01")


def test_legacy_dataset_names_are_still_handled():
    """Test legacy dataset names are still handled."""
    raw = _macro_raw("yyyymm", ["202001", "202002", "202003"])
    with patch(
        "tidyfinance.data_download.pd.read_csv", return_value=raw
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = _download_data_macro_predictors("macro_predictors_monthly")
        assert out["date"].iloc[0] == pd.Timestamp("2020-02-01")


def test_correct_google_sheets_url_is_constructed():
    """Test correct Google Sheets URL is constructed."""
    captured = {}

    def fake_read_csv(url, *a, **kw):
        captured["url"] = url
        return _macro_raw("yyyymm", ["202001", "202002", "202003"])

    with patch(
        "tidyfinance.data_download.pd.read_csv", side_effect=fake_read_csv
    ):
        _download_data_macro_predictors(
            dataset="monthly", sheet_id="test_sheet"
        )

    expected = (
        "https://docs.google.com/spreadsheets/d/"
        "test_sheet"
        "/gviz/tq?tqx=out:csv&sheet=Monthly"
    )
    assert captured["url"] == expected


def test_empty_download_informs_and_returns_empty_dataframe():
    """Test empty download informs and returns empty dataframe."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("download failed"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            out = _download_data_macro_predictors("monthly")
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            out = _download_data_macro_predictors("quarterly")
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            out = _download_data_macro_predictors("annual")
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
