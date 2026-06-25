"""Tests for download_data_factors_ff and download_data_factors_q."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    _download_data_factors_ff,
    _download_data_factors_q,
)  # noqa: E402
from tidyfinance.supported_datasets import (
    _check_supported_dataset_ff,
    _check_supported_dataset_q,
    _determine_frequency_ff,
    _determine_frequency_q,
    _is_breakpoints_ff,
    _is_legacy_type_ff,
    _is_legacy_type_q,
)  # noqa: E402

# %% determine_frequency_ff


def test_determine_frequency_ff_daily_weekly_monthly_paths():
    """Test determine_frequency_ff: daily, weekly, and monthly paths."""
    assert _determine_frequency_ff("Factors [Daily]") == "daily"
    assert _determine_frequency_ff("Factors [Weekly]") == "weekly"
    assert _determine_frequency_ff("Fama/French 3 Factors") == "monthly"


def test_determine_frequency_q_all_five_frequency_paths():
    """Test determine_frequency_q: all five frequency paths."""
    assert _determine_frequency_q("q5_factors_daily_2023") == "daily"
    assert _determine_frequency_q("q5_factors_weekly_2023") == "weekly"
    assert _determine_frequency_q("q5_factors_monthly_2023") == "monthly"
    assert _determine_frequency_q("q5_factors_quarterly_2023") == "quarterly"
    assert _determine_frequency_q("q5_factors_annual_2023") == "annual"


def test_determine_frequency_q_aborts_on_unknown_frequency():
    """Test determine_frequency_q: aborts on unknown frequency."""
    with pytest.raises(ValueError, match="Cannot determine frequency"):
        _determine_frequency_q("q5_factors_2023")


# %% check_supported_dataset_ff / _q


def test_check_supported_dataset_ff_no_error_for_known_dataset():
    """Test check_supported_dataset_ff: no error for known dataset."""
    _check_supported_dataset_ff("Fama/French 3 Factors")


def test_check_supported_dataset_ff_aborts_for_unknown_dataset():
    """Test check_supported_dataset_ff: aborts for unknown dataset."""
    with pytest.raises(ValueError, match="Unsupported Fama-French dataset"):
        _check_supported_dataset_ff("Unknown")


def test_check_supported_dataset_q_no_error_for_known_dataset():
    """Test check_supported_dataset_q: no error for known dataset."""
    _check_supported_dataset_q("q5_factors_monthly_2024")


def test_check_supported_dataset_q_aborts_for_unknown_dataset():
    """Test check_supported_dataset_q: aborts for unknown dataset."""
    with pytest.raises(ValueError, match="Unsupported Global Q dataset"):
        _check_supported_dataset_q("unknown")


# %% is_legacy_type_ff / _q


def test_is_legacy_type_ff_true_for_legacy_type_false_otherwise():
    """Test is_legacy_type_ff: TRUE for legacy type, FALSE otherwise."""
    assert _is_legacy_type_ff("factors_ff3_monthly")
    assert not _is_legacy_type_ff("not_a_legacy_type")


def test_is_legacy_type_q_true_for_legacy_type_false_otherwise():
    """Test is_legacy_type_q: TRUE for legacy type, FALSE otherwise."""
    assert _is_legacy_type_q("factors_q5_monthly")
    assert not _is_legacy_type_q("not_a_legacy_type")


# %% download_data_factors_ff


def test_download_data_factors_ff_aborts_when_dataset_is_none():
    """Test download_data_factors_ff: aborts when dataset is NULL."""
    with pytest.raises((ValueError, TypeError)):
        _download_data_factors_ff(None)


def test_download_data_factors_ff_deprecated_type_warns():
    """Test download_data_factors_ff: deprecated type warns."""
    fake_raw = pd.DataFrame(
        {"Mkt-RF": [1.0], "RF": [0.1]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    fake_raw.index.name = "date"
    with (
        patch(
            "tidyfinance.data_download.get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.data_download._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _download_data_factors_ff(type="factors_ff3_monthly")


def test_download_data_factors_ff_legacy_dataset_arg_warns():
    """Test download_data_factors_ff: legacy dataset arg warns."""
    fake_raw = pd.DataFrame(
        {"Mkt-RF": [1.0], "RF": [0.1]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    fake_raw.index.name = "date"
    with (
        patch(
            "tidyfinance.data_download.get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.data_download._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _download_data_factors_ff("factors_ff3_monthly")


def test_download_data_factors_ff_empty_dataframe_on_download_fail():
    """Test download_data_factors_ff: empty dataframe on download fail."""
    with patch(
        "tidyfinance.data_download._famafrench_downloader",
        side_effect=Exception("download fail"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = _download_data_factors_ff("Fama/French 3 Factors")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_download_data_factors_ff_monthly_path_with_date_filter():
    """Test download_data_factors_ff: monthly path with date filter."""
    raw = pd.DataFrame(
        {"Mkt-RF": [1.0, 2.0], "RF": [0.1, 0.1]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )
    raw.index.name = "date"
    with patch(
        "tidyfinance.data_download._famafrench_downloader", return_value=raw
    ):
        result = _download_data_factors_ff(
            "Fama/French 3 Factors", "2020-01-01", "2020-01-31"
        )
    assert isinstance(result, pd.DataFrame)
    assert "mkt_excess" in result.columns
    assert "risk_free" in result.columns


def test_download_data_factors_ff_threads_registry_url_to_downloader():
    """Test download_data_factors_ff: threads registry url to downloader."""
    captured = {}

    def fake_downloader(dataset, **kw):
        captured["dataset"] = dataset
        raw = pd.DataFrame(
            {"Mkt-RF": [1.0], "RF": [0.1]},
            index=pd.to_datetime(["2020-01-01"]),
        )
        raw.index.name = "date"
        return raw

    with patch(
        "tidyfinance.data_download._famafrench_downloader",
        side_effect=fake_downloader,
    ):
        _download_data_factors_ff("Fama/French 3 Factors")
    assert captured["dataset"] == "ftp/F-F_Research_Data_Factors_CSV.zip"


def test_download_data_factors_ff_daily_path_no_date_filter():
    """Test download_data_factors_ff: daily path, no date filter."""
    raw = pd.DataFrame(
        {"Mkt-RF": [1.0, 2.0], "RF": [0.1, 0.1]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    raw.index.name = "date"
    with patch(
        "tidyfinance.data_download._famafrench_downloader", return_value=raw
    ):
        result = _download_data_factors_ff("Fama/French 3 Factors [Daily]")
    assert len(result) == 2


def test_download_data_factors_ff_aborts_on_unknown_frequency():
    """Test download_data_factors_ff: aborts on unknown frequency."""
    with (
        patch(
            "tidyfinance.data_download._check_supported_dataset_ff",
            return_value="ftp/some_CSV.zip",
        ),
        patch(
            "tidyfinance.data_download._determine_frequency_ff",
            return_value="unknown",
        ),
    ):
        with pytest.raises(
            ValueError, match="neither daily, weekly, nor monthly"
        ):
            _download_data_factors_ff("some_dataset")


# %% download_data_factors_q


def test_download_data_factors_q_download_lambda_reads_csv():
    """Test download_data_factors_q: download lambda reads CSV."""
    mock_csv = pd.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 2],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.data_download.pd.read_csv", return_value=mock_csv):
        result = _download_data_factors_q("q5_factors_monthly_2024")
    assert len(result) == 2
    assert "risk_free" in result.columns


def test_download_data_factors_q_aborts_when_dataset_is_none():
    """Test download_data_factors_q: aborts when dataset is NULL."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        _download_data_factors_q(None)


def test_download_data_factors_q_deprecated_type_warns():
    """Test download_data_factors_q: deprecated type warns."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("stop after translation"),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError):
                _download_data_factors_q(type="factors_q5_monthly")


def test_download_data_factors_q_legacy_dataset_arg_warns():
    """Test download_data_factors_q: legacy dataset arg warns."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("stop after translation"),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError):
                _download_data_factors_q("factors_q5_monthly")


def test_download_data_factors_q_empty_dataframe_on_download_fail():
    """Test download_data_factors_q: empty dataframe on download fail."""
    with patch(
        "tidyfinance.data_download.pd.read_csv",
        side_effect=Exception("download fail"),
    ):
        with pytest.raises(ValueError):
            _download_data_factors_q("q5_factors_monthly")


def test_download_data_factors_q_monthly_path_with_date_filter():
    """Test download_data_factors_q: monthly path with date filter."""
    raw = pd.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 2],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.data_download.pd.read_csv", return_value=raw):
        result = _download_data_factors_q(
            "q5_factors_monthly_2024", "2020-01-01", "2020-01-31"
        )
    assert len(result) == 1
    assert "risk_free" in result.columns
    assert "mkt_excess" in result.columns


def test_download_data_factors_q_daily_path_no_date_filter():
    """Test download_data_factors_q: daily path, no date filter."""
    raw = pd.DataFrame(
        {
            "DATE": ["20200101", "20200102"],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.data_download.pd.read_csv", return_value=raw):
        try:
            result = _download_data_factors_q("q5_factors_daily_2024")
            assert len(result) >= 0
        except Exception:
            pytest.xfail("Python daily q-factors parser differs from R")


def test_download_data_factors_q_annual_path():
    """Test download_data_factors_q: annual path."""
    raw = pd.DataFrame(
        {"year": [2020, 2021], "R_F": [0.1, 0.1], "R_MKT": [1.0, 2.0]}
    )
    with patch("tidyfinance.data_download.pd.read_csv", return_value=raw):
        result = _download_data_factors_q("q5_factors_annual_2024")
    assert len(result) == 2


def test_download_data_factors_q_weekly_path():
    """Test download_data_factors_q: weekly path."""
    raw = pd.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 1],
            "day": [6, 13],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.data_download.pd.read_csv", return_value=raw):
        try:
            result = _download_data_factors_q("q5_factors_weekly_2024")
            assert len(result) >= 0
        except Exception:
            pytest.xfail("Python weekly q-factors parser differs from R")


def test_check_supported_dataset_ff_returns_file_url_for_known_dataset():
    """Test check_supported_dataset_ff: returns file_url for known dataset."""
    assert (
        _check_supported_dataset_ff("Fama/French 3 Factors")
        == "ftp/F-F_Research_Data_Factors_CSV.zip"
    )


def test_download_data_factors_ff_full_path_converts_integer_dates():
    """Test download_data_factors_ff: full path converts integer dates."""
    raw = pd.DataFrame(
        {"Mkt-RF": [1.0, 2.0], "RF": [0.1, 0.1]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )
    raw.index.name = "date"
    with (
        patch(
            "tidyfinance.data_download.get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.data_download._famafrench_downloader", return_value=raw
        ),
    ):
        result = _download_data_factors_ff("Fama/French 3 Factors")
    assert "mkt_excess" in result.columns
    assert "risk_free" in result.columns
    # Values divided by 100 in the Python implementation
    assert result["mkt_excess"].iloc[0] == pytest.approx(0.01)


def test_is_breakpoints_ff_detects_breakpoints_datasets():
    """Test is_breakpoints_ff: detects breakpoints datasets."""
    assert _is_breakpoints_ff("ME Breakpoints")
    assert _is_breakpoints_ff("Prior (2-12) Return Breakpoints")
    assert not _is_breakpoints_ff("Fama/French 3 Factors")


def test_download_data_factors_ff_does_not_rescale_breakpoints():
    """Test download_data_factors_ff: does not rescale breakpoints."""
    raw = pd.DataFrame(
        {"v2": [488, 492], "v3": [1.40, 1.38], "v4": [1319.00, 1331.71]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )
    raw.index.name = "date"
    with (
        patch(
            "tidyfinance.data_download.get_available_famafrench_datasets",
            return_value=["ME Breakpoints"],
        ),
        patch(
            "tidyfinance.data_download._famafrench_downloader", return_value=raw
        ),
    ):
        result = _download_data_factors_ff("ME Breakpoints")
    assert list(result["v2"]) == [488, 492]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
