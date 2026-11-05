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
from tidyfinance.download import download_data  # noqa: E402
from tidyfinance.download_open_source import (  # noqa: E402
    _OsapFileNotFound,
    download_data_osap,
)

DOWNLOAD = "tidyfinance.download_open_source._download_osap_file"
GOOGLE_DRIVE = "tidyfinance.download_open_source._fetch_google_drive_csv"


def osap_file(data):
    """Serve 'data' as the parquet file of every download and record the
    requested URLs."""
    urls = []

    def fake_download(url, path):
        urls.append(url)
        data.write_parquet(path)

    return patch(DOWNLOAD, side_effect=fake_download), urls


def warning_messages(record):
    return [str(w.message) for w in record]


# %% Google Drive


def test_downloads_and_processes_a_google_drive_file():
    """Test downloads and processes a Google Drive file."""
    raw = pl.DataFrame(
        {"date": ["2020-01-31", "2020-02-29"], "LongName": [1, 2]}
    )
    with patch(GOOGLE_DRIVE, return_value=raw):
        result = download_data_osap(sheet_id="abc")

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "long_name"]
    # Dates are aligned to the beginning of the month.
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]
    # Percentage returns are scaled to numeric (decimal) values.
    assert result["long_name"].to_list() == [0.01, 0.02]


def _html_response(content_type, body):
    """Mock a Google Drive response that is an HTML page."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": content_type}
    response.text = body
    response.content = body.encode("utf-8")
    return response


def test_reports_an_exceeded_google_drive_quota():
    """Test reports an exceeded Google Drive quota."""
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
            result = download_data_osap(sheet_id="abc")

    messages = warning_messages(record)
    assert any("download quota of the file is exceeded" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert result.shape == (0, 0)


def test_rejects_other_html_pages_instead_of_parsing_them():
    """Test rejects other HTML pages instead of parsing them."""
    response = _html_response(
        "text/html", "<html><body>Sign in</body></html>"
    )
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=response,
    ):
        with pytest.warns(UserWarning) as record:
            result = download_data_osap(sheet_id="abc")

    messages = warning_messages(record)
    assert any("HTML page instead of a CSV" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert result.width == 0


def test_keeps_sheet_id_as_the_third_positional_argument():
    """Test keeps sheet_id as the third positional argument."""
    requested = []

    def fake_google_drive(url):
        requested.append(url)
        return pl.DataFrame({"date": ["2020-01-31"], "value": [3.0]})

    with patch(GOOGLE_DRIVE, side_effect=fake_google_drive):
        result = download_data_osap("2020-01-01", "2020-12-31", "abc")

    assert requested[0].endswith("id=abc")
    assert result["value"].to_list() == [0.03]


def test_selects_signals_from_a_google_drive_file():
    """Test selects signals from a Google Drive file."""
    raw = pl.DataFrame({"date": ["2020-01-31"], "BM": [1.0], "Mom12m": [2.0]})
    with patch(GOOGLE_DRIVE, return_value=raw):
        result = download_data_osap(sheet_id="abc", signals="BM")
        assert result.columns == ["date", "bm"]

        with pytest.raises(ValueError, match="unknown names.*Nope"):
            download_data_osap(sheet_id="abc", signals="Nope")


# %% Hugging Face: long-short returns


def test_downloads_the_latest_release_from_hugging_face_by_default():
    """Test downloads the latest release from Hugging Face by default."""
    mock, urls = osap_file(
        pl.DataFrame(
            {
                "date": [dt.date(2020, 1, 31), dt.date(2020, 2, 29)],
                "LongName": [1.0, 2.0],
            }
        )
    )
    with mock:
        result = download_data_osap()

    assert urls == [
        "https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/"
        "Data-Release-2026.10/Portfolios/Full%20Sets%20OP/"
        "PredictorLSretWide.parquet"
    ]
    assert result.columns == ["date", "long_name"]
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]
    assert result["long_name"].to_list() == [0.01, 0.02]


def test_downloads_an_earlier_release_from_its_branch():
    """Test downloads an earlier release from its branch."""
    mock, urls = osap_file(
        pl.DataFrame({"date": [dt.date(2020, 1, 31)], "value": [1.0]})
    )
    with mock:
        download_data_osap(release="2025.10")
        download_data_osap(release="2021.03")

    assert "/resolve/2025.10/Portfolios/Full%20Sets%20OP/" in urls[0]
    assert "/resolve/2021.03/Portfolios/Full_Sets_Main/" in urls[1]


def test_finds_the_latest_release_on_a_branch_named_like_the_release():
    """Test finds the latest release on a branch named like the release."""
    urls = []

    def fake_download(url, path):
        urls.append(url)
        if "Data-Release" in url:
            raise _OsapFileNotFound(url)
        data = pl.DataFrame({"date": [dt.date(2020, 1, 31)], "value": [1.0]})
        data.write_parquet(path)

    with patch(DOWNLOAD, side_effect=fake_download):
        result = download_data_osap()

    assert len(urls) == 2
    assert "/resolve/2026.10/" in urls[1]
    assert result.height == 1


def test_reports_a_missing_file_of_an_earlier_release():
    """Test reports a missing file of an earlier release."""
    with patch(DOWNLOAD, side_effect=_OsapFileNotFound("missing")):
        with pytest.warns(UserWarning) as record:
            result = download_data_osap(release="2025.10")

    assert any(
        "Returning an empty dataset" in m for m in warning_messages(record)
    )
    assert result.height == 0


def test_rejects_unsupported_releases():
    """Test rejects unsupported releases."""
    for release in ("2020.05", "latest", ["2026.10", "2025.10"]):
        with pytest.raises(ValueError, match="must be one of"):
            download_data_osap(release=release)


def test_rejects_release_and_sheet_id_together():
    """Test rejects release and sheet_id together."""
    with pytest.raises(ValueError, match="either"):
        download_data_osap(release="2025.10", sheet_id="abc")

    # A forwarded None is the same as leaving out the release.
    with patch(
        GOOGLE_DRIVE,
        return_value=pl.DataFrame({"date": ["2020-01-31"], "value": [1.0]}),
    ):
        result = download_data_osap(sheet_id="abc", release=None)
    assert result.height == 1


def test_filters_rows_when_both_dates_are_supplied():
    """Test filters rows when both dates are supplied."""
    mock, _ = osap_file(
        pl.DataFrame(
            {
                "date": [
                    dt.date(2020, 1, 31),
                    dt.date(2020, 2, 29),
                    dt.date(2020, 3, 31),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )
    )
    with mock:
        result = download_data_osap(
            start_date="2020-02-01", end_date="2020-02-28"
        )

    assert result.height == 1
    assert result["date"].to_list() == [dt.date(2020, 2, 1)]
    assert result["value"].to_list() == [0.02]


def test_returns_empty_dataframe_when_the_hugging_face_download_fails():
    """Test returns empty dataframe when the Hugging Face download fails."""
    with patch(DOWNLOAD, side_effect=ConnectionError("connection refused")):
        with pytest.warns(UserWarning) as record:
            result = download_data_osap()

    messages = warning_messages(record)
    assert any("connection refused" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 0


def test_reads_the_long_short_returns_of_a_single_factor():
    """Test reads the long-short returns of a single factor."""
    mock, _ = osap_file(
        pl.DataFrame(
            {"date": [dt.date(2020, 1, 31)], "BM": [1.0], "Mom12m": [2.0]}
        )
    )
    with mock:
        result = download_data_osap(signals="BM")

    assert result.columns == ["date", "bm"]
    assert result["bm"].to_list() == [0.01]


# %% Hugging Face: signal documentation


def test_downloads_the_signal_documentation():
    """Test downloads the signal documentation."""
    mock, urls = osap_file(
        pl.DataFrame(
            {
                "Acronym": ["AM", "BM"],
                "Predictability in OP": ["1_clear", "1_clear"],
                "T-Stat": [2.1, 3.4],
            }
        )
    )
    with mock:
        result = download_data_osap(
            start_date="2020-01-01",
            end_date="2020-12-31",
            dataset="signal_doc",
        )

    assert urls[0].endswith(
        "/resolve/Data-Release-2026.10/SignalDoc.parquet"
    )
    assert result.columns == ["acronym", "predictability_in_op", "t_stat"]
    # Acronyms are kept as OSAP names, which 'signals' takes.
    assert result["acronym"].to_list() == ["AM", "BM"]


def test_rejects_releases_without_a_signal_documentation():
    """Test rejects releases without a signal documentation."""
    with pytest.raises(ValueError, match="must be one of"):
        download_data_osap(dataset="signal_doc", release="2021.04")


def test_returns_empty_dataframe_when_the_signal_documentation_fails():
    """Test returns empty dataframe when the signal documentation fails."""
    with patch(DOWNLOAD, side_effect=ConnectionError("connection refused")):
        with pytest.warns(UserWarning) as record:
            result = download_data_osap(dataset="signal_doc")

    messages = warning_messages(record)
    assert any("connection refused" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert result.height == 0


# %% Hugging Face: firm-level predictors


def test_downloads_and_processes_firm_level_predictors(capsys):
    """Test downloads and processes firm-level predictors."""
    mock, urls = osap_file(
        pl.DataFrame(
            {
                "permno": [10000, 10000, 10001],
                "yyyymm": [201912, 202001, 202002],
                "BM": [0.5, 0.6, 0.7],
                "Mom12m": [0.1, 0.2, 0.3],
                "Other": [1.0, 2.0, 3.0],
            }
        )
    )
    with mock:
        result = download_data_osap(
            start_date="2020-01-01",
            end_date="2020-12-31",
            dataset="predictors",
            signals=["BM", "Mom12m"],
        )

    assert "about 2 GB" in capsys.readouterr().err
    assert urls[0].endswith(
        "/Firm%20Level%20Characteristics/Predictors.parquet"
    )
    assert result.columns == ["permno", "date", "bm", "mom12m"]
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]
    # Signals are not scaled.
    assert result["bm"].to_list() == [0.6, 0.7]


def test_downloads_all_signed_predictors_by_default():
    """Test downloads all signed predictors by default."""
    mock, urls = osap_file(
        pl.DataFrame(
            {
                "permno": [10000],
                "yyyymm": [202001],
                "BM": [0.5],
                "Mom12m": [0.1],
            }
        )
    )
    with mock:
        result = download_data_osap(dataset="signed_predictors")

    assert urls[0].endswith("signed_predictors_dl_wide.parquet")
    assert result.columns == ["permno", "date", "bm", "mom12m"]


def test_rejects_unknown_signals_instead_of_reporting_a_failed_download():
    """Test rejects unknown signals instead of reporting a failed
    download."""
    mock, _ = osap_file(
        pl.DataFrame({"permno": [10000], "yyyymm": [202001], "BM": [0.5]})
    )
    with mock:
        with pytest.raises(ValueError, match=r"unknown names.*'bm'"):
            download_data_osap(dataset="predictors", signals=["BM", "bm"])


# %% Argument validation and dispatch


def test_validates_dataset_signals_and_sheet_id():
    """Test validates dataset, signals, and sheet_id."""
    with pytest.raises(ValueError, match="must be one of"):
        download_data_osap(dataset="placebos")
    with pytest.raises(ValueError, match="only supported for"):
        download_data_osap(dataset="signal_doc", sheet_id="abc")
    with pytest.raises(ValueError, match="not supported for"):
        download_data_osap(dataset="signal_doc", signals="BM")
    with pytest.raises(ValueError, match="non-empty string or list"):
        download_data_osap(dataset="predictors", signals=[])


def test_download_data_passes_the_osap_arguments_through():
    """Test download_data passes dataset, release, signals, and sheet_id
    through."""
    captured = {}

    def fake_osap(**kwargs):
        captured.update(kwargs)
        return pl.DataFrame({"ok": [True]})

    with patch(
        "tidyfinance.download.download_data_osap", side_effect=fake_osap
    ):
        download_data("Open Source Asset Pricing")
        assert captured["dataset"] == "long_short_returns"

        download_data(
            "Open Source Asset Pricing",
            "signal_doc",
            release="2025.10",
        )
        assert captured["dataset"] == "signal_doc"
        assert captured["release"] == "2025.10"

        download_data(
            "Open Source Asset Pricing", signals="BM", sheet_id="abc"
        )
        assert captured["signals"] == "BM"
        assert captured["sheet_id"] == "abc"


def test_converts_names_to_snake_case():
    """Test converts names to snake case."""
    result = [
        _transfrom_to_snake_case(c)
        for c in [
            "LongName",
            "two words",
            "__Already___Odd__",
            "BM",
            "BMdec",
            "Mom12m",
            "T-Stat",
        ]
    ]
    assert result == [
        "long_name",
        "two_words",
        "already_odd",
        "bm",
        "bmdec",
        "mom12m",
        "t_stat",
    ]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
