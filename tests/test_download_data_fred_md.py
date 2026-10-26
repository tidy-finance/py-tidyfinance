"""Tests for download_data_fred_md (FRED-MD / FRED-QD)."""

import datetime as dt
import io
import os
import sys
import zipfile
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance import download_data, list_supported_datasets  # noqa: E402
from tidyfinance.download_open_source import (  # noqa: E402
    _apply_fred_md_tcode,
    _fetch_fred_md_bytes,
    _fetch_fred_md_text,
    _fred_md_wide,
    _looks_like_fred_md,
    _vintage_label,
    download_data_fred_md,
)

# A tiny FRED-MD-shaped CSV: header, the Transform: tcode row, then M/D/YYYY levels.
_CSV = (
    "sasdate,LEVELSER,LOGDIFFSER\n"
    "Transform:,1,5\n"
    "1/1/2020,100,100\n"
    "2/1/2020,101,110\n"
    "3/1/2020,102,121\n"
)

_HTML = "<!DOCTYPE html>\n<html><body>not found</body></html>"


def _zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _fake_get(*, csv=_CSV, zip_files=None, fail_csv=False):
    """Build a requests.get side effect: CSV text for .csv URLs, zip bytes for .zip URLs."""

    def _get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if url.endswith(".zip"):
            if zip_files is None:
                raise RuntimeError("no archive")
            resp.content = _zip_bytes(zip_files)
            return resp
        if fail_csv or csv is None:
            raise RuntimeError("not hosted")
        resp.text = csv
        return resp

    return _get


def _patch(**kwargs):
    return patch(
        "tidyfinance.download_open_source.requests.get",
        side_effect=_fake_get(**kwargs),
    )


def test_apply_tcode_transforms():
    """Each McCracken-Ng tcode (1-7) applies the expected causal transform."""
    import numpy as np

    x = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_array_equal(_apply_fred_md_tcode(x, 1), x)  # level
    np.testing.assert_array_equal(
        _apply_fred_md_tcode(x, 2), [np.nan, 1, 2, 4]
    )  # diff
    np.testing.assert_array_equal(
        _apply_fred_md_tcode(x, 3), [np.nan, np.nan, 1, 2]
    )  # diff^2
    np.testing.assert_allclose(_apply_fred_md_tcode(x, 4), np.log(x))
    assert np.isnan(_apply_fred_md_tcode(x, 5)[0])  # dlog leads with NaN
    np.testing.assert_allclose(
        _apply_fred_md_tcode(x, 5)[1:], np.diff(np.log(x))
    )
    assert np.isnan(_apply_fred_md_tcode(x, 6)[:2]).all()
    np.testing.assert_allclose(
        _apply_fred_md_tcode(x, 6)[2:], np.diff(np.log(x), n=2)
    )
    assert np.isnan(_apply_fred_md_tcode(x, 7)[0])
    with pytest.raises(ValueError, match="tcode"):
        _apply_fred_md_tcode(x, 9)


def test_vintage_label_extracts_yyyy_mm_from_archived_filenames():
    assert _vintage_label("2020-03-md.csv") == "2020-03"
    assert _vintage_label("fred-md_2020m3.csv") == "2020-03"
    assert _vintage_label("fred-qd_2018m05.csv") == "2018-05"
    assert _vintage_label("readme.txt") is None


def test_looks_like_fred_md_guards_against_non_csv_and_blank_content():
    assert not _looks_like_fred_md(None)
    assert not _looks_like_fred_md("")
    assert not _looks_like_fred_md("   \n  \n")
    assert not _looks_like_fred_md(_HTML)
    assert not _looks_like_fred_md("\r\n\r\n<!DOCTYPE html>")
    assert _looks_like_fred_md(_CSV)


def _response(status_code, **attrs):
    resp = MagicMock(status_code=status_code, **attrs)
    if status_code != 200:
        resp.raise_for_status.side_effect = RuntimeError(f"{status_code} Error")
    return resp


def test_fetch_fred_md_text_returns_the_response_body_on_success():
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=_response(200, text=_CSV),
    ):
        assert _fetch_fred_md_text("https://example.org/current.csv") == _CSV


def test_fetch_fred_md_text_raises_on_a_non_200_response():
    with (
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=_response(404),
        ),
        pytest.raises(RuntimeError, match="404"),
    ):
        _fetch_fred_md_text("https://example.org/current.csv")


def test_fetch_fred_md_bytes_returns_the_response_body_on_success():
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=_response(200, content=b"\x01\x02\x03"),
    ):
        assert (
            _fetch_fred_md_bytes("https://example.org/archive.zip")
            == b"\x01\x02\x03"
        )


def test_fetch_fred_md_bytes_raises_on_a_non_200_response():
    with (
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=_response(500),
        ),
        pytest.raises(RuntimeError, match="500"),
    ):
        _fetch_fred_md_bytes("https://example.org/archive.zip")


def test_latest_is_wide_raw_levels():
    """vintage='latest' returns a wide [date, <series...>] frame of raw levels."""
    with _patch():
        result = download_data_fred_md("FRED-MD")
    assert isinstance(result, pl.DataFrame)
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]
    assert "vintage" not in result.columns
    assert result["date"][0] == dt.date(2020, 1, 1)
    assert result["LEVELSER"].to_list() == [100, 101, 102]


def test_latest_raises_when_the_current_file_is_not_valid_csv():
    with (
        _patch(csv=_HTML),
        pytest.raises(ValueError, match="not a valid CSV"),
    ):
        download_data_fred_md("FRED-MD")


def test_fred_md_wide_drops_the_unnamed_column_of_a_trailing_comma():
    csv_trailing_comma = (
        "sasdate,LEVELSER,LOGDIFFSER,\n"
        "Transform:,1,5,\n"
        "1/1/2020,100,100,\n"
        "2/1/2020,101,110,\n"
    )
    result = _fred_md_wide(csv_trailing_comma, transform=True)
    assert result.columns == ["date", "LEVELSER", "LOGDIFFSER"]
    assert result["LEVELSER"].to_list() == [100, 101]


def test_transform_applies_per_series_tcode():
    """transform=True applies each series' tcode; a tcode-1 series is unchanged."""
    with _patch():
        result = download_data_fred_md("FRED-MD", transform=True)
    assert result["LEVELSER"].to_list() == [100, 101, 102]  # unchanged
    logdiff = result["LOGDIFFSER"]  # tcode 5: dlog, leads with NaN
    assert logdiff[0] is None
    assert logdiff[1] == pytest.approx(0.09531, abs=1e-4)


def test_specific_vintage_individual_adds_column():
    """A recent vintage (hosted individually) gets a 'vintage' column after 'date'."""
    with _patch():
        result = download_data_fred_md("FRED-MD", vintage="2026-01")
    assert result.columns[:2] == ["date", "vintage"]
    assert result["vintage"].unique().to_list() == ["2026-01"]


def test_vintage_neither_archived_nor_individually_hosted_raises():
    with (
        _patch(csv=_HTML),
        pytest.raises(ValueError, match="Could not fetch FRED-MD/QD vintage"),
    ):
        download_data_fred_md("FRED-MD", vintage="2026-06")


def test_vintage_extracted_from_archive():
    """An older vintage is extracted from the covering historical archive zip."""
    files = {"2020-03.csv": _CSV}
    with _patch(zip_files=files, fail_csv=True):
        result = download_data_fred_md("FRED-MD", vintage="2020-03")
    assert result["vintage"].unique().to_list() == ["2020-03"]
    assert result["LEVELSER"].to_list() == [100, 101, 102]


def test_vintage_covered_by_an_archive_but_missing_from_it_raises():
    files = {"2020-01.csv": _CSV}
    with (
        _patch(zip_files=files, fail_csv=True),
        pytest.raises(ValueError, match="not found in its FRED-MD/QD archive"),
    ):
        download_data_fred_md("FRED-MD", vintage="2020-03")


def test_all_stacks_vintages_wide():
    """vintage='all' stacks every archived vintage into a wide real-time panel."""
    files = {"2020-01.csv": _CSV, "2020-02.csv": _CSV}
    with _patch(
        zip_files=files, fail_csv=True
    ):  # recent individual pulls all fail
        result = download_data_fred_md("FRED-MD", vintage="all")
    assert result.columns[:2] == ["date", "vintage"]
    assert sorted(result["vintage"].unique()) == ["2020-01", "2020-02"]


def test_all_skips_archived_files_without_a_vintage_label():
    """A CSV whose name carries no vintage label is skipped, and HTML
    placeholders for recent months are rejected."""
    files = {
        "notes.csv": "not a vintage file",
        "2024-11.csv": _CSV,
        "2024-12.csv": _CSV,
    }
    with _patch(zip_files=files, csv=_HTML):
        result = download_data_fred_md("FRED-MD", vintage="all")
    assert result.columns[:2] == ["date", "vintage"]
    assert sorted(result["vintage"].unique()) == ["2024-11", "2024-12"]


def test_all_raises_when_no_vintage_can_be_downloaded():
    with (
        _patch(zip_files={}, csv=_HTML),
        pytest.raises(
            ValueError, match="No FRED-MD/QD vintages could be downloaded"
        ),
    ):
        download_data_fred_md("FRED-MD", vintage="all")


def test_all_fills_in_individually_hosted_vintages():
    with _patch(zip_files={}):
        result = download_data_fred_md("FRED-MD", vintage="all")
    assert result.columns[:2] == ["date", "vintage"]
    assert "2024-12" in result["vintage"].to_list()


def test_invalid_vintage_raises():
    with pytest.raises(ValueError, match="vintage must be"):
        download_data_fred_md("FRED-MD", vintage="banana")


def test_invalid_database_raises():
    with pytest.raises(ValueError, match="Unsupported database"):
        download_data_fred_md("not_a_database")


def test_supported_datasets_registered():
    """FRED-MD and FRED-QD are discoverable via list_supported_datasets (under the FRED domain)."""
    datasets = list_supported_datasets()
    fred = datasets[datasets["domain"] == "FRED"]
    assert {"FRED-MD", "FRED-QD"} <= set(fred["type"])


def test_download_data_dispatch_fred_md():
    """download_data("FRED", "FRED-MD") routes to the FRED-MD handler (wide output)."""
    with _patch():
        result = download_data("FRED", "FRED-MD")
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]


def test_download_data_dispatch_fred_qd():
    """download_data("FRED", "FRED-QD") routes to the FRED-QD handler."""
    with _patch():
        result = download_data("FRED", "FRED-QD")
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]


if __name__ == "__main__":
    pytest.main([__file__])
