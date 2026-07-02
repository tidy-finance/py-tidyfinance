"""Tests for download_data_fred_md (FRED-MD / FRED-QD)."""

import io
import os
import sys
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance import download_data, list_supported_datasets  # noqa: E402
from tidyfinance.download_open_source import (  # noqa: E402
    _apply_fred_md_tcode,
    _download_data_fred_md,
)

# A tiny FRED-MD-shaped CSV: header, the Transform: tcode row, then M/D/YYYY levels.
_CSV = (
    "sasdate,LEVELSER,LOGDIFFSER\n"
    "Transform:,1,5\n"
    "1/1/2020,100,100\n"
    "2/1/2020,101,110\n"
    "3/1/2020,102,121\n"
)


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
    assert np.isnan(_apply_fred_md_tcode(x, 5)[0])  # dlog leads with NaN
    with pytest.raises(ValueError, match="tcode"):
        _apply_fred_md_tcode(x, 9)


def test_latest_is_wide_raw_levels():
    """vintage='latest' returns a wide [date, <series...>] frame of raw levels."""
    with _patch():
        result = _download_data_fred_md("fred_md_monthly")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]
    assert "vintage" not in result.columns
    assert result["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert result["LEVELSER"].tolist() == [100, 101, 102]


def test_transform_applies_per_series_tcode():
    """transform=True applies each series' tcode; a tcode-1 series is unchanged."""
    with _patch():
        result = _download_data_fred_md("fred_md_monthly", transform=True)
    assert result["LEVELSER"].tolist() == [100, 101, 102]  # tcode 1: unchanged
    logdiff = result["LOGDIFFSER"]  # tcode 5: dlog, leads with NaN
    assert pd.isna(logdiff.iloc[0])
    assert logdiff.iloc[1] == pytest.approx(0.09531, abs=1e-4)


def test_specific_vintage_individual_adds_column():
    """A recent vintage (hosted individually) gets a 'vintage' column after 'date'."""
    with _patch():
        result = _download_data_fred_md("fred_md_monthly", vintage="2026-01")
    assert list(result.columns[:2]) == ["date", "vintage"]
    assert result["vintage"].unique().tolist() == ["2026-01"]


def test_vintage_extracted_from_archive():
    """An older vintage is extracted from the covering historical archive zip."""
    files = {"2020-03.csv": _CSV}
    with _patch(zip_files=files, fail_csv=True):
        result = _download_data_fred_md("fred_md_monthly", vintage="2020-03")
    assert result["vintage"].unique().tolist() == ["2020-03"]
    assert "LEVELSER" in result.columns


def test_all_stacks_vintages_wide():
    """vintage='all' stacks every archived vintage into a wide real-time panel."""
    files = {"2020-01.csv": _CSV, "2020-02.csv": _CSV}
    with _patch(
        zip_files=files, fail_csv=True
    ):  # recent individual pulls all fail
        result = _download_data_fred_md("fred_md_monthly", vintage="all")
    assert list(result.columns[:2]) == ["date", "vintage"]
    assert sorted(result["vintage"].unique()) == ["2020-01", "2020-02"]


def test_invalid_vintage_raises():
    with pytest.raises(ValueError, match="vintage must be"):
        _download_data_fred_md("fred_md_monthly", vintage="banana")


def test_invalid_dataset_raises():
    with pytest.raises(ValueError, match="Unsupported FRED-MD/QD dataset"):
        _download_data_fred_md("not_a_dataset")


def test_supported_datasets_registered():
    """FRED-MD and FRED-QD are discoverable via list_supported_datasets."""
    datasets = list_supported_datasets()
    types = set(datasets["type"])
    assert {"fred_md_monthly", "fred_qd_quarterly"} <= types
    assert {"FRED-MD", "FRED-QD"} <= set(datasets["domain"])


def test_download_data_dispatch_fred_md():
    """The public download_data routes the FRED-MD domain to the handler (wide output)."""
    with _patch():
        result = download_data("FRED-MD", "fred_md_monthly")
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]


def test_download_data_dispatch_fred_qd():
    """The public download_data routes the FRED-QD domain to the handler."""
    with _patch():
        result = download_data("FRED-QD", "fred_qd_quarterly")
    assert list(result.columns) == ["date", "LEVELSER", "LOGDIFFSER"]


if __name__ == "__main__":
    pytest.main([__file__])
