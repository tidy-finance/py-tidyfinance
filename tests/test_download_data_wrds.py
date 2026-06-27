"""Tests for the download_data_wrds dispatcher."""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import _download_data_wrds  # noqa: E402
from tidyfinance.supported_datasets import _is_legacy_type  # noqa: E402


def test_wrds_helper_functions_validate_inputs():
    """Test WRDS helper functions validate inputs."""
    from tidyfinance.supported_datasets import _check_supported_dataset_wrds, _is_legacy_type_wrds  # noqa: E402

    assert _is_legacy_type_wrds("wrds_crsp_monthly")
    assert not _is_legacy_type_wrds("crsp_monthly")
    assert _check_supported_dataset_wrds("crsp_monthly") is None
    with pytest.raises(ValueError, match="Unsupported WRDS dataset"):
        _check_supported_dataset_wrds("bad")


def test_download_data_wrds_requires_dataset():
    """Test download_data_wrds requires dataset."""
    with pytest.raises((ValueError, TypeError), match="dataset"):
        _download_data_wrds()


def test_download_data_wrds_dispatches_crsp_datasets():
    """Test download_data_wrds dispatches CRSP datasets."""
    captured = {}

    def fake_crsp(dataset, start_date, end_date, **kwargs):
        captured["args"] = (dataset, start_date, end_date, kwargs)
        return "crsp_result"

    with patch(
        "tidyfinance.download_wrds._download_data_wrds_crsp",
        side_effect=fake_crsp,
    ):
        out = _download_data_wrds(
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-12-31",
            permno=1,
        )

    assert out == "crsp_result"
    assert captured["args"][0] == "crsp_monthly"
    assert captured["args"][1] == "2020-01-01"
    assert captured["args"][2] == "2020-12-31"
    assert captured["args"][3]["permno"] == 1


def test_download_data_wrds_dispatches_compustat_datasets():
    """Test download_data_wrds dispatches Compustat datasets."""
    captured = {}

    def fake_compustat(dataset, start_date, end_date, **kwargs):
        captured["args"] = (dataset, start_date, end_date, kwargs)
        return "compustat_result"

    with patch(
        "tidyfinance.download_wrds._download_data_wrds_compustat",
        side_effect=fake_compustat,
    ):
        out = _download_data_wrds(
            dataset="compustat_annual",
            start_date="2020-01-01",
            end_date="2020-12-31",
            gvkey="001690",
        )

    assert out == "compustat_result"
    assert captured["args"][0] == "compustat_annual"
    assert captured["args"][3]["gvkey"] == "001690"


def test_download_data_wrds_dispatches_link_and_bond_datasets():
    """Test download_data_wrds dispatches link and bond datasets."""
    with patch(
        "tidyfinance.download_wrds._download_data_wrds_ccm_links",
        return_value=("ccm", "linktype=LU"),
    ):
        assert _download_data_wrds("ccm_links", linktype="LU") == (
            "ccm",
            "linktype=LU",
        )

    with patch(
        "tidyfinance.download_wrds._download_data_wrds_fisd",
        return_value="fisd_result",
    ):
        assert _download_data_wrds("fisd", issuer="ABC") == "fisd_result"

    captured = {}

    def fake_trace(start_date=None, end_date=None, **kwargs):
        captured["args"] = (start_date, end_date, kwargs)
        return "trace_result"

    with patch(
        "tidyfinance.download_wrds._download_data_wrds_trace_enhanced",
        side_effect=fake_trace,
    ):
        assert (
            _download_data_wrds("trace_enhanced", cusips="00101JAH9")
            == "trace_result"
        )
        assert captured["args"][2]["cusips"] == "00101JAH9"


def test_download_data_wrds_handles_deprecated_type_argument():
    """Test download_data_wrds handles deprecated type argument."""
    with patch(
        "tidyfinance.download_wrds._download_data_wrds_crsp",
        side_effect=lambda ds, s, e, **kw: ds,
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = _download_data_wrds(type="wrds_crsp_monthly")
        assert out == "crsp_monthly"


def test_download_data_wrds_handles_legacy_dataset_values():
    """Test download_data_wrds handles legacy dataset values."""
    with patch(
        "tidyfinance.download_wrds._download_data_wrds_crsp",
        side_effect=lambda ds, s, e, **kw: ds,
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = _download_data_wrds("wrds_crsp_daily")
        assert out == "crsp_daily"

    # Sanity that the legacy helper would have recognised the type
    assert _is_legacy_type("wrds_crsp_daily")


def test_download_data_wrds_rejects_unsupported_datasets():
    """Test download_data_wrds rejects unsupported datasets."""
    with pytest.raises(ValueError, match="Unsupported"):
        _download_data_wrds("bad")


def test_download_data_wrds_final_fallback_errors():
    """Test download_data_wrds final fallback errors."""
    with pytest.raises(ValueError, match="Unsupported"):
        _download_data_wrds("other")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
