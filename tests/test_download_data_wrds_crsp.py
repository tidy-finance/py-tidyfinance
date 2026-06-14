"""Tests for download_data_wrds_crsp."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    _download_data_wrds_crsp,
)  # noqa: E402


def test_crsp_dataset_validation_rejects_unsupported_values():
    """Test CRSP dataset validation rejects unsupported values."""
    from tidyfinance.data_download import (  # noqa
        _check_supported_dataset_wrds_crsp,
    )
    with pytest.raises(ValueError, match="Unsupported CRSP dataset"):
        _check_supported_dataset_wrds_crsp("bad")
    _check_supported_dataset_wrds_crsp("crsp_monthly")
    _check_supported_dataset_wrds_crsp("crsp_daily")


def test_crsp_argument_validation_covers_required_inputs():
    """Test CRSP argument validation covers required inputs."""
    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"):
        with pytest.raises((ValueError, TypeError)):
            _download_data_wrds_crsp()

        with pytest.raises(ValueError, match="batch_size"):
            _download_data_wrds_crsp("crsp_monthly", batch_size=0)

        with pytest.raises(ValueError, match="version"):
            _download_data_wrds_crsp("crsp_monthly", version="bad")

        with pytest.raises(ValueError, match="Unsupported CRSP dataset"):
            _download_data_wrds_crsp("bad")


def test_deprecated_type_inputs_are_translated_to_dataset():
    """Test deprecated type inputs are translated to dataset."""
    seen = {}

    def fake_check(dataset):
        seen["dataset"] = dataset
        raise ValueError("stop after translation")

    with patch(
        "tidyfinance.data_download._check_supported_dataset_wrds_crsp",
        side_effect=fake_check,
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError, match="stop after translation"):
                _download_data_wrds_crsp(type="wrds_crsp_monthly")
        assert seen["dataset"] == "crsp_monthly"

        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError, match="stop after translation"):
                _download_data_wrds_crsp(dataset="wrds_bad")
        assert seen["dataset"] == "bad"


def _mock_monthly_query_result():
    return pd.DataFrame(
        {
            "permno": [1, 1, 2, 3],
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-01-01",
                    "2020-01-01",
                ]
            ),
            "calculation_date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-01-31",
                    "2020-01-31",
                ]
            ),
            "ret": [0.10, 0.20, 0.30, 0.40],
            "shrout": [10, 20, 0, 40],
            "prc": [5, 6, 7, 8],
            "primaryexch": ["N", "A", "Q", "Z"],
            "siccd": [5100, 5300, 6500, 9500],
            "first_crsp_date": pd.to_datetime(["2000-01-01"] * 4),
            "mthvol": [11, 12, 13, 14],
        }
    )


def _mock_risk_free_monthly():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "risk_free": [0.01, 0.01],
        }
    )


def _mock_daily_query_result():
    return pd.DataFrame(
        {
            "permno": [1, 1, 1, 1, 2, 3],
            "date": pd.to_datetime(
                [
                    "2001-01-15",
                    "2001-02-15",
                    "2002-02-15",
                    "2004-02-15",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "ret": [0.10, 0.20, 0.30, 0.40, 0.50, None],
            "dlyprc": [10, 10, 10, 10, 0, 8],
            "dlyvol": [20, 18, 16, -99, 5, 6],
            "dlyfacprc": [1, 1, 1, 1, 1, 1],
            "primaryexch": ["Q", "Q", "Q", "Q", "N", "A"],
        }
    )


def _mock_risk_free_daily():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2001-01-15",
                    "2001-02-15",
                    "2002-02-15",
                    "2004-02-15",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "risk_free": [0.01] * 6,
        }
    )


def test_monthly_crsp_v2_is_processed():
    """Test monthly CRSP v2 is processed."""
    monthly = _mock_monthly_query_result()

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query", return_value=monthly
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_monthly(),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            additional_columns=["mthvol"],
        )

    assert isinstance(out, pd.DataFrame)
    assert "mthvol" in out.columns
    assert "mktcap" in out.columns


def test_daily_crsp_v2_validates_and_adjusts_volume():
    """Test daily CRSP v2 validates and adjusts volume."""
    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"), patch(
        "tidyfinance.data_download.pd.read_sql",
        return_value=pd.DataFrame({"permno": [1, 2, 3]}),
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query",
        return_value=_mock_daily_query_result(),
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_daily(),
    ):
        with pytest.raises(ValueError, match="adjust_volume"):
            _download_data_wrds_crsp(
                dataset="crsp_daily",
                start_date="2001-01-01",
                end_date="2020-12-31",
                version="v2",
                adjust_volume=True,
                additional_columns=["dlyprc"],
            )

        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            adjust_volume=True,
            additional_columns=[
                "dlyprc",
                "dlyvol",
                "dlyfacprc",
                "primaryexch",
            ],
        )

    assert "vol_adj" in out.columns
    assert "prc_adj" in out.columns
    # dlyvol/dlyprc/dlyfacprc are dropped after adjust_volume
    assert "dlyvol" not in out.columns


def test_daily_crsp_v2_handles_empty_batches():
    """Test daily CRSP v2 handles empty batches."""
    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"), patch(
        "tidyfinance.data_download.pd.read_sql",
        return_value=pd.DataFrame({"permno": [1, 2, 3]}),
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query",
        return_value=_mock_daily_query_result(),
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_daily(),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            batch_size=1,
        )

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_ccm_links_are_added_when_requested():
    """Test CCM links are added when requested."""
    monthly = _mock_monthly_query_result()
    ccm_links = pd.DataFrame(
        {
            "permno": [1, 1],
            "gvkey": ["001", None],
            "linkdt": pd.to_datetime(["2019-01-01", "2019-01-01"]),
            "linkenddt": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query", return_value=monthly
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_monthly(),
    ), patch(
        "tidyfinance.data_download._download_data_wrds_ccm_links",
        return_value=ccm_links,
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            add_ccm_links=True,
        )

    assert "gvkey" in out.columns


def _mock_monthly_v1_msf():
    """Mock crsp.msf + msenames join for v1."""
    return pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.to_datetime(["2020-01-15", "2020-02-15"]),
            "ret": [0.10, 0.20],
            "shrout": [10, 10],
            "altprc": [5.0, 5.5],
            "cfacpr": [1.0, 1.0],
            "exchcd": [1, 1],  # NYSE
            "siccd": [5100, 5100],  # Wholesale
        }
    )


def _mock_monthly_v1_msedelist():
    """Mock crsp.msedelist — empty (no delistings)."""
    return pd.DataFrame(
        {
            "permno": pd.Series([], dtype=int),
            "dlstdt": pd.Series([], dtype="datetime64[ns]"),
            "dlret": pd.Series([], dtype=float),
            "dlstcd": pd.Series([], dtype=float),
        }
    )


def _mock_monthly_v1_first_crsp_date():
    """Mock first_crsp_date per permno."""
    return pd.DataFrame(
        {
            "permno": [1],
            "first_crsp_date": pd.to_datetime(["2000-01-15"]),
        }
    )


def test_monthly_crsp_v1_is_processed():
    """Test monthly CRSP v1 is processed."""
    msf_data = _mock_monthly_v1_msf()
    msedelist = _mock_monthly_v1_msedelist()
    first_crsp_date = _mock_monthly_v1_first_crsp_date()

    # Each call to pd.read_sql_query returns the next mock in sequence:
    # 1) msf+msenames query, 2) msedelist, 3) first_crsp_date
    sql_results = iter([msf_data, msedelist, first_crsp_date])

    def fake_read_sql_query(sql, con, **kwargs):
        return next(sql_results)

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query",
        side_effect=fake_read_sql_query,
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_monthly(),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-12-31",
            version="v1",
        )

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    # Expected v1-specific columns
    assert "mktcap" in out.columns
    assert "exchange" in out.columns
    assert "industry" in out.columns
    assert "ret_adj" in out.columns
    assert "prc_adj" in out.columns
    assert "listing_age" in out.columns
    assert "ret_excess" in out.columns
    # Exchange mapping from exchcd=1 -> NYSE
    assert (out["exchange"] == "NYSE").all()
    # Industry from siccd=5100 -> Wholesale
    assert (out["industry"] == "Wholesale").all()
    # mktcap = |shrout * 1000 * altprc| / 1e6 = |10 * 1000 * 5| / 1e6 = 0.05
    assert abs(out["mktcap"].iloc[0] - 0.05) < 1e-12


def _mock_daily_v1_dsf():
    """Mock crsp.dsf + msenames join for v1."""
    return pd.DataFrame(
        {
            "permno": [1, 1, 1, 1, 2],
            "date": pd.to_datetime(
                [
                    "2001-01-15",
                    "2001-06-15",
                    "2002-06-15",
                    "2004-06-15",
                    "2020-01-02",
                ]
            ),
            "ret": [0.10, 0.20, 0.30, 0.40, 0.50],
            "prc": [10.0, 12.0, 15.0, 20.0, 25.0],
            "vol": [100, 200, 300, -99, 500],
            "cfacpr": [1.0, 1.0, 1.0, 1.0, 1.0],
            "exchcd": [3, 3, 3, 3, 1],  # NASDAQ, NASDAQ, NASDAQ, NASDAQ, NYSE
        }
    )


def _mock_daily_v1_msedelist():
    return pd.DataFrame(
        {
            "permno": pd.Series([], dtype=int),
            "dlstdt": pd.Series([], dtype="datetime64[ns]"),
            "dlret": pd.Series([], dtype=float),
        }
    )


def _mock_daily_v1_permnos():
    return pd.DataFrame({"permno": [1, 2]})


def test_daily_crsp_v1_validates_and_adjusts_volume():
    """Test daily CRSP v1 validates and adjusts volume."""
    permnos = _mock_daily_v1_permnos()
    dsf = _mock_daily_v1_dsf()
    msedelist = _mock_daily_v1_msedelist()

    sql_query_results = iter([dsf, msedelist])

    def fake_read_sql_query(sql, con, **kw):
        return next(sql_query_results)

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=permnos
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query",
        side_effect=fake_read_sql_query,
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_daily(),
    ):
        # Wrong adjust_volume columns -> error
        with pytest.raises(ValueError, match="prc"):
            _download_data_wrds_crsp(
                dataset="crsp_daily",
                start_date="2001-01-01",
                end_date="2020-12-31",
                version="v1",
                adjust_volume=True,
                additional_columns=["prc"],
            )

    # Reset iterators for the successful call
    sql_query_results = iter([dsf, msedelist])

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=permnos
    ), patch(
        "tidyfinance.data_download.pd.read_sql_query",
        side_effect=fake_read_sql_query,
    ), patch(
        "tidyfinance.data_download._download_data_risk_free",
        return_value=_mock_risk_free_daily(),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v1",
            adjust_volume=True,
            additional_columns=["prc", "vol", "cfacpr", "exchcd"],
            batch_size=500,
        )

    assert isinstance(out, pd.DataFrame)
    assert "vol_adj" in out.columns
    assert "prc_adj" in out.columns


def test_daily_crsp_v1_handles_empty_batches():
    """Test daily CRSP v1 handles empty batches."""
    # No permnos -> processed_data stays empty
    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"), patch(
        "tidyfinance.data_download.pd.read_sql",
        return_value=pd.DataFrame({"permno": []}),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v1",
            batch_size=1,
        )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
