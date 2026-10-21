"""Tests for download_data_wrds_crsp."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import _download_data_wrds_crsp  # noqa: E402


def test_crsp_dataset_validation_rejects_unsupported_values():
    """Test CRSP dataset validation rejects unsupported values."""
    from tidyfinance.supported_datasets import _check_supported_dataset_wrds_crsp  # noqa: E402

    with pytest.raises(ValueError, match="Unsupported CRSP dataset"):
        _check_supported_dataset_wrds_crsp("bad")
    _check_supported_dataset_wrds_crsp("crsp_monthly")
    _check_supported_dataset_wrds_crsp("crsp_daily")


def test_crsp_argument_validation_covers_required_inputs():
    """Test CRSP argument validation covers required inputs."""
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
    ):
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
        "tidyfinance.download_wrds._check_supported_dataset_wrds_crsp",
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
    return pl.DataFrame(
        {
            "permno": [1, 1, 2, 3],
            "date": [
                dt.date(2020, 1, 1),
                dt.date(2020, 2, 1),
                dt.date(2020, 1, 1),
                dt.date(2020, 1, 1),
            ],
            "calculation_date": [
                dt.date(2020, 1, 31),
                dt.date(2020, 2, 29),
                dt.date(2020, 1, 31),
                dt.date(2020, 1, 31),
            ],
            "ret": [0.10, 0.20, 0.30, 0.40],
            "shrout": [10, 20, 0, 40],
            "prc": [5, 6, 7, 8],
            "primaryexch": ["N", "A", "Q", "Z"],
            "siccd": [5100, 5300, 6500, 9500],
            "first_crsp_date": [dt.date(2000, 1, 1)] * 4,
            "mthvol": [11, 12, 13, 14],
        }
    )


def _mock_risk_free_monthly():
    return pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "risk_free": [0.01, 0.01],
        }
    )


def _mock_daily_query_result():
    return pl.DataFrame(
        {
            "permno": [1, 1, 1, 1, 2, 3],
            "date": [
                dt.date(2001, 1, 15),
                dt.date(2001, 2, 15),
                dt.date(2002, 2, 15),
                dt.date(2004, 2, 15),
                dt.date(2020, 1, 2),
                dt.date(2020, 1, 3),
            ],
            "ret": [0.10, 0.20, 0.30, 0.40, 0.50, None],
            "dlyprc": [10, 10, 10, 10, 0, 8],
            "dlyvol": [20, 18, 16, -99, 5, 6],
            "dlyfacprc": [1, 1, 1, 1, 1, 1],
            "primaryexch": ["Q", "Q", "Q", "Q", "N", "A"],
        }
    )


def _mock_risk_free_daily():
    return pl.DataFrame(
        {
            "date": [
                dt.date(2001, 1, 15),
                dt.date(2001, 2, 15),
                dt.date(2002, 2, 15),
                dt.date(2004, 2, 15),
                dt.date(2020, 1, 2),
                dt.date(2020, 1, 3),
            ],
            "risk_free": [0.01] * 6,
        }
    )


def test_monthly_crsp_v2_is_processed():
    """Test monthly CRSP v2 is processed."""
    monthly = _mock_monthly_query_result()

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=monthly),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_monthly(),
        ),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            additional_columns=["mthvol"],
        )

    assert isinstance(out, pl.DataFrame)
    assert "mthvol" in out.columns
    assert "mktcap" in out.columns
    # Expected column order (v2):
    # ..., siccd, <additional_columns>, listing_age, mktcap, mktcap_lag, ...
    # In particular listing_age precedes mktcap (regression test for the
    # swapped columns 9-10 reported in issue #36).
    assert out.columns == [
        "permno",
        "date",
        "calculation_date",
        "ret",
        "shrout",
        "prc",
        "primaryexch",
        "siccd",
        "mthvol",
        "listing_age",
        "mktcap",
        "mktcap_lag",
        "prc_lag",
        "exchange",
        "industry",
        "ret_excess",
    ]
    assert out.columns.index("listing_age") < out.columns.index("mktcap")
    # prc_lag is the previous month's price of the same stock
    prc_jan = out.filter(
        (pl.col("permno") == 1) & (pl.col("date") == dt.date(2020, 1, 1))
    )["prc"][0]
    prc_lag_feb = out.filter(
        (pl.col("permno") == 1) & (pl.col("date") == dt.date(2020, 2, 1))
    )["prc_lag"][0]
    assert prc_lag_feb == prc_jan


def test_daily_crsp_v2_validates_and_adjusts_volume():
    """Test daily CRSP v2 validates and adjusts volume."""
    permnos = pl.DataFrame({"permno": [1, 2, 3]})

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            # 1) distinct permnos, 2) the single daily batch query
            side_effect=[permnos, _mock_daily_query_result()],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_daily(),
        ),
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
    permnos = pl.DataFrame({"permno": [1, 2, 3]})

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            # 1) distinct permnos, then one query per batch (batch_size=1)
            side_effect=[
                permnos,
                _mock_daily_query_result(),
                _mock_daily_query_result(),
                _mock_daily_query_result(),
            ],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_daily(),
        ),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v2",
            batch_size=1,
        )

    assert isinstance(out, pl.DataFrame)
    assert len(out) > 0


def test_ccm_links_are_added_when_requested():
    """Test CCM links are added when requested."""
    monthly = _mock_monthly_query_result()
    ccm_links = pl.DataFrame(
        {
            "permno": [1, 1],
            "gvkey": ["001", None],
            "linkdt": [dt.date(2019, 1, 1), dt.date(2019, 1, 1)],
            "linkenddt": [dt.date(2020, 12, 31), dt.date(2020, 12, 31)],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=monthly),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_monthly(),
        ),
        patch(
            "tidyfinance.download_wrds._download_data_wrds_ccm_links",
            return_value=ccm_links,
        ),
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
    return pl.DataFrame(
        {
            "permno": [1, 1],
            "date": [dt.date(2020, 1, 15), dt.date(2020, 2, 15)],
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
    return pl.DataFrame(
        schema={
            "permno": pl.Int64,
            "dlstdt": pl.Date,
            "dlret": pl.Float64,
            "dlstcd": pl.Float64,
        }
    )


def _mock_monthly_v1_first_crsp_date():
    """Mock first_crsp_date per permno."""
    return pl.DataFrame(
        {
            "permno": [1],
            "first_crsp_date": [dt.date(2000, 1, 15)],
        }
    )


def test_monthly_crsp_v1_is_processed():
    """Test monthly CRSP v1 is processed."""
    msf_data = _mock_monthly_v1_msf()
    msedelist = _mock_monthly_v1_msedelist()
    first_crsp_date = _mock_monthly_v1_first_crsp_date()

    # Each call to _read_sql returns the next mock in sequence:
    # 1) msf+msenames query, 2) msedelist, 3) first_crsp_date
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=[msf_data, msedelist, first_crsp_date],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_monthly(),
        ),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-12-31",
            version="v1",
        )

    assert isinstance(out, pl.DataFrame)
    assert len(out) > 0
    # Expected v1-specific columns
    assert "mktcap" in out.columns
    assert "exchange" in out.columns
    assert "industry" in out.columns
    assert "ret_adj" in out.columns
    assert "prc_adj" in out.columns
    assert "prc_lag" in out.columns
    assert "listing_age" in out.columns
    assert "ret_excess" in out.columns
    # Exchange mapping from exchcd=1 -> NYSE
    assert (out["exchange"] == "NYSE").all()
    # Industry from siccd=5100 -> Wholesale
    assert (out["industry"] == "Wholesale").all()
    # mktcap = |shrout * 1000 * altprc| / 1e6 = |10 * 1000 * 5| / 1e6 = 0.05
    assert abs(out["mktcap"][0] - 0.05) < 1e-12


def test_read_sql_casts_decimal_columns_to_float():
    """Postgres 'numeric' columns arrive as pl.Decimal and must be
    normalized to Float64 for downstream arithmetic."""
    from decimal import Decimal

    from tidyfinance.download_wrds import _read_sql

    decimal_frame = pl.DataFrame(
        {
            "permno": [1, 2],
            "amount": [Decimal("1.50"), Decimal("2.25")],
        }
    )
    with patch("polars.read_database", return_value=decimal_frame):
        out = _read_sql("SELECT 1", "con")

    assert out.schema["amount"] == pl.Float64
    assert out["amount"].to_list() == [1.5, 2.25]


def test_monthly_crsp_v1_ccm_links_with_float_permno():
    """WRDS stores v1 permno as double precision; the read paths must
    cast it so the join with the Int64-permno ccm link table works."""
    as_float = pl.col("permno").cast(pl.Float64)
    msf_data = _mock_monthly_v1_msf().with_columns(as_float)
    msedelist = _mock_monthly_v1_msedelist().with_columns(as_float)
    first_crsp_date = _mock_monthly_v1_first_crsp_date().with_columns(
        as_float
    )
    ccm_links = pl.DataFrame(
        {
            "permno": pl.Series([1], dtype=pl.Int64),
            "gvkey": ["001"],
            "linkdt": [dt.date(2019, 1, 1)],
            "linkenddt": [dt.date(2020, 12, 31)],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=[msf_data, msedelist, first_crsp_date],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_monthly(),
        ),
        patch(
            "tidyfinance.download_wrds._download_data_wrds_ccm_links",
            return_value=ccm_links,
        ),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-12-31",
            version="v1",
            add_ccm_links=True,
        )

    assert "gvkey" in out.columns
    assert out["permno"].dtype == pl.Int64


def _mock_daily_v1_dsf():
    """Mock crsp.dsf + msenames join for v1."""
    return pl.DataFrame(
        {
            "permno": [1, 1, 1, 1, 2],
            "date": [
                dt.date(2001, 1, 15),
                dt.date(2001, 6, 15),
                dt.date(2002, 6, 15),
                dt.date(2004, 6, 15),
                dt.date(2020, 1, 2),
            ],
            "ret": [0.10, 0.20, 0.30, 0.40, 0.50],
            "prc": [10.0, 12.0, 15.0, 20.0, 25.0],
            "vol": [100, 200, 300, -99, 500],
            "cfacpr": [1.0, 1.0, 1.0, 1.0, 1.0],
            "exchcd": [3, 3, 3, 3, 1],  # NASDAQ, NASDAQ, NASDAQ, NASDAQ, NYSE
        }
    )


def _mock_daily_v1_msedelist():
    return pl.DataFrame(
        schema={
            "permno": pl.Int64,
            "dlstdt": pl.Date,
            "dlret": pl.Float64,
        }
    )


def _mock_daily_v1_permnos():
    return pl.DataFrame({"permno": [1, 2]})


def test_daily_crsp_v1_validates_and_adjusts_volume():
    """Test daily CRSP v1 validates and adjusts volume."""
    permnos = _mock_daily_v1_permnos()
    dsf = _mock_daily_v1_dsf()
    msedelist = _mock_daily_v1_msedelist()

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=[permnos, dsf, msedelist],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_daily(),
        ),
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

    # Reset the SQL sequence for the successful call:
    # 1) distinct permnos, 2) the dsf batch query, 3) the batch msedelist
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=[permnos, dsf, msedelist],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_daily(),
        ),
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

    assert isinstance(out, pl.DataFrame)
    assert "vol_adj" in out.columns
    assert "prc_adj" in out.columns


def test_daily_crsp_v1_handles_empty_batches():
    """Test daily CRSP v1 handles empty batches."""
    # No permnos -> processed_data stays empty
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            return_value=pl.DataFrame(schema={"permno": pl.Int64}),
        ),
    ):
        out = _download_data_wrds_crsp(
            dataset="crsp_daily",
            start_date="2001-01-01",
            end_date="2020-12-31",
            version="v1",
            batch_size=1,
        )

    assert isinstance(out, pl.DataFrame)
    assert len(out) == 0


def test_crsp_v1_end_date_past_2024_raises():
    """v1 with end_date > 2024-12-31 must raise."""
    with pytest.raises(ValueError, match="end_date"):
        _download_data_wrds_crsp(
            dataset="crsp_monthly",
            end_date="2025-01-01",
            version="v1",
        )


def test_crsp_v1_end_date_boundary_2024_12_31_is_allowed():
    """v1 with end_date == 2024-12-31 must not raise the boundary error."""
    # Mocks so the function runs to completion without WRDS access.
    msf = pl.DataFrame(
        {
            "permno": [1],
            "date": [dt.date(2024, 12, 31)],
            "ret": [0.01],
            "shrout": [10],
            "altprc": [5.0],
            "cfacpr": [1.0],
            "exchcd": [1],
            "siccd": [5100],
        }
    )
    msedelist = pl.DataFrame(
        schema={
            "permno": pl.Int64,
            "dlstdt": pl.Date,
            "dlret": pl.Float64,
            "dlstcd": pl.Float64,
        }
    )
    first_crsp = pl.DataFrame(
        {
            "permno": [1],
            "first_crsp_date": [dt.date(2000, 1, 15)],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=[msf, msedelist, first_crsp],
        ),
        patch(
            "tidyfinance.download_wrds._download_data_risk_free",
            return_value=_mock_risk_free_monthly(),
        ),
    ):
        # Should not raise the boundary error
        out = _download_data_wrds_crsp(
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2024-12-31",
            version="v1",
        )

    assert isinstance(out, pl.DataFrame)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
