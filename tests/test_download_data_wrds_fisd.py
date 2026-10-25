"""Tests for download_data_wrds_fisd."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tidyfinance as tf  # noqa: E402
from tidyfinance.download_wrds import download_data_wrds_fisd  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_backend():
    """Ensure the global backend never leaks between tests."""
    tf.set_backend("pandas")
    yield
    tf.set_backend("pandas")


def test_downloads_filtered_fisd_data_for_usa_issuers():
    """Test downloads filtered FISD data for USA issuers."""
    issue_filtered = pl.DataFrame(
        {
            "complete_cusip": ["111111111", "222222222"],
            "maturity": [dt.date(2030, 1, 1), dt.date(2031, 1, 1)],
            "offering_amt": [100, 200],
            "offering_date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "dated_date": [dt.date(2020, 1, 2), dt.date(2020, 2, 2)],
            "interest_frequency": ["2", "2"],
            "coupon": [5, 6],
            "last_interest_date": [dt.date(2029, 12, 31), dt.date(2030, 12, 31)],
            "issue_id": [1, 2],
            "issuer_id": [1, 2],
        }
    )
    issuer = pl.DataFrame(
        {
            "issuer_id": [1, 2],
            "sic_code": ["1234", "9999"],
            "country_domicile": ["USA", "CAN"],
        }
    )

    def fake_read_sql(query, conn, *a, **kw):
        if "fisd_mergedissuer" in str(query):
            return issuer
        return issue_filtered

    disconnected = {"value": False}

    def fake_disconnect(conn):
        disconnected["value"] = True

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch(
            "tidyfinance.download_wrds.disconnect_connection",
            side_effect=fake_disconnect,
        ),
        patch(
            "tidyfinance.download_wrds._read_sql", side_effect=fake_read_sql
        ),
    ):
        result = download_data_wrds_fisd()

    assert disconnected["value"]
    expected_cols = [
        "complete_cusip",
        "maturity",
        "offering_amt",
        "offering_date",
        "dated_date",
        "interest_frequency",
        "coupon",
        "last_interest_date",
        "issue_id",
        "issuer_id",
        "sic_code",
    ]
    for col in expected_cols:
        assert col in result.columns
    assert len(result) == 1
    assert result["complete_cusip"][0] == "111111111"
    assert result["sic_code"][0] == "1234"


def test_download_data_fisd_polars_returns_date_columns():
    """With the polars backend, the FISD calendar-date columns must
    come out as polars Date, not Datetime (issue #66). The mocked
    frames carry datetime columns, mirroring a source that hands back
    timestamps rather than plain dates."""
    issue_filtered = pl.DataFrame(
        {
            "complete_cusip": ["111111111"],
            "maturity": [dt.datetime(2030, 1, 1)],
            "offering_amt": [100],
            "offering_date": [dt.datetime(2020, 1, 1)],
            "dated_date": [dt.datetime(2020, 1, 2)],
            "interest_frequency": ["2"],
            "coupon": [5],
            "last_interest_date": [dt.datetime(2029, 12, 31)],
            "issue_id": [1],
            "issuer_id": [1],
        }
    )
    issuer = pl.DataFrame(
        {
            "issuer_id": [1],
            "sic_code": ["1234"],
            "country_domicile": ["USA"],
        }
    )

    def fake_read_sql(query, conn, *a, **kw):
        if "fisd_mergedissuer" in str(query):
            return issuer
        return issue_filtered

    tf.set_backend("polars")
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql", side_effect=fake_read_sql
        ),
    ):
        out = tf.download_data(domain="WRDS", dataset="fisd")

    assert isinstance(out, pl.DataFrame)
    for column in [
        "maturity",
        "offering_date",
        "dated_date",
        "last_interest_date",
    ]:
        assert out.schema[column] == pl.Date, column


def test_returns_requested_additional_columns():
    """Test returns requested additional columns."""
    issue_filtered = pl.DataFrame(
        {
            "complete_cusip": ["111111111"],
            "maturity": [dt.date(2030, 1, 1)],
            "offering_amt": [100],
            "offering_date": [dt.date(2020, 1, 1)],
            "dated_date": [dt.date(2020, 1, 2)],
            "interest_frequency": ["2"],
            "coupon": [5],
            "last_interest_date": [dt.date(2029, 12, 31)],
            "issue_id": [1],
            "issuer_id": [1],
            "asset_backed": ["N"],
            "defeased": ["N"],
        }
    )
    issuer = pl.DataFrame(
        {
            "issuer_id": [1],
            "sic_code": ["1234"],
            "country_domicile": ["USA"],
        }
    )

    def fake_read_sql(query, conn, *a, **kw):
        if "fisd_mergedissuer" in str(query):
            return issuer
        return issue_filtered

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds._read_sql", side_effect=fake_read_sql
        ),
    ):
        result = download_data_wrds_fisd(
            additional_columns=["asset_backed", "defeased"]
        )

    expected_cols = [
        "complete_cusip",
        "maturity",
        "offering_amt",
        "offering_date",
        "dated_date",
        "interest_frequency",
        "coupon",
        "last_interest_date",
        "issue_id",
        "issuer_id",
        "asset_backed",
        "defeased",
        "sic_code",
    ]
    for col in expected_cols:
        assert col in result.columns
    assert result["asset_backed"][0] == "N"
    assert result["defeased"][0] == "N"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
