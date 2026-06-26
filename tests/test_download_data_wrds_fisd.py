"""Tests for download_data_wrds_fisd."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import _download_data_wrds_fisd  # noqa: E402


def test_downloads_filtered_fisd_data_for_usa_issuers():
    """Test downloads filtered FISD data for USA issuers."""
    issue_filtered = pd.DataFrame(
        {
            "complete_cusip": ["111111111", "222222222"],
            "maturity": pd.to_datetime(["2030-01-01", "2031-01-01"]),
            "offering_amt": [100, 200],
            "offering_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "dated_date": pd.to_datetime(["2020-01-02", "2020-02-02"]),
            "interest_frequency": ["2", "2"],
            "coupon": [5, 6],
            "last_interest_date": pd.to_datetime(["2029-12-31", "2030-12-31"]),
            "issue_id": [1, 2],
            "issuer_id": [1, 2],
        }
    )
    issuer = pd.DataFrame(
        {
            "issuer_id": [1, 2],
            "sic_code": ["1234", "9999"],
            "country_domicile": ["USA", "CAN"],
        }
    )

    def fake_read_sql(query, conn, *a, **kw):
        if "fisd_mergedissue " in str(query) or (
            "fisd_mergedissue" in str(query) and "issuer" not in str(query)
        ):
            return issue_filtered
        return issuer

    def fake_read_sql_query(sql=None, con=None, *a, **kw):
        return fake_read_sql(sql, con)

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
            "tidyfinance.download_wrds.pd.read_sql_query",
            side_effect=fake_read_sql_query,
        ),
        patch(
            "tidyfinance.download_wrds.pd.read_sql", side_effect=fake_read_sql
        ),
    ):
        result = _download_data_wrds_fisd()

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
    assert result["complete_cusip"].iloc[0] == "111111111"
    assert result["sic_code"].iloc[0] == "1234"


def test_returns_requested_additional_columns():
    """Test returns requested additional columns."""
    issue_filtered = pd.DataFrame(
        {
            "complete_cusip": ["111111111"],
            "maturity": pd.to_datetime(["2030-01-01"]),
            "offering_amt": [100],
            "offering_date": pd.to_datetime(["2020-01-01"]),
            "dated_date": pd.to_datetime(["2020-01-02"]),
            "interest_frequency": ["2"],
            "coupon": [5],
            "last_interest_date": pd.to_datetime(["2029-12-31"]),
            "issue_id": [1],
            "issuer_id": [1],
            "asset_backed": ["N"],
            "defeased": ["N"],
        }
    )
    issuer = pd.DataFrame(
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

    def fake_read_sql_query(sql=None, con=None, *a, **kw):
        return fake_read_sql(sql, con)

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch(
            "tidyfinance.download_wrds.pd.read_sql_query",
            side_effect=fake_read_sql_query,
        ),
        patch(
            "tidyfinance.download_wrds.pd.read_sql", side_effect=fake_read_sql
        ),
    ):
        result = _download_data_wrds_fisd(
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
    assert result["asset_backed"].iloc[0] == "N"
    assert result["defeased"].iloc[0] == "N"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
