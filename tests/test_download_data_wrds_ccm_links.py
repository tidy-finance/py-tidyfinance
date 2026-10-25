"""Tests for download_data_wrds_ccm_links."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import download_data_wrds_ccm_links  # noqa: E402


def test_downloads_default_ccm_links_and_replaces_missing_end_dates():
    """Test downloads default CCM links and replaces missing end dates."""
    sql_result = pl.DataFrame(
        {
            "permno": [1, 2],
            "gvkey": ["001", "002"],
            "linkdt": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "linkenddt": [None, dt.date(2021, 2, 1)],
        },
        schema={
            "permno": pl.Int64,
            "gvkey": pl.String,
            "linkdt": pl.Date,
            "linkenddt": pl.Date,
        },
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="conn"
        ),
        patch("tidyfinance.download_wrds._read_sql", return_value=sql_result),
        patch("tidyfinance.download_wrds.disconnect_connection") as mock_disc,
    ):
        result = download_data_wrds_ccm_links()

    mock_disc.assert_called_once_with("conn")
    assert set(["permno", "gvkey", "linkdt", "linkenddt"]).issubset(
        result.columns
    )
    assert result["permno"].to_list() == [1, 2]
    # Missing linkenddt is replaced with today's date
    assert result["linkenddt"][0] is not None
    assert result["linkenddt"][1] == dt.date(2021, 2, 1)


def test_passes_custom_link_filters_to_the_ccm_query():
    """Test passes custom link filters to the CCM query."""
    # With linktype="LU" and linkprim="C" filters applied at SQL level,
    # only the LU+C row would be returned.
    sql_result = pl.DataFrame(
        {
            "permno": [3],
            "gvkey": ["003"],
            "linkdt": [dt.date(2020, 3, 1)],
            "linkenddt": [dt.date(2021, 3, 1)],
        },
        schema={
            "permno": pl.Int64,
            "gvkey": pl.String,
            "linkdt": pl.Date,
            "linkenddt": pl.Date,
        },
    )

    captured = {}

    def fake_read_sql(query, conn, *a, **kw):
        captured["query"] = query
        return sql_result

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="conn"
        ),
        patch(
            "tidyfinance.download_wrds._read_sql", side_effect=fake_read_sql
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
    ):
        result = download_data_wrds_ccm_links(linktype=["LU"], linkprim=["C"])

    assert len(result) == 1
    assert result["permno"][0] == 3
    assert result["gvkey"][0] == "003"
    assert "'LU'" in captured["query"]
    assert "'C'" in captured["query"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
