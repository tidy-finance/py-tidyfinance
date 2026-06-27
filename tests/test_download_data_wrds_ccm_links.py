"""Tests for download_data_wrds_ccm_links."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import _download_data_wrds_ccm_links  # noqa: E402


def test_downloads_default_ccm_links_and_replaces_missing_end_dates():
    """Test downloads default CCM links and replaces missing end dates."""
    sql_result = pd.DataFrame(
        {
            "permno": [1, 2],
            "gvkey": ["001", "002"],
            "linkdt": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "linkenddt": [pd.NaT, pd.Timestamp("2021-02-01")],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="conn"
        ),
        patch("tidyfinance.download_wrds.pd.read_sql", return_value=sql_result),
        patch("tidyfinance.download_wrds.disconnect_connection") as mock_disc,
    ):
        result = _download_data_wrds_ccm_links()

    mock_disc.assert_called_once_with("conn")
    assert set(["permno", "gvkey", "linkdt", "linkenddt"]).issubset(
        result.columns
    )
    assert list(result["permno"]) == [1, 2]
    # Missing linkenddt is replaced with today's date
    assert pd.notna(result["linkenddt"].iloc[0])
    assert result["linkenddt"].iloc[1] == pd.Timestamp("2021-02-01")


def test_passes_custom_link_filters_to_the_ccm_query():
    """Test passes custom link filters to the CCM query."""
    # With linktype="LU" and linkprim="C" filters applied at SQL level,
    # only the LU+C row would be returned.
    sql_result = pd.DataFrame(
        {
            "permno": [3],
            "gvkey": ["003"],
            "linkdt": pd.to_datetime(["2020-03-01"]),
            "linkenddt": pd.to_datetime(["2021-03-01"]),
        }
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
            "tidyfinance.download_wrds.pd.read_sql", side_effect=fake_read_sql
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
    ):
        result = _download_data_wrds_ccm_links(linktype=["LU"], linkprim=["C"])

    assert len(result) == 1
    assert result["permno"].iloc[0] == 3
    assert result["gvkey"].iloc[0] == "003"
    assert "'LU'" in captured["query"]
    assert "'C'" in captured["query"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
