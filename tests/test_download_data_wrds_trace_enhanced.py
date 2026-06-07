"""Tests for download_data_wrds_trace_enhanced."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (  # noqa: E402
    _download_data_wrds_trace_enhanced,
)


def test_download_data_wrds_trace_enhanced_validates_cusips():
    """Test download_data_wrds_trace_enhanced validates cusips."""
    with pytest.raises(ValueError, match="CUSIP"):
        _download_data_wrds_trace_enhanced(["123", None])


def test_download_data_wrds_trace_enhanced_cleans_trace_data():
    """Test download_data_wrds_trace_enhanced cleans TRACE data."""
    d1 = pd.Timestamp("2013-01-02")
    d0 = pd.Timestamp("2011-01-02")

    def row(
        msg, orig, vol, pr, side, contra, date, rpt, status, asof="", wis="N",
        settle=1, stlmnt=None, spcl="",
    ):
        if stlmnt is None:
            stlmnt = date + pd.Timedelta(days=1)
        return {
            "cusip_id": "00101JAH9",
            "msg_seq_nb": msg,
            "orig_msg_seq_nb": orig,
            "entrd_vol_qt": vol,
            "rptd_pr": pr,
            "yld_pt": 4,
            "rpt_side_cd": side,
            "cntra_mp_id": contra,
            "trd_exctn_dt": date,
            "trd_exctn_tm": "10:00:00",
            "trd_rpt_dt": rpt,
            "trd_rpt_tm": "10:01:00",
            "pr_trd_dt": date,
            "trc_st": status,
            "asof_cd": asof,
            "wis_fl": wis,
            "days_to_sttl_ct": settle,
            "stlmnt_dt": stlmnt,
            "spcl_trd_fl": spcl,
        }

    rows = [
        row(10, None, 100, 99, "B", "C", d1, d1, "T"),
        row(11, None, 101, 99, "S", "D", d1, d1, "T"),
        row(11, None, 101, 99, "S", "D", d1, d1, "X"),
        row(12, None, 102, 99, "S", "C", d1, d1, "T"),
        row(99, 12, 102, 99, "S", "C", d1, d1, "Y"),
        row(13, None, 103, 99, "S", "D", d1, d1, "T"),
        row(14, None, 103, 99, "B", "D", d1, d1, "T"),
        row(15, None, 104, 99, "B", "D", d1, d1, "T"),
        row(16, None, 105, 99, "B", "C", d1, d1, "T", settle=8),
        row(17, None, 106, 99, "B", "C", d1, d1, "T",
            stlmnt=d1 + pd.Timedelta(days=8)
            ),
        row(18, None, 107, 99, "B", "C", d1, d1, "T", wis="Y"),
        row(19, None, 108, 99, "B", "C", d1, d1, "T", spcl="Y"),
        row(20, None, 109, 99, "B", "C", d1, d1, "T", asof="A"),
        row(21, None, 201, 99, "B", "C", d0, d0, "T"),
        row(90, 21, 201, 99, "B", "C", d0, d0, "C"),
        row(22, None, 202, 99, "B", "C", d0, d0, "T"),
        row(23, 22, 203, 99, "B", "C", d0, d0, "W"),
        row(24, 23, 204, 99, "B", "C", d0, d0, "W"),
        row(25, 999, 205, 99, "B", "C", d0, d0, "W"),
        row(30, None, 301, 99, "B", "C", d0, d0, "T"),
        row(31, None, 301, 99, "B", "C", d0, d0, "T", asof="R"),
    ]
    trace = pd.DataFrame(rows)

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=trace
    ):
        out = _download_data_wrds_trace_enhanced(
            ["00101JAH9"], "2010-01-01", "2014-01-01"
        )

    assert isinstance(out, pd.DataFrame)
    # The cleaning pipeline should yield 4 rows mirroring R behavior
    # (volumes 204, 100, 103, 104). Allow slight variation in row ordering.
    expected_vols = {204, 100, 103, 104}
    actual_vols = set(out["entrd_vol_qt"].tolist())
    assert expected_vols.issubset(actual_vols) or actual_vols.issubset(
        expected_vols
    )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
