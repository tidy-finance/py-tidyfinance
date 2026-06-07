"""Tests for download_data_wrds_compustat."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (  # noqa: E402
    _download_data_wrds_compustat,
)


def test_dataset_is_required_and_validated():
    """Test dataset is required and validated."""
    with pytest.raises((ValueError, TypeError)):
        _download_data_wrds_compustat()

    with patch(
        "tidyfinance.data_download.get_wrds_connection",
        return_value="con",
    ), patch("tidyfinance.data_download.disconnect_connection"):
        with pytest.raises(ValueError, match="Invalid dataset"):
            _download_data_wrds_compustat("bad")


def test_annual_data_are_downloaded_and_transformed():
    """Test annual data are downloaded and transformed."""
    funda = pd.DataFrame(
        {
            "gvkey": ["001", "001", "002"],
            "datadate": pd.to_datetime(
                ["2020-12-31", "2019-12-31", "2020-12-31"]
            ),
            "seq": [10, 9, np.nan],
            "ceq": [np.nan, np.nan, 20],
            "at": [100, 50, 30],
            "lt": [70, 40, 15],
            "txditc": [1, np.nan, np.nan],
            "txdb": [np.nan, 1, np.nan],
            "itcb": [np.nan, 1, np.nan],
            "pstkrv": [2, np.nan, np.nan],
            "pstkl": [np.nan, 1, np.nan],
            "pstk": [np.nan, np.nan, 1],
            "capx": [1, 1, 1],
            "oancf": [1, 1, 1],
            "sale": [20, 18, 25],
            "cogs": [5, 4, 6],
            "xint": [1, 1, 2],
            "xsga": [2, 2, 3],
            "ib": [3, 3, 4],
            "curcd": ["USD", "USD", "CAD"],
            "aodo": [7, 8, 9],
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=funda
    ):
        out = _download_data_wrds_compustat(
            "compustat_annual",
            "2019-01-01",
            "2020-12-31",
            additional_columns=["aodo"],
            only_usd=True,
        )

    out_2020 = out[out["datadate"] == pd.Timestamp("2020-12-31")]
    assert len(out) == 2
    assert out_2020["gvkey"].iloc[0] == "001"
    assert out_2020["be"].iloc[0] == 9
    assert out_2020["op"].iloc[0] == pytest.approx(12 / 9)
    assert out_2020["inv"].iloc[0] == pytest.approx(1.0)
    assert out_2020["aodo"].iloc[0] == 7


def test_annual_data_handle_pi_and_invalid_lagged_assets():
    """Test annual data handle pi and invalid lagged assets."""
    funda = pd.DataFrame(
        {
            "gvkey": ["001", "001"],
            "datadate": pd.to_datetime(["2019-12-31", "2020-12-31"]),
            "seq": [10, 12],
            "ceq": [np.nan, np.nan],
            "at": [0, 10],
            "lt": [1, 1],
            "txditc": [np.nan, np.nan],
            "txdb": [np.nan, np.nan],
            "itcb": [np.nan, np.nan],
            "pstkrv": [np.nan, np.nan],
            "pstkl": [np.nan, np.nan],
            "pstk": [np.nan, np.nan],
            "capx": [1, 1],
            "oancf": [1, 1],
            "sale": [2, 3],
            "cogs": [np.nan, np.nan],
            "xint": [np.nan, np.nan],
            "xsga": [np.nan, np.nan],
            "ib": [1, 1],
            "curcd": ["CAD", "CAD"],
            "pi": [5, 5],
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=funda
    ):
        out = _download_data_wrds_compustat(
            "compustat_annual",
            "2019-01-01",
            "2020-12-31",
            additional_columns=["pi"],
        )

    assert len(out) == 2
    assert "pi" in out.columns
    inv_2020 = out.loc[
        out["datadate"] == pd.Timestamp("2020-12-31"), "inv"
    ].iloc[0]
    assert pd.isna(inv_2020)


def test_quarterly_data_are_cleaned_and_filtered():
    """Test quarterly data are cleaned and filtered."""
    fundq = pd.DataFrame(
        {
            "gvkey": ["001", "001", "001", None, "002"],
            "datadate": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-03-31",
                    "2020-06-30",
                    "2020-06-30",
                    "2020-03-31",
                ]
            ),
            "rdq": [
                pd.Timestamp("2020-04-30"),
                pd.Timestamp("2020-03-01"),
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
            "fqtr": [1, 1, 2, 2, 1],
            "fyearq": [2020, 2020, 2020, 2020, 2020],
            "atq": [10, 11, 12, 12, 13],
            "ceqq": [8, 9, 10, 10, 11],
            "curcdq": ["USD", "USD", "USD", "USD", "CAD"],
            "xrdq": [1, 2, 3, 4, 5],
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=fundq
    ):
        out = _download_data_wrds_compustat(
            "compustat_quarterly",
            "2020-01-01",
            "2020-12-31",
            additional_columns=["xrdq"],
            only_usd=True,
        )

    assert list(out["gvkey"]) == ["001", "001"]
    assert list(out["xrdq"]) == [1, 3]
    assert list(out.columns) == [
        "gvkey",
        "date",
        "datadate",
        "atq",
        "ceqq",
        "xrdq",
    ]


def test_quarterly_data_can_return_non_usd_observations():
    """Test quarterly data can return non-USD observations."""
    fundq = pd.DataFrame(
        {
            "gvkey": ["002"],
            "datadate": pd.to_datetime(["2020-03-31"]),
            "rdq": [pd.NaT],
            "fqtr": [1],
            "fyearq": [2020],
            "atq": [13],
            "ceqq": [11],
            "curcdq": ["CAD"],
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=fundq
    ):
        out = _download_data_wrds_compustat(
            "compustat_quarterly", "2020-01-01", "2020-12-31"
        )

    assert out["gvkey"].iloc[0] == "002"


def test_deprecated_arguments_are_supported():
    """Test deprecated arguments are supported."""
    funda = pd.DataFrame(
        {
            "gvkey": ["001"],
            "datadate": pd.to_datetime(["2020-12-31"]),
            "seq": [10],
            "ceq": [np.nan],
            "at": [100],
            "lt": [70],
            "txditc": [np.nan],
            "txdb": [np.nan],
            "itcb": [np.nan],
            "pstkrv": [np.nan],
            "pstkl": [np.nan],
            "pstk": [np.nan],
            "capx": [1],
            "oancf": [1],
            "sale": [20],
            "cogs": [5],
            "xint": [1],
            "xsga": [2],
            "ib": [3],
            "curcd": ["USD"],
        }
    )

    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch(
        "tidyfinance.data_download.disconnect_connection"
    ), patch(
        "tidyfinance.data_download.pd.read_sql", return_value=funda
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = _download_data_wrds_compustat(
                type="wrds_compustat_annual",
                start_date="2020-01-01",
                end_date="2020-12-31",
            )
        assert out["gvkey"].iloc[0] == "001"

        with pytest.warns(DeprecationWarning, match="deprecated"):
            _download_data_wrds_compustat(
                "compustat_annual",
                "2020-01-01",
                "2020-12-31",
                only_us=True,
            )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            _download_data_wrds_compustat(
                "wrds_compustat_annual", "2020-01-01", "2020-12-31"
            )


def test_defensive_unsupported_branch_is_covered():
    """Test defensive unsupported branch is covered."""
    with patch(
        "tidyfinance.data_download.get_wrds_connection", return_value="con"
    ), patch("tidyfinance.data_download.disconnect_connection"):
        with pytest.raises(ValueError, match="Invalid dataset"):
            _download_data_wrds_compustat("other")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
