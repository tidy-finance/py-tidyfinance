"""Tests for download_data_wrds_compustat."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import download_data_wrds_compustat  # noqa: E402


def test_dataset_is_required_and_validated():
    """Test dataset is required and validated."""
    with pytest.raises((ValueError, TypeError)):
        download_data_wrds_compustat()

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection",
            return_value="con",
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
    ):
        with pytest.raises(ValueError, match="Invalid dataset"):
            download_data_wrds_compustat("bad")


def test_annual_data_are_downloaded_and_transformed():
    """Test annual data are downloaded and transformed."""
    funda = pl.DataFrame(
        {
            "gvkey": ["001", "001", "002"],
            "datadate": [
                dt.date(2020, 12, 31),
                dt.date(2019, 12, 31),
                dt.date(2020, 12, 31),
            ],
            "seq": [10.0, 9.0, None],
            "ceq": [None, None, 20.0],
            "at": [100.0, 50.0, 30.0],
            "lt": [70.0, 40.0, 15.0],
            "txditc": [1.0, None, None],
            "txdb": [None, 1.0, None],
            "itcb": [None, 1.0, None],
            "pstkrv": [2.0, None, None],
            "pstkl": [None, 1.0, None],
            "pstk": [None, None, 1.0],
            "capx": [1.0, 1.0, 1.0],
            "oancf": [1.0, 1.0, 1.0],
            "sale": [20.0, 18.0, 25.0],
            "cogs": [5.0, 4.0, 6.0],
            "xint": [1.0, 1.0, 2.0],
            "xsga": [2.0, 2.0, 3.0],
            "ib": [3.0, 3.0, 4.0],
            "curcd": ["USD", "USD", "CAD"],
            "aodo": [7, 8, 9],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=funda),
    ):
        out = download_data_wrds_compustat(
            "compustat_annual",
            "2019-01-01",
            "2020-12-31",
            additional_columns=["aodo"],
            only_usd=True,
        )

    out_2020 = out.filter(pl.col("datadate") == dt.date(2020, 12, 31))
    assert len(out) == 2
    assert out_2020["gvkey"][0] == "001"
    assert out_2020["be"][0] == 9
    assert out_2020["op"][0] == pytest.approx(12 / 9)
    assert out_2020["inv"][0] == pytest.approx(1.0)
    assert out_2020["aodo"][0] == 7


def test_annual_data_handle_pi_and_invalid_lagged_assets():
    """Test annual data handle pi and invalid lagged assets."""
    funda = pl.DataFrame(
        {
            "gvkey": ["001", "001"],
            "datadate": [dt.date(2019, 12, 31), dt.date(2020, 12, 31)],
            "seq": [10.0, 12.0],
            "ceq": [None, None],
            "at": [0.0, 10.0],
            "lt": [1.0, 1.0],
            "txditc": [None, None],
            "txdb": [None, None],
            "itcb": [None, None],
            "pstkrv": [None, None],
            "pstkl": [None, None],
            "pstk": [None, None],
            "capx": [1.0, 1.0],
            "oancf": [1.0, 1.0],
            "sale": [2.0, 3.0],
            "cogs": [None, None],
            "xint": [None, None],
            "xsga": [None, None],
            "ib": [1.0, 1.0],
            "curcd": ["CAD", "CAD"],
            "pi": [5, 5],
        },
        schema_overrides={
            "ceq": pl.Float64,
            "txditc": pl.Float64,
            "txdb": pl.Float64,
            "itcb": pl.Float64,
            "pstkrv": pl.Float64,
            "pstkl": pl.Float64,
            "pstk": pl.Float64,
            "cogs": pl.Float64,
            "xint": pl.Float64,
            "xsga": pl.Float64,
        },
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=funda),
    ):
        out = download_data_wrds_compustat(
            "compustat_annual",
            "2019-01-01",
            "2020-12-31",
            additional_columns=["pi"],
        )

    assert len(out) == 2
    assert "pi" in out.columns
    inv_2020 = out.filter(pl.col("datadate") == dt.date(2020, 12, 31))["inv"][0]
    assert inv_2020 is None


def test_quarterly_data_are_cleaned_and_filtered():
    """Test quarterly data are cleaned and filtered."""
    fundq = pl.DataFrame(
        {
            "gvkey": ["001", "001", "001", None, "002"],
            "datadate": [
                dt.date(2020, 3, 31),
                dt.date(2020, 3, 31),
                dt.date(2020, 6, 30),
                dt.date(2020, 6, 30),
                dt.date(2020, 3, 31),
            ],
            "rdq": [
                dt.date(2020, 4, 30),
                dt.date(2020, 3, 1),
                None,
                None,
                None,
            ],
            "fqtr": [1, 1, 2, 2, 1],
            "fyearq": [2020, 2020, 2020, 2020, 2020],
            "atq": [10, 11, 12, 12, 13],
            "ceqq": [8, 9, 10, 10, 11],
            "curcdq": ["USD", "USD", "USD", "USD", "CAD"],
            "xrdq": [1, 2, 3, 4, 5],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=fundq),
    ):
        out = download_data_wrds_compustat(
            "compustat_quarterly",
            "2020-01-01",
            "2020-12-31",
            additional_columns=["xrdq"],
            only_usd=True,
        )

    assert out["gvkey"].to_list() == ["001", "001"]
    assert out["xrdq"].to_list() == [1, 3]
    assert out.columns == [
        "gvkey",
        "date",
        "datadate",
        "atq",
        "ceqq",
        "xrdq",
    ]


def test_quarterly_base_columns_can_be_requested():
    """Test base quarterly columns are returned when requested."""
    fundq = pl.DataFrame(
        {
            "gvkey": ["001", "002"],
            "datadate": [dt.date(2020, 3, 31), dt.date(2020, 3, 31)],
            "rdq": [dt.date(2020, 4, 30), None],
            "fqtr": [1, 1],
            "fyearq": [2020, 2020],
            "atq": [10, 13],
            "ceqq": [8, 11],
            "curcdq": ["USD", "CAD"],
        }
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=fundq),
    ):
        out = download_data_wrds_compustat(
            "compustat_quarterly",
            "2020-01-01",
            "2020-12-31",
            additional_columns=["rdq", "fyearq", "fqtr", "curcdq"],
        )

    assert out.columns == [
        "gvkey",
        "date",
        "datadate",
        "atq",
        "ceqq",
        "rdq",
        "fyearq",
        "fqtr",
        "curcdq",
    ]
    assert out["fyearq"].to_list() == [2020, 2020]
    assert out["fqtr"].to_list() == [1, 1]
    assert out["curcdq"].to_list() == ["USD", "CAD"]
    assert out["rdq"][0] == dt.date(2020, 4, 30)


def test_quarterly_data_can_return_non_usd_observations():
    """Test quarterly data can return non-USD observations."""
    fundq = pl.DataFrame(
        {
            "gvkey": ["002"],
            "datadate": [dt.date(2020, 3, 31)],
            "rdq": [None],
            "fqtr": [1],
            "fyearq": [2020],
            "atq": [13],
            "ceqq": [11],
            "curcdq": ["CAD"],
        },
        schema_overrides={"rdq": pl.Date},
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=fundq),
    ):
        out = download_data_wrds_compustat(
            "compustat_quarterly", "2020-01-01", "2020-12-31"
        )

    assert out["gvkey"][0] == "002"


def test_deprecated_arguments_are_supported():
    """Test deprecated arguments are supported."""
    funda = pl.DataFrame(
        {
            "gvkey": ["001"],
            "datadate": [dt.date(2020, 12, 31)],
            "seq": [10.0],
            "ceq": [None],
            "at": [100.0],
            "lt": [70.0],
            "txditc": [None],
            "txdb": [None],
            "itcb": [None],
            "pstkrv": [None],
            "pstkl": [None],
            "pstk": [None],
            "capx": [1.0],
            "oancf": [1.0],
            "sale": [20.0],
            "cogs": [5.0],
            "xint": [1.0],
            "xsga": [2.0],
            "ib": [3.0],
            "curcd": ["USD"],
        },
        schema_overrides={
            "ceq": pl.Float64,
            "txditc": pl.Float64,
            "txdb": pl.Float64,
            "itcb": pl.Float64,
            "pstkrv": pl.Float64,
            "pstkl": pl.Float64,
            "pstk": pl.Float64,
        },
    )

    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
        patch("tidyfinance.download_wrds._read_sql", return_value=funda),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            out = download_data_wrds_compustat(
                type="wrds_compustat_annual",
                start_date="2020-01-01",
                end_date="2020-12-31",
            )
        assert out["gvkey"][0] == "001"

        with pytest.warns(DeprecationWarning, match="deprecated"):
            download_data_wrds_compustat(
                "compustat_annual",
                "2020-01-01",
                "2020-12-31",
                only_us=True,
            )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            download_data_wrds_compustat(
                "wrds_compustat_annual", "2020-01-01", "2020-12-31"
            )


def test_defensive_unsupported_branch_is_covered():
    """Test defensive unsupported branch is covered."""
    with (
        patch(
            "tidyfinance.download_wrds.get_wrds_connection", return_value="con"
        ),
        patch("tidyfinance.download_wrds.disconnect_connection"),
    ):
        with pytest.raises(ValueError, match="Invalid dataset"):
            download_data_wrds_compustat("other")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
