"""Tests for add_lagged_columns."""

import datetime as dt
import os
import sys

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.lagging import add_lagged_columns  # noqa: E402


def test_exact_lag_with_by_returns_correct_lagged_values():
    """Test exact lag with by returns correct lagged values."""
    dates = pl.date_range(
        dt.date(2023, 1, 1), dt.date(2023, 4, 1), "1mo", eager=True
    )
    data = pl.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": dates.to_list() * 2,
            "size": [float(i) for i in range(1, 9)],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo", by="permno")
    g1 = result.filter(pl.col("permno") == 1)
    assert g1["size_lag"][0] is None
    assert g1["size_lag"][1] == 1
    assert g1["size_lag"][2] == 2


def test_exact_lag_without_by_returns_correct_values():
    """Test exact lag without by (by=None) returns correct values."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo")
    assert result["size_lag"][0] is None
    assert result["size_lag"][1] == 1
    assert result["size_lag"][2] == 2


def test_window_lag_handles_all_src_date_conditions():
    """Test window lag: NA, in-window, and below-lower-bound cases."""
    data = pl.DataFrame(
        {
            "date": [
                dt.date(2023, 1, 1),
                dt.date(2023, 2, 1),
                dt.date(2023, 6, 1),
            ],
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo", max_lag="2mo")
    # Jan: no source at all -> null
    assert result["size_lag"][0] is None
    # Feb: src_date Jan within window [Dec, Jan] -> 1
    assert result["size_lag"][1] == 1
    # Jun: closest src is Feb which is below the [Apr, May] window -> null
    assert result["size_lag"][2] is None


def test_drop_na_skips_na_source_rows_in_window_lag():
    """Test drop_na: missing source rows are skipped when drop_na=True."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 5, 1), "1mo", eager=True
            ),
            "size": [1.0, None, None, 4.0, 5.0],
        }
    )
    r_keep = add_lagged_columns(data, cols="size", lag="1mo", max_lag="3mo")
    r_drop = add_lagged_columns(
        data, cols="size", lag="1mo", max_lag="3mo", drop_na=True
    )
    # Apr (index 3): window [Jan, Mar]; without drop_na, closest = Mar
    # (missing)
    assert r_keep["size_lag"][3] is None
    # Apr with drop_na: missing sources skipped, closest valid = Jan -> 1
    assert r_drop["size_lag"][3] == 1


def test_ff_adjustment_without_by_uses_year_grouping_only():
    """Test ff_adjustment without by uses year grouping only."""
    data = pl.DataFrame(
        {
            "date": [
                dt.date(2022, 6, 1),
                dt.date(2022, 12, 1),
                dt.date(2023, 6, 1),
                dt.date(2023, 7, 1),
            ],
            "size": [10.0, 20.0, 30.0, 40.0],
        }
    )
    # 2022: Dec kept (20), used Jul-2023 to Jun-2024.
    result = add_lagged_columns(data, cols="size", ff_adjustment=True)
    assert result["size_lag"].to_list() == [None, None, None, 20]


def test_ff_adjustment_with_by_updates_in_july_for_every_fiscal_year_end():
    """Test ff_adjustment with by updates in July for every fiscal
    year-end."""
    dates = pl.date_range(
        dt.date(2020, 1, 1), dt.date(2022, 12, 1), "1mo", eager=True
    )
    data = pl.DataFrame(
        {
            "permno": [1] * len(dates) + [2] * len(dates),
            "date": dates.to_list() * 2,
        }
    )
    fye = ((pl.col("permno") == 1) & (pl.col("date").dt.month() == 12)) | (
        (pl.col("permno") == 2) & (pl.col("date").dt.month() == 6)
    )
    data = data.with_columns(
        be=pl.when(fye).then(
            pl.col("date").dt.year().cast(pl.Float64) * 10 + pl.col("permno")
        )
    )

    result = add_lagged_columns(
        data,
        cols="be",
        by="permno",
        drop_na=True,
        ff_adjustment=True,
    )

    assert result.columns == ["permno", "date", "be", "be_lag"]
    assert result.height == data.height
    jul21_jun22 = result.filter(
        pl.col("date").is_between(dt.date(2021, 7, 1), dt.date(2022, 6, 1))
    )
    assert jul21_jun22.filter(pl.col("permno") == 1)[
        "be_lag"
    ].to_list() == [20201] * 12
    assert jul21_jun22.filter(pl.col("permno") == 2)[
        "be_lag"
    ].to_list() == [20202] * 12
    assert result.filter(pl.col("date") == dt.date(2022, 7, 1))[
        "be_lag"
    ].to_list() == [20211, 20212]


def test_ff_adjustment_ignores_lags_with_a_message(capsys):
    """Test ff_adjustment ignores lags with a message."""
    data = pl.DataFrame(
        {
            "date": [dt.date(2020, 12, 1), dt.date(2021, 7, 1)],
            "size": [1.0, 2.0],
        }
    )
    result = add_lagged_columns(
        data, cols="size", lag="1mo", ff_adjustment=True
    )
    assert "'lag' is ignored" in capsys.readouterr().err
    assert result["size_lag"].to_list() == [None, 1]

    add_lagged_columns(
        data, cols="size", lag="1mo", max_lag="2mo", ff_adjustment=True
    )
    assert "'lag' and 'max_lag' are ignored" in capsys.readouterr().err

    add_lagged_columns(data, cols="size", ff_adjustment=True)
    assert capsys.readouterr().err == ""


def test_lag_is_required_without_ff_adjustment():
    """Test lag is required without ff_adjustment."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="required"):
        add_lagged_columns(data, cols="size")




def test_error_when_date_column_is_absent_from_data():
    """Test error when date column is absent from data."""
    with pytest.raises(ValueError, match="date"):
        add_lagged_columns(pl.DataFrame({"x": [1]}), cols="x", lag="1mo")


def test_error_when_lag_is_negative():
    """Test error when lag is negative."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="non-negative"):
        add_lagged_columns(data, cols="size", lag=-1)


def test_error_when_max_lag_is_less_than_lag():
    """Test error when max_lag is less than lag."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="max_lag"):
        add_lagged_columns(data, cols="size", lag="3mo", max_lag="1mo")


def test_error_when_requested_column_is_absent_from_data():
    """Test error when a requested column is absent from data."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(data, cols="no_such_col", lag="1mo")


def test_error_when_by_column_is_absent_from_data():
    """Test error when a by column is absent from data."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(data, cols="size", lag="1mo", by="no_such_grp")


def test_error_when_join_key_is_not_unique():
    """Test error when join key is not unique."""
    data = pl.DataFrame(
        {
            "date": [dt.date(2023, 1, 1)] * 2,
            "size": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        add_lagged_columns(data, cols="size", lag="1mo")


def test_warns_when_upper_helper_column_already_exists():
    """Test add_lagged_columns warns when '_upper' already exists and
    max_lag is used."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
            "_upper": [0, 0, 0],
        }
    )
    with pytest.warns(UserWarning, match="overwrites.*'_upper'"):
        result = add_lagged_columns(
            data, cols="size", lag="1mo", max_lag="2mo"
        )
    assert result["size_lag"].to_list() == [None, 1.0, 2.0]
    assert "_upper" not in result.columns


def test_data_options_specifies_date_column_name():
    """Test data_options dict specifies the date column name."""
    from tidyfinance.portfolios import data_options

    data = pl.DataFrame(
        {
            "my_date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
        }
    )
    opts = data_options(date="my_date")
    result = add_lagged_columns(data, cols="size", lag="1mo", data_options=opts)
    assert result["size_lag"][0] is None
    assert result["size_lag"][1] == 1


def _daily_values():
    return pl.DataFrame(
        {
            "date": [
                dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(10)
            ],
            "value": list(range(1, 11)),
        }
    )


def test_integer_lag_counts_days():
    result = add_lagged_columns(_daily_values(), cols="value", lag=1)
    assert result["value_lag"].to_list() == [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_window_lag_works_with_integer_lags():
    result = add_lagged_columns(_daily_values(), cols="value", lag=1, max_lag=3)
    assert result["value_lag"].to_list() == [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
