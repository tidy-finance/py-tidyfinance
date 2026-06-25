"""Tests for add_lagged_columns."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import add_lagged_columns  # noqa: E402


def test_exact_lag_with_by_returns_correct_lagged_values():
    """Test exact lag with by returns correct lagged values."""
    data = pd.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": list(pd.date_range("2023-01-01", periods=4, freq="MS")) * 2,
            "size": [float(i) for i in range(1, 9)],
        }
    )
    result = add_lagged_columns(
        data, cols="size", lag=pd.DateOffset(months=1), by="permno"
    )
    g1 = result[result["permno"] == 1].reset_index(drop=True)
    assert pd.isna(g1["size_lag"].iloc[0])
    assert g1["size_lag"].iloc[1] == 1
    assert g1["size_lag"].iloc[2] == 2


def test_exact_lag_without_by_returns_correct_values():
    """Test exact lag without by (by=None) returns correct values."""
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(data, cols="size", lag=pd.DateOffset(months=1))
    assert pd.isna(result["size_lag"].iloc[0])
    assert result["size_lag"].iloc[1] == 1
    assert result["size_lag"].iloc[2] == 2


def test_window_lag_handles_all_src_date_conditions():
    """Test window lag: NA, in-window, and below-lower-bound cases."""
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-06-01"]),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=2),
    )
    # Jan: no source at all → NA
    assert pd.isna(result["size_lag"].iloc[0])
    # Feb: src_date Jan within window [Dec, Jan] → 1
    assert result["size_lag"].iloc[1] == 1
    # Jun: closest src is Feb which is below the [Apr, May] window → NA
    assert pd.isna(result["size_lag"].iloc[2])


def test_drop_na_skips_na_source_rows_in_window_lag():
    """Test drop_na: NaN source rows are skipped when drop_na=True."""
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
            "size": [1.0, float("nan"), float("nan"), 4.0, 5.0],
        }
    )
    r_keep = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
    )
    r_drop = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
        drop_na=True,
    )
    # Apr (index 3): window [Jan, Mar]; without drop_na, closest = Mar (NaN)
    assert pd.isna(r_keep["size_lag"].iloc[3])
    # Apr with drop_na: NaN sources skipped, closest valid = Jan → 1
    assert r_drop["size_lag"].iloc[3] == 1


def test_ff_adjustment_without_by_uses_year_grouping_only():
    """Test ff_adjustment without by uses year grouping only."""
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-01", "2022-12-01", "2023-06-01"]),
            "size": [10.0, 20.0, 30.0],
        }
    )
    result = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=6),
        ff_adjustment=True,
    )
    # ff: 2022 → Dec kept (20); 2023 → Jun kept (30).
    # Shifted +6m: Jun-2023 (20), Dec-2023 (30).
    jun23 = result["date"] == pd.Timestamp("2023-06-01")
    assert result.loc[jun23, "size_lag"].iloc[0] == 20


def test_non_default_date_col_is_respected():
    """Test non-default date_col uses the specified column name."""
    data = pd.DataFrame(
        {
            "my_date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=1),
        date_col="my_date",
    )
    assert pd.isna(result["size_lag"].iloc[0])
    assert result["size_lag"].iloc[1] == 1


def test_error_when_date_column_is_absent_from_data():
    """Test error when date column is absent from data."""
    with pytest.raises(ValueError, match="date"):
        add_lagged_columns(
            pd.DataFrame({"x": [1]}),
            cols="x",
            lag=pd.DateOffset(months=1),
        )


def test_error_when_lag_is_negative():
    """Test error when lag is negative."""
    data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "size": [1.0]})
    with pytest.raises(ValueError, match="non-negative"):
        add_lagged_columns(data, cols="size", lag=-1)


def test_error_when_max_lag_is_less_than_lag():
    """Test error when max_lag is less than lag."""
    data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "size": [1.0]})
    with pytest.raises(ValueError, match="max_lag"):
        add_lagged_columns(
            data,
            cols="size",
            lag=pd.DateOffset(months=3),
            max_lag=pd.DateOffset(months=1),
        )


def test_error_when_requested_column_is_absent_from_data():
    """Test error when a requested column is absent from data."""
    data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(
            data, cols="no_such_col", lag=pd.DateOffset(months=1)
        )


def test_error_when_by_column_is_absent_from_data():
    """Test error when a by column is absent from data."""
    data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(
            data,
            cols="size",
            lag=pd.DateOffset(months=1),
            by="no_such_grp",
        )


def test_error_when_join_key_is_not_unique():
    """Test error when join key is not unique."""
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01")] * 2,
            "size": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        add_lagged_columns(data, cols="size", lag=pd.DateOffset(months=1))


def test_error_when_upper_helper_column_already_exists():
    """Test error when '_upper' helper column already exists in data."""
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "size": [1.0, 2.0, 3.0],
            "_upper": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="_upper"):
        add_lagged_columns(
            data,
            cols="size",
            lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=2),
        )


def test_data_options_specifies_date_column_name():
    """Test data_options dict specifies the date column name."""
    from tidyfinance.core import data_options

    data = pd.DataFrame(
        {
            "my_date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "size": [1.0, 2.0, 3.0],
        }
    )
    opts = data_options(date="my_date")
    result = add_lagged_columns(
        data,
        cols="size",
        lag=pd.DateOffset(months=1),
        data_options=opts,
    )
    assert pd.isna(result["size_lag"].iloc[0])
    assert result["size_lag"].iloc[1] == 1


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
