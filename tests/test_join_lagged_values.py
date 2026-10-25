"""Tests for join_lagged_values."""

import datetime as dt
import os
import sys

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.lagging import join_lagged_values  # noqa: E402


def test_normal_join_adds_lagged_columns_for_matching_date_windows():
    """Test normal join adds lagged columns for matching date windows."""
    orig = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "date": [
                dt.date(2023, 2, 1),
                dt.date(2023, 4, 1),
                dt.date(2023, 5, 1),
            ],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [10.0],
            "y": [20.0],
        }
    )
    result = join_lagged_values(
        orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
    )
    # T = Jan-2023: window [Feb-2023, Apr-2023]
    # Feb, Apr -> match; May -> no match
    assert result["x"][0] == 10
    assert result["x"][1] == 10
    assert result["x"][2] is None
    assert result["y"][0] == 20


def test_ff_adjustment_uses_the_last_observation_per_id_year():
    """Test ff_adjustment uses the last observation per id-year."""
    orig = pl.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "date": [
                dt.date(2023, 6, 1),
                dt.date(2023, 7, 1),
                dt.date(2024, 6, 1),
                dt.date(2024, 7, 1),
            ],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1, 1],
            "date": [dt.date(2022, 6, 1), dt.date(2022, 12, 1)],
            "x": [5.0, 10.0],
        }
    )
    # Jun-2022 is dropped, Dec-2022 (last in 2022) is used Jul-2023 to
    # Jun-2024.
    result = join_lagged_values(orig, new, id_keys="id", ff_adjustment=True)
    assert result["x"].to_list() == [None, 10, 10, None]


def test_ff_adjustment_updates_in_july_for_every_fiscal_year_end():
    """Test ff_adjustment updates in July for every fiscal year-end."""
    dates = pl.date_range(
        dt.date(2021, 1, 1), dt.date(2022, 12, 1), "1mo", eager=True
    )
    orig = pl.DataFrame(
        {
            "id": ["dec"] * len(dates) + ["jun"] * len(dates),
            "date": dates.to_list() * 2,
        }
    )
    new = pl.DataFrame(
        {
            "id": ["dec", "jun"],
            "date": [dt.date(2020, 12, 1), dt.date(2020, 6, 1)],
            "x": [1.0, 2.0],
        }
    )
    result = join_lagged_values(orig, new, id_keys="id", ff_adjustment=True)

    in_window = pl.col("date").is_between(
        dt.date(2021, 7, 1), dt.date(2022, 6, 1)
    )
    assert result.filter(in_window & (pl.col("id") == "dec"))[
        "x"
    ].to_list() == [1] * 12
    assert result.filter(in_window & (pl.col("id") == "jun"))[
        "x"
    ].to_list() == [2] * 12
    assert result.filter(~in_window)["x"].is_null().all()


def test_ff_adjustment_works_with_end_of_month_dates():
    """Test ff_adjustment works with end-of-month dates."""
    orig = pl.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "date": [
                dt.date(2021, 6, 30),
                dt.date(2021, 7, 31),
                dt.date(2022, 6, 30),
                dt.date(2022, 7, 31),
            ],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2020, 9, 30)],
            "x": [3.0],
        }
    )
    result = join_lagged_values(orig, new, id_keys="id", ff_adjustment=True)
    assert result["x"].to_list() == [None, 3, 3, None]


def test_ff_adjustment_ignores_lags_with_a_message(capsys):
    """Test ff_adjustment ignores lags with a message."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2021, 7, 1)]})
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2020, 12, 1)],
            "x": [1.0],
        }
    )
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag="1mo",
        max_lag="2mo",
        ff_adjustment=True,
    )
    assert "'min_lag' and 'max_lag' are ignored" in capsys.readouterr().err
    assert result["x"].to_list() == [1]


def test_min_lag_and_max_lag_are_required_without_ff_adjustment():
    """Test min_lag and max_lag are required without ff_adjustment."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2021, 7, 1)]})
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2020, 12, 1)],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="required"):
        join_lagged_values(orig, new, id_keys="id", min_lag="1mo")


def test_non_default_date_col_uses_specified_column():
    """Test non-default date_col uses the specified column."""
    orig = pl.DataFrame(
        {
            "id": [1, 1],
            "my_date": [dt.date(2023, 2, 1), dt.date(2023, 5, 1)],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "my_date": [dt.date(2023, 1, 1)],
            "x": [7.0],
        }
    )
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag="1mo",
        max_lag="3mo",
        date_col="my_date",
    )
    assert result["x"][0] == 7
    assert result["x"][1] is None


def test_error_when_id_keys_is_not_a_character_vector():
    """Test error when id_keys is not a string or list of strings."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2023, 1, 1)]})
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="string"):
        join_lagged_values(orig, new, id_keys=1, min_lag="1mo", max_lag="3mo")


def test_error_when_date_column_missing_from_original_data():
    """Test error when date column missing from original_data."""
    orig = pl.DataFrame({"id": [1], "other": [1]})
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="original_data"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_error_when_date_column_missing_from_new_data():
    """Test error when date column missing from new_data."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2023, 1, 1)]})
    new = pl.DataFrame({"id": [1], "other": [1]})
    with pytest.raises(ValueError, match="new_data"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_error_when_id_keys_column_missing_from_original_data():
    """Test error when id_keys column missing from original_data."""
    orig = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "val": [1.0]})
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="original_data"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_error_when_id_keys_column_missing_from_new_data():
    """Test error when id_keys column missing from new_data."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2023, 1, 1)]})
    new = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "x": [1.0]})
    with pytest.raises(ValueError, match="new_data"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_error_when_new_data_has_no_columns_besides_id_keys_and_date():
    """Test error when new_data has no columns besides id_keys and date."""
    orig = pl.DataFrame({"id": [1], "date": [dt.date(2023, 1, 1)]})
    new = pl.DataFrame({"id": [1], "date": [dt.date(2023, 1, 1)]})
    with pytest.raises(ValueError, match="columns besides"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_error_when_new_data_column_already_exists_in_original_data():
    """Test error when new_data column already exists in original_data."""
    orig = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [0.0],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="already exist"):
        join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )


def test_data_options_specifies_date_column_name():
    """Test data_options dict specifies the date column name."""
    from tidyfinance.portfolios import data_options

    orig = pl.DataFrame(
        {
            "id": [1, 1],
            "my_date": [dt.date(2023, 2, 1), dt.date(2023, 5, 1)],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "my_date": [dt.date(2023, 1, 1)],
            "x": [7.0],
        }
    )
    opts = data_options(date="my_date")
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag="1mo",
        max_lag="3mo",
        data_options=opts,
    )
    assert result["x"][0] == 7
    assert result["x"][1] is None


def test_warns_when_lower_helper_column_already_exists():
    """Test join_lagged_values warns when '_lower' already exists in
    new_data."""
    orig = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 2, 1)],
        }
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2023, 1, 1)],
            "x": [10.0],
            "_lower": [0],
        }
    )
    with pytest.warns(UserWarning, match="overwrites.*'_lower'"):
        result = join_lagged_values(
            orig, new, id_keys="id", min_lag="1mo", max_lag="3mo"
        )
    assert result.columns == ["id", "date", "x"]
    assert result["x"].to_list() == [10.0]


def test_warns_when_new_data_has_year_and_ff_adjustment_is_true():
    """Test join_lagged_values warns when new_data already has '_year'
    and ff_adjustment is True."""
    orig = pl.DataFrame(
        {"id": [1, 1], "date": [dt.date(2021, 6, 1), dt.date(2021, 7, 1)]}
    )
    new = pl.DataFrame(
        {
            "id": [1],
            "date": [dt.date(2020, 12, 1)],
            "bm": [0.5],
            "_year": [None],
        }
    )
    with pytest.warns(UserWarning, match="overwrites.*'_year'"):
        result = join_lagged_values(
            orig, new, id_keys="id", ff_adjustment=True
        )
    assert result.columns == ["id", "date", "bm"]
    assert result["bm"].to_list() == [None, 0.5]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
