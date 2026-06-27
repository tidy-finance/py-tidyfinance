"""Tests for join_lagged_values."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.lagging import join_lagged_values  # noqa: E402


def test_normal_join_adds_lagged_columns_for_matching_date_windows():
    """Test normal join adds lagged columns for matching date windows."""
    orig = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "date": pd.to_datetime(["2023-02-01", "2023-04-01", "2023-05-01"]),
        }
    )
    new = pd.DataFrame(
        {
            "id": [1],
            "date": pd.to_datetime(["2023-01-01"]),
            "x": [10.0],
            "y": [20.0],
        }
    )
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
    )
    # T = Jan-2023: window [Feb-2023, Apr-2023]
    # Feb, Apr → match; May → no match
    assert result["x"].iloc[0] == 10
    assert result["x"].iloc[1] == 10
    assert pd.isna(result["x"].iloc[2])
    assert result["y"].iloc[0] == 20


def test_ff_adjustment_removes_non_last_observations_per_id_year():
    """Test ff_adjustment removes non-last observations per id-year."""
    orig = pd.DataFrame(
        {
            "id": [1, 1],
            "date": pd.to_datetime(["2022-08-01", "2023-03-01"]),
        }
    )
    new = pd.DataFrame(
        {
            "id": [1, 1],
            "date": pd.to_datetime(["2022-06-01", "2022-12-01"]),
            "x": [5.0, 10.0],
        }
    )
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
        ff_adjustment=True,
    )
    # ff: Jun-2022 dropped, Dec-2022 kept (last in 2022).
    # Dec window: [Jan-2023, Mar-2023]
    #   Aug-2022: no match → NA
    #   Mar-2023: match    → 10
    assert pd.isna(result["x"].iloc[0])
    assert result["x"].iloc[1] == 10


def test_non_default_date_col_uses_specified_column():
    """Test non-default date_col uses the specified column."""
    orig = pd.DataFrame(
        {
            "id": [1, 1],
            "my_date": pd.to_datetime(["2023-02-01", "2023-05-01"]),
        }
    )
    new = pd.DataFrame(
        {
            "id": [1],
            "my_date": pd.to_datetime(["2023-01-01"]),
            "x": [7.0],
        }
    )
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
        date_col="my_date",
    )
    assert result["x"].iloc[0] == 7
    assert pd.isna(result["x"].iloc[1])


def test_error_when_id_keys_is_not_a_character_vector():
    """Test error when id_keys is not a string or list of strings."""
    orig = pd.DataFrame({"id": [1], "date": [pd.Timestamp("2023-01-01")]})
    new = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="string"):
        join_lagged_values(
            orig,
            new,
            id_keys=1,
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_date_column_missing_from_original_data():
    """Test error when date column missing from original_data."""
    orig = pd.DataFrame({"id": [1], "other": [1]})
    new = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="original_data"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_date_column_missing_from_new_data():
    """Test error when date column missing from new_data."""
    orig = pd.DataFrame({"id": [1], "date": [pd.Timestamp("2023-01-01")]})
    new = pd.DataFrame({"id": [1], "other": [1]})
    with pytest.raises(ValueError, match="new_data"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_id_keys_column_missing_from_original_data():
    """Test error when id_keys column missing from original_data."""
    orig = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "val": [1.0]})
    new = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="original_data"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_id_keys_column_missing_from_new_data():
    """Test error when id_keys column missing from new_data."""
    orig = pd.DataFrame({"id": [1], "date": [pd.Timestamp("2023-01-01")]})
    new = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "x": [1.0]})
    with pytest.raises(ValueError, match="new_data"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_new_data_has_no_columns_besides_id_keys_and_date():
    """Test error when new_data has no columns besides id_keys and date."""
    orig = pd.DataFrame({"id": [1], "date": [pd.Timestamp("2023-01-01")]})
    new = pd.DataFrame({"id": [1], "date": [pd.Timestamp("2023-01-01")]})
    with pytest.raises(ValueError, match="columns besides"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_error_when_new_data_column_already_exists_in_original_data():
    """Test error when new_data column already exists in original_data."""
    orig = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [0.0],
        }
    )
    new = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="already exist"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


def test_data_options_specifies_date_column_name():
    """Test data_options dict specifies the date column name."""
    from tidyfinance.portfolios import data_options

    orig = pd.DataFrame(
        {
            "id": [1, 1],
            "my_date": pd.to_datetime(["2023-02-01", "2023-05-01"]),
        }
    )
    new = pd.DataFrame(
        {
            "id": [1],
            "my_date": pd.to_datetime(["2023-01-01"]),
            "x": [7.0],
        }
    )
    opts = data_options(date="my_date")
    result = join_lagged_values(
        orig,
        new,
        id_keys="id",
        min_lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
        data_options=opts,
    )
    assert result["x"].iloc[0] == 7
    assert pd.isna(result["x"].iloc[1])


def test_error_when_lower_helper_column_already_exists():
    """Test error when '_lower' helper column already exists in new_data."""
    orig = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-02-01")],
        }
    )
    new = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.Timestamp("2023-01-01")],
            "x": [10.0],
            "_lower": [0],
        }
    )
    with pytest.raises(ValueError, match="_lower"):
        join_lagged_values(
            orig,
            new,
            id_keys="id",
            min_lag=pd.DateOffset(months=1),
            max_lag=pd.DateOffset(months=3),
        )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
