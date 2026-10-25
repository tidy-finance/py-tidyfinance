"""Tests for assign_portfolio."""

import datetime as dt
import os
import sys
import warnings as _warnings

import numpy as np
import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import assign_portfolio  # noqa: E402

# %% mock breakpoint functions


def mock_breakpoint_function(
    data, sorting_variable, bp_options=None, data_options=None
):
    """Quantile-based mock breakpoint function."""
    n = (bp_options or {}).get("n_portfolios", 5)
    probs = np.linspace(0, 1, n + 1)
    return np.quantile(
        data.get_column(sorting_variable).drop_nulls().to_numpy(), probs
    )


def mock_breakpoint_percentiles(
    data, sorting_variable, bp_options=None, data_options=None
):
    """Percentile-based mock breakpoint function."""
    percentiles = (bp_options or {}).get("percentiles", [0.2, 0.4, 0.6, 0.8])
    probs = [0] + list(percentiles) + [1]
    return np.quantile(
        data.get_column(sorting_variable).drop_nulls().to_numpy(), probs
    )


# %% size tests


def test_assign_portfolio_returns_vector_same_length_as_data():
    """Test assign_portfolio returns a vector the same length as data."""
    rng = np.random.default_rng()
    data = pl.DataFrame({"id": range(100), "value": rng.random(100)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert len(result) == len(data)


def test_assign_portfolio_returns_integer_portfolio_indices():
    """Test assign_portfolio returns integer portfolio indices."""
    rng = np.random.default_rng()
    data = pl.DataFrame({"id": range(100), "value": rng.random(100)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result.dtype == pl.Int64


def test_missing_values_keep_the_integer_type():
    """Test missing sorting values give null in an integer series."""
    data = pl.DataFrame({"value": [1.0, None, 3.0, float("nan"), 5.0]})
    result = assign_portfolio(
        data, "value", breakpoint_options={"n_portfolios": 2}
    )
    assert result.dtype == pl.Int64
    assert result.to_list() == [1, None, 2, None, 2]


# %% portfolio range tests


def test_portfolio_indices_within_expected_range_for_5_portfolios():
    """Test portfolio indices are within expected range for 5 portfolios."""
    rng = np.random.default_rng(42)
    data = pl.DataFrame({"id": range(200), "value": rng.random(200)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result.drop_nulls().is_between(1, 5).all()


def test_portfolio_indices_within_expected_range_for_10_portfolios():
    """Test portfolio indices are within expected range for 10 portfolios."""
    rng = np.random.default_rng(42)
    data = pl.DataFrame({"id": range(1000), "value": rng.random(1000)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 10},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result.drop_nulls().is_between(1, 10).all()


def test_all_portfolio_buckets_are_populated_with_sufficient_distinct_data():
    """Test all portfolio buckets are populated with sufficient distinct data."""
    data = pl.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert sorted(result.drop_nulls().unique().to_list()) == [1, 2, 3, 4, 5]


# %% monotonicity / tie tests


def test_higher_sorting_variable_values_get_higher_or_equal_portfolios():
    """Test higher sorting variable values get higher or equal portfolios."""
    data = pl.DataFrame({"id": range(500), "value": np.arange(1, 501)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert (result.diff().drop_nulls() >= 0).all()


def test_rows_with_same_sorting_variable_value_get_same_portfolio():
    """Test rows with the same sorting variable value get the same portfolio."""
    data = pl.DataFrame(
        {"id": range(10), "value": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]}
    )
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 3},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result[0] == result[1]
    assert result[2] == result[3]
    assert result[4] == result[5]
    assert result[6] == result[7]
    assert result[8] == result[9]


# %% constant variable tests


def test_constant_sorting_variable_returns_all_1s_with_a_warning():
    """Test constant sorting variable returns all 1s with a warning."""
    data = pl.DataFrame({"id": range(50), "value": [42] * 50})
    with pytest.warns(UserWarning, match="sorting variable is constant"):
        result = assign_portfolio(
            data,
            "value",
            breakpoint_options={"n_portfolios": 5},
            breakpoint_function=mock_breakpoint_function,
        )
    assert result.to_list() == [1.0] * 50


def test_constant_sorting_variable_returns_vector_of_correct_length():
    """Test constant sorting variable returns vector of correct length."""
    data = pl.DataFrame({"id": range(10), "value": [0] * 10})
    with pytest.warns(UserWarning, match="constant"):
        result = assign_portfolio(
            data, "value", breakpoint_function=mock_breakpoint_function
        )
    assert len(result) == 10


def test_two_distinct_values_with_2_portfolios_produces_two_groups():
    """Test two distinct values with 2 portfolios produces two groups."""
    data = pl.DataFrame({"id": range(100), "value": [1] * 50 + [2] * 50})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 2},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result.drop_nulls().is_in([1, 2]).all()
    lows = result.filter(data.get_column("value") == 1)
    highs = result.filter(data.get_column("value") == 2)
    assert (lows.to_numpy() <= highs.to_numpy()).all()


def test_single_row_data_frame_with_constant_variable_triggers_warning():
    """Test single-row data frame with constant variable triggers warning."""
    data = pl.DataFrame({"id": [1], "value": [5]})
    with pytest.warns(UserWarning, match="constant"):
        result = assign_portfolio(
            data,
            "value",
            breakpoint_options={"n_portfolios": 3},
            breakpoint_function=mock_breakpoint_function,
        )
    assert result.to_list() == [1.0]


# %% cluster warning tests


def test_warning_when_clusters_reduce_number_of_portfolios():
    """Test warning is issued when clusters reduce the number of portfolios."""
    data = pl.DataFrame({"id": range(100), "value": [1] * 50 + [100] * 50})

    def mock_bp_5(data, sv, bp_options=None, data_options=None):
        return np.array([1, 20, 40, 60, 80, 100])

    with pytest.warns(UserWarning, match="number of portfolios differs"):
        assign_portfolio(data, "value", breakpoint_function=mock_bp_5)


def test_no_cluster_warning_when_portfolios_match_expected_count():
    """Test no cluster warning when portfolios match expected count."""
    data = pl.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        assign_portfolio(
            data,
            "value",
            breakpoint_options={"n_portfolios": 5},
            breakpoint_function=mock_breakpoint_function,
        )


# %% custom breakpoint function tests


def test_custom_breakpoint_function_is_used_correctly():
    """Test custom breakpoint function is used correctly."""

    def median_bp(data, sv, bp_options=None, data_options=None):
        vals = data.get_column(sv).to_numpy()
        return np.array([vals.min(), np.median(vals), vals.max()])

    data = pl.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    result = assign_portfolio(data, "value", breakpoint_function=median_bp)
    assert result.drop_nulls().is_in([1, 2]).all()
    below_median = result.filter(data.get_column("value") <= 50)
    assert (below_median == 1).all()


def test_breakpoint_options_are_forwarded_to_the_breakpoint_function():
    """Test breakpoint_options are forwarded to the breakpoint function."""

    def custom_bp(data, sv, bp_options=None, data_options=None):
        n = bp_options["custom_n"]
        probs = np.linspace(0, 1, n + 1)
        return np.quantile(
            data.get_column(sv).drop_nulls().to_numpy(), probs
        )

    data = pl.DataFrame({"id": range(200), "value": np.arange(1, 201)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"custom_n": 4},
        breakpoint_function=custom_bp,
    )
    assert result.drop_nulls().is_between(1, 4).all()


def test_data_options_are_forwarded_to_the_breakpoint_function():
    """Test data_options are forwarded to the breakpoint function."""

    def bp_with_opts(data, sv, bp_options=None, data_options=None):
        assert data_options is not None
        assert data_options["date"] == "my_date"
        probs = np.linspace(0, 1, 4)
        return np.quantile(
            data.get_column(sv).drop_nulls().to_numpy(), probs
        )

    data = pl.DataFrame(
        {
            "id": range(100),
            "my_date": [dt.date(2024, 1, 1)] * 100,
            "value": np.arange(1, 101),
        }
    )
    assign_portfolio(
        data,
        "value",
        breakpoint_function=bp_with_opts,
        data_options={"date": "my_date"},
    )


def test_null_breakpoint_options_works_without_error():
    """Test None breakpoint_options works without error."""

    def bp_fn(data, sv, bp_options=None, data_options=None):
        probs = np.linspace(0, 1, 4)
        return np.quantile(
            data.get_column(sv).drop_nulls().to_numpy(), probs
        )

    data = pl.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    assign_portfolio(
        data,
        "value",
        breakpoint_options=None,
        breakpoint_function=bp_fn,
    )


def test_null_data_options_works_without_error():
    """Test None data_options works without error."""
    data = pl.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 3},
        breakpoint_function=mock_breakpoint_function,
        data_options=None,
    )


# %% input data tests


def test_negative_values_are_handled_correctly():
    """Test negative values are handled correctly."""
    data = pl.DataFrame({"id": range(100), "value": np.arange(-50, 50)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert len(result) == 100
    assert result.drop_nulls().is_between(1, 5).all()
    assert (result.diff().drop_nulls() >= 0).all()


def test_decimal_floating_point_values_are_handled_correctly():
    """Test decimal/floating-point values are handled correctly."""
    rng = np.random.default_rng(7)
    data = pl.DataFrame({"id": range(200), "value": rng.standard_normal(200)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert len(result) == 200
    assert result.drop_nulls().is_between(1, 5).all()


def test_function_works_with_large_datasets():
    """Test function works with large datasets."""
    rng = np.random.default_rng(1)
    n = 100_000
    data = pl.DataFrame({"id": np.arange(n), "value": rng.standard_normal(n)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 10},
        breakpoint_function=mock_breakpoint_function,
    )
    assert len(result) == n
    assert result.drop_nulls().is_between(1, 10).all()
    assert sorted(result.drop_nulls().unique().to_list()) == list(
        range(1, 11)
    )


def test_extreme_values_are_assigned_to_boundary_portfolios():
    """Test extreme values (min and max) are assigned to boundary portfolios."""
    data = pl.DataFrame({"id": range(5), "value": [0, 25, 50, 75, 100]})

    def bp_fn(data, sv, bp_options=None, data_options=None):
        return np.array([0, 25, 50, 75, 100])

    result = assign_portfolio(data, "value", breakpoint_function=bp_fn)
    assert result[0] == 1
    assert result[4] == 4


def test_na_values_in_sorting_variable_produce_na_portfolio_assignments():
    """Test missing sorting variable values produce null assignments."""
    data = pl.DataFrame(
        {"id": range(10), "value": [1, 2, 3, 4, 5, 6, 7, 8, None, None]}
    )

    def bp_fn(data, sv, bp_options=None, data_options=None):
        vals = data.get_column(sv).drop_nulls().to_numpy()
        probs = np.linspace(0, 1, 4)
        return np.quantile(vals, probs)

    result = assign_portfolio(data, "value", breakpoint_function=bp_fn)
    assert result[8] is None
    assert result[9] is None
    assert result.slice(0, 8).is_not_null().all()


def test_function_works_with_different_column_names_for_sorting_variable():
    """Test function works with different column names for sorting variable."""
    rng = np.random.default_rng()
    data = pl.DataFrame(
        {
            "company_id": range(100),
            "market_cap": rng.uniform(1e6, 1e9, 100),
        }
    )
    result = assign_portfolio(
        data,
        "market_cap",
        breakpoint_options={"n_portfolios": 5},
        breakpoint_function=mock_breakpoint_function,
    )
    assert len(result) == 100
    assert result.drop_nulls().is_between(1, 5).all()


# %% percentile / spread tests


def test_percentile_based_breakpoints_produce_correct_number_of_groups():
    """Test percentile-based breakpoints produce correct number of groups."""
    data = pl.DataFrame({"id": range(500), "value": np.arange(1, 501)})
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"percentiles": [0.2, 0.4, 0.6, 0.8]},
        breakpoint_function=mock_breakpoint_percentiles,
    )
    assert result.drop_nulls().is_between(1, 5).all()
    assert sorted(result.drop_nulls().unique().to_list()) == [1, 2, 3, 4, 5]


def test_two_portfolios_split_data_roughly_in_half():
    """Test two portfolios split data roughly in half."""
    rng = np.random.default_rng(10)
    data = pl.DataFrame(
        {"id": range(1000), "value": rng.standard_normal(1000)}
    )
    result = assign_portfolio(
        data,
        "value",
        breakpoint_options={"n_portfolios": 2},
        breakpoint_function=mock_breakpoint_function,
    )
    assert result.drop_nulls().is_in([1, 2]).all()
    counts = result.drop_nulls().value_counts()
    assert (counts.get_column("count") > 400).all()


# %% spy tests


def test_breakpoint_function_receives_all_four_arguments():
    """Test breakpoint function receives all four arguments."""
    received = {}

    def spy_bp(data, sorting_variable, bp_options=None, data_options=None):
        received["data"] = data
        received["sorting_variable"] = sorting_variable
        received["bp_options"] = bp_options
        received["data_options"] = data_options
        return np.array([0.0, 0.5, 1.0])

    data = pl.DataFrame({"id": range(10), "value": np.linspace(0, 1, 10)})
    my_bp_opts = {"n_portfolios": 2}
    my_data_opts = {"date": "date_col"}
    assign_portfolio(
        data,
        "value",
        breakpoint_options=my_bp_opts,
        breakpoint_function=spy_bp,
        data_options=my_data_opts,
    )
    assert received["data"] is data
    assert received["sorting_variable"] == "value"
    assert received["bp_options"] == my_bp_opts
    assert received["data_options"] == my_data_opts


# %% direct value-correctness tests


def test_normal_assignment_returns_correct_portfolio_indices():
    """Test normal assignment returns correct portfolio indices."""
    data = pl.DataFrame({"x": [10, 60, 100]})

    def bp_fn(data, sv, bo=None, do=None):
        return np.array([0, 50, 100])

    result = assign_portfolio(data, "x", breakpoint_function=bp_fn)
    assert result.to_list() == [1, 2, 2]


def test_constant_sorting_variable_warns_and_returns_all_1():
    """Test constant sorting variable warns and returns all 1."""
    data = pl.DataFrame({"x": [5] * 6})
    with pytest.warns(UserWarning, match="constant"):
        result = assign_portfolio(data, "x")
    assert result.to_list() == [1] * 6
    assert result.dtype == pl.Int64


def test_na_breakpoints_warns_and_returns_na_series():
    """Test NaN breakpoints warns and returns an all-null series."""
    data = pl.DataFrame({"x": [1, 2, 3, 4, 5]})

    def bp_na(data, sv, bo=None, do=None):
        return np.array([np.nan, np.nan])

    with pytest.warns(UserWarning, match="missing breakpoints"):
        result = assign_portfolio(data, "x", breakpoint_function=bp_na)
    assert result.is_null().all()
    assert len(result) == 5
    assert result.dtype == pl.Int64


def test_cluster_warning_when_portfolios_collapse_due_to_ties():
    """Test cluster warning when portfolios collapse due to ties."""
    data = pl.DataFrame({"x": [1, 2, 3]})

    def bp_fn(data, sv, bo=None, do=None):
        return np.array([0, 25, 50, 75, 100])

    with pytest.warns(UserWarning, match="clusters"):
        result = assign_portfolio(data, "x", breakpoint_function=bp_fn)
    assert (result == 1).all()


if __name__ == "__main__":
    pytest.main([__file__])
