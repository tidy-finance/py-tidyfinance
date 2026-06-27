"""Tests for compute_breakpoints."""

import os
import sys
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import breakpoint_options, compute_breakpoints, data_options  # noqa: E402

# %% validation tests


def test_error_if_breakpoint_options_is_not_a_list():
    """Test error if breakpoint_options is not a dict."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError, match="dictionary"):
        compute_breakpoints(data, "value", "not_a_dict")


def test_error_if_both_n_portfolios_and_percentiles_are_provided():
    """Test error if both n_portfolios and percentiles are provided."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError, match="not both"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(n_portfolios=5, percentiles=[0.5]),
        )


def test_error_if_neither_n_portfolios_nor_percentiles_are_provided():
    """Test error if neither n_portfolios nor percentiles are provided."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError, match="must provide"):
        compute_breakpoints(data, "value", breakpoint_options())


def test_error_if_n_portfolios_is_1_or_less():
    """Test error if n_portfolios is 1 or less."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    # 1 and 0 are rejected by breakpoint_options' own validator
    with pytest.raises(ValueError):
        compute_breakpoints(data, "value", breakpoint_options(n_portfolios=1))
    with pytest.raises(ValueError):
        compute_breakpoints(data, "value", breakpoint_options(n_portfolios=0))


def test_error_if_breakpoints_exchanges_column_is_missing_from_data():
    """Test error if breakpoints_exchanges column is missing from data."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError, match="exchange"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE"]),
        )


# %% basic n_portfolios behavior


def test_returns_n_portfolios_plus_1_breakpoints():
    """Test returns n_portfolios + 1 breakpoints."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    for n in (2, 5, 10):
        bp = compute_breakpoints(
            data, "value", breakpoint_options(n_portfolios=n)
        )
        assert len(bp) == n + 1


def test_breakpoints_are_in_ascending_order():
    """Test breakpoints are in ascending order."""
    rng = np.random.default_rng()
    data = pd.DataFrame({"id": range(1000), "value": rng.standard_normal(1000)})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=5))
    assert (np.diff(bp) >= 0).all()


def test_breakpoints_are_numeric():
    """Test breakpoints are numeric."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=5))
    assert np.issubdtype(bp.dtype, np.floating)


def test_first_breakpoint_equals_min_last_approximately_equals_max():
    """Test first breakpoint equals min, last approximately equals max."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=5))
    assert bp[0] == data["value"].min()
    # Last breakpoint = max + tiny epsilon (1e-20)
    assert abs(bp[5] - (data["value"].max() + 1e-20)) < 1e-15


def test_n_portfolios_2_gives_3_breakpoints():
    """Test n_portfolios = 2 gives 3 breakpoints (min, median, max)."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=2))
    assert len(bp) == 3


# %% percentiles behavior


def test_percentiles_produce_correct_number_of_breakpoints():
    """Test percentiles produce correct number of breakpoints."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    pcts = [0.2, 0.4, 0.6, 0.8]
    bp = compute_breakpoints(
        data, "value", breakpoint_options(percentiles=pcts)
    )
    # breakpoints = [0, *percentiles, 1] -> length = len(percentiles) + 2
    assert len(bp) == len(pcts) + 2


def test_single_percentile_produces_3_breakpoints():
    """Test single percentile produces 3 breakpoints (median split)."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    bp = compute_breakpoints(
        data, "value", breakpoint_options(percentiles=[0.5])
    )
    assert len(bp) == 3


def test_percentile_breakpoints_are_ascending():
    """Test percentile breakpoints are ascending."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"id": range(500), "value": rng.standard_normal(500)})
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(percentiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
    )
    assert (np.diff(bp) >= 0).all()


def test_n_portfolios_5_and_percentiles_give_same_breakpoints():
    """Test n_portfolios = 5 and equivalent percentiles give same breakpoints."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    bp_n = compute_breakpoints(
        data, "value", breakpoint_options(n_portfolios=5)
    )
    bp_p = compute_breakpoints(
        data, "value", breakpoint_options(percentiles=[0.2, 0.4, 0.6, 0.8])
    )
    np.testing.assert_allclose(bp_n, bp_p)


def test_n_portfolios_10_and_decile_percentiles_give_same_breakpoints():
    """Test n_portfolios = 10 and decile percentiles give same breakpoints."""
    rng = np.random.default_rng(7)
    data = pd.DataFrame({"id": range(5000), "value": rng.standard_normal(5000)})
    bp_n = compute_breakpoints(
        data, "value", breakpoint_options(n_portfolios=10)
    )
    bp_p = compute_breakpoints(
        data,
        "value",
        breakpoint_options(percentiles=np.arange(0.1, 1.0, 0.1).tolist()),
    )
    np.testing.assert_allclose(bp_n, bp_p)


# %% exchange filtering


def test_breakpoints_exchanges_filters_data_before_computing_breakpoints():
    """Test breakpoints_exchanges filters data before computing breakpoints."""
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "id": range(200),
            "exchange": ["NYSE"] * 100 + ["NASDAQ"] * 100,
            "value": np.concatenate(
                [
                    rng.standard_normal(100) + 10,
                    rng.standard_normal(100) + 50,
                ]
            ),
        }
    )
    bp_all = compute_breakpoints(
        data, "value", breakpoint_options(n_portfolios=5)
    )
    bp_nyse = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE"]),
    )
    assert not np.allclose(bp_all, bp_nyse)
    assert (bp_nyse < 20).all()


def test_multiple_exchanges_can_be_specified():
    """Test multiple exchanges can be specified."""
    data = pd.DataFrame(
        {
            "id": range(300),
            "exchange": ["NYSE"] * 100 + ["NASDAQ"] * 100 + ["AMEX"] * 100,
            "value": np.concatenate(
                [np.arange(1, 101), np.arange(201, 301), np.arange(401, 501)]
            ),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(
            n_portfolios=5, breakpoints_exchanges=["NYSE", "AMEX"]
        ),
    )
    assert bp[0] == 1
    assert bp[-1] > 400


def test_custom_exchange_column_name_via_data_options():
    """Test custom exchange column name via data_options."""
    data = pd.DataFrame(
        {
            "id": range(100),
            "exch": ["NYSE"] * 50 + ["NASDAQ"] * 50,
            "value": np.arange(1, 101),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE"]),
        data_options=data_options(exchange="exch"),
    )
    assert bp[0] >= 1
    assert bp[-1] <= 50 + 1e-15


def test_error_when_custom_exchange_column_does_not_exist():
    """Test error when custom exchange column does not exist."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError, match="nonexistent"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE"]),
            data_options=data_options(exchange="nonexistent"),
        )


def test_null_data_options_uses_defaults_without_error():
    """Test None data_options uses defaults without error."""
    data = pd.DataFrame(
        {
            "id": range(100),
            "exchange": ["NYSE"] * 100,
            "value": np.arange(1, 101),
        }
    )
    compute_breakpoints(data, "value", breakpoint_options(n_portfolios=5))


# %% interior epsilon


def test_interior_breakpoints_are_slightly_larger_than_raw_quantiles():
    """Test interior breakpoints are slightly larger than raw quantiles."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=5))
    raw_q = np.quantile(data["value"].values, np.linspace(0, 1, 6))
    assert bp[0] == raw_q[0]
    for i in range(1, 6):
        assert abs(bp[i] - (raw_q[i] + 1e-20)) < 1e-25


# %% smooth_bunching


def test_smooth_bunching_handles_clustering_on_both_edges():
    """Test smooth_bunching handles clustering on both edges."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate(
                [
                    np.zeros(200),
                    np.linspace(1, 99, 100),
                    np.full(200, 100),
                ]
            ),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=True),
    )
    assert len(bp) == 6
    assert bp[0] == 0
    assert bp[5] >= 100
    assert (bp[1:5] > 0).all()
    assert (bp[1:5] <= 100 + 1e-10).all()


def test_smooth_bunching_with_both_edges_and_percentiles_emits_warning():
    """Test smooth_bunching with both edges and percentiles emits warning."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate(
                [
                    np.zeros(200),
                    np.linspace(1, 99, 100),
                    np.full(200, 100),
                ]
            ),
        }
    )
    with pytest.warns(UserWarning, match="smooth_bunching"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(
                percentiles=[0.2, 0.4, 0.6, 0.8], smooth_bunching=True
            ),
        )


def test_smooth_bunching_handles_clustering_only_on_lower_edge():
    """Test smooth_bunching handles clustering only on lower edge."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate([np.zeros(200), np.linspace(1, 100, 300)]),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=True),
    )
    assert len(bp) == 6
    assert bp[0] == 0
    assert (bp[1:6] > 0).all()


def test_lower_edge_bunching_with_percentiles_emits_warning():
    """Test lower edge bunching with percentiles emits warning."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate([np.zeros(200), np.linspace(1, 100, 300)]),
        }
    )
    with pytest.warns(UserWarning, match="smooth_bunching"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(
                percentiles=[0.2, 0.4, 0.6, 0.8], smooth_bunching=True
            ),
        )


def test_smooth_bunching_handles_clustering_only_on_upper_edge():
    """Test smooth_bunching handles clustering only on upper edge."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate(
                [np.linspace(0, 99, 300), np.full(200, 100)]
            ),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=True),
    )
    assert len(bp) == 6
    assert bp[5] >= 100
    assert (bp[0:5] < 100).all()


def test_upper_edge_bunching_with_percentiles_emits_warning():
    """Test upper edge bunching with percentiles emits warning."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate(
                [np.linspace(0, 99, 300), np.full(200, 100)]
            ),
        }
    )
    with pytest.warns(UserWarning, match="smooth_bunching"):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(
                percentiles=[0.2, 0.4, 0.6, 0.8], smooth_bunching=True
            ),
        )


def test_smooth_bunching_false_does_not_alter_clustered_breakpoints():
    """Test smooth_bunching = False does not alter clustered breakpoints."""
    data = pd.DataFrame(
        {
            "id": range(500),
            "value": np.concatenate(
                [
                    np.zeros(200),
                    np.linspace(1, 99, 100),
                    np.full(200, 100),
                ]
            ),
        }
    )
    bp = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=False),
    )
    raw_q = np.quantile(data["value"].values, np.linspace(0, 1, 6))
    assert bp[0] == raw_q[0]
    for i in range(1, 6):
        assert abs(bp[i] - (raw_q[i] + 1e-20)) < 1e-25


def test_smooth_bunching_none_triggers_error():
    """Test smooth_bunching = None triggers error."""
    data = pd.DataFrame({"id": range(100), "value": np.arange(1, 101)})
    with pytest.raises(ValueError):
        compute_breakpoints(
            data,
            "value",
            breakpoint_options(n_portfolios=5, smooth_bunching=None),
        )


def test_smooth_bunching_true_with_no_clustering_gives_same_as_false():
    """Test smooth_bunching = True with no clustering gives same as False."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    bp_smooth = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=True),
    )
    bp_plain = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=False),
    )
    np.testing.assert_allclose(bp_smooth, bp_plain)


# %% edge cases


def test_na_values_in_sorting_variable_are_ignored():
    """Test NaN values in sorting variable are ignored."""
    data_clean = pd.DataFrame(
        {"id": range(100), "value": np.arange(1, 101).astype(float)}
    )
    data_na = pd.DataFrame(
        {
            "id": range(120),
            "value": np.concatenate(
                [np.arange(1, 101).astype(float), np.full(20, np.nan)]
            ),
        }
    )
    bp_clean = compute_breakpoints(
        data_clean, "value", breakpoint_options(n_portfolios=5)
    )
    bp_na = compute_breakpoints(
        data_na, "value", breakpoint_options(n_portfolios=5)
    )
    np.testing.assert_allclose(bp_clean, bp_na)


def test_works_with_arbitrary_column_names():
    """Test works with arbitrary column names."""
    rng = np.random.default_rng()
    data = pd.DataFrame(
        {
            "company_id": range(200),
            "market_cap": rng.uniform(1e6, 1e9, 200),
        }
    )
    bp = compute_breakpoints(
        data, "market_cap", breakpoint_options(n_portfolios=5)
    )
    assert len(bp) == 6
    assert (np.diff(bp) >= 0).all()


def test_works_with_very_small_data():
    """Test works with very small data (n = 2)."""
    data = pd.DataFrame({"id": [1, 2], "value": [1.0, 10.0]})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=2))
    assert len(bp) == 3
    assert bp[0] == 1


def test_works_with_n_3_and_3_portfolios():
    """Test works with n = 3 and 3 portfolios (one obs per portfolio)."""
    data = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 5.0, 10.0]})
    bp = compute_breakpoints(data, "value", breakpoint_options(n_portfolios=3))
    assert len(bp) == 4


def test_non_uniform_percentiles_produce_correct_breakpoints():
    """Test non-uniform percentiles produce correct breakpoints."""
    data = pd.DataFrame({"id": range(1000), "value": np.arange(1, 1001)})
    bp = compute_breakpoints(
        data, "value", breakpoint_options(percentiles=[0.1, 0.5, 0.9])
    )
    assert len(bp) == 5
    assert abs(bp[2] - 500) < 5


def test_100_portfolios_work_with_sufficient_data():
    """Test 100 portfolios work with sufficient data."""
    rng = np.random.default_rng(99)
    data = pd.DataFrame(
        {"id": range(100_000), "value": rng.standard_normal(100_000)}
    )
    bp = compute_breakpoints(
        data, "value", breakpoint_options(n_portfolios=100)
    )
    assert len(bp) == 101
    assert (np.diff(bp) >= 0).all()


# %% integration with assign_portfolio (replaces R's findInterval tests)


def test_breakpoints_produce_valid_portfolio_assignments():
    """Test breakpoints produce valid portfolio assignments."""
    from tidyfinance.portfolios import assign_portfolio

    rng = np.random.default_rng(42)
    data = pd.DataFrame({"id": range(1000), "value": rng.standard_normal(1000)})
    portfolios = assign_portfolio(
        data, "value", breakpoint_options={"n_portfolios": 5}
    )
    valid = portfolios.dropna()
    assert valid.between(1, 5).all()
    assert sorted(valid.unique().tolist()) == [1, 2, 3, 4, 5]


def test_exchange_filtered_breakpoints_produce_valid_assignments():
    """Test exchange-filtered breakpoints produce valid assignments on full data."""
    from tidyfinance.portfolios import assign_portfolio

    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "id": range(500),
            "exchange": ["NYSE"] * 250 + ["NASDAQ"] * 250,
            "value": rng.standard_normal(500),
        }
    )
    portfolios = assign_portfolio(
        data,
        "value",
        breakpoint_options={
            "n_portfolios": 5,
            "breakpoints_exchanges": ["NYSE"],
        },
    )
    valid = portfolios.dropna()
    assert valid.between(1, 5).all()
    assert len(portfolios) == 500


def test_smooth_bunching_produces_more_distinct_interior_values():
    """Test smooth bunching produces more distinct interior values."""
    data = pd.DataFrame(
        {
            "id": range(1000),
            "value": np.concatenate(
                [
                    np.zeros(400),
                    np.linspace(1, 99, 200),
                    np.full(400, 100),
                ]
            ),
        }
    )
    bp_smooth = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=True),
    )
    bp_raw = compute_breakpoints(
        data,
        "value",
        breakpoint_options(n_portfolios=5, smooth_bunching=False),
    )
    n_distinct_smooth = len(np.unique(np.round(bp_smooth, 10)))
    n_distinct_raw = len(np.unique(np.round(bp_raw, 10)))
    assert n_distinct_smooth >= n_distinct_raw


def test_function_is_deterministic():
    """Test function is deterministic (same input = same output)."""
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"id": range(500), "value": rng.standard_normal(500)})
    opts = breakpoint_options(n_portfolios=5)
    bp1 = compute_breakpoints(data, "value", opts)
    bp2 = compute_breakpoints(data, "value", opts)
    np.testing.assert_array_equal(bp1, bp2)


# %% breakpoints_min_size_threshold


def test_min_size_threshold_with_exchanges_filters_small_stocks():
    """Test breakpoints_min_size_threshold with exchanges filters small stocks."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "id": range(200),
            "exchange": ["NYSE"] * 100 + ["NASDAQ"] * 100,
            "mktcap_lag": np.concatenate([np.arange(1, 101)] * 2),
            "sorting_var": rng.standard_normal(200),
        }
    )
    bp_no_filter = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE"]),
    )
    bp_with_filter = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(
            n_portfolios=5,
            breakpoints_exchanges=["NYSE"],
            breakpoints_min_size_threshold=0.2,
        ),
    )
    assert len(bp_with_filter) == 6
    assert not np.allclose(bp_no_filter, bp_with_filter)


def test_min_size_threshold_without_exchanges_uses_full_sample():
    """Test breakpoints_min_size_threshold without exchanges uses full sample."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "id": range(200),
            "exchange": ["NYSE"] * 100 + ["NASDAQ"] * 100,
            "mktcap_lag": np.arange(1, 201),
            "sorting_var": rng.standard_normal(200),
        }
    )
    bp_no_filter = compute_breakpoints(
        data, "sorting_var", breakpoint_options(n_portfolios=5)
    )
    bp_with_filter = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=5, breakpoints_min_size_threshold=0.2),
    )
    assert len(bp_with_filter) == 6
    assert not np.allclose(bp_no_filter, bp_with_filter)


def test_min_size_threshold_produces_correct_breakpoints():
    """Test breakpoints_min_size_threshold produces correct breakpoints."""
    rng = np.random.default_rng(7)
    mktcap = np.arange(1, 101)
    sorting_var = rng.standard_normal(100)
    data = pd.DataFrame({"mktcap_lag": mktcap, "sorting_var": sorting_var})
    size_cutoff = np.quantile(mktcap, 0.2)
    above = mktcap > size_cutoff
    expected = np.quantile(sorting_var[above], np.linspace(0, 1, 6))
    expected[1:] += 1e-20

    bp = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=5, breakpoints_min_size_threshold=0.2),
    )
    np.testing.assert_allclose(bp, expected)


def test_min_size_threshold_excludes_stocks_exactly_at_cutoff():
    """Test breakpoints_min_size_threshold excludes stocks exactly at the cutoff."""
    mktcap = np.array([10, 20, 30, 40, 50], dtype=float)
    sorting_var = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = pd.DataFrame({"mktcap_lag": mktcap, "sorting_var": sorting_var})
    size_cutoff = np.quantile(mktcap, 0.25)
    above = mktcap > size_cutoff
    # boundary stock at exactly the cutoff is excluded
    assert not above[mktcap == size_cutoff].any()

    expected = np.quantile(sorting_var[above], np.linspace(0, 1, 4))
    expected[1:] += 1e-20
    bp = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=3, breakpoints_min_size_threshold=0.25),
    )
    np.testing.assert_allclose(bp, expected)


def test_min_size_threshold_with_na_mktcap_excludes_na_rows():
    """Test breakpoints_min_size_threshold with NaN mktcap excludes NaN rows."""
    data = pd.DataFrame(
        {
            "mktcap_lag": [np.nan, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "sorting_var": [99, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    bp = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=3, breakpoints_min_size_threshold=0.2),
    )
    assert (~np.isnan(bp)).all()
    assert len(bp) == 4

    mktcap_clean = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    size_cutoff = np.quantile(mktcap_clean, 0.2)
    above = ~pd.isna(data["mktcap_lag"].values) & (
        data["mktcap_lag"].values > size_cutoff
    )
    expected = np.quantile(
        data["sorting_var"].values[above], np.linspace(0, 1, 4)
    )
    expected[1:] += 1e-20
    np.testing.assert_allclose(bp, expected)


def test_min_size_threshold_none_default_has_no_effect():
    """Test breakpoints_min_size_threshold = None (default) has no effect."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "id": range(100),
            "exchange": ["NYSE"] * 100,
            "mktcap_lag": np.arange(1, 101),
            "sorting_var": rng.standard_normal(100),
        }
    )
    bp_default = compute_breakpoints(
        data, "sorting_var", breakpoint_options(n_portfolios=5)
    )
    bp_explicit_none = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(n_portfolios=5, breakpoints_min_size_threshold=None),
    )
    np.testing.assert_array_equal(bp_default, bp_explicit_none)


def test_min_size_threshold_errors_when_mktcap_column_missing():
    """Test breakpoints_min_size_threshold errors when mktcap column is missing."""
    data = pd.DataFrame(
        {"id": range(100), "sorting_var": np.arange(1, 101).astype(float)}
    )
    with pytest.raises(ValueError, match="mktcap_lag"):
        compute_breakpoints(
            data,
            "sorting_var",
            breakpoint_options(
                n_portfolios=5, breakpoints_min_size_threshold=0.2
            ),
        )


def test_min_size_threshold_works_with_custom_data_options():
    """Test breakpoints_min_size_threshold works with custom data_options."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "id": range(100),
            "listing": ["NYSE"] * 100,
            "mcap": np.arange(1, 101),
            "sorting_var": rng.standard_normal(100),
        }
    )
    bp = compute_breakpoints(
        data,
        "sorting_var",
        breakpoint_options(
            n_portfolios=5,
            breakpoints_exchanges=["NYSE"],
            breakpoints_min_size_threshold=0.2,
        ),
        data_options=data_options(exchange="listing", mktcap_lag="mcap"),
    )
    assert len(bp) == 6


# %% small-block summary tests (mirrors R's "Standard data" section)


_df = pd.DataFrame(
    {
        "x": np.arange(1, 21),
        "exchange": ["NYSE", "NASDAQ"] * 10,
        "mktcap_lag": np.arange(1, 21),
    }
)

_df_both = pd.DataFrame(
    {"x": np.concatenate([np.zeros(10), np.arange(1, 6), np.full(10, 10)])}
)
_df_lower = pd.DataFrame(
    {"x": np.concatenate([np.zeros(15), np.arange(1, 11)])}
)
_df_upper = pd.DataFrame(
    {"x": np.concatenate([np.arange(1, 11), np.full(15, 20)])}
)


def test_non_list_breakpoint_options_raises_an_error():
    """Test non-dict breakpoint_options raises an error."""
    with pytest.raises(ValueError):
        compute_breakpoints(_df, "x", 5)


def test_both_n_portfolios_and_percentiles_raises_an_error():
    """Test providing both n_portfolios and percentiles raises an error."""
    with pytest.raises(ValueError):
        compute_breakpoints(
            _df,
            "x",
            {"n_portfolios": 4, "percentiles": [0.25, 0.75]},
        )


def test_neither_n_portfolios_nor_percentiles_raises_an_error():
    """Test providing neither n_portfolios nor percentiles raises an error."""
    with pytest.raises(ValueError):
        compute_breakpoints(_df, "x", {})


def test_exchanges_set_but_column_absent_raises_error():
    """Test breakpoints_exchanges set but column absent raises an error."""
    with pytest.raises(ValueError):
        compute_breakpoints(
            pd.DataFrame({"x": np.arange(1, 11)}),
            "x",
            {"n_portfolios": 3, "breakpoints_exchanges": "NYSE"},
        )


def test_min_size_threshold_set_but_mktcap_lag_absent_raises_error():
    """Test min_size_threshold set but mktcap_lag column absent raises error."""
    with pytest.raises(ValueError):
        compute_breakpoints(
            pd.DataFrame({"x": np.arange(1, 11)}),
            "x",
            {"n_portfolios": 3, "breakpoints_min_size_threshold": 0.2},
        )


def test_n_portfolios_le_1_raises_an_error():
    """Test n_portfolios <= 1 raises an error."""
    with pytest.raises(ValueError):
        compute_breakpoints(_df, "x", {"n_portfolios": 1})


def test_exchange_filter_leaving_no_rows_warns_and_returns_nan():
    """Test exchange filter leaving no rows warns and returns NaN."""
    with pytest.warns(UserWarning):
        result = compute_breakpoints(
            _df,
            "x",
            {"n_portfolios": 3, "breakpoints_exchanges": "AMEX"},
        )
    assert len(result) == 1
    assert np.isnan(result[0])


def test_n_portfolios_with_no_filters_returns_n_plus_1_breakpoints():
    """Test n_portfolios with no filters returns n + 1 breakpoints."""
    result = compute_breakpoints(_df, "x", {"n_portfolios": 4})
    assert len(result) == 5


def test_percentiles_with_exchange_filter_computes_breakpoints_on_filtered_data():
    """Test percentiles with exchange filter computes breakpoints on filtered data."""
    result = compute_breakpoints(
        _df,
        "x",
        {
            "percentiles": [0.25, 0.5, 0.75],
            "breakpoints_exchanges": "NYSE",
        },
    )
    assert len(result) == 5


def test_min_size_with_exchange_filter_uses_exchange_filtered_mktcap_reference():
    """Test min_size_threshold with exchange filter uses filtered mktcap."""
    result = compute_breakpoints(
        _df,
        "x",
        {
            "n_portfolios": 3,
            "breakpoints_exchanges": "NYSE",
            "breakpoints_min_size_threshold": 0.2,
        },
    )
    assert len(result) == 4


def test_min_size_without_exchange_filter_uses_full_dataset_as_reference():
    """Test min_size_threshold without exchange filter uses full dataset."""
    result = compute_breakpoints(
        _df,
        "x",
        {"n_portfolios": 3, "breakpoints_min_size_threshold": 0.2},
    )
    assert len(result) == 4


def test_smooth_bunching_with_both_edge_bunching_warns_and_recomputes():
    """Test smooth_bunching with both-edge bunching warns and recomputes."""
    with pytest.warns(UserWarning):
        result = compute_breakpoints(
            _df_both,
            "x",
            {
                "percentiles": [1 / 3, 2 / 3],
                "smooth_bunching": True,
            },
        )
    assert len(result) == 4


def test_smooth_bunching_with_lower_edge_bunching_warns_and_recomputes():
    """Test smooth_bunching with lower-edge bunching warns and recomputes."""
    with pytest.warns(UserWarning):
        result = compute_breakpoints(
            _df_lower,
            "x",
            {
                "percentiles": [1 / 3, 2 / 3],
                "smooth_bunching": True,
            },
        )
    assert len(result) == 4


def test_smooth_bunching_with_upper_edge_bunching_warns_and_recomputes():
    """Test smooth_bunching with upper-edge bunching warns and recomputes."""
    with pytest.warns(UserWarning):
        result = compute_breakpoints(
            _df_upper,
            "x",
            {
                "percentiles": [1 / 3, 2 / 3],
                "smooth_bunching": True,
            },
        )
    assert len(result) == 4


def test_breakpoints_exchanges_accepts_list_of_strings():
    """Test breakpoints_exchanges accepts a list of strings."""
    result = breakpoint_options(
        n_portfolios=5, breakpoints_exchanges=["NYSE", "AMEX"]
    )
    assert result["breakpoints_exchanges"] == ["NYSE", "AMEX"]


def test_breakpoints_exchanges_rejects_empty_list():
    """Test breakpoints_exchanges rejects an empty list."""
    with pytest.raises(ValueError, match="breakpoints_exchanges"):
        breakpoint_options(n_portfolios=5, breakpoints_exchanges=[])


def test_breakpoints_exchanges_rejects_list_with_non_strings():
    """Test breakpoints_exchanges rejects a list containing non-strings."""
    with pytest.raises(ValueError, match="breakpoints_exchanges"):
        breakpoint_options(n_portfolios=5, breakpoints_exchanges=["NYSE", 1])


if __name__ == "__main__":
    pytest.main([__file__])
