"""Tests for the global data frame backend (pandas/polars)."""

import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tidyfinance as tf  # noqa: E402
from tidyfinance.backend import (  # noqa: E402
    _convert_output,
    _to_pandas_input,
    get_backend,
)


@pytest.fixture(autouse=True)
def _restore_backend():
    """Ensure the global backend never leaks between tests."""
    tf.set_backend("pandas")
    yield
    tf.set_backend("pandas")


pythonpytestmark = pytest.mark.filterwarnings(
    "ignore:Returning pseudo data:UserWarning",
)


# %% set_backend / get_backend


def test_default_backend_is_pandas():
    assert get_backend() == "pandas"


def test_set_and_get_backend():
    tf.set_backend("polars")
    assert tf.get_backend() == "polars"


def test_set_backend_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid backend"):
        tf.set_backend("spark")


# %% _convert_output


def test_convert_output_passthrough_for_pandas_backend():
    df = pd.DataFrame({"a": [1, 2]})
    assert _convert_output(df) is df


def test_convert_output_returns_polars():
    tf.set_backend("polars")
    out = _convert_output(pd.DataFrame({"a": [1, 2]}))
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["a"]


def test_convert_output_preserves_named_index_as_column():
    tf.set_backend("polars")
    df = pd.DataFrame({"v": [1.0, 2.0]})
    df.index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    df.index.name = "date"
    out = _convert_output(df)
    assert "date" in out.columns


def test_convert_output_drops_default_rangeindex():
    tf.set_backend("polars")
    out = _convert_output(pd.DataFrame({"a": [1, 2]}))
    assert "index" not in out.columns


def test_convert_output_leaves_series_alone():
    tf.set_backend("polars")
    s = pd.Series([1, 2, 3], name="x")
    assert _convert_output(s) is s


def test_convert_output_leaves_dict_alone():
    tf.set_backend("polars")
    d = {"a": 1}
    assert _convert_output(d) is d


# %% _to_pandas_input


def test_to_pandas_input_converts_polars_frame():
    out = _to_pandas_input(pl.DataFrame({"a": [1, 2]}))
    assert isinstance(out, pd.DataFrame)


def test_to_pandas_input_collects_lazyframe():
    out = _to_pandas_input(pl.DataFrame({"a": [1, 2]}).lazy())
    assert isinstance(out, pd.DataFrame)


def test_to_pandas_input_passes_through_pandas():
    df = pd.DataFrame({"a": [1, 2]})
    assert _to_pandas_input(df) is df


# %% download_data integration (network-free pseudo domain)


def test_download_data_returns_pandas_by_default():
    with pytest.warns(UserWarning, match="pseudo data"):
        out = tf.download_data("Pseudo Data", "crsp_monthly")
    assert isinstance(out, pd.DataFrame)


def test_download_data_returns_polars_when_configured():
    tf.set_backend("polars")
    with pytest.warns(UserWarning, match="pseudo data"):
        out = tf.download_data("Pseudo Data", "crsp_monthly")
    assert isinstance(out, pl.DataFrame)


# %% core function honors the backend (input + output round-trip)


def _lag_input():
    return pd.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": list(pd.date_range("2023-01-01", periods=4, freq="MS")) * 2,
            "size": [float(i) for i in range(1, 9)],
        }
    )


def test_core_function_returns_polars_for_polars_input():
    tf.set_backend("polars")
    data = pl.from_pandas(_lag_input())
    out = tf.add_lagged_columns(
        data, cols="size", lag=pd.DateOffset(months=1), by="permno"
    )
    assert isinstance(out, pl.DataFrame)
    assert "size_lag" in out.columns


def test_core_function_returns_pandas_under_pandas_backend():
    out = tf.add_lagged_columns(
        _lag_input(), cols="size", lag=pd.DateOffset(months=1), by="permno"
    )
    assert isinstance(out, pd.DataFrame)


def test_core_accepts_polars_input_even_on_pandas_backend():
    data = pl.from_pandas(_lag_input())
    out = tf.add_lagged_columns(
        data, cols="size", lag=pd.DateOffset(months=1), by="permno"
    )
    assert isinstance(out, pd.DataFrame)


def test_series_returning_function_stays_pandas_under_polars():
    tf.set_backend("polars")
    rng = np.random.default_rng(42)
    data = pl.from_pandas(
        pd.DataFrame({"id": range(100), "value": rng.random(100)})
    )
    result = tf.assign_portfolio(
        data, "value", breakpoint_options={"n_portfolios": 5}
    )
    assert isinstance(result, pd.Series)


# %% boundary-only wrapping protects internal cross-calls


def test_in_module_implementations_remain_unwrapped():
    """The wrapping is applied only at the public package boundary. The
    in-module implementations (which core functions call internally,
    e.g. implement_portfolio_sort -> filter_sorting_data ->
    compute_portfolio_returns -> assign_portfolio) must stay pandas-only
    so internal callers never receive polars under the polars backend."""
    import tidyfinance.core as core

    tf.set_backend("polars")
    out = core.add_lagged_columns(
        _lag_input(), cols="size", lag=pd.DateOffset(months=1), by="permno"
    )
    assert isinstance(out, pd.DataFrame)


# %% Smoke tests: every wrapped function should round-trip its data
# argument through the polars backend without raising and produce the
# expected output type.


def _panel_with_returns():
    """Five-asset, two-year panel — enough per cross-section to fit
    three breakpoints without ties."""
    rng = np.random.default_rng(7)
    n_permnos = 5
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    n = n_permnos * len(dates)
    return pd.DataFrame(
        {
            "permno": np.repeat(np.arange(1, n_permnos + 1), len(dates)),
            "date": list(dates) * n_permnos,
            "ret_excess": rng.standard_normal(n) * 0.02,
            "mkt_excess": rng.standard_normal(n) * 0.02,
            "size": rng.uniform(1.0, 100.0, n),
            "mktcap_lag": rng.uniform(1.0, 100.0, n),
            "exchange": ["NYSE"] * n,
        }
    )


def test_compute_breakpoints_polars_input_returns_ndarray():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.compute_breakpoints(
        data, "size", breakpoint_options={"n_portfolios": 3}
    )
    # compute_breakpoints returns np.ndarray; backend leaves arrays
    # alone.
    assert isinstance(out, np.ndarray)


def test_compute_portfolio_returns_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.compute_portfolio_returns(
        sorting_data=data,
        sorting_variables="size",
        breakpoint_options_main={"n_portfolios": 3},
    )
    assert isinstance(out, pl.DataFrame)


def test_compute_long_short_returns_round_trips_polars():
    tf.set_backend("polars")
    portfolio_returns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "portfolio": [1, 1, 1, 3, 3, 3],
            "ret_excess_vw": [0.01, -0.02, 0.005, 0.03, 0.01, 0.04],
        }
    )
    out = tf.compute_long_short_returns(pl.from_pandas(portfolio_returns))
    assert isinstance(out, pl.DataFrame)


def test_compute_rolling_value_returns_pandas_under_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.compute_rolling_value(
        data,
        f=lambda d: d["ret_excess"].mean(),
        period="month",
        periods=6,
    )
    # compute_rolling_value returns np.ndarray; backend leaves arrays
    # alone.
    assert isinstance(out, np.ndarray)


def test_create_summary_statistics_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.create_summary_statistics(data, ["ret_excess", "size"])
    assert isinstance(out, pl.DataFrame)


def test_estimate_betas_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.estimate_betas(
        data,
        model="ret_excess ~ mkt_excess",
        lookback=12,
        min_obs=8,
    )
    assert isinstance(out, pl.DataFrame)


def test_estimate_fama_macbeth_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.estimate_fama_macbeth(
        data,
        model="ret_excess ~ mkt_excess",
    )
    assert isinstance(out, pl.DataFrame)


def test_estimate_model_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.estimate_model(data, "ret_excess ~ mkt_excess")
    assert isinstance(out, pl.DataFrame)


def test_compute_portfolio_returns_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    out = tf.compute_portfolio_returns(
        data=data,
        sorting_variables="size",
        sorting_method="univariate",
        breakpoint_options_main={"n_portfolios": 3},
    )
    assert isinstance(out, pl.DataFrame)


def test_implement_portfolio_sort_round_trips_polars():
    tf.set_backend("polars")
    data = pl.from_pandas(_panel_with_returns())
    pso = tf.portfolio_sort_options(
        breakpoint_options_main=tf.breakpoint_options(n_portfolios=3)
    )
    out = tf.implement_portfolio_sort(
        data=data,
        sorting_variables="size",
        sorting_method="univariate",
        portfolio_sort_options=pso,
    )
    assert isinstance(out, pl.DataFrame)


def test_list_supported_datasets_returns_polars():
    tf.set_backend("polars")
    out = tf.list_supported_datasets()
    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) >= {"type", "dataset_name", "domain"}


def test_list_supported_datasets_as_vector_unchanged():
    tf.set_backend("polars")
    out = tf.list_supported_datasets(as_vector=True)
    # Returns a list of strings; backend leaves non-frame outputs
    # alone.
    assert isinstance(out, list)


def test_list_supported_indexes_returns_polars():
    tf.set_backend("polars")
    out = tf.list_supported_indexes()
    assert isinstance(out, pl.DataFrame)


def test_process_trace_data_round_trips_polars():
    tf.set_backend("polars")
    trace = pd.DataFrame(
        {
            "cusip_id": ["00077D1AA"] * 2,
            "msg_seq_nb": [1, 2],
            "orig_msg_seq_nb": [1, 2],
            "trd_rpt_dt": pd.to_datetime(["2015-01-05", "2015-01-06"]),
            "trd_rpt_tm": ["09:31:00", "09:36:00"],
            "trd_exctn_dt": pd.to_datetime(["2015-01-04", "2015-01-05"]),
            "trd_exctn_tm": ["09:30:00", "09:35:00"],
            "rptd_pr": [100.0, 100.5],
            "entrd_vol_qt": [1000, 1500],
            "yld_pt": [5.0, 5.0],
            "rpt_side_cd": ["B", "S"],
            "cntra_mp_id": ["D", "D"],
            "trc_st": ["T", "T"],
            "asof_cd": [None, None],
            "wis_fl": ["N", "N"],
            "days_to_sttl_ct": [2, 2],
            "stlmnt_dt": pd.to_datetime(["2015-01-06", "2015-01-07"]),
            "spcl_trd_fl": [None, None],
        }
    )
    out = tf.process_trace_data(pl.from_pandas(trace))
    assert isinstance(out, pl.DataFrame)


# %% run all tests
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
