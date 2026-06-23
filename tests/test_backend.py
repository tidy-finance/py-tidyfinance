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
    out = tf.download_data("Pseudo Data", "crsp_monthly")
    assert isinstance(out, pd.DataFrame)


def test_download_data_returns_polars_when_configured():
    tf.set_backend("polars")
    out = tf.download_data("Pseudo Data", "crsp_monthly")
    assert isinstance(out, pl.DataFrame)


# %% core function honors the backend (input + output round-trip)

def _lag_input():
    return pd.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": list(pd.date_range("2023-01-01", periods=4, freq="MS"))
            * 2,
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
