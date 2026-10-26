"""Tests for the global data frame backend (pandas/polars).

The package computes in polars internally. The backend wrapper converts
pandas inputs to polars on entry and converts polars outputs back to
pandas when the active backend is 'pandas' (the default).
"""

import datetime
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
    _to_polars_input,
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


def test_default_backend_is_polars():
    import importlib

    from tidyfinance import backend

    # The autouse fixture sets the backend, so reload the module to read
    # the value it starts with.
    assert importlib.reload(backend).get_backend() == "polars"


def test_set_and_get_backend():
    tf.set_backend("polars")
    assert tf.get_backend() == "polars"


def test_set_backend_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid backend"):
        tf.set_backend("spark")


# %% _to_polars_input


def test_to_polars_input_converts_pandas_frame():
    out = _to_polars_input(pd.DataFrame({"a": [1, 2]}))
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["a"]


def test_to_polars_input_collects_lazyframe():
    out = _to_polars_input(pl.DataFrame({"a": [1, 2]}).lazy())
    assert isinstance(out, pl.DataFrame)


def test_to_polars_input_passes_through_polars():
    df = pl.DataFrame({"a": [1, 2]})
    assert _to_polars_input(df) is df


def test_to_polars_input_preserves_named_index_as_column():
    df = pd.DataFrame({"v": [1.0, 2.0]})
    df.index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    df.index.name = "date"
    out = _to_polars_input(df)
    assert "date" in out.columns
    assert out.schema["date"] == pl.Date


def test_to_polars_input_casts_date_column_to_polars_date():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "v": [1.0, 2.0]}
    )
    out = _to_polars_input(df)
    assert out.schema["date"] == pl.Date
    assert out["date"].dt.strftime("%Y-%m-%d").to_list() == [
        "2020-01-31",
        "2020-02-29",
    ]


def test_to_polars_input_casts_wrds_date_columns_to_polars_date():
    """Date-typed WRDS columns (TRACE, Compustat, CCM links, FISD) must
    come in as polars Date, not Datetime (issue #66)."""
    days = pd.to_datetime(["2019-01-02", "2019-01-03"])
    df = pd.DataFrame(
        {
            "trd_exctn_dt": days,
            "trd_rpt_dt": days,
            "stlmnt_dt": days,
            "datadate": days,
            "rdq": days,
            "linkdt": days,
            "linkenddt": days,
            "maturity": days,
            "offering_date": days,
            "dated_date": days,
            "last_interest_date": days,
            "calculation_date": days,
        }
    )
    out = _to_polars_input(df)
    for column in df.columns:
        assert out.schema[column] == pl.Date, column


def test_to_polars_input_keeps_unknown_datetime_columns():
    """Only known calendar-date columns are cast; other datetime
    columns (e.g. user-supplied timestamps) keep their type."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-31 09:30:00"]),
            "trd_exctn_dt": pd.to_datetime(["2020-01-31"]),
        }
    )
    out = _to_polars_input(df)
    assert isinstance(out.schema["timestamp"], pl.Datetime)
    assert out.schema["trd_exctn_dt"] == pl.Date


def test_to_polars_input_keeps_timezone_aware_datetime_columns():
    """Timezone-aware datetimes are never cast: taking the UTC calendar
    date could differ from the wall-clock date."""
    # fixed-offset timezone so the test needs no zoneinfo database
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    df = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2020-01-01 23:30:00"]).tz_localize(
                tz
            ),
        }
    )
    out = _to_polars_input(df)
    assert isinstance(out.schema["trd_exctn_dt"], pl.Datetime)
    assert out.schema["trd_exctn_dt"].time_zone is not None


def test_to_polars_input_keeps_non_datetime_date_named_columns():
    """A known date-column name that is not datetime-typed (e.g. a
    string) passes through unchanged."""
    df = pd.DataFrame({"datadate": ["2020-01-31"], "v": [1.0]})
    out = _to_polars_input(df)
    assert out.schema["datadate"] == pl.String


def test_to_polars_input_drops_default_rangeindex():
    out = _to_polars_input(pd.DataFrame({"a": [1, 2]}))
    assert "index" not in out.columns


def test_to_polars_input_converts_pandas_series():
    out = _to_polars_input(pd.Series([1, 2, 3], name="x"))
    assert isinstance(out, pl.Series)


def test_to_polars_input_leaves_dict_alone():
    d = {"a": 1}
    assert _to_polars_input(d) is d


# %% _convert_output


def test_convert_output_passthrough_for_polars_backend():
    tf.set_backend("polars")
    df = pl.DataFrame({"a": [1, 2]})
    assert _convert_output(df) is df


def test_convert_output_returns_pandas_by_default():
    out = _convert_output(pl.DataFrame({"a": [1, 2]}))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a"]


def test_convert_output_normalizes_date_to_datetime64_ns():
    out = _convert_output(
        pl.DataFrame({"date": [datetime.date(2020, 1, 31)], "v": [1.0]})
    )
    assert str(out["date"].dtype) == "datetime64[ns]"


def test_convert_output_converts_series():
    out = _convert_output(pl.Series("x", [1.0, 2.0]))
    assert isinstance(out, pd.Series)


def test_convert_output_recurses_into_dicts():
    out = _convert_output(
        {
            "coefficients": pl.DataFrame({"a": [1.0]}),
            "residuals": np.array([1.0, 2.0]),
        }
    )
    assert isinstance(out["coefficients"], pd.DataFrame)
    assert isinstance(out["residuals"], np.ndarray)


def test_convert_output_leaves_arrays_alone():
    arr = np.array([1.0, 2.0])
    assert _convert_output(arr) is arr


def test_convert_output_leaves_dict_of_scalars_alone():
    d = {"a": 1}
    assert _convert_output(d) == {"a": 1}


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


def _lag_input_pandas():
    return pd.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": list(pd.date_range("2023-01-01", periods=4, freq="MS")) * 2,
            "size": [float(i) for i in range(1, 9)],
        }
    )


def _lag_input_polars():
    return pl.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": (
                pl.date_range(
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 4, 1),
                    interval="1mo",
                    eager=True,
                ).to_list()
                * 2
            ),
            "size": [float(i) for i in range(1, 9)],
        }
    )


def test_core_function_returns_polars_for_polars_input():
    tf.set_backend("polars")
    out = tf.add_lagged_columns(
        _lag_input_polars(), cols="size", lag="1mo", by="permno"
    )
    assert isinstance(out, pl.DataFrame)
    assert "size_lag" in out.columns


def test_core_function_returns_pandas_under_pandas_backend():
    out = tf.add_lagged_columns(
        _lag_input_pandas(),
        cols="size",
        lag=pd.DateOffset(months=1),
        by="permno",
    )
    assert isinstance(out, pd.DataFrame)


def test_core_accepts_polars_input_even_on_pandas_backend():
    out = tf.add_lagged_columns(
        _lag_input_polars(), cols="size", lag="1mo", by="permno"
    )
    assert isinstance(out, pd.DataFrame)


def test_series_returning_function_matches_backend():
    rng = np.random.default_rng(42)
    data = pl.DataFrame({"id": range(100), "value": rng.random(100)})
    result = tf.assign_portfolio(
        data, "value", breakpoint_options={"n_portfolios": 5}
    )
    assert isinstance(result, pd.Series)
    tf.set_backend("polars")
    result = tf.assign_portfolio(
        data, "value", breakpoint_options={"n_portfolios": 5}
    )
    assert isinstance(result, pl.Series)


def test_series_output_aligns_with_input_frame_index():
    """Row-aligned Series outputs carry the pandas input's index, so
    'df["pf"] = tf.assign_portfolio(df, ...)' on a filtered frame with
    a non-default index does not misalign rows."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "exchange": ["NYSE", "NASDAQ"] * 50,
            "mktcap": rng.random(100),
        }
    )
    sub = data[data["exchange"] == "NYSE"]
    result = tf.assign_portfolio(
        sub, "mktcap", breakpoint_options={"n_portfolios": 5}
    )
    assert isinstance(result, pd.Series)
    assert result.index.equals(sub.index)


def test_rolling_value_callback_receives_backend_frame():
    """compute_rolling_value hands the window to the callback in the
    active backend's frame type, so pandas-style callbacks (e.g.
    '.iloc') keep working under the default backend."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([f"2020-{m:02d}-01" for m in range(1, 7)]),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out = tf.compute_rolling_value(
        df,
        f=lambda x: x["value"].iloc[-1],
        period="month",
        periods=2,
        min_obs=1,
    )
    assert out[-1] == 6.0

    tf.set_backend("polars")
    out_pl = tf.compute_rolling_value(
        pl.from_pandas(df),
        f=lambda x: x["value"][-1],
        period="month",
        periods=2,
        min_obs=1,
    )
    assert out_pl[-1] == 6.0


def test_day_count_dateoffset_lags_by_days():
    """'pd.DateOffset(k)' without keyword components means k days in
    pandas; it must not collapse to a zero lag."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.date_range("2020-01-01", periods=5)),
            "size": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = tf.add_lagged_columns(df, cols="size", lag=pd.DateOffset(2))
    expected = tf.add_lagged_columns(df, cols="size", lag="2d")
    pd.testing.assert_frame_equal(out, expected)
    assert out["size_lag"].iloc[2] == 1.0


# %% boundary-only wrapping protects internal cross-calls


def test_in_module_implementations_are_polars_only():
    """The wrapping is applied only at the public package boundary. The
    in-module implementations (which core functions call internally)
    operate on polars and return polars regardless of the backend."""
    import tidyfinance.lagging as lagging

    out = lagging.add_lagged_columns(
        _lag_input_polars(), cols="size", lag="1mo", by="permno"
    )
    assert isinstance(out, pl.DataFrame)


# %% Smoke tests: every wrapped function should round-trip its data
# argument through the backend without raising and produce the
# expected output type.


def _panel_with_returns():
    """Five-asset, two-year panel — enough per cross-section to fit
    three breakpoints without ties."""
    rng = np.random.default_rng(7)
    n_permnos = 5
    dates = pl.date_range(
        datetime.date(2020, 1, 1),
        datetime.date(2021, 12, 1),
        interval="1mo",
        eager=True,
    ).to_list()
    n = n_permnos * len(dates)
    return pl.DataFrame(
        {
            "permno": np.repeat(np.arange(1, n_permnos + 1), len(dates)),
            "date": dates * n_permnos,
            "ret_excess": rng.standard_normal(n) * 0.02,
            "mkt_excess": rng.standard_normal(n) * 0.02,
            "size": rng.uniform(1.0, 100.0, n),
            "mktcap_lag": rng.uniform(1.0, 100.0, n),
            "exchange": ["NYSE"] * n,
        }
    )


def test_compute_breakpoints_polars_input_returns_ndarray():
    tf.set_backend("polars")
    out = tf.compute_breakpoints(
        _panel_with_returns(), "size", breakpoint_options={"n_portfolios": 3}
    )
    # compute_breakpoints returns np.ndarray; backend leaves arrays
    # alone.
    assert isinstance(out, np.ndarray)


def test_compute_portfolio_returns_round_trips_polars():
    tf.set_backend("polars")
    out = tf.compute_portfolio_returns(
        data=_panel_with_returns(),
        sorting_variables="size",
        sorting_method="univariate",
        breakpoint_options_main={"n_portfolios": 3},
    )
    assert isinstance(out, pl.DataFrame)


def test_compute_long_short_returns_round_trips_pandas():
    portfolio_returns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "portfolio": [1, 1, 1, 3, 3, 3],
            "ret_excess_vw": [0.01, -0.02, 0.005, 0.03, 0.01, 0.04],
        }
    )
    out = tf.compute_long_short_returns(portfolio_returns)
    assert isinstance(out, pd.DataFrame)


def test_compute_rolling_value_returns_ndarray_under_polars():
    tf.set_backend("polars")
    out = tf.compute_rolling_value(
        _panel_with_returns(),
        f=lambda d: d["ret_excess"].mean(),
        period="month",
        periods=6,
    )
    # compute_rolling_value returns np.ndarray; backend leaves arrays
    # alone.
    assert isinstance(out, np.ndarray)


def test_create_summary_statistics_round_trips_polars():
    tf.set_backend("polars")
    out = tf.create_summary_statistics(
        _panel_with_returns(), ["ret_excess", "size"]
    )
    assert isinstance(out, pl.DataFrame)


def test_estimate_betas_round_trips_polars():
    tf.set_backend("polars")
    out = tf.estimate_betas(
        _panel_with_returns(),
        model="ret_excess ~ mkt_excess",
        lookback="12mo",
        min_obs=8,
    )
    assert isinstance(out, pl.DataFrame)


def test_estimate_fama_macbeth_round_trips_polars():
    tf.set_backend("polars")
    out = tf.estimate_fama_macbeth(
        _panel_with_returns(),
        model="ret_excess ~ mkt_excess",
    )
    assert isinstance(out, pl.DataFrame)


def test_estimate_model_round_trips_polars():
    tf.set_backend("polars")
    out = tf.estimate_model(_panel_with_returns(), "ret_excess ~ mkt_excess")
    assert isinstance(out, pl.DataFrame)


def test_estimate_model_dict_output_converts_to_pandas():
    out = tf.estimate_model(
        _panel_with_returns(),
        "ret_excess ~ mkt_excess",
        output=["coefficients", "tstats"],
    )
    assert isinstance(out, dict)
    assert isinstance(out["coefficients"], pd.DataFrame)
    assert isinstance(out["tstats"], pd.DataFrame)


def test_implement_portfolio_sort_round_trips_polars():
    tf.set_backend("polars")
    pso = tf.portfolio_sort_options(
        breakpoint_options_main=tf.breakpoint_options(n_portfolios=3)
    )
    out = tf.implement_portfolio_sort(
        data=_panel_with_returns(),
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


def test_list_supported_datasets_returns_pandas_by_default():
    out = tf.list_supported_datasets()
    assert isinstance(out, pd.DataFrame)


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





# %% run all tests
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
