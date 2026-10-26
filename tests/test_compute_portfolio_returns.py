"""Tests for compute_portfolio_returns."""

import datetime as dt
import os
import sys

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import (  # noqa: E402
    breakpoint_options,
    compute_breakpoints,
    compute_portfolio_returns,
)

# %% test data helper


def _month_starts(start: str, n_months: int) -> list:
    """Return 'n_months' consecutive month-start dates from 'start'."""
    first = dt.date.fromisoformat(start)
    offsets = [first.month - 1 + i for i in range(n_months)]
    return [
        dt.date(first.year + offset // 12, offset % 12 + 1, 1)
        for offset in offsets
    ]


def make_data(
    n_stocks=30, n_months=6, start="2020-01-01", seed=42, with_mktcap=True
):
    """Construct a stock-month panel for tests."""
    rng = np.random.default_rng(seed)
    dates = _month_starts(start, n_months)
    permnos = [permno for _ in dates for permno in range(1, n_stocks + 1)]
    date_col = [date for date in dates for _ in range(n_stocks)]
    n = len(permnos)
    columns = {
        "permno": permnos,
        "date": date_col,
        "ret_excess": rng.standard_normal(n),
        "size": rng.uniform(50, 150, n),
        "bm": rng.uniform(0.5, 2, n),
        "exchange": rng.choice(["NYSE", "NASDAQ"], n),
    }
    if with_mktcap:
        columns["mktcap_lag"] = rng.uniform(100, 1000, n)
    return pl.DataFrame(columns)


bp3 = breakpoint_options(n_portfolios=3)
bp2 = breakpoint_options(n_portfolios=2)


# %% argument validation


def test_quiet_must_be_a_single_logical():
    """Test quiet must be a single boolean."""
    with pytest.raises(ValueError, match="quiet"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            quiet="yes",
        )


def test_sorting_variables_cannot_be_empty():
    """Test sorting_variables cannot be empty."""
    with pytest.raises(ValueError, match="sorting variable"):
        compute_portfolio_returns(
            make_data(),
            [],
            "univariate",
            breakpoint_options_main=bp3,
        )


def test_invalid_sorting_method_raises_error():
    """Test invalid sorting_method raises error."""
    with pytest.raises(ValueError, match="sorting method"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bad-method",
            breakpoint_options_main=bp3,
        )


def test_bivariate_without_secondary_breakpoints_warns():
    """Test bivariate without secondary breakpoints emits warning."""
    with pytest.warns(UserWarning, match="breakpoint_options_secondary"):
        try:
            compute_portfolio_returns(
                make_data(),
                ["size", "bm"],
                "bivariate-dependent",
                breakpoint_options_main=bp3,
                quiet=True,
            )
        except Exception:
            pass


def test_cap_weight_outside_0_1_raises_error():
    """Test cap_weight outside [0, 1] raises error."""
    with pytest.raises(ValueError, match="cap_weight"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            cap_weight=2,
        )


def test_errors_when_breakpoint_options_main_is_missing():
    """Test errors when breakpoint_options_main is missing."""
    with pytest.raises(ValueError, match="breakpoint_options_main"):
        compute_portfolio_returns(make_data(), "size", "univariate")


def test_missing_column_raises_error():
    """Test missing column raises error."""
    with pytest.raises(ValueError, match="Missing columns"):
        compute_portfolio_returns(
            make_data(),
            "no_such_col",
            "univariate",
            breakpoint_options_main=bp3,
        )


def test_rebalancing_month_outside_1_12_raises_error():
    """Test rebalancing_month outside 1-12 raises error."""
    with pytest.raises(ValueError, match="rebalancing_month"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "univariate",
            rebalancing_month=13,
            breakpoint_options_main=bp3,
        )


# %% univariate periodic


def test_univariate_periodic_without_mktcap_returns_ew_only():
    """Test univariate periodic without mktcap returns ew only."""
    result = compute_portfolio_returns(
        make_data(with_mktcap=False),
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert list(result.columns) == ["portfolio", "date", "ret_excess_ew"]
    assert len(result) == 3 * 6


def test_univariate_periodic_with_mktcap_returns_all_ret_cols():
    """Test univariate periodic with mktcap returns all return columns."""
    result = compute_portfolio_returns(
        make_data(),
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert list(result.columns) == [
        "portfolio",
        "date",
        "ret_excess_vw",
        "ret_excess_ew",
        "ret_excess_vw_capped",
    ]
    assert len(result) == 3 * 6


def test_all_na_mktcap_lag_gives_na_value_weighted_returns():
    """Test all-missing mktcap_lag gives null value-weighted returns."""
    data = make_data().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("mktcap_lag")
    )
    result = compute_portfolio_returns(
        data,
        "size",
        "univariate",
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert result["ret_excess_vw"].is_null().all()


def test_all_na_sorting_var_emits_message_and_returns_0_rows():
    """Test all-missing sorting variable warns and returns 0 rows."""
    data = make_data().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("size")
    )
    with pytest.warns(UserWarning, match="empty panel"):
        result = compute_portfolio_returns(
            data,
            "size",
            "univariate",
            breakpoint_options_main=bp3,
        )
    assert len(result) == 0


def test_univariate_sort_with_two_variables_raises_error():
    """Test univariate sort with two variables raises error."""
    with pytest.raises(ValueError, match="one sorting variable"):
        compute_portfolio_returns(
            make_data(),
            ["size", "bm"],
            "univariate",
            breakpoint_options_main=bp3,
        )


# %% univariate annual


def test_univariate_annual_no_data_in_month_raises_early_error():
    """Test univariate annual no data in month raises early error."""
    with pytest.raises(ValueError, match="No observations match"):
        compute_portfolio_returns(
            make_data(n_months=6, start="2020-01-01"),
            "size",
            "univariate",
            rebalancing_month=12,
            breakpoint_options_main=bp3,
        )


def test_univariate_annual_rebalancing_produces_valid_output():
    """Test univariate annual rebalancing produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        "size",
        "univariate",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        quiet=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0


def test_min_portfolio_size_produces_nas_and_emits_message():
    """Test min_portfolio_size produces nulls and emits warning."""
    with pytest.warns(UserWarning, match="missing"):
        result = compute_portfolio_returns(
            make_data(n_stocks=5),
            "size",
            "univariate",
            breakpoint_options_main=bp3,
            min_portfolio_size=10,
        )
    assert result["ret_excess_ew"].is_null().any()


# %% bivariate-dependent


def test_bivariate_dependent_with_one_variable_raises_error():
    """Test bivariate-dependent with one variable raises error."""
    with pytest.raises(ValueError, match="two sorting variables"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bivariate-dependent",
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
        )


def test_bivariate_dependent_periodic_produces_valid_output():
    """Test bivariate-dependent periodic produces valid output."""
    result = compute_portfolio_returns(
        make_data(),
        ["size", "bm"],
        "bivariate-dependent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0


def test_bivariate_dependent_annual_produces_valid_output():
    """Test bivariate-dependent annual produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        ["size", "bm"],
        "bivariate-dependent",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0


def test_bivariate_dependent_annual_no_match_raises_early_error():
    """Test bivariate-dependent annual no-match raises early error."""
    with pytest.raises(ValueError, match="No observations match"):
        compute_portfolio_returns(
            make_data(n_months=6, start="2020-01-01"),
            ["size", "bm"],
            "bivariate-dependent",
            rebalancing_month=12,
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
            quiet=True,
        )


def test_all_na_mktcap_in_bivariate_coerces_nan_vw_to_na():
    """Test all-missing mktcap in bivariate produces null vw returns."""
    data = make_data().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("mktcap_lag")
    )
    result = compute_portfolio_returns(
        data,
        ["size", "bm"],
        "bivariate-dependent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert result["ret_excess_vw"].is_null().all()


# %% bivariate-independent


def test_bivariate_independent_with_one_variable_raises_error():
    """Test bivariate-independent with one variable raises error."""
    with pytest.raises(ValueError, match="two sorting variables"):
        compute_portfolio_returns(
            make_data(),
            "size",
            "bivariate-independent",
            breakpoint_options_main=bp3,
            breakpoint_options_secondary=bp2,
        )


def test_bivariate_independent_periodic_produces_valid_output():
    """Test bivariate-independent periodic produces valid output."""
    result = compute_portfolio_returns(
        make_data(),
        ["size", "bm"],
        "bivariate-independent",
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0


def test_bivariate_independent_annual_produces_valid_output():
    """Test bivariate-independent annual produces valid output."""
    result = compute_portfolio_returns(
        make_data(n_months=14, start="2019-07-01"),
        ["size", "bm"],
        "bivariate-independent",
        rebalancing_month=7,
        breakpoint_options_main=bp3,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0


# %% capped value weights


def cap_panel():
    """One date, two portfolios. NYSE stocks are small, NASDAQ stocks
    are large, so the NYSE and the all-stock percentiles differ."""
    return pl.DataFrame(
        {
            "permno": list(range(1, 11)),
            "date": [dt.date(2020, 1, 1)] * 10,
            "size": [float(i) for i in range(1, 11)],
            "exchange": ["NYSE"] * 5 + ["NASDAQ"] * 5,
            "mktcap_lag": [1.0, 2, 3, 4, 5, 60, 70, 80, 90, 100],
            "ret_excess": [0.01] * 5 + [0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )


def expected_capped(data, cap):
    """Capped value-weighted returns of the two size portfolios."""
    w = np.minimum(data["mktcap_lag"].to_numpy(), cap)
    ret = data["ret_excess"].to_numpy()
    top = data["size"].to_numpy() > 5
    return [
        np.average(ret[~top], weights=w[~top]),
        np.average(ret[top], weights=w[top]),
    ]


def test_capped_weights_use_the_nyse_percentile_by_default():
    """Test capped weights use the NYSE percentile by default."""
    data = cap_panel()
    out = compute_portfolio_returns(
        data,
        "size",
        "univariate",
        breakpoint_options_main=bp2,
        quiet=True,
    )
    nyse_cap = np.quantile([1, 2, 3, 4, 5], 0.8)
    np.testing.assert_allclose(
        out["ret_excess_vw_capped"].to_list(),
        expected_capped(data, nyse_cap),
    )


def test_cap_weight_exchanges_none_caps_at_the_percentile_of_all_stocks():
    """Test cap_weight_exchanges=None caps at the percentile of all
    stocks."""
    data = cap_panel()
    out = compute_portfolio_returns(
        data,
        "size",
        "univariate",
        breakpoint_options_main=bp2,
        cap_weight_exchanges=None,
        quiet=True,
    )
    all_cap = np.quantile(data["mktcap_lag"].to_numpy(), 0.8)
    np.testing.assert_allclose(
        out["ret_excess_vw_capped"].to_list(),
        expected_capped(data, all_cap),
    )


def test_capping_falls_back_to_all_stocks_without_an_exchange_column():
    """Test capping falls back to all stocks without an exchange
    column."""
    data = cap_panel().drop("exchange")
    with pytest.warns(UserWarning, match="not found"):
        out = compute_portfolio_returns(
            data,
            "size",
            "univariate",
            breakpoint_options_main=bp2,
        )
    all_cap = np.quantile(data["mktcap_lag"].to_numpy(), 0.8)
    np.testing.assert_allclose(
        out["ret_excess_vw_capped"].to_list(),
        expected_capped(data, all_cap),
    )


def test_capping_falls_back_to_all_stocks_on_dates_without_nyse_stocks():
    """Test capping falls back to all stocks on dates without NYSE
    stocks."""
    data = cap_panel().with_columns(exchange=pl.lit("NASDAQ"))
    with pytest.warns(UserWarning, match="on 1 date"):
        out = compute_portfolio_returns(
            data,
            "size",
            "univariate",
            breakpoint_options_main=bp2,
        )
    all_cap = np.quantile(data["mktcap_lag"].to_numpy(), 0.8)
    np.testing.assert_allclose(
        out["ret_excess_vw_capped"].to_list(),
        expected_capped(data, all_cap),
    )


def test_cap_weight_exchanges_is_validated():
    """Test cap_weight_exchanges is validated."""
    with pytest.raises(ValueError, match="cap_weight_exchanges"):
        compute_portfolio_returns(
            cap_panel(),
            "size",
            "univariate",
            breakpoint_options_main=bp2,
            cap_weight_exchanges=1,
        )


# %% dependent sorts with a minimum size threshold


def dependent_min_mktcap_panel():
    """Secondary sort splits at the median size. The main breakpoints
    exclude stocks below the 25th size percentile of the whole date
    (min_mktcap 25.75), not of each size half. In the small half, the
    main median is then 38 (stocks 26 to 50), so stock 35 is in the low
    portfolio. A minimum within the small half (13.25) would set the
    median at 32 and put stock 35 in the high one. Only stock 35 has a
    non-zero return, so its portfolio is the only one with a non-zero
    return."""
    return pl.DataFrame(
        {
            "permno": list(range(1, 101)),
            "date": [dt.date(2020, 7, 1)] * 100,
            "mktcap_lag": [float(i) for i in range(1, 101)],
            "x": [float(i) for i in range(1, 101)],
            "ret_excess": [1.0 if i == 35 else 0.0 for i in range(1, 101)],
        }
    )


def dependent_min_mktcap_sort(data, **kwargs):
    """Dependent sort with a minimum size threshold on the main
    breakpoints."""
    return compute_portfolio_returns(
        data,
        ["x", "mktcap_lag"],
        "bivariate-dependent",
        breakpoint_options_main=breakpoint_options(
            n_portfolios=2,
            breakpoints_min_size_threshold=0.25,
        ),
        breakpoint_options_secondary=bp2,
        cap_weight_exchanges=None,
        quiet=True,
        **kwargs,
    )


def test_dependent_sorts_compute_the_minimum_market_cap_once_per_date():
    """Test dependent sorts compute the minimum market cap once per
    date."""
    out = dependent_min_mktcap_sort(dependent_min_mktcap_panel())
    assert out.filter(pl.col("portfolio") == 1)["ret_excess_ew"][0] > 0
    assert out.filter(pl.col("portfolio") == 2)["ret_excess_ew"][0] == 0
    assert "_min_mktcap_main" not in out.columns


def test_dependent_sorts_with_annual_rebalancing_use_per_date_min_mktcap():
    """Test dependent sorts with annual rebalancing use per-date
    min_mktcap."""
    out = dependent_min_mktcap_sort(
        dependent_min_mktcap_panel(),
        rebalancing_month=7,
    )
    assert out.filter(pl.col("portfolio") == 1)["ret_excess_ew"][0] > 0
    assert out.filter(pl.col("portfolio") == 2)["ret_excess_ew"][0] == 0


def test_dependent_sorts_pass_options_unchanged_to_custom_breakpoints():
    """Test dependent sorts pass options unchanged to custom
    breakpoints."""
    received = []

    def custom_breakpoints(
        data, sorting_variable, breakpoint_options, data_options=None
    ):
        received.append(breakpoint_options)
        return compute_breakpoints(data, sorting_variable, breakpoint_options)

    dependent_min_mktcap_sort(
        dependent_min_mktcap_panel(),
        breakpoint_function_main=custom_breakpoints,
    )
    assert received
    assert all("breakpoints_min_mktcap" not in x for x in received)


def test_dependent_min_mktcap_with_exchanges_requires_the_exchange_column():
    """Test dependent min_mktcap with exchanges requires the exchange
    column."""
    with pytest.raises(ValueError, match="exchange"):
        compute_portfolio_returns(
            dependent_min_mktcap_panel(),
            ["x", "mktcap_lag"],
            "bivariate-dependent",
            breakpoint_options_main=breakpoint_options(
                n_portfolios=2,
                breakpoints_exchanges="NYSE",
                breakpoints_min_size_threshold=0.25,
            ),
            breakpoint_options_secondary=bp2,
            cap_weight_exchanges=None,
            quiet=True,
        )


def test_a_missing_return_names_the_missing_return_as_a_cause():
    """Test a missing stock return makes the portfolio return missing."""
    data = pl.DataFrame(
        {
            "permno": range(1, 7),
            "date": [dt.date(2020, 1, 1)] * 6,
            "s": [float(x) for x in range(1, 7)],
            "ret_excess": [0.01, 0.02, 0.03, 0.04, 0.05, None],
        }
    )
    with pytest.warns(UserWarning, match="missing return"):
        result = compute_portfolio_returns(
            data,
            "s",
            "univariate",
            breakpoint_options_main=breakpoint_options(n_portfolios=2),
        )
    assert result["ret_excess_ew"].to_list() == pytest.approx([0.02, None])


def test_a_missing_return_is_skipped_across_secondary_portfolios():
    """Test a bivariate cell with a missing return drops out of the
    average across secondary portfolios."""
    data = pl.DataFrame(
        {
            "permno": range(1, 9),
            "date": [dt.date(2020, 1, 1)] * 8,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "ret_excess": [0.01, None, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )
    result = compute_portfolio_returns(
        data,
        ["x", "y"],
        "bivariate-independent",
        breakpoint_options_main=bp2,
        breakpoint_options_secondary=bp2,
        quiet=True,
    )
    assert result["ret_excess_ew"].to_list() == pytest.approx([0.02, 0.065])


def test_the_complete_panel_is_ordered_by_portfolio_and_date():
    """Test the output is sorted by portfolio and date, whatever the
    input order, and missing portfolio-dates are kept as nulls."""
    d = pl.DataFrame(
        {
            "permno": [1, 2, 2, 1, 2],
            "date": [
                dt.date(2020, 2, 1),
                dt.date(2020, 1, 1),
                dt.date(2020, 2, 1),
                dt.date(2020, 3, 1),
                dt.date(2020, 3, 1),
            ],
            "size": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ret_excess": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    options = breakpoint_options(n_portfolios=2)
    # 2020-01-01 has a single stock, which triggers a one-portfolio warning
    with pytest.warns(UserWarning):
        result = compute_portfolio_returns(
            d, "size", "univariate", breakpoint_options_main=options, quiet=True
        )
    dates = [dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1)]
    assert result["portfolio"].to_list() == [1, 1, 1, 2, 2, 2]
    assert result["date"].to_list() == dates * 2
    assert result["ret_excess_ew"].to_list() == [
        0.02,
        0.01,
        0.04,
        None,
        0.03,
        0.05,
    ]

    with pytest.warns(UserWarning):
        shuffled = compute_portfolio_returns(
            d[[4, 2, 0, 3, 1]],
            "size",
            "univariate",
            breakpoint_options_main=options,
            quiet=True,
        )
    assert_frame_equal(shuffled, result)


if __name__ == "__main__":
    pytest.main([__file__])
