"""Tests for staggered portfolios (holding_period and skip)."""

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
    compute_portfolio_returns,
    filter_options,
    implement_portfolio_sort,
    portfolio_sort_options,
)

RET_COLS = ["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"]

# Single-stock cohorts produce constant sorting variables on purpose.
ignore_constant = pytest.mark.filterwarnings("ignore:.*constant")


def _month_starts(n_months, start=dt.date(2020, 1, 1)):
    """Return 'n_months' consecutive month-start dates from 'start'."""
    return [
        dt.date(
            start.year + (start.month - 1 + i) // 12,
            (start.month - 1 + i) % 12 + 1,
            1,
        )
        for i in range(n_months)
    ]


def staggered_panel(n_stocks=40, n_months=12):
    """Balanced panel: every stock in every month, so a staggered sort
    with K cohorts equals the average of K monthly sorts on the signal
    lagged by 0, ..., K - 1 months."""
    rng = np.random.default_rng(324)
    dates = _month_starts(n_months)
    n = n_stocks * n_months
    return pl.DataFrame(
        {
            "permno": [p for p in range(1, n_stocks + 1) for _ in dates],
            "date": dates * n_stocks,
            "signal": rng.standard_normal(n),
            "size": rng.uniform(1, 100, n),
            "mktcap_lag": rng.uniform(10, 1000, n),
            "ret_excess": rng.normal(0.01, 0.05, n),
            "exchange": rng.choice(["NYSE", "NASDAQ"], n),
        }
    )


def lagged_sort_average(data, holding_period, skip=0, **kwargs):
    """Average of monthly sorts on the sorting variables lagged by
    skip, ..., skip + holding_period - 1 months."""
    kwargs.setdefault("cap_weight_exchanges", None)
    lags = range(skip, skip + holding_period)
    sorts = [
        compute_portfolio_returns(
            data.with_columns(
                [
                    pl.col(c).shift(k).over("permno")
                    for c in ("signal", "size")
                ]
            ),
            quiet=True,
            **kwargs,
        )
        for k in lags
    ]
    return (
        pl.concat(sorts)
        .group_by(["portfolio", "date"])
        .agg([pl.col(c).mean() for c in RET_COLS])
    )


def sorted_by_portfolio_date(data):
    return data.sort(["portfolio", "date"])


def test_staggered_univariate_sort_averages_k_lagged_monthly_sorts():
    """Test staggered univariate sort averages K lagged monthly sorts."""
    data = staggered_panel()
    # Missing market caps must not enter the cap of held stocks as zeros.
    data = data.with_columns(
        mktcap_lag=pl.when(pl.int_range(pl.len()).is_in([4, 49, 149, 299]))
        .then(None)
        .otherwise(pl.col("mktcap_lag"))
    )
    out = compute_portfolio_returns(
        data,
        "signal",
        "univariate",
        holding_period=3,
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
        cap_weight_exchanges=None,
        quiet=True,
    )
    expected = lagged_sort_average(
        data,
        3,
        sorting_variables="signal",
        sorting_method="univariate",
        breakpoint_options_main=breakpoint_options(n_portfolios=5),
    ).filter(pl.col("date") >= dt.date(2020, 3, 1))

    assert out["date"].min() == dt.date(2020, 3, 1)
    assert out.height == 5 * 10
    out = sorted_by_portfolio_date(out)
    expected = sorted_by_portfolio_date(expected)
    for c in RET_COLS:
        np.testing.assert_allclose(out[c].to_numpy(), expected[c].to_numpy())


def test_staggered_sort_caps_at_the_nyse_percentile_like_monthly_sorts():
    """Test the default NYSE cap of held stocks matches the cap of the
    lagged monthly sorts, also with missing market caps."""
    data = staggered_panel().with_columns(
        mktcap_lag=pl.when(pl.int_range(pl.len()).is_in([4, 49, 149, 299]))
        .then(None)
        .otherwise(pl.col("mktcap_lag"))
    )
    out = sorted_by_portfolio_date(
        compute_portfolio_returns(
            data,
            "signal",
            "univariate",
            holding_period=3,
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
            quiet=True,
        )
    )
    expected = sorted_by_portfolio_date(
        lagged_sort_average(
            data,
            3,
            cap_weight_exchanges="NYSE",
            sorting_variables="signal",
            sorting_method="univariate",
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
        ).filter(pl.col("date") >= dt.date(2020, 3, 1))
    )

    np.testing.assert_allclose(
        out["ret_excess_vw_capped"].to_numpy(),
        expected["ret_excess_vw_capped"].to_numpy(),
    )
    # The NYSE cap differs from the cap of all stocks.
    all_stocks = sorted_by_portfolio_date(
        compute_portfolio_returns(
            data,
            "signal",
            "univariate",
            holding_period=3,
            breakpoint_options_main=breakpoint_options(n_portfolios=5),
            cap_weight_exchanges=None,
            quiet=True,
        )
    )
    assert not np.allclose(
        out["ret_excess_vw_capped"].to_numpy(),
        all_stocks["ret_excess_vw_capped"].to_numpy(),
    )


def test_staggered_sort_with_skip_holds_cohorts_from_later_periods():
    """Test staggered sort with skip holds cohorts from later periods."""
    data = staggered_panel()
    out = sorted_by_portfolio_date(
        compute_portfolio_returns(
            data,
            "signal",
            "univariate",
            holding_period=2,
            skip=1,
            breakpoint_options_main=breakpoint_options(n_portfolios=2),
            cap_weight_exchanges=None,
            quiet=True,
        )
    )
    expected = sorted_by_portfolio_date(
        lagged_sort_average(
            data,
            2,
            skip=1,
            sorting_variables="signal",
            sorting_method="univariate",
            breakpoint_options_main=breakpoint_options(n_portfolios=2),
        ).filter(pl.col("date") >= dt.date(2020, 3, 1))
    )

    assert out["date"].min() == dt.date(2020, 3, 1)
    np.testing.assert_allclose(
        out["ret_excess_ew"].to_numpy(),
        expected["ret_excess_ew"].to_numpy(),
    )


@pytest.mark.parametrize(
    "method", ["bivariate-dependent", "bivariate-independent"]
)
def test_staggered_bivariate_sorts_average_k_lagged_monthly_sorts(method):
    """Test staggered bivariate sorts average K lagged monthly sorts."""
    data = staggered_panel(n_stocks=60)
    args = {
        "sorting_variables": ["signal", "size"],
        "sorting_method": method,
        "breakpoint_options_main": breakpoint_options(n_portfolios=3),
        "breakpoint_options_secondary": breakpoint_options(n_portfolios=2),
    }
    out = sorted_by_portfolio_date(
        compute_portfolio_returns(
            data,
            holding_period=2,
            cap_weight_exchanges=None,
            quiet=True,
            **args,
        )
    )
    expected = sorted_by_portfolio_date(
        lagged_sort_average(data, 2, **args).filter(
            pl.col("date") >= dt.date(2020, 2, 1)
        )
    )

    for c in RET_COLS:
        np.testing.assert_allclose(out[c].to_numpy(), expected[c].to_numpy())


@ignore_constant
def test_held_stocks_stay_in_their_cohort_without_a_sorting_variable():
    """Test held stocks stay in their cohort without a sorting variable.

    Two stocks, two portfolios. Stock 2 has no signal in February, but
    stays in the high portfolio of the January cohort."""
    data = pl.DataFrame(
        {
            "permno": [1, 2, 1, 2],
            "date": [dt.date(2020, 1, 1)] * 2 + [dt.date(2020, 2, 1)] * 2,
            "signal": [1.0, 2.0, 1.0, None],
            "mktcap_lag": [1.0] * 4,
            "ret_excess": [0.0, 0.0, 0.1, 0.2],
        }
    )
    out = compute_portfolio_returns(
        data,
        "signal",
        "univariate",
        holding_period=2,
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
        cap_weight_exchanges=None,
        quiet=True,
    )
    feb = out.filter(pl.col("date") == dt.date(2020, 2, 1))
    # Low: January cohort (stock 1) and February cohort (stock 1) -> 0.1.
    # High: only the January cohort holds stock 2 -> 0.2.
    low = feb.filter(pl.col("portfolio") == 1)["ret_excess_ew"][0]
    high = feb.filter(pl.col("portfolio") == 2)["ret_excess_ew"][0]
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.2)
    assert not (out["date"] == dt.date(2020, 1, 1)).any()


def test_holding_period_and_skip_are_validated():
    """Test holding_period and skip are validated."""
    data = staggered_panel(n_stocks=4, n_months=3)
    args = {
        "breakpoint_options_main": breakpoint_options(n_portfolios=2),
        "quiet": True,
    }
    with pytest.raises(ValueError, match="holding_period"):
        compute_portfolio_returns(
            data, "signal", "univariate", holding_period=0, **args
        )
    with pytest.raises(ValueError, match="holding_period"):
        compute_portfolio_returns(
            data, "signal", "univariate", holding_period=1.5, **args
        )
    with pytest.raises(ValueError, match="skip"):
        compute_portfolio_returns(
            data, "signal", "univariate", skip=-1, **args
        )
    with pytest.raises(ValueError, match="rebalancing_month"):
        compute_portfolio_returns(
            data,
            "signal",
            "univariate",
            holding_period=3,
            rebalancing_month=7,
            **args,
        )


def test_holding_period_1_and_skip_0_reproduce_monthly_sorts():
    """Test holding_period = 1 and skip = 0 reproduce monthly sorts."""
    data = staggered_panel()
    args = {
        "breakpoint_options_main": breakpoint_options(n_portfolios=5),
        "cap_weight_exchanges": None,
        "quiet": True,
    }
    assert_frame_equal(
        compute_portfolio_returns(
            data, "signal", "univariate", holding_period=1, skip=0, **args
        ),
        compute_portfolio_returns(data, "signal", "univariate", **args),
    )


@ignore_constant
def test_staggered_sorts_screen_only_at_formation():
    """Test staggered sorts screen only at formation.

    Stock 1 passes the price screen in January but not in February.
    With formation screens, it stays in the January cohort in February
    (-0.5), while the February cohort's low portfolio holds stock 2
    (0.1), so the February low return is (-0.5 + 0.1) / 2. With screens
    in every period, stock 1 also leaves the January cohort, so only
    the February cohort remains (0.1)."""
    data = pl.DataFrame(
        {
            "permno": [1, 2, 1, 2],
            "date": [dt.date(2020, 1, 1)] * 2 + [dt.date(2020, 2, 1)] * 2,
            "signal": [1.0, 2.0, 1.0, 2.0],
            "prc_lag": [10.0, 10.0, 1.0, 10.0],
            "mktcap_lag": [1.0] * 4,
            "ret_excess": [0.0, 0.0, -0.5, 0.1],
        }
    )
    options = portfolio_sort_options(
        filter_options=filter_options(min_stock_price=5),
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
    )
    out = implement_portfolio_sort(
        data,
        "signal",
        "univariate",
        holding_period=2,
        portfolio_sort_options=options,
        cap_weight_exchanges=None,
        quiet=True,
    )
    feb_low = out.filter(
        (pl.col("portfolio") == 1) & (pl.col("date") == dt.date(2020, 2, 1))
    )["ret_excess_ew"][0]
    assert feb_low == pytest.approx((-0.5 + 0.1) / 2)

    options_ex_post = portfolio_sort_options(
        filter_options=filter_options(
            min_stock_price=5,
            timing="every_period",
        ),
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
    )
    with pytest.warns(UserWarning, match="look-ahead"):
        out_ex_post = implement_portfolio_sort(
            data,
            "signal",
            "univariate",
            holding_period=2,
            portfolio_sort_options=options_ex_post,
            cap_weight_exchanges=None,
            quiet=True,
        )
    feb_low_ex_post = out_ex_post.filter(
        (pl.col("portfolio") == 1) & (pl.col("date") == dt.date(2020, 2, 1))
    )["ret_excess_ew"][0]
    assert feb_low_ex_post == pytest.approx(0.1)


def test_a_holding_period_longer_than_the_sample_returns_an_empty_panel():
    """Test a holding period longer than the sample returns an empty
    panel."""
    data = staggered_panel(n_stocks=10, n_months=3)
    out = compute_portfolio_returns(
        data,
        "signal",
        "univariate",
        holding_period=6,
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
        cap_weight_exchanges=None,
        quiet=True,
    )
    assert out.height == 0


def test_held_stocks_are_capped_on_dates_without_formation_rows():
    """Test held stocks are capped on dates without formation rows.

    No signals in March: no cohort is formed, but the February cohort
    is still held. Its capped weights use the cap of all stocks in
    March."""
    data = staggered_panel(n_stocks=20, n_months=3).with_columns(
        signal=pl.when(pl.col("date") == dt.date(2020, 3, 1))
        .then(None)
        .otherwise(pl.col("signal"))
    )
    out = compute_portfolio_returns(
        data,
        "signal",
        "univariate",
        holding_period=2,
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
        cap_weight_exchanges=None,
        quiet=True,
    )
    march = out.filter(pl.col("date") == dt.date(2020, 3, 1))
    assert march.height == 2
    assert march["ret_excess_vw_capped"].null_count() == 0


if __name__ == "__main__":
    pytest.main([__file__])
