"""Test script for tidyfinance package."""

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
from tidyfinance.lagging import add_lagged_columns  # noqa: E402
from tidyfinance.portfolios import (  # noqa: E402
    breakpoint_options,
    compute_breakpoints,
)
from tidyfinance.regression import (  # noqa: E402
    _newey_west_se,
    estimate_betas,
    estimate_fama_macbeth,
)
from tidyfinance.utilities import create_summary_statistics  # noqa: E402


def _month_starts(start: dt.date, periods: int) -> list[dt.date]:
    out = []
    year, month = start.year, start.month
    for _ in range(periods):
        out.append(dt.date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def _month_ends(start_year: int, start_month: int, periods: int):
    import calendar

    out = []
    year, month = start_year, start_month
    for _ in range(periods):
        out.append(dt.date(year, month, calendar.monthrange(year, month)[1]))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


# %% Helper function to create test data
def create_test_data():
    np.random.seed(42)  # For reproducibility
    dates = _month_starts(dt.date(2023, 1, 1), 10)
    return pl.DataFrame(
        {
            "permno": np.repeat([1, 2], 10),
            "date": dates * 2,
            "bm": np.random.uniform(0.5, 1.5, 20),
            "size": np.random.uniform(100, 200, 20),
        }
    )


# %% Tests
def test_add_lagged_columns():
    """Test that lagged columns are added correctly"""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm", "size"],
        lag="3mo",
        by="permno",
    )

    # Check if lagged columns exist
    assert "bm_lag" in result.columns
    assert "size_lag" in result.columns

    # Check if the number of rows is preserved
    assert len(result) == len(data)


def test_negative_lag():
    """Test that negative lag raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lagged_columns(data, cols=["bm", "size"], lag=-1)


def test_invalid_max_lag():
    """Test that max_lag < lag raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lagged_columns(
            data,
            cols=["bm", "size"],
            lag="3mo",
            max_lag="1mo",
        )


def test_without_grouping():
    """Test function works without grouping"""
    data = create_test_data().filter(pl.col("permno") == 1).drop("permno")
    result = add_lagged_columns(
        data,
        cols=["bm", "size"],
        lag="3mo",
    )

    assert "bm_lag" in result.columns
    assert "size_lag" in result.columns
    assert len(result) == len(data)


def test_preserve_original_values():
    """Test that original column values are preserved"""
    data = create_test_data()
    result = add_lagged_columns(data, cols=["bm", "size"], lag=3, by="permno")

    assert result["bm"].to_list() == data["bm"].to_list()
    assert result["size"].to_list() == data["size"].to_list()


def test_lag_values_correctness():
    """Test that lag values are correct"""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm"],
        lag="1mo",
        by="permno",
    )

    # For each permno group, check if lag values are correct
    for permno in [1, 2]:
        group_data = result.filter(pl.col("permno") == permno).sort("date")
        orig_values = group_data["bm"].to_list()
        lag_values = group_data["bm_lag"].to_list()

        # Lagged values equal originals shifted by 1 month
        assert lag_values[1:] == orig_values[:-1]
        assert lag_values[0] is None  # First value has no source


def test_window_lag_produces_single_column():
    """Test that window lag (lag != max_lag) produces a single column."""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm"],
        lag="1mo",
        max_lag="3mo",
        by="permno",
    )

    # Window mode: one lag column per source col (no per-step columns)
    assert "bm_lag" in result.columns
    assert len(result) == len(data)


def test_invalid_column():
    """Test that invalid column names raise error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lagged_columns(data, cols=["invalid_column"], lag=1)


def test_invalid_date_column():
    """Test that invalid date column raises error"""
    data = create_test_data()
    with pytest.raises(ValueError):
        add_lagged_columns(data, cols=["bm"], lag=1, date_col="invalid_date")


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    permnos = [1, 2]
    return pl.DataFrame(
        {
            "date": dates * len(permnos),
            "permno": np.repeat(permnos, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "mkt_excess": np.random.randn(len(dates) * len(permnos)),
        }
    )


def test_estimate_rolling_betas_basic(sample_data: pl.DataFrame) -> None:
    lookback = "30d"
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)
    assert not result.is_empty(), "Result should not be empty"
    assert "beta_mkt_excess" in result.columns, (
        "Output should include beta estimate for mkt_excess"
    )


def test_estimate_rolling_betas_min_obs(sample_data: pl.DataFrame) -> None:
    """Windows with fewer than min_obs observations are dropped from
    the output rather than returned as null rows."""
    lookback = "30d"
    min_obs = 10
    full = estimate_betas(
        sample_data, "ret_excess ~ mkt_excess", lookback, min_obs=1
    )
    result = estimate_betas(
        sample_data, "ret_excess ~ mkt_excess", lookback, min_obs=min_obs
    )
    assert result.height > 0, "Result should have valid estimates"
    assert result["beta_mkt_excess"].is_null().sum() == 0, (
        "Sub-min_obs windows should be dropped, not null"
    )
    assert result.height < full.height, (
        "min_obs should remove the early sub-min_obs windows"
    )


def test_estimate_betas_min_obs_non_positive_raises(
    sample_data: pl.DataFrame,
) -> None:
    """min_obs <= 0 raises a ValueError."""
    for bad in (0, -5):
        with pytest.raises(ValueError, match="min_obs must be a positive"):
            estimate_betas(
                sample_data, "ret_excess ~ mkt_excess", "30d", min_obs=bad
            )


def test_estimate_betas_default_min_obs_is_80_percent(
    sample_data: pl.DataFrame,
) -> None:
    """min_obs defaults to 80% of lookback when not provided."""
    lookback = "30d"
    default = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)
    explicit = estimate_betas(
        sample_data,
        "ret_excess ~ mkt_excess",
        lookback,
        min_obs=round(30 * 0.8),
    )
    assert_frame_equal(default, explicit)


def test_estimate_betas_without_intercept_omits_intercept_column(
    sample_data: pl.DataFrame,
) -> None:
    """A '- 1' formula omits the intercept column."""
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess - 1", "30d")
    assert "intercept" not in result.columns
    assert "beta_mkt_excess" in result.columns


def test_estimate_betas_match_per_window_ols(
    sample_data: pl.DataFrame,
) -> None:
    """Estimated betas match a per-window OLS fit."""
    lookback = 30
    result = estimate_betas(
        sample_data, "ret_excess ~ mkt_excess", f"{lookback}d"
    )

    # The fixture has one observation per calendar day, so the 30-day
    # calendar window is exactly the trailing 30 rows.
    group = sample_data.filter(pl.col("permno") == 1).sort("date")
    i = 50
    window = group.slice(i - lookback + 1, lookback)
    design = np.column_stack(
        [np.ones(window.height), window["mkt_excess"].to_numpy()]
    )
    expected = np.linalg.lstsq(
        design, window["ret_excess"].to_numpy(), rcond=None
    )[0]

    row = result.filter(
        (pl.col("permno") == 1) & (pl.col("date") == group["date"][i])
    )
    np.testing.assert_allclose(
        row.select(["intercept", "beta_mkt_excess"]).to_numpy()[0],
        expected,
        rtol=1e-8,
    )


def test_estimate_betas_custom_id_column(
    sample_data: pl.DataFrame,
) -> None:
    """A non-default stock identifier column is honored."""
    renamed = sample_data.rename({"permno": "gvkey"})
    result = estimate_betas(
        renamed, "ret_excess ~ mkt_excess", "30d", id_col="gvkey"
    )
    assert "gvkey" in result.columns
    assert "permno" not in result.columns


def test_estimate_betas_invalid_formula_raises(
    sample_data: pl.DataFrame,
) -> None:
    """A formula without '~' raises a ValueError."""
    with pytest.raises(ValueError, match="must contain '~'"):
        estimate_betas(sample_data, "ret_excess mkt_excess", "30d")


# %% estimate_betas: calendar vs positional windows


def _gappy_monthly_panel() -> pl.DataFrame:
    """Monthly panel for one stock with a three-month hole in 2020."""
    gap = {dt.date(2020, 6, 1), dt.date(2020, 7, 1), dt.date(2020, 8, 1)}
    keep = [d for d in _month_starts(dt.date(2019, 1, 1), 36) if d not in gap]
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "permno": [1] * len(keep),
            "date": keep,
            "ret_excess": rng.standard_normal(len(keep)) * 0.05,
            "mkt_excess": rng.standard_normal(len(keep)) * 0.05,
        }
    )


def test_calendar_lookback_window_spans_calendar_periods() -> None:
    """A calendar window covers the trailing N periods, so a gap in the
    history shrinks the window rather than reaching further back."""
    data = _gappy_monthly_panel()
    result = estimate_betas(
        data, "ret_excess ~ mkt_excess", lookback="12mo", min_obs=5
    )

    target = dt.date(2020, 12, 1)
    got = result.filter(pl.col("date") == target)["beta_mkt_excess"][0]

    # 2020-01 through 2020-12 with the three-month gap: nine observations.
    window = data.filter(pl.col("date").is_between(dt.date(2020, 1, 1), target))
    assert window.height == 9
    design = np.column_stack(
        [np.ones(window.height), window["mkt_excess"].to_numpy()]
    )
    expected = np.linalg.lstsq(
        design, window["ret_excess"].to_numpy(), rcond=None
    )[0][1]
    assert got == pytest.approx(expected, rel=1e-10)


def test_calendar_and_positional_lookbacks_differ_on_gappy_panels() -> None:
    """The deprecated positional window reaches past the gap and so
    gives a different estimate than the calendar window."""
    data = _gappy_monthly_panel()
    calendar = estimate_betas(
        data, "ret_excess ~ mkt_excess", lookback="12mo", min_obs=5
    )
    with pytest.warns(DeprecationWarning, match="calendar window"):
        positional = estimate_betas(
            data, "ret_excess ~ mkt_excess", lookback=12, min_obs=5
        )

    target = dt.date(2020, 12, 1)
    a = calendar.filter(pl.col("date") == target)["beta_mkt_excess"][0]
    b = positional.filter(pl.col("date") == target)["beta_mkt_excess"][0]
    assert a != pytest.approx(b, rel=1e-12)


def test_calendar_and_positional_agree_on_balanced_panels() -> None:
    """With one observation per period and no gaps the two windows
    coincide, which is why the book's monthly examples are unaffected."""
    dates = _month_starts(dt.date(2020, 1, 1), 36)
    rng = np.random.default_rng(9)
    data = pl.DataFrame(
        {
            "permno": [1] * 36,
            "date": dates,
            "ret_excess": rng.standard_normal(36) * 0.05,
            "mkt_excess": rng.standard_normal(36) * 0.05,
        }
    )
    calendar = estimate_betas(
        data, "ret_excess ~ mkt_excess", lookback="12mo", min_obs=10
    )
    with pytest.warns(DeprecationWarning):
        positional = estimate_betas(
            data, "ret_excess ~ mkt_excess", lookback=12, min_obs=10
        )
    # The calendar path drops sub-min_obs windows;
    # the deprecated positional path keeps them as null rows.
    positional = positional.filter(pl.col("beta_mkt_excess").is_not_null())
    assert_frame_equal(calendar, positional)


def test_calendar_lookback_pools_observations_within_a_period() -> None:
    """Daily observations with a monthly window collapse to one row per
    month, each fitted on every observation in the trailing months."""
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    days = [d for d in days if d.weekday() < 5]
    rng = np.random.default_rng(3)
    data = pl.DataFrame(
        {
            "permno": [1] * len(days),
            "date": days,
            "ret_excess": rng.standard_normal(len(days)) * 0.01,
            "mkt_excess": rng.standard_normal(len(days)) * 0.01,
        }
    )
    result = estimate_betas(
        data, "ret_excess ~ mkt_excess", lookback="3mo", min_obs=40
    )

    # One row per month, dated at the start of the month. The first
    # month (Jan 2020) has fewer than min_obs=40 observations in its
    # trailing window and is dropped.
    assert result.height == data["date"].dt.truncate("1mo").n_unique() - 1
    assert result["date"].to_list() == sorted(result["date"].to_list())
    assert all(d.day == 1 for d in result["date"].to_list())

    target = dt.date(2020, 6, 1)
    got = result.filter(pl.col("date") == target)["beta_mkt_excess"][0]
    window = data.filter(
        pl.col("date").is_between(dt.date(2020, 4, 1), dt.date(2020, 6, 30))
    )
    design = np.column_stack(
        [np.ones(window.height), window["mkt_excess"].to_numpy()]
    )
    expected = np.linalg.lstsq(
        design, window["ret_excess"].to_numpy(), rcond=None
    )[0][1]
    assert got == pytest.approx(expected, rel=1e-10)


def test_integer_lookback_is_deprecated(sample_data: pl.DataFrame) -> None:
    """A bare integer still works but warns."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        estimate_betas(sample_data, "ret_excess ~ mkt_excess", 30)


def test_invalid_lookback_string_raises(sample_data: pl.DataFrame) -> None:
    """An unparseable or unsupported duration raises a ValueError."""
    for bad in ("60", "60y", "sixty", "-3mo", ""):
        with pytest.raises(ValueError, match="lookback"):
            estimate_betas(sample_data, "ret_excess ~ mkt_excess", bad)


def test_subday_lookback_requires_datetime_column(
    sample_data: pl.DataFrame,
) -> None:
    """An hour/minute/second window needs a datetime date column."""
    with pytest.raises(ValueError, match="requires a datetime"):
        estimate_betas(sample_data, "ret_excess ~ mkt_excess", "6h")


def test_default_min_obs_rounds_like_r() -> None:
    """min_obs defaults to round(0.8 * lookback), not truncation
    (6 -> 5, not 4)."""
    dates = _month_starts(dt.date(2020, 1, 1), 12)
    rng = np.random.default_rng(11)
    data = pl.DataFrame(
        {
            "permno": [1] * 12,
            "date": dates,
            "ret_excess": rng.standard_normal(12) * 0.05,
            "mkt_excess": rng.standard_normal(12) * 0.05,
        }
    )
    default = estimate_betas(data, "ret_excess ~ mkt_excess", "6mo")
    assert_frame_equal(
        default,
        estimate_betas(data, "ret_excess ~ mkt_excess", "6mo", min_obs=5),
    )


def sample_data_fmb() -> pl.DataFrame:
    np.random.seed(42)
    dates = _month_ends(2020, 1, 12)
    permnos = range(50)
    return pl.DataFrame(
        {
            "date": dates * len(permnos),
            "permno": np.repeat(list(permnos), len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "beta": np.random.randn(len(dates) * len(permnos)),
            "bm": np.random.randn(len(dates) * len(permnos)),
            "log_mktcap": np.random.randn(len(dates) * len(permnos)),
        }
    )


def test_estimate_fama_macbeth_basic(sample_data: pl.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap"
    )
    assert not result.is_empty(), "Result should not be empty"
    assert "risk_premium" in result.columns, (
        "Output should include risk premia estimates"
    )


def test_estimate_fama_macbeth_vcov(sample_data: pl.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap", vcov="iid"
    )
    assert "t_statistic" in result.columns, (
        "Output should include t-statistics based on vcov choice"
    )


def test_estimate_fama_macbeth_iid_t_statistic_is_mean_over_se() -> None:
    """With vcov='iid', the standard error is sd / sqrt(T) of the
    cross-sectional estimates and the t-statistic is risk_premium /
    standard_error, without an extra sqrt(T) factor."""
    data = sample_data_fmb()
    result = estimate_fama_macbeth(data, "ret_excess ~ beta", vcov="iid")

    slopes = []
    for group in data.sort("date").partition_by("date", maintain_order=True):
        x = np.column_stack([np.ones(group.height), group["beta"].to_numpy()])
        y = group["ret_excess"].to_numpy()
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        slopes.append(coef[1])
    slopes = np.array(slopes)
    expected_se = slopes.std(ddof=1) / np.sqrt(slopes.size)

    row = result.filter(pl.col("factor") == "beta")
    assert row["risk_premium"][0] == pytest.approx(slopes.mean())
    assert row["standard_error"][0] == pytest.approx(expected_se)
    assert row["t_statistic"][0] == pytest.approx(slopes.mean() / expected_se)


def test_estimate_fama_macbeth_invalid_vcov_raises() -> None:
    """An unsupported vcov option raises a ValueError."""
    with pytest.raises(ValueError, match="vcov must be either"):
        estimate_fama_macbeth(
            sample_data_fmb(),
            "ret_excess ~ beta + bm + log_mktcap",
            vcov="bogus",
        )


def test_estimate_fama_macbeth_missing_date_column_raises() -> None:
    """A missing date column raises a ValueError."""
    data = sample_data_fmb().drop("date")
    with pytest.raises(ValueError, match="must contain a date column"):
        estimate_fama_macbeth(data, "ret_excess ~ beta + bm + log_mktcap")


def test_estimate_fama_macbeth_small_cross_section_raises() -> None:
    """A date grouping with too few rows raises a ValueError.

    Every date grouping must have more rows than the number of
    variables in the model (dependent variable included); the model
    below has 4 variables, so a 4-row cross-section fails.
    """
    data = sample_data_fmb().filter(
        (pl.col("date") != pl.col("date").min()) | (pl.col("permno") < 4)
    )
    with pytest.raises(ValueError, match="more rows than the number"):
        estimate_fama_macbeth(data, "ret_excess ~ beta + bm + log_mktcap")


def test_estimate_fama_macbeth_data_options_date_column() -> None:
    """data_options={'date': ...} maps the date column like date_col."""
    data = sample_data_fmb()
    expected = estimate_fama_macbeth(
        data, "ret_excess ~ beta + bm + log_mktcap"
    )
    result = estimate_fama_macbeth(
        data.rename({"date": "month"}),
        "ret_excess ~ beta + bm + log_mktcap",
        data_options={"date": "month"},
    )
    assert result.equals(expected)


def test_estimate_fama_macbeth_unrecognized_vcov_options_raises() -> None:
    """Keys that sandwich::NeweyWest would reject raise instead of
    being silently ignored."""
    with pytest.raises(ValueError, match="Unrecognized vcov_options"):
        estimate_fama_macbeth(
            sample_data_fmb(),
            "ret_excess ~ beta + bm + log_mktcap",
            vcov_options={"kernel": "Bartlett"},
        )


def test_estimate_fama_macbeth_n_equals_number_of_periods() -> None:
    """The reported 'n' equals the number of distinct periods."""
    data = sample_data_fmb()
    result = estimate_fama_macbeth(data, "ret_excess ~ beta + bm + log_mktcap")
    assert (result["n"] == data["date"].n_unique()).all()


def test_estimate_fama_macbeth_detail() -> None:
    """detail=True returns coefficients and summary statistics."""
    data = sample_data_fmb()
    result = estimate_fama_macbeth(
        data, "ret_excess ~ beta + bm + log_mktcap", detail=True
    )
    assert set(result.keys()) == {"coefficients", "summary_statistics"}
    assert "risk_premium" in result["coefficients"].columns
    summary_statistics = result["summary_statistics"]
    assert list(summary_statistics.columns) == [
        "r_squared",
        "adj_r_squared",
        "n_obs",
    ]
    assert len(summary_statistics) == 1
    assert 0 <= summary_statistics["r_squared"][0] <= 1


# Fixed series with reference values computed in R via
# sqrt(as.numeric(sandwich::NeweyWest(lm(y ~ 1), ...))). These lock the
# Python estimator to R's sandwich::NeweyWest (issue #35).
_NW_FIXED_SERIES = np.array(
    [
        0.01,
        -0.02,
        0.015,
        0.03,
        -0.01,
        0.005,
        0.02,
        -0.025,
        0.01,
        0.0,
        0.018,
        -0.012,
        0.022,
        -0.008,
        0.014,
        0.006,
        -0.019,
        0.011,
        0.027,
        -0.003,
    ]
)


def test_newey_west_se_matches_r_default() -> None:
    """Default (prewhite=1, automatic NW1994 bandwidth) matches R sandwich."""
    se = _newey_west_se(_NW_FIXED_SERIES)  # lag=None, prewhite=1
    assert se == pytest.approx(0.000646974246259443, rel=1e-9)


def test_newey_west_se_matches_r_no_prewhitening() -> None:
    """prewhite=0 (automatic bandwidth) matches R sandwich."""
    se = _newey_west_se(_NW_FIXED_SERIES, prewhite=0)
    assert se == pytest.approx(0.00094140519968821, rel=1e-9)


def test_newey_west_se_matches_r_fixed_lag() -> None:
    """Explicit lag, with and without prewhitening, matches R sandwich."""
    se_pw0 = _newey_west_se(_NW_FIXED_SERIES, lag=3, prewhite=0)
    se_pw1 = _newey_west_se(_NW_FIXED_SERIES, lag=3, prewhite=1)
    assert se_pw0 == pytest.approx(0.00177500880279507, rel=1e-9)
    assert se_pw1 == pytest.approx(0.00140568050935899, rel=1e-9)


def test_newey_west_se_legacy_path_equals_statsmodels_hac() -> None:
    """The deprecated maxlags=6 path (lag=6, prewhite=0) equals statsmodels'
    HAC(maxlags=6), the pre-PR behavior. Anchors the legacy path to an
    absolute reference so it cannot silently regress."""
    se = _newey_west_se(_NW_FIXED_SERIES, lag=6, prewhite=0)
    assert se == pytest.approx(0.0012883225527793873, rel=1e-9)


def _sample_data_fmb_parity() -> pl.DataFrame:
    """Deterministic panel; reference values produced in R
    (vcov='newey-west') on the identical data."""
    rng = np.random.default_rng(987654)
    dates = _month_ends(2000, 1, 48)
    recs = []
    for d in dates:
        beta = rng.normal(1, 0.3, size=40)
        bm = rng.normal(0.5, 0.2, size=40)
        size = rng.normal(10, 1, size=40)
        eps = rng.normal(0, 0.05, size=40)
        ret = 0.002 + 0.0015 * beta - 0.003 * bm + 0.0008 * size + eps
        for p in range(40):
            recs.append((d, p, ret[p], beta[p], bm[p], size[p]))
    return pl.DataFrame(
        recs,
        schema=["date", "permno", "ret_excess", "beta", "bm", "size"],
        orient="row",
    )


def test_estimate_fama_macbeth_newey_west_matches_r() -> None:
    """End-to-end Fama-MacBeth t-statistics match the R reference
    values exactly.

    Reference (sandwich::NeweyWest default) rounded to 3 decimals:
    intercept -0.792, beta 2.301, bm 1.005, size 0.887.
    """
    out = estimate_fama_macbeth(
        _sample_data_fmb_parity(), "ret_excess ~ beta + bm + size"
    )
    t = dict(zip(out["factor"].to_list(), out["t_statistic"].to_list()))
    rp = dict(zip(out["factor"].to_list(), out["risk_premium"].to_list()))
    # Reference values are rounded to 3 decimals, so compare within half a
    # unit in the last place (abs=5e-4).
    assert t["intercept"] == pytest.approx(-0.792, abs=5e-4)
    assert t["beta"] == pytest.approx(2.301, abs=5e-4)
    assert t["bm"] == pytest.approx(1.005, abs=5e-4)
    assert t["size"] == pytest.approx(0.887, abs=5e-4)
    assert rp["beta"] == pytest.approx(0.007, abs=5e-4)


def test_estimate_fama_macbeth_maxlags_deprecated() -> None:
    """The legacy 'maxlags' key warns and maps to lag with prewhite=0."""
    data = sample_data_fmb()
    with pytest.warns(DeprecationWarning, match="maxlags"):
        legacy = estimate_fama_macbeth(
            data,
            "ret_excess ~ beta + bm + log_mktcap",
            vcov_options={"maxlags": 6},
        )
    explicit = estimate_fama_macbeth(
        data,
        "ret_excess ~ beta + bm + log_mktcap",
        vcov_options={"lag": 6, "prewhite": 0},
    )
    assert_frame_equal(legacy, explicit)


def sample_data_summary() -> pl.DataFrame:
    np.random.seed(42)
    return pl.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=100),
            "x": np.random.randn(100),
            "y": np.random.randint(0, 100, size=100),
            "z": np.random.randint(0, 100, size=100),
        }
    )


def test_create_summary_statistics_basic(sample_data) -> None:
    result = create_summary_statistics(sample_data_summary(), ["x", "y"])
    assert not result.is_empty(), "Result should not be empty"
    assert "mean" in result.columns, "Output should include mean calculation"


def test_create_summary_statistics_by_group(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], by="group"
    )
    assert "group" in result.columns, "Output should include group column"
    assert "mean" in result.columns, "Output should include mean calculation"
    # One row per (group, variable) combination in tidy long format.
    assert result.height == 4


def test_create_summary_statistics_detail(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], detail=True
    )
    assert "q01" in result.columns, (
        "Detailed statistics should include 1st percentile"
    )
    assert "q99" in result.columns, (
        "Detailed statistics should include 99th percentile"
    )


def test_create_summary_statistics_accepts_boolean() -> None:
    """Test boolean columns are summarized as proportion of True."""
    df = pl.DataFrame(
        {
            "flag": [True, False, True, True],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = create_summary_statistics(df, ["flag", "x"])
    flag_mean = result.filter(pl.col("variable") == "flag")["mean"][0]
    assert abs(flag_mean - 0.75) < 1e-12, (
        "Boolean mean should equal the proportion of True"
    )


def test_create_summary_statistics_rejects_strings() -> None:
    """Test string dtype columns raise ValueError."""
    df = pl.DataFrame({"name": ["A", "B", "C"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not numeric or boolean"):
        create_summary_statistics(df, ["name", "x"])


def test_create_summary_statistics_handles_na() -> None:
    """NA values are dropped before statistics are computed."""
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    result = create_summary_statistics(df, ["x"])
    assert result["n"][0] == 3
    assert result["mean"][0] == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def sample_data_ls() -> pl.DataFrame:
    np.random.seed(42)
    dates = _month_ends(2020, 1, 10)
    portfolios = [1, 2]
    return pl.DataFrame(
        {
            "date": dates * len(portfolios),
            "portfolio": np.repeat(portfolios, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(portfolios)),
        }
    )


def test_breakpoint_options_default():
    options = breakpoint_options()
    assert options["smooth_bunching"] is False, (
        "Default smooth_bunching should be False"
    )


def test_breakpoint_options_custom():
    options = breakpoint_options(
        n_portfolios=5,
        percentiles=[0.2, 0.4, 0.6, 0.8],
        breakpoint_exchanges="NYSE",
    )
    assert options["n_portfolios"] == 5, "Custom n_portfolios should be 5"
    assert options["breakpoint_exchanges"] == "NYSE", (
        "Custom exchange should be 'NYSE'"
    )


def test_breakpoint_options_invalid():
    with pytest.raises(ValueError):
        breakpoint_options(n_portfolios=-1)  # Invalid n_portfolios
    with pytest.raises(ValueError):
        breakpoint_options(percentiles=[-0.1, 1.2])  # Invalid percentiles


def sample_data_breakpoints() -> pl.DataFrame:
    np.random.seed(42)
    return pl.DataFrame(
        {
            "id": np.arange(1, 101),
            "exchange": np.random.choice(["NYSE", "NASDAQ"], 100),
            "market_cap": np.random.uniform(100, 1000, 100),
        }
    )


def test_compute_breakpoints_n_portfolios(
    sample_data=sample_data_breakpoints(),
):
    breakpoints = compute_breakpoints(
        sample_data, "market_cap", {"n_portfolios": 5}
    )
    assert len(breakpoints) >= 2, (
        "Breakpoints should include at least min/max boundaries"
    )


def test_compute_breakpoints_percentiles(sample_data=sample_data_breakpoints()):
    breakpoints = compute_breakpoints(
        sample_data, "market_cap", {"percentiles": [0.2, 0.4, 0.6, 0.8]}
    )
    assert len(breakpoints) >= 2, (
        "Breakpoints should include at least min/max boundaries"
    )


def test_compute_breakpoints_invalid_options(
    sample_data=sample_data_breakpoints(),
):
    with pytest.raises(ValueError):
        compute_breakpoints(
            sample_data,
            "market_cap",
            {"n_portfolios": 5, "percentiles": [0.2, 0.4]},
        )
    with pytest.raises(ValueError):
        compute_breakpoints(sample_data, "market_cap", {})


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
