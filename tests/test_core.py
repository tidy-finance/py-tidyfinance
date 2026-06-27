"""Test script for tidyfinance package."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from tidyfinance.lagging import add_lagged_columns  # noqa: E402
from tidyfinance.portfolios import breakpoint_options, compute_breakpoints  # noqa: E402
from tidyfinance.regression import _newey_west_se, estimate_betas, estimate_fama_macbeth  # noqa: E402
from tidyfinance.utilities import create_summary_statistics  # noqa: E402


# %% Helper function to create test data
def create_test_data():
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2023-01-01", periods=10, freq="MS")
    data = {
        "permno": np.repeat([1, 2], 10),
        "date": np.tile(dates, 2),
        "bm": np.random.uniform(0.5, 1.5, 20),
        "size": np.random.uniform(100, 200, 20),
    }
    return pd.DataFrame(data)


# %% Tests
def test_add_lagged_columns():
    """Test that lagged columns are added correctly"""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm", "size"],
        lag=pd.DateOffset(months=3),
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
            lag=pd.DateOffset(months=3),
            max_lag=pd.DateOffset(months=1),
        )


def test_without_grouping():
    """Test function works without grouping"""
    data = (
        create_test_data()
        .query("permno == 1")
        .drop(columns="permno")
        .reset_index(drop=True)
    )
    result = add_lagged_columns(
        data,
        cols=["bm", "size"],
        lag=pd.DateOffset(months=3),
    )

    assert "bm_lag" in result.columns
    assert "size_lag" in result.columns
    assert len(result) == len(data)


def test_preserve_original_values():
    """Test that original column values are preserved"""
    data = create_test_data()
    result = add_lagged_columns(data, cols=["bm", "size"], lag=3, by="permno")

    # Convert to lists for comparison
    assert result.get("bm").to_list() == data.get("bm").to_list()
    assert result.get("size").to_list() == data.get("size").to_list()


def test_lag_values_correctness():
    """Test that lag values are correct"""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm"],
        lag=pd.DateOffset(months=1),
        by="permno",
    )

    # For each permno group, check if lag values are correct
    for permno in [1, 2]:
        group_data = result.query("permno == @permno").sort_values("date")
        orig_values = group_data["bm"].to_list()
        lag_values = group_data["bm_lag"].to_list()

        # Lagged values equal originals shifted by 1 month
        assert lag_values[1:] == orig_values[:-1]
        assert np.isnan(lag_values[0])  # First value has no source


def test_window_lag_produces_single_column():
    """Test that window lag (lag != max_lag) produces a single column."""
    data = create_test_data()
    result = add_lagged_columns(
        data,
        cols=["bm"],
        lag=pd.DateOffset(months=1),
        max_lag=pd.DateOffset(months=3),
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
def sample_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    permnos = [1, 2]
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(permnos)),
            "permno": np.repeat(permnos, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "mkt_excess": np.random.randn(len(dates) * len(permnos)),
        }
    )
    return data


def test_estimate_rolling_betas_basic(sample_data: pd.DataFrame) -> None:
    lookback = 30
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)
    assert not result.empty, "Result should not be empty"
    assert "mkt_excess" in result.columns, (
        "Output should include beta estimate for mkt_excess"
    )


def test_estimate_rolling_betas_min_obs(sample_data: pd.DataFrame) -> None:
    lookback = 30
    min_obs = 10
    result = estimate_betas(
        sample_data, "ret_excess ~ mkt_excess", lookback, min_obs=min_obs
    )
    assert result.shape[0] > 0, "Result should have valid estimates"
    assert result["mkt_excess"].isna().sum() > 0, (
        "Some estimates should be NaN due to min_obs constraint"
    )


def test_estimate_betas_min_obs_non_positive_raises(
    sample_data: pd.DataFrame,
) -> None:
    """min_obs <= 0 raises a ValueError."""
    for bad in (0, -5):
        with pytest.raises(ValueError, match="min_obs must be a positive"):
            estimate_betas(
                sample_data, "ret_excess ~ mkt_excess", 30, min_obs=bad
            )


def test_estimate_betas_default_min_obs_is_80_percent(
    sample_data: pd.DataFrame,
) -> None:
    """min_obs defaults to 80% of lookback when not provided."""
    lookback = 30
    default = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)
    explicit = estimate_betas(
        sample_data,
        "ret_excess ~ mkt_excess",
        lookback,
        min_obs=int(lookback * 0.8),
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_estimate_betas_without_intercept_omits_intercept_column(
    sample_data: pd.DataFrame,
) -> None:
    """A '- 1' formula omits the Intercept column."""
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess - 1", 30)
    assert "Intercept" not in result.columns
    assert "mkt_excess" in result.columns


def test_estimate_betas_match_per_window_ols(
    sample_data: pd.DataFrame,
) -> None:
    """Estimated betas match a per-window OLS fit."""
    lookback = 30
    result = estimate_betas(sample_data, "ret_excess ~ mkt_excess", lookback)

    group = (
        sample_data[sample_data["permno"] == 1]
        .sort_values("date")
        .reset_index(drop=True)
    )
    i = 50
    window = group.iloc[i - lookback + 1 : i + 1]
    design = np.column_stack(
        [np.ones(len(window)), window["mkt_excess"].values]
    )
    expected = np.linalg.lstsq(design, window["ret_excess"].values, rcond=None)[
        0
    ]

    row = result[
        (result["permno"] == 1) & (result["date"] == group.loc[i, "date"])
    ]
    np.testing.assert_allclose(
        row[["Intercept", "mkt_excess"]].values[0], expected, rtol=1e-8
    )


def test_estimate_betas_custom_id_column(
    sample_data: pd.DataFrame,
) -> None:
    """A non-default stock identifier column is honored."""
    renamed = sample_data.rename(columns={"permno": "gvkey"})
    result = estimate_betas(
        renamed, "ret_excess ~ mkt_excess", 30, id_col="gvkey"
    )
    assert "gvkey" in result.columns
    assert "permno" not in result.columns


def test_estimate_betas_invalid_formula_raises(
    sample_data: pd.DataFrame,
) -> None:
    """A formula without '~' raises a ValueError."""
    with pytest.raises(ValueError, match="must contain '~'"):
        estimate_betas(sample_data, "ret_excess mkt_excess", 30)


def sample_data_fmb() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=12, freq="ME")
    permnos = range(50)
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(permnos)),
            "permno": np.repeat(permnos, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(permnos)),
            "beta": np.random.randn(len(dates) * len(permnos)),
            "bm": np.random.randn(len(dates) * len(permnos)),
            "log_mktcap": np.random.randn(len(dates) * len(permnos)),
        }
    )
    return data


def test_estimate_fama_macbeth_basic(sample_data: pd.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap"
    )
    assert not result.empty, "Result should not be empty"
    assert "risk_premium" in result.columns, (
        "Output should include risk premia estimates"
    )


def test_estimate_fama_macbeth_vcov(sample_data: pd.DataFrame) -> None:
    result = estimate_fama_macbeth(
        sample_data_fmb(), "ret_excess ~ beta + bm + log_mktcap", vcov="iid"
    )
    assert "t_statistic" in result.columns, (
        "Output should include t-statistics based on vcov choice"
    )


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
    data = sample_data_fmb().drop(columns="date")
    with pytest.raises(ValueError, match="must contain a date column"):
        estimate_fama_macbeth(data, "ret_excess ~ beta + bm + log_mktcap")


def test_estimate_fama_macbeth_n_equals_number_of_periods() -> None:
    """The reported 'n' equals the number of distinct periods."""
    data = sample_data_fmb()
    result = estimate_fama_macbeth(data, "ret_excess ~ beta + bm + log_mktcap")
    assert (result["n"] == data["date"].nunique()).all()


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


def _sample_data_fmb_parity() -> pd.DataFrame:
    """Deterministic panel; reference values produced by r-tidyfinance's
    estimate_fama_macbeth (vcov='newey-west') on the identical data."""
    rng = np.random.default_rng(987654)
    dates = pd.date_range("2000-01-31", periods=48, freq="ME")
    recs = []
    for d in dates:
        beta = rng.normal(1, 0.3, size=40)
        bm = rng.normal(0.5, 0.2, size=40)
        size = rng.normal(10, 1, size=40)
        eps = rng.normal(0, 0.05, size=40)
        ret = 0.002 + 0.0015 * beta - 0.003 * bm + 0.0008 * size + eps
        for p in range(40):
            recs.append((d, p, ret[p], beta[p], bm[p], size[p]))
    return pd.DataFrame(
        recs, columns=["date", "permno", "ret_excess", "beta", "bm", "size"]
    )


def test_estimate_fama_macbeth_newey_west_matches_r() -> None:
    """End-to-end Fama-MacBeth t-statistics match r-tidyfinance exactly.

    Reference (sandwich::NeweyWest default) rounded to 3 decimals:
    intercept -0.792, beta 2.301, bm 1.005, size 0.887.
    """
    out = estimate_fama_macbeth(
        _sample_data_fmb_parity(), "ret_excess ~ beta + bm + size"
    )
    t = out.set_index("factor")["t_statistic"].to_dict()
    rp = out.set_index("factor")["risk_premium"].to_dict()
    # Reference values are rounded to 3 decimals, so compare within half a
    # unit in the last place (abs=5e-4).
    assert t["Intercept"] == pytest.approx(-0.792, abs=5e-4)
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
    pd.testing.assert_frame_equal(legacy, explicit)


def sample_data_summary() -> pd.DataFrame:
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=100),
            "x": np.random.randn(100),
            "y": np.random.randint(0, 100, size=100),
            "z": np.random.randint(0, 100, size=100),
        }
    )
    return data


def test_create_summary_statistics_basic(sample_data) -> None:
    result = create_summary_statistics(sample_data_summary(), ["x", "y"])
    assert not result.empty, "Result should not be empty"
    assert "mean" in result.columns, "Output should include mean calculation"


def test_create_summary_statistics_by_group(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], by="group"
    )
    assert "group" in result.columns, "Output should include group column"
    assert "mean" in result.columns.get_level_values(1), (
        "Output should include mean calculation"
    )


def test_create_summary_statistics_detail(sample_data) -> None:
    result = create_summary_statistics(
        sample_data_summary(), ["x", "y"], detail=True
    )
    assert "1%" in result.columns, (
        "Detailed statistics should include 1st percentile"
    )
    assert "99%" in result.columns, (
        "Detailed statistics should include 99th percentile"
    )


def test_create_summary_statistics_accepts_boolean() -> None:
    """Test boolean columns are summarized as proportion of True."""
    df = pd.DataFrame(
        {
            "flag": [True, False, True, True],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = create_summary_statistics(df, ["flag", "x"])
    flag_row = result[result["variable"] == "flag"].iloc[0]
    assert abs(flag_row["mean"] - 0.75) < 1e-12, (
        "Boolean mean should equal the proportion of True"
    )


def test_create_summary_statistics_rejects_strings() -> None:
    """Test object dtype columns raise ValueError."""
    df = pd.DataFrame({"name": ["A", "B", "C"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not numeric or boolean"):
        create_summary_statistics(df, ["name", "x"])


def test_create_summary_statistics_handles_na() -> None:
    """NA values are dropped before statistics are computed."""
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = create_summary_statistics(df, ["x"])
    assert result["count"].iloc[0] == 3
    assert result["mean"].iloc[0] == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def sample_data_ls() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=10, freq="ME")
    portfolios = [1, 2]
    data = pd.DataFrame(
        {
            "date": np.tile(dates, len(portfolios)),
            "portfolio": np.repeat(portfolios, len(dates)),
            "ret_excess": np.random.randn(len(dates) * len(portfolios)),
        }
    )
    return data


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


def sample_data_breakpoints() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
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
