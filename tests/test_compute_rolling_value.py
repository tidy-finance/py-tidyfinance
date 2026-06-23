"""Tests for compute_rolling_value."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import compute_rolling_value  # noqa: E402


def make_df(n_months=24, seed=42):
    """Construct a monthly panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    return pd.DataFrame(
        {"date": dates, "value": rng.standard_normal(n_months)}
    )


# %% validation tests


def test_errors_when_data_has_no_date_column():
    """Test errors when data has no date column."""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="date"):
        compute_rolling_value(df, lambda x: x["value"].mean())


def test_errors_when_date_column_is_not_date_class():
    """Test errors when date column is not datetime dtype."""
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="datetime"):
        compute_rolling_value(df, lambda x: x["value"].mean())


def test_errors_when_period_is_not_a_single_string():
    """Test errors when period is not a single string."""
    df = make_df()
    with pytest.raises(ValueError, match="period"):
        compute_rolling_value(df, lambda x: x["value"].mean(), period=12)


# %% basic behavior tests


def test_returns_numeric_vector_with_length_equal_to_nrow_of_data():
    """Test returns a numeric vector with length equal to nrow(data)."""
    df = make_df()
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=4
    )
    assert isinstance(out, np.ndarray)
    assert len(out) == len(df)


def test_min_obs_defaults_to_periods_early_windows_return_nan():
    """Test min_obs defaults to periods; early windows return NaN."""
    df = make_df()
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=4
    )
    # First 3 windows have fewer than 4 obs -> NaN
    assert np.isnan(out[:3]).all()
    assert not np.isnan(out[3:]).any()


def test_custom_min_obs_produces_fewer_leading_nas():
    """Test custom min_obs produces fewer leading NaNs."""
    df = make_df()
    out = compute_rolling_value(
        df,
        lambda x: x["value"].mean(),
        period="month",
        periods=4,
        min_obs=2,
    )
    # Only the first window (1 obs) is NaN
    assert np.isnan(out[0])
    assert not np.isnan(out[1:]).any()


def test_min_obs_larger_than_periods_makes_more_windows_nan():
    """Test min_obs larger than periods makes more windows NaN."""
    df = make_df()
    out = compute_rolling_value(
        df,
        lambda x: x["value"].mean(),
        period="month",
        periods=4,
        min_obs=10,
    )
    # No window has 10 obs -> all NaN
    assert np.isnan(out).all()


# %% NaN handling


def test_rows_with_na_values_are_dropped_before_applying_f():
    """Test rows with NaN values are dropped before applying f."""
    df = make_df(n_months=6)
    df.loc[2, "value"] = np.nan
    out = compute_rolling_value(
        df, lambda x: len(x), period="month", periods=4, min_obs=1
    )
    # At i=2, the window contains 3 rows (0, 1, 2), but row 2 has NaN
    # -> dropna gives 2 rows -> f returns 2
    assert out[2] == 2


def test_window_returns_nan_when_complete_cases_less_than_min_obs_due_to_nas():
    """Test window returns NaN when complete cases < min_obs due to NaNs."""
    df = make_df(n_months=6)
    df.loc[:2, "value"] = np.nan  # First 3 values are NaN
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=4, min_obs=3
    )
    # At i=2: window has 3 rows but all NaN -> dropna gives 0 -> NaN
    assert np.isnan(out[2])


# %% correctness tests


def test_rolling_mean_with_periods_1_equals_original_values():
    """Test rolling mean with periods = 1 equals the original values."""
    df = make_df(n_months=6)
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=1
    )
    np.testing.assert_allclose(out, df["value"].values)


def test_rolling_mean_with_periods_3_computes_correct_values():
    """Test rolling mean with periods = 3 computes correct values."""
    df = make_df(n_months=6)
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=3
    )
    expected_at_5 = df["value"].iloc[3:6].mean()
    assert abs(out[5] - expected_at_5) < 1e-12


def test_rolling_sum_works_correctly():
    """Test rolling sum works correctly."""
    df = make_df(n_months=6)
    out = compute_rolling_value(
        df, lambda x: x["value"].sum(), period="month", periods=3
    )
    expected_at_5 = df["value"].iloc[3:6].sum()
    assert abs(out[5] - expected_at_5) < 1e-12


def test_rolling_sd_works_correctly():
    """Test rolling sd works correctly."""
    df = make_df(n_months=6)
    out = compute_rolling_value(
        df, lambda x: x["value"].std(), period="month", periods=4
    )
    expected_at_5 = df["value"].iloc[2:6].std()
    assert abs(out[5] - expected_at_5) < 1e-12


# %% callable / lambda


def test_accepts_lambda_function():
    """Test accepts a lambda function for f."""
    df = make_df(n_months=6)
    out = compute_rolling_value(
        df, lambda x: x["value"].max(), period="month", periods=3
    )
    assert not np.isnan(out[-1])


def test_accepts_a_regular_function():
    """Test accepts a regular function for f."""
    df = make_df(n_months=6)

    def my_mean(x):
        return x["value"].mean()

    out = compute_rolling_value(df, my_mean, period="month", periods=3)
    assert not np.isnan(out[-1])


# %% period variations


def test_works_with_period_quarter():
    """Test works with period = 'quarter'."""
    df = make_df(n_months=24)
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="quarter", periods=4
    )
    assert isinstance(out, np.ndarray)
    assert len(out) == len(df)


def test_works_with_period_year():
    """Test works with period = 'year'."""
    df = make_df(n_months=36)
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="year", periods=2
    )
    assert isinstance(out, np.ndarray)
    assert len(out) == len(df)


# %% multi-column / edge cases


def test_works_with_multiple_columns():
    """Test works with multiple columns (e.g., regression residuals)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "y": rng.standard_normal(12),
            "x": rng.standard_normal(12),
        }
    )
    out = compute_rolling_value(
        df,
        lambda x: (x["y"] - x["x"]).mean(),
        period="month",
        periods=3,
    )
    assert isinstance(out, np.ndarray)


def test_single_row_data_frame_works():
    """Test single-row data frame works."""
    df = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=1, freq="MS"), "value": [1.0]}
    )
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=1
    )
    assert out[0] == 1.0


def test_single_row_data_frame_returns_nan_when_min_obs_gt_1():
    """Test single-row data frame returns NaN when min_obs > 1."""
    df = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=1, freq="MS"), "value": [1.0]}
    )
    out = compute_rolling_value(
        df,
        lambda x: x["value"].mean(),
        period="month",
        periods=1,
        min_obs=2,
    )
    assert np.isnan(out[0])


def test_all_na_value_column_returns_all_nans():
    """Test all-NaN value column returns all NaNs."""
    df = make_df(n_months=6)
    df["value"] = np.nan
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=3
    )
    assert np.isnan(out).all()


def test_periods_equal_to_nrow_uses_full_history_for_last_row():
    """Test periods = nrow(data) uses full history for last row."""
    df = make_df(n_months=12)
    out = compute_rolling_value(
        df, lambda x: x["value"].mean(), period="month", periods=12
    )
    expected = df["value"].mean()
    assert abs(out[-1] - expected) < 1e-12


# %% custom data_options


def test_data_options_with_non_default_date_column_works():
    """Test data_options with non-default date column works."""
    df = pd.DataFrame(
        {
            "my_date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "value": np.arange(6, dtype=float),
        }
    )
    out = compute_rolling_value(
        df,
        lambda x: x["value"].mean(),
        period="month",
        periods=3,
        data_options={"date": "my_date"},
    )
    assert isinstance(out, np.ndarray)
    assert len(out) == 6


def test_non_default_date_column_produces_same_results_as_default():
    """Test non-default date column produces same results as default."""
    df_default = make_df(n_months=6)
    df_custom = df_default.rename(columns={"date": "my_date"})
    out_default = compute_rolling_value(
        df_default, lambda x: x["value"].mean(), period="month", periods=3
    )
    out_custom = compute_rolling_value(
        df_custom,
        lambda x: x["value"].mean(),
        period="month",
        periods=3,
        data_options={"date": "my_date"},
    )
    np.testing.assert_allclose(
        out_default[~np.isnan(out_default)],
        out_custom[~np.isnan(out_custom)],
    )


def test_errors_when_mapped_date_column_is_absent_from_data():
    """Test errors when mapped date column is absent from data."""
    df = make_df(n_months=6)
    with pytest.raises(ValueError, match="my_date"):
        compute_rolling_value(
            df,
            lambda x: x["value"].mean(),
            data_options={"date": "my_date"},
        )


def test_errors_when_mapped_date_column_is_not_date():
    """Test errors when mapped date column is not datetime dtype."""
    df = pd.DataFrame(
        {"my_date": ["2020-01-01", "2020-02-01"], "value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="datetime"):
        compute_rolling_value(
            df,
            lambda x: x["value"].mean(),
            data_options={"date": "my_date"},
        )


def test_errors_when_data_options_date_is_none():
    """Test errors when data_options['date'] is None."""
    df = make_df()
    with pytest.raises(ValueError, match="date"):
        compute_rolling_value(
            df,
            lambda x: x["value"].mean(),
            data_options={"date": None},
        )


def test_errors_when_data_options_date_is_not_character():
    """Test errors when data_options['date'] is not a string."""
    df = make_df()
    with pytest.raises(ValueError, match="date"):
        compute_rolling_value(
            df,
            lambda x: x["value"].mean(),
            data_options={"date": 1},
        )


def test_errors_when_data_options_date_has_length_gt_1():
    """Test errors when data_options['date'] is a list."""
    df = make_df()
    with pytest.raises(ValueError, match="date"):
        compute_rolling_value(
            df,
            lambda x: x["value"].mean(),
            data_options={"date": ["a", "b"]},
        )


if __name__ == "__main__":
    pytest.main([__file__])
