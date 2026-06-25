"""Tests for estimate_model."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import estimate_model  # noqa: E402


def _ols_reference(df, dep, regressors):
    """Independent OLS fit (intercept + regressors) via numpy.

    Returns (params, tvalues, residuals) as dicts/arrays keyed by
    'Intercept' and the regressor names, computed straight from the
    normal equations so the test does not depend on the estimator under
    test or on statsmodels.
    """
    sub = df[[dep] + regressors].dropna()
    y = sub[dep].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(sub)), sub[regressors].to_numpy(float)])
    names = ["Intercept"] + regressors
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, k = x.shape
    sigma2 = (resid @ resid) / (n - k)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tvalues = beta / se
    return (
        dict(zip(names, beta)),
        dict(zip(names, tvalues)),
        resid,
    )


def make_test_data():
    """Construct the standard test panel (100 rows, 4 columns)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "ret_excess": rng.standard_normal(100),
            "mkt_excess": rng.standard_normal(100),
            "smb": rng.standard_normal(100),
            "hml": rng.standard_normal(100),
        }
    )


def make_test_data_with_na():
    """Like make_test_data but with the first 10 rows having NaNs."""
    df = make_test_data()
    df.loc[:4, "ret_excess"] = np.nan
    df.loc[5:9, "mkt_excess"] = np.nan
    return df


def make_small_data():
    """Tiny data frame for insufficient-obs tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "ret_excess": rng.standard_normal(3),
            "mkt_excess": rng.standard_normal(3),
            "smb": rng.standard_normal(3),
        }
    )


# %% validation


def test_invalid_output_values_raise_an_error():
    """Test invalid output values raise an error."""
    df = make_test_data()
    with pytest.raises(ValueError, match="output"):
        estimate_model(df, "ret_excess ~ mkt_excess", output="invalid")
    with pytest.raises(ValueError, match="output"):
        estimate_model(
            df,
            "ret_excess ~ mkt_excess",
            output=["coefficients", "bogus"],
        )


def test_missing_independent_variables_raise_an_error():
    """Test missing independent variables raise an error."""
    df = make_test_data()
    with pytest.raises(ValueError, match="missing"):
        estimate_model(df, "ret_excess ~ nonexistent_var")
    with pytest.raises(ValueError, match="nonexistent_var"):
        estimate_model(df, "ret_excess ~ mkt_excess + nonexistent_var")


def test_column_named_intercept_raises_an_error():
    """Test column named 'intercept' raises an error."""
    df = make_test_data()
    df["intercept"] = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="intercept"):
        estimate_model(df, "ret_excess ~ intercept")


# %% default output


def test_default_output_returns_a_dataframe_of_coefficients():
    """Test default output returns a DataFrame of coefficients."""
    result = estimate_model(make_test_data(), "ret_excess ~ mkt_excess")
    assert isinstance(result, pd.DataFrame)
    assert "intercept" in result.columns
    assert "mkt_excess" in result.columns
    assert result.shape[1] == 2


def test_coefficients_match_ols_reference():
    """Test coefficients match an independent numpy OLS fit."""
    df = make_test_data()
    result = estimate_model(df, "ret_excess ~ mkt_excess + smb + hml")
    params, _, _ = _ols_reference(
        df, "ret_excess", ["mkt_excess", "smb", "hml"]
    )
    assert abs(result["intercept"].iloc[0] - params["Intercept"]) < 1e-10
    assert abs(result["mkt_excess"].iloc[0] - params["mkt_excess"]) < 1e-10
    assert abs(result["smb"].iloc[0] - params["smb"]) < 1e-10
    assert abs(result["hml"].iloc[0] - params["hml"]) < 1e-10


def test_model_without_intercept_omits_intercept_column():
    """Test model without intercept omits intercept column."""
    result = estimate_model(make_test_data(), "ret_excess ~ mkt_excess - 1")
    assert "intercept" not in result.columns
    assert "mkt_excess" in result.columns


# %% tstats


def test_tstats_output_returns_t_statistics_as_dataframe():
    """Test tstats output returns t-statistics as DataFrame."""
    df = make_test_data()
    result = estimate_model(
        df, "ret_excess ~ mkt_excess + smb", output="tstats"
    )
    assert isinstance(result, pd.DataFrame)
    assert "intercept" in result.columns
    assert "mkt_excess" in result.columns
    assert "smb" in result.columns

    _, tvalues, _ = _ols_reference(df, "ret_excess", ["mkt_excess", "smb"])
    assert abs(result["mkt_excess"].iloc[0] - tvalues["mkt_excess"]) < 1e-10


# %% residuals


def test_residuals_output_returns_numeric_vector_of_correct_length():
    """Test residuals output returns numeric vector of correct length."""
    df = make_test_data()
    result = estimate_model(df, "ret_excess ~ mkt_excess", output="residuals")
    assert isinstance(result, np.ndarray)
    assert len(result) == len(df)


def test_residuals_match_ols_reference():
    """Test residuals match an independent numpy OLS fit."""
    df = make_test_data()
    result = estimate_model(
        df, "ret_excess ~ mkt_excess + smb", output="residuals"
    )
    _, _, resid = _ols_reference(df, "ret_excess", ["mkt_excess", "smb"])
    np.testing.assert_allclose(result, resid, atol=1e-10)


def test_residuals_are_na_where_data_has_missing_values():
    """Test residuals are NaN where data has missing values."""
    df = make_test_data_with_na()
    result = estimate_model(df, "ret_excess ~ mkt_excess", output="residuals")
    assert len(result) == len(df)
    assert np.isnan(result[:10]).all()
    assert not np.isnan(result[10:]).any()


# %% multiple outputs


def test_multiple_outputs_return_a_named_dict():
    """Test multiple outputs return a named dict."""
    result = estimate_model(
        make_test_data(),
        "ret_excess ~ mkt_excess",
        output=["coefficients", "tstats", "residuals"],
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"coefficients", "tstats", "residuals"}
    assert isinstance(result["coefficients"], pd.DataFrame)
    assert isinstance(result["tstats"], pd.DataFrame)
    assert isinstance(result["residuals"], np.ndarray)


def test_two_outputs_return_a_named_dict_with_two_elements():
    """Test two outputs return a named dict with two elements."""
    result = estimate_model(
        make_test_data(),
        "ret_excess ~ mkt_excess",
        output=["coefficients", "residuals"],
    )
    assert isinstance(result, dict)
    assert len(result) == 2
    assert set(result.keys()) == {"coefficients", "residuals"}


# %% insufficient observations


def test_insufficient_observations_return_na_coefficients():
    """Test insufficient observations return NaN coefficients."""
    result = estimate_model(
        make_small_data(), "ret_excess ~ mkt_excess", min_obs=100
    )
    assert isinstance(result, pd.DataFrame)
    assert result.isna().all().all()


def test_insufficient_observations_return_na_residuals():
    """Test insufficient observations return NaN residuals."""
    df = make_small_data()
    result = estimate_model(
        df, "ret_excess ~ mkt_excess", min_obs=100, output="residuals"
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == len(df)
    assert np.isnan(result).all()


def test_insufficient_observations_return_na_tstats():
    """Test insufficient observations return NaN tstats."""
    result = estimate_model(
        make_small_data(),
        "ret_excess ~ mkt_excess",
        min_obs=100,
        output="tstats",
    )
    assert isinstance(result, pd.DataFrame)
    assert result.isna().all().all()


# %% misc


def test_model_with_multiple_independent_variables_works():
    """Test model with multiple independent variables works."""
    result = estimate_model(
        make_test_data(), "ret_excess ~ mkt_excess + smb + hml"
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 4


def test_min_obs_1_works_with_minimal_data():
    """Test min_obs = 1 works with minimal data."""
    tiny = pd.DataFrame({"ret_excess": [1.0, 2.0], "mkt_excess": [1.0, 2.0]})
    result = estimate_model(tiny, "ret_excess ~ mkt_excess", min_obs=1)
    assert isinstance(result, pd.DataFrame)


def test_residuals_are_na_when_too_few_complete_cases_after_na_removal():
    """Test residuals are NaN when too few complete cases after NaN removal."""
    df = make_test_data()
    df.loc[:98, "ret_excess"] = np.nan
    result = estimate_model(
        df,
        "ret_excess ~ mkt_excess + smb",
        min_obs=5,
        output="residuals",
    )
    assert np.isnan(result).all()


def test_coefficients_are_na_when_all_rows_are_na():
    """Test coefficients are NaN when all rows are NaN."""
    df = make_test_data()
    df["ret_excess"] = np.nan
    result = estimate_model(df, "ret_excess ~ mkt_excess", min_obs=1)
    assert isinstance(result, pd.DataFrame)
    assert result.isna().all().all()


def test_tstats_are_na_when_all_model_rows_are_na():
    """Test tstats are NaN when all model rows are NaN."""
    df = make_test_data()
    df["ret_excess"] = np.nan
    result = estimate_model(
        df, "ret_excess ~ mkt_excess", min_obs=1, output="tstats"
    )
    assert isinstance(result, pd.DataFrame)
    assert result.isna().all().all()


# %% minimal data


_df = pd.DataFrame({"y": np.arange(1.0, 11.0), "x": np.arange(1.0, 11.0)})


def test_invalid_output_raises_an_error():
    """Test invalid output raises an error."""
    with pytest.raises(ValueError):
        estimate_model(_df, "y ~ x", output="foo")


def test_column_named_intercept_in_model_raises_an_error():
    """Test column named 'intercept' in model raises an error."""
    d = pd.DataFrame({"y": np.arange(1, 11), "intercept": np.arange(1, 11)})
    with pytest.raises(ValueError):
        estimate_model(d, "y ~ intercept")


def test_independent_variable_absent_from_data_raises_an_error():
    """Test independent variable absent from data raises an error."""
    with pytest.raises(ValueError):
        estimate_model(_df, "y ~ x + z")


def test_single_output_is_returned_directly_not_wrapped_in_a_list():
    """Test single output is returned directly, not wrapped in a list."""
    result = estimate_model(_df, "y ~ x")
    assert isinstance(result, pd.DataFrame)
    assert "intercept" in result.columns


def test_model_without_intercept_skips_column_rename():
    """Test model without intercept skips column rename."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        result = estimate_model(_df, "y ~ x - 1", output="tstats")
    assert isinstance(result, pd.DataFrame)
    assert "intercept" not in result.columns


def test_residuals_are_na_for_rows_with_missing_values_non_na_elsewhere():
    """Test residuals are NaN for rows with missing values, non-NaN elsewhere."""
    d = pd.DataFrame(
        {
            "y": list(np.arange(1.0, 10.0)) + [np.nan],
            "x": np.arange(1.0, 11.0),
        }
    )
    result = estimate_model(d, "y ~ x", output="residuals")
    assert np.isnan(result[9])
    assert not np.isnan(result[:9]).any()


def test_multiple_outputs_returns_a_named_dict_minimal():
    """Test multiple outputs returns a named dict (minimal data)."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        result = estimate_model(
            _df, "y ~ x", output=["coefficients", "tstats", "residuals"]
        )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"coefficients", "tstats", "residuals"}


def test_insufficient_obs_returns_na_for_all_three_outputs():
    """Test insufficient observations returns NaN for all three outputs."""
    d = pd.DataFrame({"y": np.arange(1.0, 6.0), "x": np.arange(1.0, 6.0)})
    result = estimate_model(
        d, "y ~ x", min_obs=10, output=["coefficients", "tstats", "residuals"]
    )
    assert result["coefficients"].isna().all().all()
    assert result["tstats"].isna().all().all()
    np.testing.assert_array_equal(result["residuals"], np.full(5, np.nan))


def test_insufficient_obs_with_no_independent_variables_returns_na_scalar():
    """Test insufficient obs with no independent variables returns NaN scalar."""
    d = pd.DataFrame({"y": np.arange(1.0, 6.0)})
    result = estimate_model(d, "y ~ 1", min_obs=10, output="coefficients")
    assert pd.isna(result)


if __name__ == "__main__":
    pytest.main([__file__])
