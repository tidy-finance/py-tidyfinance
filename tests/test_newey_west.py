"""Tests for the internal Newey-West HAC helpers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.core import (  # noqa: E402
    _ar1_ols_residuals,
    _newey_west_bandwidth,
    _newey_west_se,
)


def make_ar1_series(
    rho: float = 0.7, n: int = 1000, seed: int = 0
) -> np.ndarray:
    """Construct an AR(1) series x_t = rho * x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


# %% _ar1_ols_residuals


def test_ar1_returns_rho_close_to_true_value():
    """Test AR(1) coefficient is close to the data-generating value."""
    x = make_ar1_series(rho=0.7, n=1000)
    rho, _ = _ar1_ols_residuals(x)
    assert abs(rho - 0.7) < 0.05


def test_ar1_returns_residuals_one_shorter_than_input():
    """Test residual vector has length n-1."""
    x = make_ar1_series(n=500)
    _, resid = _ar1_ols_residuals(x)
    assert resid.shape == (499,)


def test_ar1_recovers_exact_coefficient_on_geometric_series():
    """Test exact recovery on a noise-free geometric series."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    rho, resid = _ar1_ols_residuals(x)
    assert abs(rho - 2.0) < 1e-10
    np.testing.assert_allclose(resid, [0, 0, 0, 0], atol=1e-10)


# %% _newey_west_bandwidth


def test_bandwidth_returns_positive_float():
    """Test bandwidth is a non-negative float."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal(200)
    bw = _newey_west_bandwidth(e, prewhite=1)
    assert isinstance(bw, float)
    assert bw >= 0


def test_bandwidth_scales_with_sample_size_on_average():
    """Test bandwidth grows with n on average across multiple seeds."""
    bws_small, bws_large = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        bws_small.append(
            _newey_west_bandwidth(rng.standard_normal(100), prewhite=1)
        )
        bws_large.append(
            _newey_west_bandwidth(rng.standard_normal(2000), prewhite=1)
        )
    assert np.mean(bws_large) > np.mean(bws_small)


def test_bandwidth_runs_for_both_prewhite_orders():
    """Test bandwidth is computed without error for prewhite in {0, 1}."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal(200)
    assert isinstance(_newey_west_bandwidth(e, prewhite=0), float)
    assert isinstance(_newey_west_bandwidth(e, prewhite=1), float)


# %% _newey_west_se


def test_newey_west_se_returns_nan_on_empty_series():
    """Test NaN return when the input is empty."""
    assert np.isnan(_newey_west_se(np.array([])))


def test_newey_west_se_returns_nan_on_single_obs():
    """Test NaN return when only one observation is supplied."""
    assert np.isnan(_newey_west_se(np.array([1.0])))


def test_newey_west_se_close_to_naive_se_for_iid_white_noise():
    """Test SE is within 10% of sd/sqrt(n) for iid white noise (no prewhitening)."""
    rng = np.random.default_rng(42)
    e = rng.standard_normal(5000)
    nw = _newey_west_se(e, prewhite=0)
    naive = e.std(ddof=1) / np.sqrt(len(e))
    assert abs(nw - naive) / naive < 0.10


def test_newey_west_se_accepts_explicit_lag():
    """Test SE is computed when 'lag' is supplied explicitly."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal(200)
    nw = _newey_west_se(e, lag=6, prewhite=0)
    assert isinstance(nw, float)
    assert nw > 0


def test_newey_west_se_drops_nan_values():
    """Test NaN values do not raise."""
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    result = _newey_west_se(x)
    assert isinstance(result, float)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
