"""Tests for _quantile, the package's sample quantile."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance._internal import _quantile  # noqa: E402


def test_matches_numpy_quantile_up_to_rounding():
    """Test matches numpy's linear quantile up to rounding."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(101)
    probs = np.linspace(0, 1, 11)
    np.testing.assert_allclose(
        _quantile(x, probs), np.quantile(x, probs), rtol=1e-12
    )


def test_returns_observations_at_whole_number_positions():
    """Test returns the observations themselves at whole positions."""
    x = np.array([40.0, 10.0, 30.0, 20.0, 50.0])
    np.testing.assert_array_equal(
        _quantile(x, [0, 0.25, 0.5, 0.75, 1]),
        [10.0, 20.0, 30.0, 40.0, 50.0],
    )


def test_uses_r_interpolation_arithmetic():
    """Test interpolates as (1 - h) * a + h * b."""
    x = np.array([0.1, 0.7])
    p = 0.3
    h = (1 + p) - 1  # position 1 + (n - 1) * p minus its floor, n = 2
    assert _quantile(x, p) == (1 - h) * 0.1 + h * 0.7


def test_ignores_nan_and_returns_scalars_for_scalar_probs():
    """Test ignores NaN and returns a float for a scalar probability."""
    result = _quantile([1.0, np.nan, 3.0], 0.5)
    assert isinstance(result, float)
    assert result == 2.0


def test_returns_nan_without_values():
    """Test returns NaN without non-missing values."""
    assert np.isnan(_quantile([np.nan], 0.5))
    assert np.isnan(_quantile([], [0.2, 0.8])).all()


if __name__ == "__main__":
    pytest.main([__file__])
