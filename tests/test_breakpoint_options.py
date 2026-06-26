"""Tests for breakpoint_options."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import breakpoint_options  # noqa: E402


def test_returns_correct_structure_with_all_arguments_including_kwargs():
    """Test returns correct structure with all arguments including kwargs."""
    result = breakpoint_options(
        n_portfolios=5,
        percentiles=[0.2, 0.8],
        breakpoints_exchanges="NYSE",
        smooth_bunching=True,
        breakpoints_min_size_threshold=0.1,
        custom_arg="test",
    )

    assert isinstance(result, dict)
    assert result["n_portfolios"] == 5
    assert result["percentiles"] == [0.2, 0.8]
    assert result["breakpoints_exchanges"] == "NYSE"
    assert result["smooth_bunching"] is True
    assert result["breakpoints_min_size_threshold"] == 0.1
    assert result["custom_arg"] == "test"


def test_returns_correct_structure_with_default_none_arguments():
    """Test returns correct structure with default (None) arguments."""
    result = breakpoint_options()

    assert isinstance(result, dict)
    assert result["n_portfolios"] is None
    assert result["percentiles"] is None
    assert result["breakpoints_exchanges"] is None
    assert result["smooth_bunching"] is False
    assert result["breakpoints_min_size_threshold"] is None


def test_n_portfolios_errors_on_non_numeric_non_positive_non_integer():
    """Test n_portfolios errors on non-numeric, non-positive, non-integer."""
    with pytest.raises(ValueError, match="n_portfolios"):
        breakpoint_options(n_portfolios="a")
    with pytest.raises(ValueError, match="n_portfolios"):
        breakpoint_options(n_portfolios=-1)
    with pytest.raises(ValueError, match="n_portfolios"):
        breakpoint_options(n_portfolios=1.5)


def test_percentiles_errors_on_non_numeric_and_out_of_range_values():
    """Test percentiles errors on non-numeric and out-of-range values."""
    with pytest.raises(ValueError, match="percentiles"):
        breakpoint_options(percentiles="a")
    with pytest.raises(ValueError, match="percentiles"):
        breakpoint_options(percentiles=[0.5, 1.5])


def test_breakpoints_exchanges_errors_on_non_character_and_empty_vector():
    """Test breakpoints_exchanges errors on non-string and empty string."""
    with pytest.raises(ValueError, match="breakpoints_exchanges"):
        breakpoint_options(breakpoints_exchanges=123)
    with pytest.raises(ValueError, match="breakpoints_exchanges"):
        breakpoint_options(breakpoints_exchanges="")


def test_smooth_bunching_errors_on_non_logical_na_and_length_gt_1():
    """Test smooth_bunching errors on non-bool, None, and lists."""
    with pytest.raises(ValueError, match="smooth_bunching"):
        breakpoint_options(smooth_bunching="yes")
    with pytest.raises(ValueError, match="smooth_bunching"):
        breakpoint_options(smooth_bunching=None)
    with pytest.raises(ValueError, match="smooth_bunching"):
        breakpoint_options(smooth_bunching=[True, False])


def test_breakpoints_min_size_threshold_errors_on_invalid_values():
    """Test breakpoints_min_size_threshold errors on invalid values."""
    with pytest.raises(ValueError, match="breakpoints_min_size_threshold"):
        breakpoint_options(breakpoints_min_size_threshold=[0.1, 0.2])
    with pytest.raises(ValueError, match="breakpoints_min_size_threshold"):
        breakpoint_options(breakpoints_min_size_threshold="a")
    with pytest.raises(ValueError, match="breakpoints_min_size_threshold"):
        breakpoint_options(breakpoints_min_size_threshold=0)
    with pytest.raises(ValueError, match="breakpoints_min_size_threshold"):
        breakpoint_options(breakpoints_min_size_threshold=1)


def test_n_portfolios_accepts_whole_valued_floats():
    """Test n_portfolios accepts whole-valued floats (R parity)."""
    result = breakpoint_options(n_portfolios=5.0)
    assert result["n_portfolios"] == 5.0


def test_n_portfolios_rejects_fractional_floats():
    """Test n_portfolios still rejects non-whole floats."""
    with pytest.raises(ValueError, match="n_portfolios"):
        breakpoint_options(n_portfolios=5.5)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
