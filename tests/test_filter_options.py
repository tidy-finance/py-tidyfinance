"""Tests for filter_options."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import filter_options  # noqa: E402


def test_filter_options_returns_dict():
    """Test filter_options returns a dict."""
    opts = filter_options()
    assert isinstance(opts, dict)


def test_filter_options_has_correct_defaults():
    """Test filter_options has correct defaults."""
    opts = filter_options()
    assert opts["exclude_financials"] is False
    assert opts["exclude_utilities"] is False
    assert opts["min_stock_price"] is None
    assert opts["min_size_quantile"] is None
    assert opts["min_listing_age"] is None
    assert opts["exclude_negative_book_equity"] is False
    assert opts["exclude_negative_earnings"] is False


def test_filter_options_accepts_non_default_logical_flags():
    """Test filter_options accepts non-default logical flags."""
    opts = filter_options(
        exclude_financials=True,
        exclude_utilities=True,
        exclude_negative_book_equity=True,
        exclude_negative_earnings=True,
    )
    assert opts["exclude_financials"] is True
    assert opts["exclude_utilities"] is True
    assert opts["exclude_negative_book_equity"] is True
    assert opts["exclude_negative_earnings"] is True


def test_filter_options_accepts_valid_numeric_thresholds():
    """Test filter_options accepts valid numeric thresholds."""
    opts = filter_options(
        min_stock_price=5,
        min_size_quantile=0.2,
        min_listing_age=12,
    )
    assert opts["min_stock_price"] == 5
    assert opts["min_size_quantile"] == 0.2
    assert opts["min_listing_age"] == 12


def test_filter_options_stores_extra_arguments_via_kwargs():
    """Test filter_options stores extra arguments via kwargs."""
    opts = filter_options(custom_filter="my_value")
    assert opts["custom_filter"] == "my_value"


def test_filter_options_errors_for_non_logical_exclude_financials():
    """Test filter_options errors for non-boolean exclude_financials."""
    with pytest.raises(ValueError, match="exclude_financials"):
        filter_options(exclude_financials="yes")
    with pytest.raises(ValueError, match="exclude_financials"):
        filter_options(exclude_financials=1)
    with pytest.raises(ValueError, match="exclude_financials"):
        filter_options(exclude_financials=None)


def test_filter_options_errors_for_non_logical_exclude_utilities():
    """Test filter_options errors for non-boolean exclude_utilities."""
    with pytest.raises(ValueError, match="exclude_utilities"):
        filter_options(exclude_utilities="yes")
    with pytest.raises(ValueError, match="exclude_utilities"):
        filter_options(exclude_utilities=None)


def test_filter_options_errors_for_non_positive_min_stock_price():
    """Test filter_options errors for non-positive min_stock_price."""
    with pytest.raises(ValueError, match="min_stock_price"):
        filter_options(min_stock_price=0)
    with pytest.raises(ValueError, match="min_stock_price"):
        filter_options(min_stock_price=-1)
    with pytest.raises(ValueError, match="min_stock_price"):
        filter_options(min_stock_price="5")
    with pytest.raises(ValueError, match="min_stock_price"):
        filter_options(min_stock_price=[1, 2])


def test_filter_options_errors_for_out_of_range_min_size_quantile():
    """Test filter_options errors for out-of-range min_size_quantile."""
    with pytest.raises(ValueError, match="min_size_quantile"):
        filter_options(min_size_quantile=0)
    with pytest.raises(ValueError, match="min_size_quantile"):
        filter_options(min_size_quantile=1)
    with pytest.raises(ValueError, match="min_size_quantile"):
        filter_options(min_size_quantile=-0.1)
    with pytest.raises(ValueError, match="min_size_quantile"):
        filter_options(min_size_quantile=1.1)


def test_filter_options_errors_for_invalid_min_listing_age():
    """Test filter_options errors for invalid min_listing_age."""
    with pytest.raises(ValueError, match="min_listing_age"):
        filter_options(min_listing_age=-1)
    with pytest.raises(ValueError, match="min_listing_age"):
        filter_options(min_listing_age="12")
    with pytest.raises(ValueError, match="min_listing_age"):
        filter_options(min_listing_age=[6, 12])


def test_filter_options_accepts_zero_as_valid_min_listing_age():
    """Test filter_options accepts zero as valid min_listing_age."""
    opts = filter_options(min_listing_age=0)
    assert opts["min_listing_age"] == 0


def test_filter_options_errors_for_non_logical_exclude_negative_book_equity():
    """Test filter_options errors for non-boolean exclude_negative_book_equity."""
    with pytest.raises(ValueError, match="exclude_negative_book_equity"):
        filter_options(exclude_negative_book_equity="TRUE")
    with pytest.raises(ValueError, match="exclude_negative_book_equity"):
        filter_options(exclude_negative_book_equity=None)
    with pytest.raises(ValueError, match="exclude_negative_book_equity"):
        filter_options(exclude_negative_book_equity=1)


def test_filter_options_errors_for_non_logical_exclude_negative_earnings():
    """Test filter_options errors for non-boolean exclude_negative_earnings."""
    with pytest.raises(ValueError, match="exclude_negative_earnings"):
        filter_options(exclude_negative_earnings="TRUE")
    with pytest.raises(ValueError, match="exclude_negative_earnings"):
        filter_options(exclude_negative_earnings=None)
    with pytest.raises(ValueError, match="exclude_negative_earnings"):
        filter_options(exclude_negative_earnings=1)


if __name__ == "__main__":
    pytest.main([__file__])
