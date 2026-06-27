"""Tests for data_options."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import data_options  # noqa: E402


def test_data_options_returns_dict():
    """Test data_options returns a dict."""
    opts = data_options()
    assert isinstance(opts, dict)


def test_data_options_has_correct_default_values():
    """Test data_options has correct default values."""
    opts = data_options()
    assert opts["id"] == "permno"
    assert opts["date"] == "date"
    assert opts["exchange"] == "exchange"
    assert opts["mktcap_lag"] == "mktcap_lag"
    assert opts["ret_excess"] == "ret_excess"
    assert opts["portfolio"] == "portfolio"
    assert opts["siccd"] == "siccd"
    assert opts["price"] == "prc_adj"
    assert opts["listing_age"] == "listing_age"
    assert opts["be"] == "be"
    assert opts["earnings"] == "ib"


def test_data_options_accepts_custom_column_names_for_core_columns():
    """Test data_options accepts custom column names for core columns."""
    opts = data_options(
        id="firm_id",
        date="month",
        exchange="exch",
        mktcap_lag="lag_mcap",
        ret_excess="excess_ret",
        portfolio="port",
    )
    assert opts["id"] == "firm_id"
    assert opts["date"] == "month"
    assert opts["exchange"] == "exch"
    assert opts["mktcap_lag"] == "lag_mcap"
    assert opts["ret_excess"] == "excess_ret"
    assert opts["portfolio"] == "port"


def test_data_options_accepts_custom_column_names_for_filter_columns():
    """Test data_options accepts custom names for filter-related columns."""
    opts = data_options(
        siccd="sic_code",
        price="adj_price",
        listing_age="age_months",
        be="book_equity",
        earnings="ni",
    )
    assert opts["siccd"] == "sic_code"
    assert opts["price"] == "adj_price"
    assert opts["listing_age"] == "age_months"
    assert opts["be"] == "book_equity"
    assert opts["earnings"] == "ni"


def test_data_options_stores_extra_arguments_via_kwargs():
    """Test data_options stores extra arguments via kwargs."""
    opts = data_options(custom_col="my_column")
    assert opts["custom_col"] == "my_column"


def test_data_options_errors_for_non_string_id():
    """Test data_options errors for non-string id."""
    with pytest.raises(ValueError, match="id"):
        data_options(id=123)
    with pytest.raises(ValueError, match="id"):
        data_options(id=["a", "b"])


def test_data_options_errors_for_non_string_date():
    """Test data_options errors for non-string date."""
    with pytest.raises(ValueError, match="date"):
        data_options(date=1)
    with pytest.raises(ValueError, match="date"):
        data_options(date=["date1", "date2"])


def test_data_options_errors_for_non_string_exchange():
    """Test data_options errors for non-string exchange."""
    with pytest.raises(ValueError, match="exchange"):
        data_options(exchange=True)
    with pytest.raises(ValueError, match="exchange"):
        data_options(exchange=["NYSE", "NASDAQ"])


def test_data_options_errors_for_non_string_mktcap_lag():
    """Test data_options errors for non-string mktcap_lag."""
    with pytest.raises(ValueError, match="mktcap_lag"):
        data_options(mktcap_lag=0)


def test_data_options_errors_for_non_string_ret_excess():
    """Test data_options errors for non-string ret_excess."""
    with pytest.raises(ValueError, match="ret_excess"):
        data_options(ret_excess=None)


def test_data_options_errors_for_non_string_portfolio():
    """Test data_options errors for non-string portfolio."""
    with pytest.raises(ValueError, match="portfolio"):
        data_options(portfolio=5)


def test_data_options_errors_for_non_string_siccd():
    """Test data_options errors for non-string siccd."""
    with pytest.raises(ValueError, match="siccd"):
        data_options(siccd=6000)


def test_data_options_errors_for_non_string_price():
    """Test data_options errors for non-string price."""
    with pytest.raises(ValueError, match="price"):
        data_options(price=True)


def test_data_options_errors_for_non_string_listing_age():
    """Test data_options errors for non-string listing_age."""
    with pytest.raises(ValueError, match="listing_age"):
        data_options(listing_age=12)


def test_data_options_errors_for_non_string_be():
    """Test data_options errors for non-string be."""
    with pytest.raises(ValueError, match="be"):
        data_options(be=1.0)


def test_data_options_errors_for_non_string_earnings():
    """Test data_options errors for non-string earnings."""
    with pytest.raises(ValueError, match="earnings"):
        data_options(earnings=1.0)
    with pytest.raises(ValueError, match="earnings"):
        data_options(earnings=["ib", "ni"])


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
