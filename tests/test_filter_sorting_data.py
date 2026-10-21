"""Tests for filter_sorting_data."""

import datetime as dt
import os
import sys

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import filter_options, filter_sorting_data  # noqa: E402


def make_data():
    """Construct a stock-level panel for tests."""
    return pl.DataFrame(
        {
            "permno": list(range(1, 11)),
            "date": [dt.date(2020, 1, 1)] * 5 + [dt.date(2020, 2, 1)] * 5,
            "exchange": ["NYSE", "NYSE", "NASDAQ", "NYSE", "NASDAQ"] * 2,
            "siccd": [6500, 2000, 4950, 3000, 6100] * 2,
            "prc_lag": [10.0, 0.5, 15.0, 20.0, 5.0] * 2,
            "mktcap_lag": [100.0, 200.0, 50.0, 500.0, 300.0] * 2,
            "listing_age": [24, 6, 60, 12, 36] * 2,
            "be": [10.0, -5.0, 50.0, 100.0, 0.0] * 2,
            "ib": [5.0, -2.0, 20.0, -10.0, 30.0] * 2,
        }
    )


def set_null(data, column, index):
    """Return 'data' with a single cell of 'column' set to null."""
    values = data.get_column(column).to_list()
    values[index] = None
    return data.with_columns(
        pl.Series(column, values, dtype=data.schema[column])
    )


# %% validation


def test_quiet_must_be_a_single_non_na_logical():
    """Test quiet must be a single boolean."""
    with pytest.raises(ValueError, match="quiet"):
        filter_sorting_data(make_data(), quiet="yes")


def test_null_filter_options_and_data_options_leave_data_unchanged():
    """Test None filter_options and data_options leave data unchanged."""
    data = make_data()
    out = filter_sorting_data(data)
    assert_frame_equal(out, data)


# %% SIC filters


def test_sic_filters_abort_when_siccd_column_is_absent():
    """Test SIC filters abort when siccd column is absent."""
    data = make_data().drop("siccd")
    with pytest.raises(ValueError, match="siccd"):
        filter_sorting_data(
            data, filter_options=filter_options(exclude_financials=True)
        )


def test_exclude_financials_removes_sic_6000_6799_keeps_na_messages():
    """Test exclude_financials removes SIC 6000-6799, keeps null, warns."""
    data = set_null(make_data(), "siccd", 0)  # null row should be kept
    with pytest.warns(UserWarning, match="exclude_financials"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_financials=True),
        )
    # null row kept; SIC 6500 and 6100 dropped
    assert out["siccd"].is_null().any()
    assert not ((out["siccd"] >= 6000) & (out["siccd"] <= 6799)).any()


def test_exclude_utilities_removes_sic_4900_4999_keeps_na_messages():
    """Test exclude_utilities removes SIC 4900-4999, keeps null, warns."""
    data = set_null(make_data(), "siccd", 0)
    with pytest.warns(UserWarning, match="exclude_utilities"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_utilities=True),
        )
    assert out["siccd"].is_null().any()
    assert not ((out["siccd"] >= 4900) & (out["siccd"] <= 4999)).any()


# %% min_stock_price


def test_min_stock_price_aborts_when_price_column_is_absent():
    """Test min_stock_price aborts when price column is absent."""
    data = make_data().drop("prc_lag")
    with pytest.raises(ValueError, match="prc_lag"):
        filter_sorting_data(
            data, filter_options=filter_options(min_stock_price=1)
        )


def test_min_stock_price_removes_below_threshold_and_na_rows():
    """Test min_stock_price removes below-threshold and null rows, warns."""
    data = set_null(make_data(), "prc_lag", 0)
    with pytest.warns(UserWarning, match="min_stock_price"):
        out = filter_sorting_data(
            data, filter_options=filter_options(min_stock_price=1)
        )
    assert (out["prc_lag"] >= 1).all()
    assert out["prc_lag"].is_not_null().all()


# %% min_size_quantile


def test_min_size_quantile_aborts_when_mktcap_lag_column_is_absent():
    """Test min_size_quantile aborts when mktcap_lag column is absent."""
    data = make_data().drop("mktcap_lag")
    with pytest.raises(ValueError, match="mktcap_lag"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_aborts_when_date_column_is_absent():
    """Test min_size_quantile aborts when date column is absent."""
    data = make_data().drop("date")
    with pytest.raises(ValueError, match="date"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_aborts_when_exchange_column_is_absent():
    """Test min_size_quantile aborts when exchange column is absent."""
    data = make_data().drop("exchange")
    with pytest.raises(ValueError, match="exchange"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_warns_on_dates_with_no_nyse_observations():
    """Test min_size_quantile warns on dates with no NYSE observations."""
    import warnings as _w

    data = make_data().filter(
        ~(
            (pl.col("date") == dt.date(2020, 1, 1))
            & (pl.col("exchange") == "NYSE")
        )
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )
    messages = [str(w.message) for w in caught]
    assert any("no NYSE stocks" in m for m in messages)


def test_min_size_quantile_removes_below_nyse_quantile_stocks_messages():
    """Test min_size_quantile removes below-NYSE-quantile stocks, warns."""
    data = make_data()
    with pytest.warns(UserWarning, match="min_size_quantile"):
        out = filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.5)
        )
    assert len(out) < len(data)


def test_min_size_quantile_treats_nan_as_missing():
    """Float NaN in mktcap_lag is treated as missing: it neither
    shifts the NYSE quantile cutoff nor passes the size filter."""
    data = pl.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "date": [dt.date(2020, 1, 1)] * 4,
            "exchange": ["NYSE"] * 4,
            "mktcap_lag": [1.0, 2.0, float("nan"), 4.0],
        }
    )
    out = filter_sorting_data(
        data,
        filter_options=filter_options(min_size_quantile=0.5),
        quiet=True,
    )
    # Cutoff from [1, 2, 4] (median 2.0); the NaN row is dropped.
    assert out["permno"].to_list() == [2, 4]


def test_min_size_quantile_emits_no_message_when_no_rows_are_removed():
    """Test min_size_quantile emits no message when no rows are removed."""
    # All rows are at or above the lowest NYSE size cutoff -> no removal
    data = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)] * 4,
            "exchange": ["NYSE", "NYSE", "NYSE", "NYSE"],
            "mktcap_lag": [100, 100, 100, 100],
        }
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        # Should not raise: no removal -> no warning
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.1)
        )


# %% min_listing_age


def test_min_listing_age_aborts_when_listing_age_column_is_absent():
    """Test min_listing_age aborts when listing_age column is absent."""
    data = make_data().drop("listing_age")
    with pytest.raises(ValueError, match="listing_age"):
        filter_sorting_data(
            data, filter_options=filter_options(min_listing_age=12)
        )


def test_min_listing_age_removes_young_and_na_stocks_messages():
    """Test min_listing_age removes young and null stocks, warns."""
    data = set_null(make_data(), "listing_age", 0)
    with pytest.warns(UserWarning, match="min_listing_age"):
        out = filter_sorting_data(
            data, filter_options=filter_options(min_listing_age=12)
        )
    assert (out["listing_age"] >= 12).all()
    assert out["listing_age"].is_not_null().all()


# %% exclude_negative_book_equity


def test_exclude_negative_book_equity_aborts_when_be_column_is_absent():
    """Test exclude_negative_book_equity aborts when be column is absent."""
    data = make_data().drop("be")
    with pytest.raises(ValueError, match="be"):
        filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_book_equity=True),
        )


def test_exclude_negative_book_equity_removes_nonpositive_and_na_messages():
    """Test exclude_negative_book_equity drops non-positive and null, warns."""
    data = set_null(make_data(), "be", 0)
    with pytest.warns(UserWarning, match="exclude_negative_book_equity"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_book_equity=True),
        )
    assert (out["be"] > 0).all()
    assert out["be"].is_not_null().all()


# %% exclude_negative_earnings


def test_exclude_negative_earnings_aborts_when_earnings_column_is_absent():
    """Test exclude_negative_earnings aborts when earnings column is absent."""
    data = make_data().drop("ib")
    with pytest.raises(ValueError, match="ib"):
        filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_earnings=True),
        )


def test_exclude_negative_earnings_removes_nonpositive_and_na_messages():
    """Test exclude_negative_earnings drops non-positive and null, warns."""
    data = set_null(make_data(), "ib", 0)
    with pytest.warns(UserWarning, match="exclude_negative_earnings"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_earnings=True),
        )
    assert (out["ib"] > 0).all()
    assert out["ib"].is_not_null().all()


# %% quiet


def test_quiet_true_suppresses_messages_across_all_filters():
    """Test quiet = True suppresses messages across all filters."""
    data = make_data()
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        filter_sorting_data(
            data,
            filter_options=filter_options(
                exclude_financials=True,
                exclude_utilities=True,
                min_stock_price=1,
                min_listing_age=12,
                exclude_negative_book_equity=True,
                exclude_negative_earnings=True,
            ),
            quiet=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
