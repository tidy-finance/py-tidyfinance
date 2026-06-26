"""Tests for filter_sorting_data."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.portfolios import filter_options, filter_sorting_data  # noqa: E402


def make_data():
    """Construct a stock-level panel for tests."""
    return pd.DataFrame(
        {
            "permno": list(range(1, 11)),
            "date": pd.to_datetime(["2020-01-01"] * 5 + ["2020-02-01"] * 5),
            "exchange": ["NYSE", "NYSE", "NASDAQ", "NYSE", "NASDAQ"] * 2,
            "siccd": [6500, 2000, 4950, 3000, 6100] * 2,
            "prc_adj": [10.0, 0.5, 15.0, 20.0, 5.0] * 2,
            "mktcap_lag": [100.0, 200.0, 50.0, 500.0, 300.0] * 2,
            "listing_age": [24, 6, 60, 12, 36] * 2,
            "be": [10.0, -5.0, 50.0, 100.0, 0.0] * 2,
            "ib": [5.0, -2.0, 20.0, -10.0, 30.0] * 2,
        }
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
    pd.testing.assert_frame_equal(out, data)


# %% SIC filters


def test_sic_filters_abort_when_siccd_column_is_absent():
    """Test SIC filters abort when siccd column is absent."""
    data = make_data().drop(columns="siccd")
    with pytest.raises(ValueError, match="siccd"):
        filter_sorting_data(
            data, filter_options=filter_options(exclude_financials=True)
        )


def test_exclude_financials_removes_sic_6000_6799_keeps_na_messages():
    """Test exclude_financials removes SIC 6000-6799, keeps NaN, warns."""
    data = make_data()
    data.loc[0, "siccd"] = np.nan  # NaN row should be kept
    with pytest.warns(UserWarning, match="exclude_financials"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_financials=True),
        )
    # NaN row kept; SIC 6500 and 6100 dropped
    assert pd.isna(out["siccd"]).any()
    assert not ((out["siccd"] >= 6000) & (out["siccd"] <= 6799)).any()


def test_exclude_utilities_removes_sic_4900_4999_keeps_na_messages():
    """Test exclude_utilities removes SIC 4900-4999, keeps NaN, warns."""
    data = make_data()
    data.loc[0, "siccd"] = np.nan
    with pytest.warns(UserWarning, match="exclude_utilities"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_utilities=True),
        )
    assert pd.isna(out["siccd"]).any()
    assert not ((out["siccd"] >= 4900) & (out["siccd"] <= 4999)).any()


# %% min_stock_price


def test_min_stock_price_aborts_when_price_column_is_absent():
    """Test min_stock_price aborts when price column is absent."""
    data = make_data().drop(columns="prc_adj")
    with pytest.raises(ValueError, match="prc_adj"):
        filter_sorting_data(
            data, filter_options=filter_options(min_stock_price=1)
        )


def test_min_stock_price_removes_below_threshold_and_na_rows():
    """Test min_stock_price removes below-threshold and NaN rows, warns."""
    data = make_data()
    data.loc[0, "prc_adj"] = np.nan
    with pytest.warns(UserWarning, match="min_stock_price"):
        out = filter_sorting_data(
            data, filter_options=filter_options(min_stock_price=1)
        )
    assert (out["prc_adj"] >= 1).all()
    assert out["prc_adj"].notna().all()


# %% min_size_quantile


def test_min_size_quantile_aborts_when_mktcap_lag_column_is_absent():
    """Test min_size_quantile aborts when mktcap_lag column is absent."""
    data = make_data().drop(columns="mktcap_lag")
    with pytest.raises(ValueError, match="mktcap_lag"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_aborts_when_date_column_is_absent():
    """Test min_size_quantile aborts when date column is absent."""
    data = make_data().drop(columns="date")
    with pytest.raises(ValueError, match="date"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_aborts_when_exchange_column_is_absent():
    """Test min_size_quantile aborts when exchange column is absent."""
    data = make_data().drop(columns="exchange")
    with pytest.raises(ValueError, match="exchange"):
        filter_sorting_data(
            data, filter_options=filter_options(min_size_quantile=0.2)
        )


def test_min_size_quantile_warns_on_dates_with_no_nyse_observations():
    """Test min_size_quantile warns on dates with no NYSE observations."""
    import warnings as _w

    data = make_data()
    data = data[
        ~(
            (data["date"] == pd.Timestamp("2020-01-01"))
            & (data["exchange"] == "NYSE")
        )
    ]
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


def test_min_size_quantile_emits_no_message_when_no_rows_are_removed():
    """Test min_size_quantile emits no message when no rows are removed."""
    # All rows are at or above the lowest NYSE size cutoff -> no removal
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
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
    data = make_data().drop(columns="listing_age")
    with pytest.raises(ValueError, match="listing_age"):
        filter_sorting_data(
            data, filter_options=filter_options(min_listing_age=12)
        )


def test_min_listing_age_removes_young_and_na_stocks_messages():
    """Test min_listing_age removes young and NaN stocks, warns."""
    data = make_data()
    data.loc[0, "listing_age"] = np.nan
    with pytest.warns(UserWarning, match="min_listing_age"):
        out = filter_sorting_data(
            data, filter_options=filter_options(min_listing_age=12)
        )
    assert (out["listing_age"] >= 12).all()
    assert out["listing_age"].notna().all()


# %% exclude_negative_book_equity


def test_exclude_negative_book_equity_aborts_when_be_column_is_absent():
    """Test exclude_negative_book_equity aborts when be column is absent."""
    data = make_data().drop(columns="be")
    with pytest.raises(ValueError, match="be"):
        filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_book_equity=True),
        )


def test_exclude_negative_book_equity_removes_nonpositive_and_na_messages():
    """Test exclude_negative_book_equity removes non-positive and NaN, warns."""
    data = make_data()
    data.loc[0, "be"] = np.nan
    with pytest.warns(UserWarning, match="exclude_negative_book_equity"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_book_equity=True),
        )
    assert (out["be"] > 0).all()
    assert out["be"].notna().all()


# %% exclude_negative_earnings


def test_exclude_negative_earnings_aborts_when_earnings_column_is_absent():
    """Test exclude_negative_earnings aborts when earnings column is absent."""
    data = make_data().drop(columns="ib")
    with pytest.raises(ValueError, match="ib"):
        filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_earnings=True),
        )


def test_exclude_negative_earnings_removes_nonpositive_and_na_messages():
    """Test exclude_negative_earnings removes non-positive and NaN, warns."""
    data = make_data()
    data.loc[0, "ib"] = np.nan
    with pytest.warns(UserWarning, match="exclude_negative_earnings"):
        out = filter_sorting_data(
            data,
            filter_options=filter_options(exclude_negative_earnings=True),
        )
    assert (out["ib"] > 0).all()
    assert out["ib"].notna().all()


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
