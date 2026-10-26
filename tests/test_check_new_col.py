"""Tests for _check_new_col and the functions that use it."""

import datetime as dt
import os
import sys
import warnings

import numpy as np
import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance._internal import _check_new_col  # noqa: E402
from tidyfinance.lagging import (  # noqa: E402
    add_lagged_columns,
    join_lagged_values,
)
from tidyfinance.portfolios import (  # noqa: E402
    breakpoint_options,
    compute_long_short_returns,
    compute_portfolio_returns,
    filter_options,
    filter_sorting_data,
)


def _month_starts(n_months: int) -> list[dt.date]:
    """Return 'n_months' consecutive month starts from January 2020."""
    return [dt.date(2020 + m // 12, m % 12 + 1, 1) for m in range(n_months)]


def make_panel() -> pl.DataFrame:
    """Minimal stock-return panel used throughout these tests."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "permno": np.repeat(np.arange(1, 21), 6),
            "date": _month_starts(6) * 20,
            "ret_excess": rng.standard_normal(120),
            "mktcap_lag": rng.uniform(100, 1000, 120),
            "bm": rng.standard_normal(120),
        }
    )


def _capture(fn) -> list[warnings.WarningMessage]:
    """Run 'fn' and return the warnings it emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return caught


# %% _check_new_col


def test_does_not_warn_for_genuinely_new_column_names():
    """Test _check_new_col does not warn for genuinely new column names."""
    assert _capture(lambda: _check_new_col(make_panel(), "bm_lag")) == []


def test_does_not_warn_when_cols_is_empty():
    """Test _check_new_col does not warn when cols is empty."""
    assert _capture(lambda: _check_new_col(make_panel(), [])) == []


def test_warns_when_ret_excess_would_be_overwritten():
    """Test _check_new_col warns when ret_excess would be overwritten."""
    with pytest.warns(UserWarning, match="[Oo]verwrit"):
        _check_new_col(make_panel(), "ret_excess")


def test_warns_when_multiple_portfolio_columns_would_be_overwritten():
    """Test _check_new_col warns when multiple portfolio columns would be
    overwritten."""
    data = make_panel().with_columns(
        portfolio_main=pl.lit(None, dtype=pl.Int64),
        portfolio_secondary=pl.lit(None, dtype=pl.Int64),
    )
    with pytest.warns(UserWarning, match="[Oo]verwrit"):
        _check_new_col(data, ["portfolio_main", "portfolio_secondary"])


def test_warning_message_contains_the_offending_column_name():
    """Test _check_new_col warning message contains the offending column
    name."""
    with pytest.warns(UserWarning, match="mktcap_lag"):
        _check_new_col(make_panel(), "mktcap_lag")


def test_uses_the_overwriting_branch_when_caller_has_no_name():
    """Test _check_new_col uses the 'Overwriting' branch when no caller is
    given."""
    with pytest.warns(UserWarning, match="^Overwriting"):
        _check_new_col(make_panel(), "ret_excess")


def test_names_the_caller_when_given():
    """Test _check_new_col names the caller when given."""
    with pytest.warns(UserWarning, match="^'my_function' overwrites"):
        _check_new_col(make_panel(), "ret_excess", "my_function")


def test_emits_exactly_one_warning_even_for_multiple_overlapping_cols():
    """Test _check_new_col emits exactly one warning even for multiple
    overlapping cols."""
    data = make_panel().with_columns(
        bm_lag=pl.lit(None, dtype=pl.Float64),
        ret_excess_lag=pl.lit(None, dtype=pl.Float64),
    )
    caught = _capture(
        lambda: _check_new_col(data, ["bm_lag", "ret_excess_lag"])
    )
    assert len(caught) == 1


def test_warns_only_for_existing_columns_not_new_ones_in_the_same_call():
    """Test _check_new_col warns only for existing columns, not new ones in
    the same call."""
    caught = _capture(lambda: _check_new_col(make_panel(), ["bm", "bm_lag"]))
    assert len(caught) == 1
    assert "'bm'" in str(caught[0].message)
    assert "bm_lag" not in str(caught[0].message)


def test_returns_the_input_data_unchanged_when_no_overlap():
    """Test _check_new_col returns the input data unchanged when there is
    no overlap."""
    data = make_panel()
    assert _check_new_col(data, "bm_lag").equals(data)


def test_drops_the_overlapping_columns_when_overlap_triggers_a_warning():
    """Test _check_new_col drops the columns it warns about, so that the
    caller can create them afresh."""
    data = make_panel()
    with pytest.warns(UserWarning):
        result = _check_new_col(data, "ret_excess")
    assert result.equals(data.drop("ret_excess"))


# %% functions that create columns


def test_add_lagged_columns_errors_when_bm_lag_already_exists_in_data():
    """Test add_lagged_columns errors when bm_lag already exists in data."""
    data = make_panel().with_columns(bm_lag=pl.lit(None, dtype=pl.Float64))
    with pytest.raises(ValueError, match="already exist"):
        add_lagged_columns(data, cols="bm", lag="3mo", by="permno")


def test_add_lagged_columns_warns_when_upper_exists_and_max_lag_is_used():
    """Test add_lagged_columns warns when '_upper' already exists and
    max_lag is used."""
    data = make_panel().with_columns(_upper=pl.lit(None))
    with pytest.warns(UserWarning, match="overwrites"):
        add_lagged_columns(
            data, cols="bm", lag="1mo", max_lag="3mo", by="permno"
        )


def test_compute_long_short_returns_warns_when_leg_already_exists():
    """Test compute_long_short_returns warns when '_leg' already exists
    in data."""
    rng = np.random.default_rng(42)
    portfolio_returns = pl.DataFrame(
        {
            "date": [d for d in _month_starts(6) for _ in range(5)],
            "portfolio": list(range(1, 6)) * 6,
            "ret_excess": rng.standard_normal(30),
            "_leg": [None] * 30,
        }
    )
    with pytest.warns(UserWarning, match="overwrites"):
        compute_long_short_returns(portfolio_returns)


def test_filter_sorting_data_warns_when_size_cutoff_already_exists():
    """Test filter_sorting_data warns when '_size_cutoff' already exists
    in data."""
    data = make_panel().with_columns(
        _size_cutoff=pl.lit(None, dtype=pl.Float64),
        exchange=pl.lit("NYSE"),
    )
    with pytest.warns(UserWarning, match="overwrites"):
        filter_sorting_data(
            data,
            filter_options=filter_options(min_size_quantile=0.2),
            quiet=True,
        )


def test_join_lagged_values_warns_when_new_data_already_has_year():
    """Test join_lagged_values warns when new_data already has '_year' and
    ff_adjustment is True."""
    panel = make_panel()
    original = panel.select("permno", "date", "ret_excess", "mktcap_lag")
    new = panel.select("permno", "date", "bm").with_columns(
        _year=pl.lit(None, dtype=pl.Float64)
    )
    with pytest.warns(UserWarning, match="overwrites"):
        join_lagged_values(
            original_data=original,
            new_data=new,
            id_keys="permno",
            ff_adjustment=True,
        )


def test_compute_portfolio_returns_warns_when_portfolio_already_exists():
    """Test compute_portfolio_returns warns when portfolio already exists
    in data."""
    data = make_panel().with_columns(portfolio=pl.lit(None, dtype=pl.Int64))
    with pytest.warns(UserWarning, match="overwrites"):
        compute_portfolio_returns(
            data,
            sorting_variables="bm",
            sorting_method="univariate",
            breakpoint_options_main=breakpoint_options(n_portfolios=3),
            quiet=True,
        )


def test_compute_portfolio_returns_ignores_an_existing_portfolio_column():
    """Test an existing portfolio column does not change the returns."""
    data = make_panel()
    expected = compute_portfolio_returns(
        data,
        ["bm", "mktcap_lag"],
        "bivariate-independent",
        breakpoint_options_main=breakpoint_options(n_portfolios=2),
        breakpoint_options_secondary=breakpoint_options(n_portfolios=2),
        quiet=True,
    )
    with pytest.warns(UserWarning, match="'portfolio_main'"):
        result = compute_portfolio_returns(
            data.with_columns(portfolio_main=pl.lit(9, dtype=pl.Int64)),
            ["bm", "mktcap_lag"],
            "bivariate-independent",
            breakpoint_options_main=breakpoint_options(n_portfolios=2),
            breakpoint_options_secondary=breakpoint_options(n_portfolios=2),
            quiet=True,
        )
    assert result.equals(expected)


if __name__ == "__main__":
    pytest.main([__file__])
