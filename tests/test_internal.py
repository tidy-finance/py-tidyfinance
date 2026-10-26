"""Test script for tidyfinance package."""

import datetime as dt
import os
import sys
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from dotenv import dotenv_values

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from tidyfinance._internal import (
    _get_random_user_agent,
    _shift_years,
    _validate_column_name,
    _validate_dates,
    _validate_flag,
    _validate_optional_number,
)
from tidyfinance.download_wrds import set_wrds_credentials
from tidyfinance.utilities import list_supported_indexes, trim, winsorize


def test_set_wrds_credentials(tmp_path, monkeypatch):
    """Test that set_wrds_credentials writes credentials to a .env file."""
    monkeypatch.chdir(tmp_path)

    test_gitignore_path = tmp_path / ".gitignore"
    test_gitignore_path.write_text("")

    inputs = iter(["test_user", "test_password", "project", "yes"])
    monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

    set_wrds_credentials()

    env_path = tmp_path / ".env"
    assert env_path.exists()

    credentials = dotenv_values(env_path)
    assert credentials["WRDS_USER"] == "test_user"
    assert credentials["WRDS_PASSWORD"] == "test_password"

    gitignore_content = test_gitignore_path.read_text().splitlines()
    assert ".env" in gitignore_content


def test_winsorize_correct_adjustment():
    """Test that winsorize correctly adjusts extreme values."""
    np.random.seed(123)
    x = np.random.randn(100)
    cut = 0.05
    winsorized_x = winsorize(x, cut)

    assert np.min(winsorized_x) == np.quantile(x, cut), (
        "Lower bound not correctly applied"
    )
    assert np.max(winsorized_x) == np.quantile(x, 1 - cut), (
        "Upper bound not correctly applied"
    )
    assert np.all(winsorized_x >= np.quantile(x, cut)), (
        "Values below lower bound not adjusted"
    )
    assert np.all(winsorized_x <= np.quantile(x, 1 - cut)), (
        "Values above upper bound not adjusted"
    )


def test_winsorize_handles_na():
    """Test that winsorize correctly handles NaN values."""
    x = np.array([np.nan, 1, 2, 3, 4, 5, np.nan])
    cut = 0.1
    winsorized_x = winsorize(x, cut)

    assert len(winsorized_x) == len(x), (
        "Output length should match input length"
    )
    assert np.all(np.isnan(winsorized_x) == np.isnan(x)), (
        "NaN values should remain unchanged"
    )
    assert np.all(
        winsorized_x[~np.isnan(winsorized_x)] >= np.nanquantile(x, cut)
    ), "Non-NaN values below lower bound not adjusted"
    assert np.all(
        winsorized_x[~np.isnan(winsorized_x)] <= np.nanquantile(x, 1 - cut)
    ), "Non-NaN values above upper bound not adjusted"


def test_winsorize_edge_cases():
    """Test winsorize with edge cases (empty input and identical values)."""
    assert np.array_equal(winsorize([], 0.1), np.array([])), (
        "Empty array should return empty array"
    )
    x = np.full(10, 1.0)
    assert np.array_equal(winsorize(x, 0.1), x), (
        "Identical values should remain unchanged"
    )


def test_trim_correct_removal():
    """Values outside the cut quantiles become NaN; the length is kept."""
    np.random.seed(123)
    x = np.random.randn(100)
    cut = 0.05

    trimmed_x = trim(x, cut)

    assert trimmed_x.shape == x.shape
    lb, ub = np.quantile(x, [cut, 1 - cut])
    outside = (x < lb) | (x > ub)
    assert np.all(np.isnan(trimmed_x[outside]))
    assert np.array_equal(trimmed_x[~outside], x[~outside])


def test_trim_handles_na():
    """Missing values stay missing and the tails are set to NaN."""
    np.random.seed(123)
    x = np.concatenate([[np.nan], np.random.randn(100), [np.nan]])
    cut = 0.05

    trimmed_x = trim(x, cut)

    assert trimmed_x.shape == x.shape
    assert np.all(np.isnan(trimmed_x[np.isnan(x)]))
    lb, ub = np.nanquantile(x, [cut, 1 - cut])
    assert np.all(np.isnan(trimmed_x[(x < lb) | (x > ub)]))


def test_trim_keeps_the_length_and_sets_the_tails_to_nan():
    out = trim(np.arange(1, 101), 0.05)
    assert out.shape == (100,)
    assert np.isnan(out).sum() == 10


def test_trim_edge_cases():
    """Test trim with edge cases such as empty input and identical values."""
    x = np.full(10, 1.0)
    assert np.array_equal(trim(x, 0.1), x), (
        "Identical values should remain unchanged"
    )


def test_list_supported_indexes():
    """Test that the function returns a DataFrame with the expected columns."""
    df = list_supported_indexes()
    assert isinstance(df, pl.DataFrame)
    assert "index" in df.columns
    assert "url" in df.columns
    assert "skip" in df.columns


def test_list_supported_indexes_content():
    """The indexes and providers match the R package."""
    df = list_supported_indexes()
    assert df.columns == ["index", "url", "skip"]
    assert df.height == 20
    assert df["index"].to_list()[-3:] == [
        "MSCI Europe",
        "MSCI ACWI",
        "S&P SmallCap 600",
    ]
    # iShares removed the German retail pages; all files are read from
    # the Swiss professional pages or BlackRock Japan.
    assert not df["url"].str.contains("/de/privatanleger/").any()


# %% _validate_dates


def test_validate_dates_both_none_returns_none_pair():
    assert _validate_dates(None, None) == (None, None)


def test_validate_dates_both_none_with_default_range_returns_two_years():
    start, end = _validate_dates(None, None, use_default_range=True)
    today = dt.date.today()
    assert isinstance(start, dt.date) and isinstance(end, dt.date)
    assert start == _shift_years(today, -2)
    assert end == _shift_years(today, -1)


def test_validate_dates_coerces_a_valid_pair_to_dates():
    start, end = _validate_dates("2022-01-01", "2022-12-31")
    assert start == dt.date(2022, 1, 1)
    assert end == dt.date(2022, 12, 31)


def test_validate_dates_start_after_end_raises():
    with pytest.raises(ValueError):
        _validate_dates("2022-12-31", "2022-01-01")


# %% option validators


def test_validate_column_name_accepts_a_single_string():
    assert _validate_column_name("permno", "id", "entity") is None


def test_validate_column_name_rejects_non_strings_and_non_scalars():
    for bad in (123, ["a", "b"], None, True):
        with pytest.raises(ValueError, match="id"):
            _validate_column_name(bad, "id", "entity")


def test_validate_flag_accepts_a_single_boolean():
    _validate_flag(True, "flag")
    _validate_flag(False, "flag")


def test_validate_flag_rejects_non_booleans_none_and_sequences():
    for bad in ("yes", 1, None, [True, False]):
        with pytest.raises(ValueError, match="flag"):
            _validate_flag(bad, "flag")


def test_validate_flag_supports_a_custom_message():
    with pytest.raises(ValueError, match="True or False"):
        _validate_flag(None, "flag", "'flag' must be True or False.")


def test_validate_optional_number_passes_for_none():
    assert _validate_optional_number(None, "msg") is None


def test_validate_optional_number_enforces_strict_lower_bound():
    _validate_optional_number(0.5, "msg", min=0, min_strict=True)
    with pytest.raises(ValueError, match="msg"):
        _validate_optional_number(0, "msg", min=0, min_strict=True)


def test_validate_optional_number_enforces_an_exclusive_0_1_range():
    def in_unit(value):
        _validate_optional_number(
            value, "msg", min=0, max=1, min_strict=True, max_strict=True
        )

    in_unit(0.5)
    for bad in (0, 1):
        with pytest.raises(ValueError, match="msg"):
            in_unit(bad)


def test_validate_optional_number_rejects_nan_non_numbers_and_sequences():
    for bad in (float("nan"), "a", [0.1, 0.2]):
        with pytest.raises(ValueError, match="msg"):
            _validate_optional_number(bad, "msg")


def test_validate_optional_number_allows_an_inclusive_non_negative_bound():
    _validate_optional_number(0, "msg", min=0)
    with pytest.raises(ValueError, match="msg"):
        _validate_optional_number(-1, "msg", min=0)


# %% _get_random_user_agent


def test_get_random_user_agent_returns_first_agent():
    with patch(
        "tidyfinance._internal.np.random.choice",
        side_effect=lambda agents: agents[0],
    ):
        user_agent = _get_random_user_agent()
    assert isinstance(user_agent, str)
    assert "Windows NT 10.0" in user_agent
    assert "Chrome/122.0.6261.111" in user_agent


def test_get_random_user_agent_returns_last_agent():
    with patch(
        "tidyfinance._internal.np.random.choice",
        side_effect=lambda agents: agents[-1],
    ):
        user_agent = _get_random_user_agent()
    assert isinstance(user_agent, str)
    assert "Firefox/125.0" in user_agent


def test_get_random_user_agent_samples_from_twenty_agents():
    with patch(
        "tidyfinance._internal.np.random.choice",
        side_effect=lambda agents: str(len(agents)),
    ):
        assert _get_random_user_agent() == "20"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
