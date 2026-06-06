"""Test script for tidyfinance package."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import download_data  # noqa: E402


def test_download_data_factors_invalid_data_set():
    with pytest.raises(ValueError, match="Unsupported domain."):
        download_data(
            domain="invalid_data_set",
            dataset="invalid",
            start_date="2020-01-01",
            end_date="2022-12-31",
        )


def test_download_data_column_ordering():
    df = download_data(
        domain="stock_prices",
        symbols="AAPL",
        start_date="2000-01-01",
        end_date="2023-12-31"
    )
    expected_columns = ['symbol', 'date', 'volume', 'open', 'low', 'high',
                        'close', 'adjusted_close']
    assert list(df.columns) == expected_columns, ("Expected columns "
                                                  f"{expected_columns}, but "
                                                  f"got {list(df.columns)}"
                                                  )


def test_download_data_requires_domain():
    """Raise when neither `domain` nor legacy `type` is supplied."""
    with pytest.raises(ValueError, match="'domain' is required"):
        download_data()


def test_download_data_rejects_unsupported_domain():
    """Raise an informative error for an unknown domain."""
    with pytest.raises(ValueError, match="Unsupported domain"):
        download_data(domain="not_a_domain")


def test_download_data_legacy_type_kwarg_warns():
    """Emit DeprecationWarning when `type=` is used and translate it."""
    with patch(
        "tidyfinance.data_download._download_data_factors_ff",
        return_value="sentinel",
    ):
        with pytest.warns(DeprecationWarning, match="'type' is deprecated"):
            result = download_data(type="factors_ff_3_monthly")
    assert result == "sentinel"


def test_download_data_legacy_type_as_domain_warns():
    """Emit DeprecationWarning when a legacy type is passed as `domain`."""
    with patch(
        "tidyfinance.data_download._download_data_factors_ff",
        return_value="sentinel",
    ):
        with pytest.warns(DeprecationWarning, match="legacy"):
            result = download_data("factors_ff_3_monthly")
    assert result == "sentinel"


def test_download_data_pseudo_dispatches_to_simulate():
    """Route domain="pseudo" through _simulate_pseudo_data."""
    with pytest.warns(UserWarning, match="pseudo"):
        result = download_data(
            domain="pseudo",
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-03-31",
            n_assets=3,
        )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert {"permno", "date", "ret"}.issubset(result.columns)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
