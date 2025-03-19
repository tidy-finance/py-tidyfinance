"""Test script for tidyfinance package."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from tidyfinance.utilities import list_supported_indexes, trim, winsorize


def test_set_wrds_credentials(tmp_path):
    """Test function for set_wrds_credentials using a temporary directory."""
    test_config_path = tmp_path / "config.yaml"
    test_gitignore_path = tmp_path / ".gitignore"
    test_credentials = {
        "WRDS": {"USER": "test_user", "PASSWORD": "test_password"}
    }

    with open(test_config_path, "w") as file:
        yaml.safe_dump(test_credentials, file)
    assert test_config_path.exists()

    with open(test_config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    assert loaded_config["WRDS"]["USER"] == "test_user"
    assert loaded_config["WRDS"]["PASSWORD"] == "test_password"

    with open(test_gitignore_path, "w") as file:
        file.write("config.yaml\n")

    assert test_gitignore_path.exists()

    with open(test_gitignore_path, "r") as file:
        gitignore_content = file.readlines()

    assert "config.yaml\n" in gitignore_content


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
    """Test that trim correctly removes extreme values."""
    np.random.seed(123)
    x = np.random.randn(100)
    cut = 0.05

    trimmed_x = trim(x, cut)

    assert np.min(trimmed_x) >= np.quantile(x, cut), (
        "Lower bound not correctly applied"
    )
    assert np.max(trimmed_x) <= np.quantile(x, 1 - cut), (
        "Upper bound not correctly applied"
    )


def test_trim_handles_na():
    """Test that trim correctly handles NaN values."""
    x = np.array([np.nan, 1, 2, 3, 4, 5, np.nan])
    cut = 0.1

    trimmed_x = trim(x, cut)

    assert not np.any(np.isnan(trimmed_x)), "NaN values should be removed"


def test_trim_edge_cases():
    """Test trim with edge cases such as empty input and identical values."""
    x = np.full(10, 1.0)
    assert np.array_equal(trim(x, 0.1), x), (
        "Identical values should remain unchanged"
    )


def test_list_supported_indexes():
    """Test that the function returns a DataFrame with the expected columns."""
    df = list_supported_indexes()
    assert isinstance(df, pd.DataFrame)
    assert "index" in df.columns
    assert "url" in df.columns
    assert "skip" in df.columns


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
