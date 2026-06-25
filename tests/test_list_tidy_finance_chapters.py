"""Tests for list_tidy_finance_chapters."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.utilities import list_tidy_finance_chapters  # noqa: E402


def test_returns_list_of_chapter_names():
    """Test returns the expected list of chapter names."""
    result = list_tidy_finance_chapters()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(chapter, str) for chapter in result)


def test_includes_known_chapter():
    """Test that a known chapter slug is present."""
    result = list_tidy_finance_chapters()
    assert "beta-estimation" in result


if __name__ == "__main__":
    pytest.main([__file__])
