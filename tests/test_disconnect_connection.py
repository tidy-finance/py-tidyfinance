"""Tests for disconnect_connection."""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.utilities import disconnect_connection  # noqa: E402


def test_returns_true_on_successful_disconnect():
    """Test disconnect_connection returns True on success."""
    connection = MagicMock()
    result = disconnect_connection(connection)

    assert result is True
    connection.close.assert_called_once()


def test_returns_false_when_close_raises():
    """Test disconnect_connection returns False when close fails."""
    connection = MagicMock()
    connection.close.side_effect = Exception("disconnection failed")

    result = disconnect_connection(connection)

    assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
