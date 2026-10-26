"""Tests for disconnect_connection."""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import disconnect_connection  # noqa: E402


def test_returns_true_on_successful_disconnect():
    """Test disconnect_connection returns True on success."""
    connection = MagicMock()
    result = disconnect_connection(connection)

    assert result is True
    connection.close.assert_called_once()


def test_propagates_errors_from_close():
    """Test disconnect_connection propagates errors from close, as in R."""
    connection = MagicMock()
    connection.close.side_effect = RuntimeError("disconnection failed")

    with pytest.raises(RuntimeError, match="disconnection failed"):
        disconnect_connection(connection)


if __name__ == "__main__":
    pytest.main([__file__])
