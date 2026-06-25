"""Tests for get_wrds_connection and load_wrds_credentials."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.utilities import (  # noqa: E402
    get_wrds_connection,
    load_wrds_credentials,
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Prevent load_dotenv from reading a local .env during tests."""
    monkeypatch.setattr(
        "tidyfinance.utilities.load_dotenv", lambda *a, **k: None
    )


def test_load_wrds_credentials_returns_tuple_when_set(monkeypatch):
    """Test load_wrds_credentials returns the credentials when set."""
    monkeypatch.setenv("WRDS_USER", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "pass")

    assert load_wrds_credentials() == ("user", "pass")


def test_load_wrds_credentials_raises_when_missing(monkeypatch):
    """Test load_wrds_credentials raises when credentials are missing."""
    monkeypatch.delenv("WRDS_USER", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="WRDS credentials not found"):
        load_wrds_credentials()


def test_get_wrds_connection_raises_when_credentials_missing(monkeypatch):
    """Test get_wrds_connection raises when credentials are missing."""
    monkeypatch.delenv("WRDS_USER", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="WRDS credentials not found"):
        get_wrds_connection()


def test_get_wrds_connection_returns_connection_when_set(monkeypatch):
    """Test get_wrds_connection returns a connection when credentials set."""
    monkeypatch.setenv("WRDS_USER", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "pass")

    fake_connection = MagicMock(name="connection")
    fake_engine = MagicMock(name="engine")
    fake_engine.connect.return_value = fake_connection

    with patch(
        "tidyfinance.utilities.create_engine", return_value=fake_engine
    ) as mock_create_engine:
        result = get_wrds_connection()

    assert result is fake_connection
    mock_create_engine.assert_called_once()
    fake_engine.connect.assert_called_once()


def test_get_wrds_connection_propagates_connect_errors(monkeypatch):
    """Test get_wrds_connection propagates errors from connecting."""
    monkeypatch.setenv("WRDS_USER", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "pass")

    fake_engine = MagicMock(name="engine")
    fake_engine.connect.side_effect = Exception("connection refused")

    with patch("tidyfinance.utilities.create_engine", return_value=fake_engine):
        with pytest.raises(Exception, match="connection refused"):
            get_wrds_connection()


if __name__ == "__main__":
    pytest.main([__file__])
