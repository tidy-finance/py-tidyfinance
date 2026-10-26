"""Tests for get_wrds_connection and _load_wrds_credentials."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_wrds import (  # noqa: E402
    _load_wrds_credentials,
    get_wrds_connection,
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Prevent load_dotenv from reading a local .env during tests."""
    monkeypatch.setattr(
        "tidyfinance.download_wrds.load_dotenv", lambda *a, **k: None
    )


def test_load_wrds_credentials_returns_tuple_when_set(monkeypatch):
    """Test _load_wrds_credentials returns the credentials when set."""
    monkeypatch.setenv("WRDS_USER", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "pass")

    assert _load_wrds_credentials() == ("user", "pass")


def test_load_wrds_credentials_raises_when_missing(monkeypatch):
    """Test _load_wrds_credentials raises when credentials are missing."""
    monkeypatch.delenv("WRDS_USER", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="WRDS credentials not found"):
        _load_wrds_credentials()


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
        "tidyfinance.download_wrds.create_engine", return_value=fake_engine
    ) as mock_create_engine:
        result = get_wrds_connection()

    assert result is fake_connection
    mock_create_engine.assert_called_once()
    fake_engine.connect.assert_called_once()


def test_get_wrds_connection_explains_failed_connections(monkeypatch):
    """Test a failed connection lists the likely causes and keeps the
    original error as the cause."""
    monkeypatch.setenv("WRDS_USER", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "pass")

    fake_engine = MagicMock(name="engine")
    fake_engine.connect.side_effect = Exception("connection refused")

    with patch(
        "tidyfinance.download_wrds.create_engine", return_value=fake_engine
    ):
        with pytest.raises(
            ConnectionError, match="Could not connect to WRDS"
        ) as err:
            get_wrds_connection()

    message = str(err.value)
    assert "two-factor authentication" in message
    assert "terms of use" in message
    assert "connection refused" in str(err.value.__cause__)


def test_get_wrds_connection_removes_credentials_from_the_error(monkeypatch):
    """Test the username and password do not appear in the error."""
    monkeypatch.setenv("WRDS_USER", "wrds_jdoe")
    monkeypatch.setenv("WRDS_PASSWORD", "s3cr3t-pw")

    fake_engine = MagicMock(name="engine")
    fake_engine.connect.side_effect = Exception(
        "connection to server failed: FATAL: password authentication "
        'failed for user "wrds_jdoe" (password "s3cr3t-pw")'
    )

    with patch(
        "tidyfinance.download_wrds.create_engine", return_value=fake_engine
    ):
        with pytest.raises(ConnectionError) as err:
            get_wrds_connection()

    cause = str(err.value.__cause__)
    assert 'for user "<WRDS_USER>"' in cause
    assert '(password "<WRDS_PASSWORD>")' in cause
    for text in (str(err.value), cause):
        assert "wrds_jdoe" not in text
        assert "s3cr3t-pw" not in text


if __name__ == "__main__":
    pytest.main([__file__])
