"""Tests for the WRDS connection and download progress messages."""

import datetime as dt
import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_pseudo import _simulate_pseudo_data  # noqa: E402
from tidyfinance.download_wrds import (  # noqa: E402
    _download_data_wrds_ccm_links,
    _report_batch_progress,
    get_wrds_connection,
)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    """Set test credentials and keep a local .env out of the tests."""
    monkeypatch.setattr(
        "tidyfinance.download_wrds.load_dotenv", lambda *a, **k: None
    )
    monkeypatch.setenv("WRDS_USER", "wrds_jdoe")
    monkeypatch.setenv("WRDS_PASSWORD", "s3cr3t-pw")


def _engine(connect_error=None):
    """Patch create_engine with an engine whose connect() succeeds or
    raises 'connect_error'."""
    engine = MagicMock(name="engine")
    if connect_error is None:
        engine.connect.return_value = MagicMock(name="connection")
    else:
        engine.connect.side_effect = connect_error
    return patch("tidyfinance.download_wrds.create_engine", return_value=engine)


def _ccm_links_result():
    return pl.DataFrame(
        {
            "permno": [1],
            "gvkey": ["001000"],
            "linkdt": [dt.date(2000, 1, 1)],
            "linkenddt": [None],
        },
        schema_overrides={"linkenddt": pl.Date},
    )


# %% get_wrds_connection


def test_get_wrds_connection_says_when_it_connects(capsys):
    """Test the connection messages, without the username."""
    with _engine():
        get_wrds_connection()

    err = capsys.readouterr().err
    assert "Connecting to WRDS. Approve the Duo request if prompted." in err
    assert "Connected to WRDS." in err
    assert "wrds_jdoe" not in err


def test_get_wrds_connection_reports_a_failed_connection(capsys):
    """Test a failed connection is reported before the error."""
    with _engine(Exception("connection refused")):
        with pytest.raises(ConnectionError):
            get_wrds_connection()

    err = capsys.readouterr().err
    assert "Connection to WRDS failed." in err
    assert "Connected to WRDS." not in err


def test_get_wrds_connection_is_silent_with_quiet(capsys):
    """Test quiet=True suppresses the connection messages."""
    with _engine():
        get_wrds_connection(quiet=True)

    assert capsys.readouterr().err == ""


def test_get_wrds_connection_validates_quiet():
    """Test quiet must be a boolean."""
    with pytest.raises(ValueError, match="quiet"):
        get_wrds_connection(quiet=None)


# %% Download steps


def test_download_reports_its_step_after_connecting(capsys):
    """Test a download is announced once connected and reports the
    elapsed time when it returns, in that order."""
    with (
        _engine(),
        patch(
            "tidyfinance.download_wrds._read_sql",
            return_value=_ccm_links_result(),
        ),
    ):
        _download_data_wrds_ccm_links()

    err = capsys.readouterr().err
    messages = [
        "Connecting to WRDS.",
        "Connected to WRDS.",
        "Downloading CRSP-Compustat links from WRDS. This can take a few "
        "minutes.",
        "Downloaded CRSP-Compustat links from WRDS. [",
    ]
    positions = [err.index(m) for m in messages]
    assert positions == sorted(positions)
    assert "wrds_jdoe" not in err


def test_download_reports_a_failed_download(capsys):
    """Test a download that fails after connecting says so."""
    with (
        _engine(),
        patch(
            "tidyfinance.download_wrds._read_sql",
            side_effect=RuntimeError("query failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="query failed"):
            _download_data_wrds_ccm_links()

    err = capsys.readouterr().err
    assert "Downloading CRSP-Compustat links from WRDS failed." in err
    assert "Downloaded CRSP-Compustat links" not in err


def test_download_is_not_announced_when_the_connection_fails(capsys):
    """Test nothing is said about the download if it never started."""
    with _engine(Exception("connection refused")):
        with pytest.raises(ConnectionError):
            _download_data_wrds_ccm_links()

    err = capsys.readouterr().err
    assert "Connection to WRDS failed." in err
    assert "CRSP-Compustat links" not in err


def test_download_is_silent_with_quiet(capsys):
    """Test quiet=True suppresses all connection and download messages."""
    with (
        _engine(),
        patch(
            "tidyfinance.download_wrds._read_sql",
            return_value=_ccm_links_result(),
        ),
    ):
        _download_data_wrds_ccm_links(quiet=True)

    assert capsys.readouterr().err == ""


# %% Daily CRSP batches


def test_batch_progress_updates_one_line_and_clears_it(capsys):
    """Test the batch progress is one line, updated in place and cleared
    after the last batch."""
    _report_batch_progress(1, 2, quiet=False)
    _report_batch_progress(2, 2, quiet=False)

    err = capsys.readouterr().err
    assert "1/2 (50%)" in err
    assert "2/2 (100%)" in err
    assert "\n" not in err
    assert err.endswith("\r")


def test_batch_progress_is_silent_with_quiet(capsys):
    """Test quiet=True suppresses the batch progress."""
    _report_batch_progress(1, 2, quiet=True)

    assert capsys.readouterr().err == ""


# %% Pseudo data


def test_pseudo_data_notice_can_be_suppressed():
    """Test quiet=True omits the pseudo-data notice, so the same call
    works for domain='WRDS' and domain='Pseudo Data'."""
    with pytest.warns(UserWarning, match="pseudo data"):
        _simulate_pseudo_data("ccm_links", n_assets=5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _simulate_pseudo_data("ccm_links", n_assets=5, quiet=True)
