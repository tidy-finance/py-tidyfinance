"""Tests for download_data_constituents."""

import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import _download_data_constituents  # noqa: E402


def _supported_indexes_df(index="DAX"):
    return pl.DataFrame({"index": [index], "url": ["url"], "skip": [0]})


def test_unsupported_indexes_fail():
    """Test unsupported indexes fail."""
    with patch(
        "tidyfinance.download_open_source.list_supported_indexes",
        return_value=_supported_indexes_df(),
    ):
        with pytest.raises(ValueError, match="not supported"):
            _download_data_constituents("UNKNOWN")


def test_download_errors_can_return_none():
    """Test download errors can return None."""
    response_mock = MagicMock()
    response_mock.status_code = 500
    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df(),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            side_effect=Exception("network error"),
        ),
    ):
        with pytest.raises(Exception):
            _download_data_constituents("DAX")


def test_non_200_responses_fail():
    """Test non-200 responses fail."""
    response_mock = MagicMock()
    response_mock.status_code = 500
    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df(),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ),
    ):
        with pytest.raises(ValueError, match="Failed to download data"):
            _download_data_constituents("DAX")


def test_german_csv_layout_is_parsed_and_cleaned():
    """Test German CSV layout is parsed and cleaned."""
    csv = "\n".join(
        [
            "Anlageklasse,Emittententicker,Name,Standort,Börse",
            "Aktien, ABC ,Alpha AG,Germany,Xetra",
            "Aktien,-,Bad AG,Germany,Xetra",
            "Bonds,BND,Bond AG,Germany,Xetra",
            "Aktien,CASH,CASH,Germany,Xetra",
            "Aktien,DAX,DAX INDEX,Germany,Xetra",
        ]
    )
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = csv

    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df("DAX"),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ),
    ):
        out = _download_data_constituents("DAX")

    assert isinstance(out, pl.DataFrame)
    assert list(out["symbol"]) == ["ABC.DE"]
    assert list(out["name"]) == ["Alpha AG"]


def test_asset_class_layout_covers_exchanges_and_symbol_rules():
    """Test Asset Class layout covers exchanges and symbol rules."""
    exchanges = [
        "Deutsche Boerse AG",
        "Boerse Berlin",
        "Borsa Italiana",
        "Nyse Euronext - Euronext Paris",
        "Euronext Amsterdam",
        "Nasdaq Omx Helsinki Ltd.",
        "Singapore Exchange",
        "Asx - All Markets",
        "London Stock Exchange",
        "SIX Swiss Exchange",
        "Tel Aviv Stock Exchange",
        "Tokyo Stock Exchange",
        "Hong Kong Stock Exchange",
        "Toronto Stock Exchange",
        "Euronext Brussels",
        "Euronext Lisbon",
        "Bovespa",
        "Mexican Stock Exchange",
        "Stockholm Stock Exchange",
        "Oslo Stock Exchange",
        "Johannesburg Stock Exchange",
        "Korea Exchange",
        "Shanghai Stock Exchange",
        "Shenzhen Stock Exchange",
        "Other Exchange",
    ]
    rows = [
        f"Equity,SYM{i + 1},Company {i + 1},Loc,{ex}"
        for i, ex in enumerate(exchanges)
    ]
    rows[0] = f"Equity,BAD SYM/PR..,NATIONAL BANK OF CANADA,Loc,{exchanges[0]}"

    csv = "\n".join(
        ["Asset Class,Ticker,Name,Location,Exchange"]
        + rows
        + [
            "Cash,IGN,Ignored,Loc,Other Exchange",
            "Equity,USD,Bad,Loc,Other Exchange",
            "Equity,IDX,MSCI WORLD,Loc,Other Exchange",
        ]
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = csv

    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df("MSCI World"),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ),
    ):
        out = _download_data_constituents("MSCI World")

    assert isinstance(out, pl.DataFrame)
    # 25 exchange rows survive; symbol uniqueness mirrors R assertions
    assert len(out) == len(exchanges)
    symbols = set(out["symbol"].to_list())
    for expected in [
        "SYM2.BE",
        "SYM3.MI",
        "SYM4.PA",
        "SYM5.AS",
        "SYM6.HE",
        "SYM7.SI",
        "SYM8.AX",
        "SYM9.L",
        "SYM10.SW",
        "SYM11.TA",
        "SYM12.T",
        "SYM13.HK",
        "SYM14.TO",
        "SYM15.BR",
        "SYM16.LS",
        "SYM17.SA",
        "SYM18.MX",
        "SYM19.ST",
        "SYM20.OL",
        "SYM21.J",
        "SYM22.KS",
        "SYM23.SS",
        "SYM24.SZ",
        "SYM25",
    ]:
        assert expected in symbols


def test_download_request_pipeline_is_executed():
    """Test download request pipeline is executed."""
    csv = (
        "Asset Class,Ticker,Name,Location,Exchange\n"
        "Equity,AAPL,Apple Inc,United States,Nasdaq"
    )
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = csv

    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=pl.DataFrame(
                {"index": ["S&P 500"], "url": ["mock-url"], "skip": [0]}
            ),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="test-agent",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ) as mock_get,
    ):
        out = _download_data_constituents("S&P 500")

    assert "AAPL" in out["symbol"].to_list()
    mock_get.assert_called()


def test_header_row_is_detected_after_preamble():
    """Test the header row is found without a fixed skip count."""
    csv = "\n".join(
        [
            "Fondspositionen per,30.09.2026",
            "",
            "Anlageklasse,Emittententicker,Name,Standort,Börse",
            "Aktien,ABC,Alpha AG,Germany,Xetra",
        ]
    )
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = csv

    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df("DAX"),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ),
    ):
        out = _download_data_constituents("DAX")

    assert out["symbol"].to_list() == ["ABC.DE"]


def test_path_reads_ishares_file_without_network(tmp_path):
    """Test 'path' reads a local iShares file and makes no request."""
    file = tmp_path / "holdings.csv"
    file.write_text(
        "Preamble\n"
        "Anlageklasse,Emittententicker,Name,Standort,Börse\n"
        "Aktien,ABC,Alpha AG,Germany,Xetra\n"
        "Aktien,DAX,DAX INDEX,Germany,Xetra\n",
        encoding="utf-8",
    )
    with patch("tidyfinance.download_open_source.requests.get") as mock_get:
        out = _download_data_constituents(index="DAX", path=file)

    mock_get.assert_not_called()
    assert out["symbol"].to_list() == ["ABC.DE"]
    assert out.columns == ["symbol", "name", "location", "exchange", "currency"]


def test_path_without_index_skips_index_name_filter(tmp_path):
    """Test 'path' works without 'index' and keeps index-named rows."""
    file = tmp_path / "holdings.csv"
    file.write_text(
        "Anlageklasse,Emittententicker,Name,Standort,Börse\n"
        "Aktien,ABC,Alpha AG,Germany,Xetra\n"
        "Aktien,DAX,DAX INDEX,Germany,Xetra\n",
        encoding="utf-8",
    )
    out = _download_data_constituents(path=file)

    assert out["symbol"].to_list() == ["ABC.DE", "DAX.DE"]


def test_path_reads_latin1_file(tmp_path):
    """Test a Latin-1 encoded holdings file is decoded."""
    file = tmp_path / "holdings.csv"
    file.write_bytes(
        (
            "Anlageklasse,Emittententicker,Name,Standort,Börse\n"
            "Aktien,ABC,Alpha AG,Germany,Xetra\n"
        ).encode("latin-1")
    )
    out = _download_data_constituents(path=file)

    assert out["symbol"].to_list() == ["ABC.DE"]


def test_unknown_file_layout_fails(tmp_path):
    """Test a file without an iShares header row fails."""
    file = tmp_path / "other.csv"
    file.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="Unknown column format"):
        _download_data_constituents(path=file)


def test_index_or_path_is_required():
    """Test calling without 'index' and 'path' fails."""
    with pytest.raises(ValueError, match="'index'.*'path'"):
        _download_data_constituents()


def test_failed_download_points_to_path_fallback():
    """Test the download error message mentions the 'path' fallback."""
    response_mock = MagicMock()
    response_mock.status_code = 404
    with (
        patch(
            "tidyfinance.download_open_source.list_supported_indexes",
            return_value=_supported_indexes_df(),
        ),
        patch(
            "tidyfinance.download_open_source._get_random_user_agent",
            return_value="ua",
        ),
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response_mock,
        ),
    ):
        with pytest.raises(ValueError, match="path="):
            _download_data_constituents("DAX")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
