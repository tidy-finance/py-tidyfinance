"""Tests for download_data_stock_prices."""

import datetime as dt
import os
import sys
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tidyfinance as tf  # noqa: E402
from tidyfinance.download_open_source import (  # noqa: E402
    _download_data_stock_prices,
)


def test_symbols_must_be_a_character_vector_without_missing_values():
    """Test symbols must be a character vector without missing values."""
    with pytest.raises((ValueError, TypeError)):
        _download_data_stock_prices(1)

    with pytest.raises((ValueError, TypeError)):
        _download_data_stock_prices(["AAPL", None])


def test_downloads_data_replaces_null_and_warns_on_failures():
    """Test downloads data, replaces NULL, and warns on failures."""
    success_body = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577836800, 1577923200],
                    "indicators": {
                        "quote": [
                            {
                                "volume": [100, None],
                                "open": [10, 11],
                                "low": [9, 10],
                                "high": [12, 13],
                                "close": [11, 12],
                            }
                        ],
                        "adjclose": [{"adjclose": [11, None]}],
                    },
                }
            ]
        }
    }
    fail_body = {
        "chart": {
            "error": {
                "code": "Not Found",
                "description": "No data found",
            }
        }
    }

    def fake_get(url, *a, **kw):
        resp = MagicMock()
        if "FAIL" in url:
            resp.status_code = 404
            resp.json.return_value = fail_body
        else:
            resp.status_code = 200
            resp.json.return_value = success_body
        return resp

    with patch(
        "tidyfinance.download_open_source.requests.get", side_effect=fake_get
    ):
        with pytest.warns(
            UserWarning, match="Failed to retrieve data for symbol FAIL"
        ):
            out = _download_data_stock_prices(
                ["AAPL", "FAIL"], "2020-01-01", "2020-01-03"
            )

    assert isinstance(out, pl.DataFrame)
    # Two AAPL rows are kept; the row with a NULL upstream value
    # retains a null rather than being dropped. FAIL is silently
    # skipped.
    assert out["symbol"].to_list() == ["AAPL", "AAPL"]
    assert out["open"].to_list() == [10, 11]
    assert out["low"].to_list() == [9, 10]
    assert out["high"].to_list() == [12, 13]
    assert out["close"].to_list() == [11, 12]
    assert out["volume"][0] == 100
    assert out["volume"][1] is None
    assert out["adjusted_close"][0] == 11
    assert out["adjusted_close"][1] is None


def _chart_body(timestamps, meta):
    values = list(range(1, len(timestamps) + 1))
    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                column: values
                                for column in (
                                    "volume",
                                    "open",
                                    "low",
                                    "high",
                                    "close",
                                )
                            }
                        ],
                        "adjclose": [{"adjclose": values}],
                    },
                }
            ],
        },
    }


@pytest.mark.parametrize(
    "meta, expected_day",
    [
        ({"exchangeTimezoneName": "Australia/Sydney"}, 16),
        ({"exchangeTimezoneName": "Pacific/Auckland"}, 16),
        ({"exchangeTimezoneName": "America/New_York"}, 15),
        ({"exchangeTimezoneName": "UTC"}, 15),
        ({"exchangeTimezoneName": None}, 15),
        ({"exchangeTimezoneName": ""}, 15),
        ({}, 15),
        (None, 15),
    ],
)
def test_dates_use_exchange_timezone_with_utc_fallback(meta, expected_day):
    # Monday's Sydney open is still Sunday in UTC.
    body = _chart_body([1584313200], meta)
    with patch("tidyfinance.download_open_source.requests.get") as get:
        get.return_value.status_code = 200
        get.return_value.json.return_value = body
        out = _download_data_stock_prices("^AXJO", "2020-03-01", "2020-03-31")

    assert out["date"].to_list() == [dt.date(2020, 3, expected_day)]
    assert out.schema["date"] == pl.Date
    assert out["close"].to_list() == [1]


@pytest.mark.parametrize(
    "timezone, utc_hour",
    [
        ("Australia/Sydney", 23),
        ("Pacific/Auckland", 21),
        ("America/New_York", 14),
        ("UTC", 10),
    ],
)
@pytest.mark.parametrize(
    "start, end, expected_days",
    [(16, 18, [16, 17, 18]), (16, 16, [16]), (14, 15, [])],
)
def test_inclusive_local_range_and_buffered_request(
    timezone, utc_hour, start, end, expected_days
):
    # Include bars before and after the requested range. Sydney and
    # Auckland opens fall on the preceding UTC date.
    timestamps = [
        int(
            dt.datetime(
                2020,
                3,
                day - (utc_hour >= 21),
                utc_hour,
                tzinfo=dt.timezone.utc,
            ).timestamp()
        )
        for day in [13, 16, 17, 18, 19]
    ]
    body = _chart_body(timestamps, {"exchangeTimezoneName": timezone})
    start_date = dt.date(2020, 3, start)
    end_date = dt.date(2020, 3, end)
    with patch("tidyfinance.download_open_source.requests.get") as get:
        get.return_value.status_code = 200
        get.return_value.json.return_value = body
        out = _download_data_stock_prices("TEST", start_date, end_date)

    assert out["date"].to_list() == [
        dt.date(2020, 3, day) for day in expected_days
    ]
    assert out.schema["date"] == pl.Date
    query = parse_qs(urlparse(get.call_args.args[0]).query)
    for key, bound in [
        ("period1", start_date - dt.timedelta(days=2)),
        ("period2", end_date + dt.timedelta(days=2)),
    ]:
        assert int(query[key][0]) == int(
            dt.datetime.combine(
                bound, dt.time(), tzinfo=dt.timezone.utc
            ).timestamp()
        )


def test_exchange_timezone_observes_daylight_saving_changes():
    # Sydney's 10:00 open is 23:00 UTC in summer, 00:00 UTC in winter.
    timestamps = [1584313200, 1594771200]
    body = _chart_body(timestamps, {"exchangeTimezoneName": "Australia/Sydney"})
    with patch("tidyfinance.download_open_source.requests.get") as get:
        get.return_value.status_code = 200
        get.return_value.json.return_value = body
        out = _download_data_stock_prices("^AXJO", "2020-03-01", "2020-07-31")

    assert out["date"].to_list() == [dt.date(2020, 3, 16), dt.date(2020, 7, 15)]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_public_download_handles_each_symbols_timezone(backend):
    def fake_get(url, **kwargs):
        timezone = "Australia/Sydney" if "^AXJO" in url else "America/New_York"
        return MagicMock(
            status_code=200,
            json=lambda: _chart_body(
                [1584313200], {"exchangeTimezoneName": timezone}
            ),
        )

    previous_backend = tf.get_backend()
    try:
        tf.set_backend(backend)
        with patch(
            "tidyfinance.download_open_source.requests.get",
            side_effect=fake_get,
        ):
            out = tf.download_data(
                "Stock Prices",
                symbols=["^AXJO", "AAPL"],
                start_date="2020-03-15",
                end_date="2020-03-16",
            )
        if backend == "pandas":
            out = pl.from_pandas(out).with_columns(pl.col("date").cast(pl.Date))
        assert out["symbol"].to_list() == ["^AXJO", "AAPL"]
        assert out["date"].to_list() == [
            dt.date(2020, 3, 16),
            dt.date(2020, 3, 15),
        ]
    finally:
        tf.set_backend(previous_backend)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
