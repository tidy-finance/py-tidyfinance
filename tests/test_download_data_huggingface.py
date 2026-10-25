"""Tests for download_data_huggingface and related helpers."""

import datetime as dt
import io
import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import tidyfinance.download_tidy_finance as tfhf  # noqa: E402
from tidyfinance.download_tidy_finance import (  # noqa: E402
    download_data_huggingface,
    _download_data_huggingface_factor_library,
    download_factor_library_grid,
    download_factor_library_ids,
    _download_factor_library_persistence,
    _download_factor_library_table,
    _factor_library_cache,
    _factor_library_file,
    _factor_library_url,
    _fetch_parquet_url,
    _filter_factor_library_grid,
    get_available_huggingface_files,
    _rate_limit_reset,
    _RateLimited,
    _read_factor_library_files,
    _Throttle,
    _wait_for_rate_limit,
)


@pytest.fixture(autouse=True)
def _empty_factor_library_cache():
    """Empty the session cache of the factor library grid for each test,
    so a grid mocked in one test does not leak into the next, and
    restore its previous content afterwards."""
    old = dict(_factor_library_cache)
    _factor_library_cache.clear()
    yield
    _factor_library_cache.clear()
    _factor_library_cache.update(old)


# Grid columns pinned to their dtypes on Hugging Face: polars would infer
# Int64 from the integer literals below, and no dtype at all for the
# all-null columns.
_GRID_DTYPES = {
    "id": pl.Int32,
    "n_portfolios_main": pl.Float64,
    "n_portfolios_secondary": pl.Float64,
    "breakpoints_min_size_threshold": pl.Float64,
}

_FACTOR_LIBRARY_URL = (
    "https://huggingface.co/datasets/tidy-finance/factor-library/resolve/main"
)


def _make_grid(ids=(1,)):
    """Build a grid with one default univariate row per id."""
    row = {
        "sorting_variable": "size",
        "sorting_variable_lag": "6m",
        "sorting_method": "univariate",
        "n_portfolios_main": 10,
        "min_size_quantile": 0.2,
        "exclude_financials": False,
        "exclude_utilities": False,
        "exclude_negative_earnings": False,
        "rebalancing": "monthly",
        "n_portfolios_secondary": None,
        "breakpoints_exchanges": "NYSE",
        "breakpoints_min_size_threshold": None,
        "weighting_scheme": "VW",
    }
    return pl.DataFrame(
        {"id": list(ids), **{col: [v] * len(ids) for col, v in row.items()}},
        schema_overrides=_GRID_DTYPES,
    )


def _make_returns(ids, rets):
    """Build a returns file with the columns and dtypes used on the Hub."""
    return pl.DataFrame(
        {"id": ids, "date": [dt.date(2020, 1, 1)] * len(ids), "ret": rets},
        schema={"id": pl.Int32, "date": pl.Date, "ret": pl.Float32},
    )


def _serve_files(files, requested):
    """Stand in for '_fetch_parquet_url', serving 'files' by file name.

    Records each requested URL and returns None, as for a missing file
    (HTTP 404), for names not in 'files'.
    """

    def fetch(url):
        requested.append(url)
        return files.get(url.rsplit("/", 1)[-1])

    return fetch


def _available(paths, sizes=None):
    """Build an available-files frame, typed even when empty."""
    if not paths:
        return pl.DataFrame(schema={"path": pl.String, "size": pl.Int64})
    return pl.DataFrame(
        {
            "path": paths,
            "size": sizes if sizes is not None else [100] * len(paths),
        }
    )


# %% get_available_huggingface_files


def test_single_page_returns_only_parquet_files():
    """Test single page: returns only parquet files."""
    page = [
        {"type": "file", "path": "data.parquet", "size": 100},
        {"type": "file", "path": "readme.txt", "size": 10},
        {"type": "directory", "path": "subdir", "size": 0},
    ]
    response_mock = MagicMock()
    response_mock.json.return_value = page
    response_mock.headers = {"Link": ""}
    response_mock.raise_for_status = MagicMock()

    with patch(
        "tidyfinance.download_tidy_finance.requests.get",
        return_value=response_mock,
    ):
        result = get_available_huggingface_files("org", "ds")

    assert len(result) == 1
    assert result.columns == ["path", "size"]
    assert result["path"][0] == "data.parquet"


def test_multi_page_paginates_until_rel_next_link_absent():
    """Test multi-page: paginates until rel=next link absent."""
    page1 = [{"type": "file", "path": "a.parquet", "size": 1}]
    page2 = [{"type": "file", "path": "b.parquet", "size": 2}]

    responses = []
    r1 = MagicMock()
    r1.json.return_value = page1
    r1.headers = {"Link": '<https://page2.example.com>; rel="next"'}
    r1.raise_for_status = MagicMock()
    responses.append(r1)

    r2 = MagicMock()
    r2.json.return_value = page2
    r2.headers = {"Link": ""}
    r2.raise_for_status = MagicMock()
    responses.append(r2)

    with patch(
        "tidyfinance.download_tidy_finance.requests.get", side_effect=responses
    ):
        result = get_available_huggingface_files("org", "ds")

    assert len(result) == 2


# %% download_data_huggingface


def test_aborts_when_dataset_is_none():
    """Test aborts when dataset is NULL."""
    with pytest.raises(ValueError):
        download_data_huggingface(dataset=None)


def test_deprecated_type_arg_warns_and_strips_hf_prefix():
    """Test deprecated type arg warns and strips hf_ prefix."""
    with patch(
        "tidyfinance.download_tidy_finance.download_factor_library_grid",
        return_value=pl.DataFrame(),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            download_data_huggingface(type="hf_factor_library_grid")


def test_legacy_hf_dataset_value_warns_and_strips_prefix():
    """Test legacy hf_ dataset value warns and strips prefix."""
    with patch(
        "tidyfinance.download_tidy_finance._download_data_huggingface_factor_library",
        return_value=pl.DataFrame(schema={"id": pl.Int64}),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            try:
                download_data_huggingface(dataset="hf_factor_library")
            except ValueError:
                pass


def test_aborts_for_unsupported_dataset():
    """Test aborts for unsupported dataset."""
    with pytest.raises(ValueError):
        download_data_huggingface(dataset="unknown")


def test_factor_library_grid_delegates_to_helper():
    """Test factor_library_grid: delegates to helper."""
    mock_grid = pl.DataFrame({"id": [1]})
    with patch(
        "tidyfinance.download_tidy_finance.download_factor_library_grid",
        return_value=mock_grid,
        create=True,
    ):
        result = download_data_huggingface("factor_library_grid")
    assert_frame_equal(result, mock_grid)


def test_high_frequency_sp500_filters_by_date_and_downloads():
    """Test high_frequency_sp500: filters by date and downloads."""
    available = _available(["date=2007-07-26/part.parquet"])
    mock_trades = pl.DataFrame({"price": [100.0]})

    with (
        patch(
            "tidyfinance.download_tidy_finance.get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=mock_trades,
        ),
    ):
        result = download_data_huggingface(
            "high_frequency_sp500", "2007-07-26", "2007-07-26"
        )

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 1


def test_factor_library_delegates_to_inner_helper():
    """Test factor_library: delegates to inner helper."""
    mock_returns = pl.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.download_tidy_finance._download_data_huggingface_factor_library",
        return_value=mock_returns,
    ):
        result = download_data_huggingface(
            "factor_library", sorting_variable="size"
        )
    assert_frame_equal(result, mock_returns)


def test_factor_library_forwards_start_date_and_end_date():
    """Test factor_library: forwards start_date and end_date."""
    captured = {}

    def fake_inner(**kwargs):
        captured.update(kwargs)
        return pl.DataFrame({"id": [1]})

    with patch(
        "tidyfinance.download_tidy_finance._download_data_huggingface_factor_library",
        side_effect=fake_inner,
    ):
        download_data_huggingface(
            "factor_library",
            sorting_variable="size",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert captured.get("sorting_variable") == "size"
    assert captured.get("start_date") == "2020-01-01"
    assert captured.get("end_date") == "2020-12-31"


def test_high_frequency_sp500_uses_sample_window_when_no_dates():
    """Test high_frequency_sp500: uses sample window when no dates."""
    available = _available(["date=2007-06-27/part.parquet"])
    with (
        patch(
            "tidyfinance.download_tidy_finance.get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=pl.DataFrame({"price": [100.0]}),
        ),
    ):
        result = download_data_huggingface("high_frequency_sp500")
    assert len(result) == 1


# %% _filter_factor_library_grid


def test_aborts_for_unsupported_filter_name():
    """Test aborts for unsupported filter name."""
    with pytest.raises(ValueError):
        _filter_factor_library_grid(bad_col="x")


def test_aborts_non_univariate_sort_without_secondary_n():
    """Test aborts: non-univariate sort without secondary n."""
    with pytest.raises(ValueError):
        _filter_factor_library_grid(
            sorting_variable="size", sorting_method="bivariate-dependent"
        )


def test_sorting_variable_optional_returns_all_with_defaults():
    """Test sorting_variable omitted: all sorting variables, defaults."""
    grid = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "sorting_variable": ["size", "bm", "size"],
            "min_size_quantile": [0.2, 0.2, 0.4],
            "exclude_financials": [False, False, False],
            "exclude_utilities": [False, False, False],
            "exclude_negative_earnings": [False, False, False],
            "sorting_variable_lag": ["6m", "6m", "6m"],
            "rebalancing": ["monthly", "monthly", "monthly"],
            "n_portfolios_main": [10, 10, 10],
            "sorting_method": ["univariate", "univariate", "univariate"],
            "n_portfolios_secondary": [None, None, None],
            "breakpoints_exchanges": ["NYSE", "NYSE", "NYSE"],
            "breakpoints_min_size_threshold": [None, None, None],
            "weighting_scheme": ["VW", "VW", "VW"],
        },
        schema_overrides=_GRID_DTYPES,
    )
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=grid,
    ):
        ids = _filter_factor_library_grid()

    # Both sorting variables returned; the non-default row (id 3) dropped.
    assert ids == [1, 2]


def test_explicit_none_removes_filter_returning_all_values():
    """Test passing None for a column returns all values for it."""
    grid = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "sorting_variable": ["size", "size", "size"],
            "min_size_quantile": [0.2, 0.4, 0.6],
            "exclude_financials": [False, False, False],
            "exclude_utilities": [False, False, False],
            "exclude_negative_earnings": [False, False, False],
            "sorting_variable_lag": ["6m", "6m", "6m"],
            "rebalancing": ["monthly", "monthly", "monthly"],
            "n_portfolios_main": [10, 10, 10],
            "sorting_method": ["univariate", "univariate", "univariate"],
            "n_portfolios_secondary": [None, None, None],
            "breakpoints_exchanges": ["NYSE", "NYSE", "NYSE"],
            "breakpoints_min_size_threshold": [None, None, None],
            "weighting_scheme": ["VW", "VW", "VW"],
        },
        schema_overrides=_GRID_DTYPES,
    )
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=grid,
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", min_size_quantile=None
        )

    # The default 0.2 screen is removed, so all size groups are returned.
    assert ids == [1, 2, 3]


def test_list_with_none_selects_the_missing_level():
    """Test [None] selects the rows where a column is missing (R's NA)."""
    grid = _make_grid([1, 2]).with_columns(
        pl.Series("min_size_quantile", [None, 0.2])
    )
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=grid,
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", min_size_quantile=[None]
        )

    # None would drop the filter and return both rows.
    assert ids == [1]


def test_fill_all_false_defaults_applied_row_filtered_out():
    """Test fill_all = FALSE: defaults applied, row filtered out."""
    grid = pl.DataFrame(
        {
            "id": [1, 2],
            "sorting_variable": ["size", "size"],
            "min_size_quantile": [0.2, 0.4],
            "exclude_financials": [False, False],
            "exclude_utilities": [False, False],
            "exclude_negative_earnings": [False, False],
            "sorting_variable_lag": ["6m", "6m"],
            "rebalancing": ["monthly", "monthly"],
            "n_portfolios_main": [10, 10],
            "sorting_method": ["univariate", "univariate"],
            "n_portfolios_secondary": [None, None],
            "breakpoints_exchanges": ["NYSE", "NYSE"],
            "breakpoints_min_size_threshold": [None, None],
            "weighting_scheme": ["VW", "VW"],
        },
        schema_overrides=_GRID_DTYPES,
    )
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=grid,
    ):
        ids = _filter_factor_library_grid(sorting_variable="size")

    assert ids == [1]


def test_fill_all_true_only_explicit_filters_applied():
    """Test fill_all = TRUE: only explicit filters applied."""
    grid = pl.DataFrame(
        {
            "id": [1, 2],
            "sorting_variable": ["size", "bm"],
            "min_size_quantile": [0.2, 0.2],
            "exclude_financials": [False, False],
            "exclude_utilities": [False, False],
            "exclude_negative_earnings": [False, False],
            "sorting_variable_lag": ["6m", "6m"],
            "rebalancing": ["monthly", "monthly"],
            "n_portfolios_main": [10, 10],
            "sorting_method": ["univariate", "univariate"],
            "n_portfolios_secondary": [None, None],
            "breakpoints_exchanges": ["NYSE", "NYSE"],
            "breakpoints_min_size_threshold": [None, None],
            "weighting_scheme": ["EW", "VW"],
        },
        schema_overrides=_GRID_DTYPES,
    )
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=grid,
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", fill_all=True
        )

    assert ids == [1]


# %% download_factor_library_grid (no direct Python equivalent)


def test_reads_the_grid_file_by_name():
    """Test the grid is read from portfolio_sort_grid.parquet directly."""
    mock_grid = pl.DataFrame({"id": [1]})
    with (
        patch(
            "tidyfinance.download_tidy_finance.get_available_huggingface_files"
        ) as listing,
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=mock_grid,
        ) as read,
    ):
        result = download_factor_library_grid()

    assert_frame_equal(result, mock_grid)
    read.assert_called_once_with(
        "https://huggingface.co/datasets/tidy-finance/factor-library-grid/"
        "resolve/main/portfolio_sort_grid.parquet"
    )
    listing.assert_not_called()


def _counting_grid_reader():
    """Return a fake '_fetch_parquet_url' that serves a new grid on each
    call (id 1, then id 2, ...) and counts the calls."""
    calls = {"n": 0}

    def read(url):
        calls["n"] += 1
        return _make_grid([calls["n"]])

    return read, calls


def test_caches_the_grid_within_the_session():
    """Test the grid is downloaded once and then reused."""
    read, calls = _counting_grid_reader()
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url", side_effect=read
    ):
        first = download_factor_library_grid()
        second = download_factor_library_grid()

    assert calls["n"] == 1
    assert second is first


def test_refresh_downloads_the_grid_again():
    """Test refresh=True downloads the grid again and caches the new
    copy."""
    read, calls = _counting_grid_reader()
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url", side_effect=read
    ):
        download_factor_library_grid()
        refreshed = download_factor_library_grid(refresh=True)
        cached = download_factor_library_grid()

    assert calls["n"] == 2
    assert refreshed["id"].to_list() == [2]
    assert cached is refreshed


def test_does_not_cache_a_failed_download():
    """Test a failed download leaves the cache empty."""
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=ConnectionError("network down"),
    ):
        with pytest.raises(ConnectionError, match="network down"):
            download_factor_library_grid()

    assert not _factor_library_cache.get("grids")


def test_aborts_on_an_invalid_refresh_argument():
    """Test refresh must be True or False."""
    for value in (None, "yes", 1):
        with pytest.raises(ValueError, match="refresh"):
            download_factor_library_grid(refresh=value)


def test_download_factor_library_ids_reuses_the_cached_grid():
    """Test repeated ID downloads download the grid only once."""
    n_grid_downloads = {"n": 0}

    def fetch(url):
        if url.endswith("portfolio_sort_grid.parquet"):
            n_grid_downloads["n"] += 1
            return _make_grid([1, 2])
        return _make_returns([1, 2], [0.0, 0.0])

    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=fetch,
    ):
        download_factor_library_ids([1])
        download_factor_library_ids([2])

    assert n_grid_downloads["n"] == 1


def test_factor_library_grid_dataset_passes_refresh():
    """Test download_data('Tidy Finance', 'factor_library_grid',
    refresh=True) downloads the grid again."""
    read, calls = _counting_grid_reader()
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url", side_effect=read
    ):
        download_data_huggingface("factor_library_grid")
        refreshed = download_data_huggingface(
            "factor_library_grid", refresh=True
        )

    assert calls["n"] == 2
    assert refreshed["id"].to_list() == [2]


# %% _factor_library_file


def test_factor_library_file_names_the_1000_id_file_of_each_id():
    """Test factor_library_file names the 1,000-id file of each id."""
    assert [_factor_library_file(i) for i in (1, 1000, 1001, 4105728)] == [
        "id_0000001-0001000.parquet",
        "id_0000001-0001000.parquet",
        "id_0001001-0002000.parquet",
        "id_4105001-4106000.parquet",
    ]


# %% _fetch_parquet_url


def _response(status_code, content=b"", headers=None):
    response = MagicMock(status_code=status_code, content=content)
    response.headers = headers or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = RuntimeError(
            f"HTTP {status_code}"
        )
    return response


def test_fetch_parquet_url_parses_the_parquet_payload():
    """Test the payload of a successful response is parsed."""
    expected = _make_returns([1], [0.01])
    payload = io.BytesIO()
    expected.write_parquet(payload)
    with patch("tidyfinance.download_tidy_finance._hf_session") as session:
        session.get.return_value = _response(200, payload.getvalue())
        result = _fetch_parquet_url("https://example.com/f.parquet")
    assert_frame_equal(result, expected)


def test_fetch_parquet_url_returns_none_for_a_missing_file():
    """Test a missing file (HTTP 404) returns None without retrying."""
    with patch("tidyfinance.download_tidy_finance._hf_session") as session:
        session.get.return_value = _response(404)
        assert _fetch_parquet_url("https://example.com/f.parquet") is None
    session.get.assert_called_once()


def test_fetch_parquet_url_retries_other_errors_then_raises():
    """Test other HTTP errors are retried, then raise ConnectionError."""
    with patch("tidyfinance.download_tidy_finance._hf_session") as session:
        session.get.return_value = _response(503)
        with pytest.raises(ConnectionError):
            _fetch_parquet_url(
                "https://example.com/f.parquet", retries=3, backoff=0
            )
    assert session.get.call_count == 3


def _float_declared_as_double():
    """Build parquet bytes like the factor library files written with
    nanoparquet: 'ret' is stored as float but declared as double in the
    embedded Arrow schema."""
    import base64

    import pyarrow as pa
    import pyarrow.parquet as pq

    declared = pa.schema(
        [("id", pa.int32()), ("date", pa.date32()), ("ret", pa.float64())]
    )
    arrow_schema = base64.b64encode(declared.serialize().to_pybytes())
    table = pa.table(
        {
            "id": pa.array([1, 2], pa.int32()),
            "date": pa.array([dt.date(2020, 1, 1)] * 2, pa.date32()),
            "ret": pa.array([0.01, -0.02], pa.float32()),
        }
    )
    payload = io.BytesIO()
    # store_schema=False drops pyarrow's own (correct) Arrow schema; the
    # mismatched one is then added to the file footer by hand.
    with pq.ParquetWriter(payload, table.schema, store_schema=False) as writer:
        writer.write_table(table)
        writer.add_key_value_metadata({"ARROW:schema": arrow_schema.decode()})

    # The fixture must carry the mismatch it is meant to reproduce.
    stored = pq.read_metadata(io.BytesIO(payload.getvalue())).metadata
    assert stored[b"ARROW:schema"] == arrow_schema
    return payload.getvalue()


def test_fetch_parquet_url_reads_float_declared_as_double():
    """Test files whose Arrow schema declares a float column as double,
    as written by nanoparquet, are read with the stored type."""
    with patch("tidyfinance.download_tidy_finance._hf_session") as session:
        session.get.return_value = _response(200, _float_declared_as_double())
        result = _fetch_parquet_url("https://example.com/f.parquet")

    assert result["id"].to_list() == [1, 2]
    assert result["ret"].to_list() == pytest.approx([0.01, -0.02])


def test_fetch_parquet_url_does_not_retry_a_file_it_cannot_parse():
    """Test a downloaded file that cannot be parsed raises right away
    instead of being downloaded again."""
    with patch("tidyfinance.download_tidy_finance._hf_session") as session:
        session.get.return_value = _response(200, b"not a parquet file")
        with pytest.raises(Exception) as err:
            _fetch_parquet_url(
                "https://example.com/f.parquet", retries=3, backoff=0
            )

    assert session.get.call_count == 1
    assert not isinstance(err.value, ConnectionError)


# %% download_factor_library_ids


def test_aborts_when_ids_are_empty():
    """Test aborts when no ids are passed, before any download."""
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid"
        ) as grid,
        pytest.raises(ValueError, match="No portfolio IDs"),
    ):
        download_factor_library_ids([])
    grid.assert_not_called()


def test_aborts_when_no_grid_rows_match_requested_ids():
    """Test aborts when no grid rows match requested ids."""
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([42]),
        ),
        patch("tidyfinance.download_tidy_finance._fetch_parquet_url") as fetch,
        pytest.raises(ValueError, match="None of the requested"),
    ):
        download_factor_library_ids([999])
    fetch.assert_not_called()


def test_downloads_the_files_that_hold_the_ids_and_joins_grid_metadata():
    """Test downloads the files that hold the ids and joins grid metadata."""
    files = {
        "id_0000001-0001000.parquet": _make_returns(
            [1, 2, 3], [0.01, 0.02, 0.99]
        ),
        "id_0001001-0002000.parquet": _make_returns([1001], [0.03]),
    }
    requested = []
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1, 2, 3, 1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, requested),
        ),
    ):
        result = download_factor_library_ids([1, 2, 1001])

    # One download per file, although the first file holds two of the ids.
    assert sorted(requested) == [f"{_FACTOR_LIBRARY_URL}/{f}" for f in files]
    assert result.columns[:3] == ["id", "date", "ret"]
    assert result["id"].to_list() == [1, 2, 1001]
    assert result["ret"].to_list() == pytest.approx([0.01, 0.02, 0.03])
    assert result.schema["ret"] == pl.Float64
    assert "weighting_scheme" in result.columns


def test_ids_without_returns_are_absent_with_a_warning():
    """Test ids without returns are dropped with a warning."""
    # id 2 has no rows in its file, and the range of id 1001 has no file
    # at all, as when none of the sorts in the range produced portfolios.
    files = {"id_0000001-0001000.parquet": _make_returns([1], [0.01])}
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1, 2, 1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, []),
        ),
        pytest.warns(UserWarning, match="without returns.*: 2, 1001"),
    ):
        result = download_factor_library_ids([1, 2, 1001])

    assert result["id"].to_list() == [1]


def test_returns_empty_frame_when_no_requested_id_has_returns():
    """Test an empty frame with all columns when no id has returns."""
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=None,
        ),
        pytest.warns(UserWarning, match="without returns"),
    ):
        result = download_factor_library_ids([1001])

    assert result.is_empty()
    assert result.columns == ["id", "date", "ret", *_make_grid().columns[1:]]


# %% _download_data_huggingface_factor_library


def test_aborts_when_ids_and_filter_args_are_combined():
    """Test aborts when ids and filter args are combined."""
    with pytest.raises(ValueError):
        _download_data_huggingface_factor_library(
            sorting_variable="size", ids=[1]
        )


def test_with_ids_delegates_to_download_factor_library_table():
    """Test with ids: delegates to download_factor_library_table."""
    mock_result = pl.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_table",
        return_value=mock_result,
    ):
        result = _download_data_huggingface_factor_library(ids=[1])
    assert_frame_equal(result, mock_result)


def test_without_ids_resolves_via_grid_then_downloads():
    """Test without ids: resolves via grid then downloads."""
    mock_result = pl.DataFrame({"id": [1], "ret": [0.01]})
    with (
        patch(
            "tidyfinance.download_tidy_finance._filter_factor_library_grid",
            return_value=[1],
        ),
        patch(
            "tidyfinance.download_tidy_finance._download_factor_library_table",
            return_value=mock_result,
        ),
    ):
        result = _download_data_huggingface_factor_library(
            sorting_variable="size"
        )
    assert_frame_equal(result, mock_result)


def test_filters_returns_to_the_requested_date_range():
    """Test filters returns to the requested date range."""
    mock_returns = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "date": [
                dt.date(2019, 12, 31),
                dt.date(2020, 6, 30),
                dt.date(2021, 1, 31),
            ],
            "ret": [0.01, 0.02, 0.03],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_table",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface_factor_library(
            ids=[1], start_date="2020-01-01", end_date="2020-12-31"
        )
    assert result["date"][0] == dt.date(2020, 6, 30)


def test_returns_full_history_when_dates_omitted():
    """Test returns full history when dates omitted."""
    mock_returns = pl.DataFrame(
        {
            "id": [1, 1],
            "date": [dt.date(2019, 12, 31), dt.date(2020, 6, 30)],
            "ret": [0.01, 0.02],
        }
    )
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_table",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface_factor_library(ids=[1])
    assert_frame_equal(result, mock_returns)


def test_date_filtering_also_applies_on_the_grid_resolved_path():
    """Test date filtering also applies on the grid-resolved path."""
    mock_returns = pl.DataFrame(
        {
            "id": [1, 1],
            "date": [dt.date(2018, 1, 31), dt.date(2020, 6, 30)],
            "ret": [0.01, 0.02],
        }
    )
    with (
        patch(
            "tidyfinance.download_tidy_finance._filter_factor_library_grid",
            return_value=[1],
        ),
        patch(
            "tidyfinance.download_tidy_finance._download_factor_library_table",
            return_value=mock_returns,
        ),
    ):
        result = _download_data_huggingface_factor_library(
            sorting_variable="size",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert result["date"][0] == dt.date(2020, 6, 30)


# %% factor library revisions, costs, persistence, and rate limit


def _make_costs(ids):
    """Build a costs file with the columns and dtypes used on the Hub."""
    n = len(ids)
    return pl.DataFrame(
        {
            "id": ids,
            "date": [dt.date(2020, 1, 1)] * n,
            "turnover": [0.1] * n,
            "cost": [0.002] * n,
            "microcap_share": [0.05] * n,
        },
        schema={
            "id": pl.Int32,
            "date": pl.Date,
            "turnover": pl.Float32,
            "cost": pl.Float32,
            "microcap_share": pl.Float32,
        },
    )


def _make_persistence():
    return pl.DataFrame(
        {
            "sorting_variable": ["size", "size", "bm"],
            "date": [
                dt.date(2019, 1, 1),
                dt.date(2020, 1, 1),
                dt.date(2020, 1, 1),
            ],
            "persistence": [0.9, 0.8, 0.7],
        }
    )


def test_factor_library_costs_delegates_with_the_costs_table():
    with patch(
        "tidyfinance.download_tidy_finance."
        "_download_data_huggingface_factor_library",
        return_value=pl.DataFrame({"id": [1]}),
    ) as inner:
        download_data_huggingface(
            "factor_library_costs", sorting_variable="size"
        )
    assert inner.call_args.kwargs["table"] == "costs"
    assert inner.call_args.kwargs["sorting_variable"] == "size"


def test_factor_library_grid_forwards_refresh_and_grid_revision():
    with patch(
        "tidyfinance.download_tidy_finance.download_factor_library_grid",
        return_value=_make_grid(),
    ) as grid:
        download_data_huggingface(
            "factor_library_grid", refresh=True, grid_revision="v1"
        )
    grid.assert_called_once_with(refresh=True, grid_revision="v1")


def test_factor_library_persistence_delegates_with_dates():
    with patch(
        "tidyfinance.download_tidy_finance."
        "_download_factor_library_persistence",
        return_value=_make_persistence(),
    ) as persistence:
        download_data_huggingface(
            "factor_library_persistence",
            start_date="2020-01-01",
            end_date="2020-12-31",
            sorting_variable="size",
        )
    persistence.assert_called_once_with(
        start_date="2020-01-01",
        end_date="2020-12-31",
        sorting_variable="size",
    )


@pytest.mark.parametrize(
    "dataset", ["factor_library_grid", "factor_library_persistence"]
)
def test_grid_and_persistence_reject_revision(dataset):
    with pytest.raises(ValueError, match="'revision' does not apply"):
        download_data_huggingface(dataset, revision="abc")


def test_filters_the_grid_of_the_requested_revision():
    requested = []

    def fetch(url):
        requested.append(url)
        return _make_grid([1])

    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=fetch,
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", grid_revision="v1"
        )
    assert ids == [1]
    assert requested == [
        "https://huggingface.co/datasets/tidy-finance/factor-library-grid/"
        "resolve/v1/portfolio_sort_grid.parquet"
    ]


def test_caches_one_grid_per_revision():
    n = {"n": 0}

    def fetch(url):
        n["n"] += 1
        return _make_grid([n["n"]])

    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=fetch,
    ):
        main = download_factor_library_grid()
        v1 = download_factor_library_grid(grid_revision="v1")
        assert download_factor_library_grid() is main
        assert download_factor_library_grid(grid_revision="v1") is v1
    assert n["n"] == 2


def test_aborts_when_the_revision_has_no_grid():
    with (
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=None,
        ),
        pytest.raises(ValueError, match="is not in revision 'v1'"),
    ):
        download_factor_library_grid(grid_revision="v1")


@pytest.mark.parametrize("value", [None, "", 1, ["main"]])
def test_aborts_on_an_invalid_revision(value):
    with pytest.raises(ValueError, match="grid_revision"):
        download_factor_library_grid(grid_revision=value)
    with pytest.raises(ValueError, match="'revision' must be"):
        download_factor_library_ids([1], revision=value)
    with pytest.raises(ValueError, match="grid_revision"):
        _download_factor_library_persistence(grid_revision=value)


def test_reads_the_returns_and_the_grid_at_their_revisions():
    requested = []

    def fetch(url):
        requested.append(url)
        if url.endswith("portfolio_sort_grid.parquet"):
            return _make_grid([1])
        return _make_returns([1], [0.01])

    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=fetch,
    ):
        download_factor_library_ids([1], revision="r1", grid_revision="g1")
    base = "https://huggingface.co/datasets/tidy-finance/"
    assert requested == [
        f"{base}factor-library-grid/resolve/g1/portfolio_sort_grid.parquet",
        f"{base}factor-library/resolve/r1/id_0000001-0001000.parquet",
    ]


def test_costs_reads_the_files_of_the_same_names_in_costs():
    requested = []
    files = {"id_0000001-0001000.parquet": _make_costs([1, 2])}
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1, 2]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, requested),
        ),
    ):
        result = _download_factor_library_table([1], "costs", "main", "main")
    assert requested == [
        f"{_FACTOR_LIBRARY_URL}/costs/id_0000001-0001000.parquet"
    ]
    assert result.columns[:5] == [
        "id",
        "date",
        "turnover",
        "cost",
        "microcap_share",
    ]
    assert result["id"].to_list() == [1]
    assert result.schema["cost"] == pl.Float64
    assert "weighting_scheme" in result.columns


def test_costs_warn_about_ids_without_costs():
    files = {"id_0000001-0001000.parquet": _make_costs([1])}
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1, 2]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, []),
        ),
        pytest.warns(UserWarning, match="without costs.*: 2"),
    ):
        result = _download_factor_library_table([1, 2], "costs", "main", "main")
    assert result["id"].to_list() == [1]


def test_costs_return_no_rows_but_all_columns_when_no_id_has_costs():
    with (
        patch(
            "tidyfinance.download_tidy_finance.download_factor_library_grid",
            return_value=_make_grid([1]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=None,
        ),
        pytest.warns(UserWarning, match="without costs"),
    ):
        result = _download_factor_library_table([1], "costs", "main", "main")
    assert result.is_empty()
    assert result.columns[:5] == [
        "id",
        "date",
        "turnover",
        "cost",
        "microcap_share",
    ]


def test_persistence_reads_the_file_of_the_grid_dataset():
    requested = []

    def fetch(url):
        requested.append(url)
        return _make_persistence()

    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=fetch,
    ):
        result = _download_factor_library_persistence(grid_revision="g1")
    assert requested == [
        "https://huggingface.co/datasets/tidy-finance/factor-library-grid/"
        "resolve/g1/sorting_variable_persistence.parquet"
    ]
    assert result.height == 3


def test_persistence_filters_by_sorting_variable_and_dates():
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        return_value=_make_persistence(),
    ):
        result = _download_factor_library_persistence(
            sorting_variable="size",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert result["sorting_variable"].to_list() == ["size"]
    assert result["date"].to_list() == [dt.date(2020, 1, 1)]


def test_persistence_aborts_for_unknown_sorting_variables():
    with (
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=_make_persistence(),
        ),
        pytest.raises(ValueError, match="Unknown sorting variable"),
    ):
        _download_factor_library_persistence(sorting_variable=["size", "x"])


def test_factor_library_url_resolves_paths_at_a_revision():
    assert _factor_library_url("factor-library", "refs/pr/1", "a.parquet") == (
        "https://huggingface.co/datasets/tidy-finance/factor-library/"
        "resolve/refs%2Fpr%2F1/a.parquet"
    )


def test_reads_the_files_in_order_and_skips_missing_files():
    files = {"b.parquet": _make_grid([2]), "a.parquet": _make_grid([1])}
    with patch(
        "tidyfinance.download_tidy_finance._fetch_parquet_url",
        side_effect=_serve_files(files, []),
    ):
        result = _read_factor_library_files(
            "factor-library", "main", ["a.parquet", "missing", "b.parquet"]
        )
    assert result[0]["id"].to_list() == [1]
    assert result[1] is None
    assert result[2]["id"].to_list() == [2]


def test_aborts_on_a_revision_that_the_dataset_does_not_have():
    with patch.object(tfhf, "_hf_session") as session:
        session.get.return_value = _response(
            404, headers={"X-Error-Code": "RevisionNotFound"}
        )
        with pytest.raises(
            ValueError,
            match="Revision 'nope' is not in 'tidy-finance/factor-library'",
        ):
            _read_factor_library_files("factor-library", "nope", ["a"])


def test_retries_files_refused_by_the_rate_limit_after_it_resets():
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited(5)
        return _make_grid([1])

    with (
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=fetch,
        ),
        patch("tidyfinance.download_tidy_finance._wait_for_rate_limit") as wait,
    ):
        result = _read_factor_library_files("factor-library", "main", ["a"])
    wait.assert_called_once_with(5)
    assert result[0]["id"].to_list() == [1]


def test_aborts_when_the_rate_limit_refuses_a_file_three_times():
    with (
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_RateLimited(7),
        ) as fetch,
        patch("tidyfinance.download_tidy_finance._wait_for_rate_limit") as wait,
        pytest.raises(RuntimeError, match="resets in 7 seconds"),
    ):
        _read_factor_library_files("factor-library", "main", ["a"])
    assert fetch.call_count == 3
    assert wait.call_count == 2


def test_fetch_parquet_url_raises_rate_limited_without_retrying():
    with patch.object(tfhf, "_hf_session") as session:
        session.get.return_value = _response(
            429, headers={"RateLimit": '"resolvers";r=0;t=41'}
        )
        with pytest.raises(_RateLimited) as err:
            _fetch_parquet_url("https://example.com/f.parquet", backoff=0)
    assert err.value.seconds == 42
    assert session.get.call_count == 1


def test_rate_limit_reset_reads_the_ratelimit_header():
    assert (
        _rate_limit_reset(
            _response(429, headers={"RateLimit": '"resolvers";r=0;t=82'})
        )
        == 83
    )
    assert _rate_limit_reset(_response(429)) == 301


def test_wait_for_rate_limit_says_how_long_it_waits(capsys):
    with patch("tidyfinance.download_tidy_finance.time.sleep") as sleep:
        _wait_for_rate_limit(3)
    sleep.assert_called_once_with(3)
    assert "Waiting 3 seconds" in capsys.readouterr().err


def test_downloads_are_throttled_across_calls():
    throttle = _Throttle(capacity=2, fill_time_s=1)
    with patch("tidyfinance.download_tidy_finance.time.sleep") as sleep:
        throttle.wait()
        throttle.wait()
        sleep.assert_not_called()
        # The bucket is empty; the third request waits for a token. The
        # mocked sleep does not refill it, so stop after the first wait.
        sleep.side_effect = lambda delay: setattr(throttle, "tokens", 1.0)
        throttle.wait()
    assert sleep.call_count == 1
    assert 0 < sleep.call_args.args[0] <= 0.5


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
