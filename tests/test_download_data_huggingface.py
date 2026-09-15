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

from tidyfinance.download_tidy_finance import (
    _download_data_huggingface,
    _download_data_huggingface_factor_library,
    _download_factor_library_grid,
    _download_factor_library_ids,
    _factor_library_file,
    _fetch_parquet_url,
    _filter_factor_library_grid,
    _get_available_huggingface_files,
)  # noqa: E402

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


# %% _get_available_huggingface_files


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
        result = _get_available_huggingface_files("org", "ds")

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
        result = _get_available_huggingface_files("org", "ds")

    assert len(result) == 2


# %% _download_data_huggingface


def test_aborts_when_dataset_is_none():
    """Test aborts when dataset is NULL."""
    with pytest.raises(ValueError):
        _download_data_huggingface(dataset=None)


def test_deprecated_type_arg_warns_and_strips_hf_prefix():
    """Test deprecated type arg warns and strips hf_ prefix."""
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_grid",
        return_value=pl.DataFrame(),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            _download_data_huggingface(type="hf_factor_library_grid")


def test_legacy_hf_dataset_value_warns_and_strips_prefix():
    """Test legacy hf_ dataset value warns and strips prefix."""
    with patch(
        "tidyfinance.download_tidy_finance._download_data_huggingface_factor_library",
        return_value=pl.DataFrame(schema={"id": pl.Int64}),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            try:
                _download_data_huggingface(dataset="hf_factor_library")
            except ValueError:
                pass


def test_aborts_for_unsupported_dataset():
    """Test aborts for unsupported dataset."""
    with pytest.raises(ValueError):
        _download_data_huggingface(dataset="unknown")


def test_factor_library_grid_delegates_to_helper():
    """Test factor_library_grid: delegates to helper."""
    mock_grid = pl.DataFrame({"id": [1]})
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_grid",
        return_value=mock_grid,
        create=True,
    ):
        result = _download_data_huggingface("factor_library_grid")
    assert_frame_equal(result, mock_grid)


def test_high_frequency_sp500_filters_by_date_and_downloads():
    """Test high_frequency_sp500: filters by date and downloads."""
    available = _available(["date=2007-07-26/part.parquet"])
    mock_trades = pl.DataFrame({"price": [100.0]})

    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=mock_trades,
        ),
    ):
        result = _download_data_huggingface(
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
        result = _download_data_huggingface(
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
        _download_data_huggingface(
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
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=pl.DataFrame({"price": [100.0]}),
        ),
    ):
        result = _download_data_huggingface("high_frequency_sp500")
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
    available = _available(["grid.parquet"])
    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=grid,
        ),
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
    available = _available(["grid.parquet"])
    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=grid,
        ),
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", min_size_quantile=None
        )

    # The default 0.2 screen is removed, so all size groups are returned.
    assert ids == [1, 2, 3]


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
    available = _available(["grid.parquet"])
    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=grid,
        ),
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
    available = _available(["grid.parquet"])
    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=grid,
        ),
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="size", fill_all=True
        )

    assert ids == [1]


# %% download_factor_library_grid (no direct Python equivalent)


def test_pulls_url_from_available_files_and_reads_parquet():
    """Test pulls url from available files and reads parquet."""
    available = _available(["grid.parquet"], [500])
    mock_grid = pl.DataFrame({"id": [1]})
    with (
        patch(
            "tidyfinance.download_tidy_finance._get_available_huggingface_files",
            return_value=available,
        ),
        patch(
            "tidyfinance.download_tidy_finance._read_parquet_url",
            return_value=mock_grid,
        ),
    ):
        result = _download_factor_library_grid()
    assert_frame_equal(result, mock_grid)


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


def _response(status_code, content=b""):
    response = MagicMock(status_code=status_code, content=content)
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


# %% _download_factor_library_ids


def test_aborts_when_ids_are_empty():
    """Test aborts when no ids are passed, before any download."""
    with (
        patch(
            "tidyfinance.download_tidy_finance._download_factor_library_grid"
        ) as grid,
        pytest.raises(ValueError, match="No portfolio IDs"),
    ):
        _download_factor_library_ids([])
    grid.assert_not_called()


def test_aborts_when_no_grid_rows_match_requested_ids():
    """Test aborts when no grid rows match requested ids."""
    with (
        patch(
            "tidyfinance.download_tidy_finance._download_factor_library_grid",
            return_value=_make_grid([42]),
        ),
        patch("tidyfinance.download_tidy_finance._fetch_parquet_url") as fetch,
        pytest.raises(ValueError, match="None of the requested"),
    ):
        _download_factor_library_ids([999])
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
            "tidyfinance.download_tidy_finance._download_factor_library_grid",
            return_value=_make_grid([1, 2, 3, 1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, requested),
        ),
    ):
        result = _download_factor_library_ids([1, 2, 1001])

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
            "tidyfinance.download_tidy_finance._download_factor_library_grid",
            return_value=_make_grid([1, 2, 1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            side_effect=_serve_files(files, []),
        ),
        pytest.warns(UserWarning, match="without returns.*: 2, 1001"),
    ):
        result = _download_factor_library_ids([1, 2, 1001])

    assert result["id"].to_list() == [1]


def test_returns_empty_frame_when_no_requested_id_has_returns():
    """Test an empty frame with all columns when no id has returns."""
    with (
        patch(
            "tidyfinance.download_tidy_finance._download_factor_library_grid",
            return_value=_make_grid([1001]),
        ),
        patch(
            "tidyfinance.download_tidy_finance._fetch_parquet_url",
            return_value=None,
        ),
        pytest.warns(UserWarning, match="without returns"),
    ):
        result = _download_factor_library_ids([1001])

    assert result.is_empty()
    assert result.columns == ["id", "date", "ret", *_make_grid().columns[1:]]


# %% _download_data_huggingface_factor_library


def test_aborts_when_ids_and_filter_args_are_combined():
    """Test aborts when ids and filter args are combined."""
    with pytest.raises(ValueError):
        _download_data_huggingface_factor_library(
            sorting_variable="size", ids=[1]
        )


def test_with_ids_delegates_to_download_factor_library_ids():
    """Test with ids: delegates to download_factor_library_ids."""
    mock_result = pl.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.download_tidy_finance._download_factor_library_ids",
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
            "tidyfinance.download_tidy_finance._download_factor_library_ids",
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
        "tidyfinance.download_tidy_finance._download_factor_library_ids",
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
        "tidyfinance.download_tidy_finance._download_factor_library_ids",
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
            "tidyfinance.download_tidy_finance._download_factor_library_ids",
            return_value=mock_returns,
        ),
    ):
        result = _download_data_huggingface_factor_library(
            sorting_variable="size",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert result["date"][0] == dt.date(2020, 6, 30)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
