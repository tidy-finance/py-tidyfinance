"""Tests for download_data_huggingface and related helpers."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.data_download import (
    _download_data_huggingface,
    _download_data_huggingface_factor_library,
    _download_factor_library_ids,
    _filter_factor_library_grid,
    _get_available_huggingface_files,
    _download_factor_library_grid
)  # noqa: E402


def _make_grid(grid_id=1):
    return pd.DataFrame(
        {
            "id": [grid_id],
            "sorting_variable": ["sv_me"],
            "sorting_variable_lag": ["6m"],
            "sorting_method": ["univariate"],
            "n_portfolios_main": [10],
            "min_size_quantile": [0.2],
            "exclude_financials": [False],
            "exclude_utilities": [False],
            "exclude_negative_earnings": [False],
            "rebalancing": ["monthly"],
            "n_portfolios_secondary": [None],
            "breakpoints_exchanges": ["NYSE"],
            "breakpoints_min_size_threshold": [None],
            "weighting_scheme": ["VW"],
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
        "tidyfinance.data_download.requests.get", return_value=response_mock
    ):
        result = _get_available_huggingface_files("org", "ds")

    assert len(result) == 1
    assert list(result.columns) == ["path", "size"]
    assert result["path"].iloc[0] == "data.parquet"


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
        "tidyfinance.data_download.requests.get", side_effect=responses
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
    with pytest.warns(DeprecationWarning, match="deprecated"):
        _download_data_huggingface(type="hf_factor_library_grid")


def test_legacy_hf_dataset_value_warns_and_strips_prefix():
    """Test legacy hf_ dataset value warns and strips prefix."""
    with patch(
        "tidyfinance.data_download._download_data_huggingface_factor_library",
        return_value=pd.DataFrame({"id": []}),
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
    mock_grid = pd.DataFrame({"id": [1]})
    with patch(
        "tidyfinance.data_download._download_factor_library_grid",
        return_value=mock_grid,
        create=True,
    ):
        result = _download_data_huggingface("factor_library_grid")
    pd.testing.assert_frame_equal(result, mock_grid)


def test_high_frequency_sp500_filters_by_date_and_downloads():
    """Test high_frequency_sp500: filters by date and downloads."""
    available = pd.DataFrame(
        {
            "path": ["date=2007-07-26/part.parquet"],
            "size": [100],
        }
    )
    mock_trades = pd.DataFrame({"price": [100.0]})

    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=mock_trades
    ):
        result = _download_data_huggingface(
            "high_frequency_sp500", "2007-07-26", "2007-07-26"
        )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_factor_library_delegates_to_inner_helper():
    """Test factor_library: delegates to inner helper."""
    mock_returns = pd.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.data_download._download_data_huggingface_factor_library",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface(
            "factor_library", sorting_variable="me"
        )
    pd.testing.assert_frame_equal(result, mock_returns)


def test_factor_library_forwards_start_date_and_end_date():
    """Test factor_library: forwards start_date and end_date."""
    captured = {}

    def fake_inner(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"id": [1]})

    with patch(
        "tidyfinance.data_download._download_data_huggingface_factor_library",
        side_effect=fake_inner,
    ):
        _download_data_huggingface(
            "factor_library",
            sorting_variable="me",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert captured.get("sorting_variable") == "me"
    assert captured.get("start_date") == "2020-01-01"
    assert captured.get("end_date") == "2020-12-31"


def test_high_frequency_sp500_uses_sample_window_when_no_dates():
    """Test high_frequency_sp500: uses sample window when no dates."""
    available = pd.DataFrame(
        {
            "path": ["date=2007-06-27/part.parquet"],
            "size": [100],
        }
    )
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet",
        return_value=pd.DataFrame({"price": [100.0]}),
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
            sorting_variable="me", sorting_method="sequential"
        )


def test_fill_all_false_defaults_applied_row_filtered_out():
    """Test fill_all = FALSE: defaults applied, row filtered out."""
    grid = pd.DataFrame(
        {
            "id": [1, 2],
            "sorting_variable": ["sv_me", "sv_me"],
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
        }
    )
    available = pd.DataFrame(
        {"path": ["grid.parquet"], "size": [100]}
    )
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=grid
    ):
        ids = _filter_factor_library_grid(sorting_variable="me")

    assert ids == [1]


def test_fill_all_true_only_explicit_filters_applied():
    """Test fill_all = TRUE: only explicit filters applied."""
    grid = pd.DataFrame(
        {
            "id": [1, 2],
            "sorting_variable": ["sv_me", "sv_bm"],
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
        }
    )
    available = pd.DataFrame(
        {"path": ["grid.parquet"], "size": [100]}
    )
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=grid
    ):
        ids = _filter_factor_library_grid(
            sorting_variable="me", fill_all=True
        )

    assert ids == [1]


# %% download_factor_library_grid (no direct Python equivalent)


def test_pulls_url_from_available_files_and_reads_parquet():
    """Test pulls url from available files and reads parquet."""
    available = pd.DataFrame(
        {"path": ["grid.parquet"], "size": [500]}
    )
    mock_grid = pd.DataFrame({"id": [1]})
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=mock_grid
    ):
        result = _download_factor_library_grid()
    pd.testing.assert_frame_equal(result, mock_grid)


# %% _download_factor_library_ids


def test_aborts_when_no_grid_rows_match_requested_ids():
    """Test aborts when no grid rows match requested ids."""
    grid = _make_grid(42)
    available = pd.DataFrame({"path": [], "size": []})
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=grid
    ):
        with pytest.raises(ValueError):
            _download_factor_library_ids([999])


def test_aborts_when_ids_have_no_matching_parquet_file():
    """Test aborts when ids have no matching parquet file."""
    grid = _make_grid(1)
    available = pd.DataFrame(
        {"path": ["unrelated/data.parquet"], "size": [100]}
    )
    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet", return_value=grid
    ):
        with pytest.raises(ValueError):
            _download_factor_library_ids([1])


def test_downloads_returns_and_joins_grid_metadata():
    """Test downloads returns and joins grid metadata."""
    fpath = (
        "sorting_variable=me/sorting_variable_lag=6m/"
        "sorting_method=univariate/n_portfolios_main=10/data.parquet"
    )
    grid = _make_grid(1)
    mock_returns = pd.DataFrame({"id": [1], "ret": [0.01]})
    available = pd.DataFrame({"path": [fpath], "size": [100]})

    def fake_read_parquet(url, *a, **kw):
        if "factor-library-grid" in str(url):
            return grid
        return mock_returns

    with patch(
        "tidyfinance.data_download._get_available_huggingface_files",
        return_value=available,
    ), patch(
        "tidyfinance.data_download.pd.read_parquet",
        side_effect=fake_read_parquet,
    ):
        result = _download_factor_library_ids([1])

    assert "ret" in result.columns
    assert "weighting_scheme" in result.columns


# %% _download_data_huggingface_factor_library


def test_aborts_when_ids_and_filter_args_are_combined():
    """Test aborts when ids and filter args are combined."""
    with pytest.raises(ValueError):
        _download_data_huggingface_factor_library(
            sorting_variable="me", ids=[1]
        )


def test_with_ids_delegates_to_download_factor_library_ids():
    """Test with ids: delegates to download_factor_library_ids."""
    mock_result = pd.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.data_download._download_factor_library_ids",
        return_value=mock_result,
    ):
        result = _download_data_huggingface_factor_library(ids=[1])
    pd.testing.assert_frame_equal(result, mock_result)


def test_without_ids_resolves_via_grid_then_downloads():
    """Test without ids: resolves via grid then downloads."""
    mock_result = pd.DataFrame({"id": [1], "ret": [0.01]})
    with patch(
        "tidyfinance.data_download._filter_factor_library_grid",
        return_value=[1],
    ), patch(
        "tidyfinance.data_download._download_factor_library_ids",
        return_value=mock_result,
    ):
        result = _download_data_huggingface_factor_library(
            sorting_variable="me"
        )
    pd.testing.assert_frame_equal(result, mock_result)


def test_filters_returns_to_the_requested_date_range():
    """Test filters returns to the requested date range."""
    mock_returns = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "date": pd.to_datetime(
                ["2019-12-31", "2020-06-30", "2021-01-31"]
            ),
            "ret": [0.01, 0.02, 0.03],
        }
    )
    with patch(
        "tidyfinance.data_download._download_factor_library_ids",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface_factor_library(
            ids=[1], start_date="2020-01-01", end_date="2020-12-31"
        )
    assert result["date"].iloc[0] == pd.Timestamp("2020-06-30")


def test_returns_full_history_when_dates_omitted():
    """Test returns full history when dates omitted."""
    mock_returns = pd.DataFrame(
        {
            "id": [1, 1],
            "date": pd.to_datetime(["2019-12-31", "2020-06-30"]),
            "ret": [0.01, 0.02],
        }
    )
    with patch(
        "tidyfinance.data_download._download_factor_library_ids",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface_factor_library(ids=[1])
    pd.testing.assert_frame_equal(result, mock_returns)


def test_date_filtering_also_applies_on_the_grid_resolved_path():
    """Test date filtering also applies on the grid-resolved path."""
    mock_returns = pd.DataFrame(
        {
            "id": [1, 1],
            "date": pd.to_datetime(["2018-01-31", "2020-06-30"]),
            "ret": [0.01, 0.02],
        }
    )
    with patch(
        "tidyfinance.data_download._filter_factor_library_grid",
        return_value=[1],
    ), patch(
        "tidyfinance.data_download._download_factor_library_ids",
        return_value=mock_returns,
    ):
        result = _download_data_huggingface_factor_library(
            sorting_variable="me",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
    assert result["date"].iloc[0] == pd.Timestamp("2020-06-30")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
