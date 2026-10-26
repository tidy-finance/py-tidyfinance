"""Tests for download_data_damodaran."""

import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download import download_data  # noqa: E402
from tidyfinance.download_open_source import (  # noqa: E402
    _check_excel_reader,
    _damodaran_data_year,
    _download_damodaran_file,
    download_data_damodaran,
    _parse_damodaran_wacc,
)
from tidyfinance.supported_datasets import list_supported_datasets  # noqa: E402


def mock_wacc_sheet(local_header="Cost of Capital (Local Currency)"):
    """A sheet in the layout of the recent files: metadata rows above the
    header, read as text without a header."""
    rows = [
        ["Date updated:", "46027"] + [None] * 10,
        ["Created by:", "Aswath Damodaran"] + [None] * 10,
        [
            "Industry Name",
            "Number of Firms",
            "Beta",
            "Cost of Equity",
            "E/(D+E)",
            "Std Dev in Stock",
            "Cost of Debt",
            "Tax Rate",
            "After-tax Cost of Debt",
            "D/(D+E)",
            "Cost of Capital",
            local_header,
        ],
        [
            "Advertising",
            "52",
            "1.2",
            "0.09",
            "0.7",
            "0.5",
            "0.06",
            "0.1",
            "0.05",
            "0.3",
            "0.08",
            "0.081",
        ],
        [
            "Total Market",
            "5994",
            "0.9",
            "0.08",
            "0.8",
            "0.4",
            "0.05",
            "0.2",
            "0.04",
            "0.2",
            "0.07",
            "0.071",
        ],
        [None] * 12,
    ]
    return pl.DataFrame(
        rows,
        schema={f"column_{i}": pl.String for i in range(1, 13)},
        orient="row",
    )


def mock_wacc_file(year):
    """A parsed WACC file with one industry."""
    return pl.DataFrame(
        {
            "year": [year],
            "industry": ["Advertising"],
            "number_of_firms": [52],
            "beta": [1.2],
            "cost_of_equity": [0.09],
            "equity_to_capital": [0.7],
            "sd_stock": [0.5],
            "cost_of_debt": [0.06],
            "tax_rate": [0.1],
            "after_tax_cost_of_debt": [0.05],
            "debt_to_capital": [0.3],
            "wacc": [0.08],
            "wacc_local": [None],
        },
        schema_overrides={"year": pl.Int64, "wacc_local": pl.Float64},
    )


def mock_response(status_code):
    """A download response with the given HTTP status."""
    response = MagicMock()
    response.status_code = status_code
    response.content = b"workbook"
    return response


def test_parses_the_industry_table_from_the_matching_sheet():
    """Test parses the industry table from the matching sheet."""
    sheets = {
        "Variables & FAQ": pl.DataFrame({"column_1": ["Notes"]}),
        "Industry Averages": mock_wacc_sheet(),
    }
    with patch(
        "tidyfinance.download_open_source._read_excel_sheets",
        return_value=sheets,
    ):
        result = _parse_damodaran_wacc(b"workbook")

    assert result.columns == [
        "year",
        "industry",
        "number_of_firms",
        "beta",
        "cost_of_equity",
        "equity_to_capital",
        "sd_stock",
        "cost_of_debt",
        "tax_rate",
        "after_tax_cost_of_debt",
        "debt_to_capital",
        "wacc",
        "wacc_local",
    ]
    assert result["industry"].to_list() == ["Advertising", "Total Market"]
    assert result["number_of_firms"].to_list() == [52, 5994]
    assert result["number_of_firms"].dtype == pl.Int64
    assert result["wacc"].to_list() == [0.08, 0.07]
    assert result["wacc_local"].to_list() == [0.081, 0.071]
    # Updated in January 2026, so the data refers to 2025.
    assert result["year"].to_list() == [2025, 2025]


def test_maps_currency_specific_local_columns_and_fills_missing_ones():
    """Test maps currency-specific local columns and fills missing ones."""
    with patch(
        "tidyfinance.download_open_source._read_excel_sheets",
        return_value={"Sheet1": mock_wacc_sheet("Cost of Capital (Euros)")},
    ):
        result = _parse_damodaran_wacc(b"workbook")
    assert result["wacc_local"].to_list() == [0.081, 0.071]

    sheet = mock_wacc_sheet().slice(1).drop("column_12")
    with patch(
        "tidyfinance.download_open_source._read_excel_sheets",
        return_value={"Sheet1": sheet},
    ):
        result = _parse_damodaran_wacc(b"workbook")
    assert result["wacc_local"].is_null().all()
    assert result["year"].is_null().all()


def test_aborts_when_no_sheet_holds_the_industry_table():
    """Test aborts when no sheet holds the industry table."""
    with patch(
        "tidyfinance.download_open_source._read_excel_sheets",
        return_value={"Sheet1": pl.DataFrame({"column_1": ["nothing"]})},
    ):
        with pytest.raises(ValueError, match="industry table"):
            _parse_damodaran_wacc(b"workbook")


def test_aborts_when_a_required_column_is_missing():
    """Test aborts when a required column is missing."""
    with patch(
        "tidyfinance.download_open_source._read_excel_sheets",
        return_value={"Sheet1": mock_wacc_sheet().drop("column_3")},
    ):
        with pytest.raises(ValueError, match="Missing columns"):
            _parse_damodaran_wacc(b"workbook")


def test_reads_the_data_year_from_excel_serials_and_iso_dates():
    """Test reads the data year from Excel serials and ISO dates."""
    assert _damodaran_data_year("46027") == 2025
    assert _damodaran_data_year("2026-01-05 00:00:00") == 2025
    assert _damodaran_data_year("Aswath Damodaran") is None
    assert _damodaran_data_year(None) is None


def test_downloads_and_parses_a_file_from_the_data_library():
    """Test downloads and parses a file from the data library."""
    with (
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=mock_response(200),
        ) as mock_get,
        patch(
            "tidyfinance.download_open_source._parse_damodaran_wacc",
            return_value="parsed",
        ),
    ):
        assert _download_damodaran_file("datasets/wacc.xls") == "parsed"

    assert mock_get.call_args.args[0] == (
        "https://pages.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls"
    )


def test_returns_none_for_missing_files_and_failed_downloads():
    """Test returns None for missing files and failed downloads."""
    with patch(
        "tidyfinance.download_open_source._parse_damodaran_wacc",
        side_effect=AssertionError("must not be called"),
    ):
        with patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=mock_response(404),
        ):
            assert _download_damodaran_file("archives/wacc25.xls") is None

        with patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=mock_response(500),
        ):
            with pytest.warns(UserWarning, match="HTTP status 500"):
                assert _download_damodaran_file("datasets/wacc.xls") is None

        with (
            patch(
                "tidyfinance.download_open_source.requests.get",
                side_effect=OSError("offline"),
            ),
            patch("tidyfinance.download_open_source.time.sleep"),
        ):
            with pytest.warns(
                UserWarning, match="Failed to download or process the resource"
            ):
                assert _download_damodaran_file("datasets/wacc.xls") is None


def test_downloads_the_current_file_by_default():
    """Test downloads the current file by default."""
    requested = []

    def fake_download(file_path):
        requested.append(file_path)
        return mock_wacc_file(2025)

    with (
        patch("tidyfinance.download_open_source._check_excel_reader"),
        patch(
            "tidyfinance.download_open_source._download_damodaran_file",
            side_effect=fake_download,
        ),
    ):
        result = download_data_damodaran()

    assert requested == ["datasets/wacc.xls"]
    assert result["region"].to_list() == ["us"]
    assert result["year"].to_list() == [2025]
    assert result.columns[:3] == ["region", "year", "industry"]


def test_stacks_archived_years_and_falls_back_to_the_current_file():
    """Test stacks archived years and falls back to the current file."""
    requested = []
    files = {
        "archives/waccEurope23.xls": mock_wacc_file(None),
        "archives/waccEurope24.xls": mock_wacc_file(None),
        "datasets/waccEurope.xls": mock_wacc_file(2025),
    }

    def fake_download(file_path):
        requested.append(file_path)
        return files.get(file_path)

    with (
        patch("tidyfinance.download_open_source._check_excel_reader"),
        patch(
            "tidyfinance.download_open_source._download_damodaran_file",
            side_effect=fake_download,
        ),
    ):
        result = download_data_damodaran(
            region="europe", year=range(2023, 2026)
        )

    assert requested == [
        "archives/waccEurope23.xls",
        "archives/waccEurope24.xls",
        "archives/waccEurope25.xls",
        "datasets/waccEurope.xls",
    ]
    assert result["year"].to_list() == [2023, 2024, 2025]
    assert result["region"].unique().to_list() == ["europe"]


def test_warns_about_unavailable_years():
    """Test warns about unavailable years."""

    def fake_download(file_path):
        if file_path == "archives/wacc98.xls":
            return mock_wacc_file(None)
        return None

    with (
        patch("tidyfinance.download_open_source._check_excel_reader"),
        patch(
            "tidyfinance.download_open_source._download_damodaran_file",
            side_effect=fake_download,
        ),
    ):
        with pytest.warns(UserWarning, match=r"us \(1997\)"):
            result = download_data_damodaran(year=[1998, 1997])

    assert result["year"].to_list() == [1998]


def test_returns_empty_dataframe_after_download_failure():
    """Test returns empty dataframe after download failure."""
    with (
        patch("tidyfinance.download_open_source._check_excel_reader"),
        patch(
            "tidyfinance.download_open_source._download_damodaran_file",
            return_value=None,
        ),
    ):
        with pytest.warns(UserWarning) as record:
            result = download_data_damodaran()

    messages = [str(w.message) for w in record]
    assert any("No Damodaran data" in m for m in messages)
    assert any("Returning an empty dataset" in m for m in messages)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


def test_validates_arguments():
    """Test validates arguments."""
    with patch("tidyfinance.download_open_source._check_excel_reader"):
        with pytest.raises(ValueError, match="Unsupported Damodaran"):
            download_data_damodaran("betas")
        with pytest.raises(ValueError, match="Unsupported region"):
            download_data_damodaran(region="mars")
        with pytest.raises(ValueError, match="whole numbers"):
            download_data_damodaran(year=2020.5)


def test_explains_a_missing_excel_reader():
    """Test explains a missing Excel reader."""
    with patch.dict(sys.modules, {"fastexcel": None}):
        with pytest.raises(ImportError, match=r"tidyfinance\[excel\]"):
            _check_excel_reader()


def test_download_data_forwards_to_download_data_damodaran():
    """Test download_data forwards to download_data_damodaran."""
    captured = {}

    def fake_damodaran(**kwargs):
        captured.update(kwargs)
        return pl.DataFrame({"ok": [True]})

    with patch(
        "tidyfinance.download.download_data_damodaran",
        side_effect=fake_damodaran,
    ):
        download_data("Damodaran", region="japan")

    assert captured == {"dataset": "wacc", "region": "japan"}


def test_lists_damodaran_as_a_supported_dataset():
    """Test lists Damodaran as a supported dataset."""
    result = list_supported_datasets(domain="Damodaran", as_vector=True)
    assert result == ["wacc"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
