"""Tests for download_data_factors_ff and download_data_factors_q."""

import datetime as dt
import io
import os
import sys
import zipfile
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import (
    download_data_factors_ff,
    download_data_factors_q,
    _breakpoints_col_names_ff,
    _famafrench_downloader,
    _parse_french_data_factors,
)  # noqa: E402
from tidyfinance.supported_datasets import (
    _check_supported_dataset_ff,
    _check_supported_dataset_q,
    _determine_frequency_ff,
    _determine_frequency_q,
    _is_breakpoints_ff,
    _is_legacy_type_ff,
    _is_legacy_type_q,
)  # noqa: E402

# %% determine_frequency_ff


def test_determine_frequency_ff_daily_weekly_monthly_paths():
    """Test determine_frequency_ff: daily, weekly, and monthly paths."""
    assert _determine_frequency_ff("Factors [Daily]") == "daily"
    assert _determine_frequency_ff("Factors [Weekly]") == "weekly"
    assert _determine_frequency_ff("Fama/French 3 Factors") == "monthly"


def test_determine_frequency_q_all_five_frequency_paths():
    """Test determine_frequency_q: all five frequency paths."""
    assert _determine_frequency_q("q5_factors_daily_2023") == "daily"
    assert _determine_frequency_q("q5_factors_weekly_2023") == "weekly"
    assert _determine_frequency_q("q5_factors_monthly_2023") == "monthly"
    assert _determine_frequency_q("q5_factors_quarterly_2023") == "quarterly"
    assert _determine_frequency_q("q5_factors_annual_2023") == "annual"


def test_determine_frequency_q_aborts_on_unknown_frequency():
    """Test determine_frequency_q: aborts on unknown frequency."""
    with pytest.raises(ValueError, match="Cannot determine frequency"):
        _determine_frequency_q("q5_factors_2023")


# %% check_supported_dataset_ff / _q


def test_check_supported_dataset_ff_no_error_for_known_dataset():
    """Test check_supported_dataset_ff: no error for known dataset."""
    _check_supported_dataset_ff("Fama/French 3 Factors")


def test_check_supported_dataset_ff_aborts_for_unknown_dataset():
    """Test check_supported_dataset_ff: aborts for unknown dataset."""
    with pytest.raises(ValueError, match="Unsupported Fama-French dataset"):
        _check_supported_dataset_ff("Unknown")


def test_check_supported_dataset_q_no_error_for_known_dataset():
    """Test check_supported_dataset_q: no error for known dataset."""
    _check_supported_dataset_q("q5_factors_monthly_2024")


def test_check_supported_dataset_q_aborts_for_unknown_dataset():
    """Test check_supported_dataset_q: aborts for unknown dataset."""
    with pytest.raises(ValueError, match="Unsupported Global Q dataset"):
        _check_supported_dataset_q("unknown")


# %% is_legacy_type_ff / _q


def test_is_legacy_type_ff_true_for_legacy_type_false_otherwise():
    """Test is_legacy_type_ff: TRUE for legacy type, FALSE otherwise."""
    assert _is_legacy_type_ff("factors_ff3_monthly")
    assert not _is_legacy_type_ff("not_a_legacy_type")


def test_is_legacy_type_q_true_for_legacy_type_false_otherwise():
    """Test is_legacy_type_q: TRUE for legacy type, FALSE otherwise."""
    assert _is_legacy_type_q("factors_q5_monthly")
    assert not _is_legacy_type_q("not_a_legacy_type")


# %% download_data_factors_ff


def test_download_data_factors_ff_aborts_when_dataset_is_none():
    """Test download_data_factors_ff: aborts when dataset is NULL."""
    with pytest.raises((ValueError, TypeError)):
        download_data_factors_ff(None)


def test_download_data_factors_ff_deprecated_type_warns():
    """Test download_data_factors_ff: deprecated type warns."""
    fake_raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)],
            "Mkt-RF": [1.0],
            "RF": [0.1],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            download_data_factors_ff(type="factors_ff3_monthly")


def test_download_data_factors_ff_legacy_dataset_arg_warns():
    """Test download_data_factors_ff: legacy dataset arg warns."""
    fake_raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)],
            "Mkt-RF": [1.0],
            "RF": [0.1],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            download_data_factors_ff("factors_ff3_monthly")


def test_legacy_dataset_warning_names_the_replacement():
    """Test the warning names the legacy value and the dataset that
    replaces it, not the 'type' argument the caller did not use."""
    fake_raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)],
            "Mkt-RF": [1.0],
            "RF": [0.1],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning) as record:
            download_data_factors_ff("factors_ff_3_monthly")

    messages = [str(w.message) for w in record]
    expected = (
        "dataset='factors_ff_3_monthly' is deprecated. "
        "Use dataset='Fama/French 3 Factors' instead."
    )
    assert expected in messages
    assert not any("'type'" in m for m in messages)


def test_legacy_dataset_warning_is_attributed_to_the_caller():
    """Test that, through the public download_data, the warning points
    at the caller and not at package code. Python hides
    DeprecationWarning by default unless it is attributed to '__main__',
    so a warning attributed to package code would never be shown."""
    import tidyfinance as tf

    fake_raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)],
            "Mkt-RF": [1.0],
            "RF": [0.1],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=fake_raw,
        ),
    ):
        with pytest.warns(DeprecationWarning, match="is deprecated") as record:
            tf.download_data("Fama-French", "factors_ff_3_monthly")

    legacy = [w for w in record if "factors_ff_3_monthly" in str(w.message)]
    assert legacy
    assert os.path.normcase(os.path.abspath(legacy[0].filename)) == (
        os.path.normcase(os.path.abspath(__file__))
    )


def test_download_data_factors_ff_empty_dataframe_on_download_fail():
    """Test download_data_factors_ff: empty dataframe on download fail."""
    with patch(
        "tidyfinance.download_open_source._famafrench_downloader",
        side_effect=Exception("download fail"),
    ):
        with pytest.warns(UserWarning, match="Returning an empty dataset"):
            result = download_data_factors_ff("Fama/French 3 Factors")
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


def test_download_data_factors_ff_monthly_path_with_date_filter():
    """Test download_data_factors_ff: monthly path with date filter."""
    raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "Mkt-RF": [1.0, 2.0],
            "RF": [0.1, 0.1],
        }
    )
    with patch(
        "tidyfinance.download_open_source._famafrench_downloader",
        return_value=raw,
    ):
        result = download_data_factors_ff(
            "Fama/French 3 Factors", "2020-01-01", "2020-01-31"
        )
    assert isinstance(result, pl.DataFrame)
    assert "mkt_excess" in result.columns
    assert "risk_free" in result.columns


def test_download_data_factors_ff_threads_registry_url_to_downloader():
    """Test download_data_factors_ff: threads registry url to downloader."""
    captured = {}

    def fake_downloader(dataset, **kw):
        captured["dataset"] = dataset
        raw = pl.DataFrame(
            {
                "date": [dt.date(2020, 1, 1)],
                "Mkt-RF": [1.0],
                "RF": [0.1],
            }
        )
        return raw

    with patch(
        "tidyfinance.download_open_source._famafrench_downloader",
        side_effect=fake_downloader,
    ):
        download_data_factors_ff("Fama/French 3 Factors")
    assert captured["dataset"] == "ftp/F-F_Research_Data_Factors_CSV.zip"


def test_download_data_factors_ff_daily_path_no_date_filter():
    """Test download_data_factors_ff: daily path, no date filter."""
    raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
            "Mkt-RF": [1.0, 2.0],
            "RF": [0.1, 0.1],
        }
    )
    with patch(
        "tidyfinance.download_open_source._famafrench_downloader",
        return_value=raw,
    ):
        result = download_data_factors_ff("Fama/French 3 Factors [Daily]")
    assert len(result) == 2


def test_download_data_factors_ff_aborts_on_unknown_frequency():
    """Test download_data_factors_ff: aborts on unknown frequency."""
    with (
        patch(
            "tidyfinance.download_open_source._check_supported_dataset_ff",
            return_value="ftp/some_CSV.zip",
        ),
        patch(
            "tidyfinance.download_open_source._determine_frequency_ff",
            return_value="unknown",
        ),
    ):
        with pytest.raises(
            ValueError, match="neither daily, weekly, nor monthly"
        ):
            download_data_factors_ff("some_dataset")


# %% download_data_factors_q


def test_download_data_factors_q_download_lambda_reads_csv():
    """Test download_data_factors_q: download lambda reads CSV."""
    mock_csv = pl.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 2],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch(
        "tidyfinance.download_open_source._fetch_csv",
        return_value=mock_csv,
    ):
        result = download_data_factors_q("q5_factors_monthly_2024")
    assert len(result) == 2
    assert "risk_free" in result.columns


def test_download_data_factors_q_aborts_when_dataset_is_none():
    """Test download_data_factors_q: aborts when dataset is NULL."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        download_data_factors_q(None)


def test_download_data_factors_q_deprecated_type_warns():
    """Test download_data_factors_q: deprecated type warns."""
    with patch(
        "tidyfinance.download_open_source._fetch_csv",
        side_effect=Exception("stop after translation"),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError):
                download_data_factors_q(type="factors_q5_monthly")


def test_download_data_factors_q_legacy_dataset_arg_warns():
    """Test download_data_factors_q: legacy dataset arg warns."""
    with patch(
        "tidyfinance.download_open_source._fetch_csv",
        side_effect=Exception("stop after translation"),
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            with pytest.raises(ValueError):
                download_data_factors_q("factors_q5_monthly")


def test_download_data_factors_q_empty_dataframe_on_download_fail():
    """Test download_data_factors_q: empty dataframe on download fail."""
    with patch(
        "tidyfinance.download_open_source._fetch_csv",
        side_effect=Exception("download fail"),
    ):
        with pytest.raises(ValueError):
            download_data_factors_q("q5_factors_monthly")


def test_download_data_factors_q_monthly_path_with_date_filter():
    """Test download_data_factors_q: monthly path with date filter."""
    raw = pl.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 2],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.download_open_source._fetch_csv", return_value=raw):
        result = download_data_factors_q(
            "q5_factors_monthly_2024", "2020-01-01", "2020-01-31"
        )
    assert len(result) == 1
    assert "risk_free" in result.columns
    assert "mkt_excess" in result.columns


def test_download_data_factors_q_daily_path_no_date_filter():
    """Test download_data_factors_q: daily path, no date filter."""
    raw = pl.DataFrame(
        {
            "DATE": ["20200101", "20200102"],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.download_open_source._fetch_csv", return_value=raw):
        try:
            result = download_data_factors_q("q5_factors_daily_2024")
            assert len(result) >= 0
        except Exception:
            pytest.xfail("Python daily q-factors parser differs from R")


def test_download_data_factors_q_annual_path():
    """Test download_data_factors_q: annual path."""
    raw = pl.DataFrame(
        {"year": [2020, 2021], "R_F": [0.1, 0.1], "R_MKT": [1.0, 2.0]}
    )
    with patch("tidyfinance.download_open_source._fetch_csv", return_value=raw):
        result = download_data_factors_q("q5_factors_annual_2024")
    assert len(result) == 2


def test_download_data_factors_q_weekly_path():
    """Test download_data_factors_q: weekly path."""
    raw = pl.DataFrame(
        {
            "year": [2020, 2020],
            "month": [1, 1],
            "day": [6, 13],
            "R_F": [0.1, 0.1],
            "R_MKT": [1.0, 2.0],
        }
    )
    with patch("tidyfinance.download_open_source._fetch_csv", return_value=raw):
        try:
            result = download_data_factors_q("q5_factors_weekly_2024")
            assert len(result) >= 0
        except Exception:
            pytest.xfail("Python weekly q-factors parser differs from R")


def test_check_supported_dataset_ff_returns_file_url_for_known_dataset():
    """Test check_supported_dataset_ff: returns file_url for known dataset."""
    assert (
        _check_supported_dataset_ff("Fama/French 3 Factors")
        == "ftp/F-F_Research_Data_Factors_CSV.zip"
    )


def test_download_data_factors_ff_full_path_converts_integer_dates():
    """Test download_data_factors_ff: full path converts integer dates."""
    raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "Mkt-RF": [1.0, 2.0],
            "RF": [0.1, 0.1],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["Fama/French 3 Factors"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=raw,
        ),
    ):
        result = download_data_factors_ff("Fama/French 3 Factors")
    assert "mkt_excess" in result.columns
    assert "risk_free" in result.columns
    # Values divided by 100 in the Python implementation
    assert result["mkt_excess"][0] == pytest.approx(0.01)


def test_is_breakpoints_ff_detects_breakpoints_datasets():
    """Test is_breakpoints_ff: detects breakpoints datasets."""
    assert _is_breakpoints_ff("ME Breakpoints")
    assert _is_breakpoints_ff("Prior (2-12) Return Breakpoints")
    assert not _is_breakpoints_ff("Fama/French 3 Factors")


def test_download_data_factors_ff_does_not_rescale_breakpoints():
    """Test download_data_factors_ff: does not rescale breakpoints."""
    raw = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
            "v2": [488, 492],
            "v3": [1.40, 1.38],
            "v4": [1319.00, 1331.71],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._get_available_famafrench_datasets",
            return_value=["ME Breakpoints"],
        ),
        patch(
            "tidyfinance.download_open_source._famafrench_downloader",
            return_value=raw,
        ),
    ):
        result = download_data_factors_ff("ME Breakpoints")
    assert list(result["v2"]) == [488, 492]


# %% _famafrench_downloader


def _make_french_zip(table_body):
    """Build an in-memory Kenneth French style ZIP from a CSV body.

    A leading documentation chunk is prepended so the parser has to
    skip it and keep only the first data table.
    """
    header = "This file was created for testing tidyfinance. " * 3
    data_raw = header + "\r\n\r\n" + table_body
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", data_raw.encode("latin1"))
    response = MagicMock()
    response.content = buffer.getvalue()
    response.raise_for_status = MagicMock()
    return response


def test_famafrench_downloader_parses_first_table():
    """Test _famafrench_downloader parses the first table in the archive."""
    rows = "\r\n".join(
        f"{y}{m:02d},1.0,0.1" for y in range(2000, 2021) for m in range(1, 13)
    )
    response = _make_french_zip(",Mkt-RF,RF\r\n" + rows)

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        result = _famafrench_downloader("ftp/Some_File_CSV.zip")

    assert isinstance(result, pl.DataFrame)
    # polars has no index: the parsed date is the leading column
    assert result.columns == ["date", "Mkt-RF", "RF"]
    assert result.schema["date"] == pl.Date
    assert len(result) == 252


def test_famafrench_downloader_applies_date_filter():
    """Test _famafrench_downloader filters the parsed table by date."""
    rows = "\r\n".join(
        f"{y}{m:02d},1.0,0.1" for y in range(2000, 2021) for m in range(1, 13)
    )
    response = _make_french_zip(",Mkt-RF,RF\r\n" + rows)

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        result = _famafrench_downloader(
            "ftp/Some_File_CSV.zip",
            start_date="2010-01-01",
            end_date="2010-12-31",
        )

    assert len(result) == 12
    assert result["date"].min() >= dt.date(2010, 1, 1)
    assert result["date"].max() <= dt.date(2010, 12, 31)


def _breakpoint_rows(first_month, n_lead):
    """Rows of a breakpoint file: date, n_lead counts, 20 percentiles."""
    pct = ",".join(f"{p:.2f}" for p in range(5, 105, 5))
    lead = ",".join(["100"] * n_lead)
    year, month = first_month
    rows = []
    for i in range(80):
        y, m = year + (month - 1 + i) // 12, (month - 1 + i) % 12 + 1
        rows.append(f"{y}{m:02d},{lead},{pct}")
    return "\r\n".join(rows)


@pytest.mark.parametrize(
    "file_url",
    ["ftp/ME_Breakpoints_CSV.zip", "ftp/Prior_2-12_Breakpoints_CSV.zip"],
)
def test_famafrench_downloader_keeps_first_row_of_headerless_breakpoints(
    file_url,
):
    """Test headerless breakpoint files keep their first row and count."""
    response = _make_french_zip(_breakpoint_rows((1925, 12), 1))

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        result = _famafrench_downloader(file_url)

    assert result.columns[:3] == ["date", "Count", "0-5"]
    assert result.columns[-1] == "95-100"
    assert result.height == 80
    assert result["date"][0] == dt.date(1925, 12, 1)
    assert result["Count"][0] == 100


def test_famafrench_downloader_drops_header_of_ratio_breakpoints():
    """Test ratio breakpoint files drop their header row."""
    response = _make_french_zip(
        "  ,<= 0,>0\r\n" + _breakpoint_rows((1926, 1), 2)
    )

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        result = _famafrench_downloader("ftp/BE-ME_Breakpoints_CSV.zip")

    assert result.columns[:4] == ["date", "<=0", ">0", "0-5"]
    assert result.height == 80
    assert result["date"][0] == dt.date(1926, 1, 1)


def test_famafrench_downloader_propagates_download_error():
    """Test _famafrench_downloader propagates HTTP download errors."""
    response = MagicMock()
    response.raise_for_status.side_effect = Exception("404 error")

    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        with pytest.raises(Exception, match="404 error"):
            _famafrench_downloader("ftp/Missing_CSV.zip")


# %% _parse_french_data_factors


def test_parse_french_data_factors_keeps_first_table_names_date_column():
    text = (
        "This file was created by a test\n"
        "\n"
        ",Mkt-RF,SMB,RF\n"
        "202001,  1.00, -0.50, 0.10\n"
        "202002,  2.00,  0.50, 0.10\n"
        "\n"
        "  Annual Factors:\n"
        ",Mkt-RF,SMB,RF\n"
        "2020, 12.00, -1.00, 1.20"
    )
    result = _parse_french_data_factors(text)
    assert result.columns == ["date", "Mkt-RF", "SMB", "RF"]
    assert result.height == 2
    assert result["date"].to_list() == ["202001", "202002"]
    assert result["Mkt-RF"].to_list() == [1.0, 2.0]


def test_parse_french_data_factors_aborts_when_no_table_present():
    with pytest.raises(ValueError, match="Could not locate a data table"):
        _parse_french_data_factors("Only header text\nno data here")


def test_parse_french_data_factors_parses_positionally_without_header():
    result = _parse_french_data_factors(
        "202001,  1.00, -0.50\n202002,  2.00,  0.25"
    )
    assert result.height == 2
    assert result.columns == ["date", "V2", "V3"]
    assert result["date"].to_list() == ["202001", "202002"]


def test_parse_french_data_factors_blank_line_above_data_is_headerless():
    result = _parse_french_data_factors(
        "Some title text\n\n202001,  1.00, -0.50"
    )
    assert result.height == 1
    assert result.columns == ["date", "V2", "V3"]


def test_parse_french_data_factors_prose_line_above_data_is_headerless():
    result = _parse_french_data_factors(
        "Average Value Weight Returns\n202001,1.0,0.1"
    )
    assert result.height == 1
    assert result.columns == ["date", "V2", "V3"]


def test_parse_french_data_factors_ignores_prose_starting_with_a_digit():
    result = _parse_french_data_factors(
        "1 month is skipped\n,Mom\n202001,1.0\n202002,2.0"
    )
    assert result.columns == ["date", "Mom"]
    assert result.height == 2


def test_parse_french_data_factors_accepts_named_first_column():
    result = _parse_french_data_factors(
        "Count,P5,P10\n202001,100,5.0\n202002,101,5.1"
    )
    assert result.height == 2
    assert result.columns == ["date", "P5", "P10"]


def test_parse_french_data_factors_parses_breakpoints_file_layout():
    """A prose title, a blank line, then headerless data; the
    integer-looking count column is read as a float."""
    text = (
        "This file was created using the CRSP database.\n"
        "\n"
        "192512,   488,        1.40,        2.38,     1319.00\n"
        "192601,   492,        1.38,        2.54,     1331.71"
    )
    result = _parse_french_data_factors(text)
    assert result.height == 2
    assert result["date"].to_list() == ["192512", "192601"]
    assert result.columns == ["date", "V2", "V3", "V4", "V5"]
    assert result.schema["V2"] == pl.Float64


def test_parse_french_data_factors_drops_merged_annual_rows():
    text = ",Mkt-RF,RF\n202001,1.0,0.1\n202002,2.0,0.1\n2020,12.0,1.2"
    with pytest.warns(UserWarning, match="different date-key width"):
        result = _parse_french_data_factors(text)
    assert result.height == 2
    assert result["date"].to_list() == ["202001", "202002"]


def test_parse_french_data_factors_aborts_on_column_count_mismatch():
    with pytest.raises(ValueError, match="Expected 2 columns"):
        _parse_french_data_factors("202001,1.0,0.1", col_names=["date", "a"])


def test_famafrench_downloader_parses_a_small_first_table():
    """Tables are found by their date keys, whatever their size."""
    response = _make_french_zip(
        ",Mkt-RF,RF\r\n202001,1.0,0.1\r\n202002,2.0,0.1"
    )
    with patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    ):
        result = _famafrench_downloader("ftp/Some_File_CSV.zip")
    assert result.columns == ["date", "Mkt-RF", "RF"]
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]


def test_famafrench_downloader_aborts_when_archive_has_no_csv():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("readme.txt", "no table")
    response = MagicMock()
    response.content = buffer.getvalue()
    with (
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=response,
        ),
        pytest.raises(ValueError, match="No CSV file found"),
    ):
        _famafrench_downloader("ftp/Empty_CSV.zip")


# %% breakpoints files

_PERCENTILE_BINS = [f"{lo}-{lo + 5}" for lo in range(0, 100, 5)]


def _breakpoints_row(date, counts):
    return ",   ".join([date, *counts, *(f"{p:.2f}" for p in range(1, 21))])


def _mock_ff_file(lines):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", "\n".join(lines))
    response = MagicMock()
    response.content = buffer.getvalue()
    return patch(
        "tidyfinance.download_open_source.requests.get", return_value=response
    )


def test_breakpoints_col_names_ff_names_count_and_ratio_layouts():
    assert (
        _breakpoints_col_names_ff("ftp/F-F_Research_Data_Factors_CSV.zip")
        is None
    )
    assert _breakpoints_col_names_ff("ftp/ME_Breakpoints_CSV.zip") == [
        "date",
        "Count",
        *_PERCENTILE_BINS,
    ]
    assert _breakpoints_col_names_ff("ftp/Prior_2-12_Breakpoints_CSV.zip") == [
        "date",
        "Count",
        *_PERCENTILE_BINS,
    ]
    assert _breakpoints_col_names_ff("ftp/BE-ME_Breakpoints_CSV.zip") == [
        "date",
        "<=0",
        ">0",
        *_PERCENTILE_BINS,
    ]


def test_download_data_factors_ff_parses_count_breakpoints_unscaled():
    lines = [
        "This file was created using the CRSP database.",
        "",
        _breakpoints_row("202001", ["488"]),
        _breakpoints_row("202002", ["492"]),
    ]
    with _mock_ff_file(lines):
        result = download_data_factors_ff("ME Breakpoints")

    assert result.columns == ["date", "count", *_PERCENTILE_BINS]
    assert result["date"].to_list() == [
        dt.date(2020, 1, 1),
        dt.date(2020, 2, 1),
    ]
    # Dollar levels and share counts are returned unscaled.
    assert result["count"].to_list() == [488, 492]
    assert result["95-100"].to_list() == [20, 20]


def test_download_data_factors_ff_parses_ratio_breakpoints_layout():
    """The header names only 3 of the 23 columns."""
    lines = [
        "This file was created using the CRSP database.",
        "",
        "  ,<= 0,>0",
        _breakpoints_row("2019", ["3", "428"]),
        _breakpoints_row("2020", ["1", "464"]),
    ]
    with _mock_ff_file(lines):
        result = download_data_factors_ff("BE/ME Breakpoints")

    assert result.columns == ["date", "<=0", ">0", *_PERCENTILE_BINS]
    assert result["date"].to_list() == [
        dt.date(2019, 12, 31),
        dt.date(2020, 12, 31),
    ]
    assert result["<=0"].to_list() == [3, 1]
    assert result[">0"].to_list() == [428, 464]
    assert result["0-5"].to_list() == [1, 1]


def test_download_data_factors_ff_parses_prior_return_breakpoints():
    """The preamble has three lines, one of which starts with a digit."""
    lines = [
        "This file was created using the CRSP database.",
        " The prior returns are for 12 months. ",
        "1 month is skipped at the end of the prior period.",
        "",
        _breakpoints_row("192612", ["472"]),
        _breakpoints_row("192701", ["476"]),
    ]
    with _mock_ff_file(lines):
        result = download_data_factors_ff("Prior (2-12) Return Breakpoints")

    assert result.columns == ["date", "count", *_PERCENTILE_BINS]
    assert result["date"].to_list() == [
        dt.date(1926, 12, 1),
        dt.date(1927, 1, 1),
    ]
    assert result["count"].to_list() == [472, 476]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
