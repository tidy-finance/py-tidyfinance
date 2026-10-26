"""Tests for download_data_jkp and list_supported_jkp_factors."""

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

from tidyfinance.download_open_source import (  # noqa: E402
    _build_jkp_reference_url,
    _build_jkp_url,
    _download_jkp_csv,
    _download_jkp_file,
    _fetch_jkp_availability,
    download_data_jkp,
)
from tidyfinance.utilities import list_supported_jkp_factors  # noqa: E402


def _manifest():
    return {
        "factors": {
            "usa": ["mkt", "all_factors", "all_themes", "value", "be_me"],
            "frontier": ["mkt", "all_factors", "aliq_mat"],
        },
        "portfolios": {"usa": ["be_me", "ret_12_1"]},
        "industry": {"usa": ["gics", "ff49"]},
        "factors_monthly_only": {"frontier": ["aliq_mat"]},
    }


# %% download_data_jkp: factors


def test_downloads_and_processes_monthly_factor_returns():
    raw = pl.DataFrame(
        {
            "location": ["usa", "usa"],
            "name": ["mkt", "mkt"],
            "freq": ["monthly", "monthly"],
            "weighting": ["vw_cap", "vw_cap"],
            "n_stocks": [494, 505],
            "date": ["1926-01-31", "1926-02-28"],
            "ret": [0.001, -0.046],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            return_value=raw,
        ),
    ):
        result = download_data_jkp(region="usa", factors="mkt")

    assert isinstance(result, pl.DataFrame)
    # Dates are aligned to the beginning of the month.
    assert result["date"].to_list() == [
        dt.date(1926, 1, 1),
        dt.date(1926, 2, 1),
    ]
    # Returns are already decimal and must not be rescaled.
    assert result["ret"].to_list() == [0.001, -0.046]


def test_keeps_daily_dates_as_is():
    raw = pl.DataFrame(
        {"date": ["2020-01-02", "2020-01-03"], "ret": [0.01, 0.02]}
    )
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            return_value=raw,
        ),
    ):
        result = download_data_jkp(
            region="usa", factors="mkt", frequency="daily"
        )

    assert result["date"].to_list() == [
        dt.date(2020, 1, 2),
        dt.date(2020, 1, 3),
    ]


def test_downloads_portfolios_and_coerces_pf_to_integer():
    raw = pl.DataFrame(
        {
            "location": ["usa", "usa"],
            "name": ["be_me", "be_me"],
            "pf": [1.0, 3.0],
            "n": [494, 497],
            "freq": ["monthly", "monthly"],
            "weighting": ["vw_cap", "vw_cap"],
            "date": ["1926-01-31", "1926-01-31"],
            "ret": [0.001, -0.002],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            return_value=raw,
        ),
    ):
        result = download_data_jkp(dataset="portfolios", factors="be_me")

    assert result["pf"].dtype.is_integer()
    assert result["pf"].to_list() == [1, 3]
    assert result["date"].to_list() == [
        dt.date(1926, 1, 1),
        dt.date(1926, 1, 1),
    ]


def test_downloads_industry_returns_at_monthly_frequency():
    raw = pl.DataFrame(
        {
            "gics": [55, 15],
            "date": ["1999-07-31", "1999-07-31"],
            "n": [170, 376],
            "location": ["usa", "usa"],
            "ret": [-0.004, -0.036],
            "freq": ["monthly", "monthly"],
            "weighting": ["vw_cap", "vw_cap"],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            return_value=raw,
        ),
    ):
        result = download_data_jkp(dataset="industry", classification="gics")

    assert result["date"].to_list() == [
        dt.date(1999, 7, 1),
        dt.date(1999, 7, 1),
    ]
    assert "gics" in result.columns


# %% download_data_jkp: reference datasets


def test_downloads_reference_cutoffs_and_renames_eom_to_date():
    raw = pl.DataFrame(
        {
            "eom": ["1925-12-31", "1926-01-31"],
            "n": [495, 508],
            "nyse_p50": [15.84, 16.25],
        }
    )
    with patch(
        "tidyfinance.download_open_source._download_jkp_csv", return_value=raw
    ):
        result = download_data_jkp(dataset="nyse_cutoffs")

    assert "date" in result.columns
    assert "eom" not in result.columns
    assert result["date"].to_list() == [
        dt.date(1925, 12, 1),
        dt.date(1926, 1, 1),
    ]


def test_return_cutoffs_selects_the_daily_file_by_frequency():
    captured_url = {}

    def fake_download(url, *args, **kwargs):
        captured_url["url"] = url
        return pl.DataFrame({"eom": ["2020-01-31"], "ret_1": [-0.14]})

    with patch(
        "tidyfinance.download_open_source._download_jkp_csv",
        side_effect=fake_download,
    ):
        download_data_jkp(dataset="return_cutoffs", frequency="daily")

    assert captured_url["url"].endswith("return_cutoffs_daily.csv")


def test_returns_empty_dataframe_when_a_reference_download_fails():
    with patch(
        "tidyfinance.download_open_source._download_jkp_csv",
        side_effect=Exception("boom"),
    ):
        with pytest.warns(UserWarning, match="download or parsing failure"):
            result = download_data_jkp(dataset="nyse_cutoffs")

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


# %% download_data_jkp: date filtering


def test_filters_rows_when_both_dates_are_supplied():
    raw = pl.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "ret": [1, 2, 3],
        }
    )
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            return_value=raw,
        ),
    ):
        result = download_data_jkp(
            region="usa",
            factors="mkt",
            start_date="2020-02-01",
            end_date="2020-02-28",
        )

    assert len(result) == 1
    assert result["date"][0] == dt.date(2020, 2, 1)


# %% download_data_jkp: validation errors


def test_aborts_on_unsupported_dataset():
    with pytest.raises(ValueError, match="dataset"):
        download_data_jkp(dataset="bogus")


def test_aborts_on_daily_request_for_the_industry_dataset():
    with pytest.raises(ValueError, match="monthly frequency"):
        download_data_jkp(dataset="industry", frequency="daily")


def test_aborts_on_invalid_region():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        return_value=_manifest(),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            download_data_jkp(region="atlantis", factors="mkt")


def test_aborts_on_factor_not_available_in_region():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        return_value=_manifest(),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            download_data_jkp(region="frontier", factors="value")


def test_aborts_on_invalid_industry_classification():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        return_value=_manifest(),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            download_data_jkp(
                dataset="industry", region="usa", classification="naics"
            )


def test_aborts_on_daily_request_for_a_monthly_only_factor():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        return_value=_manifest(),
    ):
        with pytest.raises(ValueError, match="only available at monthly"):
            download_data_jkp(
                region="frontier", factors="aliq_mat", frequency="daily"
            )


def test_aborts_on_invalid_frequency_or_weighting():
    with pytest.raises(ValueError, match="frequency"):
        download_data_jkp(frequency="weekly")
    with pytest.raises(ValueError, match="weighting"):
        download_data_jkp(weighting="gdp_weighted")


def test_returns_empty_dataframe_when_the_manifest_is_unavailable():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        side_effect=Exception("boom"),
    ):
        with pytest.warns(UserWarning, match="download failure"):
            result = download_data_jkp()

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


def test_returns_empty_dataframe_when_a_factor_download_fails():
    with (
        patch(
            "tidyfinance.download_open_source._fetch_jkp_availability",
            return_value=_manifest(),
        ),
        patch(
            "tidyfinance.download_open_source._download_jkp_file",
            side_effect=Exception("boom"),
        ),
    ):
        with pytest.warns(UserWarning, match="download or parsing failure"):
            result = download_data_jkp(region="usa", factors="mkt")

    assert len(result) == 0


# %% _build_jkp_url / _build_jkp_reference_url


def test_builds_the_bracketed_s3_object_keys():
    assert _build_jkp_url(
        "factors", "usa", "all_factors", "monthly", "vw_cap"
    ) == (
        "https://jkpfactors-data.s3.amazonaws.com/public/"
        "%5Busa%5D_%5Ball_factors%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
    )
    assert _build_jkp_url(
        "portfolios", "usa", "be_me", "monthly", "vw_cap"
    ) == (
        "https://jkpfactors-data.s3.amazonaws.com/public/portfolios/"
        "%5Busa%5D_%5Bbe_me%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
    )
    # Industry always uses the monthly key regardless of frequency.
    assert _build_jkp_url("industry", "usa", "gics", "daily", "ew") == (
        "https://jkpfactors-data.s3.amazonaws.com/public/industry/"
        "%5Busa%5D_%5Bgics%5D_%5Bmonthly%5D_%5Bew%5D.zip"
    )


def test_builds_reference_file_urls():
    assert _build_jkp_reference_url("nyse_cutoffs", "monthly").endswith(
        "/public/other/nyse_cutoffs.csv"
    )
    assert _build_jkp_reference_url("return_cutoffs", "monthly").endswith(
        "/public/other/return_cutoffs.csv"
    )
    assert _build_jkp_reference_url("return_cutoffs", "daily").endswith(
        "/public/other/return_cutoffs_daily.csv"
    )


# %% list_supported_jkp_factors


def test_list_supported_jkp_factors_returns_regions_and_per_region_values():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        return_value=_manifest(),
    ):
        regions = list_supported_jkp_factors()
        assert isinstance(regions, pl.DataFrame)
        assert regions["region"].to_list() == ["usa", "frontier"]
        assert "be_me" in list_supported_jkp_factors("usa")["factor"].to_list()
        assert list_supported_jkp_factors("usa", dataset="industry")[
            "factor"
        ].to_list() == ["gics", "ff49"]
        with pytest.raises(ValueError, match="Unsupported"):
            list_supported_jkp_factors("atlantis")


def test_list_supported_jkp_factors_aborts_on_unsupported_dataset():
    with pytest.raises(ValueError, match="dataset"):
        list_supported_jkp_factors(dataset="bogus")


def test_list_supported_jkp_factors_returns_empty_dataframe_on_failure():
    with patch(
        "tidyfinance.download_open_source._fetch_jkp_availability",
        side_effect=Exception("boom"),
    ):
        with pytest.warns(UserWarning, match="download failure"):
            result = list_supported_jkp_factors()

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 0


# %% download helpers


def _jkp_response(**attrs):
    return MagicMock(status_code=200, **attrs)


def _zip_of(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buffer.getvalue()


def test_fetch_jkp_availability_parses_the_json_manifest():
    response = _jkp_response()
    response.json.return_value = {"factors": {"usa": ["mkt", "be_me"]}}
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=response,
    ):
        result = _fetch_jkp_availability()
    assert result["factors"]["usa"] == ["mkt", "be_me"]


def test_download_jkp_file_unzips_and_reads_the_first_csv():
    content = _zip_of({"data.csv": "location,date,ret\nusa,2020-01-31,0.01"})
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=_jkp_response(content=content),
    ):
        result = _download_jkp_file("https://example.com/x.zip")
    assert result["location"].to_list() == ["usa"]
    assert result["ret"].to_list() == [0.01]


def test_download_jkp_file_aborts_when_the_archive_has_no_csv():
    with (
        patch(
            "tidyfinance.download_open_source.requests.get",
            return_value=_jkp_response(content=_zip_of({"readme.txt": "x"})),
        ),
        pytest.raises(ValueError, match="No CSV file"),
    ):
        _download_jkp_file("https://example.com/x.zip")


def test_download_jkp_csv_reads_a_plain_csv_file():
    with patch(
        "tidyfinance.download_open_source.requests.get",
        return_value=_jkp_response(content=b"eom,n\n2020-01-31,5\n"),
    ):
        result = _download_jkp_csv("https://example.com/x.csv")
    assert result["n"].to_list() == [5]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
