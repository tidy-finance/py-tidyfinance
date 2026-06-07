"""Tests for ``tidyfinance.supported_datasets``."""

from __future__ import annotations

import pandas as pd
import pytest

from tidyfinance.supported_datasets import (
    _FF_DATASETS,
    _FF_LEGACY_DATASETS,
    _MACRO_DATASETS,
    _OTHER_DATASETS,
    _PSEUDO_DATASETS,
    _Q_DATASETS,
    _WRDS_DATASETS,
    _check_supported_domain,
    _is_legacy_type,
    _parse_type_to_domain_dataset,
    list_supported_datasets,
)


# %% Module-level table structure

def test_ff_table_has_expected_columns_and_domain():
    assert _FF_DATASETS, "FF table should not be empty"
    assert {"type", "dataset_name", "file_url", "domain"} <= set(
        _FF_DATASETS[0].keys()
    )
    assert all(row["domain"] == "Fama-French" for row in _FF_DATASETS)


def test_ff_legacy_table_has_expected_columns_and_domain():
    assert _FF_LEGACY_DATASETS, "FF legacy table should not be empty"
    assert {"type", "dataset_name", "file_url", "domain"} <= set(
        _FF_LEGACY_DATASETS[0].keys()
    )
    assert all(
        row["domain"] == "Fama-French" for row in _FF_LEGACY_DATASETS
    )


def test_q_table_has_expected_columns_and_domain():
    assert _Q_DATASETS, "Q table should not be empty"
    assert {"type", "dataset_name", "domain"} <= set(_Q_DATASETS[0].keys())
    assert all(row["domain"] == "Global Q" for row in _Q_DATASETS)


def test_macro_table_has_expected_columns_and_domain():
    assert _MACRO_DATASETS, "Macro table should not be empty"
    assert {"type", "dataset_name", "domain"} <= set(
        _MACRO_DATASETS[0].keys()
    )
    assert all(row["domain"] == "Goyal-Welch" for row in _MACRO_DATASETS)


def test_wrds_table_has_expected_columns_and_domain():
    assert _WRDS_DATASETS, "WRDS table should not be empty"
    assert {"type", "dataset_name", "domain"} <= set(
        _WRDS_DATASETS[0].keys()
    )
    assert all(row["domain"] == "WRDS" for row in _WRDS_DATASETS)


def test_pseudo_table_has_expected_columns_and_domain():
    assert _PSEUDO_DATASETS, "Pseudo table should not be empty"
    assert {"type", "dataset_name", "domain"} <= set(
        _PSEUDO_DATASETS[0].keys()
    )
    assert all(row["domain"] == "pseudo" for row in _PSEUDO_DATASETS)


def test_other_table_has_expected_columns():
    assert _OTHER_DATASETS, "Other table should not be empty"
    assert {"type", "dataset_name", "domain"} <= set(
        _OTHER_DATASETS[0].keys()
    )


# %% list_supported_datasets()

def test_default_call_returns_dataframe_with_all_domains():
    result = list_supported_datasets()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["type", "dataset_name", "domain"]
    present = set(result["domain"].unique())
    assert {"Fama-French", "Global Q", "WRDS"}.issubset(present)


def test_default_call_excludes_ff_legacy_table():
    # FF legacy types must NOT appear in the master listing.
    result = list_supported_datasets()
    legacy_only = {
        row["type"]
        for row in _FF_LEGACY_DATASETS
        if row["type"] not in {r["type"] for r in _FF_DATASETS}
    }
    assert legacy_only.isdisjoint(set(result["type"]))


def test_domain_filter_returns_only_matching_rows():
    result = list_supported_datasets(domain="WRDS")
    assert (result["domain"] == "WRDS").all()
    assert len(result) == len(_WRDS_DATASETS)


def test_domain_filter_accepts_list():
    result = list_supported_datasets(domain=["WRDS", "Global Q"])
    assert set(result["domain"].unique()) == {"WRDS", "Global Q"}


def test_as_vector_returns_list_of_types():
    result = list_supported_datasets(as_vector=True)
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)
    assert not isinstance(result, pd.DataFrame)


def test_total_row_count_matches_sum_of_components():
    expected = (
        len(_Q_DATASETS)
        + len(_FF_DATASETS)
        + len(_MACRO_DATASETS)
        + len(_WRDS_DATASETS)
        + len(_PSEUDO_DATASETS)
        + len(_OTHER_DATASETS)
    )
    assert len(list_supported_datasets()) == expected


# %% _parse_type_to_domain_dataset

def test_parse_type_ff():
    assert _parse_type_to_domain_dataset("factors_ff_3_monthly") == (
        "factors_ff",
        "Fama/French 3 Factors",
    )


def test_parse_type_ff_legacy():
    assert _parse_type_to_domain_dataset("factors_ff3_monthly") == (
        "factors_ff",
        "Fama/French 3 Factors",
    )


def test_parse_type_q_strips_csv_suffix():
    # The Q dataset names do not actually end in ".csv" in the current
    # tribble, but the suffix-stripping logic must still hold.
    assert _parse_type_to_domain_dataset("factors_q5_daily") == (
        "factors_q",
        "q5_factors_daily_2024",
    )


def test_parse_type_macro_strips_prefix():
    assert _parse_type_to_domain_dataset("macro_predictors_monthly") == (
        "macro_predictors",
        "monthly",
    )


def test_parse_type_wrds_strips_prefix():
    assert _parse_type_to_domain_dataset("wrds_crsp_monthly") == (
        "wrds",
        "crsp_monthly",
    )


def test_parse_type_hf_prefix():
    assert _parse_type_to_domain_dataset("hf_sp500") == (
        "tidyfinance",
        "sp500",
    )


@pytest.mark.parametrize(
    "simple", ["constituents", "fred", "stock_prices", "osap"]
)
def test_parse_type_simple_domain(simple):
    assert _parse_type_to_domain_dataset(simple) == (simple, None)


def test_parse_type_unknown_raises():
    with pytest.raises(ValueError, match="Cannot parse legacy type"):
        _parse_type_to_domain_dataset("unknown")

# %% _is_legacy_type

@pytest.mark.parametrize(
    "simple", ["constituents", "fred", "stock_prices", "osap"]
)
def test_is_legacy_type_false_for_simple_domains(simple):
    assert _is_legacy_type(simple) is False


def test_is_legacy_type_true_for_ff_type():
    assert _is_legacy_type("factors_ff_3_monthly") is True


def test_is_legacy_type_true_for_ff_legacy_type():
    assert _is_legacy_type("factors_ff3_monthly") is True


def test_is_legacy_type_true_for_wrds_type():
    assert _is_legacy_type("wrds_crsp_monthly") is True


def test_is_legacy_type_false_for_tidyfinance_other_type():
    # `risk_free` is in OTHER with domain == "tidyfinance" and must NOT be
    # treated as a legacy type.
    assert _is_legacy_type("risk_free") is False


def test_is_legacy_type_false_for_unknown_string():
    assert _is_legacy_type("missing") is False


# %% _check_supported_domain

@pytest.mark.parametrize(
    "domain",
    [
        "famafrench",
        "factors_ff",
        "globalq",
        "factors_q",
        "macro_predictors",
        "wrds",
        "pseudo",
        "constituents",
        "fred",
        "stock_prices",
        "osap",
        "tidyfinance",
    ],
)
def test_check_supported_domain_accepts_known(domain):
    # Should not raise
    _check_supported_domain(domain)


def test_check_supported_domain_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported domain"):
        _check_supported_domain("unknown")


# %% run all tests
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
