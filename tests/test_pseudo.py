"""Tests for pseudo (simulated) WRDS-shaped data."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_pseudo import _download_data_pseudo_ccm_links, _download_data_pseudo_compustat, _download_data_pseudo_crsp, _simulate_pseudo_data, _simulate_pseudo_identifiers
from tidyfinance.download import download_data  # noqa: E402

# %% router


def test_simulate_pseudo_data_requires_dataset():
    """Raise when no dataset is supplied."""
    with pytest.raises(ValueError, match="'dataset' is required"):
        _simulate_pseudo_data()


def test_simulate_pseudo_data_rejects_unsupported_dataset():
    """Raise an informative error for an unknown pseudo dataset."""
    with pytest.raises(ValueError, match="Unsupported pseudo dataset"):
        _simulate_pseudo_data("bad")


def test_simulate_pseudo_data_emits_notice():
    """Emit a UserWarning mentioning 'pseudo' on dispatch."""
    with pytest.warns(UserWarning, match="pseudo"):
        _simulate_pseudo_data(
            "crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-03-31",
            n_assets=2,
        )


def test_simulate_pseudo_data_dispatches_to_crsp():
    """Route crsp_* datasets through _download_data_pseudo_crsp."""
    with (
        patch(
            "tidyfinance.download_pseudo._download_data_pseudo_crsp",
            return_value=pd.DataFrame({"kind": ["crsp"]}),
        ),
        pytest.warns(UserWarning),
    ):
        out = _simulate_pseudo_data("crsp_monthly", n_assets=2)
    assert out["kind"].iloc[0] == "crsp"


def test_simulate_pseudo_data_dispatches_to_compustat():
    """Route compustat_* datasets through _download_data_pseudo_compustat."""
    with (
        patch(
            "tidyfinance.download_pseudo._download_data_pseudo_compustat",
            return_value=pd.DataFrame({"kind": ["compustat"]}),
        ),
        pytest.warns(UserWarning),
    ):
        out = _simulate_pseudo_data("compustat_annual", n_assets=2)
    assert out["kind"].iloc[0] == "compustat"


def test_simulate_pseudo_data_dispatches_to_ccm():
    """Route ccm_links through _download_data_pseudo_ccm_links."""
    with (
        patch(
            "tidyfinance.download_pseudo._download_data_pseudo_ccm_links",
            return_value=pd.DataFrame({"kind": ["ccm"]}),
        ),
        pytest.warns(UserWarning),
    ):
        out = _simulate_pseudo_data("ccm_links", n_assets=2)
    assert out["kind"].iloc[0] == "ccm"


def test_download_data_routes_pseudo():
    """download_data(domain='Pseudo Data', ...) reaches _simulate_pseudo_data."""
    with patch(
        "tidyfinance.download._simulate_pseudo_data",
        return_value=pd.DataFrame({"sentinel": [1]}),
    ):
        out = download_data(
            domain="Pseudo Data",
            dataset="crsp_monthly",
            start_date="2020-01-01",
            end_date="2020-06-30",
        )
    assert out["sentinel"].iloc[0] == 1


# %% pseudo CRSP


def test_pseudo_crsp_monthly_schema():
    """Monthly CRSP returns expected columns and row count."""
    crsp = _download_data_pseudo_crsp(
        dataset="crsp_monthly",
        start_date="2020-01-01",
        end_date="2020-06-30",
        n_assets=5,
        seed=1234,
    )
    expected = {
        "permno",
        "date",
        "calculation_date",
        "ret",
        "shrout",
        "prc",
        "primaryexch",
        "siccd",
        "listing_age",
        "mktcap",
        "mktcap_lag",
        "exchange",
        "industry",
        "ret_excess",
    }
    assert expected.issubset(set(crsp.columns))
    assert "gvkey" not in crsp.columns
    assert len(crsp) == 5 * 6


def test_pseudo_crsp_add_ccm_links_appends_gvkey():
    """add_ccm_links=True appends gvkey from the shared universe."""
    crsp = _download_data_pseudo_crsp(
        dataset="crsp_monthly",
        start_date="2020-01-01",
        end_date="2020-06-30",
        add_ccm_links=True,
        n_assets=5,
        seed=1234,
    )
    ccm = _download_data_pseudo_ccm_links(n_assets=5, seed=1234)
    assert "gvkey" in crsp.columns
    assert set(crsp["gvkey"].unique()) == set(ccm["gvkey"])


def test_pseudo_crsp_daily_schema_weekdays_only():
    """Daily CRSP returns weekday-only rows and the right schema."""
    crsp_daily = _download_data_pseudo_crsp(
        dataset="crsp_daily",
        start_date="2020-01-01",
        end_date="2020-01-31",
        n_assets=4,
        seed=1234,
    )
    assert list(crsp_daily.columns) == ["permno", "date", "ret", "ret_excess"]
    assert len(crsp_daily) == 4 * 23  # 23 weekdays in Jan 2020
    assert (crsp_daily["date"].dt.weekday < 5).all()


def test_pseudo_crsp_validates_dataset():
    """Per-dataset CRSP generator validates its dataset argument."""
    with pytest.raises(ValueError, match="'dataset' is required"):
        _download_data_pseudo_crsp()
    with pytest.raises(ValueError, match="Unsupported CRSP dataset"):
        _download_data_pseudo_crsp("compustat_annual")


def test_pseudo_crsp_additional_columns_monthly():
    """additional_columns are honored on monthly CRSP."""
    crsp = _download_data_pseudo_crsp(
        "crsp_monthly",
        "2020-01-01",
        "2020-03-31",
        additional_columns=["vol"],
        n_assets=3,
        seed=1234,
    )
    assert "vol" in crsp.columns


def test_pseudo_crsp_additional_columns_and_ccm_links_daily():
    """additional_columns and add_ccm_links work on daily CRSP."""
    crsp_daily = _download_data_pseudo_crsp(
        "crsp_daily",
        "2020-01-01",
        "2020-01-15",
        additional_columns=["vol"],
        add_ccm_links=True,
        n_assets=3,
        seed=1234,
    )
    assert {"vol", "gvkey"}.issubset(crsp_daily.columns)


# %% pseudo Compustat


def test_pseudo_compustat_annual_schema():
    """Annual Compustat returns expected columns and row count."""
    comp = _download_data_pseudo_compustat(
        dataset="compustat_annual",
        start_date="2020-01-01",
        end_date="2024-12-31",
        n_assets=5,
        seed=1234,
    )
    expected = {
        "gvkey",
        "date",
        "datadate",
        "at",
        "ib",
        "be",
        "op",
        "inv",
        "at_lag",
    }
    assert expected.issubset(set(comp.columns))
    assert len(comp) == 5 * 5


def test_pseudo_compustat_quarterly_schema():
    """Quarterly Compustat returns quarter-end datadates."""
    compq = _download_data_pseudo_compustat(
        dataset="compustat_quarterly",
        start_date="2020-01-01",
        end_date="2020-12-31",
        n_assets=5,
        seed=1234,
    )
    assert list(compq.columns) == ["gvkey", "date", "datadate", "atq", "ceqq"]
    assert len(compq) == 5 * 4
    month_day = compq["datadate"].dt.strftime("%m-%d")
    assert month_day.isin(["03-31", "06-30", "09-30", "12-31"]).all()


def test_pseudo_compustat_validates_dataset():
    """Per-dataset Compustat generator validates its dataset argument."""
    with pytest.raises(ValueError, match="'dataset' is required"):
        _download_data_pseudo_compustat()
    with pytest.raises(ValueError, match="Unsupported Compustat dataset"):
        _download_data_pseudo_compustat("crsp_monthly")


def test_pseudo_compustat_additional_columns_annual():
    """additional_columns are honored on annual Compustat."""
    comp = _download_data_pseudo_compustat(
        dataset="compustat_annual",
        start_date="2020-01-01",
        end_date="2022-12-31",
        additional_columns=["ib", "ni"],
        n_assets=3,
        seed=1234,
    )
    assert {"ib", "ni"}.issubset(comp.columns)


def test_pseudo_compustat_additional_columns_quarterly():
    """additional_columns are honored on quarterly Compustat."""
    compq = _download_data_pseudo_compustat(
        "compustat_quarterly",
        "2020-01-01",
        "2020-06-30",
        additional_columns=["saleq", "niq"],
        n_assets=3,
        seed=1234,
    )
    assert {"saleq", "niq"}.issubset(compq.columns)


# %% pseudo CCM links


def test_pseudo_ccm_links_full_universe():
    """CCM links cover the full identifier universe."""
    ccm = _download_data_pseudo_ccm_links(n_assets=10, seed=1234)
    assert len(ccm) == 10
    assert list(ccm.columns) == ["permno", "gvkey", "linkdt", "linkenddt"]


# %% determinism + cross-dataset consistency


def test_pseudo_output_is_deterministic_in_seed():
    """Same (seed, n_assets) yields identical output across calls."""
    a1 = _download_data_pseudo_crsp(
        "crsp_monthly",
        "2020-01-01",
        "2020-03-31",
        n_assets=5,
        seed=1234,
    )
    a2 = _download_data_pseudo_crsp(
        "crsp_monthly",
        "2020-01-01",
        "2020-03-31",
        n_assets=5,
        seed=1234,
    )
    pd.testing.assert_frame_equal(a1, a2)

    b1 = _download_data_pseudo_crsp(
        "crsp_monthly",
        "2020-01-01",
        "2020-03-31",
        n_assets=5,
        seed=42,
    )
    assert not np.array_equal(a1["ret"].to_numpy(), b1["ret"].to_numpy())


def test_pseudo_identifier_universe_matches_across_datasets():
    """CRSP, Compustat, and CCM share the same identifier universe."""
    crsp = _download_data_pseudo_crsp(
        "crsp_monthly",
        "2020-01-01",
        "2020-06-30",
        add_ccm_links=True,
        n_assets=7,
        seed=1234,
    )
    comp = _download_data_pseudo_compustat(
        "compustat_annual",
        "2020-01-01",
        "2024-12-31",
        n_assets=7,
        seed=1234,
    )
    ccm = _download_data_pseudo_ccm_links(n_assets=7, seed=1234)

    assert set(crsp["gvkey"].unique()) == set(ccm["gvkey"])
    assert set(crsp["gvkey"].unique()) == set(comp["gvkey"].unique())
    assert set(crsp["permno"].unique()) == set(ccm["permno"])


# %% identifier helper


def test_simulate_pseudo_identifiers_rejects_invalid_n_assets():
    """n_assets must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        _simulate_pseudo_identifiers(n_assets=0)
    with pytest.raises(ValueError, match="positive integer"):
        _simulate_pseudo_identifiers(n_assets=-5)


# %% run all tests
if __name__ == "__main__":
    pytest.main([__file__])
