"""Pseudo (simulated) WRDS-shaped data.

Generates synthetic CRSP, Compustat, and CCM-links data with the same
column layout as the real WRDS tables, for testing and tutorials
without a WRDS subscription. Values are simulated and not suitable
for inference.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._internal import _validate_dates


# %% Pseudo identifier universe
# Industry and exchange mixes calibrated to the empirical frequencies of
# the real CRSP universe; SIC codes are drawn from the conventional
# range for the assigned industry so that downstream filters drop the
# intended firms.

_INDUSTRIES = [
    ("Agriculture", 0.00319),
    ("Construction", 0.0113),
    ("Finance", 0.185),
    ("Manufacturing", 0.339),
    ("Mining", 0.0508),
    ("Public", 0.0779),
    ("Retail", 0.0620),
    ("Services", 0.169),
    ("Transportation", 0.0493),
    ("Utilities", 0.0180),
    ("Wholesale", 0.0357),
]

_EXCHANGES = [
    ("AMEX", 0.113),
    ("NASDAQ", 0.671),
    ("NYSE", 0.216),
]

_SIC_RANGES = {
    "Agriculture": (100, 999),
    "Mining": (1000, 1499),
    "Construction": (1500, 1799),
    "Manufacturing": (1800, 3999),
    "Transportation": (4000, 4899),
    "Utilities": (4900, 4999),
    "Wholesale": (5000, 5199),
    "Retail": (5200, 5999),
    "Finance": (6000, 6799),
    "Services": (7000, 8999),
    "Public": (9000, 9999),
}

_PRIMARYEXCH_LOOKUP = {"NYSE": "N", "AMEX": "A", "NASDAQ": "Q"}

_SUPPORTED_PSEUDO_DATASETS = (
    "crsp_monthly",
    "crsp_daily",
    "compustat_annual",
    "compustat_quarterly",
    "ccm_links",
)


def _simulate_pseudo_identifiers(
    n_assets: int = 1000, seed: int = 1234
) -> pd.DataFrame:
    """Draw a pseudo universe of stock identifiers.

    Fully determined by ''(seed, n_assets)'' so calls to different
    pseudo datasets (CRSP, Compustat, CCM links) share the same
    identifier mapping and join cleanly.

    Returns
    -------
    pd.DataFrame
        One row per pseudo firm with columns ''permno'', ''permco'',
        ''gvkey'', ''exchange'', ''industry'', and ''siccd''.
    """
    n_assets = int(n_assets)
    if n_assets <= 0:
        raise ValueError("'n_assets' must be a single positive integer.")

    rng = np.random.default_rng(seed)
    industries = np.array([n for n, _ in _INDUSTRIES])
    industry_probs = np.array([p for _, p in _INDUSTRIES])
    industry_probs = industry_probs / industry_probs.sum()
    exchanges = np.array([n for n, _ in _EXCHANGES])
    exchange_probs = np.array([p for _, p in _EXCHANGES])
    exchange_probs = exchange_probs / exchange_probs.sum()

    exchange = rng.choice(exchanges, size=n_assets, p=exchange_probs)
    industry = rng.choice(industries, size=n_assets, p=industry_probs)
    siccd = np.array([
        rng.integers(_SIC_RANGES[ind][0], _SIC_RANGES[ind][1] + 1)
        for ind in industry
    ])

    return pd.DataFrame({
        "permno": np.arange(1, n_assets + 1),
        "permco": np.arange(1, n_assets + 1),
        "gvkey": [f"{i + 10000:06d}" for i in range(1, n_assets + 1)],
        "exchange": exchange,
        "industry": industry,
        "siccd": siccd,
    })


# %% Router

def _check_supported_dataset_pseudo(dataset: str) -> None:
    """Raise when ''dataset'' is not a supported pseudo dataset."""
    if dataset not in _SUPPORTED_PSEUDO_DATASETS:
        joined = ", ".join(repr(d) for d in _SUPPORTED_PSEUDO_DATASETS)
        raise ValueError(
            f"Unsupported pseudo dataset: {dataset!r}. "
            f"Supported datasets: {joined}."
        )


def _simulate_pseudo_data(
    dataset: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Internal router invoked when ''domain='Pseudo Data'''.

    Validates ''dataset'', emits a notice that pseudo data is being
    returned, and dispatches to the per-dataset generator. Users access
    pseudo data via ''download_data(domain='Pseudo Data', ...)'' or the
    per-dataset ''_download_data_pseudo_*()'' functions.
    """
    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_pseudo(dataset)

    warnings.warn(
        'Returning pseudo data from domain="Pseudo Data". Schema matches '
        'domain="WRDS", but values are simulated and not suitable '
        "for inference.",
        UserWarning,
        stacklevel=2,
    )

    if dataset.startswith("crsp"):
        return _download_data_pseudo_crsp(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    elif dataset.startswith("compustat"):
        return _download_data_pseudo_compustat(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    else:
        return _download_data_pseudo_ccm_links(**kwargs)


# %% Pseudo CRSP

def _download_data_pseudo_crsp(
    dataset: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    version: str = "v2",
    additional_columns: Optional[list] = None,
    add_ccm_links: bool = False,
    adjust_volume: bool = False,
    batch_size: int = 500,
    n_assets: int = 1000,
    seed: int = 1234,
) -> pd.DataFrame:
    """Generate pseudo CRSP data with the WRDS CRSP schema.

    Returns simulated panel data that mirrors the column layout of
    '_download_data_wrds_crsp'. Useful for testing and for reproducing
    the workflow of CRSP-based analyses without a WRDS subscription. The
    generated values are random draws and are not suitable for
    inference. Both 'crsp_monthly' and 'crsp_daily' are supported; the
    daily panel uses weekdays (Monday through Friday) only so the
    calendar approximates a trading-day grid.

    Parameters
    ----------
    dataset : str
        Which CRSP variant to simulate. One of 'crsp_monthly' or
        'crsp_daily'.
    start_date : str or datetime-like, optional
        Inclusive lower bound of the simulated panel, in 'YYYY-MM-DD'
        format. Falls back to a default range when omitted.
    end_date : str or datetime-like, optional
        Inclusive upper bound of the simulated panel, in 'YYYY-MM-DD'
        format.
    version : str, default 'v2'
        Accepted for API compatibility with '_download_data_wrds_crsp';
        the pseudo schema follows the v2 output.
    additional_columns : list of str, optional
        Extra column names appended to the panel. Filled with plausible
        random draws so call sites continue to work; values themselves
        are not economically meaningful.
    add_ccm_links : bool, default False
        When 'True', a 'gvkey' column derived from the same pseudo
        identifier universe used by '_download_data_pseudo_ccm_links' is
        appended.
    adjust_volume : bool, default False
        Accepted for API compatibility; ignored for pseudo data.
    batch_size : int, default 500
        Accepted for API compatibility; ignored for pseudo data.
    n_assets : int, default 1000
        Number of pseudo firms in the universe.
    seed : int, default 1234
        Random seed controlling the pseudo identifier universe and the
        simulated values. Identical '(seed, n_assets)' pairs produce
        identical output across calls and match the identifier universe
        used by '_download_data_pseudo_compustat' and
        '_download_data_pseudo_ccm_links'.

    Returns
    -------
    pandas.DataFrame
        For 'crsp_monthly', a DataFrame with columns 'permno', 'date',
        'calculation_date', 'ret', 'shrout', 'prc', 'primaryexch',
        'siccd', 'listing_age', 'mktcap', 'mktcap_lag', 'exchange',
        'industry', and 'ret_excess'. For 'crsp_daily', a DataFrame
        with columns 'permno', 'date', 'ret', and 'ret_excess'. When
        'add_ccm_links=True', a 'gvkey' column is appended.

    Examples
    --------
    >>> from tidyfinance._pseudo import _download_data_pseudo_crsp
    >>> monthly = _download_data_pseudo_crsp(
    ...     'crsp_monthly',
    ...     start_date='2020-01-01',
    ...     end_date='2024-12-31',
    ...     n_assets=20,
    ... )
    >>> daily = _download_data_pseudo_crsp(
    ...     'crsp_daily',
    ...     start_date='2020-01-01',
    ...     end_date='2020-03-31',
    ...     n_assets=20,
    ... )
    """
    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")
    if dataset not in ("crsp_monthly", "crsp_daily"):
        raise ValueError(
            f"Unsupported CRSP dataset: {dataset!r}. Supported pseudo "
            "datasets: 'crsp_monthly', 'crsp_daily'."
        )

    start_date, end_date = _validate_dates(
        start_date, end_date, use_default_range=True
    )

    identifiers = _simulate_pseudo_identifiers(
        n_assets=n_assets, seed=seed
    )

    if dataset == "crsp_monthly":
        panel = _simulate_pseudo_crsp_monthly(
            identifiers, start_date, end_date, additional_columns, seed
        )
    else:
        panel = _simulate_pseudo_crsp_daily(
            identifiers, start_date, end_date, additional_columns, seed
        )

    if add_ccm_links:
        panel = panel.merge(
            identifiers[["permno", "gvkey"]], on="permno", how="left"
        )

    return panel


def _simulate_pseudo_crsp_monthly(
    identifiers: pd.DataFrame,
    start_date,
    end_date,
    additional_columns,
    seed: int,
) -> pd.DataFrame:
    """Monthly CRSP pseudo panel."""
    start_m = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end_m = pd.Timestamp(end_date).to_period("M").to_timestamp()
    months = pd.date_range(start_m, end_m, freq="MS")

    rng = np.random.default_rng(seed + 1)

    panel = (
        identifiers.merge(pd.DataFrame({"date": months}), how="cross")
        .sort_values(["permno", "date"])
        .reset_index(drop=True)
    )
    n = len(panel)

    panel = panel.assign(
        calculation_date=panel["date"] + pd.offsets.MonthEnd(0),
        shrout=rng.uniform(1, 50, size=n) * 1000,
        prc=rng.uniform(1, 1000, size=n),
        ret=rng.normal(0.008, 0.10, size=n),
    )
    panel = panel.assign(
        mktcap=panel["shrout"] * panel["prc"] / 1000,
        primaryexch=panel["exchange"].map(_PRIMARYEXCH_LOOKUP),
    )
    panel = panel.assign(
        listing_age=panel.groupby("permno").cumcount(),
        mktcap_lag=panel.groupby("permno")["mktcap"].shift(1),
    )
    panel = panel.assign(
        ret_excess=np.maximum(
            panel["ret"] - rng.uniform(0, 0.004, size=n), -1
        )
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel[col] = rng.normal(size=n)

    base_cols = [
        "permno", "date", "calculation_date", "ret", "shrout", "prc",
        "primaryexch", "siccd", "listing_age", "mktcap", "mktcap_lag",
        "exchange", "industry", "ret_excess",
    ]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel[base_cols + extra_cols]


def _simulate_pseudo_crsp_daily(
    identifiers: pd.DataFrame,
    start_date,
    end_date,
    additional_columns,
    seed: int,
) -> pd.DataFrame:
    """Daily CRSP pseudo panel — weekdays only."""
    all_days = pd.date_range(start_date, end_date, freq="D")
    weekdays = all_days[all_days.weekday < 5]

    rng = np.random.default_rng(seed + 2)

    panel = (
        identifiers[["permno"]]
        .merge(pd.DataFrame({"date": weekdays}), how="cross")
        .sort_values(["permno", "date"])
        .reset_index(drop=True)
    )
    n = len(panel)

    panel = panel.assign(
        ret=rng.normal(0.0004, 0.02, size=n),
    )
    panel = panel.assign(
        ret_excess=np.maximum(
            panel["ret"] - rng.uniform(0, 0.0002, size=n), -1
        )
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel[col] = rng.normal(size=n)

    base_cols = ["permno", "date", "ret", "ret_excess"]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel[base_cols + extra_cols]


# %% Pseudo Compustat

def _download_data_pseudo_compustat(
    dataset: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    additional_columns: Optional[list] = None,
    only_usd: bool = False,
    n_assets: int = 1000,
    seed: int = 1234,
) -> pd.DataFrame:
    """Generate pseudo Compustat data with the WRDS schema.

    Returns simulated panel data that mirrors the column layout of
    '_download_data_wrds_compustat'. Useful for testing and for
    reproducing the workflow of Compustat-based analyses without a WRDS
    subscription. The generated values are random draws and are not
    suitable for inference. Both 'compustat_annual' and
    'compustat_quarterly' are supported.

    Parameters
    ----------
    dataset : str
        Which Compustat variant to simulate. One of 'compustat_annual'
        or 'compustat_quarterly'.
    start_date : str or datetime-like, optional
        Inclusive lower bound of the simulated panel, in 'YYYY-MM-DD'
        format.
    end_date : str or datetime-like, optional
        Inclusive upper bound of the simulated panel, in 'YYYY-MM-DD'
        format.
    additional_columns : list of str, optional
        Extra column names appended to the panel. Filled with plausible
        random draws; values themselves are not economically meaningful.
    only_usd : bool, default False
        Accepted for API compatibility with
        '_download_data_wrds_compustat'. The pseudo universe is treated
        as USD-denominated, so this flag has no effect.
    n_assets : int, default 1000
        Number of pseudo firms in the universe.
    seed : int, default 1234
        Random seed controlling the pseudo identifier universe and the
        simulated values. Identical '(seed, n_assets)' pairs produce
        identical output across calls and match the identifier universe
        used by '_download_data_pseudo_crsp' and
        '_download_data_pseudo_ccm_links'.

    Returns
    -------
    pandas.DataFrame
        For 'compustat_annual', a DataFrame with 'gvkey', 'date',
        'datadate', the financial-statement variables 'seq', 'ceq',
        'at', 'lt', 'txditc', 'txdb', 'itcb', 'pstkrv', 'pstkl',
        'pstk', 'capx', 'oancf', 'sale', 'cogs', 'xint', 'xsga', 'ib',
        'curcd', plus the derived 'be', 'op', 'at_lag', 'inv', and any
        requested 'additional_columns'. For 'compustat_quarterly', a
        DataFrame with 'gvkey', 'date', 'datadate', 'atq', 'ceqq', and
        any requested 'additional_columns'.

    Examples
    --------
    >>> from tidyfinance._pseudo import _download_data_pseudo_compustat
    >>> annual = _download_data_pseudo_compustat(
    ...     'compustat_annual',
    ...     start_date='2020-01-01',
    ...     end_date='2024-12-31',
    ...     n_assets=20,
    ... )
    >>> quarterly = _download_data_pseudo_compustat(
    ...     'compustat_quarterly',
    ...     start_date='2020-01-01',
    ...     end_date='2024-12-31',
    ...     n_assets=20,
    ... )
    """
    _ = only_usd  # kept for API parity
    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")
    if dataset not in ("compustat_annual", "compustat_quarterly"):
        raise ValueError(
            f"Unsupported Compustat dataset: {dataset!r}. Supported "
            "pseudo datasets: 'compustat_annual', "
            "'compustat_quarterly'."
        )

    start_date, end_date = _validate_dates(
        start_date, end_date, use_default_range=True
    )

    identifiers = _simulate_pseudo_identifiers(
        n_assets=n_assets, seed=seed
    )

    if dataset == "compustat_annual":
        return _simulate_pseudo_compustat_annual(
            identifiers, start_date, end_date, additional_columns, seed
        )
    return _simulate_pseudo_compustat_quarterly(
        identifiers, start_date, end_date, additional_columns, seed
    )


def _simulate_pseudo_compustat_annual(
    identifiers: pd.DataFrame,
    start_date,
    end_date,
    additional_columns,
    seed: int,
) -> pd.DataFrame:
    """Annual Compustat pseudo panel."""
    years = np.arange(
        pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1
    )
    rng = np.random.default_rng(seed + 4)

    panel = (
        identifiers[["gvkey"]]
        .merge(pd.DataFrame({"year": years}), how="cross")
        .sort_values(["gvkey", "year"])
        .reset_index(drop=True)
    )
    n = len(panel)

    # AR-1-like cumulative growth per gvkey -> at
    panel["growth"] = rng.normal(0.05, 0.30, size=n)
    panel["at"] = panel.groupby("gvkey")["growth"].transform(
        lambda x: 100 * np.exp(np.cumsum(x))
    )
    panel = panel.drop(columns="growth")

    datadate = pd.to_datetime(panel["year"].astype(str) + "-12-31")
    panel = panel.assign(
        datadate=datadate,
        date=datadate.dt.to_period("M").dt.to_timestamp(),
        seq=panel["at"] * rng.uniform(0.3, 0.7, size=n),
    )
    panel = panel.assign(
        ceq=panel["seq"] * rng.uniform(0.8, 1.0, size=n),
        lt=panel["at"] - panel["seq"],
        txditc=panel["at"] * rng.uniform(0.0, 0.05, size=n),
    )
    panel = panel.assign(
        txdb=panel["txditc"] * rng.uniform(0.0, 1.0, size=n),
    )
    panel = panel.assign(
        itcb=panel["txditc"] - panel["txdb"],
        pstkrv=panel["at"] * rng.uniform(0.0, 0.02, size=n),
    )
    panel = panel.assign(
        pstkl=panel["pstkrv"],
        pstk=panel["pstkrv"],
        capx=panel["at"] * rng.uniform(0.02, 0.10, size=n),
        oancf=panel["at"] * rng.normal(0.07, 0.05, size=n),
        sale=panel["at"] * rng.uniform(0.5, 1.5, size=n),
    )
    panel = panel.assign(
        cogs=panel["sale"] * rng.uniform(0.5, 0.8, size=n),
        xsga=panel["sale"] * rng.uniform(0.05, 0.20, size=n),
        xint=panel["at"] * rng.uniform(0.005, 0.03, size=n),
        ib=panel["at"] * rng.normal(0.05, 0.10, size=n),
        curcd="USD",
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel[col] = rng.normal(size=n)

    panel = panel.assign(
        be=(
            panel["seq"]
            .combine_first(panel["ceq"] + panel["pstk"])
            .combine_first(panel["at"] - panel["lt"])
            + panel["txditc"].combine_first(
                panel["txdb"] + panel["itcb"]
            ).fillna(0)
            - panel["pstkrv"]
            .combine_first(panel["pstkl"])
            .combine_first(panel["pstk"])
            .fillna(0)
        ),
    )
    panel = panel.assign(
        op=(
            panel["sale"]
            - panel["cogs"].fillna(0)
            - panel["xsga"].fillna(0)
            - panel["xint"].fillna(0)
        ) / panel["be"]
    )

    lag = (
        panel[["gvkey", "year", "at"]]
        .rename(columns={"at": "at_lag"})
        .assign(year=lambda df: df["year"] + 1)
    )
    panel = panel.merge(lag, on=["gvkey", "year"], how="left")
    panel = panel.assign(
        inv=np.where(
            panel["at_lag"] <= 0, np.nan, panel["at"] / panel["at_lag"] - 1
        )
    )

    first_cols = ["gvkey", "date", "datadate"]
    other_cols = [c for c in panel.columns if c not in first_cols + ["year"]]
    return panel[first_cols + other_cols]


def _simulate_pseudo_compustat_quarterly(
    identifiers: pd.DataFrame,
    start_date,
    end_date,
    additional_columns,
    seed: int,
) -> pd.DataFrame:
    """Quarterly Compustat pseudo panel."""
    start_q = pd.Timestamp(start_date).to_period("Q").to_timestamp()
    end_q = pd.Timestamp(end_date).to_period("Q").to_timestamp()
    q_starts = pd.date_range(start_q, end_q, freq="QS")
    q_ends = q_starts + pd.offsets.QuarterEnd(0)

    rng = np.random.default_rng(seed + 3)

    panel = (
        identifiers[["gvkey"]]
        .merge(pd.DataFrame({"datadate": q_ends}), how="cross")
        .sort_values(["gvkey", "datadate"])
        .reset_index(drop=True)
    )
    n = len(panel)

    panel["growth"] = rng.normal(0.012, 0.15, size=n)
    panel["atq"] = panel.groupby("gvkey")["growth"].transform(
        lambda x: 100 * np.exp(np.cumsum(x))
    )
    panel = panel.drop(columns="growth")

    panel = panel.assign(
        date=panel["datadate"].dt.to_period("M").dt.to_timestamp(),
        ceqq=panel["atq"] * rng.uniform(0.2, 0.6, size=n),
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel[col] = rng.normal(size=n)

    base_cols = ["gvkey", "date", "datadate", "atq", "ceqq"]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel[base_cols + extra_cols]


# %% Pseudo CCM links

def _download_data_pseudo_ccm_links(
    n_assets: int = 1000,
    seed: int = 1234,
    linktype: Optional[list] = None,
    linkprim: Optional[list] = None,
) -> pd.DataFrame:
    """Generate a pseudo CRSP-Compustat linking table.

    Returns a simulated linking table with the same column layout as
    '_download_data_wrds_ccm_links'. Every pseudo 'permno' is linked to
    its corresponding 'gvkey' for the full sample horizon (1925 through
    2099). The 'linktype' and 'linkprim' arguments are accepted for API
    compatibility and ignored.

    Parameters
    ----------
    n_assets : int, default 1000
        Number of pseudo firms in the universe.
    seed : int, default 1234
        Random seed controlling the pseudo identifier universe.
    linktype : list of str, optional
        Accepted for API compatibility with
        '_download_data_wrds_ccm_links'; ignored for pseudo data.
    linkprim : list of str, optional
        Accepted for API compatibility with
        '_download_data_wrds_ccm_links'; ignored for pseudo data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'permno', 'gvkey', 'linkdt', and
        'linkenddt', one row per pseudo firm.

    Examples
    --------
    >>> from tidyfinance._pseudo import _download_data_pseudo_ccm_links
    >>> links = _download_data_pseudo_ccm_links(n_assets=10)
    """
    _ = (linktype, linkprim)
    identifiers = _simulate_pseudo_identifiers(
        n_assets=n_assets, seed=seed
    )
    return identifiers[["permno", "gvkey"]].assign(
        linkdt=pd.Timestamp("1925-12-31"),
        linkenddt=pd.Timestamp("2099-12-31"),
    )[["permno", "gvkey", "linkdt", "linkenddt"]]
