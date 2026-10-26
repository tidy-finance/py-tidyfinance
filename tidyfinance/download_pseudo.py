"""Pseudo (simulated) WRDS-shaped data.

Generates synthetic CRSP, Compustat, and CCM-links data with the same
column layout as the real WRDS tables, for testing and tutorials
without a WRDS subscription. Values are simulated and not suitable
for inference.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Optional

import numpy as np
import polars as pl

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
) -> pl.DataFrame:
    """Draw a pseudo universe of stock identifiers.

    Fully determined by '(seed, n_assets)' so calls to different
    pseudo datasets (CRSP, Compustat, CCM links) share the same
    identifier mapping and join cleanly.

    Returns
    -------
    pl.DataFrame
        One row per pseudo firm with columns 'permno', 'permco',
        'gvkey', 'exchange', 'industry', and 'siccd'.
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
    siccd = np.array(
        [
            rng.integers(_SIC_RANGES[ind][0], _SIC_RANGES[ind][1] + 1)
            for ind in industry
        ]
    )

    return pl.DataFrame(
        {
            "permno": np.arange(1, n_assets + 1),
            "permco": np.arange(1, n_assets + 1),
            "gvkey": [f"{i + 10000:06d}" for i in range(1, n_assets + 1)],
            "exchange": exchange,
            "industry": industry,
            "siccd": siccd,
        }
    )


# %% Router


def _check_supported_dataset_pseudo(dataset: str) -> None:
    """
    Validate that 'dataset' is a supported pseudo-data name.

    Parameters
    ----------
    dataset : str
        Dataset name to check against '_SUPPORTED_PSEUDO_DATASETS'.

    Raises
    ------
    ValueError
        If 'dataset' is not one of the supported pseudo datasets, with
        the full set of accepted names included in the message.
    """
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
    quiet: bool = False,
    **kwargs,
) -> pl.DataFrame:
    """
    Dispatch a pseudo-data request to the matching per-dataset generator.

    Validates 'dataset', emits a UserWarning that the returned panel
    is simulated and unsuitable for inference, and forwards the call
    to 'download_data_pseudo_crsp',
    'download_data_pseudo_compustat', or
    'download_data_pseudo_ccm_links' based on the 'dataset' prefix.
    Reached when callers use 'download_data(domain="Pseudo Data", ...)'.

    Parameters
    ----------
    dataset : str
        Name of the pseudo dataset to simulate. Must be one of the
        names listed in '_SUPPORTED_PSEUDO_DATASETS' (e.g.,
        'crsp_monthly', 'crsp_daily', 'compustat_annual',
        'compustat_quarterly', 'ccm_links').
    start_date : str, optional
        Lower date bound forwarded to the per-dataset generator.
        Ignored for 'ccm_links'.
    end_date : str, optional
        Upper date bound forwarded to the per-dataset generator.
        Ignored for 'ccm_links'.
    quiet : bool, default False
        If True, the pseudo-data notice is not shown. Accepted so that
        'quiet' works the same for 'domain="WRDS"' and
        'domain="Pseudo Data"'.
    **kwargs
        Additional generator-specific arguments (e.g., 'n_assets',
        'seed', 'version', 'additional_columns', 'adjust_volume').

    Returns
    -------
    pl.DataFrame
        Pseudo panel with the column layout of the corresponding WRDS
        dataset.

    Raises
    ------
    ValueError
        If 'dataset' is None or not in '_SUPPORTED_PSEUDO_DATASETS'.
    """
    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_pseudo(dataset)

    if not quiet:
        warnings.warn(
            'Returning pseudo data from domain="Pseudo Data". Schema matches '
            'domain="WRDS", but values are simulated and not suitable '
            "for inference.",
            UserWarning,
            stacklevel=2,
        )

    if dataset.startswith("crsp"):
        return download_data_pseudo_crsp(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    elif dataset.startswith("compustat"):
        return download_data_pseudo_compustat(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    else:
        return download_data_pseudo_ccm_links(**kwargs)


# %% Pseudo CRSP


def download_data_pseudo_crsp(
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
) -> pl.DataFrame:
    """Generate pseudo CRSP data with the WRDS CRSP schema.

    Returns simulated panel data that mirrors the column layout of
    'download_data_wrds_crsp'. Useful for testing and for reproducing
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
        Accepted for API compatibility with 'download_data_wrds_crsp';
        the pseudo schema follows the v2 output.
    additional_columns : list of str, optional
        Extra column names appended to the panel. Filled with plausible
        random draws so call sites continue to work; values themselves
        are not economically meaningful.
    add_ccm_links : bool, default False
        When 'True', a 'gvkey' column derived from the same pseudo
        identifier universe used by 'download_data_pseudo_ccm_links' is
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
        used by 'download_data_pseudo_compustat' and
        'download_data_pseudo_ccm_links'.

    Returns
    -------
    pl.DataFrame
        For 'crsp_monthly', a DataFrame with columns 'permno', 'date',
        'calculation_date', 'ret', 'shrout', 'prc', 'primaryexch',
        'siccd', 'listing_age', 'mktcap', 'mktcap_lag', 'prc_lag',
        'exchange', 'industry', and 'ret_excess'. For 'crsp_daily', a DataFrame
        with columns 'permno', 'date', 'ret', and 'ret_excess'. When
        'add_ccm_links=True', a 'gvkey' column is appended.

    Examples
    --------
    ```python
    from tidyfinance import download_data_pseudo_crsp
    monthly = download_data_pseudo_crsp(
        'crsp_monthly',
        start_date='2020-01-01',
        end_date='2024-12-31',
        n_assets=20,
    )
    daily = download_data_pseudo_crsp(
        'crsp_daily',
        start_date='2020-01-01',
        end_date='2020-03-31',
        n_assets=20,
    )
    ```
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

    identifiers = _simulate_pseudo_identifiers(n_assets=n_assets, seed=seed)

    if dataset == "crsp_monthly":
        panel = _simulate_pseudo_crsp_monthly(
            identifiers, start_date, end_date, additional_columns, seed
        )
    else:
        panel = _simulate_pseudo_crsp_daily(
            identifiers, start_date, end_date, additional_columns, seed
        )

    if add_ccm_links:
        panel = panel.join(
            identifiers.select("permno", "gvkey"),
            on="permno",
            how="left",
            maintain_order="left",
        )

    return panel


def _simulate_pseudo_crsp_monthly(
    identifiers: pl.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    additional_columns,
    seed: int,
) -> pl.DataFrame:
    """
    Generate a monthly CRSP pseudo panel.

    Crosses 'identifiers' with month-start dates between 'start_date'
    and 'end_date', then draws share counts, prices, and returns
    from i.i.d. distributions (shrout uniform in [1, 50]k;
    prc uniform in [1, 1000]; ret Normal(0.008, 0.10)). Derived
    columns ('mktcap', 'mktcap_lag', 'prc_lag', 'listing_age',
    'primaryexch', 'ret_excess') are computed deterministically from
    the draws.

    Parameters
    ----------
    identifiers : pl.DataFrame
        Per-permno identifier frame with at least 'permno', 'exchange',
        'siccd', and 'industry' columns; produced by
        '_simulate_pseudo_identifiers'.
    start_date, end_date : datetime.date
        Inclusive sample range (the first month-start at or before
        'start_date' and the last month-start at or before 'end_date').
    additional_columns : list of str or None
        Extra column names to attach. Each is filled with i.i.d.
        standard normal draws.
    seed : int
        Base seed; the function uses 'seed + 1' to keep draws
        independent of sibling generators.

    Returns
    -------
    pl.DataFrame
        Monthly panel with columns 'permno', 'date',
        'calculation_date', 'ret', 'shrout', 'prc', 'primaryexch',
        'siccd', 'listing_age', 'mktcap', 'mktcap_lag', 'prc_lag',
        'exchange', 'industry', 'ret_excess', followed by any requested
        'additional_columns'.
    """
    months = pl.date_range(
        start_date.replace(day=1),
        end_date.replace(day=1),
        interval="1mo",
        eager=True,
    ).alias("date")

    rng = np.random.default_rng(seed + 1)

    panel = identifiers.join(pl.DataFrame({"date": months}), how="cross").sort(
        "permno", "date", maintain_order=True
    )
    n = panel.height

    shrout = rng.uniform(1, 50, size=n) * 1000
    prc = rng.uniform(1, 1000, size=n)
    ret = rng.normal(0.008, 0.10, size=n)

    panel = panel.with_columns(
        calculation_date=pl.col("date").dt.month_end(),
        shrout=pl.Series(shrout),
        prc=pl.Series(prc),
        ret=pl.Series(ret),
    ).with_columns(
        mktcap=pl.Series(shrout * prc / 1000),
        primaryexch=pl.col("exchange").replace_strict(
            _PRIMARYEXCH_LOOKUP, default=None
        ),
    )
    panel = panel.with_columns(
        listing_age=pl.int_range(pl.len()).over("permno"),
        mktcap_lag=pl.col("mktcap").shift(1).over("permno"),
        prc_lag=pl.col("prc").shift(1).over("permno"),
    )
    ret_excess = np.maximum(ret - rng.uniform(0, 0.004, size=n), -1)
    panel = panel.with_columns(ret_excess=pl.Series(ret_excess))

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel = panel.with_columns(pl.Series(col, rng.normal(size=n)))

    base_cols = [
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
        "prc_lag",
        "exchange",
        "industry",
        "ret_excess",
    ]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel.select(base_cols + extra_cols)


def _simulate_pseudo_crsp_daily(
    identifiers: pl.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    additional_columns,
    seed: int,
) -> pl.DataFrame:
    """
    Generate a daily CRSP pseudo panel restricted to weekdays.

    Crosses 'identifiers' with weekday dates between 'start_date' and
    'end_date' (Mondays through Fridays). For each row draws a daily
    return from Normal(0.0004, 0.02) and a small risk-free haircut to
    produce 'ret_excess'.

    Parameters
    ----------
    identifiers : pl.DataFrame
        Identifier frame; only the 'permno' column is used.
    start_date, end_date : datetime.date
        Inclusive date range. Weekend days are dropped before the
        cross-join.
    additional_columns : list of str or None
        Extra column names to attach. Each is filled with i.i.d.
        standard normal draws.
    seed : int
        Base seed; the function uses 'seed + 2'.

    Returns
    -------
    pl.DataFrame
        Daily panel with columns 'permno', 'date', 'ret',
        'ret_excess', followed by any requested 'additional_columns'.
    """
    all_days = pl.date_range(
        start_date, end_date, interval="1d", eager=True
    ).alias("date")
    weekdays = all_days.filter(all_days.dt.weekday() <= 5)

    rng = np.random.default_rng(seed + 2)

    panel = (
        identifiers.select("permno")
        .join(pl.DataFrame({"date": weekdays}), how="cross")
        .sort("permno", "date", maintain_order=True)
    )
    n = panel.height

    ret = rng.normal(0.0004, 0.02, size=n)
    panel = panel.with_columns(ret=pl.Series(ret))
    ret_excess = np.maximum(ret - rng.uniform(0, 0.0002, size=n), -1)
    panel = panel.with_columns(ret_excess=pl.Series(ret_excess))

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel = panel.with_columns(pl.Series(col, rng.normal(size=n)))

    base_cols = ["permno", "date", "ret", "ret_excess"]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel.select(base_cols + extra_cols)


# %% Pseudo Compustat


def download_data_pseudo_compustat(
    dataset: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    additional_columns: Optional[list] = None,
    only_usd: bool = False,
    n_assets: int = 1000,
    seed: int = 1234,
) -> pl.DataFrame:
    """Generate pseudo Compustat data with the WRDS schema.

    Returns simulated panel data that mirrors the column layout of
    'download_data_wrds_compustat'. Useful for testing and for
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
        'download_data_wrds_compustat'. The pseudo universe is treated
        as USD-denominated, so this flag has no effect.
    n_assets : int, default 1000
        Number of pseudo firms in the universe.
    seed : int, default 1234
        Random seed controlling the pseudo identifier universe and the
        simulated values. Identical '(seed, n_assets)' pairs produce
        identical output across calls and match the identifier universe
        used by 'download_data_pseudo_crsp' and
        'download_data_pseudo_ccm_links'.

    Returns
    -------
    pl.DataFrame
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
    ```python
    from tidyfinance import download_data_pseudo_compustat
    annual = download_data_pseudo_compustat(
        'compustat_annual',
        start_date='2020-01-01',
        end_date='2024-12-31',
        n_assets=20,
    )
    quarterly = download_data_pseudo_compustat(
        'compustat_quarterly',
        start_date='2020-01-01',
        end_date='2024-12-31',
        n_assets=20,
    )
    ```
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

    identifiers = _simulate_pseudo_identifiers(n_assets=n_assets, seed=seed)

    if dataset == "compustat_annual":
        return _simulate_pseudo_compustat_annual(
            identifiers, start_date, end_date, additional_columns, seed
        )
    return _simulate_pseudo_compustat_quarterly(
        identifiers, start_date, end_date, additional_columns, seed
    )


def _simulate_pseudo_compustat_annual(
    identifiers: pl.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    additional_columns,
    seed: int,
) -> pl.DataFrame:
    """
    Generate an annual Compustat pseudo panel.

    Crosses 'identifiers' with calendar years from 'start_date.year'
    through 'end_date.year' inclusive. Total assets ('at') evolve as
    cumulative exponentiated Normal(0.05, 0.30) growth shocks per
    'gvkey' starting from 100. Other balance-sheet items ('seq',
    'ceq', 'lt', 'pstk', etc.) are drawn as fixed fractions of 'at',
    so cross-sectional ratios are plausible without reflecting any
    real firm.

    Parameters
    ----------
    identifiers : pl.DataFrame
        Identifier frame; only the 'gvkey' column is used.
    start_date, end_date : datetime.date
        Inclusive date range. The year component drives the panel;
        sub-annual resolution is ignored.
    additional_columns : list of str or None
        Extra column names to attach. Each is filled with i.i.d.
        standard normal draws.
    seed : int
        Base seed; the function uses 'seed + 4'.

    Returns
    -------
    pl.DataFrame
        Annual panel with 'gvkey', 'datadate' (Dec 31 of each year),
        'date' (month-start of December), and the Compustat-style
        accounting columns, followed by any requested
        'additional_columns'.
    """
    years = np.arange(start_date.year, end_date.year + 1)
    rng = np.random.default_rng(seed + 4)

    panel = (
        identifiers.select("gvkey")
        .join(pl.DataFrame({"year": years}), how="cross")
        .sort("gvkey", "year", maintain_order=True)
    )
    n = panel.height

    # AR-1-like cumulative growth per gvkey -> at. The panel is sorted
    # by gvkey with one contiguous, equally sized block of years per
    # gvkey, so the groupwise cumulative sum is a row-wise cumsum on
    # the reshaped draw matrix (bit-identical to the per-group numpy
    # computation of the pandas reference implementation).
    growth = rng.normal(0.05, 0.30, size=n)
    at = 100 * np.exp(growth.reshape(-1, years.size).cumsum(axis=1)).ravel()
    panel = panel.with_columns(at=pl.Series(at))

    panel = panel.with_columns(
        datadate=pl.date(pl.col("year"), 12, 31),
        date=pl.date(pl.col("year"), 12, 1),
        seq=pl.col("at") * pl.Series(rng.uniform(0.3, 0.7, size=n)),
    )
    panel = panel.with_columns(
        ceq=pl.col("seq") * pl.Series(rng.uniform(0.8, 1.0, size=n)),
        lt=pl.col("at") - pl.col("seq"),
        txditc=pl.col("at") * pl.Series(rng.uniform(0.0, 0.05, size=n)),
    )
    panel = panel.with_columns(
        txdb=pl.col("txditc") * pl.Series(rng.uniform(0.0, 1.0, size=n)),
    )
    panel = panel.with_columns(
        itcb=pl.col("txditc") - pl.col("txdb"),
        pstkrv=pl.col("at") * pl.Series(rng.uniform(0.0, 0.02, size=n)),
    )
    panel = panel.with_columns(
        pstkl=pl.col("pstkrv"),
        pstk=pl.col("pstkrv"),
        capx=pl.col("at") * pl.Series(rng.uniform(0.02, 0.10, size=n)),
        oancf=pl.col("at") * pl.Series(rng.normal(0.07, 0.05, size=n)),
        sale=pl.col("at") * pl.Series(rng.uniform(0.5, 1.5, size=n)),
    )
    panel = panel.with_columns(
        cogs=pl.col("sale") * pl.Series(rng.uniform(0.5, 0.8, size=n)),
        xsga=pl.col("sale") * pl.Series(rng.uniform(0.05, 0.20, size=n)),
        xint=pl.col("at") * pl.Series(rng.uniform(0.005, 0.03, size=n)),
        ib=pl.col("at") * pl.Series(rng.normal(0.05, 0.10, size=n)),
        curcd=pl.lit("USD"),
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel = panel.with_columns(pl.Series(col, rng.normal(size=n)))

    panel = panel.with_columns(
        be=(
            pl.coalesce(
                pl.col("seq"),
                pl.col("ceq") + pl.col("pstk"),
                pl.col("at") - pl.col("lt"),
            )
            + pl.coalesce(
                pl.col("txditc"), pl.col("txdb") + pl.col("itcb")
            ).fill_null(0)
            - pl.coalesce(
                pl.col("pstkrv"), pl.col("pstkl"), pl.col("pstk")
            ).fill_null(0)
        ),
    )
    panel = panel.with_columns(
        op=(
            pl.col("sale")
            - pl.col("cogs").fill_null(0)
            - pl.col("xsga").fill_null(0)
            - pl.col("xint").fill_null(0)
        )
        / pl.col("be")
    )

    panel = panel.with_columns(
        at_lag=pl.col("at").shift(1).over("gvkey"),
    )
    panel = panel.with_columns(
        inv=pl.when(pl.col("at_lag") <= 0)
        .then(None)
        .otherwise(pl.col("at") / pl.col("at_lag") - 1)
    )

    first_cols = ["gvkey", "date", "datadate"]
    other_cols = [c for c in panel.columns if c not in first_cols + ["year"]]
    return panel.select(first_cols + other_cols)


def _simulate_pseudo_compustat_quarterly(
    identifiers: pl.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    additional_columns,
    seed: int,
) -> pl.DataFrame:
    """
    Generate a quarterly Compustat pseudo panel.

    Crosses 'identifiers' with quarter-end dates between 'start_date'
    and 'end_date'. Total assets ('atq') evolve as cumulative
    exponentiated Normal(0.012, 0.15) growth shocks per 'gvkey'
    starting from 100. Common equity ('ceqq') is drawn as a fixed
    fraction of 'atq'.

    Parameters
    ----------
    identifiers : pl.DataFrame
        Identifier frame; only the 'gvkey' column is used.
    start_date, end_date : datetime.date
        Inclusive date range, rounded to quarter starts and ends.
    additional_columns : list of str or None
        Extra column names to attach. Each is filled with i.i.d.
        standard normal draws.
    seed : int
        Base seed; the function uses 'seed + 3'.

    Returns
    -------
    pl.DataFrame
        Quarterly panel with 'gvkey', 'datadate' (quarter-end),
        'date' (month-start), 'atq', 'ceqq', followed by any
        requested 'additional_columns'.
    """
    start_q = start_date.replace(
        month=3 * ((start_date.month - 1) // 3) + 1, day=1
    )
    end_q = end_date.replace(month=3 * ((end_date.month - 1) // 3) + 1, day=1)
    q_starts = pl.date_range(start_q, end_q, interval="3mo", eager=True)
    q_ends = q_starts.dt.offset_by("2mo").dt.month_end().alias("datadate")

    rng = np.random.default_rng(seed + 3)

    panel = (
        identifiers.select("gvkey")
        .join(pl.DataFrame({"datadate": q_ends}), how="cross")
        .sort("gvkey", "datadate", maintain_order=True)
    )
    n = panel.height

    # Same contiguous-block trick as in the annual generator: the
    # groupwise cumulative growth is a row-wise cumsum on the reshaped
    # draw matrix.
    growth = rng.normal(0.012, 0.15, size=n)
    atq = 100 * np.exp(growth.reshape(-1, q_ends.len()).cumsum(axis=1)).ravel()
    panel = panel.with_columns(atq=pl.Series(atq))

    panel = panel.with_columns(
        date=pl.col("datadate").dt.truncate("1mo"),
        ceqq=pl.col("atq") * pl.Series(rng.uniform(0.2, 0.6, size=n)),
    )

    additional_columns = additional_columns or []
    for col in additional_columns:
        if col not in panel.columns:
            panel = panel.with_columns(pl.Series(col, rng.normal(size=n)))

    base_cols = ["gvkey", "date", "datadate", "atq", "ceqq"]
    extra_cols = [c for c in additional_columns if c not in base_cols]
    return panel.select(base_cols + extra_cols)


# %% Pseudo CCM links


def download_data_pseudo_ccm_links(
    n_assets: int = 1000,
    seed: int = 1234,
    linktype: list[str] | tuple[str, ...] = ("LU", "LC"),
    linkprim: list[str] | tuple[str, ...] = ("P", "C"),
) -> pl.DataFrame:
    """Generate a pseudo CRSP-Compustat linking table.

    Returns a simulated linking table with the same column layout as
    'download_data_wrds_ccm_links'. Every pseudo 'permno' is linked to
    its corresponding 'gvkey' for the full sample horizon (1925 through
    2099). The 'linktype' and 'linkprim' arguments are accepted for API
    compatibility and ignored.

    Parameters
    ----------
    n_assets : int, default 1000
        Number of pseudo firms in the universe.
    seed : int, default 1234
        Random seed controlling the pseudo identifier universe.
    linktype : list or tuple of str, optional
        Accepted for API compatibility with
        'download_data_wrds_ccm_links'; ignored for pseudo data.
    linkprim : list or tuple of str, optional
        Accepted for API compatibility with
        'download_data_wrds_ccm_links'; ignored for pseudo data.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns 'permno', 'gvkey', 'linkdt', and
        'linkenddt', one row per pseudo firm.

    Examples
    --------
    ```python
    from tidyfinance import download_data_pseudo_ccm_links
    links = download_data_pseudo_ccm_links(n_assets=10)
    ```
    """
    _ = (linktype, linkprim)
    identifiers = _simulate_pseudo_identifiers(n_assets=n_assets, seed=seed)
    return identifiers.select(
        "permno",
        "gvkey",
        linkdt=pl.lit(dt.date(1925, 12, 31)),
        linkenddt=pl.lit(dt.date(2099, 12, 31)),
    )
