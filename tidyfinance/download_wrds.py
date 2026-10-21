"""WRDS connection and downloads for tidyfinance."""

import contextvars
import datetime as dt
import functools
import inspect
import math
import os
import re
import time
import warnings

import polars as pl
from dotenv import dotenv_values, load_dotenv, set_key
from sqlalchemy import URL, create_engine

from ._internal import (
    _deprecate_legacy_dataset,
    _inform,
    _validate_dates,
    _validate_flag,
)
from .download_tidy_finance import _download_data_risk_free
from .supported_datasets import (
    _check_supported_dataset_wrds,
    _check_supported_dataset_wrds_crsp,
    _is_legacy_type_wrds,
)


# %% Download progress


class _WrdsDownloadStep:
    """Progress of one WRDS download, announced once connected."""

    def __init__(self, what: str):
        self.what = what
        self.start = None

    def announce(self) -> None:
        """Say that the download starts; called by 'get_wrds_connection'
        after the connection succeeded."""
        if self.start is None:
            _inform(
                f"Downloading {self.what} from WRDS. This can take a few "
                "minutes."
            )
            self.start = time.perf_counter()


# The download step of the WRDS download function that is running, so
# that 'get_wrds_connection' can announce it right after connecting.
_WRDS_DOWNLOAD_STEP = contextvars.ContextVar(
    "tidyfinance_wrds_download_step", default=None
)


def _wrds_download_step(label):
    """Report the progress of a WRDS download.

    Decorator for the WRDS download functions. Unless the call passes
    'quiet=True', the download is announced once the WRDS connection
    succeeds ('get_wrds_connection' says when it connects), and when the
    function returns, a message reports the elapsed time, or that the
    download failed.

    Parameters
    ----------
    label : str or callable
        Name of the data, e.g. 'CRSP monthly data'. A callable receives
        the arguments of the call as a dict and returns the name, or
        None to report nothing (daily CRSP shows a batch progress line
        instead).

    Returns
    -------
    callable
        The decorator.
    """

    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                arguments = signature.bind_partial(*args, **kwargs).arguments
            except TypeError:
                # Invalid arguments: let the function raise its own error.
                return func(*args, **kwargs)
            what = label(arguments) if callable(label) else label
            if arguments.get("quiet", False) is not False:
                what = None

            step = _WrdsDownloadStep(what) if what is not None else None
            token = _WRDS_DOWNLOAD_STEP.set(step)
            try:
                result = func(*args, **kwargs)
            except BaseException:
                if step is not None and step.start is not None:
                    _inform(f"Downloading {step.what} from WRDS failed.")
                raise
            finally:
                _WRDS_DOWNLOAD_STEP.reset(token)
            if step is not None and step.start is not None:
                elapsed = time.perf_counter() - step.start
                _inform(f"Downloaded {step.what} from WRDS. [{elapsed:.1f}s]")
            return result

        return wrapper

    return decorator


def _report_batch_progress(j: int, batches: int, quiet: bool) -> None:
    """Show the progress of the daily CRSP download in one line.

    The line is updated in place and cleared after the last batch.

    Parameters
    ----------
    j : int
        Number of the batch that has just been downloaded (1-based).
    batches : int
        Total number of batches.
    quiet : bool
        If True, nothing is shown.
    """
    if quiet:
        return
    line = (
        f"Downloading CRSP daily data from WRDS in batches: {j}/{batches} "
        f"({j / batches:.0%})"
    )
    _inform(f"\r{line}", end="")
    if j == batches:
        _inform("\r" + " " * len(line) + "\r", end="")


def _read_sql(query, con) -> pl.DataFrame:
    """Run a SQL query against an open WRDS connection.

    Single seam for all WRDS SQL reads so tests can patch it and
    return polars fixtures.

    Parameters
    ----------
    query : str
        The SQL query to execute.
    con : object
        An open connection, e.g. from 'get_wrds_connection'.

    Returns
    -------
    pl.DataFrame
        The query result. Postgres 'numeric' columns (inferred as
        'pl.Decimal') are cast to Float64, since downstream arithmetic
        and checks like 'is_infinite' expect floats.
    """
    out = pl.read_database(query, con, infer_schema_length=None)
    decimal_cols = [
        name
        for name, dtype in out.schema.items()
        if isinstance(dtype, pl.Decimal)
    ]
    if decimal_cols:
        out = out.with_columns(
            [pl.col(c).cast(pl.Float64) for c in decimal_cols]
        )
    return out


def _cast_date_columns(df: pl.DataFrame, columns) -> pl.DataFrame:
    """Cast the given columns to 'pl.Date' where present.

    Real Postgres 'date' columns already arrive as 'pl.Date'; this
    normalizes datetime-typed inputs (e.g. test fixtures).
    """
    exprs = [pl.col(c).cast(pl.Date) for c in columns if c in df.columns]
    return df.with_columns(exprs) if exprs else df


def _cast_columns(df: pl.DataFrame, dtypes: dict) -> pl.DataFrame:
    """Cast columns to the given dtypes where present."""
    exprs = [
        pl.col(c).cast(dtype) for c, dtype in dtypes.items() if c in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _download_data_wrds(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Download data from WRDS.

    Acts as a wrapper to download data from various WRDS datasets
    including CRSP, Compustat, and CCM links based on the specified
    dataset. It is designed to handle different datasets by
    redirecting to the appropriate specific data download function.

    Parameters
    ----------
    dataset : str
        A string specifying the dataset to download. Supported values
        are 'crsp_monthly' and 'crsp_daily' for CRSP data,
        'compustat_annual' and 'compustat_quarterly' for Compustat
        data, 'ccm_links' for CCM links data, 'fisd' for FISD data, or
        'trace_enhanced' for TRACE data.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped and a DeprecationWarning is emitted.
    **kwargs
        Additional arguments passed to specific download functions
        depending on the 'dataset'. For instance, 'quiet=True'
        suppresses the messages about the WRDS connection and the
        download progress.

    Returns
    -------
    pl.DataFrame
        A data frame containing the requested data, with the structure
        and contents depending on the specified 'dataset'.

    Examples
    --------
    ```python
    from tidyfinance import download_data_wrds
    crsp_monthly = download_data_wrds(
        'crsp_monthly', '2020-01-01', '2020-12-31'
    )
    compustat_annual = download_data_wrds(
        'compustat_annual', '2020-01-01', '2020-12-31'
    )
    ccm_links = download_data_wrds('ccm_links')
    fisd = download_data_wrds('fisd')
    trace_enhanced = download_data_wrds(
        'trace_enhanced', cusips=['00101JAH9']
    )
    ```
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if dataset is not None and _is_legacy_type_wrds(dataset):
        replacement = re.sub(r"^wrds_", "", dataset)
        _deprecate_legacy_dataset(dataset, replacement)
        dataset = replacement

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_wrds(dataset)

    if dataset.startswith("crsp"):
        return _download_data_wrds_crsp(dataset, start_date, end_date, **kwargs)
    elif dataset.startswith("compustat"):
        return _download_data_wrds_compustat(
            dataset, start_date, end_date, **kwargs
        )
    elif dataset == "ccm_links":
        return _download_data_wrds_ccm_links(**kwargs)
    elif dataset == "fisd":
        return _download_data_wrds_fisd(**kwargs)
    elif dataset == "trace_enhanced":
        return _download_data_wrds_trace_enhanced(
            start_date=start_date, end_date=end_date, **kwargs
        )
    else:
        raise ValueError(f"Unsupported WRDS dataset: {dataset!r}.")


def _listing_age_expr() -> pl.Expr:
    """Months between 'date' and 'first_crsp_date', clipped at 0."""
    return (
        (
            (pl.col("date").dt.year() - pl.col("first_crsp_date").dt.year())
            * 12
            + (pl.col("date").dt.month() - pl.col("first_crsp_date").dt.month())
            - (
                pl.col("date").dt.day() < pl.col("first_crsp_date").dt.day()
            ).cast(pl.Int32)
        )
        .clip(lower_bound=0)
        .alias("listing_age")
    )


def _exchange_expr_v2() -> pl.Expr:
    """Vectorized equivalent of '_assign_exchange' on 'primaryexch'."""
    return (
        pl.when(pl.col("primaryexch") == "N")
        .then(pl.lit("NYSE"))
        .when(pl.col("primaryexch") == "A")
        .then(pl.lit("AMEX"))
        .when(pl.col("primaryexch") == "Q")
        .then(pl.lit("NASDAQ"))
        .otherwise(pl.lit("Other"))
        .alias("exchange")
    )


def _exchange_expr_v1() -> pl.Expr:
    """Map legacy numeric 'exchcd' codes to exchange labels."""
    return (
        pl.when(pl.col("exchcd").is_in([1, 31]))
        .then(pl.lit("NYSE"))
        .when(pl.col("exchcd").is_in([2, 32]))
        .then(pl.lit("AMEX"))
        .when(pl.col("exchcd").is_in([3, 33]))
        .then(pl.lit("NASDAQ"))
        .otherwise(pl.lit("Other"))
        .alias("exchange")
    )


def _industry_expr(siccd: pl.Expr) -> pl.Expr:
    """Vectorized equivalent of '_assign_industry' on a SIC code."""
    return (
        pl.when(siccd.is_between(1, 999))
        .then(pl.lit("Agriculture"))
        .when(siccd.is_between(1000, 1499))
        .then(pl.lit("Mining"))
        .when(siccd.is_between(1500, 1799))
        .then(pl.lit("Construction"))
        .when(siccd.is_between(2000, 3999))
        .then(pl.lit("Manufacturing"))
        .when(siccd.is_between(4000, 4899))
        .then(pl.lit("Transportation"))
        .when(siccd.is_between(4900, 4999))
        .then(pl.lit("Utilities"))
        .when(siccd.is_between(5000, 5199))
        .then(pl.lit("Wholesale"))
        .when(siccd.is_between(5200, 5999))
        .then(pl.lit("Retail"))
        .when(siccd.is_between(6000, 6799))
        .then(pl.lit("Finance"))
        .when(siccd.is_between(7000, 8999))
        .then(pl.lit("Services"))
        .when(siccd.is_between(9000, 9999))
        .then(pl.lit("Public"))
        .otherwise(pl.lit("Missing"))
        .alias("industry")
    )


def _zero_to_null(name: str) -> pl.Expr:
    """Replace exact zeros in a column with null."""
    return (
        pl.when(pl.col(name) == 0)
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
    )


def _inf_to_null(expr: pl.Expr) -> pl.Expr:
    """Replace +/-inf in a float expression with null."""
    return pl.when(expr.is_infinite()).then(None).otherwise(expr)


def _gao_ritter_vol_adj(nasdaq_cond: pl.Expr) -> pl.Expr:
    """Gao and Ritter (2010) NASDAQ volume adjustment."""
    gr_date_1 = dt.date(2001, 2, 1)
    gr_date_2 = dt.date(2002, 1, 1)
    gr_date_3 = dt.date(2004, 1, 1)
    vol = pl.col("vol")
    date = pl.col("date")
    return (
        pl.when(nasdaq_cond & (date < gr_date_1))
        .then(vol / 2.0)
        .when(nasdaq_cond & (date >= gr_date_1) & (date < gr_date_2))
        .then(vol / 1.8)
        .when(nasdaq_cond & (date >= gr_date_2) & (date < gr_date_3))
        .then(vol / 1.6)
        .when(nasdaq_cond & (date >= gr_date_3))
        .then(vol / 1.0)
        .otherwise(vol)
        .alias("vol_adj")
    )


def _crsp_step_label(arguments: dict):
    """Name of the CRSP download step; daily CRSP shows batches instead."""
    requested = arguments.get("dataset") or arguments.get("type") or ""
    return "CRSP monthly data" if "crsp_monthly" in requested else None


@_wrds_download_step(_crsp_step_label)
def _download_data_wrds_crsp(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    batch_size: int = 500,
    version: str = "v2",
    additional_columns: list = None,
    add_ccm_links: bool = False,
    adjust_volume: bool = False,
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Download data from WRDS CRSP.

    Downloads and processes stock return data from the CRSP database
    for a specified period. Users can choose between monthly and daily
    datasets. The function also adjusts returns for delisting and
    calculates market capitalization and excess returns over the
    risk-free rate.

    Parameters
    ----------
    dataset : str
        A string specifying the CRSP dataset to download:
        'crsp_monthly' or 'crsp_daily'.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped and a DeprecationWarning is emitted.
    batch_size : int, optional
        An optional integer specifying the batch size for processing
        daily data, with a default of 500.
    version : str, optional
        An optional character specifying which CRSP version to use.
        'v2' (the default) uses the updated second version of CRSP,
        and 'v1' downloads the legacy version of CRSP.
    additional_columns : list of str, optional
        Extra column names from the underlying CRSP source table to
        return alongside the standard output. For 'crsp_monthly' the
        source is 'crsp.msf_v2' (or 'crsp.msf' joined with
        'crsp.msenames' under 'version="v1"'); for 'crsp_daily' the
        source is 'crsp.dsf_v2' (or 'crsp.dsf' for 'v1'). Pass any
        column from those tables (e.g., 'mthvol', 'mthvolflg' for
        monthly; 'dlyvol', 'dlyfacprc' for daily). When
        'adjust_volume=True' for 'crsp_daily', this list must include
        the columns the adjustment needs ('dlyprc', 'dlyvol',
        'dlyfacprc', 'primaryexch' for 'v2'; 'prc', 'vol', 'cfacpr',
        'exchcd' for 'v1'); a 'ValueError' is raised otherwise.
    add_ccm_links : bool, optional
        A boolean indicating whether CRSP-Compustat links should be
        added automatically using '_download_data_wrds_ccm_links'.
        Defaults to False.
    adjust_volume : bool, optional
        A boolean indicating whether daily CRSP trading volume data
        should be adjusted according to Gao and Ritter (2010).
        Defaults to False. Note that cumulative price adjustment
        factors are computed from the data in memory; results may be
        incorrect if 'start_date' excludes early observations for some
        permnos.
    quiet : bool, optional
        If True, the messages about the WRDS connection and the download
        progress are suppressed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        A data frame containing CRSP stock returns, adjusted for
        delistings, along with calculated market capitalization and
        excess returns over the risk-free rate. The structure of the
        returned data frame depends on the selected dataset.

    References
    ----------
    Gao, X., and Ritter, J. R. (2010). The marketing of seasoned
    equity offerings. Journal of Financial Economics, 97(1), 33-52.
    https://doi.org/10.1016/j.jfineco.2010.03.007

    Examples
    --------
    ```python
    from tidyfinance import download_data_wrds_crsp
    crsp_monthly = download_data_wrds_crsp(
        'crsp_monthly', '2020-11-01', '2020-12-31'
    )
    crsp_daily = download_data_wrds_crsp(
        'crsp_daily', '2020-12-01', '2020-12-31'
    )
    download_data_wrds_crsp(
        'crsp_monthly',
        '2020-11-01',
        '2020-12-31',
        additional_columns=['mthvol', 'mthvolflg'],
    )
    ```
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if dataset is not None and _is_legacy_type_wrds(dataset):
        replacement = re.sub(r"^wrds_", "", dataset)
        _deprecate_legacy_dataset(dataset, replacement)
        dataset = replacement

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_wrds_crsp(dataset)

    start_date, end_date = _validate_dates(start_date, end_date)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be an integer larger than 0.")

    if version not in ["v1", "v2"]:
        raise ValueError("version must be 'v1' or 'v2'.")

    if (
        version == "v1"
        and end_date is not None
        and end_date > dt.date(2024, 12, 31)
    ):
        raise ValueError(
            "end_date must not be later than December 2024 for "
            "version='v1'. CRSP discontinued the legacy version at the "
            "end of 2024. Use version='v2' for more recent data."
        )

    additional_columns_list = additional_columns or []

    if adjust_volume and dataset != "crsp_daily":
        raise ValueError("adjust_volume is only supported for 'crsp_daily'.")

    if adjust_volume:
        if version == "v1":
            required = {"prc", "vol", "cfacpr", "exchcd"}
            cols_str = "prc, vol, cfacpr, and exchcd"
        else:
            required = {"dlyprc", "dlyvol", "dlyfacprc", "primaryexch"}
            cols_str = "dlyprc, dlyvol, primaryexch, and dlyfacprc"
        if not required.issubset(set(additional_columns_list)):
            raise ValueError(
                f"{cols_str} must be contained in additional_columns "
                "for adjust_volume=True."
            )

    wrds_connection = get_wrds_connection(quiet=quiet)
    additional_columns_sql = (
        ", ".join(additional_columns_list) if additional_columns_list else ""
    )

    crsp_data_parts = []
    try:
        if "crsp_monthly" in dataset:
            if version == "v1":
                # Query 1: msf joined with msenames (shrcd 10/11)
                additional_cols_select = (
                    ", "
                    + ", ".join(f"msf.{c}" for c in additional_columns_list)
                    if additional_columns_list
                    else ""
                )
                msf_query = f"""
                    SELECT msf.permno, msf.date, msf.ret, msf.shrout,
                           msf.altprc, msf.cfacpr,
                           msn.exchcd, msn.siccd
                           {additional_cols_select}
                    FROM crsp.msf AS msf
                    INNER JOIN crsp.msenames AS msn
                      ON msf.permno = msn.permno
                      AND msf.date BETWEEN msn.namedt AND msn.nameendt
                      AND msn.shrcd IN (10, 11)
                    WHERE msf.date BETWEEN '{start_date}'
                          AND '{end_date}'
                """
                # WRDS stores v1 permno as double precision; cast to
                # Int64 so joins (e.g. with ccm links) match dtypes.
                msf_data = _cast_columns(
                    _cast_date_columns(
                        _read_sql(msf_query, wrds_connection), ["date"]
                    ),
                    {"permno": pl.Int64, "siccd": pl.Int64},
                )

                # Query 2: msedelist (delisting events)
                msedelist_query = (
                    "SELECT permno, dlstdt, dlret, dlstcd FROM crsp.msedelist"
                )
                msedelist = _cast_columns(
                    _cast_date_columns(
                        _read_sql(msedelist_query, wrds_connection), ["dlstdt"]
                    ),
                    {"permno": pl.Int64},
                )

                # Query 3: first_crsp_date per permno
                first_date_query = (
                    "SELECT permno, MIN(namedt) AS first_crsp_date "
                    "FROM crsp.msenames GROUP BY permno"
                )
                first_crsp_date = _cast_columns(
                    _cast_date_columns(
                        _read_sql(first_date_query, wrds_connection),
                        ["first_crsp_date"],
                    ),
                    {"permno": pl.Int64},
                )

                disconnect_connection(wrds_connection)

                # calculation_date + month-floored date
                crsp_monthly = msf_data.with_columns(
                    calculation_date=pl.col("date"),
                    date=pl.col("date").dt.truncate("1mo"),
                    shrout=pl.col("shrout") * 1000,
                )

                # Join delisting on (permno, month-floored dlstdt)
                if msedelist.height > 0:
                    msedelist = msedelist.with_columns(
                        date=pl.col("dlstdt").dt.truncate("1mo")
                    ).select("permno", "date", "dlret", "dlstcd")
                    crsp_monthly = crsp_monthly.join(
                        msedelist,
                        on=["permno", "date"],
                        how="left",
                        maintain_order="left",
                    )
                else:
                    crsp_monthly = crsp_monthly.with_columns(
                        dlret=pl.lit(None, dtype=pl.Float64),
                        dlstcd=pl.lit(None, dtype=pl.Float64),
                    )

                # listing_age (months elapsed, clipped at 0)
                crsp_monthly = (
                    crsp_monthly.join(
                        first_crsp_date,
                        on="permno",
                        how="left",
                        maintain_order="left",
                    )
                    .with_columns(_listing_age_expr())
                    .drop("first_crsp_date")
                )

                # mktcap (millions); zero -> null
                crsp_monthly = crsp_monthly.with_columns(
                    mktcap=(pl.col("shrout") * pl.col("altprc")).abs() / 1e6
                ).with_columns(_zero_to_null("mktcap"))

                # mktcap_lag and prc_lag via self-join shifted by 1 month.
                # In the legacy SIZ format, a negative price is the
                # average of bid and ask when there was no closing trade;
                # the sign is only a flag, so prc_lag is its absolute value.
                mktcap_lag_df = crsp_monthly.select(
                    "permno",
                    pl.col("date").dt.offset_by("1mo"),
                    pl.col("mktcap").alias("mktcap_lag"),
                    pl.when(pl.col("altprc") == 0)
                    .then(None)
                    .otherwise(pl.col("altprc").abs())
                    .alias("prc_lag"),
                )
                crsp_monthly = crsp_monthly.join(
                    mktcap_lag_df,
                    on=["permno", "date"],
                    how="left",
                    maintain_order="left",
                )

                # exchange via exchcd (numeric, v1 codes) and industry
                # from SIC code
                crsp_monthly = crsp_monthly.with_columns(
                    _exchange_expr_v1(),
                    _industry_expr(pl.col("siccd").fill_null(-1)),
                )

                # ret_adj from delisting code (Shumway-style)
                crsp_monthly = crsp_monthly.with_columns(
                    ret_adj=(
                        pl.when(pl.col("dlstcd").is_null())
                        .then(pl.col("ret"))
                        .when(pl.col("dlret").is_not_null())
                        .then(pl.col("dlret"))
                        .when(
                            pl.col("dlstcd").is_in([500, 520, 580, 584])
                            | pl.col("dlstcd").is_between(551, 574)
                        )
                        .then(-0.30)
                        .when(pl.col("dlstcd") == 100)
                        .then(pl.col("ret"))
                        .otherwise(-1.0)
                    )
                ).drop("dlret", "dlstcd")

                # prc_adj = |altprc nullified-at-zero| / cfacpr
                prc_zeroed = (
                    pl.when(pl.col("altprc") == 0)
                    .then(None)
                    .otherwise(pl.col("altprc"))
                )
                crsp_monthly = crsp_monthly.with_columns(
                    prc_adj=_inf_to_null(prc_zeroed.abs() / pl.col("cfacpr"))
                )

                # Merge risk-free and compute excess return
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date, end_date=end_date
                )
                crsp_monthly = (
                    crsp_monthly.join(
                        risk_free_monthly,
                        how="left",
                        on="date",
                        maintain_order="left",
                    )
                    .with_columns(
                        ret_excess=pl.col("ret_adj") - pl.col("risk_free")
                    )
                    .drop("risk_free")
                    .drop_nulls(subset=["ret_excess", "mktcap"])
                )

                processed_data = crsp_monthly
            elif version == "v2":
                crsp_query = f"""
                    SELECT msf.permno,
                        date_trunc('month', msf.mthcaldt)::date AS date,
                        msf.mthcaldt AS calculation_date,
                        msf.mthret AS ret, msf.shrout,
                        msf.mthprc AS prc,
                        ssih.primaryexch, ssih.siccd,
                        fcd.first_crsp_date
                        {", " + additional_columns_sql if additional_columns_sql else ""}
                    FROM crsp.msf_v2 AS msf
                    INNER JOIN crsp.stksecurityinfohist AS ssih
                    ON msf.permno = ssih.permno AND
                        ssih.secinfostartdt <= msf.mthcaldt AND
                        msf.mthcaldt <= ssih.secinfoenddt
                    LEFT JOIN (
                        SELECT permno,
                            MIN(secinfostartdt) AS first_crsp_date
                        FROM crsp.stksecurityinfohist
                        GROUP BY permno
                    ) AS fcd ON msf.permno = fcd.permno
                    WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}'
                    AND ssih.sharetype = 'NS'
                    AND ssih.securitytype = 'EQTY'
                    AND ssih.securitysubtype = 'COM'
                    AND ssih.usincflg = 'Y'
                    AND ssih.issuertype in ('ACOR', 'CORP')
                    AND ssih.primaryexch in ('N', 'A', 'Q')
                    AND ssih.conditionaltype in ('RW', 'NW')
                    AND ssih.tradingstatusflg = 'A'
                    """

                crsp_monthly = _cast_columns(
                    _cast_date_columns(
                        _read_sql(crsp_query, wrds_connection),
                        ["date", "calculation_date", "first_crsp_date"],
                    ),
                    {"permno": pl.Int64, "siccd": pl.Int64},
                )
                crsp_monthly = (
                    crsp_monthly.with_columns(shrout=pl.col("shrout") * 1000)
                    # listing_age is assigned before mktcap to keep the
                    # documented column order:
                    # ..., siccd, listing_age, mktcap, mktcap_lag, ...
                    .with_columns(_listing_age_expr())
                    .with_columns(
                        mktcap=pl.col("shrout") * pl.col("prc") / 1000000
                    )
                    .with_columns(_zero_to_null("mktcap"))
                    .drop("first_crsp_date")
                )

                # CIZ (v2) prices are always positive, so prc_lag is the
                # previous month's price as is.
                mktcap_lag = crsp_monthly.select(
                    "permno",
                    pl.col("date").dt.offset_by("1mo"),
                    pl.col("mktcap").alias("mktcap_lag"),
                    pl.col("prc").alias("prc_lag"),
                )

                crsp_monthly = crsp_monthly.join(
                    mktcap_lag,
                    how="left",
                    on=["permno", "date"],
                    maintain_order="left",
                ).with_columns(
                    _exchange_expr_v2(),
                    _industry_expr(pl.col("siccd")),
                )
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date,
                    end_date=end_date,
                )
                crsp_monthly = (
                    crsp_monthly.join(
                        risk_free_monthly,
                        how="left",
                        on="date",
                        maintain_order="left",
                    )
                    .with_columns(
                        ret_excess=pl.col("ret") - pl.col("risk_free")
                    )
                    .drop("risk_free")
                    .drop_nulls(subset=["ret_excess", "mktcap"])
                )
                processed_data = crsp_monthly
        elif "crsp_daily" in dataset:
            if version == "v1":
                # Distinct permnos from dsf within the date range
                permnos_query = (
                    f"SELECT DISTINCT permno FROM crsp.dsf "
                    f"WHERE date BETWEEN '{start_date}' "
                    f"AND '{end_date}'"
                )
                permnos = _cast_columns(
                    _read_sql(permnos_query, wrds_connection),
                    {"permno": pl.Int64},
                )
                permnos = [str(p) for p in permnos["permno"].to_list()]

                if len(permnos) > 0:
                    batches = math.ceil(len(permnos) / batch_size)
                    risk_free_daily = _download_data_risk_free(
                        start_date=start_date,
                        end_date=end_date,
                        frequency="daily",
                    )

                    for j in range(1, batches + 1):
                        permno_batch = permnos[
                            ((j - 1) * batch_size) : (
                                min(j * batch_size, len(permnos))
                            )
                        ]
                        permno_batch_str = ", ".join(
                            f"'{p}'" for p in permno_batch
                        )
                        permno_in = f"({permno_batch_str})"

                        dsf_query = f"""
                            SELECT dsf.permno, dsf.date, dsf.ret
                                {", " + additional_columns_sql if additional_columns_sql else ""}
                            FROM crsp.dsf AS dsf
                            INNER JOIN crsp.msenames AS msn
                              ON dsf.permno = msn.permno
                              AND dsf.date BETWEEN msn.namedt
                                  AND msn.nameendt
                              AND msn.shrcd IN (10, 11)
                            WHERE dsf.permno IN {permno_in}
                            AND dsf.date BETWEEN '{start_date}'
                                AND '{end_date}'
                        """
                        # WRDS stores v1 permno as double precision;
                        # cast to Int64 so joins match dtypes.
                        crsp_daily_sub = _cast_columns(
                            _cast_date_columns(
                                _read_sql(dsf_query, wrds_connection),
                                ["date"],
                            ),
                            {"permno": pl.Int64},
                        ).drop_nulls(subset=["permno", "date", "ret"])

                        if crsp_daily_sub.is_empty():
                            continue

                        # Per-batch msedelist
                        msedelist_query = (
                            "SELECT permno, dlstdt, dlret "
                            "FROM crsp.msedelist "
                            f"WHERE permno IN {permno_in}"
                        )
                        msedelist_sub = _cast_columns(
                            _cast_date_columns(
                                _read_sql(msedelist_query, wrds_connection),
                                ["dlstdt"],
                            ),
                            {"permno": pl.Int64},
                        ).drop_nulls()

                        # Merge dsf with msedelist on (permno, date=dlstdt)
                        if not msedelist_sub.is_empty():
                            crsp_daily_sub = crsp_daily_sub.join(
                                msedelist_sub.rename({"dlstdt": "date"}),
                                on=["permno", "date"],
                                how="left",
                                maintain_order="left",
                            )

                            # Bind delisting-only rows that don't
                            # match any dsf date
                            matched_dates = (
                                crsp_daily_sub.filter(
                                    pl.col("dlret").is_not_null()
                                )
                                .select("permno", "date")
                                .unique(maintain_order=True)
                            )
                            unmatched = msedelist_sub.join(
                                matched_dates.rename({"date": "dlstdt"}),
                                on=["permno", "dlstdt"],
                                how="anti",
                            )
                            if not unmatched.is_empty():
                                unmatched = unmatched.rename(
                                    {"dlstdt": "date"}
                                )
                                crsp_daily_sub = pl.concat(
                                    [crsp_daily_sub, unmatched],
                                    how="diagonal_relaxed",
                                )

                            # ret = dlret if not null
                            crsp_daily_sub = crsp_daily_sub.with_columns(
                                ret=pl.coalesce("dlret", "ret")
                            ).drop("dlret")

                            # Filter date <= permno's last delisting
                            # date (or end_date if no delisting)
                            permno_dlstdt = msedelist_sub.group_by(
                                "permno", maintain_order=True
                            ).agg(
                                pl.col("dlstdt").max().alias("_permno_dlstdt")
                            )
                            crsp_daily_sub = (
                                crsp_daily_sub.join(
                                    permno_dlstdt,
                                    on="permno",
                                    how="left",
                                    maintain_order="left",
                                )
                                .with_columns(
                                    pl.col("_permno_dlstdt").fill_null(
                                        end_date
                                    )
                                )
                                .filter(
                                    pl.col("date") <= pl.col("_permno_dlstdt")
                                )
                                .drop("_permno_dlstdt")
                            )

                        # Merge risk_free, compute ret_excess
                        crsp_daily_sub = (
                            crsp_daily_sub.join(
                                risk_free_daily,
                                on="date",
                                how="left",
                                maintain_order="left",
                            )
                            .with_columns(
                                ret_excess=pl.col("ret") - pl.col("risk_free")
                            )
                            .drop("risk_free")
                        )

                        crsp_data_parts.append(crsp_daily_sub)
                        _report_batch_progress(j, batches, quiet)

                crsp_data = (
                    pl.concat(crsp_data_parts, how="diagonal_relaxed")
                    if crsp_data_parts
                    else pl.DataFrame()
                )

                # Gao-Ritter volume adjustment for NASDAQ (v1: exchcd==3)
                if adjust_volume and not crsp_data.is_empty():
                    crsp_data = (
                        crsp_data.sort(["permno", "date"], maintain_order=True)
                        .with_columns(
                            pl.when(pl.col("vol") == -99)
                            .then(None)
                            .otherwise(pl.col("vol"))
                            .alias("vol"),
                            _zero_to_null("prc"),
                        )
                        .with_columns(
                            prc_adj=_inf_to_null(
                                pl.col("prc").abs() / pl.col("cfacpr")
                            )
                        )
                        .with_columns(
                            _gao_ritter_vol_adj(pl.col("exchcd") == 3)
                        )
                    )

                processed_data = crsp_data
            elif version == "v2":
                permnos = _cast_columns(
                    _read_sql(
                        "SELECT DISTINCT permno FROM crsp.stksecurityinfohist",
                        wrds_connection,
                    ),
                    {"permno": pl.Int64},
                )
                permnos = [str(p) for p in permnos["permno"].to_list()]
                batches = math.ceil(len(permnos) / batch_size)

                risk_free_daily = _download_data_risk_free(
                    start_date=start_date,
                    end_date=end_date,
                    frequency="daily",
                )

                for j in range(1, batches + 1):
                    permno_batch = permnos[
                        ((j - 1) * batch_size) : (
                            min(j * batch_size, len(permnos))
                        )
                    ]
                    permno_batch_formatted = ", ".join(
                        f"'{permno}'" for permno in permno_batch
                    )
                    permno_string = f"({permno_batch_formatted})"

                    crsp_daily_sub_query = f"""
                        SELECT dsf.permno, dlycaldt AS date, dlyret AS ret
                            {", " + additional_columns_sql if additional_columns_sql else ""}
                        FROM crsp.dsf_v2 AS dsf
                        INNER JOIN crsp.stksecurityinfohist AS ssih
                        ON dsf.permno = ssih.permno AND
                            ssih.secinfostartdt <= dsf.dlycaldt AND
                            dsf.dlycaldt <= ssih.secinfoenddt
                        WHERE dsf.permno IN {permno_string}
                        AND dlycaldt BETWEEN '{start_date}' AND '{end_date}'
                        AND ssih.sharetype = 'NS'
                        AND ssih.securitytype = 'EQTY'
                        AND ssih.securitysubtype = 'COM'
                        AND ssih.usincflg = 'Y'
                        AND ssih.issuertype in ('ACOR', 'CORP')
                        AND ssih.primaryexch in ('N', 'A', 'Q')
                        AND ssih.conditionaltype in ('RW', 'NW')
                        AND ssih.tradingstatusflg = 'A'
                        """

                    crsp_daily_sub = _cast_columns(
                        _cast_date_columns(
                            _read_sql(crsp_daily_sub_query, wrds_connection),
                            ["date"],
                        ),
                        {"permno": pl.Int64},
                    ).drop_nulls(subset=["permno", "date", "ret"])

                    if not crsp_daily_sub.is_empty():
                        crsp_daily_sub = (
                            crsp_daily_sub.join(
                                risk_free_daily,
                                on="date",
                                how="left",
                                maintain_order="left",
                            )
                            .with_columns(
                                ret_excess=pl.col("ret") - pl.col("risk_free")
                            )
                            .drop("risk_free")
                        )

                    crsp_data_parts.append(crsp_daily_sub)
                    _report_batch_progress(j, batches, quiet)

                crsp_data = (
                    pl.concat(crsp_data_parts, how="diagonal_relaxed")
                    if crsp_data_parts
                    else pl.DataFrame()
                )

                if adjust_volume and not crsp_data.is_empty():
                    crsp_data = (
                        crsp_data.sort(["permno", "date"], maintain_order=True)
                        .with_columns(
                            cfacpr=pl.col("dlyfacprc")
                            .cum_prod()
                            .over("permno"),
                            vol=pl.when(pl.col("dlyvol") == -99)
                            .then(None)
                            .otherwise(pl.col("dlyvol")),
                            prc=pl.when(pl.col("dlyprc") == 0)
                            .then(None)
                            .otherwise(pl.col("dlyprc")),
                        )
                        .with_columns(
                            prc_adj=_inf_to_null(
                                pl.col("prc").abs() / pl.col("cfacpr")
                            )
                        )
                        .with_columns(
                            _gao_ritter_vol_adj(pl.col("primaryexch") == "Q")
                        )
                        .drop("dlyvol", "dlyprc", "dlyfacprc")
                    )

                processed_data = crsp_data
        else:
            raise ValueError(
                "Invalid dataset. Use 'crsp_monthly' or 'crsp_daily'."
            )
    finally:
        disconnect_connection(wrds_connection)

    if add_ccm_links:
        ccm_links = _download_data_wrds_ccm_links(quiet=quiet)
        merged = processed_data.select("permno", "date").join(
            ccm_links, on="permno", how="inner", maintain_order="left"
        )
        valid_links = merged.filter(
            pl.col("gvkey").is_not_null()
            & (pl.col("linkdt") <= pl.col("date"))
            & (pl.col("date") <= pl.col("linkenddt"))
        ).select("permno", "gvkey", "date")
        processed_data = processed_data.join(
            valid_links,
            on=["permno", "date"],
            how="left",
            maintain_order="left",
        )

    return processed_data


@_wrds_download_step("CRSP-Compustat links")
def _download_data_wrds_ccm_links(
    linktype: list[str] = ["LU", "LC"],
    linkprim: list[str] = ["P", "C"],
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Download data from the WRDS CRSP/Compustat Merged (CCM) links database.

    Parameters
    ----------
    linktype : list of str, optional
        A list of strings indicating the types of links to download.
        Defaults to ["LU", "LC"].
    linkprim : list of str, optional
        A list of strings indicating the primacy of the links.
        Defaults to ["P", "C"].
    quiet : bool, optional
        If True, the messages about the WRDS connection and the download
        progress are suppressed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        A data frame containing columns permno, gvkey, linkdt, and
        linkenddt (missing end dates replaced with today's date).
    """
    conn = get_wrds_connection(quiet=quiet)

    query = f"""
        SELECT lpermno AS permno, gvkey, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ({",".join(f"'{lt}'" for lt in linktype)})
        AND linkprim IN ({",".join(f"'{lp}'" for lp in linkprim)})
    """

    ccm_links = _cast_columns(
        _cast_date_columns(_read_sql(query, conn), ["linkdt", "linkenddt"]),
        {"permno": pl.Int64},
    )

    ccm_links = ccm_links.with_columns(
        pl.col("linkenddt").fill_null(dt.date.today())
    )

    disconnect_connection(conn)

    return ccm_links


def _compustat_step_label(arguments: dict):
    """Name of the Compustat download step, e.g. 'Compustat annual data'."""
    requested = arguments.get("dataset") or arguments.get("type")
    if not requested:
        return None
    name = re.sub(r"^wrds_", "", requested).replace("compustat_", "Compustat ")
    return f"{name} data"


@_wrds_download_step(_compustat_step_label)
def _download_data_wrds_compustat(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    additional_columns: list = None,
    only_usd: bool = False,
    only_us: bool = None,
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Download data from WRDS Compustat.

    Downloads financial data from the WRDS Compustat database for a
    given dataset, start date, and end date. The function filters the
    data according to industry format, data format, and consolidation
    level, and returns the most current data for each reporting
    period. Additionally, the annual data also includes the calculated
    book equity ('be'), operating profitability ('op'), and investment
    ('inv') for each company following Fama and French (1993, 2015),
    as well as income before extraordinary items ('ib').

    Parameters
    ----------
    dataset : str
        The dataset to download ('compustat_annual' or
        'compustat_quarterly').
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    type : str, optional
        Deprecated. Use 'dataset' instead. If supplied, a leading
        'wrds_' prefix is stripped (e.g., 'wrds_compustat_annual'
        becomes 'compustat_annual') and a DeprecationWarning is
        emitted.
    additional_columns : list of str, optional
        Additional columns from the Compustat table as a list of
        strings.
    only_usd : bool, optional
        A boolean indicating whether only USD-denominated shares
        should be returned (i.e., excluding Canadian firms). Defaults
        to False.
    only_us : bool, optional
        Deprecated. Use 'only_usd' instead. If supplied, the value is
        forwarded to 'only_usd' and a DeprecationWarning is emitted.
    quiet : bool, optional
        If True, the messages about the WRDS connection and the download
        progress are suppressed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        A data frame with financial data for the specified period,
        including variables for book equity ('be'), operating
        profitability ('op'), investment ('inv'), and others.

    References
    ----------
    Fama, E. F., and French, K. R. (1993). Common risk factors in the
    returns on stocks and bonds. Journal of Financial Economics,
    33(1), 3-56. https://doi.org/10.1016/0304-405X(93)90023-5

    Fama, E. F., and French, K. R. (2015). A five-factor asset pricing
    model. Journal of Financial Economics, 116(1), 1-22.
    https://doi.org/10.1016/j.jfineco.2014.10.010

    Examples
    --------
    ```python
    from tidyfinance import download_data_wrds_compustat
    download_data_wrds_compustat(
        'compustat_annual', '2020-01-01', '2020-12-31'
    )
    download_data_wrds_compustat(
        'compustat_quarterly', '2020-01-01', '2020-12-31'
    )
    download_data_wrds_compustat(
        'compustat_annual', additional_columns=['aodo', 'aldo']
    )
    ```
    """
    if type is not None:
        warnings.warn(
            "The 'type' argument is deprecated. Use 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", type)

    if only_us is not None:
        warnings.warn(
            "The 'only_us' argument is deprecated. Use 'only_usd' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        only_usd = only_us

    if dataset is not None and dataset.startswith("wrds_"):
        replacement = re.sub(r"^wrds_", "", dataset)
        _deprecate_legacy_dataset(dataset, replacement)
        dataset = replacement

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    start_date, end_date = _validate_dates(start_date, end_date)

    if dataset not in ["compustat_annual", "compustat_quarterly"]:
        raise ValueError(
            (
                "Invalid dataset specified. "
                "Use 'compustat_annual' or 'compustat_quarterly'."
            )
        )

    wrds_connection = get_wrds_connection(quiet=quiet)
    _base_annual_cols = {
        "gvkey",
        "datadate",
        "seq",
        "ceq",
        "at",
        "lt",
        "txditc",
        "txdb",
        "itcb",
        "pstkrv",
        "pstkl",
        "pstk",
        "capx",
        "oancf",
        "sale",
        "cogs",
        "xint",
        "xsga",
        "ib",
        "curcd",
    }
    _base_quarterly_cols = {
        "gvkey",
        "datadate",
        "rdq",
        "fqtr",
        "fyearq",
        "atq",
        "ceqq",
        "curcdq",
    }
    extra_annual = [
        c
        for c in (additional_columns or [])
        if c != "curcd" and c not in _base_annual_cols
    ]
    extra_quarterly = [
        c
        for c in (additional_columns or [])
        if c != "curcdq" and c not in _base_quarterly_cols
    ]
    additional_columns_annual = ", ".join(extra_annual) if extra_annual else ""
    additional_columns_quarterly = (
        ", ".join(extra_quarterly) if extra_quarterly else ""
    )

    if "compustat_annual" in dataset:
        query = f"""
            SELECT gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb,
                pstkrv, pstkl, pstk, capx, oancf, sale, cogs, xint, xsga,
                ib, curcd
                {", " + additional_columns_annual if additional_columns_annual else ""}
            FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """

        compustat = _cast_date_columns(
            _read_sql(query, wrds_connection), ["datadate"]
        )
        disconnect_connection(wrds_connection)

        # Compute Book Equity (be)
        compustat = compustat.with_columns(
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
            )
        ).with_columns(
            be=pl.when(pl.col("be") <= 0).then(None).otherwise(pl.col("be"))
        )
        # Compute Operating Profitability (op)
        compustat = compustat.with_columns(
            op=(
                pl.col("sale")
                - (
                    pl.col("cogs").fill_null(0)
                    + pl.col("xsga").fill_null(0)
                    + pl.col("xint").fill_null(0)
                )
            )
            / pl.col("be")
        )
        # Keep the latest report per company per year
        compustat = (
            compustat.with_columns(year=pl.col("datadate").dt.year())
            .sort("datadate", maintain_order=True)
            .unique(subset=["gvkey", "year"], keep="last", maintain_order=True)
        )
        # Compute Investment (inv)
        compustat_lag = compustat.select(
            "gvkey",
            (pl.col("year") + 1).alias("year"),
            pl.col("at").alias("at_lag"),
        )

        compustat = (
            compustat.join(
                compustat_lag,
                how="left",
                on=["gvkey", "year"],
                maintain_order="left",
            )
            .with_columns(inv=pl.col("at") / pl.col("at_lag") - 1)
            .with_columns(
                inv=pl.when(pl.col("at_lag") <= 0)
                .then(None)
                .otherwise(pl.col("inv"))
            )
        )

        if only_usd:
            compustat = compustat.filter(pl.col("curcd") == "USD")

        processed_data = compustat.with_columns(
            date=pl.col("datadate").dt.truncate("1mo")
        ).drop("year", "at_lag", "curcd")

    elif "compustat_quarterly" in dataset:
        query = f"""
            SELECT gvkey, datadate, rdq, fqtr, fyearq, atq, ceqq, curcdq
                {", " + additional_columns_quarterly if additional_columns_quarterly else ""}
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """

        compustat = _cast_date_columns(
            _read_sql(query, wrds_connection), ["datadate", "rdq"]
        )
        disconnect_connection(wrds_connection)

        # Ensure necessary columns are not missing
        compustat = (
            compustat.drop_nulls(subset=["gvkey", "datadate", "fyearq", "fqtr"])
            .with_columns(date=pl.col("datadate").dt.truncate("1mo"))
            .sort("datadate", descending=True, maintain_order=True)
            .unique(
                subset=["gvkey", "fyearq", "fqtr"],
                keep="first",
                maintain_order=True,
            )
            .sort(
                ["gvkey", "date", "rdq"],
                nulls_last=True,
                maintain_order=True,
            )
            .unique(subset=["gvkey", "date"], keep="first", maintain_order=True)
            .filter(pl.col("rdq").is_null() | (pl.col("date") < pl.col("rdq")))
        )

        if only_usd:
            compustat = compustat.filter(pl.col("curcdq") == "USD")

        base_output_cols = ["gvkey", "date", "datadate", "atq", "ceqq"]
        requested_quarterly = [
            c for c in (additional_columns or []) if c not in base_output_cols
        ]
        processed_data = compustat.select(base_output_cols + requested_quarterly)

    return processed_data


@_wrds_download_step("FISD bond characteristics")
def _download_data_wrds_fisd(
    additional_columns: list = None, quiet: bool = False
) -> pl.DataFrame:
    """
    Download filtered FISD data from WRDS.

    Establishes a connection to the WRDS database to download a
    filtered subset of the FISD (Fixed Income Securities Database).
    The function filters the 'fisd_mergedissue' and
    'fisd_mergedissuer' tables based on several criteria related to
    the securities, such as security level, bond type, coupon type,
    and others, focusing on specific attributes that denote the nature
    of the securities. It finally returns a data frame with selected
    fields from the 'fisd_mergedissue' table after joining it with
    issuer information from the 'fisd_mergedissuer' table for issuers
    domiciled in the USA.

    Parameters
    ----------
    additional_columns : list of str, optional
        Additional columns from the FISD table as a list of strings.
    quiet : bool, optional
        If True, the messages about the WRDS connection and the download
        progress are suppressed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        A data frame containing a subset of FISD data with fields
        related to the bond's characteristics and issuer information.
        This includes complete CUSIP, maturity date, offering amount,
        offering date, dated date, interest frequency, coupon, last
        interest date, issue ID, issuer ID, and SIC code of the
        issuer.

    Examples
    --------
    ```python
    from tidyfinance import download_data_wrds_fisd
    fisd = download_data_wrds_fisd()
    fisd_extended = download_data_wrds_fisd(
        additional_columns=['asset_backed', 'defeased']
    )
    ```
    """
    wrds_connection = get_wrds_connection(quiet=quiet)

    if additional_columns:
        if not all(
            re.match(r"^[a-z_][a-z0-9_]*$", col) for col in additional_columns
        ):
            raise ValueError("Column names must be valid SQL identifiers.")

    additional_columns_str = _process_additional_columns(additional_columns)

    fisd_query = (
        "SELECT complete_cusip, maturity, offering_amt, offering_date, "
        "dated_date, interest_frequency, coupon, last_interest_date, "
        f"issue_id, issuer_id{additional_columns_str} "
        "FROM fisd.fisd_mergedissue "
        "WHERE security_level = 'SEN' "
        "AND (slob = 'N' OR slob IS NULL) "
        "AND security_pledge IS NULL "
        "AND (asset_backed = 'N' OR asset_backed IS NULL) "
        "AND (defeased = 'N' OR defeased IS NULL) "
        "AND defeased_date IS NULL "
        "AND bond_type IN ('CDEB', 'CMTN', 'CMTZ', 'CZ', 'USBN') "
        "AND (pay_in_kind != 'Y' OR pay_in_kind IS NULL) "
        "AND pay_in_kind_exp_date IS NULL "
        "AND (yankee = 'N' OR yankee IS NULL) "
        "AND (canadian = 'N' OR canadian IS NULL) "
        "AND foreign_currency = 'N' "
        "AND coupon_type IN ('F', 'Z') "
        "AND fix_frequency IS NULL "
        "AND coupon_change_indicator = 'N' "
        "AND interest_frequency IN ('0', '1', '2', '4', '12') "
        "AND rule_144a = 'N' "
        "AND (private_placement = 'N' OR private_placement IS NULL) "
        "AND defaulted = 'N' "
        "AND filing_date IS NULL "
        "AND settlement IS NULL "
        "AND convertible = 'N' "
        "AND exchange IS NULL "
        "AND (putable = 'N' OR putable IS NULL) "
        "AND (unit_deal = 'N' OR unit_deal IS NULL) "
        "AND (exchangeable = 'N' OR exchangeable IS NULL) "
        "AND perpetual = 'N' "
        "AND (preferred_security = 'N' OR preferred_security IS NULL)"
    )

    fisd = _cast_columns(
        _cast_date_columns(
            _read_sql(fisd_query, wrds_connection),
            ["maturity", "offering_date", "dated_date", "last_interest_date"],
        ),
        {
            "complete_cusip": pl.String,
            "interest_frequency": pl.String,
            "issue_id": pl.Int64,
            "issuer_id": pl.Int64,
        },
    )

    fisd_issuer_query = (
        "SELECT issuer_id, sic_code, country_domicile "
        "FROM fisd.fisd_mergedissuer"
    )

    fisd_issuer = _cast_columns(
        _read_sql(fisd_issuer_query, wrds_connection),
        {
            "issuer_id": pl.Int64,
            "sic_code": pl.String,
            "country_domicile": pl.String,
        },
    )

    fisd = (
        fisd.join(
            fisd_issuer, how="inner", on="issuer_id", maintain_order="left"
        )
        .filter(pl.col("country_domicile") == "USA")
        .drop("country_domicile")
    )

    disconnect_connection(wrds_connection)

    return fisd


@_wrds_download_step("enhanced TRACE data")
def _download_data_wrds_trace_enhanced(
    cusips: list,
    start_date: str = None,
    end_date: str = None,
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Download Enhanced TRACE data from WRDS.

    Establishes a connection to the WRDS database to download the
    specified CUSIPs trade messages from the Trade Reporting and
    Compliance Engine (TRACE). The trade data is cleaned as suggested
    by Dick-Nielsen (2009, 2014).

    Parameters
    ----------
    cusips : list of str
        A character vector specifying the 9-digit CUSIPs to download.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, a subset of the
        dataset is returned.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, a subset of the
        dataset is returned.
    quiet : bool, optional
        If True, the messages about the WRDS connection and the download
        progress are suppressed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        A data frame containing the cleaned trade messages from TRACE
        for the selected CUSIPs over the time window specified. Output
        variables include identifying information (i.e., CUSIP, trade
        date/time) and trade-specific information (i.e., price/yield,
        volume, counterparty, and reporting side).

    References
    ----------
    Dick-Nielsen, J. (2009). Liquidity biases in TRACE. Journal of
    Fixed Income, 19(2), 43-55.
    https://doi.org/10.3905/jfi.2009.19.2.043

    Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working
    Paper. https://doi.org/10.2139/ssrn.2337908

    Examples
    --------
    ```python
    from tidyfinance import download_data_wrds_trace_enhanced
    download_data_wrds_trace_enhanced(
        ['00101JAH9'], '2019-01-01', '2021-12-31'
    )
    ```
    """
    if not all(isinstance(cusip, str) and len(cusip) == 9 for cusip in cusips):
        raise ValueError("All CUSIPs must be 9-character strings.")

    wrds_connection = get_wrds_connection(quiet=quiet)

    cusip_string = _format_cusips(cusips)

    query = (
        "SELECT cusip_id, bond_sym_id, trd_exctn_dt, "
        "trd_exctn_tm, days_to_sttl_ct, lckd_in_ind, "
        "wis_fl, sale_cndtn_cd, msg_seq_nb, "
        "trc_st, trd_rpt_dt, trd_rpt_tm, "
        "entrd_vol_qt, rptd_pr, yld_pt, "
        "asof_cd, orig_msg_seq_nb, rpt_side_cd, "
        "cntra_mp_id, stlmnt_dt, spcl_trd_fl "
        "FROM trace.trace_enhanced "
        f"WHERE cusip_id IN {cusip_string} "
    )

    if start_date and end_date:
        query += f"AND trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'"

    trace_enhanced_raw = _read_sql(query, wrds_connection)
    disconnect_connection(wrds_connection)

    trace_enhanced = process_trace_data(trace_enhanced_raw)

    return trace_enhanced


def get_wrds_connection(quiet: bool = False) -> object:
    """Establish a connection to Wharton Research Data Services (WRDS).

    Opens a SQLAlchemy connection to the WRDS PostgreSQL database. Credentials
    are read from the 'WRDS_USER' and 'WRDS_PASSWORD' environment variables,
    which can also be supplied through a '.env' file in the working directory
    ('load_dotenv' is called before the lookup). The connection uses the
    'postgresql+psycopg2' driver, requires SSL, and targets
    'wrds-pgdata.wharton.upenn.edu' on port 9737. Pool pre-ping is enabled so
    stale connections are detected and recycled automatically.

    Parameters
    ----------
    quiet : bool, default False
        If True, the messages about the WRDS connection and the download
        progress are suppressed.

    Returns
    -------
    sqlalchemy.engine.Connection
        An open SQLAlchemy connection to the WRDS database. Pass it to any
        'download_data_wrds_*' helper or use it directly with
        'polars.read_database'. Close it via 'disconnect_connection' when
        done.

    Raises
    ------
    ValueError
        If 'WRDS_USER' or 'WRDS_PASSWORD' is not set in the environment or
        '.env' file. The 'load_wrds_credentials' helper performs this check
        before the connection is opened.
    ConnectionError
        If the connection fails, e.g. because the database host is
        unreachable or the credentials are rejected. The message lists the
        likely causes. The original error is kept as the cause
        ('__cause__'), with the username and password replaced by
        placeholders.

    Notes
    -----
    A WRDS username and password alone are not enough to connect. Before
    the first connection:

    1. Get a WRDS account through an institution that subscribes to the
       data you need (e.g., CRSP and Compustat).
    2. Enroll in two-factor authentication (Duo) for your WRDS account.
       WRDS requires it for all connections, including connections from
       Python.
    3. Log in to the WRDS website (https://wrds-www.wharton.upenn.edu)
       once and accept the terms of use, if WRDS asks you to.
    4. Store your username and password with 'set_wrds_credentials', or
       set the 'WRDS_USER' and 'WRDS_PASSWORD' environment variables.
    5. When you connect, approve the Duo push notification on your
       registered device. The connection waits until you approve it.

    If the connection fails, the error message lists these steps. Unless
    'quiet=True', the function says when it starts connecting and when
    the connection succeeds, so you know when to approve the Duo request
    and when the download starts.

    See Also
    --------
    set_wrds_credentials : Interactively prompts for credentials and writes
        them to a '.env' file in the project or home directory.
    disconnect_connection : Closes the connection returned by this function.

    Examples
    --------
    ```python
    import os
    os.environ['WRDS_USER'] = 'your_username'
    os.environ['WRDS_PASSWORD'] = 'your_password'
    from tidyfinance import get_wrds_connection, disconnect_connection
    con = get_wrds_connection()
    # ... query WRDS via con ...
    disconnect_connection(con)
    ```
    """
    _validate_flag(quiet, "quiet")

    wrds_user, wrds_password = load_wrds_credentials()
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=wrds_user,
        password=wrds_password,
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        database="wrds",
    )
    engine = create_engine(
        url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )

    if not quiet:
        # The username is deliberately not shown, as messages can end up
        # in public logs (e.g., CI or rendered documents).
        _inform("Connecting to WRDS. Approve the Duo request if prompted.")
    try:
        connection = engine.connect()
    except Exception as e:
        if not quiet:
            _inform("Connection to WRDS failed.")
        # The driver error can contain the username (e.g. "password
        # authentication failed for user ..."), and error messages can end
        # up in public logs, so the credentials are replaced.
        cause = RuntimeError(
            _redact_wrds_credentials(
                f"{type(e).__name__}: {e}", wrds_user, wrds_password
            )
        )
        raise ConnectionError(
            "Could not connect to WRDS.\n"
            "- Check that 'WRDS_USER' and 'WRDS_PASSWORD' are correct, "
            "e.g., with 'set_wrds_credentials()'.\n"
            "- WRDS requires two-factor authentication (Duo): enroll your "
            "account and approve the push notification when you connect.\n"
            "- Log in to https://wrds-www.wharton.upenn.edu once and accept "
            "the terms of use if WRDS asks you to.\n"
            "- See help(get_wrds_connection) for details."
        ) from cause

    if not quiet:
        _inform("Connected to WRDS.")
        # Announce the download of the calling WRDS download function.
        step = _WRDS_DOWNLOAD_STEP.get()
        if step is not None:
            step.announce()
    return connection


def _redact_wrds_credentials(message: str, user: str, password: str) -> str:
    """Replace WRDS credentials in an error message by placeholders.

    Parameters
    ----------
    message : str
        Error message, e.g. of a failed connection.
    user : str
        WRDS username to replace by '<WRDS_USER>'.
    password : str
        WRDS password to replace by '<WRDS_PASSWORD>'.

    Returns
    -------
    str
        The message without the username and password.
    """
    # The password is replaced first, in case it contains the username.
    for placeholder, value in (
        ("<WRDS_PASSWORD>", password),
        ("<WRDS_USER>", user),
    ):
        if value:
            message = message.replace(value, placeholder)
    return message


def disconnect_connection(connection: object) -> bool:
    """Close an open WRDS database connection safely.

    Attempts to close the supplied connection. Any exception raised by
    the underlying driver is swallowed so the call can be used in
    'finally' blocks without masking the original error.

    Parameters
    ----------
    connection : object
        The connection object to close. Typically the value returned by
        'get_wrds_connection'.

    Returns
    -------
    bool
        'True' on successful disconnection, 'False' if the close call
        raised an exception.

    Examples
    --------
    ```python
    from tidyfinance import get_wrds_connection, disconnect_connection
    con = get_wrds_connection()
    disconnect_connection(con)
    ```
    """
    try:
        connection.close()
        return True
    except Exception:
        return False


def load_wrds_credentials() -> tuple:
    """Load WRDS credentials from environment variables or a '.env' file.

    Reads 'WRDS_USER' and 'WRDS_PASSWORD' from the process environment.
    If neither is set, also looks for a '.env' file in the working
    directory via 'load_dotenv'.

    Returns
    -------
    tuple of (str, str)
        '(wrds_user, wrds_password)' pair suitable for building a
        SQLAlchemy connection URL.

    Raises
    ------
    ValueError
        If either 'WRDS_USER' or 'WRDS_PASSWORD' is missing after the
        '.env' file has been loaded.
    """
    load_dotenv()

    wrds_user: str = os.getenv("WRDS_USER")
    wrds_password: str = os.getenv("WRDS_PASSWORD")

    if not wrds_user or not wrds_password:
        raise ValueError(
            "WRDS credentials not found. Please set 'WRDS_USER' "
            "and 'WRDS_PASSWORD' as environment variables, e.g. via a "
            ".env file."
        )

    return wrds_user, wrds_password


def _process_additional_columns(additional_columns):
    """Validate and format additional column names for SQL queries.

    Parameters
    ----------
    additional_columns : list of str or None
        Column names to append to a SQL SELECT clause. Each name must
        be a valid lowercase SQL identifier (letters, digits, underscores).

    Returns
    -------
    str
        A string like ", col1, col2" ready to splice into a SELECT,
        or an empty string if no additional columns were provided.

    Raises
    ------
    ValueError
        If any column name contains characters other than lowercase
        letters, digits, or underscores.
    """
    if not additional_columns:
        return ""
    if not all(
        re.match(r"^[a-z_][a-z0-9_]*$", col) for col in additional_columns
    ):
        raise ValueError("Column names must be valid SQL identifiers.")
    return ", " + ", ".join(additional_columns)


def process_trace_data(trace_all: pl.DataFrame) -> pl.DataFrame:
    """Clean Enhanced TRACE trade reports.

    Applies the Dick-Nielsen cleaning protocol to the raw Enhanced
    TRACE message stream, removing cancellations, corrections, and
    reversals and producing one observation per executed corporate
    bond trade.

    Parameters
    ----------
    trace_all : pl.DataFrame
        Raw Enhanced TRACE messages with the message-status columns
        used by the cleaning protocol ('trc_st', 'msg_seq_nb',
        'orig_msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm',
        'trd_exctn_dt', 'trd_exctn_tm', 'cusip_id', 'entrd_vol_qt',
        'rptd_pr', 'rpt_side_cd', 'cntra_mp_id', 'asof_cd').

    Returns
    -------
    pl.DataFrame
        Cleaned trade panel containing only executed trades with
        cancellations, corrections, and reversals already removed.

    Notes
    -----
    Trades are cleaned under two regimes split by 2012-02-06, the date
    the Enhanced TRACE message-status format changed. Trades reported
    on or after this cutoff are cleaned with the post-2012 logic
    (status flags 'T', 'R', 'X', 'C', 'Y'); earlier trades use the
    pre-2012 logic (status flags 'T', 'C', 'W' plus 'asof_cd'
    reversals). The cutoff date follows Dick-Nielsen (2014).

    References
    ----------
    Dick-Nielsen, J. (2009). Liquidity biases in TRACE. Journal of
    Fixed Income, 19(2), 43-55.
    https://doi.org/10.3905/jfi.2009.19.2.043

    Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working
    Paper. https://ssrn.com/abstract=2337908
    """
    cutoff = dt.date(2012, 2, 6)

    # Normalize calendar-date columns to pl.Date (raw user-supplied
    # frames may carry datetimes).
    trace_all = _cast_date_columns(
        trace_all, ["trd_exctn_dt", "trd_rpt_dt", "stlmnt_dt"]
    )

    # The pandas reference merged with how="left" and no explicit keys,
    # which joins on all shared columns; the polars joins below list
    # those shared columns explicitly.
    post_keys = [
        "cusip_id",
        "msg_seq_nb",
        "entrd_vol_qt",
        "rptd_pr",
        "rpt_side_cd",
        "cntra_mp_id",
        "trd_exctn_dt",
        "trd_exctn_tm",
    ]

    # Post 2012-02-06
    # Trades (trc_st = T) and correction (trc_st = R)
    trace_post_TR = trace_all.filter(
        pl.col("trc_st").is_in(["T", "R"]) & (pl.col("trd_rpt_dt") >= cutoff)
    )

    # Cancellations (trc_st = X) and correction cancellations (trc_st = C)
    trace_post_XC = (
        trace_all.filter(
            pl.col("trc_st").is_in(["X", "C"])
            & (pl.col("trd_rpt_dt") >= cutoff)
        )
        .select(post_keys)
        .with_columns(drop=pl.lit(True))
    )

    # Cleaning corrected and cancelled trades
    trace_post_TR = (
        trace_post_TR.join(
            trace_post_XC, on=post_keys, how="left", maintain_order="left"
        )
        .filter(pl.col("drop").ne_missing(True))
        .drop("drop")
    )

    # Reversals (trc_st = Y)
    trace_post_Y = (
        trace_all.filter(
            (pl.col("trc_st") == "Y") & (pl.col("trd_rpt_dt") >= cutoff)
        )
        .select(
            "cusip_id",
            pl.col("orig_msg_seq_nb").alias("msg_seq_nb"),
            "entrd_vol_qt",
            "rptd_pr",
            "rpt_side_cd",
            "cntra_mp_id",
            "trd_exctn_dt",
            "trd_exctn_tm",
        )
        .with_columns(drop=pl.lit(True))
    )

    # Clean reversals
    # Match the orig_msg_seq_nb of Y-message to msg_seq_nb of main message
    trace_post = (
        trace_post_TR.join(
            trace_post_Y, on=post_keys, how="left", maintain_order="left"
        )
        .filter(pl.col("drop").ne_missing(True))
        .drop("drop")
    )

    # Enhanced TRACE: Pre 2012-02-06
    # Pre 2012-02-06
    # Trades (trc_st = T)
    trace_pre_T = trace_all.filter(
        (pl.col("trc_st") == "T") & (pl.col("trd_rpt_dt") < cutoff)
    )

    # Cancellations (trc_st = C)
    trace_pre_C = (
        trace_all.filter(
            (pl.col("trc_st") == "C") & (pl.col("trd_rpt_dt") < cutoff)
        )
        .select(
            "cusip_id",
            pl.col("orig_msg_seq_nb").alias("msg_seq_nb"),
            "entrd_vol_qt",
            "rptd_pr",
            "rpt_side_cd",
            "cntra_mp_id",
            "trd_exctn_dt",
            "trd_exctn_tm",
        )
        .with_columns(drop=pl.lit(True))
    )

    # Remove cancellations from trades
    # Match orig_msg_seq_nb of C-message to msg_seq_nb of main message
    trace_pre_T = (
        trace_pre_T.join(
            trace_pre_C, on=post_keys, how="left", maintain_order="left"
        )
        .filter(pl.col("drop").ne_missing(True))
        .drop("drop")
    )

    # Corrections (trc_st = W)
    trace_pre_W = trace_all.filter(
        (pl.col("trc_st") == "W") & (pl.col("trd_rpt_dt") < cutoff)
    )

    # Implement corrections in a loop
    # Correction control
    correction_control = trace_pre_W.height
    correction_control_last = trace_pre_W.height

    # Correction loop
    while correction_control > 0:
        # Create placeholder
        # Only identifying columns of trace_pre_T (for joins)
        placeholder_trace_pre_T = trace_pre_T.select(
            "cusip_id",
            "trd_exctn_dt",
            pl.col("msg_seq_nb").alias("orig_msg_seq_nb"),
        ).with_columns(matched_T=pl.lit(True))

        w_keys = ["cusip_id", "trd_exctn_dt", "orig_msg_seq_nb"]
        trace_pre_W_joined = trace_pre_W.join(
            placeholder_trace_pre_T,
            on=w_keys,
            how="left",
            maintain_order="left",
        )

        # Corrections that correct some msg
        trace_pre_W_correcting = trace_pre_W_joined.filter(
            pl.col("matched_T").eq_missing(True)
        ).drop("matched_T")

        # Corrections that do not correct some msg
        trace_pre_W = trace_pre_W_joined.filter(
            pl.col("matched_T").ne_missing(True)
        ).drop("matched_T")

        # Create placeholder
        # Only identifying columns of trace_pre_W_correcting (for anti-joins)
        placeholder_trace_pre_W_correcting = trace_pre_W_correcting.select(
            "cusip_id",
            "trd_exctn_dt",
            pl.col("orig_msg_seq_nb").alias("msg_seq_nb"),
        ).with_columns(corrected=pl.lit(True))

        # Delete msgs that are corrected
        trace_pre_T = (
            trace_pre_T.join(
                placeholder_trace_pre_W_correcting,
                on=["cusip_id", "trd_exctn_dt", "msg_seq_nb"],
                how="left",
                maintain_order="left",
            )
            .filter(pl.col("corrected").ne_missing(True))
            .drop("corrected")
        )

        # Add correction msgs
        trace_pre_T = pl.concat(
            [trace_pre_T, trace_pre_W_correcting], how="diagonal_relaxed"
        )

        # Escape if no corrections remain or they cannot be matched
        correction_control = trace_pre_W.height

        if correction_control == correction_control_last:
            break
        else:
            correction_control_last = trace_pre_W.height
            continue

    sort_cols = [
        "cusip_id",
        "trd_exctn_dt",
        "trd_exctn_tm",
        "trd_rpt_dt",
        "trd_rpt_tm",
    ]

    # Reversals (asof_cd = R)
    # Record reversals
    trace_pre_R = trace_pre_T.filter(pl.col("asof_cd") == "R").sort(
        sort_cols, maintain_order=True
    )

    # Prepare final data
    trace_pre = trace_pre_T.filter(
        pl.col("asof_cd").is_null()
        | ~pl.col("asof_cd").is_in(["R", "X", "D"])
    ).sort(sort_cols, maintain_order=True)

    # Add grouped row numbers
    group_cols = [
        "cusip_id",
        "trd_exctn_dt",
        "entrd_vol_qt",
        "rptd_pr",
        "rpt_side_cd",
        "cntra_mp_id",
    ]
    trace_pre_R = trace_pre_R.with_columns(
        seq=pl.int_range(pl.len()).over(group_cols)
    )
    trace_pre = trace_pre.with_columns(
        seq=pl.int_range(pl.len()).over(group_cols)
    )

    # Select columns for reversal cleaning
    trace_pre_R = trace_pre_R.select(group_cols + ["seq"]).with_columns(
        reversal=pl.lit(True)
    )

    # Remove reversals and the reversed trade
    trace_pre = (
        trace_pre.join(
            trace_pre_R,
            on=group_cols + ["seq"],
            how="left",
            maintain_order="left",
        )
        .filter(pl.col("reversal").ne_missing(True))
        .drop("reversal", "seq")
    )

    # Combine pre and post trades
    trace_clean = pl.concat([trace_pre, trace_post], how="diagonal_relaxed")

    # Keep agency sells and unmatched agency buys
    trace_agency_sells = trace_clean.filter(
        (pl.col("cntra_mp_id") == "D") & (pl.col("rpt_side_cd") == "S")
    )

    # Placeholder for trace_agency_sells with relevant columns
    agency_keys = ["cusip_id", "trd_exctn_dt", "entrd_vol_qt", "rptd_pr"]
    placeholder_trace_agency_sells = trace_agency_sells.select(
        agency_keys
    ).with_columns(matched=pl.lit(True))

    # Agency buys that are unmatched
    trace_agency_buys_filtered = (
        trace_clean.filter(
            (pl.col("cntra_mp_id") == "D") & (pl.col("rpt_side_cd") == "B")
        )
        .join(
            placeholder_trace_agency_sells,
            on=agency_keys,
            how="left",
            maintain_order="left",
        )
        .filter(pl.col("matched").ne_missing(True))
        .drop("matched")
    )

    # Non-agency
    trace_nonagency = trace_clean.filter(pl.col("cntra_mp_id") == "C")

    # Agency cleaned
    trace_clean = pl.concat(
        [trace_nonagency, trace_agency_sells, trace_agency_buys_filtered],
        how="diagonal_relaxed",
    )

    # Additional Filters
    trace_add_filters = (
        trace_clean.with_columns(
            days_to_sttl_ct2=(
                pl.col("stlmnt_dt") - pl.col("trd_exctn_dt")
            ).dt.total_days(),
            days_to_sttl_ct=pl.col("days_to_sttl_ct")
            .cast(pl.String)
            .cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("days_to_sttl_ct").is_null()
            | (pl.col("days_to_sttl_ct") <= 7)
        )
        .filter(
            pl.col("days_to_sttl_ct2").is_null()
            | (pl.col("days_to_sttl_ct2") <= 7)
        )
        .filter(pl.col("wis_fl") == "N")
        .filter(pl.col("spcl_trd_fl").is_null() | (pl.col("spcl_trd_fl") == ""))
        .filter(pl.col("asof_cd").is_null() | (pl.col("asof_cd") == ""))
    )

    # Only keep necessary columns
    trace_final = trace_add_filters.sort(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm"], maintain_order=True
    ).select(
        "cusip_id",
        "trd_exctn_dt",
        "trd_exctn_tm",
        "rptd_pr",
        "entrd_vol_qt",
        "yld_pt",
        "rpt_side_cd",
        "cntra_mp_id",
    )

    return trace_final


def set_wrds_credentials() -> None:
    """Set WRDS credentials in a '.env' file.

    Prompts interactively for the WRDS username and password and writes
    them to a '.env' file as 'WRDS_USER' and 'WRDS_PASSWORD'. The
    location is chosen at the prompt: 'project' writes to the current
    working directory, 'home' writes to the user's home directory. If
    a '.env' file already contains WRDS credentials, the user is asked
    before overwriting. After saving, the user is offered to append
    '.env' to a sibling '.gitignore' (recommended).

    The resulting '.env' file is the credentials source consumed by
    'get_wrds_connection' via 'load_wrds_credentials'.

    Storing credentials is not enough to connect to WRDS: you also need
    to enroll in two-factor authentication (Duo) and accept the WRDS
    terms of use. See the Notes section of 'get_wrds_connection'.

    Returns
    -------
    None
        Called for its side effects: writing '.env' and, optionally,
        updating '.gitignore'.

    See Also
    --------
    get_wrds_connection : Opens a WRDS connection using the credentials
        stored by this function.

    Examples
    --------
    ```python
    from tidyfinance import set_wrds_credentials
    set_wrds_credentials()
    ```
    """
    wrds_user = input("Enter your WRDS username: ")
    wrds_password = input("Enter your WRDS password: ")
    location_choice = (
        input(
            "Where do you want to store the .env "
            "file? Enter 'project' for project directory or "
            "'home' for home directory: "
        )
        .strip()
        .lower()
    )

    if location_choice == "project":
        env_path = os.path.join(os.getcwd(), ".env")
        gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    elif location_choice == "home":
        env_path = os.path.join(os.path.expanduser("~"), ".env")
        gitignore_path = os.path.join(os.path.expanduser("~"), ".gitignore")
    else:
        print(
            "Invalid choice. Please start again and enter 'project' or 'home'."
        )
        return

    existing = dotenv_values(env_path) if os.path.exists(env_path) else {}

    if existing.get("WRDS_USER") and existing.get("WRDS_PASSWORD"):
        overwrite_choice = (
            input(
                "Credentials already exist. Do you want to "
                "overwrite them? Enter 'yes' or 'no': "
            )
            .strip()
            .lower()
        )
        if overwrite_choice != "yes":
            print("Aborted. Credentials already exist.")
            return

    if os.path.exists(gitignore_path):
        add_gitignore = (
            input(
                "Do you want to add .env to .gitignore? "
                "It is highly recommended! "
                "Enter 'yes' or 'no': "
            )
            .strip()
            .lower()
        )
        if add_gitignore == "yes":
            with open(gitignore_path, "r") as file:
                gitignore_lines = file.readlines()
            if ".env\n" not in gitignore_lines:
                with open(gitignore_path, "a") as file:
                    file.write(".env\n")
                print(".env added to .gitignore.")
        elif add_gitignore == "no":
            print(".env NOT added to .gitignore.")
        else:
            print("Invalid choice. Please start again and enter 'yes' or 'no'.")
            return

    set_key(env_path, "WRDS_USER", wrds_user)
    set_key(env_path, "WRDS_PASSWORD", wrds_password)

    print(
        "WRDS credentials have been set and saved in .env in your "
        f"{location_choice} directory."
    )


def _assign_exchange(primaryexch):
    """
    Map a CRSP primary-exchange code to a readable label.

    Parameters
    ----------
    primaryexch : str
        Single-letter primary-exchange code from CRSP CIZ
        ('stksecurityinfohist.primaryexch').

    Returns
    -------
    str
        'NYSE', 'AMEX', 'NASDAQ', or 'Other'.
    """
    if primaryexch == "N":
        return "NYSE"
    elif primaryexch == "A":
        return "AMEX"
    elif primaryexch == "Q":
        return "NASDAQ"
    else:
        return "Other"


def _assign_industry(siccd):
    """
    Map a Standard Industrial Classification code to a coarse industry label.

    Parameters
    ----------
    siccd : int
        Four-digit SIC code (CRSP 'siccd' or Compustat 'sich').

    Returns
    -------
    str
        One of 'Agriculture', 'Mining', 'Construction',
        'Manufacturing', 'Transportation', 'Utilities', 'Wholesale',
        'Retail', 'Finance', 'Services', 'Public', or 'Missing'
        (when the SIC code is outside the documented ranges).
    """
    if 1 <= siccd <= 999:
        return "Agriculture"
    elif 1000 <= siccd <= 1499:
        return "Mining"
    elif 1500 <= siccd <= 1799:
        return "Construction"
    elif 2000 <= siccd <= 3999:
        return "Manufacturing"
    elif 4000 <= siccd <= 4899:
        return "Transportation"
    elif 4900 <= siccd <= 4999:
        return "Utilities"
    elif 5000 <= siccd <= 5199:
        return "Wholesale"
    elif 5200 <= siccd <= 5999:
        return "Retail"
    elif 6000 <= siccd <= 6799:
        return "Finance"
    elif 7000 <= siccd <= 8999:
        return "Services"
    elif 9000 <= siccd <= 9999:
        return "Public"
    else:
        return "Missing"


def _format_cusips(cusips):
    """
    Format a list of CUSIPs as a parenthesized SQL 'IN' clause.

    Parameters
    ----------
    cusips : list of str
        CUSIP identifiers.

    Returns
    -------
    str
        SQL-ready string of the form "('cusip1', 'cusip2', ...)" or
        "()" when the input is empty.
    """
    if not cusips:
        return "()"

    cusip_batch_formatted = ", ".join(f"'{cusip}'" for cusip in cusips)
    return f"({cusip_batch_formatted})"
