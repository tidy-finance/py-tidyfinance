"""WRDS connection and downloads for tidyfinance."""

import os
import re
import warnings

import numpy as np
import pandas as pd
from dotenv import dotenv_values, load_dotenv, set_key
from sqlalchemy import URL, create_engine, text

from ._internal import _validate_dates
from .download_tidy_finance import _download_data_risk_free
from .supported_datasets import (_check_supported_dataset_wrds,
                                 _check_supported_dataset_wrds_crsp,
                                 _is_legacy_type_wrds)


def _download_data_wrds(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    **kwargs,
) -> pd.DataFrame:
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
        depending on the 'dataset'.

    Returns
    -------
    pd.DataFrame
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
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'crsp_monthly' instead of 'wrds_crsp_monthly'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

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
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
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
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'crsp_monthly' instead of 'wrds_crsp_monthly'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

    if dataset is None:
        raise ValueError("Argument 'dataset' is required.")

    _check_supported_dataset_wrds_crsp(dataset)

    start_date, end_date = _validate_dates(start_date, end_date)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be an integer larger than 0.")

    if version not in ["v1", "v2"]:
        raise ValueError("version must be 'v1' or 'v2'.")

    if version == "v1" and pd.Timestamp(end_date) > pd.Timestamp("2024-12-31"):
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

    wrds_connection = get_wrds_connection()
    additional_columns_sql = (
        ", ".join(additional_columns_list) if additional_columns_list else ""
    )

    crsp_data = pd.DataFrame()
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
                msf_query = text(f"""
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
                """)
                msf_data = pd.read_sql_query(
                    msf_query,
                    con=wrds_connection,
                    parse_dates={"date"},
                )

                # Query 2: msedelist (delisting events)
                msedelist_query = text(
                    "SELECT permno, dlstdt, dlret, dlstcd FROM crsp.msedelist"
                )
                msedelist = pd.read_sql_query(
                    msedelist_query,
                    con=wrds_connection,
                    parse_dates={"dlstdt"},
                )

                # Query 3: first_crsp_date per permno
                first_date_query = text(
                    "SELECT permno, MIN(namedt) AS first_crsp_date "
                    "FROM crsp.msenames GROUP BY permno"
                )
                first_crsp_date = pd.read_sql_query(
                    first_date_query,
                    con=wrds_connection,
                    parse_dates={"first_crsp_date"},
                )

                disconnect_connection(wrds_connection)

                # calculation_date + month-floored date
                crsp_monthly = msf_data.assign(
                    calculation_date=lambda x: pd.to_datetime(x["date"]),
                    date=lambda x: (
                        pd.to_datetime(x["date"])
                        .dt.to_period("M")
                        .dt.start_time
                    ),
                    shrout=lambda x: x["shrout"] * 1000,
                )

                # Join delisting on (permno, month-floored dlstdt)
                if len(msedelist) > 0:
                    msedelist = msedelist.assign(
                        date=lambda x: (
                            pd.to_datetime(x["dlstdt"])
                            .dt.to_period("M")
                            .dt.start_time
                        )
                    )[["permno", "date", "dlret", "dlstcd"]]
                    crsp_monthly = crsp_monthly.merge(
                        msedelist, on=["permno", "date"], how="left"
                    )
                else:
                    crsp_monthly["dlret"] = np.nan
                    crsp_monthly["dlstcd"] = np.nan

                # listing_age (months elapsed, clipped at 0)
                crsp_monthly = (
                    crsp_monthly.merge(first_crsp_date, on="permno", how="left")
                    .assign(
                        listing_age=lambda df: (
                            (df["date"].dt.year - df["first_crsp_date"].dt.year)
                            * 12
                            + (
                                df["date"].dt.month
                                - df["first_crsp_date"].dt.month
                            )
                            - (
                                df["date"].dt.day < df["first_crsp_date"].dt.day
                            ).astype(int)
                        ).clip(lower=0)
                    )
                    .drop(columns="first_crsp_date")
                )

                # mktcap (millions); zero -> NaN
                crsp_monthly["mktcap"] = (
                    crsp_monthly["shrout"] * crsp_monthly["altprc"]
                ).abs() / 1e6
                crsp_monthly.loc[crsp_monthly["mktcap"] == 0, "mktcap"] = np.nan

                # mktcap_lag via self-join shifted by 1 month
                mktcap_lag_df = crsp_monthly.assign(
                    date=lambda x: x["date"] + pd.DateOffset(months=1)
                )[["permno", "date", "mktcap"]].rename(
                    columns={"mktcap": "mktcap_lag"}
                )
                crsp_monthly = crsp_monthly.merge(
                    mktcap_lag_df, on=["permno", "date"], how="left"
                )

                # exchange via exchcd (numeric, v1 codes)
                exchange_map = {
                    1: "NYSE",
                    31: "NYSE",
                    2: "AMEX",
                    32: "AMEX",
                    3: "NASDAQ",
                    33: "NASDAQ",
                }
                crsp_monthly["exchange"] = (
                    crsp_monthly["exchcd"].map(exchange_map).fillna("Other")
                )

                # industry from SIC code
                crsp_monthly["industry"] = (
                    crsp_monthly["siccd"]
                    .fillna(-1)
                    .astype(int)
                    .apply(_assign_industry)
                )

                # ret_adj from delisting code (Shumway-style)
                def _compute_ret_adj_v1(row):
                    if pd.isna(row["dlstcd"]):
                        return row["ret"]
                    if not pd.isna(row["dlret"]):
                        return row["dlret"]
                    code = row["dlstcd"]
                    if code in (500, 520, 580, 584) or (551 <= code <= 574):
                        return -0.30
                    if code == 100:
                        return row["ret"]
                    return -1.0

                crsp_monthly["ret_adj"] = crsp_monthly.apply(
                    _compute_ret_adj_v1, axis=1
                )
                crsp_monthly = crsp_monthly.drop(columns=["dlret", "dlstcd"])

                # prc_adj = |altprc nullified-at-zero| / cfacpr
                prc_zeroed = crsp_monthly["altprc"].replace(0, np.nan)
                crsp_monthly["prc_adj"] = (
                    prc_zeroed.abs() / crsp_monthly["cfacpr"]
                ).replace([np.inf, -np.inf], np.nan)

                # Merge risk-free and compute excess return
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date, end_date=end_date
                )
                crsp_monthly = (
                    crsp_monthly.merge(risk_free_monthly, how="left", on="date")
                    .assign(ret_excess=lambda x: x["ret_adj"] - x["risk_free"])
                    .drop(columns="risk_free")
                    .dropna(subset=["ret_excess", "mktcap"])
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

                crsp_monthly = (
                    pd.read_sql_query(
                        sql=crsp_query,
                        con=wrds_connection,
                        dtype={"permno": int, "siccd": int},
                        parse_dates={
                            "date",
                            "calculation_date",
                            "first_crsp_date",
                        },
                    )
                    .assign(shrout=lambda x: x["shrout"] * 1000)
                    # listing_age is assigned before mktcap to keep the
                    # documented column order:
                    # ..., siccd, listing_age, mktcap, mktcap_lag, ...
                    .assign(
                        listing_age=lambda df: (
                            (df["date"].dt.year - df["first_crsp_date"].dt.year)
                            * 12
                            + (
                                df["date"].dt.month
                                - df["first_crsp_date"].dt.month
                            )
                            - (
                                df["date"].dt.day < df["first_crsp_date"].dt.day
                            ).astype(int)
                        ).clip(lower=0)
                    )
                    .assign(mktcap=lambda x: x["shrout"] * x["prc"] / 1000000)
                    .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
                    .drop(columns=["first_crsp_date"])
                )

                mktcap_lag = crsp_monthly.assign(
                    date=lambda x: x["date"] + pd.DateOffset(months=1),
                    mktcap_lag=lambda x: x["mktcap"],
                ).get(["permno", "date", "mktcap_lag"])

                crsp_monthly = crsp_monthly.merge(
                    mktcap_lag, how="left", on=["permno", "date"]
                ).assign(
                    exchange=lambda x: x["primaryexch"].apply(_assign_exchange),
                    industry=lambda x: x["siccd"].apply(_assign_industry),
                )
                risk_free_monthly = _download_data_risk_free(
                    start_date=start_date,
                    end_date=end_date,
                )
                crsp_monthly = (
                    crsp_monthly.merge(risk_free_monthly, how="left", on="date")
                    .assign(ret_excess=lambda x: x["ret"] - x["risk_free"])
                    .drop(columns=["risk_free"])
                    .dropna(subset=["ret_excess", "mktcap"])
                )
                processed_data = crsp_monthly
        elif "crsp_daily" in dataset:
            if version == "v1":
                # Distinct permnos from dsf within the date range
                permnos_query = text(
                    f"SELECT DISTINCT permno FROM crsp.dsf "
                    f"WHERE date BETWEEN '{start_date}' "
                    f"AND '{end_date}'"
                )
                permnos = pd.read_sql(
                    permnos_query,
                    con=wrds_connection,
                    dtype={"permno": int},
                )
                permnos = list(permnos["permno"].astype(str))

                if len(permnos) > 0:
                    batches = int(np.ceil(len(permnos) / batch_size))
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

                        dsf_query = text(f"""
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
                        """)
                        crsp_daily_sub = pd.read_sql_query(
                            dsf_query,
                            con=wrds_connection,
                            parse_dates={"date"},
                        ).dropna(subset=["permno", "date", "ret"])

                        if crsp_daily_sub.empty:
                            continue

                        # Per-batch msedelist
                        msedelist_query = text(
                            "SELECT permno, dlstdt, dlret "
                            "FROM crsp.msedelist "
                            f"WHERE permno IN {permno_in}"
                        )
                        msedelist_sub = pd.read_sql_query(
                            msedelist_query,
                            con=wrds_connection,
                            parse_dates={"dlstdt"},
                        ).dropna()

                        # Merge dsf with msedelist on (permno, date=dlstdt)
                        if not msedelist_sub.empty:
                            crsp_daily_sub = crsp_daily_sub.merge(
                                msedelist_sub.rename(
                                    columns={"dlstdt": "date"}
                                ),
                                on=["permno", "date"],
                                how="left",
                            )

                            # Bind delisting-only rows that don't
                            # match any dsf date
                            matched_dates = crsp_daily_sub[
                                crsp_daily_sub["dlret"].notna()
                            ][["permno", "date"]].drop_duplicates()
                            unmatched = (
                                msedelist_sub.merge(
                                    matched_dates.rename(
                                        columns={"date": "dlstdt"}
                                    ),
                                    on=["permno", "dlstdt"],
                                    how="left",
                                    indicator=True,
                                )
                                .query("_merge == 'left_only'")
                                .drop(columns="_merge")
                            )
                            if not unmatched.empty:
                                unmatched = unmatched.rename(
                                    columns={"dlstdt": "date"}
                                )
                                for col in crsp_daily_sub.columns:
                                    if col not in unmatched.columns:
                                        unmatched[col] = np.nan
                                unmatched = unmatched[crsp_daily_sub.columns]
                                crsp_daily_sub = pd.concat(
                                    [crsp_daily_sub, unmatched],
                                    ignore_index=True,
                                )

                            # ret = dlret if not NaN
                            crsp_daily_sub["ret"] = np.where(
                                crsp_daily_sub["dlret"].notna(),
                                crsp_daily_sub["dlret"],
                                crsp_daily_sub["ret"],
                            )
                            crsp_daily_sub = crsp_daily_sub.drop(
                                columns="dlret"
                            )

                            # Filter date <= permno's last delisting
                            # date (or end_date if no delisting)
                            permno_dlstdt = (
                                msedelist_sub.groupby("permno")["dlstdt"]
                                .max()
                                .reset_index()
                                .rename(columns={"dlstdt": "_permno_dlstdt"})
                            )
                            crsp_daily_sub = crsp_daily_sub.merge(
                                permno_dlstdt,
                                on="permno",
                                how="left",
                            )
                            crsp_daily_sub["_permno_dlstdt"] = crsp_daily_sub[
                                "_permno_dlstdt"
                            ].fillna(pd.Timestamp(end_date))
                            crsp_daily_sub = crsp_daily_sub[
                                crsp_daily_sub["date"]
                                <= crsp_daily_sub["_permno_dlstdt"]
                            ].drop(columns="_permno_dlstdt")

                        # Merge risk_free, compute ret_excess
                        crsp_daily_sub = (
                            crsp_daily_sub.merge(
                                risk_free_daily,
                                on="date",
                                how="left",
                            )
                            .assign(
                                ret_excess=lambda x: x["ret"] - x["risk_free"]
                            )
                            .drop(columns="risk_free")
                        )

                        crsp_data = pd.concat([crsp_data, crsp_daily_sub])

                # Gao-Ritter volume adjustment for NASDAQ (v1: exchcd==3)
                if adjust_volume and not crsp_data.empty:
                    gr_date_1 = pd.Timestamp("2001-02-01")
                    gr_date_2 = pd.Timestamp("2002-01-01")
                    gr_date_3 = pd.Timestamp("2004-01-01")

                    crsp_data = (
                        crsp_data.sort_values(["permno", "date"])
                        .assign(
                            vol=lambda df: df["vol"].replace(-99, np.nan),
                            prc=lambda df: df["prc"].replace(0, np.nan),
                        )
                        .assign(
                            prc_adj=lambda df: (
                                df["prc"].abs() / df["cfacpr"]
                            ).replace([np.inf, -np.inf], np.nan)
                        )
                        .assign(
                            vol_adj=lambda df: np.select(
                                [
                                    (df["exchcd"] == 3)
                                    & (df["date"] < gr_date_1),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_1)
                                    & (df["date"] < gr_date_2),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_2)
                                    & (df["date"] < gr_date_3),
                                    (df["exchcd"] == 3)
                                    & (df["date"] >= gr_date_3),
                                ],
                                [
                                    df["vol"] / 2.0,
                                    df["vol"] / 1.8,
                                    df["vol"] / 1.6,
                                    df["vol"] / 1.0,
                                ],
                                default=df["vol"],
                            )
                        )
                    )

                processed_data = crsp_data
            elif version == "v2":
                permnos = pd.read_sql(
                    sql="SELECT DISTINCT permno FROM crsp.stksecurityinfohist",
                    con=wrds_connection,
                    dtype={"permno": int},
                )
                permnos = list(permnos["permno"].astype(str))
                batches = np.ceil(len(permnos) / batch_size).astype(int)

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

                    crsp_daily_sub = pd.read_sql_query(
                        sql=crsp_daily_sub_query,
                        con=wrds_connection,
                        dtype={"permno": int},
                        parse_dates={"date"},
                    ).dropna(subset=["permno", "date", "ret"])

                    if not crsp_daily_sub.empty:
                        crsp_daily_sub = (
                            crsp_daily_sub.merge(
                                risk_free_daily,
                                on="date",
                                how="left",
                            )
                            .assign(
                                ret_excess=lambda x: x["ret"] - x["risk_free"]
                            )
                            .drop(columns=["risk_free"])
                        )

                    print(
                        f"Batch {j} out of {batches} done "
                        f"({(j / batches) * 100:.2f}%)\n"
                    )

                    crsp_data = pd.concat([crsp_data, crsp_daily_sub])

                if adjust_volume and not crsp_data.empty:
                    gr_date_1 = pd.Timestamp("2001-02-01")
                    gr_date_2 = pd.Timestamp("2002-01-01")
                    gr_date_3 = pd.Timestamp("2004-01-01")

                    crsp_data = (
                        crsp_data.sort_values(["permno", "date"])
                        .assign(
                            cfacpr=lambda df: df.groupby("permno")[
                                "dlyfacprc"
                            ].cumprod(),
                            vol=lambda df: df["dlyvol"].replace(-99, np.nan),
                            prc=lambda df: df["dlyprc"].replace(0, np.nan),
                        )
                        .assign(
                            prc_adj=lambda df: (
                                df["prc"].abs() / df["cfacpr"]
                            ).replace([np.inf, -np.inf], np.nan)
                        )
                        .assign(
                            vol_adj=lambda df: np.select(
                                [
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] < gr_date_1),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_1)
                                    & (df["date"] < gr_date_2),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_2)
                                    & (df["date"] < gr_date_3),
                                    (df["primaryexch"] == "Q")
                                    & (df["date"] >= gr_date_3),
                                ],
                                [
                                    df["vol"] / 2.0,
                                    df["vol"] / 1.8,
                                    df["vol"] / 1.6,
                                    df["vol"] / 1.0,
                                ],
                                default=df["vol"],
                            )
                        )
                        .drop(columns=["dlyvol", "dlyprc", "dlyfacprc"])
                    )

                processed_data = crsp_data
        else:
            raise ValueError(
                "Invalid dataset. Use 'crsp_monthly' or 'crsp_daily'."
            )
    finally:
        disconnect_connection(wrds_connection)

    if add_ccm_links:
        ccm_links = _download_data_wrds_ccm_links()
        merged = processed_data[["permno", "date"]].merge(
            ccm_links, on="permno", how="inner"
        )
        valid_links = merged.query(
            "gvkey.notna() and linkdt <= date <= linkenddt"
        )[["permno", "gvkey", "date"]]
        processed_data = processed_data.merge(
            valid_links, on=["permno", "date"], how="left"
        )

    return processed_data


def _download_data_wrds_ccm_links(
    linktype: list[str] = ["LU", "LC"], linkprim: list[str] = ["P", "C"]
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        A data frame containing columns permno, gvkey, linkdt, and
        linkenddt (missing end dates replaced with today's date).
    """
    conn = get_wrds_connection()

    query = f"""
        SELECT lpermno AS permno, gvkey, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ({",".join(f"'{lt}'" for lt in linktype)})
        AND linkprim IN ({",".join(f"'{lp}'" for lp in linkprim)})
    """

    ccm_links = pd.read_sql(query, conn)

    ccm_links["linkenddt"] = ccm_links["linkenddt"].fillna(pd.Timestamp.today())

    disconnect_connection(conn)

    return ccm_links


def _download_data_wrds_compustat(
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    additional_columns: list = None,
    only_usd: bool = False,
    only_us: bool = None,
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
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
        warnings.warn(
            "Passing 'wrds_'-prefixed dataset names is deprecated. "
            "Use 'compustat_annual' instead of 'wrds_compustat_annual'.",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset = re.sub(r"^wrds_", "", dataset)

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

    wrds_connection = get_wrds_connection()
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
        query = text(f"""
            SELECT gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb,
                pstkrv, pstkl, pstk, capx, oancf, sale, cogs, xint, xsga,
                ib, curcd
                {", " + additional_columns_annual if additional_columns_annual else ""}
            FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """)

        compustat = pd.read_sql(query, wrds_connection)
        disconnect_connection(wrds_connection)

        # Compute Book Equity (be)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            compustat = compustat.assign(
                be=lambda x: (
                    x["seq"]
                    .combine_first(x["ceq"] + x["pstk"])
                    .combine_first(x["at"] - x["lt"])
                    + x["txditc"].combine_first(x["txdb"] + x["itcb"]).fillna(0)
                    - x["pstkrv"]
                    .combine_first(x["pstkl"])
                    .combine_first(x["pstk"])
                    .fillna(0)
                )
            ).assign(
                be=lambda x: x["be"].apply(lambda y: np.nan if y <= 0 else y)
            )
        # Compute Operating Profitability (op)
        compustat = compustat.assign(
            op=lambda df: (
                (
                    df["sale"]
                    - df[["cogs", "xsga", "xint"]].fillna(0).sum(axis=1)
                )
                / df["be"]
            )
        )
        # Keep the latest report per company per year
        compustat = (
            compustat.assign(
                year=lambda x: pd.DatetimeIndex(x["datadate"]).year
            )
            .sort_values("datadate")
            .groupby(["gvkey", "year"])
            .tail(1)
            .reset_index(drop=True)
        )
        # Compute Investment (inv)
        compustat_lag = (
            compustat.get(["gvkey", "year", "at"])
            .assign(year=lambda x: x["year"] + 1)
            .rename(columns={"at": "at_lag"})
        )

        compustat = (
            compustat.merge(compustat_lag, how="left", on=["gvkey", "year"])
            .assign(inv=lambda x: x["at"] / x["at_lag"] - 1)
            .assign(inv=lambda x: np.where(x["at_lag"] <= 0, np.nan, x["inv"]))
        )

        if only_usd:
            compustat = compustat[compustat["curcd"] == "USD"]

        processed_data = compustat.assign(
            date=lambda df: (
                pd.to_datetime(df["datadate"]).dt.to_period("M").dt.start_time
            )
        ).drop(columns=["year", "at_lag", "curcd"])

    elif "compustat_quarterly" in dataset:
        query = text(f"""
            SELECT gvkey, datadate, rdq, fqtr, fyearq, atq, ceqq, curcdq
                {", " + additional_columns_quarterly if additional_columns_quarterly else ""}
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
        """)

        compustat = pd.read_sql(query, wrds_connection)
        disconnect_connection(wrds_connection)

        # Ensure necessary columns are not missing
        compustat = (
            compustat.dropna(subset=["gvkey", "datadate", "fyearq", "fqtr"])
            .assign(
                date=lambda df: (
                    pd.to_datetime(df["datadate"])
                    .dt.to_period("M")
                    .dt.start_time
                )
            )
            .sort_values("datadate", ascending=False, kind="stable")
            .drop_duplicates(subset=["gvkey", "fyearq", "fqtr"], keep="first")
            .sort_values(
                ["gvkey", "date", "rdq"], na_position="last", kind="stable"
            )
            .drop_duplicates(subset=["gvkey", "date"], keep="first")
            .query("rdq.isna() or date < rdq")
        )

        if only_usd:
            compustat = compustat[compustat["curcdq"] == "USD"]

        processed_data = compustat.get(
            ["date"] + _base_quarterly_cols + extra_quarterly
        )

    return processed_data


def _download_data_wrds_fisd(additional_columns: list = None) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
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
    wrds_connection = get_wrds_connection()

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

    fisd = pd.read_sql_query(
        sql=fisd_query,
        con=wrds_connection,
        dtype={
            "complete_cusip": str,
            "interest_frequency": str,
            "issue_id": int,
            "issuer_id": int,
        },
        parse_dates={
            "maturity",
            "offering_date",
            "dated_date",
            "last_interest_date",
        },
    )

    fisd_issuer_query = (
        "SELECT issuer_id, sic_code, country_domicile "
        "FROM fisd.fisd_mergedissuer"
    )

    fisd_issuer = pd.read_sql_query(
        sql=fisd_issuer_query,
        con=wrds_connection,
        dtype={"issuer_id": int, "sic_code": str, "country_domicile": str},
    )

    fisd = (
        fisd.merge(fisd_issuer, how="inner", on="issuer_id")
        .query("country_domicile == 'USA'")
        .drop(columns="country_domicile")
    )

    disconnect_connection(wrds_connection)

    return fisd.reset_index(drop=True)


def _download_data_wrds_trace_enhanced(
    cusips: list, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
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

    wrds_connection = get_wrds_connection()

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

    trace_enhanced_raw = pd.read_sql(
        query,
        wrds_connection,
        parse_dates={"trd_exctn_dt", "trd_rpt_dt", "stlmnt_dt"},
    )
    disconnect_connection(wrds_connection)

    trace_enhanced = process_trace_data(trace_enhanced_raw)

    return trace_enhanced


def get_wrds_connection() -> object:
    """Establish a connection to Wharton Research Data Services (WRDS).

    Opens a SQLAlchemy connection to the WRDS PostgreSQL database. Credentials
    are read from the 'WRDS_USER' and 'WRDS_PASSWORD' environment variables,
    which can also be supplied through a '.env' file in the working directory
    ('load_dotenv' is called before the lookup). The connection uses the
    'postgresql+psycopg2' driver, requires SSL, and targets
    'wrds-pgdata.wharton.upenn.edu' on port 9737. Pool pre-ping is enabled so
    stale connections are detected and recycled automatically.

    Returns
    -------
    sqlalchemy.engine.Connection
        An open SQLAlchemy connection to the WRDS database. Pass it to any
        'download_data_wrds_*' helper or use it directly with
        'pd.read_sql_query'. Close it via 'disconnect_connection' when done.

    Raises
    ------
    ValueError
        If 'WRDS_USER' or 'WRDS_PASSWORD' is not set in the environment or
        '.env' file. The 'load_wrds_credentials' helper performs this check
        before the connection is opened.
    sqlalchemy.exc.OperationalError
        If the database host is unreachable or the credentials are rejected.

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
    return engine.connect()


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


def process_trace_data(trace_all: pd.DataFrame) -> pd.DataFrame:
    """Clean Enhanced TRACE trade reports.

    Applies the Dick-Nielsen cleaning protocol to the raw Enhanced
    TRACE message stream, removing cancellations, corrections, and
    reversals and producing one observation per executed corporate
    bond trade.

    Parameters
    ----------
    trace_all : pd.DataFrame
        Raw Enhanced TRACE messages with the message-status columns
        used by the cleaning protocol ('trc_st', 'msg_seq_nb',
        'orig_msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm',
        'trd_exctn_dt', 'trd_exctn_tm', 'cusip_id', 'entrd_vol_qt',
        'rptd_pr', 'rpt_side_cd', 'cntra_mp_id', 'asof_cd').

    Returns
    -------
    pd.DataFrame
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
    # Post 2012-02-06
    # Trades (trc_st = T) and correction (trc_st = R)
    trace_post_TR = trace_all.query("trc_st in ['T', 'R']").query(
        "trd_rpt_dt >= '2012-02-06'"
    )

    # Cancellations (trc_st = X) and correction cancellations (trc_st = C)
    trace_post_XC = (
        trace_all.query("trc_st in ['X', 'C']")
        .query("trd_rpt_dt >= '2012-02-06'")
        .get(
            [
                "cusip_id",
                "msg_seq_nb",
                "entrd_vol_qt",
                "rptd_pr",
                "rpt_side_cd",
                "cntra_mp_id",
                "trd_exctn_dt",
                "trd_exctn_tm",
            ]
        )
        .assign(drop=True)
    )

    # Cleaning corrected and cancelled trades
    trace_post_TR = (
        trace_post_TR.merge(trace_post_XC, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Reversals (trc_st = Y)
    trace_post_Y = (
        trace_all.query("trc_st == 'Y'")
        .query("trd_rpt_dt >= '2012-02-06'")
        .get(
            [
                "cusip_id",
                "orig_msg_seq_nb",
                "entrd_vol_qt",
                "rptd_pr",
                "rpt_side_cd",
                "cntra_mp_id",
                "trd_exctn_dt",
                "trd_exctn_tm",
            ]
        )
        .assign(drop=True)
        .rename(columns={"orig_msg_seq_nb": "msg_seq_nb"})
    )

    # Clean reversals
    # Match the orig_msg_seq_nb of Y-message to msg_seq_nb of main message
    trace_post = (
        trace_post_TR.merge(trace_post_Y, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Enhanced TRACE: Pre 2012-02-06
    # Pre 2012-02-06
    # Trades (trc_st = T)
    trace_pre_T = trace_all.query("trd_rpt_dt < '2012-02-06'")

    # Cancellations (trc_st = C)
    trace_pre_C = (
        trace_all.query("trc_st == 'C'")
        .query("trd_rpt_dt < '2012-02-06'")
        .get(
            [
                "cusip_id",
                "orig_msg_seq_nb",
                "entrd_vol_qt",
                "rptd_pr",
                "rpt_side_cd",
                "cntra_mp_id",
                "trd_exctn_dt",
                "trd_exctn_tm",
            ]
        )
        .assign(drop=True)
        .rename(columns={"orig_msg_seq_nb": "msg_seq_nb"})
    )

    # Remove cancellations from trades
    # Match orig_msg_seq_nb of C-message to msg_seq_nb of main message
    trace_pre_T = (
        trace_pre_T.merge(trace_pre_C, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Corrections (trc_st = W)
    trace_pre_W = trace_all.query("trc_st == 'W'").query(
        "trd_rpt_dt < '2012-02-06'"
    )

    # Implement corrections in a loop
    # Correction control
    correction_control = len(trace_pre_W)
    correction_control_last = len(trace_pre_W)

    # Correction loop
    while correction_control > 0:
        # Create placeholder
        ## Only identifying columns of trace_pre_T (for joins)
        placeholder_trace_pre_T = (
            trace_pre_T.get(["cusip_id", "trd_exctn_dt", "msg_seq_nb"])
            .rename(columns={"msg_seq_nb": "orig_msg_seq_nb"})
            .assign(matched_T=True)
        )

        # Corrections that correct some msg
        trace_pre_W_correcting = (
            trace_pre_W.merge(placeholder_trace_pre_T, how="left")
            .query("matched_T == True")
            .drop(columns="matched_T")
        )

        # Corrections that do not correct some msg
        trace_pre_W = (
            trace_pre_W.merge(placeholder_trace_pre_T, how="left")
            .query("matched_T != True")
            .drop(columns="matched_T")
        )

        # Create placeholder
        # Only identifying columns of trace_pre_W_correcting (for anti-joins)
        placeholder_trace_pre_W_correcting = (
            trace_pre_W_correcting.get(
                ["cusip_id", "trd_exctn_dt", "orig_msg_seq_nb"]
            )
            .rename(columns={"orig_msg_seq_nb": "msg_seq_nb"})
            .assign(corrected=True)
        )

        # Delete msgs that are corrected
        trace_pre_T = (
            trace_pre_T.merge(placeholder_trace_pre_W_correcting, how="left")
            .query("corrected != True")
            .drop(columns="corrected")
        )

        # Add correction msgs
        trace_pre_T = pd.concat([trace_pre_T, trace_pre_W_correcting])

        # Escape if no corrections remain or they cannot be matched
        correction_control = len(trace_pre_W)

        if correction_control == correction_control_last:
            break
        else:
            correction_control_last = len(trace_pre_W)
            continue

    # Reversals (asof_cd = R)
    # Record reversals
    trace_pre_R = trace_pre_T.query("asof_cd == 'R'").sort_values(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"]
    )

    # Prepare final data
    trace_pre = trace_pre_T.query(
        "asof_cd == None | asof_cd.isnull() | asof_cd not in ['R', 'X', 'D']"
    ).sort_values(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"]
    )

    # Add grouped row numbers
    trace_pre_R["seq"] = trace_pre_R.groupby(
        [
            "cusip_id",
            "trd_exctn_dt",
            "entrd_vol_qt",
            "rptd_pr",
            "rpt_side_cd",
            "cntra_mp_id",
        ]
    ).cumcount()

    trace_pre["seq"] = trace_pre.groupby(
        [
            "cusip_id",
            "trd_exctn_dt",
            "entrd_vol_qt",
            "rptd_pr",
            "rpt_side_cd",
            "cntra_mp_id",
        ]
    ).cumcount()

    # Select columns for reversal cleaning
    trace_pre_R = trace_pre_R.get(
        [
            "cusip_id",
            "trd_exctn_dt",
            "entrd_vol_qt",
            "rptd_pr",
            "rpt_side_cd",
            "cntra_mp_id",
            "seq",
        ]
    ).assign(reversal=True)

    # Remove reversals and the reversed trade
    trace_pre = (
        trace_pre.merge(trace_pre_R, how="left")
        .query("reversal != True")
        .drop(columns=["reversal", "seq"])
    )

    # Combine pre and post trades
    trace_clean = pd.concat([trace_pre, trace_post])

    # Keep agency sells and unmatched agency buys
    trace_agency_sells = trace_clean.query(
        "cntra_mp_id == 'D' & rpt_side_cd == 'S'"
    )

    # Placeholder for trace_agency_sells with relevant columns
    placeholder_trace_agency_sells = trace_agency_sells.get(
        ["cusip_id", "trd_exctn_dt", "entrd_vol_qt", "rptd_pr"]
    ).assign(matched=True)

    # Agency buys that are unmatched
    trace_agency_buys_filtered = (
        trace_clean.query("cntra_mp_id == 'D' & rpt_side_cd == 'B'")
        .merge(placeholder_trace_agency_sells, how="left")
        .query("matched != True")
        .drop(columns="matched")
    )

    # Non-agency
    trace_nonagency = trace_clean.query("cntra_mp_id == 'C'")

    # Agency cleaned
    trace_clean = pd.concat(
        [trace_nonagency, trace_agency_sells, trace_agency_buys_filtered]
    )

    # Additional Filters
    trace_add_filters = (
        trace_clean.assign(
            days_to_sttl_ct2=lambda x: (
                (x["stlmnt_dt"] - x["trd_exctn_dt"]).dt.days
            )
        )
        .assign(
            days_to_sttl_ct=lambda x: pd.to_numeric(
                x["days_to_sttl_ct"], errors="coerce"
            )
        )
        .query("days_to_sttl_ct.isnull() | days_to_sttl_ct <= 7")
        .query("days_to_sttl_ct2.isnull() | days_to_sttl_ct2 <= 7")
        .query("wis_fl == 'N'")
        .query("spcl_trd_fl.isnull() | spcl_trd_fl == ''")
        .query("asof_cd.isnull() | asof_cd == ''")
    )

    # Only keep necessary columns
    trace_final = trace_add_filters.sort_values(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm"]
    ).get(
        [
            "cusip_id",
            "trd_exctn_dt",
            "trd_exctn_tm",
            "rptd_pr",
            "entrd_vol_qt",
            "yld_pt",
            "rpt_side_cd",
            "cntra_mp_id",
        ]
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
