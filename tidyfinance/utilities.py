"""Utility functions module for tidyfinance."""

import os
import webbrowser

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_wrds_connection(config_path: str = "config.yaml") -> object:
    """
    Establish a connection to Wharton Research Data Services (WRDS) database.

    The function retrieves WRDS credentials from environment variables or
    a config.yaml file  and connects to the WRDS PostgreSQL database using
    SQLAlchemy.

    Parameters
    ----------
        config_path (str): Path to the configuration file.
        Default is "config.yaml".

    Returns
    -------
        object: A connection object to interact with the WRDS database.
    """
    wrds_user, wrds_password = load_wrds_credentials(config_path)

    engine = create_engine(
        (
            f"postgresql+psycopg2://{wrds_user}:{wrds_password}"
            "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
        ),
        connect_args={"sslmode": "require"},
    )
    return engine.connect()


def disconnect_connection(connection: object) -> bool:
    """Close the database connection.

    Parameters
    ----------
        connection (object): The active database connection to be closed.

    Returns
    -------
        bool: True if disconnection was successful, False otherwise.
    """
    try:
        connection.close()
        return True
    except Exception:
        return False


def list_supported_indexes() -> pd.DataFrame:
    """
    Return a DataFrame containing information on supported financial indexes.

    Each index is associated with a URL pointing to a CSV file containing
    the holdings of the index and a `skip` value indicating the number of
    lines to skip when reading the CSV.

    Returns
    -------
        pd.DataFrame: A DataFrame with three columns:
            - index (str): The name of the financial index
            (e.g., "DAX", "S&P 500").
            - url (str): The URL to the CSV file containing holdings data.
            - skip (int): The number of lines to skip when reading CSV file.
    """
    data = [
        (
            "DAX",
            "https://www.ishares.com/de/privatanleger/de/produkte/251464/ishares-dax-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=DAXEX_holdings&dataType=fund",
            2,
        ),
        (
            "EURO STOXX 50",
            "https://www.ishares.com/de/privatanleger/de/produkte/251783/ishares-euro-stoxx-50-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXW1_holdings&dataType=fund",
            2,
        ),
        (
            "Dow Jones Industrial Average",
            "https://www.ishares.com/de/privatanleger/de/produkte/251770/ishares-dow-jones-industrial-average-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXI3_holdings&dataType=fund",
            2,
        ),
        (
            "Russell 1000",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239707/ishares-russell-1000-etf/1495092304805.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
            9,
        ),
        (
            "Russell 2000",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239710/ishares-russell-2000-etf/1495092304805.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
            9,
        ),
        (
            "Russell 3000",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239714/ishares-russell-3000-etf/1495092304805.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
            9,
        ),
        (
            "S&P 100",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/239723/ishares-sp-100-etf/1495092304805.ajax?fileType=csv&fileName=OEF_holdings&dataType=fund",
            9,
        ),
        (
            "S&P 500",
            "https://www.ishares.com/de/privatanleger/de/produkte/253743/ishares-sp-500-b-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=SXR8_holdings&dataType=fund",
            2,
        ),
        (
            "Nasdaq 100",
            "https://www.ishares.com/de/privatanleger/de/produkte/251896/ishares-nasdaq100-ucits-etf-de-fund/1478358465952.ajax?fileType=csv&fileName=EXXT_holdings&dataType=fund",
            2,
        ),
        (
            "FTSE 100",
            "https://www.ishares.com/de/privatanleger/de/produkte/251795/ishares-ftse-100-ucits-etf-inc-fund/1478358465952.ajax?fileType=csv&fileName=IUSZ_holdings&dataType=fund",
            2,
        ),
        (
            "MSCI World",
            "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=EUNL_holdings&dataType=fund",
            2,
        ),
        (
            "Nikkei 225",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/253742/ishares-nikkei-225-ucits-etf/1495092304805.ajax?fileType=csv&fileName=CSNKY_holdings&dataType=fund",
            2,
        ),
        (
            "TOPIX",
            "https://www.blackrock.com/jp/individual-en/en/products/279438/fund/1480664184455.ajax?fileType=csv&fileName=1475_holdings&dataType=fund",
            2,
        ),
    ]
    return pd.DataFrame(data, columns=["index", "url", "skip"])


def list_tidy_finance_chapters() -> list:
    """
    Return a list of available chapters in the Tidy Finance resource.

    Returns
    -------
        list: A list where each element is the name of a chapter available in
        the Tidy Finance resource.
    """
    return [
        "setting-up-your-environment",
        "working-with-stock-returns",
        "modern-portfolio-theory",
        "capital-asset-pricing-model",
        "financial-statement-analysis",
        "discounted-cash-flow-analysis",
        "accessing-and-managing-financial-data",
        "wrds-crsp-and-compustat",
        "trace-and-fisd",
        "other-data-providers",
        "beta-estimation",
        "univariate-portfolio-sorts",
        "size-sorts-and-p-hacking",
        "value-and-bivariate-sorts",
        "replicating-fama-and-french-factors",
        "fama-macbeth-regressions",
        "fixed-effects-and-clustered-standard-errors",
        "difference-in-differences",
        "factor-selection-via-machine-learning",
        "option-pricing-via-machine-learning",
        "parametric-portfolio-policies",
        "constrained-optimization-and-backtesting",
        "wrds-dummy-data",
        "cover-and-logo-design",
        "clean-enhanced-trace-with-r",
        "proofs",
        "changelog",
    ]


def load_wrds_credentials(config_path: str = "./config.yaml") -> tuple:
    """
    Load WRDS credentials from a config.yaml file if env variables are not set.

    Parameters
    ----------
        config_path (str): Path to the configuration file.
        Default is "config.yaml".

    Returns
    -------
        tuple: A tuple containing (wrds_user (str), wrds_password (str)).===================================================================================================================================================================================================================================================================================================================================
    """
    load_dotenv()

    wrds_user: str = os.getenv("WRDS_USER")
    wrds_password: str = os.getenv("WRDS_PASSWORD")

    if not wrds_user or not wrds_password:
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                wrds_user = config.get("WRDS", {}).get("USER", "")
                wrds_password = config.get("WRDS", {}).get("PASSWORD", "")

    if not wrds_user or not wrds_password:
        raise ValueError(
            "WRDS credentials not found. Please set 'WRDS_USER' "
            "and 'WRDS_PASSWORD' as environment variables or "
            "in config.yaml."
        )

    return wrds_user, wrds_password


def open_tidy_finance_website(chapter: str = None) -> None:
    """Open the Tidy Finance website or a specific chapter in a browser.

    Parameters
    ----------
        chapter (str, optional): Name of the chapter to open. Defaults to None.

    Returns
    -------
        None
    """
    base_url = "https://www.tidy-finance.org/python/"

    if chapter:
        tidy_finance_chapters = list_tidy_finance_chapters()
        if chapter in tidy_finance_chapters:
            final_url = f"{base_url}{chapter}.html"
        else:
            final_url = base_url
    else:
        final_url = base_url

    webbrowser.open(final_url)


def process_trace_data(trace_all: pd.DataFrame) -> pd.DataFrame:
    """
    Process TRACE data by filtering trades, handling exception.

    Parameters
    ----------
        trace_all (pd.DataFrame): The raw TRACE data.

    Returns
    -------
        pd.DataFrame: The cleaned and processed TRACE data.
    """
    # Post 2012-06-02
    ## Trades (trc_st = T) and correction (trc_st = R)
    trace_post_TR = trace_all.query("trc_st in ['T', 'R']").query(
        "trd_rpt_dt >= '2012-06-02'"
    )

    # Cancellations (trc_st = X) and correction cancellations (trc_st = C)
    trace_post_XC = (
        trace_all.query("trc_st in ['X', 'C']")
        .query("trd_rpt_dt >= '2012-06-02'")
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

    ## Cleaning corrected and cancelled trades
    trace_post_TR = (
        trace_post_TR.merge(trace_post_XC, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Reversals (trc_st = Y)
    trace_post_Y = (
        trace_all.query("trc_st == 'Y'")
        .query("trd_rpt_dt >= '2012-06-02'")
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
    ## Match the orig_msg_seq_nb of Y-message to msg_seq_nb of main message
    trace_post = (
        trace_post_TR.merge(trace_post_Y, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Enhanced TRACE: Pre 06-02-2012
    # Pre 06-02-12
    ## Trades (trc_st = T)
    trace_pre_T = trace_all.query("trd_rpt_dt < '2012-06-02'")

    # Cancellations (trc_st = C)
    trace_pre_C = (
        trace_all.query("trc_st == 'C'")
        .query("trd_rpt_dt < '2012-06-02'")
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
    ## Match orig_msg_seq_nb of C-message to msg_seq_nb of main message
    trace_pre_T = (
        trace_pre_T.merge(trace_pre_C, how="left")
        .query("drop != True")
        .drop(columns="drop")
    )

    # Corrections (trc_st = W)
    trace_pre_W = trace_all.query("trc_st == 'W'").query(
        "trd_rpt_dt < '2012-06-02'"
    )

    # Implement corrections in a loop
    ## Correction control
    correction_control = len(trace_pre_W)
    correction_control_last = len(trace_pre_W)

    ## Correction loop
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
        ## Only identifying columns of trace_pre_W_correcting (for anti-joins)
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
    ## Record reversals
    trace_pre_R = trace_pre_T.query("asof_cd == 'R'").sort_values(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"]
    )

    ## Prepare final data
    trace_pre = trace_pre_T.query(
        "asof_cd == None | asof_cd.isnull() | asof_cd not in ['R', 'X', 'D']"
    ).sort_values(
        ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "trd_rpt_dt", "trd_rpt_tm"]
    )

    ## Add grouped row numbers
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

    ## Select columns for reversal cleaning
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

    ## Remove reversals and the reversed trade
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
    """Set WRDS credentials in the environment.

    Prompts the user for WRDS credentials and stores them in a YAML
    configuration file.

    The user can choose to store the credentials in the project directory or
    the home directory. If credentials already exist, the user is prompted for
    confirmation before overwriting them. Additionally, the user is given an
    option to add the configuration file to .gitignore.

    Returns
    -------
        - Saves the WRDS credentials in a `config.yaml` file
        - Optionally adds `config.yaml` to `.gitignore`
    """
    wrds_user = input("Enter your WRDS username: ")
    wrds_password = input("Enter your WRDS password: ")
    location_choice = (
        input(
            "Where do you want to store the config.yaml "
            "file? Enter 'project' for project directory or "
            "'home' for home directory: "
        )
        .strip()
        .lower()
    )

    if location_choice == "project":
        config_path = os.path.join(os.getcwd(), "config.yaml")
        gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    elif location_choice == "home":
        config_path = os.path.join(os.path.expanduser("~"), "config.yaml")
        gitignore_path = os.path.join(os.path.expanduser("~"), ".gitignore")
    else:
        print(
            "Invalid choice. Please start again and enter 'project' or 'home'."
        )
        return

    config: dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}

    if (
        "WRDS" in config
        and "USER" in config["WRDS"]
        and "PASSWORD" in config["WRDS"]
    ):
        overwrite_choice = (
            input(
                "Credentials already exist. Do you want to "
                "overwrite them? Enter 'yes' or 'no': "
            )
            .strip()
            .lower()
        )
        if overwrite_choice != "yes":
            print("Aborted. Credentials already exist and are not overwritten.")
            return

    if os.path.exists(gitignore_path):
        add_gitignore = (
            input(
                "Do you want to add config.yaml to .gitignore? "
                "It is highly recommended! "
                "Enter 'yes' or 'no': "
            )
            .strip()
            .lower()
        )
        if add_gitignore == "yes":
            with open(gitignore_path, "r") as file:
                gitignore_lines = file.readlines()
            if "config.yaml\n" not in gitignore_lines:
                with open(gitignore_path, "a") as file:
                    file.write("config.yaml\n")
                print("config.yaml added to .gitignore.")
        elif add_gitignore == "no":
            print("config.yaml NOT added to .gitignore.")
        else:
            print("Invalid choice. Please start again and enter 'yes' or 'no'.")
            return

    config["WRDS"] = {"USER": wrds_user, "PASSWORD": wrds_password}

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    print(
        "WRDS credentials have been set and saved in config.yaml in your "
        f"{location_choice} directory."
    )


def winsorize(x: np.ndarray, cut: float) -> np.ndarray:
    """Winsorize a numeric vector by replacing extreme values.

    Parameters
    ----------
        x (pd.Series): Numeric vector to winsorize.
        cut (float): Proportion to replace at both ends.

    Returns
    -------
        pd.Series: Winsorized vector.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        # raise ValueError("x must be an numpy array")

    if not (0 <= cut <= 0.5):
        raise ValueError("'cut' must be inside [0, 0.5].")

    if x.size == 0:
        return x

    x = np.array(x)  # Convert input to numpy array if not already
    lb, ub = np.nanquantile(x, [cut, 1 - cut])  # Compute quantiles
    x = np.clip(x, lb, ub)  # Winsorize values
    return x


def trim(x: np.ndarray, cut: float) -> np.ndarray:
    """
    Remove values in a numeric vector beyond the specified quantiles.

    Parameters
    ----------
        x (np.ndarray): A numeric array to be trimmed.
        cut (float): The proportion of data to be trimmed from both ends
        (must be between [0, 0.5]).

    Returns
    -------
        np.ndarray: A numeric array with extreme values removed.
    """
    if not (0 <= cut <= 0.5):
        raise ValueError("'cut' must be inside [0, 0.5].")

    lb = np.nanquantile(x, cut)
    ub = np.nanquantile(x, 1 - cut)

    return x[(x >= lb) & (x <= ub)]
