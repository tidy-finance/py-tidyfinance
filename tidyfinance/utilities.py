"""Utility functions module for tidyfinance."""

import warnings
import webbrowser

import numpy as np
import polars as pl

_SUMMARY_QUANTILES_BASIC = (0.50,)
_SUMMARY_QUANTILES_DETAIL = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)


def _quantile_label(p: float) -> str:
    """Return the R-style quantile column label, e.g. 0.05 -> 'q05'."""
    return f"q{round(p * 100):02d}"


def create_summary_statistics(
    data: pl.DataFrame,
    variables: list,
    by: str = None,
    detail: bool = False,
    drop_na: bool = False,
) -> pl.DataFrame:
    """Create summary statistics for specified variables.

    Computes a set of summary statistics for numeric and boolean
    variables in a data frame. It allows users to select specific
    variables for summarization and can calculate statistics for the
    whole dataset or within groups specified by the 'by' argument.
    Additional detail levels for quantiles can be included.

    The function first checks that all specified variables are of a
    numeric dtype (int, float, or bool). If any variables fail this
    check, a 'ValueError' is raised listing the offending columns.
    Boolean columns are summarized as their numeric equivalent — for
    example, the 'mean' of a boolean column is the proportion of True.

    The basic set of summary statistics includes the count of non-null
    values ('n'), 'mean', standard deviation ('sd'), minimum ('min'),
    median ('q50'), and maximum ('max'). If 'detail' is True, the
    function also computes the 1st, 5th, 10th, 25th, 75th, 90th, 95th,
    and 99th percentiles ('q01' through 'q99'). Statistics are computed
    for the whole dataset, or separately for each group when 'by' is
    supplied.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing the variables to be summarized.
    variables : list of str
        List of column names in the data frame to summarize. These
        variables must be of a numeric dtype (int, float, or bool).
    by : str, optional
        Column name to group the data before summarizing. If None (the
        default), summary statistics are computed across all
        observations.
    detail : bool, default False
        Whether to compute detailed summary statistics, including
        additional quantiles. When False, computes basic statistics
        (n, mean, sd, min, q50, max). When True, additional quantiles
        (q01, q05, q10, q25, q75, q90, q95, q99) are computed.
    drop_na : bool, default False
        Whether to drop missing values for each variable before
        summarizing.

    Returns
    -------
    pl.DataFrame
        Data frame with summary statistics for each selected variable.
        If 'by' is specified, the output includes the grouping variable
        as well. Each row represents a variable (and a group if 'by' is
        used), and each column contains the computed statistics.

    Examples
    --------
    ```python
    import polars as pl
    from tidyfinance import create_summary_statistics
    data = pl.DataFrame({
        'ret': [0.01, -0.02, 0.03, None, 0.005],
        'size': [100, 200, 150, 300, 250],
        'group': ['A', 'A', 'B', 'B', 'A'],
    })
    # Basic summary across all observations
    create_summary_statistics(data, ['ret', 'size'])
    # Grouped summary
    create_summary_statistics(data, ['ret', 'size'], by='group')
    # Detailed quantiles
    create_summary_statistics(data, ['ret'], detail=True)
    ```
    """
    non_numeric_vars = [
        var
        for var in variables
        if not (
            data.schema[var].is_numeric() or data.schema[var] == pl.Boolean
        )
    ]
    if non_numeric_vars:
        raise ValueError(
            "The following columns are not numeric or boolean: "
            f"{', '.join(non_numeric_vars)}"
        )

    # Cast boolean columns to float so their mean equals the proportion
    # of True in the original.
    bool_cols = [v for v in variables if data.schema[v] == pl.Boolean]
    if bool_cols:
        data = data.with_columns(
            [pl.col(c).cast(pl.Float64) for c in bool_cols]
        )

    if drop_na:
        data = data.drop_nulls(subset=variables)

    quantiles = (
        _SUMMARY_QUANTILES_DETAIL if detail else _SUMMARY_QUANTILES_BASIC
    )

    id_cols = [by] if by else []
    long = data.select(id_cols + list(variables)).unpivot(
        index=id_cols, variable_name="variable", value_name="_value"
    )
    if long.schema["_value"].is_float():
        # Treat float NaN like missing so statistics ignore it, matching
        # the pandas/R behavior for NA values.
        long = long.with_columns(pl.col("_value").fill_nan(None))

    aggs = [
        pl.col("_value").count().alias("n"),
        pl.col("_value").mean().alias("mean"),
        pl.col("_value").std().alias("sd"),
        pl.col("_value").min().alias("min"),
        *[
            pl.col("_value")
            .quantile(p, interpolation="linear")
            .alias(_quantile_label(p))
            for p in quantiles
        ],
        pl.col("_value").max().alias("max"),
    ]

    group_cols = id_cols + ["variable"]
    result = long.group_by(group_cols, maintain_order=True).agg(aggs)

    # Order quantile columns around the median as documented.
    ordered_stats = (
        ["n", "mean", "sd", "min"]
        + [_quantile_label(p) for p in quantiles]
        + ["max"]
    )
    return result.select(group_cols + ordered_stats).sort(group_cols)


def list_supported_indexes() -> pl.DataFrame:
    """Return a DataFrame of supported financial indexes.

    Each row corresponds to one index and pairs the index name with the
    URL of an iShares CSV holdings file plus the number of header rows
    to skip when parsing it. This table powers
    'download_data_constituents'.

    Returns
    -------
    polars.DataFrame
        A DataFrame with three columns:

        - 'index': name of the financial index (e.g. 'DAX', 'S&P 500').
        - 'url': URL of the CSV file with constituent holdings.
        - 'skip': number of preamble rows in the provider file. Kept for
          compatibility; the header row is now detected automatically.

    Examples
    --------
    ```python
    from tidyfinance import list_supported_indexes
    supported_indexes = list_supported_indexes()
    print(supported_indexes)
    ```
    """
    data = [
        (
            "DAX",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/251464/ishares-dax-ucits-etf-de-fund/1495092304805.ajax?fileType=csv&fileName=DAXEX_holdings&dataType=fund",
            2,
        ),
        (
            "EURO STOXX 50",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/251783/ishares-euro-stoxx-50-ucits-etf-de-fund/1495092304805.ajax?fileType=csv&fileName=EXW1_holdings&dataType=fund",
            2,
        ),
        (
            "Dow Jones Industrial Average",
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/251770/ishares-dow-jones-industrial-average-ucits-etf-de-fund/1495092304805.ajax?fileType=csv&fileName=EXI3_holdings&dataType=fund",
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
            "https://www.ishares.com/ch/professionelle-anleger/de/produkte/253743/ishares-sp-500-b-ucits-etf-acc-fund/1495092304805.ajax?fileType=csv&fileName=SXR8_holdings&dataType=fund",
            2,
        ),
        (
            "Nasdaq 100",
            "https://www.ishares.com/de/privatanleger/de/produkte/251896/ishares-nasdaq100-ucits-etf-de-fund/1495092304805.ajax?fileType=csv&fileName=EXXT_holdings&dataType=fund",
            2,
        ),
        (
            "FTSE 100",
            "https://www.ishares.com/de/privatanleger/de/produkte/251795/ishares-ftse-100-ucits-etf-inc-fund/1495092304805.ajax?fileType=csv&fileName=IUSZ_holdings&dataType=fund",
            2,
        ),
        (
            "MSCI World",
            "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1495092304805.ajax?fileType=csv&fileName=EUNL_holdings&dataType=fund",
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
    return pl.DataFrame(
        data, schema=["index", "url", "skip"], orient="row"
    )


def list_supported_jkp_factors(
    region: str = None, dataset: str = "factors"
) -> pl.DataFrame:
    """Return the regions and factors supported by Global Factor Data.

    Queries the live availability manifest of
    `Global Factor Data <https://jkpfactors.com/data>`_ and returns the
    regions and selectors that can be passed to 'download_data_jkp'.

    Parameters
    ----------
    region : str, optional
        A region or country code. If provided, the function returns
        the selectors (factor codes, or industry classifications when
        'dataset="industry"') available for that region. If 'None'
        (the default), it returns the available region codes.
    dataset : str, default 'factors'
        The Global Factor Data product to query, one of 'factors',
        'portfolios', or 'industry'.

    Returns
    -------
    pl.DataFrame
        When 'region' is 'None', a data frame with a single 'region'
        column listing the available region codes. When 'region' is
        provided, a data frame with a 'region' column and a 'factor'
        column listing the selectors (factor codes, or industry
        classifications when 'dataset="industry"') available for that
        region.

    Examples
    --------
    ```python
    from tidyfinance import list_supported_jkp_factors
    list_supported_jkp_factors()
    list_supported_jkp_factors('usa')
    list_supported_jkp_factors('usa', dataset='portfolios')
    ```
    """
    # Imported lazily to avoid a circular import: 'download_open_source'
    # imports 'list_supported_indexes' from this module at load time.
    from .download_open_source import _fetch_jkp_availability

    supported_datasets = ("factors", "portfolios", "industry")
    if dataset not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. "
            f"Supported datasets: {', '.join(supported_datasets)}."
        )

    try:
        availability = _fetch_jkp_availability()
    except Exception:
        warnings.warn(
            "Returning an empty dataset due to download failure.",
            UserWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema={"region": pl.String})

    regions = list(availability.get(dataset, {}).keys())

    if region is None:
        return pl.DataFrame({"region": regions})

    if region not in regions:
        raise ValueError(
            f"Unsupported region: {region!r}. Use "
            "list_supported_jkp_factors() to see valid regions."
        )

    return pl.DataFrame(
        {
            "region": [region] * len(availability[dataset][region]),
            "factor": availability[dataset][region],
        }
    )


def list_tidy_finance_chapters() -> list:
    """Return the chapter slugs of the Tidy Finance book.

    Provides the URL-safe identifiers used to address chapters on the
    Tidy Finance website. The list can be passed to
    'open_tidy_finance_website' or formatted into other links.

    Returns
    -------
    list of str
        Chapter slugs in the order they appear in the book. Each slug
        corresponds to a chapter page on https://www.tidy-finance.org.

    Examples
    --------
    ```python
    from tidyfinance import list_tidy_finance_chapters
    list_tidy_finance_chapters()
    ```
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


def open_tidy_finance_website(chapter: str = None) -> None:
    """Open the Tidy Finance website or a specific chapter.

    If 'chapter' is omitted, the main landing page of the Python
    edition of Tidy Finance is opened. Otherwise the URL for the
    requested chapter is constructed and opened. An unknown chapter
    name falls back to the main page.

    Parameters
    ----------
    chapter : str, optional
        Slug of the chapter to open (e.g. 'beta-estimation'). Must
        match an entry returned by 'list_tidy_finance_chapters'.

    Returns
    -------
    None
        The function is called for its side effect of launching the
        default browser at the appropriate URL.

    Examples
    --------
    ```python
    from tidyfinance import open_tidy_finance_website
    open_tidy_finance_website()
    open_tidy_finance_website('beta-estimation')
    ```
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


def winsorize(x, cut: float) -> np.ndarray:
    """Winsorize a numeric vector at symmetric quantiles.

    Replaces values below the lower 'cut' quantile and above the upper
    '1 - cut' quantile with the corresponding quantile boundaries. The
    length of the vector is preserved.

    Parameters
    ----------
    x : numpy.ndarray or polars.Series
        A numeric vector to winsorize. Inputs that are not already
        arrays are coerced via 'numpy.asarray'.
    cut : float
        Proportion of observations replaced at each tail. For example,
        'cut=0.05' clips the lowest and highest five percent. Must lie
        in '[0, 0.5]'.

    Returns
    -------
    numpy.ndarray
        Vector with the same length as 'x' in which extreme values have
        been replaced by the lower and upper quantile cutoffs.

    Examples
    --------
    ```python
    import numpy as np
    from tidyfinance import winsorize
    rng = np.random.default_rng(123)
    data = rng.standard_normal(100)
    winsorized = winsorize(data, 0.05)
    ```
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if not (0 <= cut <= 0.5):
        raise ValueError("'cut' must be inside [0, 0.5].")

    if x.size == 0:
        return x

    x = np.asarray(x, dtype=float)
    lb, ub = np.nanquantile(x, [cut, 1 - cut])
    return np.clip(x, lb, ub)


def trim(x, cut: float) -> np.ndarray:
    """Trim a numeric vector by removing extreme observations.

    Drops values below the lower 'cut' quantile and above the upper
    '1 - cut' quantile. The returned vector is therefore shorter than
    the input.

    Parameters
    ----------
    x : numpy.ndarray or polars.Series
        A numeric vector to trim.
    cut : float
        Proportion of observations removed at each tail. For example,
        'cut=0.05' removes the lowest and highest five percent. Must
        lie in '[0, 0.5]'.

    Returns
    -------
    numpy.ndarray
        Vector with the extreme observations removed.

    Examples
    --------
    ```python
    import numpy as np
    from tidyfinance import trim
    rng = np.random.default_rng(123)
    data = rng.standard_normal(100)
    trimmed = trim(data, 0.05)
    ```
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if not (0 <= cut <= 0.5):
        raise ValueError("'cut' must be inside [0, 0.5].")

    lb = np.nanquantile(x, cut)
    ub = np.nanquantile(x, 1 - cut)

    return x[(x >= lb) & (x <= ub)]
