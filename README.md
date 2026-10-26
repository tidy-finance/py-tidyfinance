# tidyfinance

![PyPI](https://img.shields.io/pypi/v/tidyfinance?label=pypi%20package)
![PyPI Downloads](https://img.shields.io/pypi/dm/tidyfinance)
[![python-package.yml](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml/badge.svg)](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml)
<!-- [![codecov.yml](https://codecov.io/gh/tidy-finance/py-tidyfinance/graph/badge.svg)](https://app.codecov.io/gh/tidy-finance/py-tidyfinance) -->
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Helper functions for empirical research in financial economics, addressing a variety of topics covered in [Scheuch, Voigt, Weiss, and Frey (2024)](https://doi.org/10.1201/9781032684307). The package is designed to provide shortcuts for issues extensively discussed in the book, facilitating easier application of its concepts. For more information and resources related to the book, visit [tidy-finance.org](https://www.tidy-finance.org).

## Installation

You can install the release version from [PyPI](https://pypi.org/project/tidyfinance/):

```
pip install tidyfinance
```

You can install the development version from GitHub:

```
pip install "git+https://github.com/tidy-finance/py-tidyfinance"
```

## Usage

Import the package:

```python
import tidyfinance as tf
```

## Choosing a Data Frame Backend

The package is implemented in [polars](https://pola.rs/), and since
version 0.9.0 the public `tidyfinance` API returns polars data frames by
default, as used throughout the book. If you prefer
[pandas](https://pandas.pydata.org/), switch the global backend with
`set_backend()`. The pandas backend needs the optional `pandas` extra:

```
pip install "tidyfinance[pandas]"
```

```python
import tidyfinance as tf

tf.set_backend("pandas")

# Returns a pandas DataFrame
data = tf.download_data(
    domain="Fama-French",
    dataset="Fama/French 3 Factors",
    start_date="2000-01-01",
    end_date="2020-12-31",
)

# Subsequent calls also return pandas
tf.estimate_model(data, "mkt_excess ~ smb + hml")
tf.set_backend("polars")  # back to the default
```

The setting applies to the whole public API, so any data-bearing function honors it. Pandas data frames are also accepted as input regardless of the active backend (they are converted to polars internally), so results from one call can be fed straight into the next. Under the default `polars` backend there is no conversion overhead at all, since polars is the native representation. You can check the current setting with `tf.get_backend()`.

> **Note:** Before version 0.9.0, the default backend was `pandas`. If your code expects pandas data frames, call `tf.set_backend("pandas")` once after importing the package.

## Download Open Source Data

The main functionality of the `tidyfinance` package centers around data download. You can download most of the data that we used in [Tidy Finance with Python](https://www.tidy-finance.org/index.html) using the `download_data()` function or its children.

The function always requires a `domain` argument and depending on the domain typically also a `dataset`. For instance, to download monthly Fama-French factors, you have to provide the dataset name according to [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):

```python
tf.download_data(
    domain="Fama-French",
    dataset="Fama/French 5 Factors (2x3) [Daily]",
    start_date="2000-01-01",
    end_date="2020-12-31",
)
```

For [q factors](https://global-q.org/index.html), you provide the relevant file name:

```python
tf.download_data(
    domain="Global Q",
    dataset="q5_factors_monthly_2024",
    start_date="2000-01-01",
    end_date="2020-12-31",
)
```

To download the Welch and Goyal (2008) macroeconomic predictors for monthly, quarterly, or annual frequency:

```python
tf.download_data(
    domain="Goyal-Welch",
    dataset="monthly",
    start_date="2000-01-01",
    end_date="2020-12-31",
)
```

To download data from [Open Source Asset Pricing](https://www.openassetpricing.com/) (OSAP):

```python
tf.download_data(
    domain="Open Source Asset Pricing",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download characteristic-managed portfolio (factor) returns from [Global Factor Data](https://jkpfactors.com/data), select a region, the factor content (a single factor, a theme, `"all_themes"`, or `"all_factors"`), a frequency, and a weighting scheme:

```python
tf.download_data(
    domain="Global Factor Data",
    region="usa",
    factors="all_factors",
    frequency="monthly",
    weighting="vw_cap",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

The `dataset` argument also gives access to the underlying long-short portfolios (`"portfolios"`), industry returns (`"industry"`), and the reference files `"nyse_cutoffs"` and `"return_cutoffs"`:

```python
tf.download_data(
    domain="Global Factor Data",
    dataset="industry",
    region="usa",
    classification="gics",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

Use `tf.list_supported_jkp_factors()` to see the available regions, or `tf.list_supported_jkp_factors("usa")` to see the factors available for a region.

To download the liquidity factors of Pastor and Stambaugh (2003) from [Robert Stambaugh's data library](https://fnce.wharton.upenn.edu/profile/stambaug/#misc):

```python
tf.download_data(
    domain="Pastor-Stambaugh",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download the mispricing factors of Stambaugh and Yuan (2017) from [Robert Stambaugh's data library](https://fnce.wharton.upenn.edu/profile/stambaug/#misc), optionally selecting `"monthly"` (the default) or `"daily"` data. Note that the source files currently end in December 2016:

```python
tf.download_data(
    domain="Stambaugh-Yuan",
    dataset="monthly",
    start_date="2015-01-01",
    end_date="2016-12-31",
)
```

To download the cost of capital (WACC) by industry from [Aswath Damodaran's data library](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/data.html), optionally selecting a `region` (e.g., `"us"`, the default, or `"europe"`) and archived years via `year`. Reading the Excel files requires the optional `excel` extra (`pip install "tidyfinance[excel]"`):

```python
tf.download_data(
    domain="Damodaran",
    dataset="wacc",
    region="us",
    year=[2023, 2024],
)
```

To download multiple series from the Federal Reserve Economic Data (FRED):

```python
tf.download_data(
    domain="FRED",
    series=["GDP", "CPIAUCNS"],
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download the FRED-MD / FRED-QD (McCracken-Ng) macroeconomic databases, a curated, balanced panel of monthly or quarterly macro series with McCracken-Ng stationarity transform codes. Set `transform=True` to apply the transforms, and `vintage` to a `"YYYY-MM"` label or `"all"` to access historical (real-time) releases:

```python
tf.download_data(domain="FRED", dataset="FRED-MD", transform=True)

tf.download_data(domain="FRED", dataset="FRED-QD", vintage="2020-03")
```

To download stock prices from Yahoo Finance:

```python
tf.download_data(
    domain="Stock Prices",
    symbols=["AAPL", "MSFT"],
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download index constituents from selected ETF holdings:

```python
tf.download_data(domain="Index Constituents", index="S&P 500")
```

If the download fails, save the holdings CSV from the ETF's web page and read it from disk instead:

```python
tf.download_data(domain="Index Constituents", path="holdings.csv")
```

You can also download high-frequency data for the S&P 500 that we host on Hugging Face:

```python
tf.download_data(
    domain="Tidy Finance",
    dataset="high_frequency_sp500",
    start_date="2007-07-26",
    end_date="2007-07-27",
)
```

## Download WRDS Data

You can also download data directly from [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/) if you have access to the underlying data and set your credentials via the environment variables `WRDS_USER` and `WRDS_PASSWORD` (e.g., in a `.env` file) or the `set_wrds_credentials()` helper function:

```python
tf.set_wrds_credentials()
```

A WRDS account alone is not enough to connect: you also need to enroll in two-factor authentication (Duo) and accept the WRDS terms of use on the WRDS website. When you connect, approve the Duo push notification on your device. See `help(tf.get_wrds_connection)` for the full setup checklist.

The WRDS downloads tell you when they connect and how long the download took. Pass `quiet=True` to suppress these messages.

To download monthly CRSP data:

```python
tf.download_data(
    domain="WRDS",
    dataset="crsp_monthly",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download annual (or quarterly) Compustat data:

```python
tf.download_data(
    domain="WRDS",
    dataset="compustat_annual",
    start_date="2020-01-01",
    end_date="2020-12-31",
)
```

To download the CRSP-Compustat linking table:

```python
tf.download_data(domain="WRDS", dataset="ccm_links")
```

To download bond characteristics from Mergent FISD:

```python
tf.download_data(domain="WRDS", dataset="fisd")
```

To download Enhanced TRACE data for selected bonds:

```python
tf.download_data(
    domain="WRDS",
    dataset="trace_enhanced",
    cusips=["00101JAH9"],
    start_date="2019-01-01",
    end_date="2021-12-31",
)
```

## Factor Library Tools

You can download pre-computed portfolio returns from the [Tidy Finance Factor Library](https://factors.tidy-finance.org) hosted on Hugging Face:

```python
tf.download_data(
    domain="Tidy Finance",
    dataset="factor_library",
    sorting_variable="assetgrowth",
)
```

The package also provides functions to construct your own factor portfolios:

```python
# Portfolio sorts and long-short returns
tf.compute_portfolio_returns()
tf.compute_long_short_returns()

# Configuration helpers
tf.breakpoint_options()
tf.data_options()

# Rolling and lagging
tf.add_lagged_columns()
tf.join_lagged_values()
tf.compute_rolling_value()

# Regression estimation
tf.estimate_model()
tf.estimate_betas()
tf.estimate_fama_macbeth()
```

## Other Helpers

We include functions to check out content from [tidy-finance.org](https://www.tidy-finance.org/index.html):

```python
tf.list_tidy_finance_chapters()
tf.open_tidy_finance_website()
```

There are also some simple helpers for regression analyses:

```python
tf.winsorize()
tf.trim()
tf.create_summary_statistics()
```
