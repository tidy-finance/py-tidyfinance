# tidyfinance

![PyPI](https://img.shields.io/pypi/v/tidyfinance?label=pypi%20package)
![PyPI Downloads](https://img.shields.io/pypi/dm/tidyfinance)
[![python-package.yml](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml/badge.svg)](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml)
<!-- [![codecov.yml](https://codecov.io/gh/tidy-finance/py-tidyfinance/graph/badge.svg)](https://app.codecov.io/gh/tidy-finance/py-tidyfinance) -->
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Helper functions for empirical research in financial economics, addressing a variety of topics covered in [Scheuch, Voigt, Weiss, and Frey (2024)](https://doi.org/10.1201/9781032684307). The package is designed to provide shortcuts for issues extensively discussed in the book, facilitating easier application of its concepts. For more information and resources related to the book, visit [tidy-finance.org/python](https://tidy-finance.org/python).

## Installation

You can install the release version from [PyPI](https://pypi.org/project/tidyfinance/):

```
pip install tidyfinance
```

You can install the development version from GitHub:

```
pip install "git+https://github.com/tidy-finance/py-tidyfinance"
```

To use the optional [polars](https://pola.rs/) backend (see below), install the `polars` extra:

```
pip install "tidyfinance[polars]"
```

## Choosing a Data Frame Backend

By default, the public `tidyfinance` API returns [pandas](https://pandas.pydata.org/) data frames. If you prefer [polars](https://pola.rs/), switch the global backend with `set_backend()`:

```python
import tidyfinance as tf

tf.set_backend("polars")

# Returns a polars DataFrame
data = tf.download_data(
  domain="Fama-French",
  dataset="factors_ff3_monthly",
  start_date="2000-01-01",
  end_date="2020-12-31"
)

# Subsequent calls also return polars
tf.estimate_model(data, "mkt_excess")

tf.set_backend("pandas")  # back to the default
```

The setting applies to the whole public API, so any data-bearing function honors it. Polars data frames are also accepted as input regardless of the active backend (they are converted to pandas internally), so results from one call can be fed straight into the next. You can check the current setting with `tf.get_backend()`.

> **Note:** The default backend is currently `pandas`, but we expect to switch the default to `polars` from version 1.0.0 onwards. To keep your code working across that change, set the backend explicitly via `tf.set_backend(...)`.

## Download Open Source Data

The main functionality of the `tidyfinance` package centers around data download. You can download most of the data that we used in Tidy Finance with R using the `download_data()` function or its children.

```python
import tidyfinance as tf
```

The function always requires a `domain` argument and depending on the domain typically also a `dataset`. For instance, to download monthly Fama-French factors, you have to provide the dataset name according to `pdr.famafrench.get_available_datasets()`:

```python
tf.download_data(
  domain="Fama-French",
  dataset="Fama/French 5 Factors (2x3) [Daily]",
  start_date="2000-01-01",
  end_date="2020-12-31"
)
```

For q factors, you provide the relevant file name:

```python
tf.download_data(
  domain="Global Q",
  dataset="q5_factors_monthly",
  start_date="2000-01-01",
  end_date="2020-12-31"
)
```

To download the Welch and Goyal (2008) macroeconomic predictors for monthly, quarterly, or annual frequency:

```python
tf.download_data(
  domain="Goyal-Welch",
  dataset="monthly",
  start_date="2000-01-01",
  end_date="2020-12-31"
)
```

To download data from Open Source Asset Pricing (OSAP):

```python
tf.download_data(
  domain="Open Source Asset Pricing",
  start_date="2020-01-01",
  end_date="2020-12-31"
)
```

To download multiple series from the Federal Reserve Economic Data (FRED):

```python
tf.download_data(
  domain="FRED",
  series=["GDP", "CPIAUCNS"],
  start_date="2020-01-01",
  end_date="2020-12-31"
)
```

To download stock prices from Yahoo Finance:

```python
tf.download_data(
  domain="Stock Prices",
  symbols=["AAPL", "MSFT"],
  start_date="2020-01-01",
  end_date="2020-12-31"
)
```

To download index constituents from selected ETF holdings:

```python
tf.download_data(
  domain="Index Constituents",
  index="S&P 500"
)
```

## Download WRDS Data

To access data from the [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/), you need to set your credentials first:

```python
tf.set_wrds_credentials()
```

To download monthly CRSP data:

```python
tf.download_data(
  domain="WRDS",
  dataset="crsp_monthly",
  start_date="2020-01-01",
  end_date="2020-12-31"
)
```

To download annual (or quaterly) Compustat data:

```python
tf.download_data(
  domain="WRDS",
  dataset="compustat_annual",
  start_date="2020-01-01",
  end_date="2020-12-31"
)
```

To download the CRSP-Compustat linking table:

```python
tf.download_data(
  domain="WRDS",
  dataset="ccm_links"
)
```

To download bond characteristics from Mergent FISD:

```python
tf.download_data(
  domain="WRDS",
  dataset="fisd"
)
```

To download Enhanced TRACE data for selected bonds:

```python
tf.download_data(
  domain="WRDS",
  dataset="trace_enhanced",
  cusips=["00101JAH9"],
  start_date="2019-01-01",
  end_date="2021-12-31"
)
```

To download high-frequency S&P 500 data or factor library data from Hugging Face:

```python
tf.download_data(
  domain="Tidy Finance",
  dataset="high_frequency_sp500",
  start_date="2007-07-26",
  end_date="2007-07-27"
)

tf.download_data(
  domain="Tidy Finance",
  dataset="factor_library",
  sorting_variable="me"
)
```

## Other Helpers

We include functions to check out content from tidy-finance.org:

```python
tf.list_tidy_finance_chapters()
tf.open_tidy_finance_website("capital-asset-pricing-model")
```

We also include (experimental) functions that can be used for different applications, but note that they might heavily change in future package versions as we try to make them more general:

```python
# Create summary statistics
help(tf.create_summary_statistics)

# Assign portfolios
help(tf.assign_portfolio)

# Estimate betas
help(tf.estimate_betas)

# Estimate Fama-MacBeth
help(tf.estimate_fama_macbeth)

# Add lag columns
help(tf.add_lagged_columns)

# Winsorize or trim
help(tf.winsorize)
help(tf.trim)
```
