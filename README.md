# tidyfinance

![PyPI](https://img.shields.io/pypi/v/tidyfinance?label=pypi%20package)
![PyPI Downloads](https://img.shields.io/pypi/dm/tidyfinance)
[![python-package.yml](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml/badge.svg)](https://github.com/tidy-finance/py-tidyfinance/actions/workflows/python-package.yml)
<!-- [![codecov.yml](https://codecov.io/gh/tidy-finance/py-tidyfinance/graph/badge.svg)](https://app.codecov.io/gh/tidy-finance/py-tidyfinance) -->
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Helper functions for empirical research in financial economics, addressing a variety of topics covered in [Scheuch, Frey, Voigt, and Weiss (2024)](https://doi.org/10.1201/9781032684307). The package is designed to provide shortcuts for issues extensively discussed in the book, facilitating easier application of its concepts. For more information and resources related to the book, visit [tidy-finance.org/python](https://tidy-finance.org/python).

## Installation

You can install the release version from [PyPI](https://pypi.org/project/tidyfinance/):

```
pip install tidyfinance
```

You can install the development version from GitHub:

```
pip install "git+https://github.com/tidy-finance/py-tidyfinance"
```

## Download Open Source Data

The main functionality of the `tidyfinance` package centers around data download. You can download most of the data that we used in Tidy Finance with R using the `download_data()` function or its children. 

```python
from tidyfinance import download_data
```

For instance, to download monthly Fama-French factors, you have to provide the dataset name according to `pdr.famafrench.get_available_datasets()`:

```python
download_data(
  domain="factors_ff",
  dataset="F-F_Research_Data_5_Factors_2x3_daily",
  start_date="2000-01-01", 
  end_date="2020-12-31"
)
```

For q factors, you provide the relevant file name:

```python
download_data(
  domain="factors_q",
  dataset="q5_factors_monthly",
  start_date="2000-01-01", 
  end_date="2020-12-31"
)
```

To download the Welch and Goyal (2008) macroeconomic predictors for monthly, quarterly, or annual frequency:

```python
download_data(
  domain="macro_predictors",
  frequency="monthly",
  start_date="2000-01-01", 
  end_date="2020-12-31"
)
```

To download data from Open Source Asset Pricing (OSAP):

```python
download_data(
  domain="osap",
  start_date="2020-01-01", 
  end_date="2020-12-31"
)
```

To download multiple series from the Federal Reserve Economic Data (FRED):

```python
download_data(
  domain="fred",
  series=["GDP", "CPIAUCNS"], 
  start_date="2020-01-01", 
  end_date="2020-12-31"
)
```

To download stock prices from Yahoo Finance:

```python
download_data(
  domain="stock_prices",
  symbols=["AAPL", "MSFT"], 
  start_date="2020-01-01", 
  end_date="2020-12-31"
)
```

To download index constituents from selected ETF holdings: 

```python
download_data(
  domain="constituents",
  index="S&P 500"
)
```

## Download WRDS Data

To access data from the [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/), you need to set your credentials first:

```python
from tidyfinance import set_wrds_credentials
set_wrds_credentials()
```

```python
download_data(
  domain="wrds",
  dataset="crsp_monthly", 
  start_date="2020-01-01", 
  end_date="2020-12-31"
)
```

```python
download_data(
  domain="wrds",
  dataset="compustat_annual", 
  start_date="2020-01-01", 
  end_date="2020-12-31"
)
```

```python
download_data(
  domain="wrds",
  dataset="ccm_links"
)
```

```python
download_data(
  domain="wrds",
  dataset="fisd"
)
```

```python
download_data(
  domain="wrds",
  dataset="trace_enhanced",
  cusips=["00101JAH9"],
  start_date="2019-01-01", 
  end_date="2021-12-31"
)
```

## Other Helpers

We include functions to check out content from tidy-finance.org:

```python
from tidyfinance import list_tidy_finance_chapters, open_tidy_finance_website

list_tidy_finance_chapters()
open_tidy_finance_website("capital-asset-pricing-model")
```

We also include (experimental) functions that can be used for different applications, but note that they might heavily change in future package versions as we try to make them more general:

```python
import tidyfinance as tf
# For portfolio sorts
help(tf.assign_portfolio)

# For beta estimation
help(tf.estimate_betas)

# For Fama-MacBeth estimation
help(tf.estimate_fama_macbeth)
```