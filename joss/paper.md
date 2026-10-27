---
title: "tidyfinance: Transparent factor construction for empirical asset pricing in Python"
tags:
  - Python
  - finance
  - asset pricing
  - portfolio sorts
  - CRSP
  - Compustat
  - reproducible research
authors:
  - name: Christoph Frey
    orcid: 0000-0002-9937-0903
    affiliation: 1
  - name: Christoph Scheuch
    orcid: 0009-0004-0423-6819
    corresponding: true
    affiliation: 2
  - name: Stefan Voigt
    orcid: 0000-0001-5619-3161
    affiliation: "3, 4"
  - name: Patrick Weiss
    orcid: 0000-0002-9282-5872
    affiliation: 5
affiliations:
  - name: Lancaster University Management School, United Kingdom
    index: 1
  - name: School of Business and Economics, Humboldt University of Berlin, Germany
    index: 2
  - name: Department of Economics, University of Copenhagen, Denmark
    index: 3
  - name: Danish Finance Institute, Denmark
    index: 4
  - name: Department of Business and Economics, Reykjavík University, Iceland
    index: 5
date: 27 October 2026
bibliography: paper.bib
---

# Summary

`tidyfinance` is a Python package that builds factor returns for empirical asset pricing from raw data. It turns each step of the construction into a function, and each construction choice into an argument with a conventional default. A default call reproduces standard practice; changing one choice takes one argument.

Building a factor takes three steps. Researchers download and clean return and accounting data, typically from CRSP and Compustat. They lag firm characteristics so that portfolios use only information available at formation. They then sort stocks into portfolios and take the return difference between the extreme portfolios. Each step involves choices about delistings, fiscal-year lags, breakpoints, weights, and rebalancing, and few studies publish these choices as code.

The package covers all three steps. `download_data()` returns cleaned panels from Wharton Research Data Services (CRSP, Compustat, TRACE, FISD), Ken French's Data Library, FRED, and other providers. Further functions lag characteristics, compute breakpoints, assign portfolios, aggregate long-short returns, and run Fama-MacBeth regressions. We write the package in polars [@polars], and it accepts and returns both polars and pandas [@mckinney.2010] data frames. It implements the same design as the R package of the same name [@tidyfinance.r; @frey.2026].

# Statement of need

Researchers who want to reproduce or modify a published factor usually rebuild its construction from scratch. Ken French's Data Library, the most widely used factor source, publishes returns without the code that produced them [@fama.2023]. Projects that do publish code release fixed scripts for a given set of factors, not functions that a researcher can call with different settings [@chen.2022; @jensen.2023; @dickerson.2023]. Yet construction choices change results. @hou.2020, @walter.2024, and @soebhag.2024 show that reasonable changes to breakpoints, filters, and weights change portfolio-sort results materially, and @akey.2026 show that the returns in Ken French's library change when its method changes.

`tidyfinance` serves researchers, PhD students, and instructors who work in Python and want factor returns they can build from raw data, inspect, and vary. They need three things. First, a cleaned CRSP-Compustat panel without writing their own SQL and cleaning code. Second, every construction choice as a parameter, so that a robustness check changes an argument instead of a script. Third, results that agree with the R package, so that teams using both languages get the same numbers.

# State of the field

Existing Python tools cover data access or estimation, but not the portfolio construction in between. The `wrds` package [@wrds.python] opens a connection to WRDS and leaves universe filters, delisting adjustments, and the CRSP-Compustat link to the user. `pandas-datareader` [@pandas.datareader] and `yfinance` [@yfinance] retrieve FRED series, French library files, and market prices, without the cleaning conventions of asset pricing studies. `linearmodels` [@linearmodels] estimates Fama-MacBeth and linear factor models on a panel of characteristics and returns that the user must already have.

Factor data releases supply finished factor returns, sometimes with the scripts that produced them, fixed at a release date. Open Asset Pricing [@chen.2022; @chen.2025] publishes the construction code for 212 cross-sectional signals, written in Python since version 2.0, and the `openassetpricing` package [@li.2024] downloads the resulting signals and portfolio returns. The Global Factor Data project publishes its construction code in Python as well [@jensen.2023]. Both are research pipelines, not libraries. To rebalance a factor in January instead of July, or to form it on a different fiscal-year lag, a user must copy the code and edit it.

PyBondLab [@rossetti.2026] comes closest. It implements portfolio sorts for corporate bonds as a callable API, with single and double sorts, staggered rebalancing, within-issuer sorts, turnover reports, and banding. `tidyfinance` differs in three ways. It starts from raw WRDS panels, because it includes data download and cleaning. It targets stocks, with options such as NYSE breakpoints, capped value weights, and industry and earnings screens. And it adds Fama-MacBeth regressions to portfolio sorts. The two packages complement each other across asset classes.

We wrote a new package because no existing project covers the path from raw WRDS data to factor returns, and because we want one API that matches the R package argument for argument.

# Software design

Two principles from the open-access textbooks *Tidy Finance with R* and *Tidy Finance with Python* [@scheuch.2023; @scheuch.2024] shape the package. Every construction choice is a parameter with a documented default. Every step is a small function that takes a data frame and returns one, so users can chain the full workflow or reuse single steps in their own code.

The public API follows the three steps of the workflow:

- **Data download.** `download_data(domain, dataset, ...)` covers 334 datasets from 13 providers. For CRSP, it keeps common stocks on NYSE, AMEX, and NASDAQ, includes delisting returns, computes lagged market capitalization, and subtracts a risk-free rate spliced from FRED Treasury bill series. For Compustat, it computes book equity with the usual priority rules. For TRACE, it applies the cleaning steps of @dicknielsen.2009. Every panel has one `date` column with the same convention across providers, so users join sources on equal dates. The `"Pseudo Data"` domain returns simulated panels with the WRDS column layout, so users without a subscription can run, test, and teach the full workflow.
- **Variable construction.** `add_lagged_columns()` and `join_lagged_values()` attach the most recent value observed within a look-back window. With `ff_adjustment=True`, they instead follow @fama.1993 and use fiscal-year data from July of the following calendar year. Further functions winsorize, trim, compute rolling values over calendar windows, and apply sample filters.
- **Portfolio sorts and estimation.** `compute_breakpoints()`, `assign_portfolio()`, `compute_portfolio_returns()`, and `compute_long_short_returns()` are the building blocks. `implement_portfolio_sort()` combines them for univariate, independent bivariate, and dependent bivariate sorts, with monthly, annual, or staggered rebalancing [@jegadeesh.1993]. `filter_options()`, `breakpoint_options()`, and `portfolio_sort_options()` collect the choices in one object. `estimate_fama_macbeth()` runs @fama.1973 regressions with prewhitened @newey.1987 standard errors and the @newey.1994 bandwidth.

The code below builds the asset growth factor of @cooper.2008 from raw CRSP and Compustat data. With `"Pseudo Data"` in place of `"WRDS"`, it runs without a subscription.

```python
import polars as pl
import tidyfinance as tf

crsp = tf.download_data(
    "WRDS",
    "crsp_monthly",
    start_date="1972-01-01",
    end_date="2024-12-31",
    add_ccm_links=True,
)
compustat = tf.download_data(
    "WRDS",
    "compustat_annual",
    start_date="1972-01-01",
    end_date="2024-12-31",
    additional_columns=["at"],
).select("gvkey", "date", "at")

signal = (
    tf.add_lagged_columns(compustat, cols="at", lag="1y", by="gvkey")
    .with_columns(asset_growth=pl.col("at") / pl.col("at_lag") - 1)
    .filter(pl.col("asset_growth").is_finite())
    .select("gvkey", "date", "asset_growth")
)
sorting_data = tf.join_lagged_values(
    crsp, signal, id_keys="gvkey", ff_adjustment=True
)

options = tf.portfolio_sort_options(
    breakpoint_options_main=tf.breakpoint_options(
        n_portfolios=5, breakpoints_exchanges="NYSE"
    )
)
portfolios = tf.implement_portfolio_sort(
    sorting_data,
    sorting_variables="asset_growth",
    sorting_method="univariate",
    rebalancing_month=7,
    portfolio_sort_options=options,
)
factor = tf.compute_long_short_returns(portfolios, direction="bottom_minus_top")
```

`factor` holds value-weighted, equal-weighted, and capped value-weighted long-short returns for each month. To control for size, exclude financial firms, or rebalance in staggered cohorts, a user changes one argument.

All computation runs in polars. `set_backend("pandas")` converts inputs and outputs at the package boundary, so pandas users receive pandas data frames without a second code path. Function names, arguments, defaults, and output columns match the R package. Tests check that the numbers match too: the Newey-West standard errors of `estimate_fama_macbeth()` agree with R's `sandwich` package [@zeileis.2004] to within $10^{-13}$. GitHub Actions runs the test suite on every pull request.

# Research impact statement

*Tidy Finance with Python* [@scheuch.2024] and its companion website, tidy-finance.org, use `tidyfinance` throughout, and instructors use both in undergraduate and graduate courses in finance and asset pricing. @frey.2026 describe the R version and use it to rebuild the asset growth factor from raw data and to show how construction choices change its premium. The package also gives access to the Tidy Finance factor library, a set of precomputed factor returns on Hugging Face, through `download_data("Tidy Finance", "factor_library")`. The package has been on PyPI since March 2025.

# AI usage disclosure

<!-- TODO: fill in. JOSS requires: tools/models used (with versions), where (code, tests, docs, paper text), nature and scope of assistance, and confirmation that the authors reviewed, edited, and validated all AI-assisted output and made the core design decisions. -->

We used generative AI tools ([TOOL, VERSION]) for [code refactoring / test scaffolding / documentation / copy-editing of this paper]. We reviewed, edited, and validated all AI-assisted output and made all design decisions.

# Acknowledgements

Stefan Voigt thanks the Danish Finance Institute for financial support. The Danish Finance Institute had no role in the design of the software or in this paper.

# References
