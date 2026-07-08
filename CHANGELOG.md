# Changelog

## Unreleased

- **Polars backend returns WRDS date columns as `Date` (#66):** With
  `set_backend("polars")`, the calendar-date columns of the WRDS
  downloads (e.g. `trd_exctn_dt` from `trace_enhanced`, `datadate` and
  `rdq` from Compustat, `linkdt`/`linkenddt` from CCM links, and the
  FISD date columns) are now cast to `polars.Date` instead of
  surfacing as `polars.Datetime`, matching the R package and the
  existing normalization of the `date` column. This fixes
  `SchemaError`s when joining or vertically stacking TRACE output
  against `Date`-typed frames. True time-of-day columns (e.g.
  `trd_exctn_tm`) are unaffected.
- **Added FRED-MD and FRED-QD macroeconomic databases:**
  `download_data("FRED", "FRED-MD")` and `download_data("FRED", "FRED-QD")`
  download the McCracken and Ng (2016, 2021) curated monthly / quarterly
  macro panels as wide tables (one column per series). `transform=True`
  applies each series' stationarity transform code (tcode). `vintage`
  selects the current release (default), a specific `"YYYY-MM"` release, or
  `"all"` — the full real-time panel across every archived vintage (recent
  vintages are hosted individually; older ones are read from the St. Louis
  Fed vintage archive zips), enabling leak-free point-in-time analysis.
- **Added Global Factor Data, Pastor-Stambaugh, and Stambaugh-Yuan
  downloads:** `download_data("Global Factor Data")` downloads
  characteristic-managed portfolio returns, the underlying long-short
  portfolios, industry returns, or reference cutoff files from
  [Global Factor Data](https://jkpfactors.com/data) (Jensen, Kelly, and
  Pedersen, 2023); the requested selection is validated against the
  library's live availability manifest, and
  `list_supported_jkp_factors()` lists the available regions and
  factors. `download_data("Pastor-Stambaugh")` downloads the liquidity
  factors of Pastor and Stambaugh (2003). `download_data("Stambaugh-Yuan")`
  downloads the mispricing factors of Stambaugh and Yuan (2017), with a
  `dataset` argument selecting `"monthly"` (default) or `"daily"` data.
- **OSAP download aligned with beginning-of-month and scaled returns:**
  `download_data("Open Source Asset Pricing")` now aligns the `date`
  column to the beginning of the month (the dataset previously returned
  end-of-month dates), matching the convention used by the other
  download functions. All predictor columns are monthly long-short
  returns expressed in percent and are now divided by 100 to return
  plain numeric (decimal) returns.
- **`sorting_variable` is now optional for `factor_library`:** Calling
  `download_data("Tidy Finance", "factor_library")` without a
  `sorting_variable` now returns the default portfolio construction for
  all sorting variables instead of raising an error. Other defaults
  (e.g. `rebalancing`, `weighting_scheme`) still apply; pass
  `fill_all=True` to leave every column unrestricted. Passing `None` for
  any filter column now removes that filter entirely and returns all
  values for that column (e.g. `min_size_quantile=None` includes all
  size groups), matching the R package's `NULL` behavior.
- **Added `detail` parameter to `estimate_fama_macbeth`:** matching the R
  package, passing `detail=True` returns a dict with `coefficients` (the
  usual risk premium estimates) and `summary_statistics` (a one-row data
  frame with the mean cross-sectional `r_squared`, `adj_r_squared`, and
  `n_obs` across all per-period regressions). The default (`detail=False`)
  behavior — returning only the coefficients data frame — is unchanged.
- **Dependencies (replaced pyfixest with formulaic):** The `pyfixest` dependency was dropped in favor of [`formulaic`](https://github.com/matthewwardrop/formulaic) plus a small internal numpy OLS helper (`_fit_ols`). `pyfixest` was used only for plain OLS with classical (IID) standard errors in `estimate_model` and the cross-sectional / IID-variance steps of `estimate_fama_macbeth`, but it pulled in `great-tables` → `multimark`, a `cffi` C-extension that ships no Python 3.14 wheels and therefore required a C/C++ toolchain (e.g. MSVC Build Tools on Windows) to install on unsupported interpreters. `_fit_ols` builds the design matrix via `formulaic` and reproduces `feols` coefficients, standard errors, t-statistics, and residuals to ~1e-9 for models without fixed effects, so results are unchanged. Note that `estimate_model` and `estimate_fama_macbeth` now perform classical OLS only (the fixed-effects / clustered-SE features of `pyfixest` were never used and are no longer available).

## v0.1.0

- Development version

## v0.1.1 (2025-03-21)

- Initial PyPI release.

## v0.1.2 (2025-07-03)

- Added new FRED url for data download
- added `curl_cffi` package to handle HTTP 429 Too Many Requests client error on Yahoo finance

## v0.2.2 (2025-10-19)

- removed `pandas_datareader` package requirements, replaced with `requests` from `curl_cffi` package
- bug fix fred download
- added "_" to internal data download function names

## v0.2.3 (2025-11-17)

- removed internal `__download_data_factors` function

## v0.2.4 (2025-11-17)

- added `get_available_famafrench_datasets` as a public function

## v0.2.5 (2025-11-20)

- fix date for monthly data in `_download_data_factors_ff`

## v0.2.6 (2026-04-02)

- Added support for Hugging Face datasets via `domain="tidyfinance"`, including `high_frequency_sp500` and `factor_library`

## v0.3.0 (2026-06-28)

- **Dependencies (removed statsmodels):** The `statsmodels` dependency was dropped; the regression-based functions now use [`pyfixest`](https://github.com/py-econometrics/pyfixest) instead. `estimate_model` and the cross-sectional / IID-variance steps of `estimate_fama_macbeth` call `pyfixest.feols`, and `estimate_betas` was rewritten to estimate rolling betas via closed-form OLS on cumulative cross-product sums (the design Gram matrix `X'X` and moment vector `X'y` are accumulated and rolled by cumulative-sum differencing, then solved once per window). This follows the [fast beta estimation](https://www.tidy-finance.org/blog/fast-beta-estimation/) approach, generalized to multiple regressors, and returns coefficients identical to ordinary least squares while avoiding a full refit per window (#49).
- **Docs (Great Docs):** Added a [Great Docs](https://opensource.posit.co/blog/2026-04-15_great-docs-introduction/) documentation site configured via `great-docs.yml`, including LLM-friendly artifacts (`llms.txt`, `llms-full.txt`). The API reference is generated from the numpydoc docstrings; build locally with `great-docs build` (on Windows set `PYTHONUTF8=1` to avoid a cp1252 decode error during post-processing). The generated `great-docs/` build directory is gitignored. (#29)
- **Breaking (Python version):** The minimum supported Python is now 3.11 (was 3.10), as required by the Great Docs toolchain.
- **Docs (R parity):** Fixed docstring discrepancies surfaced by the rendered reference, aligning the Python docs with r-tidyfinance: `breakpoint_options` (removed a duplicated `breakpoints_exchanges` entry and documented the previously undocumented `breakpoints_min_size_threshold`), `create_summary_statistics` (enumerated the reported statistics and detail quantiles), `compute_portfolio_returns` / `implement_portfolio_sort` (`min_portfolio_size` univariate/bivariate semantics and the "set to 0 to deactivate" behavior), `estimate_betas` (`lookback` annotated as `int` to match its use as an observation-count window), and `winsorize` (corrected the `x` type to `np.ndarray` and documented the `[0, 0.5]` range for `cut`).
- **Polars support:** the public API can now work with polars data frames via a global backend. Call `tidyfinance.set_backend("polars")` (default `"pandas"`; `get_backend()` reports the current setting). When set to `"polars"`, the data-bearing functions (`download_data`, the `estimate_*`/`compute_*` family, `add_lagged_columns`, `assign_portfolio`'s frame inputs, `list_supported_datasets`, etc.) return polars data frames, and all of them also accept polars input regardless of the active backend (converted to pandas internally). DataFrame outputs convert; Series/dict/ndarray returns (e.g. `assign_portfolio`) are left as-is, and date indices are preserved as columns. Internals remain pandas-based for now. Requires the optional `polars` dependency (`pip install tidyfinance[polars]`) (#42).
- **Breaking (WRDS credentials):** WRDS credentials are now read exclusively from environment variables (e.g. via a `.env` file). Support for `config.yaml` has been removed: `set_wrds_credentials()` now writes a `.env` file (with `WRDS_USER` and `WRDS_PASSWORD`), and `get_wrds_connection()` no longer accepts a `config_path` argument. The `pyyaml` dependency was dropped. Migrate any existing `config.yaml` credentials into a `.env` file or environment variables.
- **Breaking (CRSP):** the monthly CRSP price column returned by `download_data(domain="wrds", dataset="crsp_monthly")` is now named `prc` (was `altprc`), aligning with r-tidyfinance and both book editions. The value is unchanged — it is `mthprc` from the CRSP v2 monthly stock file; `altprc` was the legacy (v1) column name and was semantically stale for v2 downloads. Update any downstream code that referenced `altprc` (including the dependent `mktcap` computation).
- **Fix (Fama-MacBeth Newey-West):** `estimate_fama_macbeth` now matches R's `sandwich::NeweyWest` defaults, so the Python and R editions agree on Newey-West t-statistics. The previous implementation used statsmodels HAC with a fixed `maxlags=6` and no prewhitening (textbook Newey-West 1987); the new numpy implementation uses VAR(1) prewhitening plus the automatic Newey & West (1994) bandwidth, Bartlett kernel, recoloring, and no finite-sample adjustment (verified against `sandwich` 3.1.1 to ~1e-13). `vcov_options` now mirrors R's interface (`lag`, `prewhite`, `adjust`) and defaults to `None`; the legacy `maxlags` key is accepted as a deprecated alias for `lag` (preserving the old no-prewhitening behavior) and emits a `DeprecationWarning` (#35).
- **Fix (CRSP column order):** `download_data(domain="wrds", dataset="crsp_monthly")` now orders `listing_age` before `mktcap` to match r-tidyfinance's `download_data_wrds_crsp()` (`..., siccd, listing_age, mktcap, mktcap_lag, ...`). Values are unchanged; only the column order differed (#36).
- **Fix (TRACE regime cutoff):** `process_trace_data` now uses the correct Dick-Nielsen (2014) enhanced-TRACE regime cutoff of `2012-02-06` (was the transposed `2012-06-02`). Samples spanning Feb 6 – Jun 2, 2012 were previously cleaned under the wrong cancellation/correction/reversal regime, producing incorrect output; samples entirely after June 2012 were unaffected. This aligns the Python edition with r-tidyfinance's `download_data_wrds_trace_enhanced()` (#34).
- `download_data()` now uses the human-readable domain names returned by `list_supported_datasets()` (e.g., `"Fama-French"`, `"Global Q"`, `"WRDS"`, `"Tidy Finance"`). The `"pseudo"` and `"tidyfinance"` domains were renamed to `"Pseudo Data"` and `"Tidy Finance"`. The previous machine-readable domain names (e.g., `"famafrench"`, `"wrds"`, `"pseudo"`, `"tidyfinance"`) are soft-deprecated but still accepted.
- **Breaking (package API):** the dataset-specific `_download_data_*` helpers (e.g. `_download_data_wrds`, `_download_data_macro_predictors`, `_download_data_constituents`, `_download_data_factors_ff`, `_download_data_factors_q`, `_download_data_osap`, `_download_data_risk_free`, `_download_data_stock_prices`) are no longer re-exported from the package root. Public access continues via the dispatcher `download_data(domain, dataset, ...)`. If you need a helper directly, import it from its defining module (e.g. `from tidyfinance.data_download import _download_data_wrds`).
