# Changelog

## Unreleased

- **Fixed:** Goyal-Welch (macro predictor) CSV parse no longer fails when the S&P Index column uses thousands separators (e.g. `"1,049.34"`). Rows after Polars' default schema-inference window were previously dropped as an empty download.

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

## v0.4.0 (2026-07-27)

- **Polars backend returns WRDS date columns as `Date` (#66):** WRDS calendar-date columns (`datadate`, `trd_exctn_dt`, CCM link dates, FISD dates) are now cast to `polars.Date` instead of `polars.Datetime`, matching the R package.
- **Added FRED-MD and FRED-QD macroeconomic databases:** `download_data("FRED", "FRED-MD" / "FRED-QD")` download the McCracken and Ng (2016, 2021) macro panels. `transform=True` applies each series' transform code; `vintage` enables point-in-time analysis.
- **Added Global Factor Data, Pastor-Stambaugh, and Stambaugh-Yuan downloads:** `download_data("Global Factor Data")` downloads portfolios, industries, or cutoffs from Jensen, Kelly, and Pedersen (2023). `download_data("Pastor-Stambaugh")` and `download_data("Stambaugh-Yuan")` download the liquidity and mispricing factors.
- **OSAP download aligned with beginning-of-month and scaled returns:** `download_data("Open Source Asset Pricing")` now uses beginning-of-month dates (was end-of-month) and decimal returns (divided by 100).
- **`sorting_variable` is now optional for `factor_library`:** it returns the default construction for all sorting variables when omitted, and passing `None` for a filter column removes that filter.
- **Added `detail` parameter to `estimate_fama_macbeth`:** `detail=True` returns coefficients plus summary statistics (mean `r_squared`, `adj_r_squared`, `n_obs`). Default unchanged.
- **Dependencies (replaced pyfixest with formulaic):** dropped `pyfixest` for `formulaic`.

## v0.5.0 (2026-07-29)

- **Polars rewrite:** the entire package is now implemented in [polars](https://pola.rs/). All module-level functions take and return `polars.DataFrame` objects, calendar-date columns are typed as `polars.Date` (matching the R package and the book), and the `polars` backend (`set_backend("polars")`, as used throughout the Tidy Finance book) is a zero-overhead pass-through. The public API and the default `pandas` backend are unchanged: pandas users keep receiving pandas frames, with conversion now happening at the package boundary (pandas in → polars internals → pandas out). `polars` moved from an optional extra to a core dependency.
- **Missing values are nulls:** data frame outputs now represent missing values as polars nulls (surfacing as `NaN`/`NaT` after conversion under the pandas backend), instead of float `NaN` sentinels.
- **Breaking (`create_summary_statistics`):** output columns now follow the R package naming — `n`, `mean`, `sd`, `min`, `q50`, `max` (plus `q01`…`q99` with `detail=True`) — and grouped summaries return a tidy long table (one row per group × variable) instead of pandas MultiIndex columns.
- **Breaking (`assign_portfolio`):** returns a `polars.Series` under the polars backend (a `pandas.Series` under the default pandas backend, as before).
- **Lag arguments accept polars duration strings:** `add_lagged_columns` / `join_lagged_values` accept lags as polars offsets (e.g. `"1mo"`, `"1y2mo"`) in addition to ints (days), `datetime.timedelta`/`pd.Timedelta`, and calendar `pd.DateOffset` objects.
- **Internal SQL and HTTP readers now parse directly into polars** (`polars.read_database` for WRDS, `polars.read_csv`/`read_parquet` on fetched bytes elsewhere).
- **Dependencies trimmed:** dropped `requests` (unused since v0.2.2 — all HTTP goes through `curl_cffi`); replaced the `dotenv` wrapper package with its actual implementation `python-dotenv`; moved `lxml` to the new optional `scraping` extra (it only powers `get_available_famafrench_datasets`, which now raises a clear ImportError pointing at `pip install tidyfinance[scraping]` when lxml is absent). The `polars` extra is kept as a no-op for backward compatibility.
- **Breaking (`estimate_betas`), aligned with r-tidyfinance:** coefficient columns are now named `intercept` and `beta_<variable>` (was `Intercept` and the bare variable name), and the identifier and date columns come first, so the output is `<id>, date, intercept, beta_<variable>, ...`.
- **Deprecated (`estimate_betas` `lookback`), aligned with r-tidyfinance:** `lookback` now accepts a duration string — `"60mo"`, `"30d"`, and the sub-day units `"h"`, `"m"`, `"s"` — which rolls over **calendar periods** exactly as R's `months(60)` does. Observations falling in the same period are pooled, so the result has one row per identifier and period with `date` floored to the period start; for daily data a `"3mo"` window therefore yields a monthly beta series fitted on every observation in the trailing three months. Passing a plain integer keeps the previous positional window (a count of consecutive observations, one output row per input row) and now emits a `DeprecationWarning`. The two agree on a gap-free panel with one observation per period, so monthly examples are unaffected; they differ when a panel has holes.
- **Fix (`estimate_betas` default `min_obs`):** the default is now `round(0.8 * lookback)` as in r-tidyfinance, rather than truncating. For `lookback` 6 this gives 5 instead of 4; `lookback` 60 is unchanged at 48.
- **Breaking (`estimate_fama_macbeth`), aligned with r-tidyfinance:** the column order is now `factor, risk_premium, n, standard_error, t_statistic` (`n` moved from last to third), the intercept row is labelled `intercept` (was `Intercept`, and now matches `estimate_model`), and rows are returned in model-term order — the intercept first, then the regressors as they appear in the formula — instead of alphabetically.
- **Breaking (`estimate_betas`):** duration lookbacks drop windows below `min_obs` instead of returning null rows.
- **Breaking (`estimate_fama_macbeth`):** too-small date groups and unknown `vcov_options` keys raise; added `data_options`.
- **Breaking (`compute_rolling_value`):** callbacks receive the active backend's frame type.
- **Breaking (TRACE):** pre-2012 messages restricted to `trc_st = 'T'`.
- **Breaking (FF breakpoints):** column names are strings (`"0-5"`), not tuples.
- **Fixed:** `pd.DateOffset(n)` day lags; Series index alignment (pandas backend); CRSP v1 ccm-link join; one-sided `risk_free` date ranges; `NaN` as missing in `filter_sorting_data`; WRDS `numeric` columns cast to float.
