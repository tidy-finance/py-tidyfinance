# Changelog

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

## Unreleased

- **Breaking (WRDS credentials):** WRDS credentials are now read exclusively from environment variables (e.g. via a `.env` file). Support for `config.yaml` has been removed: `set_wrds_credentials()` now writes a `.env` file (with `WRDS_USER` and `WRDS_PASSWORD`), and `get_wrds_connection()` no longer accepts a `config_path` argument. The `pyyaml` dependency was dropped. Migrate any existing `config.yaml` credentials into a `.env` file or environment variables.
- **Breaking (CRSP):** the monthly CRSP price column returned by `download_data(domain="wrds", dataset="crsp_monthly")` is now named `prc` (was `altprc`), aligning with r-tidyfinance and both book editions. The value is unchanged — it is `mthprc` from the CRSP v2 monthly stock file; `altprc` was the legacy (v1) column name and was semantically stale for v2 downloads. Update any downstream code that referenced `altprc` (including the dependent `mktcap` computation).
- **Fix (Fama-MacBeth Newey-West):** `estimate_fama_macbeth` now matches R's `sandwich::NeweyWest` defaults, so the Python and R editions agree on Newey-West t-statistics. The previous implementation used statsmodels HAC with a fixed `maxlags=6` and no prewhitening (textbook Newey-West 1987); the new numpy implementation uses VAR(1) prewhitening plus the automatic Newey & West (1994) bandwidth, Bartlett kernel, recoloring, and no finite-sample adjustment (verified against `sandwich` 3.1.1 to ~1e-13). `vcov_options` now mirrors R's interface (`lag`, `prewhite`, `adjust`) and defaults to `None`; the legacy `maxlags` key is accepted as a deprecated alias for `lag` (preserving the old no-prewhitening behavior) and emits a `DeprecationWarning` (#35).
- **Fix (CRSP column order):** `download_data(domain="wrds", dataset="crsp_monthly")` now orders `listing_age` before `mktcap` to match r-tidyfinance's `download_data_wrds_crsp()` (`..., siccd, listing_age, mktcap, mktcap_lag, ...`). Values are unchanged; only the column order differed (#36).
- **Fix (TRACE regime cutoff):** `process_trace_data` now uses the correct Dick-Nielsen (2014) enhanced-TRACE regime cutoff of `2012-02-06` (was the transposed `2012-06-02`). Samples spanning Feb 6 – Jun 2, 2012 were previously cleaned under the wrong cancellation/correction/reversal regime, producing incorrect output; samples entirely after June 2012 were unaffected. This aligns the Python edition with r-tidyfinance's `download_data_wrds_trace_enhanced()` (#34).
- `download_data()` now uses the human-readable domain names returned by `list_supported_datasets()` (e.g., `"Fama-French"`, `"Global Q"`, `"WRDS"`, `"Tidy Finance"`). The `"pseudo"` and `"tidyfinance"` domains were renamed to `"Pseudo Data"` and `"Tidy Finance"`. The previous machine-readable domain names (e.g., `"famafrench"`, `"wrds"`, `"pseudo"`, `"tidyfinance"`) are soft-deprecated but still accepted.
