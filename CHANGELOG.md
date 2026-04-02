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

## v0.2.5 (2026-04-02)

- Added support for Hugging Face datasets via `domain="tidyfinance"`, including `high_frequency_sp500` and `factor_library`
