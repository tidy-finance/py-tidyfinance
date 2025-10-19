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
