"""Download dispatcher for tidyfinance."""

import warnings

import pandas as pd

from .download_open_source import (_download_data_constituents,
                                   _download_data_factors_ff,
                                   _download_data_factors_q,
                                   _download_data_fred,
                                   _download_data_macro_predictors,
                                   _download_data_osap,
                                   _download_data_stock_prices)
from .download_pseudo import _simulate_pseudo_data
from .download_tidy_finance import (_download_data_huggingface,
                                    _download_data_risk_free)
from .download_wrds import _download_data_wrds
from .supported_datasets import (_check_supported_domain, _is_legacy_type,
                                 _parse_type_to_domain_dataset,
                                 _resolve_domain_alias)


def download_data(
    domain: str = None,
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    type: str = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download and process data based on domain and dataset.

    Downloads and processes data based on the specified domain (e.g.,
    Fama-French factors, Global Q factors, or macro predictors), dataset,
    and date range. The function checks whether the specified domain is
    supported and then delegates to the appropriate function for
    downloading and processing the data.

    Parameters
    ----------
    domain : str
        The domain of the dataset to download, given as one of the
        canonical names returned by 'list_supported_datasets()':
        'Fama-French', 'Global Q', 'Goyal-Welch', 'WRDS', 'Pseudo Data',
        'Index Constituents', 'FRED', 'Stock Prices',
        'Open Source Asset Pricing', 'Tidy Finance'. The previous
        short names (e.g. 'famafrench', 'wrds', 'pseudo') are still
        accepted but deprecated and will be removed in a future release.
    dataset : str, optional
        The specific dataset to download within the domain.
    start_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the start date for the data. If not provided, the full dataset
        or a subset is returned, depending on the dataset type.
    end_date : str, optional
        A character string or date in 'YYYY-MM-DD' format specifying
        the end date for the data. If not provided, the full dataset
        or a subset is returned, depending on the dataset type.
    type : str, optional
        Deprecated. Use 'domain' and 'dataset' instead. If provided, a
        DeprecationWarning is emitted and the legacy type is
        translated to a ('domain', 'dataset') pair via
        'list_supported_datasets'.
    **kwargs
        Additional arguments passed to specific download functions
        depending on 'domain'. For instance, if 'domain' is
        'Index Constituents', arguments are passed to
        '_download_data_constituents'. If 'domain' is 'Tidy Finance' and
        'dataset' is 'factor_library', arguments are either filter
        inputs (e.g., 'sorting_variable', 'rebalancing', 'fill_all') or
        an explicit 'ids' vector that bypasses the grid filter and
        downloads the specified portfolios directly via
        '_download_factor_library_ids'; see
        '_download_data_huggingface' for details.

    Returns
    -------
    pd.DataFrame
        A data frame with processed data, including dates and the
        relevant financial metrics, filtered by the specified date
        range.

    Examples
    --------
    ```python
    from tidyfinance import download_data
    download_data(
        'Fama-French',
        'Fama/French 5 Factors (2x3) [Daily]',
        '2000-01-01',
        '2020-12-31',
    )
    download_data(
        'Goyal-Welch', 'monthly', '2000-01-01', '2020-12-31'
    )
    download_data('Index Constituents', index='DAX')
    download_data('FRED', series=['GDP', 'CPIAUCNS'])
    download_data('Stock Prices', symbols=['AAPL', 'MSFT'])
    download_data(
        'Tidy Finance', 'risk_free', '2020-01-01', '2020-12-31'
    )
    download_data(
        'Tidy Finance',
        'high_frequency_sp500',
        '2007-07-26',
        '2007-07-27',
    )
    download_data(
        'Tidy Finance',
        'factor_library',
        sorting_variable='52w',
        rebalancing='annual',
    )
    download_data('Tidy Finance', 'factor_library', ids=[1, 2, 3])
    download_data('Tidy Finance', 'factor_library_grid')
    ```
    """
    if type is not None:
        warnings.warn(
            "'type' is deprecated; use 'domain' and 'dataset' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain, dataset = _parse_type_to_domain_dataset(type)

    if domain is not None and _is_legacy_type(domain):
        warnings.warn(
            "Passing a legacy 'type' string as 'domain' is deprecated; "
            "use 'domain' and 'dataset' instead. "
            "See list_supported_datasets() for the mapping.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain, dataset = _parse_type_to_domain_dataset(domain)

    if domain is None:
        raise ValueError("Argument 'domain' is required.")

    domain = _resolve_domain_alias(domain)

    _check_supported_domain(domain)

    if domain == "Fama-French":
        processed_data = _download_data_factors_ff(
            dataset=dataset, start_date=start_date, end_date=end_date
        )
    elif domain == "Global Q":
        processed_data = _download_data_factors_q(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Goyal-Welch":
        processed_data = _download_data_macro_predictors(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "WRDS":
        processed_data = _download_data_wrds(
            dataset=dataset, start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Index Constituents":
        processed_data = _download_data_constituents(dataset=dataset, **kwargs)
    elif domain == "FRED":
        processed_data = _download_data_fred(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Stock Prices":
        processed_data = _download_data_stock_prices(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Open Source Asset Pricing":
        processed_data = _download_data_osap(
            start_date=start_date, end_date=end_date, **kwargs
        )
    elif domain == "Tidy Finance":
        if dataset == "risk_free":
            processed_data = _download_data_risk_free(
                start_date=start_date, end_date=end_date, **kwargs
            )
        else:
            processed_data = _download_data_huggingface(
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
    elif domain == "Pseudo Data":
        processed_data = _simulate_pseudo_data(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    else:
        raise ValueError("Unsupported domain.")
    return processed_data
