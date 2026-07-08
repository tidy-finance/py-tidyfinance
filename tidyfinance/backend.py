"""Global data frame backend for tidyfinance.

The backend controls the type of data frame returned by the public
tidyfinance API. The default is 'pandas'; after
'set_backend("polars")' the functions return 'polars.DataFrame'
objects instead. Polars data frames are also accepted as input
regardless of the active backend (they are converted to pandas
internally), so results from one call can be fed straight into the next.

Examples
--------
```python
import tidyfinance as tf
tf.set_backend("polars")
data = tf.download_data("Fama-French", "factors_ff_3_monthly")
tf.estimate_model(data, "mkt_excess ~ smb + hml")
tf.set_backend("pandas")  # back to the default
```
"""

import functools

_VALID_BACKENDS = ("pandas", "polars")

_BACKEND = "pandas"

# Calendar-date columns handled by the download functions. Internally
# pandas stores them as datetime64 (there is no plain date dtype), so
# they would surface as 'polars.Datetime' under the polars backend.
# '_convert_output' casts them to 'polars.Date' so they match the R
# package and can be joined or stacked against 'Date'-typed frames.
# Some names ('rdq', 'trd_rpt_dt', 'stlmnt_dt') are dropped before the
# downloads return and are listed defensively for raw frames that
# users pass through the analytics functions.
_DATE_COLUMNS = frozenset(
    {
        # all download functions
        "date",
        # WRDS CRSP
        "calculation_date",
        # WRDS Compustat
        "datadate",
        "rdq",
        # WRDS CCM links
        "linkdt",
        "linkenddt",
        # WRDS FISD
        "maturity",
        "offering_date",
        "dated_date",
        "last_interest_date",
        # WRDS Enhanced TRACE
        "trd_exctn_dt",
        "trd_rpt_dt",
        "stlmnt_dt",
    }
)


def set_backend(backend: str) -> None:
    """Set the global data frame backend for the tidyfinance API.

    Parameters
    ----------
    backend : str
        Either 'pandas' (the default) or 'polars'.

    Raises
    ------
    ValueError
        If 'backend' is not a recognized value.
    ImportError
        If 'backend' is 'polars' but the optional 'polars' package is
        not installed.

    Notes
    -----
    The polars backend wraps the public API at the package boundary:
    polars inputs are converted to pandas before each call, and pandas
    outputs are converted to polars on return. Chained calls therefore
    round-trip through pandas on every step, which adds a measurable
    cost on large panels. The pandas backend is a pass-through with
    zero conversion overhead.

    On conversion to polars, known calendar-date columns (e.g. 'date',
    'datadate', 'trd_exctn_dt') are cast from 'polars.Datetime' to
    'polars.Date', since pandas has no plain date dtype and would
    otherwise surface them as datetimes. Any time-of-day component in
    a column with one of these names is therefore dropped on output.
    Timezone-aware datetime columns are never cast.
    """
    global _BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Valid backends: "
            f"{', '.join(_VALID_BACKENDS)}."
        )
    if backend == "polars":
        try:
            import polars  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'polars' backend requires the optional 'polars' "
                "package. Install it via "
                "'pip install tidyfinance[polars]'."
            ) from e
    _BACKEND = backend


def get_backend() -> str:
    """Return the active data frame backend ('"pandas"' or '"polars"')."""
    return _BACKEND


def _is_polars_obj(obj) -> bool:
    """Return True for polars DataFrame/LazyFrame/Series without
    importing polars (so the check is cheap when polars is absent)."""
    module = type(obj).__module__ or ""
    return module.split(".")[0] == "polars" and type(obj).__name__ in (
        "DataFrame",
        "LazyFrame",
        "Series",
    )


def _to_pandas_input(obj):
    """Convert a polars input to pandas, leaving anything else as-is."""
    if _is_polars_obj(obj):
        if type(obj).__name__ == "LazyFrame":
            obj = obj.collect()
        return obj.to_pandas()
    return obj


def _convert_output(obj):
    """Convert a pandas data frame to the active backend.

    With the '"pandas"' backend (or for anything that is not a pandas
    data frame, e.g. a Series, dict, or ndarray) the object is returned
    unchanged. With the '"polars"' backend, a pandas data frame is
    converted via :func:'polars.from_pandas'. A non-default index (a
    named index or a non-'RangeIndex', such as a date index) is
    preserved as a column, since polars has no concept of an index.
    Known calendar-date columns ('_DATE_COLUMNS', e.g. 'date',
    'datadate', 'trd_exctn_dt') are cast from 'polars.Datetime' to
    'polars.Date' so they print as 'YYYY-MM-DD' and join or stack
    cleanly against 'Date'-typed frames. Other datetime columns pass
    through unchanged.
    """
    if get_backend() != "polars":
        return obj

    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        return obj

    import polars as pl

    include_index = not (
        isinstance(obj.index, pd.RangeIndex) and obj.index.name is None
    )
    out = pl.from_pandas(obj, include_index=include_index)
    date_casts = [
        pl.col(name).cast(pl.Date)
        for name in out.columns
        if name in _DATE_COLUMNS
        and out.schema[name] == pl.Datetime
        # never cast timezone-aware datetimes: casting would take the
        # UTC calendar date, which can differ from the wall-clock date
        and out.schema[name].time_zone is None
    ]
    if date_casts:
        out = out.with_columns(date_casts)
    return out


def _use_backend(func):
    """Wrap a public function so it honors the active backend.

    Polars data frames passed as arguments are converted to pandas
    before the call; a pandas data frame returned by the call is
    converted to the active backend afterwards. Non-data-frame
    arguments and return values pass through untouched. Apply this at
    the public API boundary only, so that internal calls between
    functions keep operating on pandas.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(_to_pandas_input(a) for a in args)
        kwargs = {k: _to_pandas_input(v) for k, v in kwargs.items()}
        return _convert_output(func(*args, **kwargs))

    return wrapper
