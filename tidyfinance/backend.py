"""Global data frame backend for tidyfinance.

The package is implemented in polars. The backend controls the type of
data frame returned by the public tidyfinance API. The default is
'pandas' for backward compatibility; after 'set_backend("polars")' the
functions return the internal 'polars.DataFrame' objects directly with
zero conversion overhead. Pandas data frames are accepted as input
regardless of the active backend (they are converted to polars
internally), so results from one call can be fed straight into the
next.

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

import polars as pl

_VALID_BACKENDS = ("pandas", "polars")

_BACKEND = "pandas"

# Calendar-date columns handled by the download functions. Internally
# polars stores them as 'polars.Date'. Pandas inputs carry them as
# datetime64 (pandas has no plain date dtype), so '_to_polars_input'
# casts tz-naive datetime columns with these names to 'polars.Date' on
# entry so they match the internal representation.
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
        If 'backend' is 'pandas' but the 'pandas' package is not
        installed.

    Notes
    -----
    The package computes in polars internally. The pandas backend wraps
    the public API at the package boundary: pandas inputs are converted
    to polars before each call, and polars outputs are converted to
    pandas on return. The polars backend is a pass-through with zero
    conversion overhead.

    On conversion from pandas, known calendar-date columns (e.g.
    'date', 'datadate', 'trd_exctn_dt') are cast from
    'polars.Datetime' to 'polars.Date', since pandas has no plain date
    dtype. Any time-of-day component in a column with one of these
    names is therefore dropped on input. Timezone-aware datetime
    columns are never cast.
    """
    global _BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Valid backends: "
            f"{', '.join(_VALID_BACKENDS)}."
        )
    if backend == "pandas":
        try:
            import pandas  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'pandas' backend requires the 'pandas' package."
            ) from e
    _BACKEND = backend


def get_backend() -> str:
    """Return the active data frame backend ('"pandas"' or '"polars"')."""
    return _BACKEND


def _is_pandas_obj(obj) -> bool:
    """Return True for pandas DataFrame/Series without importing pandas
    (so the check is cheap when pandas is absent)."""
    module = type(obj).__module__ or ""
    return module.split(".")[0] == "pandas" and type(obj).__name__ in (
        "DataFrame",
        "Series",
    )


def _cast_date_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast known tz-naive datetime calendar-date columns to pl.Date."""
    date_casts = [
        pl.col(name).cast(pl.Date)
        for name in frame.columns
        if name in _DATE_COLUMNS
        and isinstance(frame.schema[name], pl.Datetime)
        # never cast timezone-aware datetimes: casting would take the
        # UTC calendar date, which can differ from the wall-clock date
        and frame.schema[name].time_zone is None
    ]
    if date_casts:
        frame = frame.with_columns(date_casts)
    return frame


def _to_polars_input(obj):
    """Convert a pandas input to polars, leaving anything else as-is.

    Pandas data frames are converted via 'polars.from_pandas'. A
    non-default index (a named index or a non-'RangeIndex', such as a
    date index) is preserved as a column, since polars has no concept
    of an index. Known calendar-date columns ('_DATE_COLUMNS') that
    arrive as tz-naive datetimes are cast to 'polars.Date' so they
    match the internal representation. Lazy frames are collected.
    """
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    if _is_pandas_obj(obj):
        if type(obj).__name__ == "Series":
            return pl.from_pandas(obj)
        import pandas as pd

        include_index = not (
            isinstance(obj.index, pd.RangeIndex) and obj.index.name is None
        )
        out = pl.from_pandas(obj, include_index=include_index)
        return _cast_date_columns(out)
    return obj


def _convert_output(obj, index=None):
    """Convert a polars object to the active backend.

    With the '"polars"' backend the object is returned unchanged. With
    the '"pandas"' backend, a 'polars.DataFrame' or 'polars.Series' is
    converted via '.to_pandas()'; tz-naive datetime-like columns are
    normalized to 'datetime64[ns]' so the output matches classic
    pandas conventions. Dict values are converted recursively (e.g.
    'estimate_model' with multiple outputs). Anything else (arrays,
    lists, scalars) passes through unchanged.

    When 'index' (the index of the pandas input) is given, a Series
    output of the same length receives it, so that row-aligned outputs
    such as 'assign_portfolio' align with the input frame under
    pandas-style assignment (e.g. 'df["pf"] = assign_portfolio(df, ...)'
    on a filtered frame with a non-default index).
    """
    if get_backend() != "pandas":
        return obj

    if isinstance(obj, dict):
        return {k: _convert_output(v, index=index) for k, v in obj.items()}

    if isinstance(obj, pl.DataFrame):
        out = obj.to_pandas()
        for name in out.columns:
            dtype = out[name].dtype
            if (
                str(dtype).startswith("datetime64")
                and getattr(dtype, "tz", None) is None
                and str(dtype) != "datetime64[ns]"
            ):
                out[name] = out[name].astype("datetime64[ns]")
        return out

    if isinstance(obj, pl.Series):
        out = obj.to_pandas()
        dtype = out.dtype
        if (
            str(dtype).startswith("datetime64")
            and getattr(dtype, "tz", None) is None
            and str(dtype) != "datetime64[ns]"
        ):
            out = out.astype("datetime64[ns]")
        if index is not None and len(index) == len(out):
            out.index = index
        return out

    return obj


def _use_backend(func):
    """Wrap a public function so it honors the active backend.

    Pandas data frames passed as arguments are converted to polars
    before the call; polars objects returned by the call are converted
    to the active backend afterwards. Non-data-frame arguments and
    return values pass through untouched. Apply this at the public API
    boundary only, so that internal calls between functions keep
    operating on polars.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Remember the index of the first pandas input so row-aligned
        # Series outputs can be re-aligned with it on conversion.
        index = None
        for value in (*args, *kwargs.values()):
            if _is_pandas_obj(value):
                index = value.index
                break
        args = tuple(_to_polars_input(a) for a in args)
        kwargs = {k: _to_polars_input(v) for k, v in kwargs.items()}
        return _convert_output(func(*args, **kwargs), index=index)

    return wrapper
