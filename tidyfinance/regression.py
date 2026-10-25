"""Estimation and regression functions for tidyfinance."""

import re
import warnings

import numpy as np
import polars as pl
from formulaic import model_matrix

from ._internal import _user_stacklevel


class _OLSFit:
    """Minimal OLS fit mirroring the parts of the pyfixest API used here.

    Exposes ``coef``, ``se``, ``tstat`` and ``resid``, reproducing
    ``pyfixest.feols(...)`` for linear models without fixed effects:
    classical (iid) inference with an ``n - k`` degrees-of-freedom
    correction (``sigma^2 * (X'X)^-1`` with ``sigma^2 = RSS / (n - k)``).
    ``coef``, ``se`` and ``tstat`` return plain dicts keyed by the model
    term names (insertion-ordered as in the design matrix).
    """

    def __init__(
        self,
        names,
        coef,
        se,
        tstat,
        resid,
        r_squared=None,
        adj_r_squared=None,
        n_obs=None,
    ):
        self._coef = dict(zip(names, (float(v) for v in coef)))
        self._se = dict(zip(names, (float(v) for v in se)))
        self._tstat = dict(zip(names, (float(v) for v in tstat)))
        self._resid = resid
        self._r_squared = r_squared
        self._adj_r_squared = adj_r_squared
        self._n_obs = n_obs

    def coef(self):
        return self._coef

    def se(self):
        return self._se

    def tstat(self):
        return self._tstat

    def resid(self):
        return self._resid

    def r_squared(self):
        return self._r_squared

    def adj_r_squared(self):
        return self._adj_r_squared

    def n_obs(self):
        return self._n_obs


def _fit_ols(model: str, data: pl.DataFrame) -> _OLSFit:
    """Fit an OLS model from a formula via formulaic and numpy.

    Replaces ``pyfixest.feols`` for the simple regressions used in this
    package. Supports the full formulaic grammar (additive terms,
    interactions, transformations, ``- 1`` to drop the intercept) and
    returns classical (iid) standard errors identical to ``feols`` for
    models without fixed effects.

    Parameters
    ----------
    model : str
        A formulaic formula string, e.g. ``'y ~ x1 + x2'``. An
        intercept (named ``'Intercept'``) is included unless ``- 1`` is
        present.
    data : pl.DataFrame
        Data containing the formula's variables.

    Returns
    -------
    _OLSFit
        Fitted model exposing ``coef``, ``se``, ``tstat`` and ``resid``.
    """
    y, x = model_matrix(model, data.to_pandas())
    names = list(x.columns)
    x_mat = np.asarray(x, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()

    n, k = x_mat.shape
    beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ beta

    dof = n - k
    if dof > 0:
        sigma2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
        se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    else:
        se = np.full(k, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)

    has_intercept = "Intercept" in names
    rss = float(resid @ resid)
    if has_intercept:
        tss = float(np.sum((y_vec - y_vec.mean()) ** 2))
    else:
        tss = float(np.sum(y_vec**2))
    r_squared = 1.0 - rss / tss if tss > 0 else np.nan
    if dof > 0 and not np.isnan(r_squared):
        denom = (n - 1) if has_intercept else n
        adj_r_squared = 1.0 - (1.0 - r_squared) * denom / dof
    else:
        adj_r_squared = np.nan

    return _OLSFit(names, beta, se, tstat, resid, r_squared, adj_r_squared, n)


# Rolling-window units accepted by 'estimate_betas'
# (months, days, hours, minutes, seconds). The keys are the polars
# duration abbreviations, so '60mo' reads like the '1mo' offsets used
# elsewhere in the package.
_LOOKBACK_UNITS = {
    "mo": "month",
    "d": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
}

_LOOKBACK_TRUNCATE = {
    "month": "1mo",
    "day": "1d",
    "hour": "1h",
    "minute": "1m",
    "second": "1s",
}

_SUBDAY_PERIODS = ("hour", "minute", "second")

_LOOKBACK_RE = re.compile(r"^(\d+)(mo|d|h|m|s)$")


def _parse_lookback(lookback) -> tuple[int, str | None]:
    """Resolve 'lookback' into a window length and a calendar unit.

    A string such as '60mo' selects a calendar window of 60 months.
    A bare integer selects the
    legacy positional window (a count of consecutive observations) and
    is deprecated.

    Returns
    -------
    tuple
        '(length, period)', where 'period' is one of 'month', 'day',
        'hour', 'minute', 'second', or 'None' for a positional window.
    """
    if isinstance(lookback, bool):
        raise ValueError("'lookback' must be a duration string or integer.")

    if isinstance(lookback, str):
        match = _LOOKBACK_RE.match(lookback)
        if match is None:
            raise ValueError(
                f"Invalid 'lookback' {lookback!r}. Use a positive count "
                "followed by one of 'mo', 'd', 'h', 'm', 's' (e.g. "
                "'60mo' for 60 months)."
            )
        length = int(match.group(1))
        if length <= 0:
            raise ValueError("'lookback' must be a positive duration.")
        return length, _LOOKBACK_UNITS[match.group(2)]

    if isinstance(lookback, int):
        if lookback <= 0:
            raise ValueError("'lookback' must be positive.")
        warnings.warn(
            "Passing 'lookback' as an integer counts consecutive "
            "observations, which differs from the calendar window used "
            "with duration strings and is deprecated. Pass a duration string such as "
            "'60mo' to roll over calendar periods.",
            DeprecationWarning,
            stacklevel=3,
        )
        return lookback, None

    raise ValueError(
        "'lookback' must be a duration string (e.g. '60mo') or an "
        f"integer; got {type(lookback).__name__}."
    )


def _period_index_expr(date_col: str, period: str) -> pl.Expr:
    """Integer counter that advances by one per calendar period.

    Months are indexed as
    'year * 12 + month' rather than by date, which sidesteps
    end-of-month arithmetic.
    """
    col = pl.col(date_col)
    if period == "month":
        return col.dt.year() * 12 + col.dt.month()
    if period == "day":
        return col.cast(pl.Date).cast(pl.Int64)
    seconds = col.dt.epoch("s")
    if period == "hour":
        return seconds // 3600
    if period == "minute":
        return seconds // 60
    return seconds


def _rolling_moment_betas(
    design: np.ndarray,
    y: np.ndarray,
    index: np.ndarray,
    lookback: int,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling closed-form OLS over a non-decreasing integer index.

    Observations that share an index value are pooled into a single
    window position by collapsing the cumulants to one row per entity
    and period before rolling. Because
    the Gram matrix 'X'X' and the moment vector 'X'y' are additive
    across observations, pooling does not change any estimate; it only
    determines where windows begin and end.

    For each distinct index value 'v', the window spans every
    observation whose index lies in '[v - lookback + 1, v]'. Passing
    'index = arange(n)' therefore reproduces a positional window of
    'lookback' consecutive observations.

    Parameters
    ----------
    design : np.ndarray
        '(n, k)' complete-case design matrix.
    y : np.ndarray
        '(n,)' response vector.
    index : np.ndarray
        '(n,)' non-decreasing integer window index.
    lookback : int
        Window length in index units.
    min_obs : int
        Minimum number of observations required in a window.

    Returns
    -------
    tuple
        '(start, betas)' where 'start' holds, for each distinct index
        value, the position of its first observation, and 'betas' is
        the matching '(m, k)' coefficient matrix. Windows with fewer
        than 'min_obs' observations, or with singular normal equations,
        contain NaN.
    """
    n, k = design.shape

    # Per-observation cross-products: the Gram matrix X'X is the sum of
    # the outer products of each design row, and X'y the sum of each row
    # scaled by y. Cumulative sums let any window be recovered by
    # differencing two prefix sums.
    gram_rows = design[:, :, None] * design[:, None, :]  # (n, k, k)
    moment_rows = design * y[:, None]  # (n, k)

    uniq, start = np.unique(index, return_index=True)
    m = uniq.size
    counts = np.diff(np.append(start, n))

    # Pool observations sharing an index value, then prefix-sum.
    gram_pooled = np.add.reduceat(gram_rows, start, axis=0)
    moment_pooled = np.add.reduceat(moment_rows, start, axis=0)

    gram_prefix = np.zeros((m + 1, k, k))
    gram_prefix[1:] = np.cumsum(gram_pooled, axis=0)
    moment_prefix = np.zeros((m + 1, k))
    moment_prefix[1:] = np.cumsum(moment_pooled, axis=0)
    count_prefix = np.zeros(m + 1, dtype=np.int64)
    count_prefix[1:] = np.cumsum(counts)

    # The window opens at the first index value not older than
    # 'v - lookback + 1'; this is what slider::slide_index_sum does.
    lo = np.searchsorted(uniq, uniq - lookback + 1, side="left")
    hi = np.arange(m) + 1
    n_window = count_prefix[hi] - count_prefix[lo]

    betas = np.full((m, k), np.nan)
    for j in np.flatnonzero(n_window >= min_obs):
        try:
            betas[j] = np.linalg.solve(
                gram_prefix[j + 1] - gram_prefix[lo[j]],
                moment_prefix[j + 1] - moment_prefix[lo[j]],
            )
        except np.linalg.LinAlgError:
            pass

    return start, betas


def estimate_betas(
    data: pl.DataFrame,
    model: str,
    lookback,
    min_obs: int = None,
    data_options: dict | None = None,
) -> pl.DataFrame:
    """Estimate rolling betas.

    Estimates rolling betas for a given model using the provided data.
    For each stock, the regression specified by 'model' is fit over a
    rolling calendar window of length 'lookback' (e.g. '"60mo"' for
    sixty months).

    The estimator avoids refitting a full regression for every window.
    Instead it accumulates the per-observation cross-products that
    define the normal equations (the design Gram matrix 'X'X' and the
    moment vector 'X'y'), takes their rolling sums via cumulative-sum
    differencing, and solves the resulting small linear system once per
    window. This closed-form approach follows the fast beta estimation
    described at
    https://www.tidy-finance.org/blog/fast-beta-estimation/ and is
    considerably faster than looping rolling regressions while
    returning the same coefficients.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing the data with a date identifier (defaults
        to 'date'), a stock identifier (defaults to 'permno'), and the
        other variables used in the model. No variable in 'model' may
        be called 'intercept'.
    model : str
        Formula describing the model to be estimated (e.g.,
        'ret_excess ~ mkt_excess + hml + smb'). An intercept is
        included unless the formula ends in '- 1' (or '+ 0').
    lookback : str or int
        Rolling window length. Pass a duration string — a positive
        count followed by one of '"mo"' (months), '"d"' (days),
        '"h"', '"m"', '"s"' — to roll over calendar periods, e.g.
        '"60mo"'. The window
        for a period 'v' spans every observation falling in the
        'lookback' periods ending at 'v', so gaps in a stock's history
        consume window space, and one row is returned per stock and
        period. Sub-day units require a datetime column.

        Passing a plain integer selects the legacy behaviour — a window
        of that many consecutive observations, one output row per input
        row — and emits a 'DeprecationWarning'.
    min_obs : int, optional
        Minimum number of observations required to estimate the model.
        Defaults to 'round(0.8 * lookback)'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'id' element
        specifies the stock identifier and the 'date' element the date
        column. Uses the 'data_options' defaults when None: 'id' ->
        'permno', 'date' -> 'date'.

    Returns
    -------
    pl.DataFrame
        Data frame with the estimated betas for each stock and time
        period. Contains the stock identifier and the date column,
        followed by one column per model term: an 'intercept' column
        (when the model includes one) and one 'beta_<variable>' column
        per regressor.

        With a calendar 'lookback' there is one row per stock and
        period, 'date' is floored to the start of the period, and
        windows with fewer than 'min_obs' observations are dropped
        from the output. With the deprecated
        integer 'lookback' there is one row per input row, 'date' is
        the observation's own date, and sub-'min_obs' windows yield
        null coefficients.

    Examples
    --------
    ```python
    from datetime import date
    import numpy as np
    import polars as pl
    from tidyfinance import estimate_betas
    rng = np.random.default_rng(1234)
    dates = pl.date_range(
        date(2020, 1, 1), date(2020, 12, 1), "1mo", eager=True
    )
    data_monthly = pl.DataFrame({
        'date': np.repeat(dates.to_numpy(), 50),
        'permno': np.tile(np.arange(1, 51), 12),
        'ret_excess': rng.normal(0, 0.1, 600),
        'mkt_excess': rng.normal(0, 0.1, 600),
        'smb': rng.normal(0, 0.1, 600),
        'hml': rng.normal(0, 0.1, 600),
    })
    estimate_betas(data_monthly, 'ret_excess ~ mkt_excess', lookback='3mo')
    ```
    """
    if data_options is None:
        data_options = {"id": "permno", "date": "date"}
    id_col = data_options.get("id", "permno")
    date_col = data_options.get("date", "date")

    length, period = _parse_lookback(lookback)

    if min_obs is None:
        # Round rather than truncate: truncation shifted the default
        # whenever 0.8 * lookback had a fractional part of .6 or .8
        # (e.g. lookback 6 gave 4 instead of 5).
        min_obs = int(round(length * 0.8))
    elif min_obs <= 0:
        raise ValueError("min_obs must be a positive integer.")

    dep_var, regressors, has_intercept = _parse_linear_formula(model)

    if "intercept" in [dep_var] + regressors:
        raise ValueError(
            "None of the columns in 'model' may be called 'intercept'. "
            "Please rename the column and try again."
        )

    if length < len(regressors):
        warnings.warn(
            "'lookback' is too low to estimate all model parameters. "
            "Consider increasing it.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    # Column names: a bare 'intercept' plus one
    # 'beta_<variable>' per regressor.
    coef_names = (["intercept"] if has_intercept else []) + [
        f"beta_{name}" for name in regressors
    ]

    if period in _SUBDAY_PERIODS and data.schema[date_col] == pl.Date:
        raise ValueError(
            f"A '{period}' lookback requires a datetime '{date_col}' "
            "column; "
            "got a date column."
        )

    model_vars = [dep_var] + regressors
    n_coefs = len(coef_names)

    results = []
    sorted_data = data.sort([id_col, date_col], maintain_order=True)
    for group in sorted_data.partition_by(id_col, maintain_order=True):
        values = group.select(
            [pl.col(c).cast(pl.Float64) for c in model_vars]
        ).to_numpy()
        complete = ~np.isnan(values).any(axis=1)
        pos = np.flatnonzero(complete)

        if pos.size == 0:
            if period is None:
                # Keep one (all-null) output row per input row.
                results.append(
                    _betas_frame(
                        group.get_column(id_col),
                        group.get_column(date_col),
                        np.full((group.height, n_coefs), np.nan),
                        coef_names,
                    )
                )
            continue

        y = values[pos, 0]
        x = values[pos, 1:]
        if has_intercept:
            design = np.column_stack([np.ones(pos.size), x])
        else:
            design = x.reshape(pos.size, -1)

        if period is None:
            index = np.arange(pos.size)
        else:
            index = (
                group.select(_period_index_expr(date_col, period).alias("_i"))
                .get_column("_i")
                .to_numpy()[pos]
            )

        start, betas = _rolling_moment_betas(design, y, index, length, min_obs)

        if period is None:
            # Realign to every input row; rows dropped for missing data
            # stay null.
            full = np.full((group.height, n_coefs), np.nan)
            full[pos[start]] = betas
            results.append(
                _betas_frame(
                    group.get_column(id_col),
                    group.get_column(date_col),
                    full,
                    coef_names,
                )
            )
        else:
            # One row per period, dated at the start of that period.
            period_dates = (
                group.get_column(date_col)
                .dt.truncate(_LOOKBACK_TRUNCATE[period])
                .gather(pos)
                .gather(start)
            )
            results.append(
                _betas_frame(
                    group.get_column(id_col).gather(pos).gather(start),
                    period_dates,
                    betas,
                    coef_names,
                )
            )

    if not results:
        return pl.DataFrame(
            schema={
                id_col: data.schema[id_col],
                date_col: data.schema[date_col],
                **{name: pl.Float64 for name in coef_names},
            }
        )

    out = pl.concat(results).select([id_col, date_col] + coef_names)
    if period is not None:
        # Windows with fewer than 'min_obs' observations are dropped
        # rather than returned as null rows.
        # The deprecated integer path keeps one row per input row for
        # backward compatibility.
        out = out.filter(
            ~pl.all_horizontal([pl.col(c).is_null() for c in coef_names])
        )
    return out


def _betas_frame(
    ids: pl.Series,
    dates: pl.Series,
    betas: np.ndarray,
    coef_names: list[str],
) -> pl.DataFrame:
    """Assemble one group's coefficients into a frame with nulls."""
    frame = pl.from_numpy(betas, schema=coef_names, orient="row")
    return frame.with_columns(pl.all().fill_nan(None)).with_columns(ids, dates)


def _parse_linear_formula(model: str) -> tuple[str, list[str], bool]:
    """Parse a simple additive regression formula.

    Splits a formula of the form 'y ~ x1 + x2 + ...' into the dependent
    variable, the list of regressor column names, and whether an
    intercept is included. An intercept is included unless the formula
    contains a '- 1' (or '+ 0') term, matching standard patsy/formulaic
    conventions. Only additive column terms are supported.

    Parameters
    ----------
    model : str
        Formula string, e.g. 'ret_excess ~ mkt_excess + smb - 1'.

    Returns
    -------
    tuple
        (dependent_variable, regressors, has_intercept).
    """
    if "~" not in model:
        raise ValueError("'model' must contain '~'.")
    lhs, rhs = model.split("~", 1)
    dep_var = lhs.strip()

    has_intercept = True
    tokens = re.split(r"[\s+]+", rhs.strip())
    regressors = []
    skip_next = False
    for tok in tokens:
        if not tok:
            continue
        if skip_next:
            skip_next = False
            continue
        if tok == "-":
            # The following token (expected to be '1') drops the
            # intercept.
            skip_next = True
            has_intercept = False
            continue
        if tok in ("1", "0"):
            if tok == "0":
                has_intercept = False
            continue
        regressors.append(tok)

    return dep_var, regressors, has_intercept


def _ar1_ols_residuals(e: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit an AR(1) by OLS without intercept or demeaning.

    Estimates rho in e_t = rho * e_{t-1} + u_t by ordinary least
    squares (no intercept, no demeaning). Used to prewhiten the
    estimating functions before forming a Newey-West long-run variance.

    Returns
    -------
    tuple
        (rho, residuals) where 'residuals' has length 'len(e) - 1'.
    """
    x = e[:-1]
    z = e[1:]
    rho = float((x @ z) / (x @ x))
    return rho, z - rho * x


def _newey_west_bandwidth(e: np.ndarray, prewhite: int) -> float:
    """Automatic Newey & West (1994) bandwidth for the Bartlett kernel.

    Computes the data-dependent truncation lag for a univariate,
    intercept-only Bartlett-kernel HAC estimator. If 'prewhite > 0',
    the series is first prewhitened by an AR(1) fit (no intercept).
    The bandwidth is the optimal one derived in Newey and West (1994).

    Parameters
    ----------
    e : np.ndarray
        The estimating-function series (typically the demeaned
        per-period coefficient).
    prewhite : int
        Order of the prewhitening AR fit. Pass 0 to disable.

    Returns
    -------
    float
        Recommended truncation lag.

    References
    ----------
    Newey, W. K., and West, K. D. (1994). Automatic lag selection in
    covariance matrix estimation. Review of Economic Studies, 61(4),
    631-653. https://doi.org/10.2307/2297912
    """
    n = e.shape[0]
    m = int(np.floor((3 if prewhite > 0 else 4) * (n / 100.0) ** (2.0 / 9.0)))
    if prewhite > 0:
        _, u = _ar1_ols_residuals(e)
        n = n - prewhite
    else:
        u = e
    m = min(m, n - 1)
    sigma = np.array([float(u[: n - j] @ u[j:]) / n for j in range(m + 1)])
    s0 = sigma[0] + 2.0 * sigma[1:].sum()
    s1 = 2.0 * np.sum(np.arange(1, m + 1) * sigma[1:])
    if s0 == 0.0:
        return 0.0
    rval = 1.1447 * ((s1 / s0) ** 2) ** (1.0 / 3.0)
    return rval * (n + prewhite) ** (1.0 / 3.0)


def _newey_west_se(
    series: np.ndarray,
    lag: int | None = None,
    prewhite: int = 1,
    adjust: bool = False,
) -> float:
    """Newey-West HAC standard error of the mean of a time series.

    Computes the Newey-West heteroskedasticity- and autocorrelation-
    consistent standard error of the sample mean of 'series'. The
    long-run variance is estimated with a Bartlett kernel; when
    'prewhite > 0', the series is first prewhitened by an AR(1) fit;
    when 'lag' is None, the truncation lag follows the automatic
    bandwidth selection of Newey and West (1994).

    Parameters
    ----------
    series : np.ndarray
        Time-ordered series (e.g. a factor's per-period risk premium).
    lag : int, optional
        Bartlett truncation lag. If None, the automatic Newey & West
        (1994) bandwidth is used.
    prewhite : int, default 1
        Order of the prewhitening AR fit. Pass 0 to disable.
    adjust : bool, default False
        Apply the 'n / (n - k)' finite-sample degrees-of-freedom
        correction.

    Returns
    -------
    float
        Newey-West HAC standard error of the sample mean. Returns NaN
        when 'series' has fewer than two non-NaN observations.

    References
    ----------
    Newey, W. K., and West, K. D. (1987). A simple, positive
    semi-definite, heteroskedasticity and autocorrelation consistent
    covariance matrix. Econometrica, 55(3), 703-708.
    https://doi.org/10.2307/1913610

    Newey, W. K., and West, K. D. (1994). Automatic lag selection in
    covariance matrix estimation. Review of Economic Studies, 61(4),
    631-653. https://doi.org/10.2307/2297912
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n_obs = y.shape[0]
    if n_obs < 2:
        return np.nan
    e = y - y.mean()
    if float(e @ e) == 0.0:
        return 0.0

    if lag is None:
        lag = int(np.floor(_newey_west_bandwidth(e, prewhite)))

    if prewhite > 0:
        rho, u = _ar1_ols_residuals(e)
        recolor = 1.0 / (1.0 - rho)
        n = n_obs - 1
    else:
        u = e
        recolor = 1.0
        n = n_obs

    weights = [1.0 - j / (lag + 1.0) for j in range(lag + 2)]
    utu = weights[0] * float(u @ u)
    for j in range(1, len(weights)):
        w = weights[j]
        if w == 0.0 or j >= n:
            continue
        utu += 2.0 * w * float(u[: n - j] @ u[j:])
    if adjust:
        utu *= n_obs / (n_obs - 1.0)
    if prewhite > 0:
        utu *= recolor * recolor
    variance = utu / (n_obs * n_obs)
    return float(np.sqrt(variance))


def estimate_fama_macbeth(
    data: pl.DataFrame,
    model: str,
    vcov: str = "newey-west",
    vcov_options: dict | None = None,
    data_options: dict | None = None,
    detail: bool = False,
) -> pl.DataFrame | dict:
    """Estimate Fama-MacBeth regressions.

    Runs one cross-sectional ordinary least squares regression per period
    of the date column, then averages the per-period coefficients to obtain
    risk premia and aggregates them into a single tidy frame.

    Parameters
    ----------
    data : pl.DataFrame
        Panel containing the dependent and independent variables named in
        'model' plus a column with the time index. Each (date, unit)
        combination should appear at most once.
    model : str
        Formula describing the cross-sectional regression
        (e.g., 'ret_excess ~ beta + bm + log_mktcap'). Standard
        formulaic syntax; an intercept is included unless the formula
        ends in '- 1'.
    vcov : {'iid', 'newey-west'}, default 'newey-west'
        Standard error treatment for the time-series average of period
        coefficients. 'iid' assumes independent and identically distributed
        errors across periods. 'newey-west' applies Newey-West
        heteroskedasticity- and autocorrelation-consistent standard errors
        with Bartlett kernel.
    vcov_options : dict, optional
        Tuning options for the Newey-West estimator. Recognized keys:

        - 'lag' : int, optional
            Bartlett truncation lag. If None (the default), the
            automatic bandwidth from Newey & West (1994) is used.
        - 'prewhite' : int, default 1
            Order of the VAR prewhitening filter applied before
            computing the long-run variance. Pass 0 to disable.
        - 'adjust' : bool, default False
            Apply a finite-sample degrees-of-freedom correction.
        - 'maxlags' : int, optional
            Deprecated alias for 'lag' (with 'prewhite' defaulting
            to 0). Emits a DeprecationWarning.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element is
        used to specify the date column of the cross-sections. Uses
        the 'data_options' default when None: 'date' -> 'date'.
    detail : bool, default False
        If 'False' (default), return only the coefficient estimates. If
        'True', return a dict with two keys: 'coefficients' (the usual
        estimates data frame) and 'summary_statistics' (a one-row data
        frame with the average cross-sectional R-squared, adjusted
        R-squared, and number of observations per cross-section).

    Returns
    -------
    pl.DataFrame or dict
        If 'detail' is 'False' (default), a data frame with one row per
        term in 'model', in model-term order (the intercept first, then
        the regressors as they appear in the formula), with columns:

        - 'factor' : term name ('intercept' or a regressor)
        - 'risk_premium' : time-series mean of cross-sectional coefficients
        - 'n' : number of periods used
        - 'standard_error' : SE of the time-series mean under 'vcov'
        - 't_statistic' : risk_premium / standard_error

        If 'detail' is 'True', a dict with two elements:

        - 'coefficients' : the same data frame described above
        - 'summary_statistics' : a one-row data frame with 'r_squared'
          (mean cross-sectional R-squared), 'adj_r_squared' (mean
          cross-sectional adjusted R-squared), and 'n_obs' (mean
          cross-sectional observation count)

    Raises
    ------
    ValueError
        If 'vcov' is not 'iid' or 'newey-west', if 'vcov_options'
        contains an unrecognized key, if the date column is missing
        from 'data', or if any date grouping has too few rows to estimate
        the cross-sectional coefficients (each grouping needs more
        rows than the number of variables in 'model').

    References
    ----------
    Fama, E. F., and MacBeth, J. D. (1973). Risk, return, and equilibrium:
    Empirical tests. Journal of Political Economy, 81(3), 607-636.
    https://doi.org/10.1086/260061

    Newey, W. K., and West, K. D. (1987). A simple, positive
    semi-definite, heteroskedasticity and autocorrelation consistent
    covariance matrix. Econometrica, 55(3), 703-708.
    https://doi.org/10.2307/1913610

    Newey, W. K., and West, K. D. (1994). Automatic lag selection in
    covariance matrix estimation. Review of Economic Studies, 61(4),
    631-653. https://doi.org/10.2307/2297912

    Examples
    --------
    ```python
    from datetime import date
    import numpy as np
    import polars as pl
    from tidyfinance import estimate_fama_macbeth
    rng = np.random.default_rng(1234)
    dates = pl.date_range(
        date(2020, 1, 1), date(2020, 12, 1), "1mo", eager=True
    )
    data = pl.DataFrame({
        'date': np.repeat(dates.to_numpy(), 50),
        'permno': np.tile(np.arange(1, 51), 12),
        'ret_excess': rng.normal(0, 0.1, 600),
        'beta': rng.normal(1, 0.2, 600),
        'bm': rng.normal(0.5, 0.1, 600),
        'log_mktcap': rng.normal(10, 1, 600),
    })
    result = estimate_fama_macbeth(data, 'ret_excess ~ beta+bm+log_mktcap')
    # Override the Newey-West settings
    result_iid = estimate_fama_macbeth(
        data,
        'ret_excess ~ beta + bm + log_mktcap',
        vcov='iid',
    )
    # Return detailed output including R-squared and observation counts
    result_detail = estimate_fama_macbeth(
        data,
        'ret_excess ~ beta + bm + log_mktcap',
        detail=True,
    )
    ```
    """
    date_col = (data_options or {}).get("date", "date")

    if vcov not in ["iid", "newey-west"]:
        raise ValueError("vcov must be either 'iid' or 'newey-west'.")

    if date_col not in data.columns:
        raise ValueError(f"The data must contain a {date_col} column.")

    # Parse the Newey-West options.
    options = dict(vcov_options or {})
    if "maxlags" in options:
        warnings.warn(
            "vcov_options key 'maxlags' is deprecated; use 'lag' (and "
            "'prewhite'). The default Newey-West estimator now uses "
            "VAR(1) prewhitening with automatic Newey-West (1994) "
            "bandwidth selection.",
            DeprecationWarning,
            stacklevel=2,
        )
        options.setdefault("lag", options.pop("maxlags"))
        options.setdefault("prewhite", 0)
    unrecognized = sorted(set(options) - {"lag", "prewhite", "adjust"})
    if unrecognized:
        raise ValueError(
            f"Unrecognized vcov_options key(s): {', '.join(unrecognized)}. "
            "Recognized keys: 'lag', 'prewhite', 'adjust'."
        )
    nw_lag = options.get("lag", None)
    nw_prewhite = int(options.get("prewhite", 1))
    nw_adjust = bool(options.get("adjust", False))

    # Run cross-sectional regressions in ascending date order. Every
    # date grouping must have more rows than the number of variables
    # in the model (dependent variable included).
    dep_var, regressors, _ = _parse_linear_formula(model)
    n_model_vars = len(dict.fromkeys([dep_var, *regressors]))
    cross_section_coefs = []
    cross_section_stats = []
    sorted_data = data.sort(date_col, maintain_order=True)
    for group in sorted_data.partition_by(date_col, maintain_order=True):
        if group.height <= n_model_vars:
            raise ValueError(
                "Each date grouping must have more rows than the number "
                "of predictors in the model to estimate coefficients. "
                "Please check your data."
            )

        model_fit = _fit_ols(model, data=group)
        cross_section_coefs.append(model_fit.coef())
        cross_section_stats.append(
            {
                "r_squared": model_fit.r_squared(),
                "adj_r_squared": model_fit.adj_r_squared(),
                "n_obs": model_fit.n_obs(),
            }
        )

    # Compute time-series averages, standard errors, t-statistics, and
    # period counts per factor under the chosen vcov. Factors are
    # reported in model-term order (intercept first, then the
    # regressors as they appear in the formula); 'coef()' is
    # insertion-ordered as in the design
    # matrix, so first-appearance order reproduces it.
    factors = list(
        dict.fromkeys(name for coefs in cross_section_coefs for name in coefs)
    )

    factor_col = []
    premium_col = []
    se_col = []
    t_col = []
    n_col = []
    for factor in factors:
        estimates = np.array(
            [coefs.get(factor, np.nan) for coefs in cross_section_coefs],
            dtype=float,
        )
        estimates = estimates[~np.isnan(estimates)]
        n = int(estimates.size)
        risk_premium = float(estimates.mean()) if n > 0 else np.nan
        if n < 2:
            se = np.nan
            t_stat = np.nan
        else:
            if vcov == "newey-west":
                se = _newey_west_se(
                    estimates,
                    lag=nw_lag,
                    prewhite=nw_prewhite,
                    adjust=nw_adjust,
                )
            else:
                se = _fit_ols(
                    "estimate ~ 1",
                    data=pl.DataFrame({"estimate": estimates}),
                ).se()["Intercept"]
            if se is None or np.isnan(se) or se == 0:
                t_stat = np.nan
            else:
                t_stat = float(estimates.mean()) / float(se)
        # The intercept row is labeled 'intercept'; formulaic names
        # the design-matrix column 'Intercept'.
        factor_col.append("intercept" if factor == "Intercept" else factor)
        premium_col.append(risk_premium)
        se_col.append(float(se) if se is not None else np.nan)
        t_col.append(t_stat)
        n_col.append(n)

    # Column order: factor, risk_premium, n,
    # standard_error, t_statistic.
    result_df = pl.DataFrame(
        {
            "factor": pl.Series(factor_col, dtype=pl.String),
            "risk_premium": pl.Series(premium_col, dtype=pl.Float64),
            "n": pl.Series(n_col, dtype=pl.Int64),
            "standard_error": pl.Series(se_col, dtype=pl.Float64),
            "t_statistic": pl.Series(t_col, dtype=pl.Float64),
        }
    ).with_columns(
        pl.col("risk_premium", "standard_error", "t_statistic").fill_nan(None)
    )

    if detail:
        stats_df = pl.DataFrame(cross_section_stats)
        summary_statistics = stats_df.select(
            pl.col("r_squared").fill_nan(None).mean(),
            pl.col("adj_r_squared").fill_nan(None).mean(),
            pl.col("n_obs").mean(),
        )
        return {
            "coefficients": result_df,
            "summary_statistics": summary_statistics,
        }

    return result_df


def estimate_model(
    data: pl.DataFrame,
    model: str,
    min_obs: int = 1,
    output="coefficients",
):
    """Estimate a linear model.

    Estimates a linear model specified by one or more independent
    variables. It checks for the presence of the specified independent
    variables in the dataset and whether the dataset has a sufficient
    number of observations. Depending on the 'output' parameter, it
    returns the model's coefficients, t-statistics, residuals, or any
    combination in a named dict.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame containing the dependent variable and one or more
        independent variables.
    model : str
        Formula string describing the model to be estimated (e.g.,
        'ret_excess ~ mkt_excess + hml + smb'). Use 'y ~ x - 1' for
        no-intercept models.
    min_obs : int, default 1
        Minimum number of observations required to estimate the model.
    output : str or list of str, default 'coefficients'
        What to return. Must contain one or more of 'coefficients',
        'residuals', and 'tstats'. If a single value is provided, the
        corresponding object is returned directly. If multiple values
        are provided, a dict is returned.

    Returns
    -------
    pl.DataFrame, np.ndarray, or dict
        If 'output' contains a single value: a data frame of
        coefficients or t-statistics, or a numeric vector of
        residuals. If 'output' contains multiple values: a dict with
        the requested elements. Coefficients and t-statistics are
        returned as one-row data frames with column names corresponding
        to the model terms (all-null when there are not enough
        observations). Residuals are returned as a numeric vector of
        length 'len(data)' with NaN for rows with missing data or
        insufficient observations.

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    from tidyfinance import estimate_model
    rng = np.random.default_rng(42)
    data = pl.DataFrame({
        'ret_excess': rng.standard_normal(100),
        'mkt_excess': rng.standard_normal(100),
        'smb': rng.standard_normal(100),
        'hml': rng.standard_normal(100),
    })
    # Estimate model with a single independent variable
    estimate_model(data, 'ret_excess ~ mkt_excess')
    # Estimate model with multiple independent variables
    estimate_model(data, 'ret_excess ~ mkt_excess + smb + hml')
    # Estimate model without intercept
    estimate_model(data, 'ret_excess ~ mkt_excess - 1')
    # Calculate residuals
    estimate_model(
        data, 'ret_excess ~ mkt_excess + smb + hml',
        output='residuals',
    )
    # Return t-statistics
    estimate_model(
        data, 'ret_excess ~ mkt_excess + smb + hml',
        output='tstats',
    )
    # Return coefficients, t-statistics, and residuals
    estimate_model(
        data, 'ret_excess ~ mkt_excess + smb + hml',
        output=['coefficients', 'tstats', 'residuals'],
    )
    ```
    """
    if isinstance(output, str):
        output_list = [output]
        return_multiple = False
    else:
        output_list = list(output)
        return_multiple = len(output_list) > 1

    valid_outputs = ("coefficients", "tstats", "residuals")
    invalid = [o for o in output_list if o not in valid_outputs]
    if invalid:
        raise ValueError(
            f"'output' must contain one or more of "
            f"{list(valid_outputs)}, not {invalid}."
        )

    if "~" not in model:
        raise ValueError("'model' must contain '~'.")
    parts = model.split("~", 1)
    dep_var = parts[0].strip()
    rhs = parts[1].strip()
    tokens = re.split(r"[\s+]+", rhs)
    independent_vars = [t for t in tokens if t and t not in ("-", "1")]

    if "intercept" in independent_vars:
        raise ValueError(
            "None of the columns in 'model' may be called 'intercept'. "
            "Please rename the column and try again."
        )

    missing_vars = [v for v in independent_vars if v not in data.columns]
    if missing_vars:
        raise ValueError(
            "The following independent variables are missing in the "
            f"data: {', '.join(missing_vars)}."
        )

    model_vars = [dep_var] + independent_vars
    mask_exprs = []
    for var in model_vars:
        expr = pl.col(var).is_not_null()
        if data.schema[var].is_float():
            expr = expr & pl.col(var).is_not_nan()
        mask_exprs.append(expr)
    complete = (
        data.select(pl.all_horizontal(mask_exprs).alias("complete"))
        .get_column("complete")
        .to_numpy()
    )
    n_complete = int(complete.sum())

    insufficient = (n_complete < min_obs) or (
        n_complete <= len(independent_vars)
    )

    fit = None
    if not insufficient:
        try:
            fit = _fit_ols(model, data=data.filter(pl.Series(complete)))
        except Exception:
            insufficient = True

    def to_df(values):
        renamed = {
            ("intercept" if name == "Intercept" else name): value
            for name, value in values.items()
        }
        frame = pl.DataFrame({name: [value] for name, value in renamed.items()})
        return frame.with_columns(pl.all().fill_nan(None))

    def na_df():
        if len(independent_vars) == 0:
            return np.nan
        return pl.DataFrame(
            {
                var: pl.Series([None], dtype=pl.Float64)
                for var in independent_vars
            }
        )

    result = {}

    if "coefficients" in output_list:
        if insufficient:
            result["coefficients"] = na_df()
        else:
            result["coefficients"] = to_df(fit.coef())

    if "tstats" in output_list:
        if insufficient:
            result["tstats"] = na_df()
        else:
            result["tstats"] = to_df(fit.tstat())

    if "residuals" in output_list:
        if insufficient:
            result["residuals"] = np.full(len(data), np.nan)
        else:
            resid = np.full(len(data), np.nan)
            resid[complete] = np.asarray(fit.resid())
            result["residuals"] = resid

    if return_multiple:
        return result
    return result[output_list[0]]
