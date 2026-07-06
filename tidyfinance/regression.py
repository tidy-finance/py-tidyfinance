"""Estimation and regression functions for tidyfinance."""

import re
import warnings

import numpy as np
import pandas as pd
from formulaic import model_matrix


class _OLSFit:
    """Minimal OLS fit mirroring the parts of the pyfixest API used here.

    Exposes ``coef``, ``se``, ``tstat`` and ``resid``, reproducing
    ``pyfixest.feols(...)`` for linear models without fixed effects:
    classical (iid) inference with an ``n - k`` degrees-of-freedom
    correction (``sigma^2 * (X'X)^-1`` with ``sigma^2 = RSS / (n - k)``).
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
        self._coef = pd.Series(coef, index=names)
        self._se = pd.Series(se, index=names)
        self._tstat = pd.Series(tstat, index=names)
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


def _fit_ols(model: str, data: pd.DataFrame) -> _OLSFit:
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
    data : pd.DataFrame
        Data containing the formula's variables.

    Returns
    -------
    _OLSFit
        Fitted model exposing ``coef``, ``se``, ``tstat`` and ``resid``.
    """
    y, x = model_matrix(model, data)
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


def estimate_betas(
    data: pd.DataFrame,
    model: str,
    lookback: int,
    min_obs: int = None,
    id_col: str = "permno",
) -> pd.DataFrame:
    """Estimate rolling betas.

    Estimates rolling betas for a given model using the provided data.
    For each stock, the regression specified by 'model' is fit over a
    rolling window of 'lookback' consecutive observations.

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
    data : pd.DataFrame
        Data frame containing the data with a date identifier (defaults
        to 'date'), a stock identifier (defaults to 'permno'), and the
        other variables used in the model.
    model : str
        Formula describing the model to be estimated (e.g.,
        'ret_excess ~ mkt_excess + hml + smb'). An intercept is
        included unless the formula ends in '- 1' (or '+ 0').
    lookback : int
        Rolling window size in number of consecutive per-stock
        observations.
    min_obs : int, optional
        Minimum number of observations required to estimate the model.
        Defaults to 80% of 'lookback'.
    id_col : str, default 'permno'
        Column name representing the stock identifier.

    Returns
    -------
    pd.DataFrame
        Data frame with the estimated betas for each stock and time
        period. Contains one column per model term (the intercept, when
        present, is named 'Intercept'), the stock identifier, and the
        'date' column. Windows with fewer than 'min_obs' observations
        yield NaN coefficients.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import estimate_betas
    rng = np.random.default_rng(1234)
    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    data_monthly = pd.DataFrame({
        'date': np.repeat(dates, 50),
        'permno': np.tile(range(1, 51), 12),
        'ret_excess': rng.normal(0, 0.1, 600),
        'mkt_excess': rng.normal(0, 0.1, 600),
        'smb': rng.normal(0, 0.1, 600),
        'hml': rng.normal(0, 0.1, 600),
    })
    estimate_betas(data_monthly, 'ret_excess ~ mkt_excess', lookback=3)
    ```
    """
    if min_obs is None:
        min_obs = int(lookback * 0.8)
    elif min_obs <= 0:
        raise ValueError("min_obs must be a positive integer.")

    dep_var, regressors, has_intercept = _parse_linear_formula(model)

    coef_names = (["Intercept"] if has_intercept else []) + regressors

    results = []
    for stock_id, group in data.groupby(id_col):
        group = group.sort_values("date")

        betas = _rolling_ols_betas(
            group,
            dep_var,
            regressors,
            has_intercept,
            lookback,
            min_obs,
        )
        betas = pd.DataFrame(betas, columns=coef_names)
        betas[id_col] = stock_id
        betas["date"] = group["date"].values
        results.append(betas)

    betas_df = pd.concat(results, ignore_index=True)
    betas_df = betas_df[coef_names + [id_col, "date"]]
    return betas_df


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


def _rolling_ols_betas(
    group: pd.DataFrame,
    dep_var: str,
    regressors: list[str],
    has_intercept: bool,
    lookback: int,
    min_obs: int,
) -> np.ndarray:
    """Rolling OLS coefficients via cumulative cross-product sums.

    Computes, for every row 'i' of 'group' (assumed sorted in time),
    the OLS coefficients of 'dep_var' on 'regressors' over the window of
    up to 'lookback' consecutive rows ending at 'i'. Rows containing
    missing values in the model variables are dropped before windowing.

    Rather than refitting a regression per window, the routine forms the
    per-observation design Gram matrix 'X'X' and moment vector 'X'y',
    accumulates them with cumulative sums, differences those to obtain
    the windowed normal equations, and solves each small system. The
    coefficients are therefore identical to ordinary least squares.

    Parameters
    ----------
    group : pd.DataFrame
        Per-stock data sorted by date.
    dep_var : str
        Dependent variable column name.
    regressors : list of str
        Regressor column names.
    has_intercept : bool
        Whether to prepend an intercept column.
    lookback : int
        Rolling window length in observations.
    min_obs : int
        Minimum number of observations required in a window.

    Returns
    -------
    np.ndarray
        Array of shape '(len(group), k)' with the estimated
        coefficients aligned to the original rows of 'group', where 'k'
        counts the intercept (if any) plus the regressors. Rows whose
        window has fewer than 'min_obs' observations, or whose normal
        equations are singular, contain NaN. Rows dropped for missing
        data also contain NaN.
    """
    n_rows = len(group)
    k = (1 if has_intercept else 0) + len(regressors)
    betas = np.full((n_rows, k), np.nan)

    model_vars = [dep_var] + regressors
    complete = group[model_vars].notna().all(axis=1).to_numpy()
    pos = np.flatnonzero(complete)
    n = pos.size
    if n == 0:
        return betas

    y = group[dep_var].to_numpy(dtype=float)[pos]
    x = group[regressors].to_numpy(dtype=float)[pos]
    if has_intercept:
        design = np.column_stack([np.ones(n), x])
    else:
        design = x if x.ndim == 2 else x.reshape(n, 0)

    # Per-observation cross-products: the Gram matrix X'X is the sum of
    # the outer products of each design row, and X'y the sum of each row
    # scaled by y. Cumulative sums let any window be recovered by
    # differencing two prefix sums.
    gram_rows = design[:, :, None] * design[:, None, :]  # (n, k, k)
    moment_rows = design * y[:, None]  # (n, k)

    gram_prefix = np.zeros((n + 1, k, k))
    gram_prefix[1:] = np.cumsum(gram_rows, axis=0)
    moment_prefix = np.zeros((n + 1, k))
    moment_prefix[1:] = np.cumsum(moment_rows, axis=0)

    i = np.arange(n)
    lo = np.maximum(0, i - lookback + 1)
    count = i + 1 - lo

    gram_win = gram_prefix[i + 1] - gram_prefix[lo]  # (n, k, k)
    moment_win = moment_prefix[i + 1] - moment_prefix[lo]  # (n, k)

    for j in np.flatnonzero(count >= min_obs):
        try:
            betas[pos[j]] = np.linalg.solve(gram_win[j], moment_win[j])
        except np.linalg.LinAlgError:
            pass

    return betas


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
    data: pd.DataFrame,
    model: str,
    vcov: str = "newey-west",
    vcov_options: dict | None = None,
    date_col: str = "date",
    detail: bool = False,
) -> pd.DataFrame | dict:
    """Estimate Fama-MacBeth regressions.

    Runs one cross-sectional ordinary least squares regression per period
    of 'date_col', then averages the per-period coefficients to obtain
    risk premia and aggregates them into a single tidy frame.

    Parameters
    ----------
    data : pd.DataFrame
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
    date_col : str, default 'date'
        Column in 'data' identifying the time index for cross-sectional
        regressions.
    detail : bool, default False
        If 'False' (default), return only the coefficient estimates. If
        'True', return a dict with two keys: 'coefficients' (the usual
        estimates data frame) and 'summary_statistics' (a one-row data
        frame with the average cross-sectional R-squared, adjusted
        R-squared, and number of observations per cross-section).

    Returns
    -------
    pd.DataFrame or dict
        If 'detail' is 'False' (default), a data frame with one row per
        term in 'model' with columns:

        - 'factor' : term name (Intercept or regressor)
        - 'risk_premium' : time-series mean of cross-sectional coefficients
        - 'standard_error' : SE of the time-series mean under 'vcov'
        - 't_statistic' : risk_premium / standard_error
        - 'n' : number of periods used

        If 'detail' is 'True', a dict with two elements:

        - 'coefficients' : the same data frame described above
        - 'summary_statistics' : a one-row data frame with 'r_squared'
          (mean cross-sectional R-squared), 'adj_r_squared' (mean
          cross-sectional adjusted R-squared), and 'n_obs' (mean
          cross-sectional observation count)

    Raises
    ------
    ValueError
        If 'vcov' is not 'iid' or 'newey-west', or if 'date_col' is
        missing from 'data'.

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
    import numpy as np
    import pandas as pd
    from tidyfinance import estimate_fama_macbeth
    rng = np.random.default_rng(1234)
    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    data = pd.DataFrame({
        'date': np.repeat(dates, 50),
        'permno': np.tile(range(1, 51), 12),
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
    if vcov not in ["iid", "newey-west"]:
        raise ValueError("vcov must be either 'iid' or 'newey-west'.")

    if date_col not in data.columns:
        raise ValueError(f"The data must contain a {date_col} column.")

    # Parse Newey-West options (mirroring R's sandwich::NeweyWest interface).
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
    nw_lag = options.get("lag", None)
    nw_prewhite = int(options.get("prewhite", 1))
    nw_adjust = bool(options.get("adjust", False))

    # Run cross-sectional regressions
    cross_section_results = []
    cross_section_stats = []
    for date, group in data.groupby(date_col):
        if len(group) <= len(model.split("~")[1].split("+")):
            continue

        model_fit = _fit_ols(model, data=group)
        params = model_fit.coef().to_dict()
        params[date_col] = date
        cross_section_results.append(params)
        cross_section_stats.append(
            {
                "r_squared": model_fit.r_squared(),
                "adj_r_squared": model_fit.adj_r_squared(),
                "n_obs": model_fit.n_obs(),
            }
        )

    risk_premiums = pd.DataFrame(cross_section_results)

    # Compute time-series averages
    price_of_risk = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")["estimate"]
        .mean()
        .reset_index()
        .rename(columns={"estimate": "risk_premium"})
    )

    # Compute standard error, t-statistic, and n per factor under
    # the chosen vcov.
    def compute_se_and_t(x):
        x = x.sort_values(date_col)
        estimate = x["estimate"].dropna()
        n = int(estimate.size)
        if n < 2:
            return pd.Series(
                {
                    "standard_error": np.nan,
                    "t_statistic": np.nan,
                    "n": n,
                }
            )
        if vcov == "newey-west":
            se = _newey_west_se(
                estimate.to_numpy(),
                lag=nw_lag,
                prewhite=nw_prewhite,
                adjust=nw_adjust,
            )
        else:
            se = _fit_ols(
                "estimate ~ 1", data=x.dropna(subset=["estimate"])
            ).se()["Intercept"]
        if se is None or np.isnan(se) or se == 0:
            t_stat = np.nan
        else:
            t_stat = float(estimate.mean()) / float(se)
        return pd.Series(
            {
                "standard_error": float(se) if se is not None else np.nan,
                "t_statistic": t_stat,
                "n": n,
            }
        )

    price_of_risk_se_t_n = (
        risk_premiums.melt(
            id_vars=date_col, var_name="factor", value_name="estimate"
        )
        .groupby("factor")
        .apply(compute_se_and_t, include_groups=False)
        .reset_index()
    )

    result_df = price_of_risk.merge(price_of_risk_se_t_n, on="factor")[
        ["factor", "risk_premium", "standard_error", "t_statistic", "n"]
    ]

    if detail:
        stats_df = pd.DataFrame(cross_section_stats)
        summary_statistics = pd.DataFrame(
            [
                {
                    "r_squared": stats_df["r_squared"].mean(),
                    "adj_r_squared": stats_df["adj_r_squared"].mean(),
                    "n_obs": stats_df["n_obs"].mean(),
                }
            ]
        )
        return {
            "coefficients": result_df,
            "summary_statistics": summary_statistics,
        }

    return result_df


def estimate_model(
    data: pd.DataFrame,
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
    data : pd.DataFrame
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
    pd.DataFrame, np.ndarray, or dict
        If 'output' contains a single value: a data frame of
        coefficients or t-statistics, or a numeric vector of
        residuals. If 'output' contains multiple values: a dict with
        the requested elements. Coefficients and t-statistics are
        returned as data frames with column names corresponding to the
        model terms. Residuals are returned as a numeric vector of
        length 'len(data)' with NaN for rows with missing data or
        insufficient observations.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    from tidyfinance import estimate_model
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
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
    complete = data[model_vars].notna().all(axis=1)
    n_complete = int(complete.sum())

    insufficient = (n_complete < min_obs) or (
        n_complete <= len(independent_vars)
    )

    fit = None
    if not insufficient:
        try:
            fit = _fit_ols(model, data=data[complete])
        except Exception:
            insufficient = True

    def to_df(series):
        renamed = series.rename({"Intercept": "intercept"})
        return pd.DataFrame([renamed.values], columns=list(renamed.index))

    def na_df():
        if len(independent_vars) == 0:
            return np.nan
        return pd.DataFrame(
            [[np.nan] * len(independent_vars)],
            columns=independent_vars,
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
            resid[complete.values] = np.asarray(fit.resid())
            result["residuals"] = resid

    if return_multiple:
        return result
    return result[output_list[0]]
