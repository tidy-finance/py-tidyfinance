#!/usr/bin/env python
"""Verify the Python tidyfinance package against the R edition.

This script runs the same set of tidyfinance computations in both the Python
package (this repo) and the R package (``tidyfinance`` on CRAN) on *identical*
input data, then compares the numeric outputs.

To guarantee the two languages see exactly the same inputs (rather than two
independent RNG draws that merely share a seed), the data is generated once in
Python, written to CSV, and read back by both editions. R is invoked through
``Rscript``; no rpy2 is required.

Usage
-----
    python scripts/verify_r_parity.py
    python scripts/verify_r_parity.py --atol 1e-10 --rtol 1e-10
    python scripts/verify_r_parity.py --keep   # keep the temp work dir

Exit status is 0 if every check is within tolerance, 1 otherwise (so the
script is usable in CI).

Requirements
------------
- This package importable (``pip install -e .`` or run from the repo root).
- R with the ``tidyfinance`` package installed and ``Rscript`` on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Prefer this repo's source over any installed build of the package, so the
# script verifies the code in the working tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tidyfinance as tf  # noqa: E402

# --------------------------------------------------------------------------- #
# Shared deterministic input data
# --------------------------------------------------------------------------- #


def make_panel() -> pd.DataFrame:
    """A deterministic firm-month panel used by every check.

    48 months x 40 firms. ``ret_excess`` is a linear function of the
    characteristics plus noise so the Fama-MacBeth regression is well posed.
    ``mktcap_lag`` is the within-firm one-month lag of ``mktcap`` (NaN in the
    first month) and drives the value-weighted portfolio returns.
    """
    rng = np.random.default_rng(987654)
    dates = pd.date_range("2000-01-31", periods=48, freq="ME")
    exchanges = np.array(["NYSE", "NASDAQ", "AMEX"])
    recs = []
    for d in dates:
        beta = rng.normal(1, 0.3, size=40)
        bm = rng.normal(0.5, 0.2, size=40)
        size = rng.normal(10, 1, size=40)
        eps = rng.normal(0, 0.05, size=40)
        ret = 0.002 + 0.0015 * beta - 0.003 * bm + 0.0008 * size + eps
        mktcap = rng.uniform(1, 100, size=40)
        exch = exchanges[rng.integers(0, 3, size=40)]
        for p in range(40):
            recs.append(
                (d, p, ret[p], beta[p], bm[p], size[p], mktcap[p], exch[p])
            )
    panel = pd.DataFrame(
        recs,
        columns=[
            "date", "permno", "ret_excess", "beta", "bm", "size",
            "mktcap", "exchange",
        ],
    )
    # Derived (consumes no RNG, so the checks above keep their references).
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    panel["mktcap_lag"] = panel.groupby("permno")["mktcap"].shift(1)
    return panel


# --------------------------------------------------------------------------- #
# Check definitions
# --------------------------------------------------------------------------- #
#
# Each check produces a "tidy" frame with one or more *key* columns (string
# identifiers used to line up rows between the two editions) and one or more
# *value* columns (the numbers we compare). The Python side is a callable; the
# R side is a snippet that reads ``panel.csv`` and writes ``r_<name>.csv`` with
# the column names already normalised to match the Python output.


@dataclass
class Check:
    name: str
    description: str
    keys: list[str]
    values: list[str]
    py_fn: Callable[[pd.DataFrame], pd.DataFrame]
    r_code: str
    # Some checks (exact integer/identical algorithm) warrant a tighter bound.
    atol: float | None = None
    rtol: float | None = None


def _py_fama_macbeth(panel: pd.DataFrame) -> pd.DataFrame:
    out = tf.estimate_fama_macbeth(panel, "ret_excess ~ beta + bm + size")
    out = out.replace({"factor": {"Intercept": "intercept"}})
    return out[["factor", "risk_premium", "t_statistic"]]


def _py_breakpoints(panel: pd.DataFrame) -> pd.DataFrame:
    bp = tf.compute_breakpoints(
        panel, "mktcap", tf.breakpoint_options(n_portfolios=10)
    )
    return pd.DataFrame(
        {"idx": [str(i) for i in range(len(bp))], "breakpoint": np.asarray(bp)}
    )


def _py_assign_portfolio(panel: pd.DataFrame) -> pd.DataFrame:
    port = tf.assign_portfolio(
        panel, "mktcap", tf.breakpoint_options(n_portfolios=10)
    )
    return pd.DataFrame(
        {"row": [str(i) for i in range(len(port))],
         "portfolio": np.asarray(port, dtype=float)}
    )


def _py_summary(panel: pd.DataFrame) -> pd.DataFrame:
    out = tf.create_summary_statistics(panel, ["beta", "bm", "size", "mktcap"])
    return out.rename(
        columns={"count": "n", "std": "sd", "50%": "q50"}
    )[["variable", "n", "mean", "sd", "min", "q50", "max"]]


def _iso(dates) -> pd.Series:
    """Render a date column as ISO ``YYYY-MM-DD`` strings.

    Both editions then key on identical text (R's readr writes Date columns
    as ISO, and a Python ``Timestamp`` would otherwise stringify with a time
    component).
    """
    return pd.to_datetime(dates).dt.strftime("%Y-%m-%d")


# The portfolio-return checks share a single univariate value-weighted sort on
# ``beta`` over the rows with a defined market-cap lag.
_PF_VARS = "beta"
_PF_N = 5
_RET_COLS = ["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"]


def _py_portfolio_returns(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.dropna(subset=["mktcap_lag"])
    pr = tf.compute_portfolio_returns(
        p, _PF_VARS, "univariate",
        breakpoint_options_main=tf.breakpoint_options(n_portfolios=_PF_N),
        quiet=True,
    )
    pr = pr.copy()
    pr["portfolio"] = pr["portfolio"].astype(int)
    pr["date"] = _iso(pr["date"])
    return pr[["portfolio", "date", *_RET_COLS]]


def _py_implement_portfolio_sort(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.dropna(subset=["mktcap_lag"])
    pso = tf.portfolio_sort_options(
        filter_options=tf.filter_options(),
        breakpoint_options_main=tf.breakpoint_options(n_portfolios=_PF_N),
    )
    out = tf.implement_portfolio_sort(
        p, _PF_VARS, "univariate", pso, quiet=True
    )
    out = out.copy()
    out["portfolio"] = out["portfolio"].astype(int)
    out["date"] = _iso(out["date"])
    return out[["portfolio", "date", *_RET_COLS]]


def _py_long_short(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.dropna(subset=["mktcap_lag"])
    pr = tf.compute_portfolio_returns(
        p, _PF_VARS, "univariate",
        breakpoint_options_main=tf.breakpoint_options(n_portfolios=_PF_N),
        quiet=True,
    )
    ls = tf.compute_long_short_returns(pr)
    ls = ls.copy()
    ls["date"] = _iso(ls["date"])
    return ls[["date", *_RET_COLS]]


def _py_rolling_value(panel: pd.DataFrame) -> pd.DataFrame:
    sub = (
        panel[panel["permno"] == 0][["date", "ret_excess"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    rv = tf.compute_rolling_value(
        sub,
        f=lambda d: d["ret_excess"].mean(),
        period="month",
        periods=6,
        min_obs=3,
    )
    return pd.DataFrame({"date": _iso(sub["date"]), "rolling": np.asarray(rv)})


CHECKS: list[Check] = [
    Check(
        name="fama_macbeth",
        description="estimate_fama_macbeth (Newey-West) risk premia & t-stats",
        keys=["factor"],
        values=["risk_premium", "t_statistic"],
        py_fn=_py_fama_macbeth,
        r_code=r"""
out <- estimate_fama_macbeth(panel, "ret_excess ~ beta + bm + size")
out <- out[, c("factor", "risk_premium", "t_statistic")]
write_check("fama_macbeth", out)
""",
    ),
    Check(
        name="breakpoints",
        description="compute_breakpoints, 10 portfolios on mktcap",
        keys=["idx"],
        values=["breakpoint"],
        py_fn=_py_breakpoints,
        r_code=r"""
bp <- compute_breakpoints(
  panel, "mktcap", breakpoint_options(n_portfolios = 10)
)
out <- data.frame(idx = as.character(seq_along(bp) - 1L), breakpoint = bp)
write_check("breakpoints", out)
""",
    ),
    Check(
        name="assign_portfolio",
        description="assign_portfolio, 10 portfolios on mktcap (exact)",
        keys=["row"],
        values=["portfolio"],
        py_fn=_py_assign_portfolio,
        atol=0.0,
        rtol=0.0,
        r_code=r"""
port <- assign_portfolio(panel, "mktcap", breakpoint_options(n_portfolios = 10))
out <- data.frame(row = as.character(seq_along(port) - 1L),
                  portfolio = as.numeric(port))
write_check("assign_portfolio", out)
""",
    ),
    Check(
        name="summary_statistics",
        description="create_summary_statistics (n, mean, sd, min, q50, max)",
        keys=["variable"],
        values=["n", "mean", "sd", "min", "q50", "max"],
        py_fn=_py_summary,
        r_code=r"""
out <- create_summary_statistics(panel, beta, bm, size, mktcap)
write_check("summary_statistics", out)
""",
    ),
    Check(
        name="portfolio_returns",
        description="compute_portfolio_returns (univariate, ew/vw/capped)",
        keys=["portfolio", "date"],
        values=["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"],
        py_fn=_py_portfolio_returns,
        r_code=r"""
pf <- panel[!is.na(panel$mktcap_lag), ]
out <- compute_portfolio_returns(
  pf, "beta", "univariate",
  breakpoint_options_main = breakpoint_options(n_portfolios = 5),
  quiet = TRUE
)
write_check("portfolio_returns", out)
""",
    ),
    Check(
        name="implement_portfolio_sort",
        description="implement_portfolio_sort (filter + univariate sort)",
        keys=["portfolio", "date"],
        values=["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"],
        py_fn=_py_implement_portfolio_sort,
        r_code=r"""
pf <- panel[!is.na(panel$mktcap_lag), ]
pso <- portfolio_sort_options(
  filter_options = filter_options(),
  breakpoint_options_main = breakpoint_options(n_portfolios = 5)
)
out <- implement_portfolio_sort(
  pf, "beta", "univariate",
  portfolio_sort_options = pso, quiet = TRUE
)
write_check("implement_portfolio_sort", out)
""",
    ),
    Check(
        name="long_short_returns",
        description="compute_long_short_returns (top minus bottom)",
        keys=["date"],
        values=["ret_excess_vw", "ret_excess_ew", "ret_excess_vw_capped"],
        py_fn=_py_long_short,
        r_code=r"""
pf <- panel[!is.na(panel$mktcap_lag), ]
pr <- compute_portfolio_returns(
  pf, "beta", "univariate",
  breakpoint_options_main = breakpoint_options(n_portfolios = 5),
  quiet = TRUE
)
out <- compute_long_short_returns(pr)
write_check("long_short_returns", out)
""",
    ),
    Check(
        name="rolling_value",
        description="compute_rolling_value (6-month rolling mean)",
        keys=["date"],
        values=["rolling"],
        py_fn=_py_rolling_value,
        r_code=r"""
sub <- panel[panel$permno == 0, c("date", "ret_excess")]
sub <- sub[order(sub$date), ]
rv <- compute_rolling_value(
  sub, .f = function(d) mean(d$ret_excess),
  period = "month", periods = 6, min_obs = 3
)
out <- data.frame(date = sub$date, rolling = rv)
write_check("rolling_value", out)
""",
    ),
]


# --------------------------------------------------------------------------- #
# R driver
# --------------------------------------------------------------------------- #

R_HEADER = r"""
suppressMessages({
  library(tidyfinance)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
workdir <- args[[1]]

panel <- readr::read_csv(file.path(workdir, "panel.csv"),
                         show_col_types = FALSE)
panel$date <- as.Date(panel$date)

write_check <- function(name, df) {
  path <- file.path(workdir, paste0("r_", name, ".csv"))
  readr::write_csv(as.data.frame(df), path)
}
"""


def run_r(workdir: Path) -> None:
    if shutil.which("Rscript") is None:
        raise SystemExit(
            "Rscript not found on PATH. Install R and the 'tidyfinance' "
            "package to run the parity check."
        )
    r_script = R_HEADER + "\n".join(c.r_code for c in CHECKS) + "\n"
    script_path = workdir / "driver.R"
    script_path.write_text(r_script)
    proc = subprocess.run(
        ["Rscript", str(script_path), str(workdir)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"R driver failed (exit {proc.returncode}). See output above. "
            "Is the R 'tidyfinance' package installed?"
        )


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    name: str
    ok: bool
    max_abs: float
    max_rel: float
    detail: str = ""
    worst: list[str] = field(default_factory=list)


def compare(check: Check, py: pd.DataFrame, r: pd.DataFrame,
            atol: float, rtol: float) -> Result:
    a = atol if check.atol is None else check.atol
    rt = rtol if check.rtol is None else check.rtol

    # Align rows on the key columns. Coerce keys to string first so both
    # editions sort identically (e.g. an index column that R reads back from
    # CSV as int would otherwise sort numerically while Python sorts it as
    # text).
    py = py.copy()
    r = r.copy()
    for k in check.keys:
        py[k] = py[k].astype(str)
        r[k] = r[k].astype(str)
    py = py.sort_values(check.keys).reset_index(drop=True)
    r = r.sort_values(check.keys).reset_index(drop=True)

    if len(py) != len(r):
        return Result(check.name, False, np.inf, np.inf,
                      f"row count differs: python={len(py)} r={len(r)}")
    for k in check.keys:
        if not (py[k].astype(str).values == r[k].astype(str).values).all():
            return Result(check.name, False, np.inf, np.inf,
                          f"key column '{k}' does not match between editions")

    max_abs = 0.0
    max_rel = 0.0
    worst: list[str] = []
    ok = True
    for col in check.values:
        pv = py[col].to_numpy(dtype=float)
        rv = r[col].to_numpy(dtype=float)
        abs_diff = np.abs(pv - rv)
        rel_diff = abs_diff / np.maximum(np.abs(rv), 1e-300)
        col_max_abs = float(np.nanmax(abs_diff)) if len(abs_diff) else 0.0
        col_max_rel = float(np.nanmax(rel_diff)) if len(rel_diff) else 0.0
        max_abs = max(max_abs, col_max_abs)
        max_rel = max(max_rel, col_max_rel)
        within = np.allclose(pv, rv, atol=a, rtol=rt, equal_nan=True)
        if not within:
            ok = False
            i = int(np.nanargmax(abs_diff))
            key = ", ".join(f"{k}={py[k].iloc[i]}" for k in check.keys)
            worst.append(
                f"{col}[{key}]: py={pv[i]:.12g} r={rv[i]:.12g} "
                f"|d|={abs_diff[i]:.3g}"
            )
    return Result(check.name, ok, max_abs, max_rel, worst=worst)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="absolute tolerance (default 1e-8)")
    parser.add_argument("--rtol", type=float, default=1e-8,
                        help="relative tolerance (default 1e-8)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary work directory")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only the named checks")
    args = parser.parse_args()

    checks = CHECKS
    if args.only:
        checks = [c for c in CHECKS if c.name in set(args.only)]
        if not checks:
            raise SystemExit(f"no checks match {args.only}")

    workdir = Path(tempfile.mkdtemp(prefix="tf_parity_"))
    try:
        panel = make_panel()
        panel.to_csv(workdir / "panel.csv", index=False)

        # Python side first, so an R failure still shows what Python produced.
        py_outputs = {c.name: c.py_fn(panel) for c in checks}

        run_r(workdir)

        results: list[Result] = []
        for c in checks:
            r_path = workdir / f"r_{c.name}.csv"
            if not r_path.exists():
                results.append(Result(c.name, False, np.inf, np.inf,
                                      "R produced no output file"))
                continue
            r_df = pd.read_csv(r_path)
            results.append(
                compare(c, py_outputs[c.name], r_df, args.atol, args.rtol)
            )
    finally:
        if args.keep:
            print(f"\n(work dir kept at {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    # Report
    print(f"\nPython vs R tidyfinance parity  (atol={args.atol}, "
          f"rtol={args.rtol})")
    print("=" * 72)
    name_w = max(len(c.name) for c in checks)
    all_ok = True
    for res, c in zip(results, checks):
        status = "PASS" if res.ok else "FAIL"
        all_ok &= res.ok
        print(f"[{status}] {res.name:<{name_w}}  max|abs|={res.max_abs:.2e}  "
              f"max|rel|={res.max_rel:.2e}  {c.description}")
        if res.detail:
            print(f"         -> {res.detail}")
        for w in res.worst:
            print(f"         -> {w}")
    print("=" * 72)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
