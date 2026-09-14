"""
Regression with serial-correlation- and cluster-robust standard errors.

Why robust standard errors are not optional here
------------------------------------------------
The dependent variable is a daily session variance share. Volatility, and
therefore anything constructed from it, is strongly persistent: a high-volatility
week produces a run of correlated observations. Ordinary least squares standard
errors assume independent errors and would understate uncertainty substantially
in this setting, making a weak result look decisive.

Heteroskedasticity- and autocorrelation-consistent (Newey-West) standard errors
are therefore the default for the primary time-series test, with the lag length
chosen by the standard data-dependent rule rather than by hand, so that the
choice cannot be tuned to the desired answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_lags(nobs: int) -> int:
    """Automatic bandwidth: floor(4 * (T/100)^(2/9)).

    The conventional data-dependent rule (Newey and West 1994). Fixing the rule
    in advance removes the temptation to select a bandwidth that produces a
    favourable standard error.
    """
    if nobs < 2:
        raise ValueError("need at least two observations")
    return max(1, int(math.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0))))


@dataclass
class RegressionResult:
    """Container for a fitted regression with robust inference."""

    params: pd.Series
    std_err: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    conf_low: pd.Series
    conf_high: pd.Series
    nobs: int
    rsquared: float
    rsquared_adj: float
    cov_type: str
    maxlags: int | None
    n_clusters: int | None
    fitted: Any

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_err,
                "t": self.tstat,
                "p": self.pvalue,
                "ci_low": self.conf_low,
                "ci_high": self.conf_high,
            }
        )

    def coef(self, name: str) -> dict[str, float]:
        if name not in self.params.index:
            raise KeyError(f"{name!r} not among {list(self.params.index)}")
        return {
            "coef": float(self.params[name]),
            "std_err": float(self.std_err[name]),
            "t": float(self.tstat[name]),
            "p": float(self.pvalue[name]),
            "ci_low": float(self.conf_low[name]),
            "ci_high": float(self.conf_high[name]),
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        head = (
            f"OLS  N = {self.nobs:,}  R2 = {self.rsquared:.4f}  "
            f"cov = {self.cov_type}"
        )
        if self.maxlags is not None:
            head += f" (maxlags = {self.maxlags})"
        if self.n_clusters is not None:
            head += f" (clusters = {self.n_clusters})"
        return head + "\n" + self.table().to_string(float_format=lambda v: f"{v: .6g}")


def _prepare(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    add_const: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    if isinstance(X, pd.Series):
        X = X.to_frame()
    X = X.astype(float)
    y = y.astype(float)

    frame = pd.concat([y.rename("__y__"), X], axis=1)
    before = len(frame)
    frame = frame.dropna()
    dropped = before - len(frame)
    if dropped:
        # Listwise deletion is intentional and reported, never silent.
        frame.attrs["dropped_rows"] = dropped

    y_clean = frame["__y__"]
    X_clean = frame.drop(columns="__y__")
    if X_clean.empty:
        raise ValueError("no regressors remain after dropping missing rows")
    if len(y_clean) <= X_clean.shape[1] + int(add_const):
        raise ValueError(
            f"only {len(y_clean)} usable observations for "
            f"{X_clean.shape[1] + int(add_const)} parameters"
        )
    if add_const:
        X_clean = sm.add_constant(X_clean, has_constant="add")
    return y_clean, X_clean


def _package(
    fitted: Any,
    cov_type: str,
    maxlags: int | None,
    n_clusters: int | None,
) -> RegressionResult:
    ci = fitted.conf_int()
    return RegressionResult(
        params=fitted.params,
        std_err=fitted.bse,
        tstat=fitted.tvalues,
        pvalue=fitted.pvalues,
        conf_low=ci.iloc[:, 0],
        conf_high=ci.iloc[:, 1],
        nobs=int(fitted.nobs),
        rsquared=float(fitted.rsquared),
        rsquared_adj=float(fitted.rsquared_adj),
        cov_type=cov_type,
        maxlags=maxlags,
        n_clusters=n_clusters,
        fitted=fitted,
    )


def ols_hac(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    maxlags: int | None = None,
    add_const: bool = True,
) -> RegressionResult:
    """OLS with Newey-West standard errors.

    ``maxlags`` defaults to the automatic rule in :func:`newey_west_lags`.
    """
    y_clean, X_clean = _prepare(y, X, add_const)
    lags = newey_west_lags(len(y_clean)) if maxlags is None else int(maxlags)
    fitted = sm.OLS(y_clean, X_clean).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return _package(fitted, "HAC", lags, None)


def ols_cluster(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    groups: pd.Series,
    add_const: bool = True,
    warn_few_clusters: int = 20,
) -> RegressionResult:
    """OLS with one-way cluster-robust standard errors.

    Cluster-robust inference relies on a large number of clusters. With few
    clusters the standard errors are biased downward and the resulting p-values
    are unreliable; the number of clusters is therefore recorded on the result
    so that this can be stated plainly rather than glossed over. With fewer than
    roughly twenty clusters, prefer randomisation inference.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    frame = pd.concat([y.rename("__y__"), X, groups.rename("__g__")], axis=1).dropna()
    y_clean = frame["__y__"].astype(float)
    g = frame["__g__"]
    X_clean = frame.drop(columns=["__y__", "__g__"]).astype(float)
    if add_const:
        X_clean = sm.add_constant(X_clean, has_constant="add")

    n_clusters = int(g.nunique())
    fitted = sm.OLS(y_clean, X_clean).fit(
        cov_type="cluster", cov_kwds={"groups": g.to_numpy()}
    )
    result = _package(fitted, "cluster", None, n_clusters)
    if n_clusters < warn_few_clusters:
        result.fitted._few_clusters_warning = (
            f"only {n_clusters} clusters; cluster-robust p-values are unreliable, "
            "use randomisation inference instead"
        )
    return result


def add_month_fixed_effects(
    X: pd.DataFrame,
    dates: pd.DatetimeIndex,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Append calendar-month dummies.

    Month fixed effects absorb the slow-moving volatility regime, so that the
    dose-response coefficient is identified from within-month variation in ETF
    flow rather than from the difference between a calm year and a volatile one.
    """
    months = pd.Series(
        pd.DatetimeIndex(dates).to_period("M").astype(str), index=X.index, name="month"
    )
    dummies = pd.get_dummies(months, prefix="m", drop_first=drop_first, dtype=float)
    return pd.concat([X, dummies], axis=1)


def standardize(s: pd.Series) -> pd.Series:
    """Zero-mean, unit-variance transform for interpretable coefficient sizes."""
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"cannot standardise {s.name!r}: zero or undefined variance")
    return (s - s.mean()) / sd
