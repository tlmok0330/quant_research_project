"""
Structural-break tests.

Two questions are asked, and they are different
-----------------------------------------------
``chow_test`` asks whether the mean changed *at a date we specified*. It is
easy to run and easy to over-interpret, because we chose the date knowing what
happened in the market.

``sup_wald_test`` asks whether the data, given no information about where to
look, selects a break near the date our hypothesis predicts. This is the
stronger evidence: it cannot be produced by our having picked a date that
happens to sit at a volatility regime change.

Inference under serial correlation
----------------------------------
Session variance shares are persistent, so the classical Chow F distribution
and the standard normal are both badly wrong here. Two responses are used:
Newey-West standard errors for the known-date test, and a moving-block
bootstrap for the unknown-date test. The block bootstrap resamples contiguous
blocks of residuals, which preserves short-range dependence, so the simulated
null distribution reflects the persistence actually present in the data.

Because the null distribution is obtained by simulation, the search statistic
itself does not need a serial-correlation correction. That allows the scan over
candidate break dates to use a closed-form expression evaluated with cumulative
sums, making the bootstrap computationally feasible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.inference.regression import ols_hac


@dataclass
class ChowResult:
    """Mean-shift test at a pre-specified date."""

    break_date: pd.Timestamp
    n_pre: int
    n_post: int
    mean_pre: float
    mean_post: float
    diff: float
    diff_se_hac: float
    t_stat_hac: float
    p_value_hac: float
    maxlags: int
    f_stat_classical: float
    p_value_classical: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Break at {self.break_date.date()}: "
            f"{self.mean_pre:.4f} -> {self.mean_post:.4f} "
            f"(diff {self.diff:+.4f}, HAC t = {self.t_stat_hac:+.2f}, "
            f"p = {self.p_value_hac:.4g}; classical p = {self.p_value_classical:.4g})"
        )


@dataclass
class SupWaldResult:
    """Unknown-date break search with a bootstrap null distribution."""

    sup_wald: float
    argmax_date: pd.Timestamp
    hypothesised_date: pd.Timestamp | None
    days_from_hypothesis: int | None
    p_value_bootstrap: float
    n_bootstrap: int
    block_length: int
    rho1: float
    trim: float
    n_candidates: int
    scan: pd.Series = field(repr=False)
    bootstrap_dist: np.ndarray = field(repr=False)

    def __str__(self) -> str:  # pragma: no cover
        s = (
            f"sup-Wald = {self.sup_wald:.2f} at {self.argmax_date.date()}  "
            f"bootstrap p = {self.p_value_bootstrap:.4g} "
            f"({self.n_bootstrap:,} reps, block = {self.block_length}, "
            f"rho1 = {self.rho1:.3f})"
        )
        if self.days_from_hypothesis is not None:
            s += f"\nselected break is {self.days_from_hypothesis:+d} days from hypothesis"
        return s


def estimate_rho1(values: np.ndarray) -> float:
    """First-order autocorrelation of a demeaned series, clipped to [0, 0.99)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    d = v - v.mean()
    denom = float(np.sum(d**2))
    if denom <= 0:
        return 0.0
    rho = float(np.sum(d[1:] * d[:-1]) / denom)
    return float(np.clip(rho, 0.0, 0.99))


def auto_block_length(n: int, rho1: float) -> int:
    """Block length for the moving-block bootstrap, adapted to persistence.

    The conventional ``n**(1/3)`` rule assumes only weak dependence. Session
    variance shares are strongly persistent, and with a block shorter than the
    dependence length the bootstrap destroys exactly the autocorrelation it is
    supposed to preserve. The simulated null then becomes too narrow and the
    test rejects series that contain no break at all, manufacturing a structural
    break out of persistence.

    The block is therefore set to at least twice the integral timescale of an
    AR(1) process with the estimated first-order autocorrelation,

        tau = (1 + rho) / (1 - rho)

    while never falling below the conventional rule.

    When the series does contain a genuine break, the residuals from a
    constant-mean fit inherit it, which inflates the estimated ``rho`` and so
    lengthens the block. That makes the test more conservative in precisely the
    case where a false positive would be most costly.
    """
    conventional = max(2, int(round(n ** (1.0 / 3.0))))
    tau = (1.0 + rho1) / max(1e-6, 1.0 - rho1)
    persistence_aware = int(np.ceil(2.0 * tau))
    return max(2, min(max(conventional, persistence_aware), n // 3))


def chow_test(
    y: pd.Series,
    break_date: str | pd.Timestamp,
    maxlags: int | None = None,
) -> ChowResult:
    """Test for a shift in the mean of ``y`` at a pre-specified date.

    Reports both the classical F statistic, which assumes independent
    homoskedastic errors and is included only because it is conventional, and a
    Newey-West t statistic, which is the one that should be believed.
    """
    y = y.dropna().astype(float).sort_index()
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must be indexed by dates")

    bd = pd.Timestamp(break_date)
    post = (y.index >= bd).astype(float)
    n_pre = int((post == 0).sum())
    n_post = int((post == 1).sum())
    if min(n_pre, n_post) < 10:
        raise ValueError(
            f"insufficient observations either side of {bd.date()}: "
            f"{n_pre} before, {n_post} after"
        )

    X = pd.DataFrame({"post": post}, index=y.index)
    fit = ols_hac(y, X, maxlags=maxlags)
    c = fit.coef("post")

    pre_vals = y[post == 0].to_numpy()
    post_vals = y[post == 1].to_numpy()

    # Classical Chow F for an intercept-only model reduces to the equal-variance
    # two-sample comparison; retained purely for conventional reporting.
    ss_pre = float(((pre_vals - pre_vals.mean()) ** 2).sum())
    ss_post = float(((post_vals - post_vals.mean()) ** 2).sum())
    all_vals = y.to_numpy()
    ss_pooled = float(((all_vals - all_vals.mean()) ** 2).sum())
    n = len(y)
    num = (ss_pooled - ss_pre - ss_post) / 1.0
    den = (ss_pre + ss_post) / (n - 2)
    f_stat = num / den if den > 0 else float("nan")
    p_classical = float(stats.f.sf(f_stat, 1, n - 2)) if np.isfinite(f_stat) else float("nan")

    return ChowResult(
        break_date=bd,
        n_pre=n_pre,
        n_post=n_post,
        mean_pre=float(pre_vals.mean()),
        mean_post=float(post_vals.mean()),
        diff=c["coef"],
        diff_se_hac=c["std_err"],
        t_stat_hac=c["t"],
        p_value_hac=c["p"],
        maxlags=int(fit.maxlags or 0),
        f_stat_classical=f_stat,
        p_value_classical=p_classical,
    )


def _wald_scan(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Squared t statistics for a mean shift at every split in [lo, hi).

    Vectorised via cumulative sums so that the whole scan is O(n), which is what
    makes a bootstrap over the scan affordable.
    """
    n = values.size
    cs = np.concatenate(([0.0], np.cumsum(values)))
    cs2 = np.concatenate(([0.0], np.cumsum(values**2)))
    total, total2 = cs[-1], cs2[-1]

    k = np.arange(lo, hi, dtype=float)
    n1 = k
    n2 = n - k

    m1 = cs[lo:hi] / n1
    m2 = (total - cs[lo:hi]) / n2

    ss1 = cs2[lo:hi] - n1 * m1**2
    ss2 = (total2 - cs2[lo:hi]) - n2 * m2**2

    s2 = (ss1 + ss2) / (n - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(s2 * (1.0 / n1 + 1.0 / n2))
        t = (m2 - m1) / se
    return np.where(np.isfinite(t), t**2, 0.0)


def _moving_block_bootstrap(
    residuals: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample contiguous blocks with replacement to length len(residuals)."""
    n = residuals.size
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    out = np.concatenate([residuals[s : s + block_length] for s in starts])
    return out[:n]


def sup_wald_test(
    y: pd.Series,
    trim: float = 0.15,
    n_bootstrap: int = 1_000,
    block_length: int | None = None,
    hypothesised_date: str | pd.Timestamp | None = None,
    seed: int = 20260903,
) -> SupWaldResult:
    """Search for a mean break at an unknown date (Quandt-Andrews style).

    Parameters
    ----------
    y
        Date-indexed series, typically monthly or daily session variance shares.
    trim
        Fraction of the sample excluded at each end. Breaks cannot be identified
        near the boundaries, and the conventional choice is 0.15.
    n_bootstrap
        Moving-block bootstrap replications used to build the null distribution.
    block_length
        Block length for the bootstrap; defaults to the conventional n^(1/3).
    hypothesised_date
        Optional. If given, the distance from the selected break to this date is
        reported, which is the quantity of interest: does the data pick the date
        our mechanism predicts, without being told to?
    """
    y = y.dropna().astype(float).sort_index()
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must be indexed by dates")
    n = len(y)
    if n < 40:
        raise ValueError(f"need at least 40 observations to search for a break, got {n}")
    if not 0.0 < trim < 0.5:
        raise ValueError("trim must lie in (0, 0.5)")

    lo = max(2, int(np.floor(trim * n)))
    hi = min(n - 2, int(np.ceil((1 - trim) * n)))
    if hi <= lo:
        raise ValueError("trimming leaves no candidate break dates")

    values = y.to_numpy()
    scan = _wald_scan(values, lo, hi)
    sup = float(np.max(scan))
    arg = int(np.argmax(scan)) + lo
    argmax_date = y.index[arg]

    rho1 = estimate_rho1(values)
    if block_length is None:
        block_length = auto_block_length(n, rho1)
    block_length = max(2, min(int(block_length), n // 2))

    rng = np.random.default_rng(seed)
    resid = values - values.mean()
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        y_star = values.mean() + _moving_block_bootstrap(resid, block_length, rng)
        boot[b] = float(np.max(_wald_scan(y_star, lo, hi)))

    # Add-one correction keeps the p-value strictly positive and unbiased.
    p_value = float((1.0 + np.sum(boot >= sup)) / (1.0 + n_bootstrap))

    days_off = None
    hyp = None
    if hypothesised_date is not None:
        hyp = pd.Timestamp(hypothesised_date)
        days_off = int((argmax_date - hyp).days)

    return SupWaldResult(
        sup_wald=sup,
        argmax_date=argmax_date,
        hypothesised_date=hyp,
        days_from_hypothesis=days_off,
        p_value_bootstrap=p_value,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        rho1=rho1,
        trim=trim,
        n_candidates=hi - lo,
        scan=pd.Series(scan, index=y.index[lo:hi], name="wald"),
        bootstrap_dist=boot,
    )


def placebo_break_scan(
    y: pd.Series,
    true_date: str | pd.Timestamp,
    offsets_months: tuple[int, ...] = (-12, -6, -3, 3, 6, 12),
    maxlags: int | None = None,
) -> pd.DataFrame:
    """Run the known-date test at the real date and at placebo dates.

    A design that reports a significant break at arbitrary dates is detecting
    general non-stationarity rather than the event, so the placebo row values
    matter as much as the headline one.
    """
    rows = []
    true_ts = pd.Timestamp(true_date)
    candidates = [(0, true_ts)] + [
        (off, true_ts + pd.DateOffset(months=off)) for off in offsets_months
    ]
    for offset, date in candidates:
        try:
            res = chow_test(y, date, maxlags=maxlags)
        except ValueError as exc:
            rows.append(
                {
                    "offset_months": offset,
                    "date": date,
                    "diff": np.nan,
                    "t_hac": np.nan,
                    "p_hac": np.nan,
                    "note": str(exc),
                }
            )
            continue
        rows.append(
            {
                "offset_months": offset,
                "date": date,
                "is_hypothesised": offset == 0,
                "n_pre": res.n_pre,
                "n_post": res.n_post,
                "mean_pre": res.mean_pre,
                "mean_post": res.mean_post,
                "diff": res.diff,
                "t_hac": res.t_stat_hac,
                "p_hac": res.p_value_hac,
                "note": "",
            }
        )
    return pd.DataFrame(rows).set_index("offset_months")
