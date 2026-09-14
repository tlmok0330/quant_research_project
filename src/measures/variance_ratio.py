"""
Lo-MacKinlay (1988) variance-ratio test with heteroskedasticity-robust inference.

Purpose in this study
---------------------
Variance shares tell us *how much* variance is realised in each session, but not
what kind. The variance ratio distinguishes the two cases that matter here:

    VR(q) close to 1   returns behave like a random walk, consistent with
                       information being incorporated as it arrives
    VR(q) below 1      negative autocorrelation, consistent with transitory
                       noise, bid-ask bounce, or overshoot and reversal
    VR(q) above 1      positive autocorrelation, consistent with a session
                       "catching up" to information generated elsewhere

The prediction under the ETF hypothesis is that the non-US session becomes more
catch-up-like after the launch, because a growing share of information now
enters during US hours and the rest of the day partially trends toward it.

Pooling across days
-------------------
A single US session contains only 78 five-minute bars, which is far too few for
a reliable variance ratio. Days are therefore pooled, but the q-period sums and
the autocovariance cross-products are accumulated only *within* contiguous
same-session blocks, so no statistic ever spans a session boundary. Pooling the
sufficient statistics rather than averaging per-day ratios preserves power and
gives a single well-defined test statistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarianceRatioResult:
    q: int
    vr: float
    z_stat: float
    p_value: float
    n_returns: int
    n_windows: int
    n_blocks: int
    theta: float

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"VR({self.q}) = {self.vr:.4f}  z = {self.z_stat:+.3f}  "
            f"p = {self.p_value:.4f}  (N = {self.n_returns:,}, blocks = {self.n_blocks:,})"
        )


def _as_blocks(returns: pd.Series | np.ndarray | list) -> list[np.ndarray]:
    if isinstance(returns, (pd.Series, np.ndarray)):
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        return [arr]
    blocks = []
    for b in returns:
        arr = np.asarray(b, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            blocks.append(arr)
    return blocks


def variance_ratio(
    returns: pd.Series | np.ndarray | list[np.ndarray],
    q: int,
) -> VarianceRatioResult:
    """Variance ratio at horizon ``q`` with a heteroskedasticity-robust z statistic.

    Parameters
    ----------
    returns
        Either a single contiguous return series, or a list of arrays treated as
        independent contiguous blocks. Statistics are never computed across a
        block boundary.
    q
        Aggregation horizon in bars. Must be at least 2.

    Notes
    -----
    Under the null of a random walk with possibly heteroskedastic increments,
    ``sqrt(N) * (VR(q) - 1)`` is asymptotically normal with variance ``theta(q)``,
    where ``theta`` is built from the fourth-moment terms ``delta_j``. Under
    conditional homoskedasticity ``theta(q)`` collapses to the familiar
    ``2(q-1)(2q-1) / (3q)``, which the test suite verifies.
    """
    if q < 2:
        raise ValueError("q must be at least 2")

    blocks = _as_blocks(returns)
    if not blocks:
        raise ValueError("no finite returns supplied")

    all_r = np.concatenate(blocks)
    n = all_r.size
    if n <= q + 1:
        raise ValueError(f"need more than q+1={q + 1} returns, got {n}")

    mu = all_r.mean()
    dev = all_r - mu
    s2_sum = float(np.sum(dev**2))
    if s2_sum <= 0:
        raise ValueError("returns have zero variance; variance ratio undefined")
    sigma_a2 = s2_sum / (n - 1)

    # q-period overlapping sums, accumulated within blocks only.
    sc_sum = 0.0
    n_windows = 0
    for b in blocks:
        if b.size < q:
            continue
        csum = np.concatenate(([0.0], np.cumsum(b)))
        agg = csum[q:] - csum[:-q]  # each is a sum of q consecutive returns
        z = agg - q * mu
        sc_sum += float(np.sum(z**2))
        n_windows += z.size
    if n_windows == 0:
        raise ValueError(f"no block is long enough to form a {q}-period sum")

    m = q * n_windows * (1.0 - q / n)
    if m <= 0:
        raise ValueError("degenerate normalisation; q is too large relative to N")
    sigma_c2 = sc_sum / m
    vr = sigma_c2 / sigma_a2

    # Heteroskedasticity-robust variance of the ratio.
    theta = 0.0
    for j in range(1, q):
        cross = 0.0
        for b in blocks:
            if b.size <= j:
                continue
            d = b - mu
            cross += float(np.sum((d[j:] ** 2) * (d[:-j] ** 2)))
        delta_j = n * cross / (s2_sum**2)
        theta += ((2.0 * (q - j)) / q) ** 2 * delta_j

    if theta <= 0:
        raise ValueError("non-positive robust variance estimate")

    z_stat = np.sqrt(n) * (vr - 1.0) / np.sqrt(theta)

    # Two-sided normal p-value without requiring scipy at import time.
    from scipy import stats as _stats

    p_value = float(2.0 * _stats.norm.sf(abs(z_stat)))

    return VarianceRatioResult(
        q=q,
        vr=float(vr),
        z_stat=float(z_stat),
        p_value=p_value,
        n_returns=int(n),
        n_windows=int(n_windows),
        n_blocks=len(blocks),
        theta=float(theta),
    )


def homoskedastic_theta(q: int) -> float:
    """Asymptotic variance of sqrt(N)(VR(q)-1) under conditional homoskedasticity."""
    return 2.0 * (q - 1) * (2 * q - 1) / (3.0 * q)


def build_blocks(
    returns: pd.Series,
    group_keys: pd.DataFrame,
) -> list[np.ndarray]:
    """Split a return series into contiguous blocks defined by grouping columns.

    Typically grouped by Eastern date and session flag, so that no statistic is
    computed across a session boundary.
    """
    frame = pd.DataFrame({"ret": returns.to_numpy()}, index=returns.index)
    for col in group_keys.columns:
        frame[col] = group_keys[col].to_numpy()

    blocks: list[np.ndarray] = []
    for _, g in frame.groupby(list(group_keys.columns), sort=True):
        arr = g["ret"].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            blocks.append(arr)
    return blocks


def variance_ratio_by_group(
    returns: pd.Series,
    labels: pd.DataFrame,
    q_values: tuple[int, ...] = (2, 4, 8),
    session_col: str = "in_us_window",
    date_col: str = "et_date",
    period_col: str | None = None,
) -> pd.DataFrame:
    """Variance ratios computed separately by session, and optionally by period.

    Parameters
    ----------
    returns
        Per-bar returns aligned to ``labels``.
    labels
        Frame containing at least ``session_col`` and ``date_col``.
    q_values
        Horizons to evaluate.
    period_col
        Optional column splitting the sample, for example a pre/post event flag.
    """
    rows = []
    outer_cols = [session_col] + ([period_col] if period_col else [])

    frame = labels.copy()
    frame["ret"] = returns.to_numpy()

    for keys, sub in frame.groupby(outer_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        blocks = build_blocks(
            pd.Series(sub["ret"].to_numpy(), index=sub.index),
            sub[[date_col]],
        )
        for q in q_values:
            try:
                res = variance_ratio(blocks, q)
            except ValueError:
                continue
            row = dict(zip(outer_cols, keys))
            row.update(
                {
                    "q": q,
                    "vr": res.vr,
                    "z_stat": res.z_stat,
                    "p_value": res.p_value,
                    "n_returns": res.n_returns,
                    "n_blocks": res.n_blocks,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)
