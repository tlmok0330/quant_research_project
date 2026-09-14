"""
Permutation and matched-control tests.

The holiday test, in detail
---------------------------
NYSE holidays are weekdays on which cryptocurrency trades exactly as usual but
ETF creation and redemption does not occur. If the ETF primary market is what
concentrates variance into the US window, then after the launch, holidays should
lack that concentration while neighbouring ordinary weekdays should show it.

A naive comparison of holidays to all other days would be confounded, because
holidays cluster in particular months and volatility varies over time. Each
holiday is therefore matched to nearby ordinary weekdays, and the quantity of
interest is the *difference* between the holiday's session share and its local
counterfactual:

    delta(h) = share_us(h) - mean(share_us of ordinary weekdays near h)

Before the launch there was no ETF primary market to be closed, so delta should
be approximately zero. After the launch, delta should be negative. The test
statistic is therefore a difference in differences,

    mean(delta | post-event) - mean(delta | pre-event)

with a directional prediction that it is negative.

Why permutation rather than a t-test
------------------------------------
There are only about nine holidays a year, so roughly eighteen post-event
observations. Normal-approximation inference is not credible at that sample
size. Permuting the pre/post labels across holidays produces an exact
finite-sample null distribution instead, and makes the low power visible rather
than hiding it behind an asymptotic p-value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd


@dataclass
class PermutationResult:
    statistic: float
    p_value: float
    n_permutations: int
    alternative: str
    null_mean: float
    null_sd: float
    null_dist: np.ndarray = field(repr=False)

    def quantile_of_observed(self) -> float:
        """Where the observed statistic sits in the null distribution."""
        return float(np.mean(self.null_dist <= self.statistic))

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"statistic = {self.statistic:+.5f}  p = {self.p_value:.4g} "
            f"({self.alternative}, {self.n_permutations:,} permutations; "
            f"null mean {self.null_mean:+.5f}, sd {self.null_sd:.5f})"
        )


def _p_from_null(
    observed: float,
    null: np.ndarray,
    alternative: Literal["two-sided", "less", "greater"],
) -> float:
    n = null.size
    if alternative == "greater":
        count = np.sum(null >= observed)
    elif alternative == "less":
        count = np.sum(null <= observed)
    else:
        count = np.sum(np.abs(null) >= abs(observed))
    return float((1.0 + count) / (1.0 + n))


def permutation_test(
    values: pd.Series | np.ndarray,
    group: pd.Series | np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 10_000,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    seed: int = 20260903,
) -> PermutationResult:
    """Permute a binary group label and recompute a statistic.

    Defaults to the difference in means, ``mean(group==1) - mean(group==0)``.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(group)
    mask = np.isfinite(v)
    v, g = v[mask], g[mask]

    g = g.astype(bool)
    if g.all() or (~g).all():
        raise ValueError("group must contain both True and False after dropping NaNs")

    if statistic is None:
        def statistic(vals: np.ndarray, grp: np.ndarray) -> float:  # noqa: F811
            return float(vals[grp.astype(bool)].mean() - vals[~grp.astype(bool)].mean())

    observed = statistic(v, g)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = statistic(v, rng.permutation(g))

    return PermutationResult(
        statistic=observed,
        p_value=_p_from_null(observed, null, alternative),
        n_permutations=n_permutations,
        alternative=alternative,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        null_dist=null,
    )


def stratified_permutation_test(
    values: pd.Series,
    group: pd.Series,
    strata: pd.Series,
    n_permutations: int = 10_000,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    seed: int = 20260903,
) -> PermutationResult:
    """Difference in means with labels permuted only *within* strata.

    Holidays are concentrated in particular calendar months, so permuting
    labels freely across the whole sample would let seasonality masquerade as an
    effect. Permuting within stratum removes that channel.
    """
    frame = pd.DataFrame(
        {"v": values.astype(float), "g": group.astype(bool), "s": strata}
    ).dropna()
    if frame["g"].all() or not frame["g"].any():
        raise ValueError("group must contain both True and False after dropping NaNs")

    v = frame["v"].to_numpy()
    g = frame["g"].to_numpy()
    s = frame["s"].to_numpy()

    def diff_means(vals: np.ndarray, grp: np.ndarray) -> float:
        return float(vals[grp].mean() - vals[~grp].mean())

    observed = diff_means(v, g)
    rng = np.random.default_rng(seed)

    stratum_positions = [np.flatnonzero(s == u) for u in pd.unique(s)]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        g_star = g.copy()
        for pos in stratum_positions:
            g_star[pos] = rng.permutation(g[pos])
        if g_star.all() or not g_star.any():
            null[i] = np.nan
            continue
        null[i] = diff_means(v, g_star)

    null = null[np.isfinite(null)]
    return PermutationResult(
        statistic=observed,
        p_value=_p_from_null(observed, null, alternative),
        n_permutations=null.size,
        alternative=alternative,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        null_dist=null,
    )


@dataclass
class MatchedControlResult:
    """Matched difference-in-differences on a closed-market day type."""

    deltas: pd.Series
    n_pre: int
    n_post: int
    mean_delta_pre: float
    mean_delta_post: float
    did_statistic: float
    permutation: PermutationResult
    unmatched_dates: list[pd.Timestamp]
    match_window_days: int
    treated_label: str

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.treated_label} matched DiD  "
            f"pre {self.mean_delta_pre:+.4f} (n={self.n_pre})  "
            f"post {self.mean_delta_post:+.4f} (n={self.n_post})  "
            f"diff {self.did_statistic:+.4f}  "
            f"p = {self.permutation.p_value:.4g}"
        )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantity": [
                    "mean delta, pre-event",
                    "mean delta, post-event",
                    "difference in differences",
                    "permutation p-value",
                    "n pre",
                    "n post",
                    "unmatched treated dates",
                ],
                "value": [
                    self.mean_delta_pre,
                    self.mean_delta_post,
                    self.did_statistic,
                    self.permutation.p_value,
                    self.n_pre,
                    self.n_post,
                    len(self.unmatched_dates),
                ],
            }
        )


def matched_deltas(
    daily: pd.DataFrame,
    value_col: str,
    treated_mask: pd.Series,
    control_mask: pd.Series,
    match_window_days: int = 5,
) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Difference between each treated day and nearby control days.

    Matching on adjacent dates controls for the local volatility regime, which
    a simple pooled comparison would not.
    """
    if value_col not in daily.columns:
        raise ValueError(f"{value_col!r} not in daily columns")
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("daily must be indexed by dates")

    values = daily[value_col].astype(float)
    treated_dates = daily.index[treated_mask.to_numpy(dtype=bool)]
    control_values = values[control_mask.to_numpy(dtype=bool)].dropna()

    deltas: dict[pd.Timestamp, float] = {}
    unmatched: list[pd.Timestamp] = []
    window = pd.Timedelta(days=match_window_days)

    for d in treated_dates:
        treated_value = values.get(d, np.nan)
        if not np.isfinite(treated_value):
            unmatched.append(d)
            continue
        nearby = control_values[
            (control_values.index >= d - window) & (control_values.index <= d + window)
        ]
        if nearby.empty:
            unmatched.append(d)
            continue
        deltas[d] = float(treated_value - nearby.mean())

    return pd.Series(deltas, name="delta").sort_index(), unmatched


def holiday_matched_test(
    daily: pd.DataFrame,
    event_date: str | pd.Timestamp,
    value_col: str = "share_us",
    match_window_days: int = 5,
    n_permutations: int = 10_000,
    alternative: Literal["two-sided", "less", "greater"] = "less",
    treated_day_types: tuple[str, ...] = ("holiday",),
    control_day_types: tuple[str, ...] = ("regular",),
    seed: int = 20260903,
) -> MatchedControlResult:
    """The holiday control test.

    Parameters
    ----------
    daily
        Frame indexed by Eastern date with ``value_col``, ``day_type`` and a
        boolean ``is_weekday`` or equivalent.
    event_date
        The ETF launch date.
    alternative
        Defaults to ``"less"``, encoding the directional prediction that the
        post-event holiday shortfall is negative. This one-sided choice is part
        of the pre-registered specification and must not be flipped after
        seeing the sign of the result.
    treated_day_types, control_day_types
        Day types forming the treated and control groups. Early-close days are
        excluded from both by default, since they are neither fully open nor
        closed.
    """
    if "day_type" not in daily.columns:
        raise ValueError("daily must contain a 'day_type' column")

    event = pd.Timestamp(event_date)

    is_weekday = daily.index.dayofweek < 5
    treated_mask = pd.Series(
        daily["day_type"].isin(treated_day_types).to_numpy() & is_weekday,
        index=daily.index,
    )
    control_mask = pd.Series(
        daily["day_type"].isin(control_day_types).to_numpy() & is_weekday,
        index=daily.index,
    )

    if treated_mask.sum() == 0:
        raise ValueError(f"no treated days of type {treated_day_types}")
    if control_mask.sum() == 0:
        raise ValueError(f"no control days of type {control_day_types}")

    deltas, unmatched = matched_deltas(
        daily, value_col, treated_mask, control_mask, match_window_days
    )
    if deltas.empty:
        raise ValueError("no treated day could be matched to a nearby control day")

    post = pd.Series(deltas.index >= event, index=deltas.index)
    n_pre, n_post = int((~post).sum()), int(post.sum())
    if min(n_pre, n_post) < 2:
        raise ValueError(
            f"need at least two matched treated days on each side of the event; "
            f"got {n_pre} before and {n_post} after"
        )

    mean_pre = float(deltas[~post].mean())
    mean_post = float(deltas[post].mean())
    did = mean_post - mean_pre

    perm = permutation_test(
        deltas,
        post,
        n_permutations=n_permutations,
        alternative=alternative,
        seed=seed,
    )

    return MatchedControlResult(
        deltas=deltas,
        n_pre=n_pre,
        n_post=n_post,
        mean_delta_pre=mean_pre,
        mean_delta_post=mean_post,
        did_statistic=did,
        permutation=perm,
        unmatched_dates=unmatched,
        match_window_days=match_window_days,
        treated_label="+".join(treated_day_types),
    )


def weekend_matched_test(
    daily: pd.DataFrame,
    event_date: str | pd.Timestamp,
    value_col: str = "share_us",
    match_window_days: int = 3,
    n_permutations: int = 10_000,
    alternative: Literal["two-sided", "less", "greater"] = "less",
    seed: int = 20260903,
) -> MatchedControlResult:
    """Weekend version of the matched control test.

    Weekends are permanently free of ETF primary-market activity and are far
    more numerous than holidays, so this test has much greater power. It is the
    weaker control conceptually, however, because weekends differ from weekdays
    in liquidity and participant mix as well as in ETF status.
    """
    event = pd.Timestamp(event_date)
    treated_mask = pd.Series(daily["day_type"].to_numpy() == "weekend", index=daily.index)
    control_mask = pd.Series(daily["day_type"].to_numpy() == "regular", index=daily.index)

    deltas, unmatched = matched_deltas(
        daily, value_col, treated_mask, control_mask, match_window_days
    )
    if deltas.empty:
        raise ValueError("no weekend could be matched to a nearby trading day")

    post = pd.Series(deltas.index >= event, index=deltas.index)
    n_pre, n_post = int((~post).sum()), int(post.sum())
    if min(n_pre, n_post) < 2:
        raise ValueError("need at least two matched weekends on each side of the event")

    mean_pre = float(deltas[~post].mean())
    mean_post = float(deltas[post].mean())

    perm = permutation_test(
        deltas, post, n_permutations=n_permutations, alternative=alternative, seed=seed
    )

    return MatchedControlResult(
        deltas=deltas,
        n_pre=n_pre,
        n_post=n_post,
        mean_delta_pre=mean_pre,
        mean_delta_post=mean_post,
        did_statistic=mean_post - mean_pre,
        permutation=perm,
        unmatched_dates=unmatched,
        match_window_days=match_window_days,
        treated_label="weekend",
    )
