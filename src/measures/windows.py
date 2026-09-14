"""
Named intraday windows, and the within-day placebo that separates the ETF
primary-market channel from the CME futures channel.

Why this module exists
----------------------
The holiday and weekend control tests establish that a mechanism is synchronised
to the NYSE calendar. They do not, on their own, establish that the mechanism is
ETF creation and redemption, because **CME Bitcoin futures observe essentially
the same holiday calendar**. A confound operating through CME rather than through
authorised-participant hedging would pass both closed-market tests exactly as the
ETF story does. That is the most serious residual threat to identification in
this study.

The two channels are separable within the day rather than across days. CME's
Bitcoin futures session runs from 18:00 ET the previous evening through 17:00 ET,
with a one-hour daily halt, so it is open for roughly 23 hours. ETF creation and
redemption is confined to 09:30-16:00 ET. There are therefore several hours every
weekday during which CME is open and the ETF primary market is shut, and those
hours provide a placebo window:

- If the post-event rise in variance share is confined to 09:30-16:00, the ETF
  channel is supported.
- If an equal rise appears in the CME-open, ETF-closed evening window, the effect
  is not specific to the ETF, and the honest description is a shift toward
  US-linked institutional hours generally.

Because the contrast is taken between two windows of the *same* day, it also nets
out anything that shifts a whole day's volatility, which is a stronger control
than a before/after comparison across days can offer.

Session times are stated in US Eastern local time throughout and therefore follow
daylight saving automatically, which is the correct behaviour: both NYSE and CME
schedule in local time, so their UTC offsets move together twice a year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from src.config import SessionSpec
from src.inference.regression import RegressionResult, ols_hac
from src.measures.realized import ReturnMethod, _validate_bars, bar_returns
from src.sessions.calendars import complete_et_days, label_bars

logger = logging.getLogger(__name__)

_MINUTES_PER_DAY = 24 * 60


def _minute_of_day(t: time) -> int:
    return t.hour * 60 + t.minute


@dataclass(frozen=True)
class NamedWindow:
    """A clock window in Eastern local time, with its venue status.

    The window is half-open, ``[start, end)``. An ``end`` at or before ``start``
    denotes a window that wraps past midnight; ``end = 00:00`` therefore means
    "to the end of the day".

    ``etf_primary_open`` and ``cme_open`` record whether each venue can transact
    during the window *on a day when that venue is operating at all*. Whether the
    day itself is a holiday is a separate, day-level fact and is never folded in
    here — that separation is what makes the holiday tests possible.
    """

    name: str
    start: time
    end: time
    etf_primary_open: bool
    cme_open: bool
    note: str = ""

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError(
                f"window {self.name!r} has start == end, which is either an empty "
                "window or the whole day; state the intent explicitly"
            )

    @property
    def wraps_midnight(self) -> bool:
        return _minute_of_day(self.end) <= _minute_of_day(self.start)

    @property
    def duration_minutes(self) -> int:
        s, e = _minute_of_day(self.start), _minute_of_day(self.end)
        return (e - s) % _MINUTES_PER_DAY

    def contains_minutes(self, minutes: np.ndarray) -> np.ndarray:
        """Boolean mask over an array of minute-of-day values."""
        s, e = _minute_of_day(self.start), _minute_of_day(self.end)
        if self.wraps_midnight:
            return (minutes >= s) | (minutes < e)
        return (minutes >= s) & (minutes < e)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.name} [{self.start:%H:%M}, {self.end:%H:%M}) ET"


# ---------------------------------------------------------------------------
# The pre-declared day partition
#
# Six windows covering the 24-hour Eastern day exactly once. Boundaries are
# chosen from published venue schedules rather than from the data:
#
#   09:30-16:00  NYSE regular session; ETF creation/redemption window.
#   16:00-17:00  US equities closed; CME still open (CME halts at 17:00 ET).
#   17:00-18:00  CME daily maintenance halt. Neither venue transacting.
#   18:00-00:00  CME reopens for the next trade date. ETF primary market shut.
#                THIS IS THE PLACEBO WINDOW.
#   00:00-06:00  Asian afternoon and evening.
#   06:00-09:30  European morning, pre-US.
#
# CME's 17:00-18:00 halt is what makes 16:00-17:00 and 17:00-18:00 separate
# windows rather than one "US late" block: only the first has CME open, so
# merging them would blur the contrast the placebo depends on.
# ---------------------------------------------------------------------------

DAY_PARTITION: tuple[NamedWindow, ...] = (
    NamedWindow(
        name="asia",
        start=time(0, 0),
        end=time(6, 0),
        etf_primary_open=False,
        cme_open=True,
        note="Asian afternoon/evening; CME open",
    ),
    NamedWindow(
        name="europe",
        start=time(6, 0),
        end=time(9, 30),
        etf_primary_open=False,
        cme_open=True,
        note="European morning, pre-US open",
    ),
    NamedWindow(
        name="us_equity",
        start=time(9, 30),
        end=time(16, 0),
        etf_primary_open=True,
        cme_open=True,
        note="NYSE regular session; ETF creation/redemption window",
    ),
    NamedWindow(
        name="us_post_close",
        start=time(16, 0),
        end=time(17, 0),
        etf_primary_open=False,
        cme_open=True,
        note="After the equity close, before the CME halt",
    ),
    NamedWindow(
        name="cme_halt",
        start=time(17, 0),
        end=time(18, 0),
        etf_primary_open=False,
        cme_open=False,
        note="CME daily maintenance halt; neither venue transacting",
    ),
    NamedWindow(
        name="cme_evening",
        start=time(18, 0),
        end=time(0, 0),
        etf_primary_open=False,
        cme_open=True,
        note="CME reopened for the next trade date; ETF primary market shut. "
        "PLACEBO WINDOW for separating the CME channel from the ETF channel.",
    ),
)

#: The primary within-day contrast declared in the pre-analysis plan (T9).
TREATED_WINDOW = "us_equity"
PLACEBO_WINDOW = "cme_evening"


def validate_partition(
    windows: tuple[NamedWindow, ...] = DAY_PARTITION,
    resolution_minutes: int = 1,
) -> None:
    """Assert that ``windows`` cover the day exactly once.

    Checked rather than assumed, because a partition with a gap silently loses
    variance from the denominator and one with an overlap double-counts it. Both
    would make the shares fail to sum to one, and neither raises on its own.
    """
    if _MINUTES_PER_DAY % resolution_minutes:
        raise ValueError("resolution must divide the day evenly")
    grid = np.arange(0, _MINUTES_PER_DAY, resolution_minutes)
    counts = np.zeros_like(grid)
    for w in windows:
        counts += w.contains_minutes(grid).astype(int)

    if (counts == 0).any():
        gaps = grid[counts == 0]
        raise ValueError(
            f"windows leave {len(gaps)} minute(s) of the day uncovered, "
            f"starting at minute {int(gaps[0])}"
        )
    if (counts > 1).any():
        overlaps = grid[counts > 1]
        raise ValueError(
            f"windows overlap on {len(overlaps)} minute(s) of the day, "
            f"starting at minute {int(overlaps[0])}"
        )


def window_variance_shares(
    bars: pd.DataFrame,
    bar_minutes: int,
    windows: tuple[NamedWindow, ...] = DAY_PARTITION,
    spec: SessionSpec | None = None,
    method: ReturnMethod | str = ReturnMethod.CLOSE_TO_CLOSE,
    min_coverage: float = 0.90,
    drop_incomplete_days: bool = True,
    require_partition: bool = True,
) -> pd.DataFrame:
    """Daily realised variance decomposed across named clock windows.

    Generalises :func:`src.measures.realized.session_variance_shares` from a
    single US window to an arbitrary set of windows, so that the same day's
    variance can be attributed to several venue-status regimes at once.

    Returns a frame indexed by Eastern date with, for each window, the realised
    variance ``rv_<name>``, its share of the day ``share_<name>`` and the bar
    count ``n_bars_<name>``, alongside the day-level total, day type and ETF
    primary-market status.
    """
    spec = spec or SessionSpec()
    _validate_bars(bars)
    if require_partition:
        validate_partition(windows, resolution_minutes=1)

    names = [w.name for w in windows]
    if len(set(names)) != len(names):
        raise ValueError(f"window names must be unique, got {names}")

    labels = label_bars(bars.index, spec=spec, bar_minutes=bar_minutes)
    rets = bar_returns(bars, method=method)

    et_time = labels["et_time"].to_numpy()
    minutes = np.array([t.hour * 60 + t.minute for t in et_time], dtype=int)

    frame = pd.DataFrame(
        {
            "sq": np.square(rets.to_numpy()),
            "et_date": labels["et_date"].to_numpy(),
            "day_type": labels["day_type"].to_numpy(),
            "etf_primary_open": labels["etf_primary_open"].to_numpy(),
        },
        index=bars.index,
    )
    for w in windows:
        mask = w.contains_minutes(minutes)
        frame[f"sq_{w.name}"] = frame["sq"].where(mask)
        frame[f"in_{w.name}"] = mask

    grouped = frame.groupby("et_date", sort=True)
    out = pd.DataFrame(
        {
            "rv_total": grouped["sq"].sum(min_count=1),
            "n_bars_total": grouped.size(),
            "day_type": grouped["day_type"].first(),
            "etf_primary_open": grouped["etf_primary_open"].first(),
        }
    )
    for w in windows:
        out[f"rv_{w.name}"] = grouped[f"sq_{w.name}"].sum(min_count=1).fillna(0.0)
        out[f"n_bars_{w.name}"] = grouped[f"in_{w.name}"].sum()

    # A zero-variance day leaves every share undefined; NaN rather than zero, so
    # that such days are excluded from tests instead of counted as observations.
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in windows:
            out[f"share_{w.name}"] = np.where(
                out["rv_total"] > 0, out[f"rv_{w.name}"] / out["rv_total"], np.nan
            )

    out.index.name = "et_date"

    if drop_incomplete_days:
        keep = complete_et_days(
            labels, bar_minutes=bar_minutes, min_coverage=min_coverage
        )
        dropped = out.index.difference(keep)
        if len(dropped):
            logger.info(
                "Dropping %d day(s) below %.0f%% bar coverage",
                len(dropped),
                min_coverage * 100,
            )
        out = out.loc[out.index.intersection(keep)]

    return out


def per_minute_intensity(
    shares: pd.DataFrame,
    windows: tuple[NamedWindow, ...] = DAY_PARTITION,
) -> pd.DataFrame:
    """Variance share per minute of window length.

    A raw share is not comparable across windows of different length: the
    six-hour CME evening window will hold more variance than the one-hour
    post-close window under any hypothesis whatsoever. Dividing by window length
    gives a quantity that is flat across windows when variance arrives uniformly
    through the day, so departures from flatness are interpretable.

    Under a random walk sampled uniformly, every column has expectation
    ``1 / 1440``.
    """
    out = pd.DataFrame(index=shares.index)
    for w in windows:
        col = f"share_{w.name}"
        if col in shares.columns:
            out[f"intensity_{w.name}"] = shares[col] / w.duration_minutes
    return out


@dataclass
class WithinDayPlaceboResult:
    """Outcome of the T9 within-day placebo test.

    ``contrast_shift`` is the primary quantity and is measured on the **log
    variance ratio** between the two windows, not on the difference of their
    variance shares. See :func:`within_day_placebo` for why the difference of
    shares is not a valid statistic here.
    """

    treated_window: str
    placebo_window: str
    event_date: pd.Timestamp
    n_pre: int
    n_post: int

    treated_shift: dict[str, float]
    placebo_shift: dict[str, float]
    contrast_shift: dict[str, float]
    share_difference_shift: dict[str, float]

    treated_intensity_shift: dict[str, float]
    placebo_intensity_shift: dict[str, float]

    regression: RegressionResult

    @property
    def etf_channel_supported(self) -> bool:
        """Whether the pattern is consistent with an ETF-specific channel.

        Requires the *contrast* between the treated and placebo windows to have
        shifted upward significantly. A rise in the treated window alone is not
        enough, because a rise shared with the placebo window is what the CME
        confound predicts.
        """
        return self.contrast_shift["coef"] > 0 and self.contrast_shift["p"] < 0.05

    @property
    def implied_relative_change(self) -> float:
        """Contrast expressed as a proportional change in the variance ratio."""
        return float(np.expm1(self.contrast_shift["coef"]))

    @property
    def verdict(self) -> str:
        t_up = self.treated_shift["coef"] > 0 and self.treated_shift["p"] < 0.05
        p_up = self.placebo_shift["coef"] > 0 and self.placebo_shift["p"] < 0.05
        if self.etf_channel_supported:
            if p_up:
                return (
                    "ETF-specific component present, but the placebo window also "
                    "rose: part of the effect is not ETF-specific"
                )
            return "consistent with an ETF-specific channel"
        if t_up and p_up:
            return (
                "both windows rose by similar amounts: NOT ETF-specific. Report as "
                "a shift toward US-linked institutional hours generally"
            )
        if t_up:
            return (
                "treated window rose but the contrast is not significant: "
                "underpowered, not evidence either way"
            )
        return "no shift in the treated window"

    def summary(self) -> pd.DataFrame:
        rows = {
            f"share {self.treated_window} (ETF open)": self.treated_shift,
            f"share {self.placebo_window} (ETF shut, CME open)": self.placebo_shift,
            "share difference (INVALID, see docstring)": self.share_difference_shift,
            "log variance ratio (PRIMARY)": self.contrast_shift,
        }
        return pd.DataFrame(
            [
                {
                    "shift": v["coef"],
                    "std_err": v["std_err"],
                    "t": v["t"],
                    "p": v["p"],
                }
                for v in rows.values()
            ],
            index=list(rows),
        )


def _post_dummy_shift(
    y: pd.Series, post: pd.Series, maxlags: int | None
) -> tuple[dict[str, float], RegressionResult]:
    reg = ols_hac(y, post.rename("post").astype(float), maxlags=maxlags)
    return reg.coef("post"), reg


def within_day_placebo(
    shares: pd.DataFrame,
    event_date: date | pd.Timestamp,
    treated_window: str = TREATED_WINDOW,
    placebo_window: str = PLACEBO_WINDOW,
    windows: tuple[NamedWindow, ...] = DAY_PARTITION,
    maxlags: int | None = None,
    open_days_only: bool = True,
) -> WithinDayPlaceboResult:
    """T9: separate the ETF channel from the CME channel within the day.

    Estimates the event-date shift in the treated window's variance share, in the
    placebo window's share, and — the quantity of interest — in the difference
    between them. Because the difference is taken within a day, any factor that
    scales the whole day's volatility cancels, which is a control that no
    across-day comparison provides.

    Parameters
    ----------
    shares
        Output of :func:`window_variance_shares`.
    event_date
        Treatment date. Days on or after it are post-treatment.
    treated_window, placebo_window
        Window names. Defaults are the pre-declared pair.
    open_days_only
        Restrict to days when the ETF primary market was operating. Default True:
        the contrast asks whether the ETF window is special *when the ETF channel
        is available*, so including holidays and weekends would dilute it. The
        closed days are the subject of T3 and T4 instead.

    Notes
    -----
    **The statistic is the log variance ratio, not the difference of shares.**
    The obvious contrast, ``share_treated - share_placebo``, is invalid here, and
    the reason is easy to miss. The two windows have different lengths: 390
    minutes for the equity session against 360 for the CME evening. When a
    common factor scales the variance of both windows equally — exactly what the
    CME confound does — the longer window gains more *share*, because share has
    the whole day's variance in its denominator. The difference of shares
    therefore rises by about 0.9 percentage points under a pure confound with no
    ETF-specific component whatsoever, and a test built on it would report the
    confound as corroboration.

    Taking

        d[t] = log(rv_treated[t] / len_treated) - log(rv_placebo[t] / len_placebo)

    removes the day-total denominator entirely, so any factor that scales the
    whole day cancels exactly rather than approximately, and the length constants
    fall into the intercept so they cannot influence the estimated shift. Under
    the confound the shift in ``d`` is identically zero; under an ETF-specific
    effect it is the log of the variance multiple. The difference of shares is
    still computed and reported, labelled as invalid, so that the discrepancy is
    visible rather than hidden.

    Interpretation is committed in advance. A significant positive contrast
    supports an ETF-specific channel. Similar rises in both windows indicate a
    mechanism tied to US hours generally rather than to creation and redemption,
    and must be reported as such. An insignificant contrast alongside a rise in
    the treated window is underpowered and is evidence for neither.
    """
    for name in (treated_window, placebo_window):
        if f"share_{name}" not in shares.columns:
            raise KeyError(
                f"share_{name!r} not in frame; available: "
                f"{[c for c in shares.columns if c.startswith('share_')]}"
            )
    if treated_window == placebo_window:
        raise ValueError("treated and placebo windows must differ")

    by_name = {w.name: w for w in windows}
    frame = shares.copy()
    if open_days_only and "etf_primary_open" in frame.columns:
        frame = frame.loc[frame["etf_primary_open"].astype(bool)]

    event_ts = pd.Timestamp(event_date)
    post = pd.Series(frame.index >= event_ts, index=frame.index, name="post")

    t_len = by_name[treated_window].duration_minutes
    p_len = by_name[placebo_window].duration_minutes

    treated = frame[f"share_{treated_window}"]
    placebo = frame[f"share_{placebo_window}"]
    rv_treated = frame[f"rv_{treated_window}"]
    rv_placebo = frame[f"rv_{placebo_window}"]

    # A window with exactly zero realised variance makes the log ratio undefined.
    # Dropped rather than floored, so that such days are excluded from the test
    # instead of contributing an arbitrary large negative value.
    usable = (
        treated.notna()
        & placebo.notna()
        & (rv_treated > 0)
        & (rv_placebo > 0)
    )
    n_dropped = int((~usable).sum())
    if n_dropped:
        logger.info("Excluding %d day(s) with a zero-variance window", n_dropped)

    treated, placebo, post = treated[usable], placebo[usable], post[usable]
    log_ratio = (
        np.log(rv_treated[usable] / t_len) - np.log(rv_placebo[usable] / p_len)
    ).rename("log_variance_ratio")

    n_pre, n_post = int((~post).sum()), int(post.sum())
    if min(n_pre, n_post) < 30:
        raise ValueError(
            f"need at least 30 days either side of the event; got {n_pre} pre "
            f"and {n_post} post"
        )

    treated_shift, _ = _post_dummy_shift(treated, post, maxlags)
    placebo_shift, _ = _post_dummy_shift(placebo, post, maxlags)
    contrast_shift, contrast_reg = _post_dummy_shift(log_ratio, post, maxlags)

    # Retained and reported as invalid: see the docstring. Keeping it visible
    # makes the window-length artefact auditable rather than a silent choice.
    share_diff_shift, _ = _post_dummy_shift(treated - placebo, post, maxlags)

    # Length-normalised shares, so the two windows are comparable per minute.
    treated_int, _ = _post_dummy_shift(treated / t_len, post, maxlags)
    placebo_int, _ = _post_dummy_shift(placebo / p_len, post, maxlags)

    return WithinDayPlaceboResult(
        treated_window=treated_window,
        placebo_window=placebo_window,
        event_date=event_ts,
        n_pre=n_pre,
        n_post=n_post,
        treated_shift=treated_shift,
        placebo_shift=placebo_shift,
        contrast_shift=contrast_shift,
        share_difference_shift=share_diff_shift,
        treated_intensity_shift=treated_int,
        placebo_intensity_shift=placebo_int,
        regression=contrast_reg,
    )
