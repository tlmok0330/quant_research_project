"""
Data-quality checks for intraday bar data.

Philosophy
----------
Checks are reported, not silently repaired. An automatic fix hides the thing we
most need to know: whether the data supports the analysis at all. Every check
therefore returns a finding with a severity, and the caller decides.

Severities
----------
``critical``  invalidates the analysis if left unaddressed
``warning``   affects interpretation and must be disclosed
``info``      worth recording in the decision log

The bar-convention check deserves special mention. Whether a bar's timestamp
marks its opening or its closing instant is usually a single line in a
provider's documentation, easy to overlook and impossible to notice later: get
it wrong and every session boundary shifts by one bar, biasing every variance
share in the study without producing any visible error. It is therefore inferred
empirically from the data and cross-checked against the configured assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["critical", "warning", "info"]


@dataclass
class Finding:
    check: str
    severity: Severity
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        mark = "ok  " if self.passed else {
            "critical": "FAIL",
            "warning": "warn",
            "info": "note",
        }[self.severity]
        return f"[{mark}] {self.check}: {self.message}"


@dataclass
class QualityReport:
    findings: list[Finding] = field(default_factory=list)
    n_bars: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def critical_failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity == "critical"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.critical_failures

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": f.check,
                    "severity": f.severity,
                    "passed": f.passed,
                    "message": f.message,
                }
                for f in self.findings
            ]
        )

    def assert_usable(self) -> None:
        """Raise if any critical check failed."""
        if self.critical_failures:
            msgs = "\n  ".join(f"{f.check}: {f.message}" for f in self.critical_failures)
            raise ValueError(f"data failed critical quality checks:\n  {msgs}")

    def __str__(self) -> str:  # pragma: no cover
        header = f"Quality report: {self.n_bars:,} bars"
        if self.first_timestamp is not None:
            header += f" from {self.first_timestamp} to {self.last_timestamp}"
        lines = [header, "-" * len(header)]
        lines += [str(f) for f in self.findings]
        n_crit, n_warn = len(self.critical_failures), len(self.warnings)
        lines.append("-" * len(header))
        lines.append(f"{n_crit} critical failure(s), {n_warn} warning(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_timezone(index: pd.DatetimeIndex) -> Finding:
    aware = index.tz is not None
    return Finding(
        check="timezone_aware",
        severity="critical",
        passed=aware,
        message=(
            f"index timezone is {index.tz}"
            if aware
            else "index is timezone-naive; session labelling cannot be trusted"
        ),
        detail={"tz": str(index.tz)},
    )


def check_monotonic_unique(index: pd.DatetimeIndex) -> list[Finding]:
    out = [
        Finding(
            check="monotonic_index",
            severity="critical",
            passed=bool(index.is_monotonic_increasing),
            message=(
                "timestamps are sorted"
                if index.is_monotonic_increasing
                else "timestamps are not sorted; consecutive-bar returns are meaningless"
            ),
        )
    ]
    n_dup = int(index.duplicated().sum())
    out.append(
        Finding(
            check="unique_timestamps",
            severity="critical",
            passed=n_dup == 0,
            message=(
                "no duplicate timestamps"
                if n_dup == 0
                else f"{n_dup:,} duplicate timestamp(s); these inflate realised variance"
            ),
            detail={"n_duplicates": n_dup},
        )
    )
    return out


def check_ohlc_consistency(bars: pd.DataFrame) -> list[Finding]:
    findings = []
    cols = [c for c in ("open", "high", "low", "close") if c in bars.columns]
    if len(cols) < 2:
        return [
            Finding(
                check="ohlc_present",
                severity="critical",
                passed=False,
                message=f"expected OHLC columns, found {list(bars.columns)}",
            )
        ]

    n_nonpos = int((bars[cols] <= 0).any(axis=1).sum())
    findings.append(
        Finding(
            check="positive_prices",
            severity="critical",
            passed=n_nonpos == 0,
            message=(
                "all prices positive"
                if n_nonpos == 0
                else f"{n_nonpos:,} bar(s) with a non-positive price; log returns undefined"
            ),
            detail={"n_non_positive": n_nonpos},
        )
    )

    if {"high", "low", "open", "close"} <= set(bars.columns):
        bad_high = int((bars["high"] < bars[["open", "close"]].max(axis=1) - 1e-9).sum())
        bad_low = int((bars["low"] > bars[["open", "close"]].min(axis=1) + 1e-9).sum())
        bad_range = int((bars["high"] < bars["low"]).sum())
        n_bad = bad_high + bad_low + bad_range
        findings.append(
            Finding(
                check="ohlc_ordering",
                severity="critical",
                passed=n_bad == 0,
                message=(
                    "high/low bracket open and close everywhere"
                    if n_bad == 0
                    else f"{n_bad:,} bar(s) violate OHLC ordering"
                ),
                detail={
                    "high_below_body": bad_high,
                    "low_above_body": bad_low,
                    "high_below_low": bad_range,
                },
            )
        )
    return findings


def check_grid_and_gaps(
    index: pd.DatetimeIndex,
    bar_minutes: int,
    max_gap_fraction: float = 0.02,
) -> list[Finding]:
    """Check bars sit on the expected time grid, and quantify missing bars."""
    findings = []
    expected = pd.date_range(index.min(), index.max(), freq=f"{bar_minutes}min")
    missing = expected.difference(index)
    frac = len(missing) / max(1, len(expected))

    findings.append(
        Finding(
            check="bar_coverage",
            severity="warning" if frac <= max_gap_fraction else "critical",
            passed=frac <= max_gap_fraction,
            message=(
                f"{len(missing):,} of {len(expected):,} expected bars missing "
                f"({frac:.2%})"
            ),
            detail={
                "n_missing": len(missing),
                "n_expected": len(expected),
                "fraction_missing": frac,
                "first_missing": str(missing[0]) if len(missing) else None,
            },
        )
    )

    off_grid = index.difference(expected)
    findings.append(
        Finding(
            check="on_grid",
            severity="warning",
            passed=len(off_grid) == 0,
            message=(
                "all timestamps lie on the expected grid"
                if len(off_grid) == 0
                else f"{len(off_grid):,} timestamp(s) off the {bar_minutes}-minute grid"
            ),
            detail={"n_off_grid": len(off_grid)},
        )
    )

    if len(index) > 1:
        deltas = pd.Series(index[1:] - index[:-1])
        runs = deltas[deltas > pd.Timedelta(minutes=bar_minutes)]
        longest = runs.max() if len(runs) else pd.Timedelta(0)
        findings.append(
            Finding(
                check="longest_gap",
                severity="info",
                passed=True,
                message=f"longest gap between consecutive bars is {longest}",
                detail={
                    "longest_gap_minutes": longest.total_seconds() / 60,
                    "n_gaps": int(len(runs)),
                },
            )
        )
    return findings


def check_grid_alignment(index: pd.DatetimeIndex, bar_minutes: int) -> Finding:
    """Check timestamps sit exactly on a ``bar_minutes`` grid from midnight."""
    utc = index.tz_convert("UTC") if index.tz is not None else index
    seconds_of_day = (
        utc.hour * 3600 + utc.minute * 60 + utc.second + utc.microsecond / 1e6
    )
    residuals = np.unique(np.asarray(seconds_of_day) % (bar_minutes * 60))
    aligned = residuals.size == 1 and abs(residuals[0]) < 1e-9
    return Finding(
        check="grid_alignment",
        severity="warning",
        passed=bool(aligned),
        message=(
            f"timestamps lie exactly on the {bar_minutes}-minute grid"
            if aligned
            else f"timestamps are off-grid; distinct offsets found: "
            f"{np.round(residuals[:5], 3).tolist()}"
        ),
        detail={"n_distinct_offsets": int(residuals.size)},
    )


def infer_bar_convention(index: pd.DatetimeIndex, bar_minutes: int) -> Finding:
    """Report whether the open/close bar convention is identifiable at all.

    It is tempting to infer the convention from the timestamp grid, on the
    reasoning that open-stamped bars start at midnight while close-stamped bars
    start one bar width later. That reasoning fails for continuously traded
    assets.

    On a 24/7 series covering whole days, a bar *closing* at midnight is simply
    the last bar of the previous day, so a midnight-aligned timestamp appears
    under both conventions and the two produce identical minute-of-day
    distributions. The convention is therefore genuinely unidentifiable from
    crypto timestamps alone.

    Two routes remain, and both are recorded in the research plan:

    1. The provider documentation, which is authoritative and must be read.
    2. A session-bounded feed from the same provider, such as equities. There
       the first bar of a trading day is stamped 09:30 under the open convention
       and 09:35 under the close convention, which *is* identifiable. Since a
       provider almost always applies one convention across products, that
       resolves the crypto case indirectly. See
       :func:`infer_convention_from_session_data`.

    Reporting the limitation is the correct behaviour here. Guessing would
    reintroduce exactly the silent one-bar shift the check exists to prevent.
    """
    alignment = check_grid_alignment(index, bar_minutes)
    if not alignment.passed:
        return Finding(
            check="bar_convention",
            severity="warning",
            passed=False,
            message=(
                "timestamps are not grid-aligned, so the bar convention cannot be "
                "inferred and the documented convention cannot be cross-checked"
            ),
            detail={"inferred_convention": None, "reason": "off_grid"},
        )

    return Finding(
        check="bar_convention",
        severity="warning",
        passed=False,
        message=(
            "bar convention is not identifiable from a continuous 24/7 timestamp "
            "grid: both conventions produce the same minute-of-day pattern. "
            "Confirm from provider documentation, and cross-check against a "
            "session-bounded feed such as equities."
        ),
        detail={"inferred_convention": None, "reason": "not_identifiable_24_7"},
    )


def infer_convention_from_session_data(
    index: pd.DatetimeIndex,
    bar_minutes: int,
    session_open_local: str = "09:30",
    tz: str = "America/New_York",
) -> Finding:
    """Infer the bar convention from a session-bounded feed such as equities.

    Unlike a 24/7 series, a feed that only carries bars while an exchange is open
    does identify the convention. Under the open-stamped convention the first bar
    of the day is stamped at the session open; under the close-stamped
    convention it is stamped one bar width later, and no bar carries the open
    timestamp itself.

    Providers normally apply a single convention across products, so a result
    here can be carried over to the crypto feed. That inference should be stated
    explicitly in the report rather than assumed.
    """
    if index.tz is None:
        raise ValueError("index must be timezone-aware")
    local = index.tz_convert(tz)
    open_time = pd.Timestamp(session_open_local).time()

    minute_of_day = np.asarray(local.hour * 60 + local.minute)
    open_minute = open_time.hour * 60 + open_time.minute

    n_at_open = int((minute_of_day == open_minute).sum())
    n_one_bar_later = int((minute_of_day == open_minute + bar_minutes).sum())

    if n_at_open == 0 and n_one_bar_later > 0:
        inferred = "close"
    elif n_at_open > 0 and n_at_open >= n_one_bar_later * 0.5:
        inferred = "open"
    else:
        return Finding(
            check="bar_convention_from_session",
            severity="warning",
            passed=False,
            message=(
                f"inconclusive: {n_at_open} bar(s) stamped at {session_open_local} "
                f"and {n_one_bar_later} one bar later"
            ),
            detail={
                "inferred_convention": None,
                "n_at_open": n_at_open,
                "n_one_bar_later": n_one_bar_later,
            },
        )

    return Finding(
        check="bar_convention_from_session",
        severity="info",
        passed=True,
        message=(
            f"session feed indicates {inferred}-stamped bars "
            f"({n_at_open} bar(s) at {session_open_local}, "
            f"{n_one_bar_later} one bar later)"
        ),
        detail={
            "inferred_convention": inferred,
            "n_at_open": n_at_open,
            "n_one_bar_later": n_one_bar_later,
        },
    )


def convention_check_via_scheduled_event(
    bars: pd.DataFrame,
    event_times_utc: list[pd.Timestamp],
    bar_minutes: int,
    window_bars: int = 3,
) -> pd.DataFrame:
    """Locate the volatility spike around scheduled announcements.

    An independent cross-check on the bar convention. A release at a known
    instant produces a burst of variance in the bar that *contains* that
    instant. Under the open-stamped convention that is the bar stamped at the
    release time; under the close-stamped convention it is the bar stamped one
    bar width later.

    Individual events are noisy, so this is only informative when averaged over
    many of them. It is offered as corroboration of the documented convention,
    not as a substitute for reading the documentation.

    Returns mean absolute return by bar offset relative to each event.
    """
    close = bars["close"].astype(float)
    rets = np.log(close).diff().abs()

    rows = []
    for event in event_times_utc:
        event = pd.Timestamp(event)
        if event.tz is None:
            raise ValueError("event times must be timezone-aware")
        for offset in range(-window_bars, window_bars + 1):
            stamp = event + pd.Timedelta(minutes=offset * bar_minutes)
            if stamp in rets.index:
                value = rets.loc[stamp]
                if np.isfinite(value):
                    rows.append({"offset_bars": offset, "abs_return": float(value)})

    if not rows:
        return pd.DataFrame(columns=["offset_bars", "mean_abs_return", "n"])

    frame = pd.DataFrame(rows)
    out = frame.groupby("offset_bars")["abs_return"].agg(["mean", "count"])
    out.columns = ["mean_abs_return", "n"]
    return out.reset_index()


def check_stale_and_outliers(
    bars: pd.DataFrame,
    max_stale_run: int = 12,
    return_sigma: float = 12.0,
) -> list[Finding]:
    """Flag repeated identical closes and extreme returns."""
    findings = []
    close = bars["close"].astype(float)

    unchanged = close.diff() == 0
    # Length of the longest run of identical closes.
    groups = (~unchanged).cumsum()
    run_lengths = unchanged.groupby(groups).sum()
    longest_run = int(run_lengths.max()) if len(run_lengths) else 0
    findings.append(
        Finding(
            check="stale_prices",
            severity="warning",
            passed=longest_run <= max_stale_run,
            message=(
                f"longest run of unchanged closes is {longest_run} bar(s)"
                + (
                    ""
                    if longest_run <= max_stale_run
                    else f", above the tolerance of {max_stale_run}; "
                    "these bars contribute zero variance and may be filled data"
                )
            ),
            detail={
                "longest_unchanged_run": longest_run,
                "fraction_unchanged": float(unchanged.mean()),
            },
        )
    )

    rets = np.log(close).diff().dropna()
    if len(rets) > 100:
        # Median absolute deviation is used rather than the standard deviation,
        # because a single extreme print inflates the very yardstick meant to
        # detect it.
        mad = float(np.median(np.abs(rets - rets.median())))
        robust_sd = 1.4826 * mad
        if robust_sd > 0:
            z = (rets - rets.median()) / robust_sd
            n_extreme = int((z.abs() > return_sigma).sum())
            worst = float(z.abs().max())
            findings.append(
                Finding(
                    check="return_outliers",
                    severity="warning",
                    passed=n_extreme == 0,
                    message=(
                        f"{n_extreme:,} return(s) beyond {return_sigma} robust sigma; "
                        f"largest is {worst:.1f} sigma"
                    ),
                    detail={
                        "n_extreme": n_extreme,
                        "max_abs_z": worst,
                        "robust_sd": robust_sd,
                    },
                )
            )
    return findings


def check_zero_volume(bars: pd.DataFrame, max_fraction: float = 0.10) -> Finding | None:
    if "volume" not in bars.columns:
        return None
    zero = bars["volume"].fillna(0) <= 0
    frac = float(zero.mean())
    return Finding(
        check="zero_volume_bars",
        severity="warning",
        passed=frac <= max_fraction,
        message=f"{int(zero.sum()):,} bar(s) with zero volume ({frac:.2%})",
        detail={"fraction_zero_volume": frac},
    )


def check_return_method_agreement(
    bars: pd.DataFrame,
    tolerance: float = 0.05,
) -> Finding | None:
    """Cross-check the two return definitions.

    On contiguous bars, close-to-close and intra-bar realised variance should
    agree closely, because each bar's open equals the previous close. A material
    disagreement means the bars are not contiguous, or are misaligned, and is a
    more sensitive detector of that than a gap count.
    """
    if not {"open", "close"} <= set(bars.columns):
        return None
    close = bars["close"].astype(float)
    open_ = bars["open"].astype(float)
    if (close <= 0).any() or (open_ <= 0).any():
        return None

    rv_c2c = float(np.nansum(np.square(np.log(close).diff())))
    rv_intra = float(np.nansum(np.square(np.log(close) - np.log(open_))))
    if rv_intra <= 0:
        return None
    rel = abs(rv_c2c - rv_intra) / rv_intra
    return Finding(
        check="return_method_agreement",
        severity="warning",
        passed=rel <= tolerance,
        message=(
            f"close-to-close and intra-bar realised variance differ by {rel:.2%}"
            + ("" if rel <= tolerance else "; bars may not be contiguous")
        ),
        detail={"rv_close_to_close": rv_c2c, "rv_intra_bar": rv_intra, "relative": rel},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def audit_bars(
    bars: pd.DataFrame,
    bar_minutes: int,
    expected_convention: str = "open",
    max_gap_fraction: float = 0.02,
    check_convention: bool = True,
    session_feed_index: pd.DatetimeIndex | None = None,
) -> QualityReport:
    """Run every check and return a consolidated report.

    Parameters
    ----------
    session_feed_index
        Optional timestamps from a session-bounded feed by the same provider,
        such as equity bars. Supplying these is the only way to resolve the bar
        convention empirically, because a 24/7 series cannot do so.
    """
    report = QualityReport(
        n_bars=len(bars),
        first_timestamp=bars.index.min() if len(bars) else None,
        last_timestamp=bars.index.max() if len(bars) else None,
    )
    if bars.empty:
        report.add(
            Finding(
                check="non_empty",
                severity="critical",
                passed=False,
                message="no bars supplied",
            )
        )
        return report

    index = pd.DatetimeIndex(bars.index)
    report.add(check_timezone(index))
    for f in check_monotonic_unique(index):
        report.add(f)
    for f in check_ohlc_consistency(bars):
        report.add(f)

    # Grid checks require a sorted, unique, tz-aware index to be meaningful.
    if report.ok:
        for f in check_grid_and_gaps(index, bar_minutes, max_gap_fraction):
            report.add(f)
        report.add(check_grid_alignment(index, bar_minutes))

        if check_convention:
            report.add(infer_bar_convention(index, bar_minutes))
            if session_feed_index is not None:
                session_finding = infer_convention_from_session_data(
                    pd.DatetimeIndex(session_feed_index), bar_minutes
                )
                report.add(session_finding)
                inferred = session_finding.detail.get("inferred_convention")
                if inferred and inferred != expected_convention:
                    report.add(
                        Finding(
                            check="bar_convention_matches_config",
                            severity="critical",
                            passed=False,
                            message=(
                                f"the provider's session feed indicates "
                                f"{inferred}-stamped bars but the configuration "
                                f"assumes {expected_convention}-stamped; every "
                                "session boundary would be shifted by one bar"
                            ),
                            detail={
                                "inferred": inferred,
                                "configured": expected_convention,
                            },
                        )
                    )

        for f in check_stale_and_outliers(bars):
            report.add(f)
        zv = check_zero_volume(bars)
        if zv is not None:
            report.add(zv)
        agree = check_return_method_agreement(bars)
        if agree is not None:
            report.add(agree)

    return report


def daily_coverage_summary(
    bars: pd.DataFrame,
    bar_minutes: int,
) -> pd.DataFrame:
    """Bars per UTC day, for spotting outages and partial days."""
    per_day = bars.groupby(pd.DatetimeIndex(bars.index).date).size()
    expected = 1440 // bar_minutes
    out = pd.DataFrame({"n_bars": per_day})
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "utc_date"
    out["expected"] = expected
    out["coverage"] = out["n_bars"] / expected
    out["is_complete"] = out["coverage"] >= 0.999
    return out
