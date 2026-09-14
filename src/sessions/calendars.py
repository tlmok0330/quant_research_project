"""
Trading-calendar and session-labelling machinery.

The central design decision here is that the US equity window is a FIXED CLOCK
WINDOW (09:30-16:00 ET) applied to every calendar day, while whether the ETF
primary market was actually open is recorded separately as a day type.

That separation is what makes the identification tests possible:

    - NYSE holidays are weekdays on which crypto trades normally but ETF
      creation and redemption does not occur. Measuring the same clock window
      on those days gives a control that holds the crypto market constant.
    - Weekends are permanently free of ETF primary-market activity.

If the window were instead defined as "whenever NYSE is open", both controls
would be undefined, because the window would not exist on those days.

A note on daylight saving: all labelling is performed by converting UTC to
local Eastern time and comparing local clock times. Conversion from UTC to a
local zone is always unambiguous, so this direction of conversion cannot
produce the ambiguous or non-existent timestamps that arise when going the
other way. The DST shift in the corresponding UTC window is therefore handled
automatically rather than by hand.
"""
from __future__ import annotations

import logging
from enum import Enum

import pandas as pd
import pandas_market_calendars as mcal

from src.config import SessionSpec

logger = logging.getLogger(__name__)

_NYSE_REGULAR_CLOSE = pd.Timestamp("16:00").time()


class DayType(str, Enum):
    """Classification of a calendar date by US equity-market status."""

    REGULAR = "regular"
    EARLY_CLOSE = "early_close"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"

    @property
    def etf_primary_open(self) -> bool:
        """Whether ETF creation and redemption could occur on this day."""
        return self in (DayType.REGULAR, DayType.EARLY_CLOSE)


def ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a UTC tz-aware index, raising rather than guessing on naive input.

    Silently assuming a timezone for naive timestamps is the single most common
    source of off-by-one-session errors, so it is refused here.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError(
            "Timestamp index is timezone-naive. Localise it explicitly (almost "
            "certainly to UTC for crypto venue data) before labelling sessions; "
            "assuming a zone here would risk shifting every session boundary."
        )
    return index.tz_convert("UTC")


def nyse_schedule(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """NYSE trading schedule with early closes flagged.

    Returns one row per NYSE trading day, indexed by the local (Eastern) trading
    date, with UTC open and close instants.
    """
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(
        start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
    )

    out = pd.DataFrame(index=pd.DatetimeIndex(sched.index).normalize())
    out.index.name = "date"

    open_utc = pd.DatetimeIndex(sched["market_open"])
    close_utc = pd.DatetimeIndex(sched["market_close"])
    if open_utc.tz is None:
        open_utc = open_utc.tz_localize("UTC")
        close_utc = close_utc.tz_localize("UTC")
    else:
        open_utc = open_utc.tz_convert("UTC")
        close_utc = close_utc.tz_convert("UTC")

    out["market_open_utc"] = open_utc
    out["market_close_utc"] = close_utc

    close_local = close_utc.tz_convert("America/New_York")
    out["close_local_time"] = close_local.time
    out["is_early_close"] = close_local.time < _NYSE_REGULAR_CLOSE
    return out


def classify_days(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Classify every calendar date in the range by US equity-market status.

    Unlike a trading calendar, this returns *all* dates, because weekends and
    holidays are treatment-control groups in this design rather than dates to
    be discarded.
    """
    all_dates = pd.date_range(start=start, end=end, freq="D")
    sched = nyse_schedule(start, end)

    day_type = pd.Series(DayType.HOLIDAY.value, index=all_dates, dtype=object)
    day_type.loc[all_dates.dayofweek >= 5] = DayType.WEEKEND.value

    trading = sched.index.intersection(all_dates)
    day_type.loc[trading] = DayType.REGULAR.value
    early = sched.index[sched["is_early_close"]].intersection(all_dates)
    day_type.loc[early] = DayType.EARLY_CLOSE.value

    out = pd.DataFrame({"day_type": day_type})
    out.index.name = "date"
    out["etf_primary_open"] = out["day_type"].map(
        lambda d: DayType(d).etf_primary_open
    )
    out["dayofweek"] = out.index.dayofweek
    out["is_weekday"] = out["dayofweek"] < 5

    # A weekday on which NYSE did not trade. This is the holiday-test group and
    # includes unscheduled closures (funerals, weather) as well as fixed holidays.
    out["is_closed_weekday"] = out["is_weekday"] & (
        out["day_type"] == DayType.HOLIDAY.value
    )
    return out


def session_windows_utc(
    dates: pd.DatetimeIndex,
    spec: SessionSpec | None = None,
) -> pd.DataFrame:
    """UTC start and end instants of the fixed clock window for each date.

    Provided for export and inspection. Labelling itself does not use this,
    because comparing local clock times is safer than reconstructing UTC bounds.
    """
    spec = spec or SessionSpec()
    dates = pd.DatetimeIndex(dates).normalize()
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    starts = pd.to_datetime(
        [f"{d.date()} {spec.start}" for d in dates]
    ).tz_localize(spec.tz, nonexistent="shift_forward", ambiguous=True)
    ends = pd.to_datetime(
        [f"{d.date()} {spec.end}" for d in dates]
    ).tz_localize(spec.tz, nonexistent="shift_forward", ambiguous=True)

    return pd.DataFrame(
        {
            "window_start_utc": starts.tz_convert("UTC"),
            "window_end_utc": ends.tz_convert("UTC"),
        },
        index=dates,
    )


def label_bars(
    index: pd.DatetimeIndex,
    spec: SessionSpec | None = None,
    bar_minutes: int | None = None,
) -> pd.DataFrame:
    """Label intraday bar timestamps with session and day-type information.

    Parameters
    ----------
    index
        Timezone-aware bar timestamps. Naive input is refused.
    spec
        Session definition. ``spec.bar_timestamp_convention`` controls whether
        the timestamp is treated as the bar's opening or closing instant.
    bar_minutes
        Bar duration. Required only when timestamps are close-stamped, in which
        case the bar is shifted back to its opening instant before labelling.

    Returns
    -------
    DataFrame indexed by the original timestamps, with columns:
        ``et_time``        local Eastern wall-clock time of the bar open
        ``et_date``        local Eastern date, used as the day boundary
        ``in_us_window``   whether the bar falls in the fixed clock window
        ``day_type``       classification of ``et_date``
        ``etf_primary_open`` whether ETF creation/redemption could occur
    """
    spec = spec or SessionSpec()
    idx = ensure_utc(index)

    convention = spec.bar_timestamp_convention.lower()
    if convention not in ("open", "close"):
        raise ValueError("bar_timestamp_convention must be 'open' or 'close'")

    if convention == "close":
        if bar_minutes is None:
            raise ValueError(
                "bar_minutes is required when bars are close-stamped, so that the "
                "opening instant can be recovered before session labelling."
            )
        effective = idx - pd.Timedelta(minutes=bar_minutes)
    else:
        effective = idx

    local = effective.tz_convert(spec.tz)
    et_time = local.time
    et_date = pd.DatetimeIndex(local.date)

    # Half-open interval [start, end): a five-minute bar opening at 15:55 belongs
    # to the session; one opening at 16:00 does not.
    in_window = pd.Series(
        [(t >= spec.start) and (t < spec.end) for t in et_time], index=idx
    )

    day_info = classify_days(et_date.min(), et_date.max())
    mapped = day_info.reindex(et_date)

    out = pd.DataFrame(
        {
            "et_time": et_time,
            "et_date": et_date,
            "in_us_window": in_window.to_numpy(),
            "day_type": mapped["day_type"].to_numpy(),
            "etf_primary_open": mapped["etf_primary_open"].to_numpy(),
        },
        index=idx,
    )
    out.index.name = "timestamp"
    return out


def utc_range_for_et_dates(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    spec: SessionSpec | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC instants needed to cover a range of complete Eastern calendar days.

    A UTC calendar day is not an Eastern calendar day: 00:00 UTC on a given date
    is 19:00 or 20:00 the previous evening in New York. Requesting bars for the
    nominal UTC range therefore leaves the first and last Eastern days partially
    covered, which would bias their variance shares downward for no reason other
    than a fetch boundary.

    Fetching this wider UTC range and then discarding incomplete Eastern days is
    the safe pattern.

    Returns
    -------
    (start_utc, end_utc) covering [start_date 00:00 ET, end_date 24:00 ET).
    """
    spec = spec or SessionSpec()
    start_local = pd.Timestamp(start_date).normalize().tz_localize(
        spec.tz, nonexistent="shift_forward", ambiguous=True
    )
    end_local = (
        pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    ).tz_localize(spec.tz, nonexistent="shift_forward", ambiguous=True)
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def complete_et_days(
    labelled: pd.DataFrame,
    bar_minutes: int,
    min_coverage: float = 0.90,
) -> pd.Index:
    """Eastern dates with enough bars to be usable.

    Applied after labelling to drop the partially covered days at the edges of a
    fetch window, and any day with a genuine data gap.
    """
    required = expected_bars_per_day(bar_minutes) * min_coverage
    counts = labelled.groupby("et_date").size()
    return counts[counts >= required].index


def expected_bars_per_day(bar_minutes: int) -> int:
    """Bars in a full 24-hour crypto day, used for coverage checks."""
    if 1440 % bar_minutes:
        raise ValueError(f"{bar_minutes}-minute bars do not divide a 24-hour day evenly")
    return 1440 // bar_minutes


def expected_bars_in_window(bar_minutes: int, spec: SessionSpec | None = None) -> int:
    """Bars inside the fixed clock window, used for coverage checks."""
    spec = spec or SessionSpec()
    start_min = spec.start.hour * 60 + spec.start.minute
    end_min = spec.end.hour * 60 + spec.end.minute
    span = end_min - start_min
    if span % bar_minutes:
        raise ValueError(
            f"{bar_minutes}-minute bars do not divide the {span}-minute window evenly"
        )
    return span // bar_minutes
