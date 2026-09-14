"""
Frozen NYSE calendar, independent of a live library call at analysis time.

The holiday and early-close dates define the treated and control groups for
tests T3 and T4. They are therefore frozen to ``data/external/nyse_calendar.csv``
by ``scripts/freeze_nyse_calendar.py`` and loaded from that file. A live
``pandas_market_calendars`` call is used only as a cross-check: any discrepancy
raises, rather than silently updating the sample.

Unscheduled closures (for example 9 January 2025, a national day of mourning)
are included because they are weekdays on which ETF creation and redemption
did not occur, which is exactly the definition the holiday test needs. A
calendar that listed only statutory holidays would miss them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import EXTERNAL_DIR, PRIMARY_SPEC
from src.data.external import ExternalDataError
from src.sessions.calendars import DayType, classify_days

CALENDAR_PATH = EXTERNAL_DIR / "nyse_calendar.csv"
EXPECTED_COLUMNS = (
    "date",
    "source",
    "frozen_on",
    "day_type",
    "etf_primary_open",
    "dayofweek",
    "is_weekday",
    "is_closed_weekday",
)


@dataclass
class FrozenCalendar:
    """Validated day-type classification covering the study sample."""

    days: pd.DataFrame
    path: Path
    source: str
    frozen_on: pd.Timestamp
    n_closed_weekdays: int

    def closed_weekdays(self) -> pd.DatetimeIndex:
        return self.days.index[self.days["is_closed_weekday"]]

    def early_closes(self) -> pd.DatetimeIndex:
        return self.days.index[self.days["day_type"] == DayType.EARLY_CLOSE.value]

    def regular_open(self) -> pd.DatetimeIndex:
        return self.days.index[self.days["day_type"] == DayType.REGULAR.value]


def load_frozen_calendar(
    path: Path | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> FrozenCalendar:
    """Read the frozen calendar and validate it against known properties."""
    path = path or CALENDAR_PATH
    if not path.exists():
        raise ExternalDataError(
            f"Frozen NYSE calendar not found at {path}. "
            "Run python scripts/freeze_nyse_calendar.py first."
        )

    days = pd.read_csv(path, parse_dates=["date", "frozen_on"])
    missing = [c for c in EXPECTED_COLUMNS if c not in days.columns]
    if missing:
        raise ExternalDataError(f"{path} is missing columns {missing}")

    days = days.set_index("date").sort_index()
    days.index = pd.DatetimeIndex(days.index).normalize()

    start = pd.Timestamp(start or PRIMARY_SPEC.sample_start)
    end = pd.Timestamp(end or PRIMARY_SPEC.sample_end)
    expected = pd.date_range(start, end, freq="D")
    if not expected.isin(days.index).all():
        gap = expected.difference(days.index)
        raise ExternalDataError(
            f"{path} does not cover the sample: {len(gap)} missing date(s), "
            f"first {gap[0].date() if len(gap) else 'n/a'}"
        )
    days = days.loc[expected]

    valid_types = {t.value for t in DayType}
    unknown = set(days["day_type"].unique()) - valid_types
    if unknown:
        raise ExternalDataError(f"{path} has unknown day_type values {unknown}")

    # Weekends must be labelled as weekends, never as holidays.
    weekend_mislabelled = days.index[days.index.dayofweek >= 5].difference(
        days.index[days["day_type"] == DayType.WEEKEND.value]
    )
    if len(weekend_mislabelled):
        raise ExternalDataError(
            f"{path} labels {len(weekend_mislabelled)} weekend date(s) as "
            f"something other than weekend, first {weekend_mislabelled[0].date()}"
        )

    # Closed weekdays must have the ETF primary market shut.
    closed = days["is_closed_weekday"].astype(bool)
    if days.loc[closed, "etf_primary_open"].any():
        raise ExternalDataError(
            f"{path} marks a closed weekday as ETF-primary-open, which would "
            "put a holiday into the treated group of T3"
        )

    n_closed = int(closed.sum())
    if n_closed < 30:
        # Four years of NYSE holidays is typically ~40. Fewer than 30 is a
        # sign the freeze dropped unscheduled closures or used the wrong calendar.
        raise ExternalDataError(
            f"{path} has only {n_closed} closed weekdays in {start.date()} to "
            f"{end.date()}; expected roughly forty"
        )

    return FrozenCalendar(
        days=days,
        path=path,
        source=str(days["source"].iloc[0]),
        frozen_on=pd.Timestamp(days["frozen_on"].iloc[0]),
        n_closed_weekdays=n_closed,
    )


def cross_check_against_live_library(frozen: FrozenCalendar) -> pd.DataFrame:
    """Compare the frozen file to a live pandas-market-calendars call.

    Returns a frame of discrepancies. Empty means agreement. Called from tests
    and from the freeze script so that a library upgrade cannot silently change
    the sample.
    """
    live = classify_days(frozen.days.index.min(), frozen.days.index.max())
    live.index = pd.DatetimeIndex(live.index).normalize()
    aligned = frozen.days[["day_type", "etf_primary_open", "is_closed_weekday"]].join(
        live[["day_type", "etf_primary_open", "is_closed_weekday"]],
        how="inner",
        lsuffix="_frozen",
        rsuffix="_live",
    )
    disagree = (
        (aligned["day_type_frozen"] != aligned["day_type_live"])
        | (aligned["etf_primary_open_frozen"] != aligned["etf_primary_open_live"])
        | (aligned["is_closed_weekday_frozen"] != aligned["is_closed_weekday_live"])
    )
    return aligned.loc[disagree]
