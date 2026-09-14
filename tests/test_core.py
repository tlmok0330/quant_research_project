"""Small, explainable checks for the submitted analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.client import normalise_bars, redact
from src.data.quality import audit_bars
from src.measures.realized import ReturnMethod, session_variance_shares
from src.measures.windows import DAY_PARTITION, window_variance_shares
from src.sessions.calendars import label_bars, utc_range_for_et_dates


def make_clean_day(date: str = "2024-01-11") -> pd.DataFrame:
    """Create one complete Eastern day of simple five minute bars."""
    start, end = utc_range_for_et_dates(date, date)
    index = pd.date_range(start, end, freq="5min", inclusive="left")
    open_price = np.full(len(index), 40_000.0)
    close_price = open_price * np.exp(0.001)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": close_price,
            "low": open_price,
            "close": close_price,
            "volume": 1.0,
        },
        index=index,
    )


def test_api_credentials_are_removed_from_messages():
    secret = "example_secret_123"
    message = redact(f"https://api.massive.com?apiKey={secret}", secret)
    assert secret not in message
    assert "REDACTED" in message


def test_massive_bar_fields_are_normalised():
    raw = [
        {
            "t": 1704067200000,
            "o": 40_000,
            "h": 40_100,
            "l": 39_900,
            "c": 40_050,
            "v": 12.5,
        }
    ]
    bars = normalise_bars(raw)
    assert list(bars.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert bars.index.tz is not None
    assert bars["close"].iloc[0] == 40_050


def test_holiday_has_the_same_clock_window_but_etf_is_closed():
    start, end = utc_range_for_et_dates("2024-07-04", "2024-07-04")
    index = pd.date_range(start, end, freq="5min", inclusive="left")
    labels = label_bars(index, bar_minutes=5)
    assert labels["in_us_window"].sum() == 78
    assert not labels["etf_primary_open"].any()


def test_session_variance_share_has_an_interpretable_value():
    bars = make_clean_day()
    daily = session_variance_shares(
        bars,
        bar_minutes=5,
        method=ReturnMethod.INTRA_BAR,
    )
    expected = 78 / 288
    assert daily["share_us"].iloc[0] == pytest.approx(expected)


def test_intraday_windows_partition_all_daily_variance():
    shares = window_variance_shares(
        make_clean_day(),
        bar_minutes=5,
        method=ReturnMethod.INTRA_BAR,
    )
    columns = [f"share_{window.name}" for window in DAY_PARTITION]
    assert shares[columns].sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_clean_bars_pass_critical_quality_checks():
    report = audit_bars(make_clean_day(), bar_minutes=5, check_convention=False)
    assert report.ok
    assert report.critical_failures == []
