"""
Central configuration: paths, event dates, and the pre-registered primary specification.

Anything in PRIMARY_SPEC was fixed before outcomes were examined. Deviations from it are
robustness checks and must be reported as such.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURE_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Event dates
#
# These are the dates on which US spot ETFs began SECONDARY-MARKET TRADING,
# which is also when authorised participants could first create and redeem.
# Approval dates precede these by one day for BTC and by two months for ETH;
# the trading date is the economically relevant one for this study.
#
# VERIFY BEFORE SUBMISSION against primary sources (SEC filings / issuer press
# releases). These are recorded here so that any later change is a visible,
# logged decision rather than a silent edit.
# ---------------------------------------------------------------------------

BTC_ETF_LAUNCH = date(2024, 1, 11)
ETH_ETF_LAUNCH = date(2024, 7, 23)

# Confounding events that must be acknowledged, not controlled away silently.
BTC_HALVING_2024 = date(2024, 4, 19)
BTC_ETF_OPTIONS_LAUNCH = date(2024, 11, 19)  # listed options on US spot BTC ETFs


# ---------------------------------------------------------------------------
# Pre-registered primary specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSpec:
    """Definition of the US equity clock window.

    The window is a FIXED CLOCK WINDOW and is applied to every calendar day,
    including weekends and holidays. This is deliberate: the holiday and weekend
    control tests require measuring the same clock window on days when the ETF
    primary market is closed. Whether NYSE actually traded is recorded separately
    as a day type, not by changing the window.
    """

    tz: str = "America/New_York"
    start: time = time(9, 30)
    end: time = time(16, 0)

    # Bar timestamps are assumed to mark the OPENING instant of the bar.
    # This MUST be verified against the Massive documentation and empirically;
    # if bars are close-stamped, set to "close" and the labeller shifts them.
    bar_timestamp_convention: str = "open"


@dataclass(frozen=True)
class PrimarySpec:
    """The specification fixed in advance of looking at any outcome."""

    bar_minutes: int = 5
    bar_minutes_fallback: int = 60
    session: SessionSpec = field(default_factory=SessionSpec)

    # Day boundary for aggregating intraday variance into a daily total.
    # ET is chosen because the session of interest is defined in ET.
    day_boundary_tz: str = "America/New_York"

    event_date: date = BTC_ETF_LAUNCH
    sample_start: date = date(2022, 1, 1)
    sample_end: date = date(2025, 12, 31)

    # Minimum fraction of expected bars that a day must have to be usable.
    min_bar_coverage: float = 0.90

    # Primary test is the dose-response regression of session variance share
    # on ETF net flow magnitude, with HAC standard errors.
    hac_max_lags: int = 10

    # Variance-ratio horizons, in bars.
    variance_ratio_k: tuple[int, ...] = (2, 4, 8)

    # Tail threshold for the fragility extension.
    tail_quantile: float = 0.95


PRIMARY_SPEC = PrimarySpec()


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

# Symbol strings are placeholders until the Massive symbology is confirmed.
TREATED_ASSETS = {
    "BTC": {"launch": BTC_ETF_LAUNCH, "symbol_hint": "BTC-USD"},
    "ETH": {"launch": ETH_ETF_LAUNCH, "symbol_hint": "ETH-USD"},
}

# Candidate untreated controls: large-cap crypto with no US spot ETF trading
# during the sample. The choice is a judgement call and results must be shown
# to be robust across at least two of these.
CONTROL_ASSET_CANDIDATES = ("XRP", "ADA", "DOGE", "LTC", "BCH")


# ---------------------------------------------------------------------------
# Credentials (never hard-coded)
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    key = os.getenv("MASSIVE_API_KEY")
    if not key:
        raise EnvironmentError(
            "MASSIVE_API_KEY is not set. Copy .env.example to .env and add the key. "
            "Never place the key in source, notebooks, or committed configuration."
        )
    return key


def get_base_url() -> str:
    return os.getenv("MASSIVE_BASE_URL", "").rstrip("/")
