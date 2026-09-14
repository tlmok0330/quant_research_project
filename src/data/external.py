"""
Loaders for external (non-Massive) data, with validation on read.

Loading is separated from acquisition on purpose. ``scripts/fetch_etf_flows.py``
downloads and caches; this module only reads the cache and checks it. That means
an analysis run never depends on a third-party website being up, and the exact
inputs behind a published number are fixed files on disk.

Choice of dose variable
-----------------------
The ETF creation and redemption channel is mechanically symmetric: a redemption
requires the authorised participant to sell spot, just as a creation requires it
to buy. If the hypothesis is that this activity concentrates price discovery
into US hours, then the relevant dose is the *magnitude* of flow, not its sign.

The absolute flow is therefore the pre-registered primary regressor, and signed
net flow is examined separately as a distinct, directional hypothesis about
price pressure. This distinction matters empirically here: about forty per cent
of post-launch days are net outflows, so the two variables are far from
equivalent and choosing between them after seeing results would be a
specification search.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EXTERNAL_DIR

FLOW_LAUNCH_DATES = {"btc": pd.Timestamp("2024-01-11"), "eth": pd.Timestamp("2024-07-23")}


class ExternalDataError(RuntimeError):
    """Raised when a cached external file is missing or fails validation."""


@dataclass
class FlowSeries:
    """Validated daily ETF flow data for one asset."""

    asset: str
    daily: pd.DataFrame
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    n_days: int
    n_outflow_days: int
    autocorr_1: float

    @property
    def abs_flow(self) -> pd.Series:
        """Pre-registered primary dose variable."""
        return self.daily["abs_flow_usd_m"]

    @property
    def net_flow(self) -> pd.Series:
        """Signed flow, for the separate directional hypothesis."""
        return self.daily["net_flow_usd_m"]

    def summary(self) -> dict:
        return {
            "asset": self.asset,
            "first_date": str(self.first_date.date()),
            "last_date": str(self.last_date.date()),
            "n_days": self.n_days,
            "n_outflow_days": self.n_outflow_days,
            "outflow_fraction": self.n_outflow_days / max(1, self.n_days),
            "total_net_flow_usd_bn": float(self.net_flow.sum() / 1000.0),
            "mean_abs_flow_usd_m": float(self.abs_flow.mean()),
            "flow_autocorr_1": self.autocorr_1,
        }


def load_etf_flows(
    asset: str = "btc",
    directory: Path | None = None,
    require_launch_match: bool = True,
) -> FlowSeries:
    """Read and validate cached daily ETF flows.

    Parameters
    ----------
    require_launch_match
        If True, verify that the first flow date matches the launch date assumed
        in ``src.config``. The two are independent: one comes from the flow
        provider, the other from the SEC and issuer record. Agreement is a
        genuine cross-check on the event date, and disagreement is a signal that
        the event date used throughout the study needs revisiting.
    """
    asset = asset.lower()
    directory = directory or EXTERNAL_DIR
    path = directory / f"etf_flows_{asset}_daily.csv"
    if not path.exists():
        raise ExternalDataError(
            f"{path} does not exist. Run 'python scripts/fetch_etf_flows.py "
            f"--asset {asset}' first."
        )

    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    frame = frame.sort_index()

    required = {"net_flow_usd_m", "abs_flow_usd_m"}
    missing = required - set(frame.columns)
    if missing:
        raise ExternalDataError(f"{path.name} is missing column(s): {sorted(missing)}")
    if frame.index.has_duplicates:
        raise ExternalDataError(f"{path.name} contains duplicate dates")
    if frame["net_flow_usd_m"].isna().all():
        raise ExternalDataError(f"{path.name} has no usable flow values")

    # A flow file with no negative values almost certainly means the source's
    # parenthesised negatives were parsed as positive numbers.
    if not (frame["net_flow_usd_m"] < 0).any():
        raise ExternalDataError(
            f"{path.name} contains no negative flows at all. The source renders "
            "outflows in parentheses; they have probably been parsed with the "
            "wrong sign, which would invert the dose variable."
        )

    weekend = frame.index.dayofweek >= 5
    if weekend.any():
        raise ExternalDataError(
            f"{path.name} contains {int(weekend.sum())} weekend date(s); ETF "
            "primary-market flow should exist only on trading days"
        )

    first = frame.index.min()
    if require_launch_match and asset in FLOW_LAUNCH_DATES:
        expected = FLOW_LAUNCH_DATES[asset]
        gap = abs((first - expected).days)
        if gap > 3:
            raise ExternalDataError(
                f"first flow date {first.date()} differs from the assumed launch "
                f"date {expected.date()} by {gap} days. Reconcile these before "
                "proceeding: the event date is used throughout the study."
            )

    return FlowSeries(
        asset=asset,
        daily=frame,
        first_date=first,
        last_date=frame.index.max(),
        n_days=len(frame),
        n_outflow_days=int((frame["net_flow_usd_m"] < 0).sum()),
        autocorr_1=float(frame["net_flow_usd_m"].autocorr(1)),
    )


def load_flows_by_fund(asset: str = "btc", directory: Path | None = None) -> pd.DataFrame:
    """Per-fund daily flows, used for the issuer-concentration robustness check."""
    directory = directory or EXTERNAL_DIR
    path = directory / f"etf_flows_{asset.lower()}_by_fund.csv"
    if not path.exists():
        raise ExternalDataError(
            f"{path} does not exist. Run scripts/fetch_etf_flows.py first."
        )
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    return frame.sort_index()


def align_flow_to_sessions(
    flow: pd.Series,
    daily_index: pd.DatetimeIndex,
    fill_closed_days: float = 0.0,
) -> pd.Series:
    """Align an ETF flow series onto the study's daily calendar.

    Flow exists only on ETF trading days. Days on which the primary market was
    closed are assigned a flow of zero rather than being left missing, because
    zero is the economically correct value: no creation or redemption occurred.
    Leaving them missing would silently drop the holiday and weekend
    observations from any regression that includes flow, which would remove the
    very control days the design depends on.
    """
    aligned = flow.reindex(daily_index)
    return aligned.fillna(fill_closed_days)


def flow_dose_features(
    flow_series: FlowSeries,
    daily_index: pd.DatetimeIndex,
    winsorise_quantile: float | None = 0.99,
) -> pd.DataFrame:
    """Build the dose variables used by the regressions.

    Parameters
    ----------
    winsorise_quantile
        Flow has a heavy right tail: a handful of launch-week and rebalance days
        are several times larger than a typical day. Left untreated, the
        dose-response slope would be determined by a few observations. The
        winsorised variable is the pre-registered primary; the untreated one is
        reported alongside it so that the effect of this choice is visible
        rather than hidden.
    """
    abs_flow = align_flow_to_sessions(flow_series.abs_flow, daily_index)
    net_flow = align_flow_to_sessions(flow_series.net_flow, daily_index)

    out = pd.DataFrame(
        {
            "abs_flow": abs_flow,
            "net_flow": net_flow,
            "has_flow": (abs_flow > 0).astype(float),
        },
        index=daily_index,
    )

    if winsorise_quantile is not None:
        active = abs_flow[abs_flow > 0]
        if len(active) > 20:
            cap = float(active.quantile(winsorise_quantile))
            out["abs_flow_winsorised"] = abs_flow.clip(upper=cap)
            out.attrs["winsorise_cap_usd_m"] = cap
            out.attrs["n_winsorised"] = int((abs_flow > cap).sum())
        else:
            out["abs_flow_winsorised"] = abs_flow
            out.attrs["winsorise_cap_usd_m"] = np.nan
            out.attrs["n_winsorised"] = 0

    # Log scale, since a marginal dollar of flow plausibly matters less on a
    # billion-dollar day than on a hundred-million-dollar day.
    out["log_abs_flow"] = np.log1p(out["abs_flow"])
    return out
