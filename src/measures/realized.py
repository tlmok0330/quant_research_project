"""
Realised-variance estimators and the session variance share, which is the
primary dependent variable of this study.

Why a share rather than a level
-------------------------------
Bitcoin's unconditional volatility changed substantially over 2022-2025, and the
post-event window coincides with a bull market. Comparing the *level* of
US-session variance before and after the ETF launch would therefore largely
measure the change in overall volatility rather than any change in where price
discovery occurs. Normalising by the day's total realised variance removes that
first-order confound by construction.

Return attribution
------------------
Two return definitions are provided.

``close_to_close`` is the literature standard: r_t = log(C_t / C_{t-1}), attributed
to the session of bar t, since that return is realised over bar t's interval.

``intra_bar`` uses r_t = log(C_t / O_t), which lies strictly inside a single bar
and therefore inside a single session, at the cost of discarding the small gap
between one bar's close and the next bar's open.

For contiguous 24/7 crypto bars the two should agree closely. A material
disagreement is itself a data-quality signal, indicating gaps or misaligned
bars, and is checked explicitly in the quality module.
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

from src.config import SessionSpec
from src.sessions.calendars import complete_et_days, label_bars

logger = logging.getLogger(__name__)

_REQUIRED_OHLC = ("open", "high", "low", "close")


class ReturnMethod(str, Enum):
    CLOSE_TO_CLOSE = "close_to_close"
    INTRA_BAR = "intra_bar"


def _validate_bars(bars: pd.DataFrame, need_high_low: bool = False) -> None:
    missing = [c for c in ("open", "close") if c not in bars.columns]
    if need_high_low:
        missing += [c for c in ("high", "low") if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {sorted(set(missing))}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError(
            "bar index is not sorted; sort before computing returns, otherwise "
            "consecutive-bar returns are meaningless"
        )
    if bars.index.has_duplicates:
        raise ValueError(
            "bar index contains duplicate timestamps; de-duplicate first, since "
            "duplicates inflate realised variance"
        )


def bar_returns(
    bars: pd.DataFrame,
    method: ReturnMethod | str = ReturnMethod.CLOSE_TO_CLOSE,
) -> pd.Series:
    """Per-bar log returns.

    The first close-to-close return of a series is undefined and returned as NaN
    rather than being filled, so that it is excluded from variance sums instead
    of silently contributing a zero.
    """
    method = ReturnMethod(method)
    _validate_bars(bars)

    close = bars["close"].astype(float)
    if (close <= 0).any():
        raise ValueError("non-positive close prices encountered; log returns undefined")

    if method is ReturnMethod.CLOSE_TO_CLOSE:
        out = np.log(close).diff()
    else:
        open_ = bars["open"].astype(float)
        if (open_ <= 0).any():
            raise ValueError("non-positive open prices encountered")
        out = np.log(close) - np.log(open_)

    return out.rename("ret")


def realized_variance(returns: pd.Series) -> float:
    """Sum of squared returns, ignoring NaNs."""
    return float(np.nansum(np.square(returns.to_numpy(dtype=float))))


def parkinson_variance(high: pd.Series, low: pd.Series) -> float:
    """Parkinson (1980) range-based variance estimator.

    Uses the intra-bar high-low range and is roughly five times more efficient
    than squared close-to-close returns under a driftless diffusion. Included as
    a cross-check that is less sensitive to microstructure noise in the closing
    print of each bar.
    """
    h = high.astype(float).to_numpy()
    l = low.astype(float).to_numpy()
    valid = (h > 0) & (l > 0) & np.isfinite(h) & np.isfinite(l)
    if not valid.any():
        return float("nan")
    log_range = np.log(h[valid] / l[valid])
    return float(np.sum(np.square(log_range)) / (4.0 * np.log(2.0)))


def session_variance_shares(
    bars: pd.DataFrame,
    bar_minutes: int,
    spec: SessionSpec | None = None,
    method: ReturnMethod | str = ReturnMethod.CLOSE_TO_CLOSE,
    min_coverage: float = 0.90,
    drop_incomplete_days: bool = True,
) -> pd.DataFrame:
    """Daily realised variance split into US-window and non-US components.

    Parameters
    ----------
    bars
        Intraday OHLC bars with a timezone-aware index.
    bar_minutes
        Bar duration, used for coverage checks and for close-stamped bars.
    spec
        Session definition.
    method
        Return definition; see module docstring.
    min_coverage
        Minimum fraction of a day's expected bars required to keep the day.
    drop_incomplete_days
        Whether to discard days below the coverage floor. Keeping them would
        bias their shares purely through missing data, so this defaults to True.

    Returns
    -------
    DataFrame indexed by Eastern date with the variance decomposition, bar
    counts, day type and ETF primary-market status.
    """
    spec = spec or SessionSpec()
    _validate_bars(bars)

    labels = label_bars(bars.index, spec=spec, bar_minutes=bar_minutes)
    rets = bar_returns(bars, method=method)

    frame = pd.DataFrame(
        {
            "ret": rets.to_numpy(),
            "et_date": labels["et_date"].to_numpy(),
            "in_us_window": labels["in_us_window"].to_numpy(),
            "day_type": labels["day_type"].to_numpy(),
            "etf_primary_open": labels["etf_primary_open"].to_numpy(),
        },
        index=bars.index,
    )
    frame["sq"] = np.square(frame["ret"])
    # Masked columns rather than groupby.apply: the apply form is roughly two
    # orders of magnitude slower over a four-year sample of five-minute bars.
    frame["sq_us"] = frame["sq"].where(frame["in_us_window"])
    frame["sq_non_us"] = frame["sq"].where(~frame["in_us_window"])

    grouped = frame.groupby("et_date", sort=True)
    out = pd.DataFrame(
        {
            "rv_total": grouped["sq"].sum(min_count=1),
            "rv_us": grouped["sq_us"].sum(min_count=1),
            "rv_non_us": grouped["sq_non_us"].sum(min_count=1),
            "n_bars_total": grouped.size(),
            "n_bars_us": grouped["in_us_window"].sum(),
            "day_type": grouped["day_type"].first(),
            "etf_primary_open": grouped["etf_primary_open"].first(),
        }
    )
    out[["rv_us", "rv_non_us"]] = out[["rv_us", "rv_non_us"]].fillna(0.0)

    # A zero-variance day would make the share undefined; leave it as NaN rather
    # than imputing, so that it is excluded from tests rather than treated as 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["share_us"] = np.where(
            out["rv_total"] > 0, out["rv_us"] / out["rv_total"], np.nan
        )

    out.index.name = "et_date"

    if drop_incomplete_days:
        keep = complete_et_days(labels, bar_minutes=bar_minutes, min_coverage=min_coverage)
        dropped = out.index.difference(keep)
        if len(dropped):
            logger.info(
                "Dropping %d day(s) below %.0f%% bar coverage: %s%s",
                len(dropped),
                min_coverage * 100,
                [str(d.date()) for d in dropped[:5]],
                " ..." if len(dropped) > 5 else "",
            )
        out = out.loc[out.index.intersection(keep)]

    return out


def session_range_shares(
    bars: pd.DataFrame,
    bar_minutes: int,
    spec: SessionSpec | None = None,
    min_coverage: float = 0.90,
    drop_incomplete_days: bool = True,
) -> pd.DataFrame:
    """Session decomposition using the Parkinson range estimator.

    Serves as a robustness check on the squared-return decomposition: if the two
    disagree materially, the result is likely driven by microstructure noise in
    bar closes rather than by genuine variance.
    """
    spec = spec or SessionSpec()
    _validate_bars(bars, need_high_low=True)

    labels = label_bars(bars.index, spec=spec, bar_minutes=bar_minutes)
    frame = pd.DataFrame(
        {
            "high": bars["high"].astype(float).to_numpy(),
            "low": bars["low"].astype(float).to_numpy(),
            "et_date": labels["et_date"].to_numpy(),
            "in_us_window": labels["in_us_window"].to_numpy(),
            "day_type": labels["day_type"].to_numpy(),
            "etf_primary_open": labels["etf_primary_open"].to_numpy(),
        },
        index=bars.index,
    )

    scale = 1.0 / (4.0 * np.log(2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sq_log_range = np.square(np.log(frame["high"] / frame["low"]))
    sq_log_range = sq_log_range.replace([np.inf, -np.inf], np.nan)
    frame["pk"] = sq_log_range * scale
    frame["pk_us"] = frame["pk"].where(frame["in_us_window"])
    frame["pk_non_us"] = frame["pk"].where(~frame["in_us_window"])

    grouped = frame.groupby("et_date", sort=True)
    out = pd.DataFrame(
        {
            "pk_total": grouped["pk"].sum(min_count=1),
            "pk_us": grouped["pk_us"].sum(min_count=1),
            "pk_non_us": grouped["pk_non_us"].sum(min_count=1),
            "n_bars_total": grouped.size(),
            "day_type": grouped["day_type"].first(),
            "etf_primary_open": grouped["etf_primary_open"].first(),
        }
    )
    out[["pk_us", "pk_non_us"]] = out[["pk_us", "pk_non_us"]].fillna(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["share_us_pk"] = np.where(
            out["pk_total"] > 0, out["pk_us"] / out["pk_total"], np.nan
        )
    out.index.name = "et_date"

    if drop_incomplete_days:
        keep = complete_et_days(labels, bar_minutes=bar_minutes, min_coverage=min_coverage)
        out = out.loc[out.index.intersection(keep)]

    return out


def share_estimator_bias(
    true_share: float,
    n_window_bars: int,
    n_total_bars: int,
    n_simulations: int = 20_000,
    seed: int = 20260903,
) -> dict[str, float]:
    """Finite-sample bias of the session variance share estimator.

    The share is a ratio of two random sums of squared returns. Even when the
    underlying variance allocation is exactly ``true_share``, the expectation of
    the ratio is not the ratio of expectations, and the estimator is pulled
    toward the *bar-count* share, ``n_window_bars / n_total_bars``.

    The size of the pull depends on how few bars the window contains. With
    five-minute bars the window holds 78 of 288 bars and the bias is negligible.
    With hourly bars it holds only about 6 of 24, and the bias becomes material.
    This is a concrete reason to prefer fine bars beyond the usual appeal to
    estimator efficiency, and it is quantified here rather than assumed away.

    Returns the mean and standard deviation of the estimator, its bias, and the
    bar-count share toward which the bias points.
    """
    if not 0.0 < true_share < 1.0:
        raise ValueError("true_share must lie strictly between 0 and 1")
    if not 0 < n_window_bars < n_total_bars:
        raise ValueError("require 0 < n_window_bars < n_total_bars")

    rng = np.random.default_rng(seed)
    n_out = n_total_bars - n_window_bars
    var_in = true_share / n_window_bars
    var_out = (1.0 - true_share) / n_out

    rv_in = rng.chisquare(n_window_bars, size=n_simulations) * var_in
    rv_out = rng.chisquare(n_out, size=n_simulations) * var_out
    shares = rv_in / (rv_in + rv_out)

    count_share = n_window_bars / n_total_bars
    return {
        "true_share": true_share,
        "mean_estimate": float(shares.mean()),
        "sd_estimate": float(shares.std(ddof=1)),
        "bias": float(shares.mean() - true_share),
        "bar_count_share": count_share,
        "n_window_bars": n_window_bars,
        "n_total_bars": n_total_bars,
    }


def volatility_signature(
    bars: pd.DataFrame,
    sampling_minutes: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60),
) -> pd.DataFrame:
    """Volatility signature plot data: average daily RV against sampling frequency.

    Microstructure noise inflates realised variance as the sampling interval
    shrinks. Plotting average RV against the sampling interval shows where that
    inflation sets in and provides an empirical justification for the chosen bar
    width rather than an arbitrary one. Requires bars at the finest frequency
    available; coarser series are produced by resampling.
    """
    _validate_bars(bars)
    rows = []
    for m in sampling_minutes:
        resampled = bars["close"].resample(f"{m}min").last().dropna()
        if len(resampled) < 3:
            continue
        rets = np.log(resampled).diff()
        daily = rets.pow(2).groupby(rets.index.date).sum()
        rows.append(
            {
                "sampling_minutes": m,
                "mean_daily_rv": float(daily.mean()),
                "median_daily_rv": float(daily.median()),
                "n_days": int(daily.shape[0]),
                "n_bars": int(len(resampled)),
            }
        )
    return pd.DataFrame(rows).set_index("sampling_minutes")
