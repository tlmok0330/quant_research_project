"""Realised-variance, session-share and named-window estimators."""

from src.measures.realized import (
    ReturnMethod,
    bar_returns,
    parkinson_variance,
    realized_variance,
    session_range_shares,
    session_variance_shares,
    share_estimator_bias,
    volatility_signature,
)
from src.measures.variance_ratio import variance_ratio, variance_ratio_by_group
from src.measures.windows import (
    DAY_PARTITION,
    PLACEBO_WINDOW,
    TREATED_WINDOW,
    NamedWindow,
    WithinDayPlaceboResult,
    per_minute_intensity,
    validate_partition,
    window_variance_shares,
    within_day_placebo,
)

__all__ = [
    "DAY_PARTITION",
    "NamedWindow",
    "PLACEBO_WINDOW",
    "ReturnMethod",
    "TREATED_WINDOW",
    "WithinDayPlaceboResult",
    "bar_returns",
    "parkinson_variance",
    "per_minute_intensity",
    "realized_variance",
    "session_range_shares",
    "session_variance_shares",
    "share_estimator_bias",
    "validate_partition",
    "variance_ratio",
    "variance_ratio_by_group",
    "volatility_signature",
    "window_variance_shares",
    "within_day_placebo",
]
