"""
The analysis pipeline, assembled once so it can be run identically on synthetic
data and on real data.

Keeping a single implementation matters for more than tidiness. If the
validation run and the real run used different code, passing the validation
would say nothing about the correctness of the real result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.inference.breaks import chow_test, placebo_break_scan, sup_wald_test
from src.inference.permutation import holiday_matched_test, weekend_matched_test
from src.inference.regression import add_month_fixed_effects, ols_hac, standardize
from src.measures.realized import ReturnMethod, session_variance_shares


@dataclass
class PipelineOutput:
    """Everything the pipeline produces, for one asset and one dataset."""

    daily: pd.DataFrame
    naive_break: Any
    dose_response: Any | None
    holiday: Any | None
    weekend: Any | None
    sup_wald: Any | None
    placebos: pd.DataFrame | None
    diagnostics: dict = field(default_factory=dict)

    def headline(self) -> dict:
        """Compact summary of the tests that matter, for tabular comparison."""
        row: dict[str, Any] = {
            "n_days": len(self.daily),
            "mean_share_pre": self.naive_break.mean_pre,
            "mean_share_post": self.naive_break.mean_post,
            "naive_diff": self.naive_break.diff,
            "naive_p": self.naive_break.p_value_hac,
        }
        if self.dose_response is not None:
            c = self.dose_response.coef("flow_z")
            row["dose_coef"] = c["coef"]
            row["dose_p"] = c["p"]
        if self.holiday is not None:
            row["holiday_did"] = self.holiday.did_statistic
            row["holiday_p"] = self.holiday.permutation.p_value
            row["holiday_n_post"] = self.holiday.n_post
        if self.weekend is not None:
            row["weekend_did"] = self.weekend.did_statistic
            row["weekend_p"] = self.weekend.permutation.p_value
        if self.sup_wald is not None:
            row["supwald_date"] = self.sup_wald.argmax_date.date()
            row["supwald_days_off"] = self.sup_wald.days_from_hypothesis
            row["supwald_p"] = self.sup_wald.p_value_bootstrap
        return row


def build_daily_panel(
    bars: pd.DataFrame,
    bar_minutes: int,
    etf_flow: pd.Series | None = None,
    method: ReturnMethod | str = ReturnMethod.CLOSE_TO_CLOSE,
    min_coverage: float = 0.90,
    exclude_early_close: bool = True,
) -> pd.DataFrame:
    """Session variance shares plus day metadata and, if supplied, ETF flow."""
    daily = session_variance_shares(
        bars,
        bar_minutes=bar_minutes,
        method=method,
        min_coverage=min_coverage,
    )
    if exclude_early_close:
        n_early_close = int((daily["day_type"] == "early_close").sum())
        daily = daily.loc[daily["day_type"] != "early_close"].copy()
        daily.attrs["n_early_close_excluded"] = n_early_close
    daily["is_weekday"] = daily.index.dayofweek < 5
    if etf_flow is not None:
        daily["etf_net_flow"] = etf_flow.reindex(daily.index)
    return daily


def run_pipeline(
    bars: pd.DataFrame,
    event_date: str | pd.Timestamp,
    bar_minutes: int = 5,
    etf_flow: pd.Series | None = None,
    method: ReturnMethod | str = ReturnMethod.CLOSE_TO_CLOSE,
    n_permutations: int = 5_000,
    n_bootstrap: int = 400,
    run_sup_wald: bool = True,
    run_placebos: bool = True,
    sup_wald_freq: str = "D",
) -> PipelineOutput:
    """Run the full set of tests on one asset's bars.

    The order here mirrors the order of the argument in the report: describe,
    then run the weak benchmark, then the tests that actually identify the
    mechanism.
    """
    event = pd.Timestamp(event_date)
    daily = build_daily_panel(
        bars, bar_minutes=bar_minutes, etf_flow=etf_flow, method=method
    )
    share = daily["share_us"].dropna()

    diagnostics = {
        "n_days_total": len(daily),
        "n_days_usable_share": int(share.notna().sum()),
        "n_regular": int((daily["day_type"] == "regular").sum()),
        "n_early_close": int(
            daily.attrs.get(
                "n_early_close_excluded",
                (daily["day_type"] == "early_close").sum(),
            )
        ),
        "n_holiday": int((daily["day_type"] == "holiday").sum()),
        "n_weekend": int((daily["day_type"] == "weekend").sum()),
        "share_mean": float(share.mean()),
        "share_sd": float(share.std()),
        "share_autocorr_1": float(share.autocorr(1)) if len(share) > 2 else np.nan,
    }

    # Weak benchmark: a level shift at the date we chose. Confounded by design.
    naive = chow_test(share, event)

    # Primary test: dose-response on flow, with month fixed effects so the
    # coefficient comes from within-month variation rather than from a
    # calm-year-versus-volatile-year comparison.
    dose = None
    if etf_flow is not None:
        post = daily.loc[daily.index >= event].copy()
        post = post.dropna(subset=["share_us", "etf_net_flow", "rv_total"])
        post = post[post["etf_net_flow"] > 0]
        if len(post) > 60 and post["etf_net_flow"].std(ddof=1) > 0:
            X = pd.DataFrame(
                {
                    "flow_z": standardize(post["etf_net_flow"]),
                    "log_rv": np.log(post["rv_total"].clip(lower=1e-12)),
                },
                index=post.index,
            )
            X = add_month_fixed_effects(X, post.index)
            dose = ols_hac(post["share_us"], X)

    # Identification: days on which crypto trades but ETF creation does not.
    holiday = None
    try:
        holiday = holiday_matched_test(
            daily, event, n_permutations=n_permutations, alternative="less"
        )
    except ValueError as exc:
        diagnostics["holiday_test_error"] = str(exc)

    weekend = None
    try:
        weekend = weekend_matched_test(
            daily, event, n_permutations=n_permutations, alternative="less"
        )
    except ValueError as exc:
        diagnostics["weekend_test_error"] = str(exc)

    # Does the data pick our date without being told where to look?
    #
    # Frequency matters more than it might appear. Aggregating to monthly means
    # leaves only about forty-eight observations, and the persistence-aware
    # block bootstrap is then so conservative that even a large injected break
    # is not detected. The daily series is noisier per observation but has
    # thirty times as many, and validation on synthetic data showed it to be far
    # better powered. Daily is therefore the default, with monthly retained as a
    # smoothed cross-check.
    sup = None
    if run_sup_wald:
        series = share if sup_wald_freq == "D" else share.resample(sup_wald_freq).mean()
        try:
            sup = sup_wald_test(
                series.dropna(),
                n_bootstrap=n_bootstrap,
                hypothesised_date=event,
            )
        except ValueError as exc:
            diagnostics["sup_wald_error"] = str(exc)

    placebos = None
    if run_placebos:
        try:
            placebos = placebo_break_scan(share, event)
        except ValueError as exc:
            diagnostics["placebo_error"] = str(exc)

    return PipelineOutput(
        daily=daily,
        naive_break=naive,
        dose_response=dose,
        holiday=holiday,
        weekend=weekend,
        sup_wald=sup,
        placebos=placebos,
        diagnostics=diagnostics,
    )


def scenario_comparison(outputs: dict[str, PipelineOutput]) -> pd.DataFrame:
    """Side-by-side headline results across scenarios or assets."""
    rows = {name: out.headline() for name, out in outputs.items()}
    return pd.DataFrame(rows).T
