"""
Difference-in-differences with staggered treatment timing.

Design
------
Three assets, treated at different times:

    BTC   treated from 11 January 2024
    ETH   treated from 23 July 2024
    a large-cap coin with no US spot ETF, never treated

With daily time fixed effects, identification comes from comparing treated and
untreated assets *on the same day*, which absorbs every crypto-wide shock:
a risk-off day, a macro print, an exchange outage. That is the main thing a
before-and-after comparison on Bitcoin alone cannot do.

The honest limitation
---------------------
Three units is very few. Cluster-robust standard errors require many clusters
and are not trustworthy here; reporting them without comment would be
misleading. Two alternatives are provided instead.

``randomization_inference`` assigns placebo treatment dates and rebuilds the
null distribution empirically, which is valid with few units because it does
not rely on asymptotics in the number of clusters.

``event_study`` estimates coefficients by period relative to treatment. Its
purpose is to test the parallel-trends assumption directly: if the pre-treatment
coefficients already trend, the design is not credible and the point estimate
should not be believed regardless of its p-value.

The halving caveat
------------------
The April 2024 halving is specific to Bitcoin and is therefore *not* absorbed by
the control asset. The Ethereum leg of the comparison is cleaner for this
reason and should be weighted accordingly in interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DiDResult:
    coefficient: float
    std_err_cluster: float | None
    n_clusters: int
    nobs: int
    n_treated_obs: int
    within_r2: float
    time_fe_freq: str
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover
        s = (
            f"DiD coefficient = {self.coefficient:+.5f}  "
            f"N = {self.nobs:,} ({self.n_treated_obs:,} treated)  "
            f"units = {self.n_clusters}  within R2 = {self.within_r2:.4f}"
        )
        if self.note:
            s += f"\n{self.note}"
        return s


@dataclass
class RandomizationResult:
    observed: float
    p_value: float
    n_draws: int
    null_mean: float
    null_sd: float
    alternative: str
    null_dist: np.ndarray = field(repr=False)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"observed = {self.observed:+.5f}  randomisation p = {self.p_value:.4g} "
            f"({self.n_draws:,} placebo draws; null mean {self.null_mean:+.5f}, "
            f"sd {self.null_sd:.5f})"
        )


def build_panel(
    series_by_asset: dict[str, pd.Series],
    treat_dates: dict[str, pd.Timestamp | str | None],
) -> pd.DataFrame:
    """Assemble a long-format panel with a treatment indicator.

    Parameters
    ----------
    series_by_asset
        Mapping of asset name to a date-indexed outcome series.
    treat_dates
        Mapping of asset name to its treatment date, or ``None`` for a
        never-treated control.
    """
    frames = []
    for asset, s in series_by_asset.items():
        if asset not in treat_dates:
            raise ValueError(f"no treatment date supplied for {asset!r}")
        td = treat_dates[asset]
        frame = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(s.index),
                "unit": asset,
                "y": s.to_numpy(dtype=float),
            }
        )
        if td is None:
            frame["treated"] = 0.0
            frame["treat_date"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        else:
            td = pd.Timestamp(td)
            frame["treated"] = (frame["date"] >= td).astype(float)
            frame["treat_date"] = pd.Series(td, index=frame.index, dtype="datetime64[ns]")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    return panel.sort_values(["unit", "date"]).reset_index(drop=True)


def _two_way_demean(
    values: np.ndarray,
    unit_codes: np.ndarray,
    time_codes: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """Absorb unit and time fixed effects by alternating projection.

    Exact in one pass for a balanced panel, and converges quickly otherwise.
    Used instead of building explicit dummy matrices so that randomisation
    inference over many placebo draws remains fast.
    """
    v = values.astype(float).copy()
    n_units = unit_codes.max() + 1
    n_times = time_codes.max() + 1
    for _ in range(max_iter):
        before = v.copy()
        unit_sum = np.bincount(unit_codes, weights=v, minlength=n_units)
        unit_cnt = np.bincount(unit_codes, minlength=n_units)
        v -= np.where(unit_cnt > 0, unit_sum / np.maximum(unit_cnt, 1), 0.0)[unit_codes]

        time_sum = np.bincount(time_codes, weights=v, minlength=n_times)
        time_cnt = np.bincount(time_codes, minlength=n_times)
        v -= np.where(time_cnt > 0, time_sum / np.maximum(time_cnt, 1), 0.0)[time_codes]

        if np.max(np.abs(v - before)) < tol:
            break
    return v


def _prep_codes(panel: pd.DataFrame, time_fe_freq: str) -> tuple[np.ndarray, np.ndarray]:
    unit_codes = pd.Categorical(panel["unit"]).codes.astype(np.int64)
    if time_fe_freq.upper() in ("D", "DAY", "DAILY"):
        time_key = panel["date"]
    else:
        time_key = pd.DatetimeIndex(panel["date"]).to_period(time_fe_freq).astype(str)
    time_codes = pd.Categorical(time_key).codes.astype(np.int64)
    return unit_codes, time_codes


def staggered_did(
    panel: pd.DataFrame,
    time_fe_freq: str = "D",
    outcome_col: str = "y",
    treat_col: str = "treated",
) -> DiDResult:
    """Two-way fixed-effects estimate of the average treatment effect.

    Notes
    -----
    A single pooled coefficient on a staggered-treatment indicator can be biased
    when treatment effects vary over time, because already-treated units act as
    controls for later-treated ones (Goodman-Bacon 2021). With one never-treated
    control and only two treated units the exposure is limited, but the event
    study should still be inspected before relying on this number.
    """
    required = {outcome_col, treat_col, "unit", "date"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    df = panel.dropna(subset=[outcome_col, treat_col]).copy()
    if df.empty:
        raise ValueError("panel has no usable rows")

    unit_codes, time_codes = _prep_codes(df, time_fe_freq)
    y_t = _two_way_demean(df[outcome_col].to_numpy(), unit_codes, time_codes)
    d_t = _two_way_demean(df[treat_col].to_numpy(), unit_codes, time_codes)

    denom = float(np.sum(d_t**2))
    if denom <= 1e-12:
        raise ValueError(
            "treatment indicator has no variation left after absorbing fixed "
            "effects; with a single treated unit and time fixed effects the "
            "coefficient is not identified"
        )
    beta = float(np.sum(d_t * y_t) / denom)

    resid = y_t - beta * d_t
    ss_tot = float(np.sum(y_t**2))
    within_r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")

    n_clusters = int(df["unit"].nunique())
    se_cluster: float | None
    try:
        cluster_meat = 0.0
        for _, idx in df.groupby("unit").indices.items():
            cluster_meat += float(np.sum(d_t[idx] * resid[idx])) ** 2
        se_cluster = float(np.sqrt(cluster_meat) / denom)
    except Exception:  # pragma: no cover - defensive
        se_cluster = None

    note = ""
    if n_clusters < 20:
        note = (
            f"Only {n_clusters} clusters. The cluster-robust standard error is "
            "reported for completeness but is downward biased at this cluster "
            "count; use randomization_inference for inference."
        )

    return DiDResult(
        coefficient=beta,
        std_err_cluster=se_cluster,
        n_clusters=n_clusters,
        nobs=len(df),
        n_treated_obs=int(df[treat_col].sum()),
        within_r2=within_r2,
        time_fe_freq=time_fe_freq,
        note=note,
    )


def event_study(
    panel: pd.DataFrame,
    bin_days: int = 30,
    max_bins: int = 6,
    outcome_col: str = "y",
    time_fe_freq: str = "D",
    reference_bin: int = -1,
) -> pd.DataFrame:
    """Coefficients by period relative to treatment, for a parallel-trends check.

    Relative time is binned (default thirty-day bins) because daily relative-time
    dummies would be far too noisy with three units.

    The pre-treatment bins are the point of this exercise. If they are
    indistinguishable from zero, the parallel-trends assumption is supported. If
    they trend, the design is not credible and the pooled estimate should not be
    interpreted causally regardless of its significance.
    """
    df = panel.dropna(subset=[outcome_col]).copy()
    treated_units = df["treat_date"].notna()
    if not treated_units.any():
        raise ValueError("no treated unit in panel")

    rel_days = (df["date"] - df["treat_date"]).dt.days
    rel_bin = np.floor(rel_days / bin_days)
    rel_bin = rel_bin.clip(lower=-max_bins, upper=max_bins - 1)
    # Never-treated units contribute only through the time fixed effects.
    df["rel_bin"] = rel_bin
    df.loc[df["treat_date"].isna(), "rel_bin"] = np.nan

    bins = sorted(b for b in df["rel_bin"].dropna().unique() if b != reference_bin)
    dummies = {}
    for b in bins:
        dummies[f"rel_{int(b):+d}"] = (df["rel_bin"] == b).astype(float).to_numpy()

    if not dummies:
        raise ValueError("no relative-time bins other than the reference bin")

    unit_codes, time_codes = _prep_codes(df, time_fe_freq)
    y_t = _two_way_demean(df[outcome_col].to_numpy(), unit_codes, time_codes)
    cols, mats = [], []
    for name, d in dummies.items():
        cols.append(name)
        mats.append(_two_way_demean(d, unit_codes, time_codes))
    X = np.column_stack(mats)

    # Least squares on the demeaned system.
    beta, *_ = np.linalg.lstsq(X, y_t, rcond=None)
    resid = y_t - X @ beta
    dof = max(1, X.shape[0] - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))

    out = pd.DataFrame(
        {
            "bin": cols,
            "rel_bin": [int(b) for b in bins],
            "coef": beta,
            "std_err": se,
        }
    )
    out["t"] = out["coef"] / out["std_err"].replace(0, np.nan)
    out["is_pre_treatment"] = out["rel_bin"] < 0
    out["days_from_treatment"] = out["rel_bin"] * bin_days
    return out.sort_values("rel_bin").reset_index(drop=True)


def randomization_inference(
    series_by_asset: dict[str, pd.Series],
    true_treat_dates: dict[str, pd.Timestamp | str | None],
    n_draws: int = 1_000,
    time_fe_freq: str = "D",
    alternative: str = "greater",
    min_side_days: int = 90,
    seed: int = 20260903,
) -> RandomizationResult:
    """Placebo-timing randomisation inference for the DiD coefficient.

    Rather than permuting which unit is treated, which offers only a handful of
    distinct assignments with three units, this draws random placebo treatment
    dates for the treated units and rebuilds the null distribution. That yields
    a valid p-value without relying on a large number of clusters.

    ``min_side_days`` keeps placebo dates away from the sample edges so that
    every draw has a usable pre- and post-period.
    """
    observed_panel = build_panel(series_by_asset, true_treat_dates)
    observed = staggered_did(observed_panel, time_fe_freq=time_fe_freq).coefficient

    treated_assets = [a for a, d in true_treat_dates.items() if d is not None]
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[set(s.dropna().index) for s in series_by_asset.values()]))
    )
    if len(all_dates) < 2 * min_side_days + 10:
        raise ValueError("sample too short for randomisation inference")
    candidates = all_dates[min_side_days:-min_side_days]

    rng = np.random.default_rng(seed)
    null = np.empty(n_draws)
    n_valid = 0
    for i in range(n_draws):
        placebo = dict(true_treat_dates)
        for a in treated_assets:
            placebo[a] = candidates[rng.integers(0, len(candidates))]
        try:
            p = build_panel(series_by_asset, placebo)
            null[n_valid] = staggered_did(p, time_fe_freq=time_fe_freq).coefficient
            n_valid += 1
        except ValueError:
            continue
    null = null[:n_valid]
    if n_valid < 50:
        raise ValueError(f"only {n_valid} valid placebo draws; cannot form a null")

    if alternative == "greater":
        count = np.sum(null >= observed)
    elif alternative == "less":
        count = np.sum(null <= observed)
    else:
        count = np.sum(np.abs(null) >= abs(observed))
    p_value = float((1.0 + count) / (1.0 + n_valid))

    return RandomizationResult(
        observed=observed,
        p_value=p_value,
        n_draws=n_valid,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        alternative=alternative,
        null_dist=null,
    )
