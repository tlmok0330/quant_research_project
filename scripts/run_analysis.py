"""Run the minimum pre-registered BTC analysis and save reproducible outputs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURE_DIR, TABLE_DIR
from src.data.external import load_etf_flows
from src.data.quality import audit_bars
from src.measures.windows import window_variance_shares, within_day_placebo
from src.validation.pipeline import run_pipeline

EVENT = pd.Timestamp("2024-01-11")
BAR_MINUTES = 5
DATA = (
    Path(__file__).resolve().parents[1]
    / "data/raw/crypto_ohlcv_BTCUSD_5min_20220101_20251231.parquet"
)


def main() -> int:
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found; fetch BTC history first")

    bars = pd.read_parquet(DATA)
    quality = audit_bars(bars, BAR_MINUTES)
    quality.table().to_csv(TABLE_DIR / "btc_data_quality.csv", index=False)
    quality.assert_usable()

    flow = load_etf_flows("btc").abs_flow
    active = flow[flow > 0]
    cap = float(active.quantile(0.99))
    dose = flow.clip(upper=cap)

    output = run_pipeline(
        bars,
        EVENT,
        BAR_MINUTES,
        dose,
        n_permutations=5_000,
        n_bootstrap=400,
        run_sup_wald=True,
        run_placebos=True,
    )

    headline = pd.DataFrame([output.headline()], index=["BTC"])
    headline["dose_p_one_sided"] = (
        headline["dose_p"] / 2
        if headline["dose_coef"].iloc[0] > 0
        else 1 - headline["dose_p"] / 2
    )
    headline.to_csv(TABLE_DIR / "btc_headline_results.csv")
    output.daily.to_parquet(
        DATA.parents[1] / "processed/btc_daily_session_measures.parquet"
    )
    if output.placebos is not None:
        output.placebos.to_csv(TABLE_DIR / "btc_placebo_dates.csv")
    if output.dose_response is not None:
        output.dose_response.table().to_csv(TABLE_DIR / "btc_dose_response.csv")

    windows = window_variance_shares(bars, BAR_MINUTES)
    t9 = within_day_placebo(windows, EVENT)
    t9.summary().to_csv(TABLE_DIR / "btc_t9_cme_placebo.csv")

    monthly = output.daily["share_us"].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly, lw=1.5)
    ax.axvline(EVENT, color="black", ls="--", lw=1, label="Spot ETF launch")
    ax.axhline(
        output.naive_break.mean_pre,
        color="tab:blue",
        ls=":",
        lw=1,
        label="Pre-event mean",
    )
    ax.set(title="BTC variance realised during 09:30–16:00 ET", ylabel="Monthly mean share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "btc_monthly_session_share.png", dpi=160)
    plt.close(fig)

    print("\nBTC HEADLINE RESULTS")
    print(headline.T.to_string(header=False))
    print("\nT9 ETF-VERSUS-CME PLACEBO")
    print(t9.summary().to_string())
    print(f"\nT9 verdict: {t9.verdict}")
    print(f"Flow winsorisation cap: USD {cap:.2f}m")
    print(f"Quality: {len(quality.critical_failures)} critical failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
