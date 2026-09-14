"""
Acquire daily US spot crypto ETF net flows and cache them locally.

Run:
    python scripts/fetch_etf_flows.py            # BTC and ETH
    python scripts/fetch_etf_flows.py --asset btc
    python scripts/fetch_etf_flows.py --inspect  # show what is cached

Source and provenance
---------------------
Daily net creation and redemption flows for the US spot ETFs are published by
Farside Investors at https://farside.co.uk/. This is a secondary source that
aggregates issuer disclosures. It is used because there is no single official
consolidated feed: each issuer reports its own shares outstanding, and
assembling those directly would mean reconciling a dozen filings per day for
two years.

Two consequences follow and are disclosed in the report rather than buried.

First, the flow series is a *secondary* source and may contain revisions or
errors. Its role is as the explanatory variable in a dose-response test, so a
small amount of measurement error attenuates the estimated coefficient toward
zero. That biases against finding an effect, which is the safe direction for a
confirmatory test.

Second, this creates a dependency on a third party outside the Massive API. A
substitute that lives entirely inside Massive is therefore also constructed:
aggregate dollar volume of the ETF tickers themselves, from the equities feed.
The two are used as alternative measures of the same underlying concept, and the
result should not depend on which is chosen. See ``docs/pre_analysis_plan.md``.

The raw HTML is cached on first download so that the analysis is reproducible
from a fixed snapshot and the site is queried once rather than repeatedly.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.config import EXTERNAL_DIR

SOURCES = {
    "btc": {
        "url": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
        "launch": "2024-01-11",
        "label": "US spot Bitcoin ETFs",
    },
    "eth": {
        "url": "https://farside.co.uk/ethereum-etf-flow-all-data/",
        "launch": "2024-07-23",
        "label": "US spot Ethereum ETFs",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research/1.0; "
        "single-fetch-then-cached)"
    )
}


def cache_paths(asset: str) -> tuple[Path, Path, Path]:
    raw = EXTERNAL_DIR / f"farside_{asset}_raw.html"
    wide = EXTERNAL_DIR / f"etf_flows_{asset}_by_fund.csv"
    daily = EXTERNAL_DIR / f"etf_flows_{asset}_daily.csv"
    return raw, wide, daily


def download_html(asset: str, force: bool = False) -> str:
    raw_path, _, _ = cache_paths(asset)
    if raw_path.exists() and not force:
        print(f"  using cached HTML: {raw_path.name} "
              f"({raw_path.stat().st_size:,} bytes)")
        return raw_path.read_text(encoding="utf-8", errors="replace")

    url = SOURCES[asset]["url"]
    print(f"  downloading {url}")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    raw_path.write_text(response.text, encoding="utf-8")
    print(f"  cached {len(response.content):,} bytes to {raw_path.name}")
    return response.text


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Parse flow figures, which use parentheses for negatives.

    Farside renders outflows as ``(123.4)`` rather than ``-123.4``. Treating a
    parenthesised value as positive would invert the sign of every outflow day,
    which would silently destroy the dose-response test, so this is handled
    explicitly and asserted in the tests.
    """
    text = series.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    negative = text.str.startswith("(") & text.str.endswith(")")
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    text = text.replace({"-": None, "": None, "nan": None, "None": None})
    values = pd.to_numeric(text, errors="coerce")
    return values.where(~negative, -values)


def parse_flow_table(html: str, asset: str) -> pd.DataFrame:
    """Extract the per-fund daily flow table from the cached page."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError("no HTML tables found on the page; layout may have changed")

    # Choose the widest table that has a parseable date column: the per-fund
    # daily table. Selecting by shape rather than position survives the site
    # adding or reordering summary tables.
    best: pd.DataFrame | None = None
    for table in tables:
        if table.shape[0] < 100 or table.shape[1] < 3:
            continue
        frame = table.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                " ".join(str(p) for p in col if "Unnamed" not in str(p)).strip()
                for col in frame.columns
            ]
        first = frame.columns[0]
        dates = pd.to_datetime(frame[first], errors="coerce", format="mixed")
        if dates.notna().sum() < 100:
            continue
        if best is None or frame.shape[1] > best.shape[1]:
            frame["__date__"] = dates
            best = frame

    if best is None:
        raise ValueError(
            "could not identify the daily flow table; inspect the cached HTML "
            f"at {cache_paths(asset)[0]}"
        )

    frame = best.dropna(subset=["__date__"]).copy()
    frame = frame.set_index("__date__").sort_index()
    frame.index.name = "date"
    frame = frame.drop(columns=[frame.columns[0]])

    for col in frame.columns:
        frame[col] = _clean_numeric(frame[col])

    frame = frame.dropna(axis=1, how="all")
    return frame


def build_daily_series(by_fund: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Collapse per-fund flows into the daily aggregate used in the analysis."""
    total_cols = [c for c in by_fund.columns if str(c).strip().lower() == "total"]
    if total_cols:
        total = by_fund[total_cols[0]]
        source = "reported Total column"
    else:
        exclude = {"total", "average", "maximum", "minimum"}
        fund_cols = [c for c in by_fund.columns if str(c).strip().lower() not in exclude]
        total = by_fund[fund_cols].sum(axis=1, min_count=1)
        source = f"sum across {len(fund_cols)} fund columns"

    out = pd.DataFrame({"net_flow_usd_m": total})
    out["abs_flow_usd_m"] = out["net_flow_usd_m"].abs()
    out["is_inflow"] = out["net_flow_usd_m"] > 0
    out["cumulative_flow_usd_m"] = out["net_flow_usd_m"].cumsum()
    out.attrs["total_source"] = source
    out.attrs["asset"] = asset
    return out


def summarise(daily: pd.DataFrame, by_fund: pd.DataFrame, asset: str) -> None:
    launch = pd.Timestamp(SOURCES[asset]["launch"])
    print(f"\n  {SOURCES[asset]['label']}")
    print(f"    funds/columns parsed : {len(by_fund.columns)}")
    print(f"    trading days         : {len(daily):,}")
    print(f"    date range           : {daily.index.min().date()} to "
          f"{daily.index.max().date()}")
    print(f"    expected launch date : {launch.date()}")
    print(f"    first data date      : {daily.index.min().date()}")
    print(f"    net flow, total      : "
          f"${daily['net_flow_usd_m'].sum() / 1000:,.1f}bn")
    print(f"    largest inflow day   : ${daily['net_flow_usd_m'].max():,.1f}m "
          f"on {daily['net_flow_usd_m'].idxmax().date()}")
    print(f"    largest outflow day  : ${daily['net_flow_usd_m'].min():,.1f}m "
          f"on {daily['net_flow_usd_m'].idxmin().date()}")
    print(f"    outflow days         : {int((~daily['is_inflow']).sum()):,} "
          f"of {len(daily):,} "
          f"({(~daily['is_inflow']).mean():.1%})")
    print(f"    flow autocorrelation : "
          f"{daily['net_flow_usd_m'].autocorr(1):.3f}")

    # Sanity checks that would catch a parsing failure.
    issues = []
    if daily.index.min() < launch - pd.Timedelta(days=5).to_pytimedelta():
        issues.append(
            f"data begins {daily.index.min().date()}, well before the launch"
        )
    if not (~daily["is_inflow"]).any():
        issues.append(
            "no outflow days found; parenthesised negatives may not have parsed"
        )
    if daily.index.has_duplicates:
        issues.append("duplicate dates present")
    weekend = daily.index.dayofweek >= 5
    if weekend.any():
        issues.append(f"{int(weekend.sum())} weekend date(s) present, which is "
                      "unexpected for ETF primary-market flow")
    if issues:
        print("    WARNINGS:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("    sanity checks: passed")


def process(asset: str, force: bool = False) -> pd.DataFrame:
    print(f"\n[{asset.upper()}]")
    html = download_html(asset, force=force)
    by_fund = parse_flow_table(html, asset)
    daily = build_daily_series(by_fund, asset)

    _, wide_path, daily_path = cache_paths(asset)
    by_fund.to_csv(wide_path)
    daily.to_csv(daily_path)
    print(f"  wrote {wide_path.name} and {daily_path.name}")
    summarise(daily, by_fund, asset)
    return daily


def write_provenance(assets: list[str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# External data provenance",
        "",
        f"Generated by `scripts/fetch_etf_flows.py` at {stamp}.",
        "",
        "## US spot crypto ETF net flows",
        "",
        "| asset | source | retrieved | local files |",
        "| --- | --- | --- | --- |",
    ]
    for asset in assets:
        _, wide, daily = cache_paths(asset)
        lines.append(
            f"| {asset.upper()} | {SOURCES[asset]['url']} | {stamp} | "
            f"`{wide.name}`, `{daily.name}` |"
        )
    lines += [
        "",
        "### Notes on use",
        "",
        "- Figures are in US$ millions of net creation less redemption, as "
        "reported by Farside Investors, which aggregates issuer disclosures.",
        "- Outflows are rendered in parentheses on the source page and are "
        "converted to negative numbers on parse. This is verified by a test, "
        "because an unnoticed sign inversion would invalidate the "
        "dose-response result.",
        "- This is a secondary source. Measurement error in an explanatory "
        "variable attenuates the estimated coefficient toward zero, so it "
        "biases against the hypothesis rather than for it.",
        "- The raw HTML snapshot is retained so that results are reproducible "
        "from fixed inputs even if the site changes.",
        "- A substitute measure built entirely from the Massive equities feed, "
        "namely aggregate ETF dollar volume, is used as a robustness check so "
        "that no conclusion rests on a single external dependency.",
        "",
    ]
    path = EXTERNAL_DIR / "PROVENANCE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["btc", "eth", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="re-download the HTML")
    parser.add_argument("--inspect", action="store_true", help="list cached files")
    args = parser.parse_args()

    print("=" * 78)
    print("US SPOT CRYPTO ETF FLOW ACQUISITION")
    print("=" * 78)

    if args.inspect:
        for file in sorted(EXTERNAL_DIR.glob("*")):
            print(f"  {file.name:40s} {file.stat().st_size:>12,} bytes")
        return 0

    assets = ["btc", "eth"] if args.asset == "both" else [args.asset]
    failures = []
    for asset in assets:
        try:
            process(asset, force=args.force)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED for {asset}: {type(exc).__name__}: {exc}")
            failures.append(asset)

    ok = [a for a in assets if a not in failures]
    if ok:
        path = write_provenance(ok)
        print(f"\nProvenance written to {path}")

    print("\n" + "=" * 78)
    if failures:
        print(f"Completed with failures for: {failures}")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
