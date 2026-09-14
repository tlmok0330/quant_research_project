"""Download and cache the BTC data used by the report.

The API key is read from .env. Raw licensed data is saved under data/raw and is
excluded from git.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.client import MassiveClient
from src.data.loader import fetch_crypto_ohlcv


def main() -> int:
    client = MassiveClient()
    bars = fetch_crypto_ohlcv(
        client=client,
        symbol="BTCUSD",
        start="2022-01-01",
        end="2025-12-31",
        freq="5min",
    )
    print(f"Downloaded or loaded {len(bars):,} bars")
    print(f"First timestamp: {bars.index.min()}")
    print(f"Last timestamp:  {bars.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
