"""
Data acquisition and caching layer.
Downloads from Massive API and saves to data/raw/ as parquet for fast reuse.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.api.client import MassiveClient

logger = logging.getLogger(__name__)
RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"


def load_or_fetch(name: str, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    """Return cached parquet if it exists, otherwise call fetch_fn and cache result."""
    path = _cache_path(name)
    if path.exists() and not force_refresh:
        logger.info("Loading %s from cache", name)
        return pd.read_parquet(path)
    logger.info("Fetching %s from API", name)
    df = fetch_fn()
    df.to_parquet(path, index=True)
    return df


def fetch_crypto_ohlcv(
    client: MassiveClient,
    symbol: str,
    start: str,
    end: str,
    freq: str = "5min",
    force_refresh: bool = False,
) -> pd.DataFrame:
    name = f"crypto_ohlcv_{symbol}_{freq}_{start}_{end}".replace("-", "")

    def _fetch() -> pd.DataFrame:
        return client.get_crypto_ohlcv(symbol, start, end, freq)

    return load_or_fetch(name, _fetch, force_refresh)


def fetch_funding_rates(
    client: MassiveClient,
    symbol: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    name = f"funding_rates_{symbol}_{start}_{end}".replace("-", "")

    def _fetch() -> pd.DataFrame:
        raw = client.get_funding_rates(symbol, start, end)
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    return load_or_fetch(name, _fetch, force_refresh)
