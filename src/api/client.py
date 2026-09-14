"""
HTTP client for the Massive REST API.

Design priorities, in order
---------------------------
1. The key must never leave the machine in a log, a traceback, a cached file or
   a committed notebook. Every outbound URL and every error message passes
   through a redaction filter, and the key is read from the environment at call
   time rather than stored on the instance in a form that a repr would print.

2. Requests must be reproducible. Raw responses are cached to disk keyed by a
   hash of the request, so an analysis can be re-run without re-querying, and
   so the exact bytes behind a published number can be inspected later.

3. Failures must be informative rather than silent. Transient errors are
   retried with exponential backoff; permanent ones raise with the status code
   and a redacted URL.

The endpoint paths are deliberately not hard-coded. At the time of writing the
API documentation has not been seen, and inventing plausible-looking paths would
produce code that appears finished but cannot work. Paths are therefore supplied
through :class:`EndpointConfig`, which is filled in once the documentation is
read, and the probe script exists to discover which of several candidate
patterns is correct.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from pandas.errors import OutOfBoundsDatetime

from src.config import RAW_DIR, get_api_key, get_base_url

logger = logging.getLogger(__name__)

_CACHE_DIR = RAW_DIR / "api_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Query-parameter names that commonly carry credentials.
_SECRET_PARAM_NAMES = ("apikey", "api_key", "apiKey", "token", "key", "access_token")


def redact(text: str, secret: str | None = None) -> str:
    """Remove credentials from a string before it reaches a log or an exception.

    Redacts both the actual key, when supplied, and anything that looks like a
    credential in a query string, so that a URL logged from an unexpected code
    path is still safe.
    """
    out = str(text)
    if secret:
        out = out.replace(secret, "***REDACTED***")
    for name in _SECRET_PARAM_NAMES:
        out = re.sub(
            rf"({re.escape(name)}=)[^&\s\"']+",
            r"\1***REDACTED***",
            out,
            flags=re.IGNORECASE,
        )
    # Bearer tokens in headers or messages.
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***REDACTED***", out)
    return out


class MassiveAPIError(RuntimeError):
    """Raised on a non-retryable API failure, with credentials removed."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(redact(message))
        self.status_code = status_code


@dataclass
class EndpointConfig:
    """Endpoint paths and parameter names, to be filled in from the documentation.

    Left empty by default on purpose. A wrong-but-plausible default would fail
    at run time in a way that looks like a network problem; an empty one fails
    immediately with a clear message.
    """

    crypto_aggregates: str = (
        "v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
    )
    crypto_trades: str = ""
    funding_rates: str = ""
    open_interest: str = ""
    equity_aggregates: str = ""
    fx_aggregates: str = ""
    options_chain: str = ""
    options_aggregates: str = ""
    reference_tickers: str = ""

    # Parameter naming varies widely between providers.
    param_symbol: str = "symbol"
    param_start: str = "start"
    param_end: str = "end"
    param_interval: str = "interval"
    param_limit: str = "limit"

    # Authentication style: "bearer", "header" or "query".
    auth_style: str = "bearer"
    auth_header_name: str = "Authorization"
    auth_query_param: str = "apiKey"

    def require(self, name: str) -> str:
        path = getattr(self, name, "")
        if not path:
            raise MassiveAPIError(
                f"endpoint {name!r} is not configured. Read the Massive API "
                "documentation, then set it in EndpointConfig (or via "
                "scripts/probe_api.py, which discovers working paths). Paths are "
                "intentionally not guessed."
            )
        return path


@dataclass
class RateLimit:
    """Simple token-bucket limiter.

    Documented limits are unknown before access is granted, so the default is
    deliberately conservative. Being throttled or, worse, having the key
    suspended during a five-day project would be far more costly than a slow
    download.
    """

    max_per_minute: int = 60
    _timestamps: list[float] = field(default_factory=list, repr=False)

    def acquire(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < 60.0]
        if len(self._timestamps) >= self.max_per_minute:
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.05
            if sleep_for > 0:
                logger.info("Rate limit reached; sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            self._timestamps = [
                t for t in self._timestamps if time.monotonic() - t < 60.0
            ]
        self._timestamps.append(time.monotonic())


@dataclass
class ProbeResult:
    """Outcome of a single exploratory request, safe to serialise."""

    path: str
    params_sent: dict
    ok: bool
    status_code: int | None
    elapsed_seconds: float
    error: str | None = None
    top_level_type: str | None = None
    top_level_keys: list[str] = field(default_factory=list)
    record_count: int | None = None
    record_keys: list[str] = field(default_factory=list)
    sample_record: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MassiveClient:
    """Thin, cautious wrapper over the Massive REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        endpoints: EndpointConfig | None = None,
        rate_limit: RateLimit | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        use_cache: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self.base_url = (base_url or get_base_url()).rstrip("/")
        if not self.base_url:
            raise MassiveAPIError(
                "MASSIVE_BASE_URL is not set. Add it to .env once the "
                "documentation confirms the correct host."
            )
        self.endpoints = endpoints or EndpointConfig()
        self.rate_limit = rate_limit or RateLimit()
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self.request_count = 0

    # -- credentials -------------------------------------------------------

    def _auth(self, params: dict) -> tuple[dict, dict]:
        """Return (headers, params) with credentials attached per auth style."""
        key = get_api_key()
        headers = {"Accept": "application/json", "User-Agent": "quant-research/0.1"}
        params = dict(params)
        style = self.endpoints.auth_style.lower()
        if style == "bearer":
            headers[self.endpoints.auth_header_name] = f"Bearer {key}"
        elif style == "header":
            headers[self.endpoints.auth_header_name] = key
        elif style == "query":
            params[self.endpoints.auth_query_param] = key
        else:
            raise MassiveAPIError(f"unknown auth_style {self.endpoints.auth_style!r}")
        return headers, params

    def __repr__(self) -> str:  # pragma: no cover - must never leak the key
        return (
            f"MassiveClient(base_url={self.base_url!r}, "
            f"requests_made={self.request_count})"
        )

    # -- caching -----------------------------------------------------------

    def _cache_path(self, path: str, params: dict) -> Path:
        # The key is excluded from the cache key so that rotating it does not
        # invalidate the cache, and so no credential material reaches a filename.
        safe = {
            k: v
            for k, v in params.items()
            if k.lower() not in {n.lower() for n in _SECRET_PARAM_NAMES}
        }
        blob = json.dumps({"path": path, "params": safe}, sort_keys=True)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:20]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")[:60]
        return self.cache_dir / f"{stem}__{digest}.json"

    # -- requests ----------------------------------------------------------

    def request(
        self,
        path: str,
        params: dict | None = None,
        allow_cache: bool | None = None,
    ) -> Any:
        """Perform a GET request with retries, rate limiting and caching."""
        params = params or {}
        use_cache = self.use_cache if allow_cache is None else allow_cache
        cache_file = self._cache_path(path, params)

        if use_cache and cache_file.exists():
            logger.debug("Cache hit for %s", path)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        if path.startswith(("https://", "http://")):
            # Pagination URLs come from Massive responses. Never forward the
            # credential to another host if a malformed response is received.
            if urlparse(path).netloc != urlparse(self.base_url).netloc:
                raise MassiveAPIError(
                    "refusing to send credentials to a pagination URL on a "
                    "different host"
                )
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}"
        headers, full_params = self._auth(params)
        key = get_api_key()

        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            self.rate_limit.acquire()
            self.request_count += 1
            try:
                response = self._session.get(
                    url, params=full_params, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                backoff = min(2**attempt, 30)
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s; retrying in %ds",
                    redact(path, key),
                    attempt,
                    self.max_retries,
                    redact(last_error, key),
                    backoff,
                )
                time.sleep(backoff)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise MassiveAPIError(
                        f"response from {url} was not valid JSON: {exc}", 200
                    ) from None
                if use_cache:
                    cache_file.write_text(
                        json.dumps(payload, indent=None), encoding="utf-8"
                    )
                return payload

            if response.status_code in (401, 403):
                raise MassiveAPIError(
                    f"authentication failed with status {response.status_code}. "
                    "Check MASSIVE_API_KEY, and remember that access may be "
                    "restricted to a Hong Kong IP address.",
                    response.status_code,
                )
            if response.status_code == 404:
                raise MassiveAPIError(
                    f"endpoint not found: {redact(url, key)}. The path is probably "
                    "wrong; run scripts/probe_api.py to discover the correct one.",
                    404,
                )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                backoff = (
                    float(retry_after) if retry_after else min(2**attempt, 60)
                )
                last_error = f"status {response.status_code}"
                logger.warning(
                    "Retryable status %d from %s; sleeping %.0fs",
                    response.status_code,
                    redact(path, key),
                    backoff,
                )
                time.sleep(backoff)
                continue

            raise MassiveAPIError(
                f"unexpected status {response.status_code} from {redact(url, key)}: "
                f"{redact(response.text[:400], key)}",
                response.status_code,
            )

        raise MassiveAPIError(
            f"giving up on {redact(path, key)} after {self.max_retries} attempts; "
            f"last error was {redact(str(last_error), key)}"
        )

    def get_crypto_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str = "5min",
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> pd.DataFrame:
        """Retrieve and normalise Massive crypto custom bars.

        ``symbol`` accepts ``BTCUSD`` or Massive's canonical ``X:BTCUSD``.
        ``freq`` is a compact multiplier/timespan such as ``5min``, ``15min``,
        ``1hour`` or ``1day``. Pagination follows Massive's returned
        ``next_url`` and remains constrained to the configured API host.
        """
        match = re.fullmatch(r"(\d+)\s*(min|minute|hour|day)", freq.lower())
        if not match:
            raise ValueError(
                "freq must look like '5min', '15minute', '1hour', or '1day'"
            )
        multiplier = int(match.group(1))
        timespan = {"min": "minute"}.get(match.group(2), match.group(2))
        ticker = symbol.upper()
        if not ticker.startswith("X:"):
            ticker = f"X:{ticker.replace('-', '').replace('/', '')}"

        template = self.endpoints.require("crypto_aggregates")
        path = template.format(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
        )
        params: dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": int(limit),
        }

        records: list[dict] = []
        while path:
            payload = self.request(path, params)
            if not isinstance(payload, dict):
                raise MassiveAPIError("crypto aggregate response was not an object")
            page = payload.get("results", [])
            if not isinstance(page, list):
                raise MassiveAPIError("crypto aggregate 'results' was not a list")
            records.extend(page)
            path = payload.get("next_url")
            params = {}

        return normalise_bars(records, timestamp_unit="ms")

    def probe(self, path: str, params: dict | None = None) -> ProbeResult:
        """Try a request and describe the response without raising.

        Used by the discovery script, where a 404 is an expected outcome rather
        than an error.
        """
        params = params or {}
        started = time.perf_counter()
        try:
            payload = self.request(path, params, allow_cache=False)
        except MassiveAPIError as exc:
            return ProbeResult(
                path=path,
                params_sent=params,
                ok=False,
                status_code=exc.status_code,
                elapsed_seconds=time.perf_counter() - started,
                error=str(exc),
            )

        elapsed = time.perf_counter() - started
        result = ProbeResult(
            path=path,
            params_sent=params,
            ok=True,
            status_code=200,
            elapsed_seconds=elapsed,
            top_level_type=type(payload).__name__,
        )

        records: list | None = None
        if isinstance(payload, dict):
            result.top_level_keys = sorted(payload.keys())
            for candidate in ("results", "data", "bars", "candles", "values", "items"):
                if isinstance(payload.get(candidate), list):
                    records = payload[candidate]
                    break
        elif isinstance(payload, list):
            records = payload

        if records is not None:
            result.record_count = len(records)
            if records and isinstance(records[0], dict):
                result.record_keys = sorted(records[0].keys())
                result.sample_record = records[0]
        return result


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------

# Field names differ across providers. Normalisation is table-driven so that
# adapting to the real schema is a data change rather than a code change.
_FIELD_ALIASES = {
    "timestamp": ("t", "ts", "time", "timestamp", "datetime", "date", "start", "T"),
    "open": ("o", "open", "openPrice", "open_price"),
    "high": ("h", "high", "highPrice", "high_price"),
    "low": ("l", "low", "lowPrice", "low_price"),
    "close": ("c", "close", "closePrice", "close_price", "last"),
    "volume": ("v", "volume", "baseVolume", "base_volume", "vol"),
    "trades": ("n", "trades", "tradeCount", "trade_count", "num_trades"),
    "vwap": ("vw", "vwap", "weightedAverage"),
}


def normalise_bars(
    records: list[dict],
    timestamp_unit: str = "ms",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Convert raw API records into a canonical OHLCV frame.

    Parameters
    ----------
    timestamp_unit
        Epoch unit if timestamps are numeric. Getting this wrong is a common and
        catastrophic error, so a sanity check on the resulting date range is
        applied rather than trusting the argument.
    """
    if not records:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).rename_axis("timestamp")

    frame = pd.DataFrame(records)
    resolved: dict[str, pd.Series] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                resolved[canonical] = frame[alias]
                break

    if "timestamp" not in resolved:
        raise MassiveAPIError(
            f"no recognisable timestamp field in records; keys were "
            f"{sorted(frame.columns)[:20]}. Add the correct alias to _FIELD_ALIASES."
        )
    missing = [c for c in ("open", "high", "low", "close") if c not in resolved]
    if missing:
        raise MassiveAPIError(
            f"missing OHLC field(s) {missing}; record keys were "
            f"{sorted(frame.columns)[:20]}"
        )

    raw_ts = resolved.pop("timestamp")
    if pd.api.types.is_numeric_dtype(raw_ts):
        try:
            index = pd.to_datetime(raw_ts, unit=timestamp_unit, utc=True)
        except (OutOfBoundsDatetime, OverflowError, ValueError) as exc:
            # A wrong epoch unit is the overwhelmingly likely cause, and the
            # underlying error does not say so.
            raise MassiveAPIError(
                f"could not parse numeric timestamps with unit={timestamp_unit!r}: "
                f"{exc}. This almost always means the wrong epoch unit; try 's', "
                "'ms', 'us' or 'ns'."
            ) from None
    else:
        index = pd.to_datetime(raw_ts, utc=True, format="mixed")

    out = pd.DataFrame(resolved)
    out.index = pd.DatetimeIndex(index)
    out.index.name = "timestamp"
    out = out.sort_index()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # A wrong epoch unit produces dates in 1970 or in the far future, which is
    # far easier to catch here than after it has propagated into a result.
    lo, hi = out.index.min(), out.index.max()
    if lo < pd.Timestamp("2005-01-01", tz="UTC") or hi > pd.Timestamp(
        "2100-01-01", tz="UTC"
    ):
        raise MassiveAPIError(
            f"parsed timestamps span {lo} to {hi}, which is implausible; "
            f"timestamp_unit={timestamp_unit!r} is probably wrong "
            "(try 's', 'ms', 'us' or 'ns')"
        )

    if tz != "UTC":
        out.index = out.index.tz_convert(tz)
    return out
