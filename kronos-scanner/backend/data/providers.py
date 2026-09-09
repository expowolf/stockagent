# data/providers.py
"""
Market data providers with ordered fallback.

Tried in order; each is skipped automatically when unconfigured:

  1. Alpaca        Official API. Free tier gives real-time IEX bars, 200
                   req/min, 7+ years of history, and — decisively — BATCHING,
                   so 160 symbols cost ~2 requests instead of 160. Free keys,
                   no funded account required. Use this if you use anything.
  2. yfinance      Free and unlimited, needs no key, but unofficial: it breaks
                   for days at a time whenever Yahoo changes an endpoint.
  3. AlphaVantage  25 requests/DAY on the free tier — about 6x short of a
                   SINGLE sweep over a 160-name universe. Restricted to an
                   explicit short list, with a persistent quota guard that
                   refuses rather than burning the allowance and failing
                   mid-morning.

The ordering matters: Alpaca is both the most reliable and the cheapest per
symbol, so it should carry the universe when available. yfinance is the
keyless default. Alpha Vantage exists purely as insurance for a handful of
names when the first two are down.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


CANONICAL_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


class ProviderError(Exception):
    """Raised when a provider cannot serve a request."""


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return canonical OHLCV, or raise ProviderError."""

    def available_for(self, ticker: str) -> bool:
        """Whether this provider may be used for `ticker` right now."""
        return True


# --------------------------------------------------------------------------- #
# yfinance (primary)
# --------------------------------------------------------------------------- #
class YFinanceProvider(DataProvider):
    name = "yfinance"

    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        if yf is None:
            raise ProviderError("yfinance not installed")

        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yfinance error: {exc}") from exc

        if raw is None or raw.empty:
            raise ProviderError("yfinance returned no rows")

        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(-1):
                raw = raw.xs(ticker, axis=1, level=-1)
            else:
                raw = raw.droplevel(-1, axis=1)

        df = raw.rename(columns=str.lower).reset_index()
        df = df.rename(columns={"index": "date", "datetime": "date", "Date": "date"})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["ticker"] = ticker.upper()
        df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]
        return df.dropna(subset=["close"])


# --------------------------------------------------------------------------- #
# Alpaca (best free option — real-time IEX, batched, official API)
# --------------------------------------------------------------------------- #
class AlpacaProvider(DataProvider):
    """
    Alpaca market data. Free tier gives real-time IEX bars, 200 req/min and
    7+ years of history — roughly a thousand times Alpha Vantage's 25/day.

    Its decisive advantage here is batching: one request returns bars for many
    symbols, so a 160-name universe costs a couple of requests instead of 160.
    Keys are free and do not require funding an account.

        export ALPACA_API_KEY_ID=...
        export ALPACA_API_SECRET_KEY=...
    """

    name = "alpaca"
    BASE = "https://data.alpaca.markets/v2/stocks/bars"
    MAX_SYMBOLS_PER_REQUEST = 100

    def __init__(
        self,
        key_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: Optional[str] = None,
    ) -> None:
        self.key_id = key_id or os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET_KEY", "")
        # "iex" is the free real-time feed; "sip" needs a paid subscription.
        self.feed = feed or os.environ.get("ALPACA_FEED", "iex")

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.secret_key and httpx is not None)

    def available_for(self, ticker: str) -> bool:
        return self.configured

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    @staticmethod
    def _timeframe(interval: str) -> str:
        return {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "60m": "1Hour", "1h": "1Hour", "1d": "1Day", "1w": "1Week",
        }.get(interval, "1Day")

    @staticmethod
    def _start_for(period: str) -> str:
        days = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 731, "5y": 1827}
        delta = timedelta(days=days.get(period, 366))
        return (datetime.utcnow() - delta).strftime("%Y-%m-%d")

    def _request(self, symbols: List[str], period: str, interval: str) -> dict:
        params = {
            "symbols": ",".join(s.upper() for s in symbols),
            "timeframe": self._timeframe(interval),
            "start": self._start_for(period),
            "limit": 10000,
            "adjustment": "split",
            "feed": self.feed,
        }
        bars: dict = {}
        page_token = None

        # Alpaca paginates; follow next_page_token until exhausted.
        for _ in range(20):
            if page_token:
                params["page_token"] = page_token
            try:
                resp = httpx.get(self.BASE, params=params, headers=self._headers(), timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"alpaca error: {exc}") from exc

            for sym, rows in (payload.get("bars") or {}).items():
                bars.setdefault(sym, []).extend(rows)

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        return bars

    @staticmethod
    def _to_frame(ticker: str, rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "date": pd.to_datetime(r["t"]).tz_localize(None),
            "open": float(r["o"]), "high": float(r["h"]),
            "low": float(r["l"]), "close": float(r["c"]),
            "volume": int(r.get("v", 0)),
        } for r in rows])
        df["ticker"] = ticker.upper()
        df = df.sort_values("date").reset_index(drop=True)
        return df[[c for c in CANONICAL_COLUMNS if c in df.columns]]

    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        if not self.configured:
            raise ProviderError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")
        bars = self._request([ticker], period, interval)
        df = self._to_frame(ticker, bars.get(ticker.upper(), []))
        if df.empty:
            raise ProviderError(f"alpaca returned no bars for {ticker}")
        return df

    def fetch_many(self, tickers: List[str], period: str = "1y", interval: str = "1d") -> dict:
        """
        Batch fetch. This is the whole reason to prefer Alpaca: 160 symbols
        cost 2 requests rather than 160.
        """
        if not self.configured:
            raise ProviderError("alpaca not configured")

        out: dict = {}
        syms = [t.upper() for t in tickers]
        for i in range(0, len(syms), self.MAX_SYMBOLS_PER_REQUEST):
            chunk = syms[i:i + self.MAX_SYMBOLS_PER_REQUEST]
            bars = self._request(chunk, period, interval)
            for sym in chunk:
                df = self._to_frame(sym, bars.get(sym, []))
                if not df.empty:
                    out[sym] = df
        return out


# --------------------------------------------------------------------------- #
# Alpha Vantage (narrow fallback)
# --------------------------------------------------------------------------- #
class AlphaVantageQuota:
    """
    Persistent daily-request counter.

    The free tier is 25/day and resets on their clock, not ours, so we track
    conservatively against local midnight and refuse rather than overspend.
    """

    def __init__(self, limit: int = 25, state_path: Optional[str] = None) -> None:
        self.limit = int(limit)
        self.path = Path(
            state_path or os.environ.get("KRONOS_AV_QUOTA_STATE", "/tmp/kronos_av_quota.json")
        )
        self._load()

    def _load(self) -> None:
        self.day = date.today().isoformat()
        self.used = 0
        try:
            data = json.loads(self.path.read_text())
            if data.get("day") == self.day:
                self.used = int(data.get("used", 0))
        except Exception:  # noqa: BLE001 - missing/corrupt state starts fresh
            pass

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"day": self.day, "used": self.used}))
        except Exception as exc:  # noqa: BLE001
            print(f"[alphavantage] could not persist quota: {exc}")

    def _roll(self) -> None:
        today = date.today().isoformat()
        if today != self.day:
            self.day, self.used = today, 0
            self._save()

    @property
    def remaining(self) -> int:
        self._roll()
        return max(0, self.limit - self.used)

    def consume(self) -> bool:
        self._roll()
        if self.used >= self.limit:
            return False
        self.used += 1
        self._save()
        return True

    def snapshot(self) -> dict:
        self._roll()
        return {"day": self.day, "used": self.used, "limit": self.limit, "remaining": self.remaining}


class AlphaVantageProvider(DataProvider):
    """
    Free tier: 25 requests/day, 5/minute.

    Restricted to an explicit short list because it cannot carry a wide
    universe — see the module docstring.
    """

    name = "alphavantage"
    BASE = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: Optional[str] = None,
        short_list: Optional[List[str]] = None,
        daily_limit: int = 25,
        min_interval_s: float = 12.5,   # 5 req/min
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        raw = short_list if short_list is not None else os.environ.get("KRONOS_FALLBACK_TICKERS", "")
        if isinstance(raw, str):
            self.short_list = {t.strip().upper() for t in raw.split(",") if t.strip()}
        else:
            self.short_list = {t.upper() for t in raw}
        self.quota = AlphaVantageQuota(limit=daily_limit)
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def available_for(self, ticker: str) -> bool:
        if not self.api_key or httpx is None:
            return False
        if not self.short_list:
            return False   # no short list means no fallback, by design
        if ticker.upper() not in self.short_list:
            return False
        return self.quota.remaining > 0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.time()

    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        if not self.api_key:
            raise ProviderError("ALPHAVANTAGE_API_KEY not set")
        if httpx is None:
            raise ProviderError("httpx not installed")
        if ticker.upper() not in self.short_list:
            raise ProviderError(f"{ticker} not on the Alpha Vantage short list")

        intraday = interval not in ("1d", "1day", "daily")
        params = {
            "symbol": ticker.upper(),
            "apikey": self.api_key,
            "outputsize": "full" if period in ("2y", "5y", "max") else "compact",
        }
        if intraday:
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = _av_interval(interval)
        else:
            params["function"] = "TIME_SERIES_DAILY_ADJUSTED"

        # Reserve quota BEFORE the call so a crash mid-request cannot
        # double-spend the daily allowance.
        if not self.quota.consume():
            raise ProviderError(
                f"Alpha Vantage daily quota exhausted ({self.quota.limit}/day)"
            )

        self._throttle()
        try:
            resp = httpx.get(self.BASE, params=params, timeout=25)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"alphavantage error: {exc}") from exc

        # Alpha Vantage signals problems in-band with HTTP 200.
        for key in ("Note", "Information", "Error Message"):
            if key in payload:
                raise ProviderError(f"alphavantage {key}: {str(payload[key])[:160]}")

        series_key = next((k for k in payload if "Time Series" in k), None)
        if series_key is None:
            raise ProviderError(f"unexpected alphavantage response: {list(payload)[:4]}")

        rows = []
        for stamp, vals in payload[series_key].items():
            rows.append({
                "date": pd.to_datetime(stamp),
                "open": float(vals.get("1. open", "nan")),
                "high": float(vals.get("2. high", "nan")),
                "low": float(vals.get("3. low", "nan")),
                "close": float(vals.get("4. close", "nan")),
                "volume": int(float(vals.get("6. volume", vals.get("5. volume", 0)))),
            })

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df["ticker"] = ticker.upper()
        df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]
        return df.dropna(subset=["close"])


def _av_interval(interval: str) -> str:
    """Map our interval strings to Alpha Vantage's."""
    return {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "60m": "60min", "1h": "60min",
    }.get(interval, "5min")


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #
class ProviderChain:
    """Try providers in order; first success wins."""

    def __init__(self, providers: Optional[List[DataProvider]] = None) -> None:
        self.providers = providers if providers is not None else default_providers()
        self.last_source: Optional[str] = None
        self.failures: dict = {}

    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        errors = []
        for prov in self.providers:
            if not prov.available_for(ticker):
                continue
            try:
                df = prov.fetch(ticker, period=period, interval=interval)
                if df is not None and not df.empty:
                    self.last_source = prov.name
                    return df
                errors.append(f"{prov.name}: empty")
            except ProviderError as exc:
                errors.append(f"{prov.name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{prov.name}: unexpected {type(exc).__name__}: {exc}")

        self.failures[ticker.upper()] = errors
        raise ProviderError(f"all providers failed for {ticker}: " + " | ".join(errors))

    def fetch_many(self, tickers: List[str], period: str = "1y", interval: str = "1d") -> dict:
        """
        Fetch a whole universe, using batch requests where a provider supports
        them and falling back to per-ticker for whatever is still missing.
        """
        out: dict = {}
        remaining = [t.upper() for t in tickers]

        for prov in self.providers:
            if not remaining or not hasattr(prov, "fetch_many"):
                continue
            if not prov.available_for(remaining[0]):
                continue
            try:
                got = prov.fetch_many(remaining, period=period, interval=interval)
            except ProviderError as exc:
                print(f"[providers] {prov.name} batch failed: {exc}")
                continue
            if got:
                self.last_source = prov.name
                out.update(got)
                remaining = [t for t in remaining if t not in out]

        for ticker in remaining:
            try:
                out[ticker] = self.fetch(ticker, period=period, interval=interval)
            except ProviderError:
                pass
        return out

    def status(self) -> dict:
        out = {"providers": [p.name for p in self.providers], "last_source": self.last_source}
        for p in self.providers:
            if isinstance(p, AlpacaProvider):
                out["alpaca"] = {"configured": p.configured, "feed": p.feed}
        for p in self.providers:
            if isinstance(p, AlphaVantageProvider):
                out["alphavantage"] = {
                    "configured": bool(p.api_key),
                    "short_list": sorted(p.short_list),
                    "quota": p.quota.snapshot(),
                }
        return out


def default_providers() -> List[DataProvider]:
    """
    Yahoo leads by choice: it needs no key, serves intraday candles down to
    1-minute, and carries the news feed the watcher reads — so quotes and
    headlines come from one consistent source.

      1. yfinance      primary. Free, keyless, intraday + news.
      2. Alpaca        used when keys are set; official API and batched, so it
                       is the better backstop when Yahoo has an outage.
      3. AlphaVantage  25 requests/day, short-list only.

    Each is skipped automatically when unconfigured. Set KRONOS_PRIMARY=alpaca
    to put Alpaca first instead.
    """
    if os.environ.get("KRONOS_PRIMARY", "yahoo").lower() in ("alpaca", "alpaca-first"):
        return [AlpacaProvider(), YFinanceProvider(), AlphaVantageProvider()]
    return [YFinanceProvider(), AlpacaProvider(), AlphaVantageProvider()]
