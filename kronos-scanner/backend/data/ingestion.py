# data/ingestion.py
"""yfinance loader with on-disk parquet caching and indicator computation."""

import os
import time
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

CACHE_DIR = os.environ.get(
    "KRONOS_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "_cache"),
)
CACHE_TTL_HOURS = float(os.environ.get("KRONOS_CACHE_TTL_HOURS", "12"))


_CHAIN = None


def _chain():
    """Lazily build the shared provider chain."""
    global _CHAIN
    if _CHAIN is None:
        from .providers import ProviderChain

        _CHAIN = ProviderChain()
    return _CHAIN


def provider_status() -> dict:
    """Which providers are configured, and Alpha Vantage quota remaining."""
    return _chain().status()


def _cache_path(ticker: str, interval: str = "1d") -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Interval is part of the key: intraday bars must never be served from a
    # daily cache entry, or a 1m scan silently gets yesterday's daily candles.
    suffix = "" if interval == "1d" else f".{interval}"
    return os.path.join(CACHE_DIR, f"{ticker.upper()}{suffix}.parquet")


def _cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    return age_hours < CACHE_TTL_HOURS


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add sma_20, sma_50, rsi_14 columns to an OHLCV frame."""
    df = df.copy()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = compute_rsi(df["close"], 14)
    return df


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance frame to lowercase OHLCV with a `date` column."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance can return a column MultiIndex for single tickers in some versions.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=-1) if ticker in raw.columns.get_level_values(-1) else raw.droplevel(-1, axis=1)

    raw = raw.rename(columns=str.lower).reset_index()
    raw = raw.rename(columns={"index": "date", "datetime": "date"})
    if "date" not in raw.columns and "Date" in raw.columns:
        raw = raw.rename(columns={"Date": "date"})

    keep = ["date", "open", "high", "low", "close", "volume"]
    raw = raw[[c for c in keep if c in raw.columns]]
    raw["ticker"] = ticker.upper()
    raw = raw.dropna(subset=["close"])
    return raw


def load_ticker(
    ticker: str,
    period: str = "2y",
    use_cache: bool = True,
    with_indicators: bool = True,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load a single ticker's daily OHLCV history.

    Returns a DataFrame with columns:
    [ticker, date, open, high, low, close, volume, sma_20, sma_50, rsi_14].
    """
    path = _cache_path(ticker, interval)

    if use_cache and _cache_fresh(path):
        df = pd.read_parquet(path)
    else:
        # Ordered provider chain: yfinance first, Alpha Vantage as a
        # short-list fallback when yfinance is down.
        from .providers import ProviderError

        try:
            df = _chain().fetch(ticker, period=period, interval=interval)
        except ProviderError as exc:
            raise RuntimeError(str(exc)) from exc

        if not df.empty and use_cache:
            df.to_parquet(path, index=False)

    if df.empty:
        return df

    if with_indicators:
        df = add_indicators(df)
    return df


def load_universe(
    tickers: Iterable[str],
    period: str = "2y",
    use_cache: bool = True,
    pause: float = 0.0,
    interval: str = "1d",
) -> dict:
    """Load many tickers. Returns {ticker: DataFrame}. Failures are skipped."""
    tickers = [str(t).upper() for t in tickers]
    out = {}

    # Serve whatever is cached and fresh without touching the network.
    need: list = []
    for t in tickers:
        path = _cache_path(t, interval)
        if use_cache and _cache_fresh(path):
            try:
                out[t] = add_indicators(pd.read_parquet(path))
                continue
            except Exception:  # noqa: BLE001 - corrupt cache entry, refetch
                pass
        need.append(t)

    # Batch-fetch the rest where the provider supports it (Alpaca does), which
    # turns a 160-request sweep into about two.
    if need:
        try:
            fetched = _chain().fetch_many(need, period=period, interval=interval)
        except Exception as exc:  # noqa: BLE001
            print(f"[ingestion] batch fetch failed, falling back per-ticker: {exc}")
            fetched = {}

        for t, df in fetched.items():
            if df is None or df.empty:
                continue
            if use_cache:
                try:
                    df.to_parquet(_cache_path(t, interval), index=False)
                except Exception:  # noqa: BLE001
                    pass
            out[t] = add_indicators(df)

        need = [t for t in need if t not in out]

    for t in need:
        try:
            df = load_ticker(t, period=period, use_cache=use_cache, interval=interval)
            if not df.empty:
                out[t.upper()] = df
        except Exception as exc:  # noqa: BLE001
            print(f"[ingestion] failed to load {t}: {exc}")
        if pause:
            time.sleep(pause)
    return out


# Default universe: liquid US large/mid caps across every major sector.
# A trend strategy needs breadth — with only a handful of names most sweeps
# find nothing, because at any moment only a small fraction of the market is
# actually setting up. The funnel makes width cheap: patterns run at ~0.3ms a
# name and token cost does not scale with universe size at all.
#
# Override wholesale with KRONOS_UNIVERSE="AAA,BBB,...".
DEFAULT_UNIVERSE = [
    # Mega-cap tech / comms
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "CSCO", "IBM", "NOW", "INTU", "AMAT", "MU", "LRCX", "KLAC",
    "ADI", "SNPS", "CDNS", "PANW", "CRWD", "SNOW", "DDOG", "NET",
    "SHOP", "UBER", "ABNB", "PLTR", "SQ", "PYPL", "COIN", "MRVL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK",
    "AXP", "V", "MA", "SPGI", "CB", "PGR", "USB", "PNC",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "DHR", "AMGN", "BMY", "GILD", "CVS", "ISRG", "VRTX", "REGN",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD",
    "KO", "PEP", "PG", "PM", "MDLZ", "CL", "KMB", "GIS",
    "DIS", "CMCSA", "BKNG", "MAR", "CMG", "LULU", "TJX", "ROST",
    # Industrials / transport
    "CAT", "DE", "BA", "GE", "HON", "UNP", "UPS", "FDX",
    "LMT", "RTX", "NOC", "GD", "MMM", "EMR", "ETN", "PH",
    # Energy / materials
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO",
    "OXY", "HAL", "DVN", "FCX", "NEM", "LIN", "APD", "NUE",
    # Utilities / REITs / staples
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA", "WELL",
    # Liquid ETFs — useful regime context for a trend strategy
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "SMH", "ARKK", "GLD", "TLT",
]
