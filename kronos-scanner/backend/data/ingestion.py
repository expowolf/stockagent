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


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker.upper()}.parquet")


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
) -> pd.DataFrame:
    """
    Load a single ticker's daily OHLCV history.

    Returns a DataFrame with columns:
    [ticker, date, open, high, low, close, volume, sma_20, sma_50, rsi_14].
    """
    path = _cache_path(ticker)

    if use_cache and _cache_fresh(path):
        df = pd.read_parquet(path)
    else:
        if yf is None:
            raise RuntimeError("yfinance is not installed; cannot fetch live data.")
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        df = _normalize(raw, ticker)
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
) -> dict:
    """Load many tickers. Returns {ticker: DataFrame}. Failures are skipped."""
    out = {}
    for t in tickers:
        try:
            df = load_ticker(t, period=period, use_cache=use_cache)
            if not df.empty:
                out[t.upper()] = df
        except Exception as exc:  # noqa: BLE001
            print(f"[ingestion] failed to load {t}: {exc}")
        if pause:
            time.sleep(pause)
    return out


# A small, liquid default universe so the app runs out of the box.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "INTC", "CSCO", "ORCL", "ADBE", "CRM", "QCOM", "TXN",
    "JPM", "BAC", "WMT", "KO", "PEP", "DIS", "NKE", "XOM",
]
