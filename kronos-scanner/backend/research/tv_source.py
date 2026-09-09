#!/usr/bin/env python3
"""
Read TradingView bars relayed from the desktop.

This is the cloud half of the relay. The desktop runs tv_relay.mjs next to
TradingView Desktop and commits bars to data/live/tv/; this reads them back in
exactly the shape the bot's own load() returns, so a strategy can be pointed at
TradingView instead of Yahoo by changing one function.

WHY A RELAY AND NOT A DIRECT CONNECTION
    tradingview.com is refused by this environment's network policy — the proxy
    answers 403 to CONNECT — and the TradingView MCP server speaks Chrome
    DevTools Protocol on the desktop's localhost:9222, which nothing in the
    cloud can reach. The repo is the only channel both sides share.

CONTRACT (identical to research.qqq_bot.load)
    DataFrame indexed by tz-aware America/New_York timestamps, columns
    open/high/low/close/volume, regular session only, or None.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ET = "America/New_York"
TV_DIR = Path(__file__).resolve().parents[3] / "data" / "live" / "tv"

# How stale a relay file may be before it is refused. The desktop relay writes
# every 60s; several minutes of silence means it died, the laptop slept, or
# TradingView was closed. Serving those bars as live would be the worst
# failure mode available — a confident signal computed on yesterday's tape.
MAX_STALE = timedelta(minutes=10)


class StaleRelay(RuntimeError):
    """Relay data exists but is too old to trade on."""


def load_tv(ticker: str, timeframe: str = "5", max_stale: timedelta = MAX_STALE,
            drop_forming_bar: bool = True):
    """
    Load relayed TradingView bars for one symbol.

    Raises StaleRelay rather than returning stale data: a caller that gets None
    logs "no data" and moves on, which is the right response to a missing feed
    and the wrong response to a dead one.
    """
    path = TV_DIR / f"{ticker.upper()}_{timeframe}.json"
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    bars = payload.get("bars") or []
    if not bars:
        return None

    fetched = payload.get("fetched_at")
    if fetched:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(
            fetched.replace("Z", "+00:00"))
        if age > max_stale:
            raise StaleRelay(
                f"{ticker}: relay last wrote {age.total_seconds()/60:.0f} min ago "
                f"(limit {max_stale.total_seconds()/60:.0f}). Desktop relay is "
                f"probably not running.")

    df = pd.DataFrame(bars).rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.to_datetime(df.pop("t"), utc=True).dt.tz_convert(ET)
    df = df.sort_index()

    # The relay flags its final bar as still forming. Acting on a partial bar is
    # acting on a number that has not happened yet, so it goes by default.
    if drop_forming_bar and payload.get("last_bar_may_be_partial") and len(df) > 1:
        df = df.iloc[:-1]

    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= 570) & (mins < 960)]        # regular session only
    df = df[~df.index.duplicated(keep="last")]
    return df if len(df) > 60 else None


def status() -> str:
    """One-line health summary per relayed file, for a quick 'is it alive'."""
    if not TV_DIR.exists():
        return "no relay directory — desktop relay has never run"
    out = []
    for p in sorted(TV_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(
                d["fetched_at"].replace("Z", "+00:00"))).total_seconds() / 60
            last = d["bars"][-1]
            out.append(f"{p.stem}: {d['bar_count']} bars · last {last['t']} "
                       f"c={last['c']} · written {age:.0f} min ago"
                       f"{'  STALE' if age > MAX_STALE.total_seconds()/60 else ''}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"{p.stem}: unreadable ({exc})")
    return "\n".join(out) or "relay directory is empty"


if __name__ == "__main__":
    print(status())
