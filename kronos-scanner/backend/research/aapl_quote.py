#!/usr/bin/env python3
"""
Most-recent AAPL quote, fetched on a runner (Yahoo reachable there).

Prints the last price, the bar it came from, and — critically — how STALE that
bar is in wall-clock minutes, so a stale print is never mistaken for live. The
free Yahoo feed is ~15 min delayed; this reports the real lag rather than
implying real-time.
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf


def main() -> int:
    df = yf.download("AAPL", period="1d", interval="1m",
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        print("no intraday data — market may be pre-open")
        return 2
    if hasattr(df.columns, "levels"):
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower).dropna(subset=["close"])
    idx = pd.to_datetime(df.index)
    et = idx.tz_convert("America/New_York") if idx.tz else \
        idx.tz_localize("UTC").tz_convert("America/New_York")

    last_t = et[-1]
    last = df["close"].iloc[-1]
    day_open = df["open"].iloc[0]
    hi = df["high"].max()
    lo = df["low"].min()
    now = datetime.now(timezone.utc)
    lag = (now - idx[-1].tz_localize("UTC") if idx.tz is None
           else now - idx[-1]).total_seconds() / 60

    print(f"AAPL last {last:.2f}  (bar {last_t:%H:%M ET})")
    print(f"today: open {day_open:.2f} · high {hi:.2f} · low {lo:.2f}")
    print(f"bar age: {lag:.0f} min behind real time "
          f"(free Yahoo feed is ~15 min delayed)")

    # Pre-compute the 2.6% event stop off THIS price, clearly labelled as an
    # anchor off a delayed quote, not a live entry.
    stop = last * (1 - 0.026)
    print(f"\nIf you entered here: stop 2.6% below = {stop:.2f} "
          f"(risk/share {last-stop:.2f})")
    print(f"  ~$110 position risks ${(last-stop)*110/last:.2f}; "
          f"full $570 risks ${(last-stop)*570/last:.2f}")
    print(f"fetched {now:%H:%M}Z — this is a DELAYED anchor, read your live "
          f"quote at 1pm and re-place the stop off that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
