#!/usr/bin/env python3
"""
AAPL around September iPhone-launch events — what actually happened.

Answers one question honestly: if you go LONG AAPL around the event, how far
against you has it typically gone, so a stop can be sized from history rather
than a hunch. Runs on a GitHub runner (Yahoo reachable there); the sandbox is
blocked from every market-data host.

No live price is invented. Entry is left to the reader's current quote; this
reports the DISTANCES (event-day range, adverse move, drift) as percentages, so
a stop can be placed off whatever the live price turns out to be.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# September iPhone announcement dates (event day). Approximate to the nearest
# known announcement; the script snaps each to the trading day on/after it.
EVENTS = [
    "2012-09-12", "2013-09-10", "2014-09-09", "2015-09-09", "2016-09-07",
    "2017-09-12", "2018-09-12", "2019-09-10", "2021-09-14", "2022-09-07",
    "2023-09-12", "2024-09-09", "2025-09-09",
    # 2020's September event (Sep 15) launched Watch/iPad, not iPhone — the
    # iPhone 12 came Oct 13 2020 — so it is deliberately excluded.
]


def main() -> int:
    df = yf.download("AAPL", start="2012-01-01", auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        print("no AAPL data")
        return 2
    if hasattr(df.columns, "levels"):
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower)
    idx = df.index

    rows = []
    for ev in EVENTS:
        d = pd.Timestamp(ev)
        pos = idx.searchsorted(d)          # first trading day on/after the event
        if pos <= 1 or pos + 5 >= len(df):
            continue
        o = df["open"].iloc[pos]
        h = df["high"].iloc[pos]
        l = df["low"].iloc[pos]
        c = df["close"].iloc[pos]
        prev_c = df["close"].iloc[pos - 1]

        day_ret = (c / prev_c - 1) * 100                      # close-to-close, event day
        intraday_range = (h - l) / o * 100                    # event-day high-low as % of open
        down_from_open = (l / o - 1) * 100                    # worst intraday dip from the open
        fwd5 = (df["close"].iloc[pos + 5] / c - 1) * 100      # 5 trading days after
        rows.append((str(idx[pos].date()), day_ret, intraday_range,
                     down_from_open, fwd5))

    if not rows:
        print("no events matched")
        return 2

    cols = ["event_day", "day_ret%", "range%", "dip_from_open%", "next5d%"]
    t = pd.DataFrame(rows, columns=cols)
    pd.set_option("display.width", 120)
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print("\n=== SUMMARY (n = %d events) ===" % len(t))
    red = (t["day_ret%"] < 0).mean() * 100
    print(f"event day closed RED: {red:.0f}% of the time  "
          f"(median day move {t['day_ret%'].median():+.2f}%)")
    print(f"event-day range: median {t['range%'].median():.2f}%, "
          f"worst {t['range%'].max():.2f}%")
    print(f"worst intraday dip from the open: median {t['dip_from_open%'].median():.2f}%, "
          f"deepest {t['dip_from_open%'].min():.2f}%")
    print(f"5 days later: median {t['next5d%'].median():+.2f}%, "
          f"positive {(t['next5d%']>0).mean()*100:.0f}% of the time")

    # Stop guidance: to survive an ordinary event-day pullback you must clear
    # the typical dip from the open. The 75th-percentile dip is a defensible
    # 'normal adverse move' — tighter than that and event noise stops you out.
    p75_dip = np.percentile(t["dip_from_open%"], 25)   # 25th pctile = a deeper-than-usual dip
    print(f"\nSTOP GUIDANCE: a long placed near the open has dipped to "
          f"{t['dip_from_open%'].median():.2f}% (median) intraday; to survive a "
          f"typical-to-deep event-day pullback, the stop needs ~{abs(p75_dip):.2f}% "
          f"of room below entry. Tighter risks a noise-out; wider over-risks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
