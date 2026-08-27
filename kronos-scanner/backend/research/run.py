#!/usr/bin/env python3
"""
Research run: walk-forward evaluation of SPY/QQQ intraday setups.

Reports each setup in-sample and out-of-sample, plus a parameter-perturbation
sweep. The point is not to find the best number — it is to find out whether any
edge survives being pushed around and paying costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import (  # noqa: E402
    HEADER, SETUPS, build, fmt, metrics, regime, simulate,
)

TICKERS = ["SPY", "QQQ"]


def load(ticker: str, interval="5m", period="60d"):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "levels"):
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower).dropna(subset=["close"])
    # Regular session only. Pre/post bars are thin, wide-spread, and would
    # flatter any breakout rule with fills nobody could actually get.
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        et = idx.tz_convert("America/New_York")
    else:
        et = idx.tz_localize("UTC").tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    df = df[(mins >= 570) & (mins < 960)]
    df.index = et[(mins >= 570) & (mins < 960)]
    return df


def main() -> int:
    data = {}
    for tk in TICKERS:
        df = load(tk)
        if df is None or len(df) < 500:
            print(f"{tk}: insufficient data ({0 if df is None else len(df)} bars)")
            continue
        data[tk] = build(df)
        print(f"{tk}: {len(df)} 5m bars, "
              f"{df.index[0].date()} -> {df.index[-1].date()}")

    if not data:
        print("\nNO DATA — cannot evaluate. Aborting rather than guessing.")
        return 2

    # Regime distribution — how much of the tape is even tradeable?
    print("\n=== REGIME DISTRIBUTION ===")
    for tk, f in data.items():
        rs = [regime(f, t) for t in range(210, f["n"])]
        tot = len(rs)
        counts = {r: rs.count(r) / tot for r in sorted(set(rs))}
        line = "  ".join(f"{k}:{v:.0%}" for k, v in counts.items())
        print(f"  {tk}  {line}")
    print("  A/B=trend up  C/D=trend dn  E=range  F=high vol  G=compress  H=no-trade")

    # Walk-forward: first 60% in-sample, last 40% out-of-sample.
    print("\n=== WALK-FORWARD (60% IS / 40% OOS, costs included) ===")
    for tk, f in data.items():
        split = int(f["n"] * 0.6)
        print(f"\n{tk} IN-SAMPLE")
        print(HEADER)
        is_res = {}
        for name, fn in SETUPS.items():
            tr = simulate(f, fn, start=210, end=split)
            is_res[name] = metrics(tr)
            print(fmt(name, is_res[name]))

        print(f"\n{tk} OUT-OF-SAMPLE")
        print(HEADER)
        for name, fn in SETUPS.items():
            tr = simulate(f, fn, start=split, end=f["n"])
            print(fmt(name, metrics(tr)))

    # Parameter perturbation — the anti-overfitting test that matters most.
    print("\n=== ROBUSTNESS: stop distance x target (all data, SPY+QQQ pooled) ===")
    print(f"  {'setup':<20} {'0.8/2R':>9} {'1.0/2R':>9} {'1.2/2R':>9} "
          f"{'1.5/2R':>9} {'1.2/2.5R':>9} {'1.2/3R':>9}")
    grid = [(0.8, 2.0), (1.0, 2.0), (1.2, 2.0), (1.5, 2.0), (1.2, 2.5), (1.2, 3.0)]
    for name, fn in SETUPS.items():
        cells = []
        for sa, tr_ in grid:
            pooled = []
            for f in data.values():
                pooled += simulate(f, fn, stop_atr=sa, target_r=tr_, start=210)
            m = metrics(pooled)
            cells.append(f"{m['exp_r']:+.3f}" if m.get("n") else "—")
        print(f"  {name:<20} " + " ".join(f"{c:>9}" for c in cells))
    print("\n  Reading: a real edge shows POSITIVE across the row. A single")
    print("  standout cell surrounded by negatives is curve-fit, not signal.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
