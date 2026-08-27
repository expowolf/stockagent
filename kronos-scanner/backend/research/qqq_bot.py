#!/usr/bin/env python3
"""
QQQ/SPY live signal bot — VWAP pullback continuation.

WHAT THIS IS, HONESTLY
    Of five setups tested walk-forward with costs, this had the largest sample
    (370+ trades) and the only broad positive region in the parameter grid:
    +0.02 to +0.04R across 1.2-1.5 ATR stops and 2-3R targets. Every other
    setup was either negative or profitable in a single cell surrounded by
    negatives.

    It is NOT a validated edge. Out-of-sample it was +0.070R on SPY and
    -0.152R on QQQ over 60 days. The expectancy is inside the noise band.
    It is shipped because the user asked for it live after seeing exactly
    these numbers.

    That is precisely why the risk controls below are strict rather than
    aggressive: when the edge is uncertain, survival has to come from position
    sizing and circuit breakers, not from conviction.

RULES
    Regime      ADX >= 18 with a stacked EMA structure (9/21/50) and price on
                the correct side of session VWAP.
    Entry       Price pulls back to touch VWAP or the 9 EMA, then closes back
                in the direction of the trend. Fill at the NEXT bar's open.
    Stop        1.2 x ATR(14).
    Target      2.5R.
    Exit        Flat by the close. No overnight risk.
    Size        Account risk / stop distance. Never the reverse.

CIRCUIT BREAKERS (from the user's own spec, section 4)
    0.5% base risk per trade, 1.0% absolute maximum.
    Daily loss limit 2.0% -> stop trading for the day.
    Three consecutive losses -> stop trading for the day.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import build, regime, TREND_DN, TREND_UP  # noqa: E402

TICKERS = ["QQQ", "SPY"]
STOP_ATR = 1.2
TARGET_R = 2.5
LOOKBACK_BARS = 4          # re-check recent bars so a 10-min cadence misses nothing

# A stop tighter than this is inside the spread and ordinary tick noise. On a
# quiet 5-minute bar ATR can fall to ~0.4 on SPY, which produces a 49c stop —
# about 0.06% of price, against ~3c of round-trip cost. That is not a trade,
# it is a coin flip paying a toll.
MIN_STOP_PCT = float(os.environ.get("KRONOS_MIN_STOP_PCT", "0.0015"))   # 0.15%
# When buying power caps the position this hard, realized risk collapses far
# below target and costs dominate. Reject rather than issue an unplayable fill.
MIN_RISK_FRACTION = float(os.environ.get("KRONOS_MIN_RISK_FRACTION", "0.5"))

ACCOUNT = float(os.environ.get("KRONOS_ACCOUNT_SIZE", "570"))
RISK_PCT = float(os.environ.get("KRONOS_RISK_PER_TRADE", "0.005"))   # 0.5% base
RISK_PCT_MAX = float(os.environ.get("KRONOS_RISK_MAX", "0.01"))      # 1.0% cap
DAILY_LOSS_LIMIT = float(os.environ.get("KRONOS_DAILY_LOSS", "0.02"))
MAX_CONSEC_LOSSES = int(os.environ.get("KRONOS_MAX_CONSEC_LOSSES", "3"))

STATE = Path(os.environ.get(
    "KRONOS_BOT_STATE",
    str(Path(__file__).resolve().parents[3] / "data" / "live" / "qqq_bot.json"),
))


def load(ticker: str):
    df = yf.download(ticker, period="5d", interval="5m",
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "levels"):
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower).dropna(subset=["close"])
    idx = pd.to_datetime(df.index)
    et = idx.tz_convert("America/New_York") if idx.tz else idx.tz_localize("UTC").tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    keep = (mins >= 570) & (mins < 960)     # regular session only
    df = df[keep]
    df.index = et[keep]
    return df if len(df) > 60 else None


def signal_at(f, t) -> int:
    """VWAP pullback, evaluated on a CLOSED bar. Returns +1 / -1 / 0."""
    r = regime(f, t)
    if r not in TREND_UP | TREND_DN:
        return 0
    touched = (f["l"][t] <= f["vwap"][t] <= f["h"][t]) or \
              (f["l"][t] <= f["e9"][t] <= f["h"][t])
    if not touched:
        return 0
    up_bar = f["c"][t] > f["o"][t]
    if r in TREND_UP and up_bar and f["c"][t] > f["vwap"][t]:
        return 1
    if r in TREND_DN and not up_bar and f["c"][t] < f["vwap"][t]:
        return -1
    return 0


def read_state() -> dict:
    try:
        s = json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        s = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if s.get("day") != today:
        # New session: reset the daily counters but keep nothing stale.
        s = {"day": today, "alerted": [], "realized_r": 0.0, "consec_losses": 0}
    s.setdefault("alerted", [])
    s.setdefault("realized_r", 0.0)
    s.setdefault("consec_losses", 0)
    return s


def write_state(s: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(s, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"[bot] state write failed: {exc}")


def halted(s: dict) -> str:
    """Circuit breakers. Returns a reason string, or '' if clear to trade."""
    if s["consec_losses"] >= MAX_CONSEC_LOSSES:
        return f"{s['consec_losses']} consecutive losses — done for the day"
    loss_r = -s["realized_r"] * RISK_PCT
    if loss_r >= DAILY_LOSS_LIMIT:
        return f"daily loss limit hit ({loss_r*100:.1f}% of account)"
    return ""


def main() -> int:
    state = read_state()
    stop_reason = halted(state)

    print(f"QQQ/SPY VWAP-pullback bot · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"account ${ACCOUNT:,.0f} · risk {RISK_PCT*100:.2f}%/trade "
          f"(${ACCOUNT*RISK_PCT:.2f}) · stop {STOP_ATR}xATR · target {TARGET_R}R")

    if stop_reason:
        print(f"HALTED: {stop_reason}")
        print("No new signals will be issued today.")
        return 0

    found = 0
    for tk in TICKERS:
        df = load(tk)
        if df is None:
            print(f"{tk}: no data")
            continue
        f = build(df)
        last = f["n"] - 1

        # Re-check the last few CLOSED bars. The newest bar is still forming,
        # so it is deliberately excluded — acting on a partial bar is acting on
        # a number that has not happened yet.
        for t in range(max(210, last - LOOKBACK_BARS), last):
            sig = signal_at(f, t)
            if sig == 0:
                continue
            key = f"{tk}:{f['idx'][t].isoformat()}"
            if key in state["alerted"]:
                continue

            atr = f["atr"][t]
            if not np.isfinite(atr) or atr <= 0:
                continue

            entry = f["o"][t + 1] if t + 1 <= last else f["c"][t]
            risk_per_share = atr * STOP_ATR

            # Floor the stop. ATR alone is not enough on a quiet tape.
            floor = entry * MIN_STOP_PCT
            if risk_per_share < floor:
                print(f"  skip {tk}: stop {risk_per_share:.2f} inside noise "
                      f"(floor {floor:.2f} = {MIN_STOP_PCT*100:.2f}% of price)")
                continue
            stop = entry - risk_per_share if sig > 0 else entry + risk_per_share
            target = entry + risk_per_share * TARGET_R if sig > 0 else entry - risk_per_share * TARGET_R

            dollar_risk = ACCOUNT * RISK_PCT
            shares = round(dollar_risk / risk_per_share, 4)
            notional = shares * entry
            if notional > ACCOUNT:
                shares = round(ACCOUNT / entry, 4)
                notional = shares * entry

            realized_risk = shares * risk_per_share
            if realized_risk < dollar_risk * MIN_RISK_FRACTION:
                print(f"  skip {tk}: buying power caps risk at ${realized_risk:.2f} "
                      f"vs ${dollar_risk:.2f} target — costs would dominate")
                continue

            side = "LONG" if sig > 0 else "SHORT"
            print(f"\nALERT: {side} {tk} @ {entry:.2f} · stop {stop:.2f} · "
                  f"target {target:.2f} · {shares:g}sh (${notional:,.0f}) · "
                  f"risk ${shares*risk_per_share:.2f}")
            print(f"       bar {f['idx'][t]:%H:%M} ET · regime {regime(f, t)} · "
                  f"ATR {atr:.2f} · R:R {TARGET_R}:1 · FLAT BY CLOSE")
            state["alerted"].append(key)
            found += 1

    if not found:
        print("\nNo setups on closed bars.")
    write_state(state)
    print(f"\nday P&L {state['realized_r']:+.2f}R · consec losses "
          f"{state['consec_losses']} · alerts today {len(state['alerted'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# run: 17:40:33Z

# run: 17:48:43Z
