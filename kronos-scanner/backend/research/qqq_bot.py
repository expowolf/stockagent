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
    Stop        0.50% of entry — a fixed percentage, not ATR.
    Target      2.5R (1.25%).
    Window      Entries only between 09:45 and 15:00 ET, today's bars only.
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
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import build, regime, TREND_DN, TREND_UP  # noqa: E402

ET = ZoneInfo("America/New_York")
TICKERS = ["QQQ", "SPY"]
# Percentage stops, not ATR. ATR collapses on a quiet tape and produced a 49c
# stop on SPY that the spread could cross — a coin flip paying a toll.
#
# 0.5% is also the number the account arithmetic forces. Fully deployed,
# notional = account, so realized risk = account x stop%. With risk set at
# 0.5%, a 0.5% stop is exactly the point where a capped position still carries
# the intended dollar risk. Anything tighter under-risks; anything wider
# over-deploys.
STOP_PCT = float(os.environ.get("KRONOS_STOP_PCT", "0.005"))    # 0.50%
TARGET_R = 2.5                                                   # -> 1.25% target
LOOKBACK_BARS = 4          # re-check recent bars so a 10-min cadence misses nothing

# Session VWAP resets at the open, so on the first bar of the day VWAP IS that
# bar's own typical price — "price touched VWAP and closed above it" is then
# nearly tautological. Wait until VWAP is an average of something.
MIN_BAR_OF_DAY = 3         # 15 minutes in

# Exit is flat-by-close, so a 1.25% target needs room to be reached. A signal
# at 15:50 gets two bars and then a forced market exit: that is not the trade
# that was measured, it is a coin flip that pays the spread.
LAST_ENTRY_MINUTE = 15 * 60     # 15:00 ET

# Loop mode: stay alive for the session instead of relying on the scheduler
# to fire twenty-one separate runs.
LOOP = os.environ.get("KRONOS_LOOP", "") not in ("", "0", "false")
SLEEP = int(os.environ.get("KRONOS_SLEEP", "300"))
# Watch until the actual bell, not until the entry cutoff. Entries stop at
# 15:00, but a position opened at 14:55 still resolves during the last hour —
# and a job that stopped watching at 15:05 would book it "flat" when it had
# really hit its stop or target. Those recorded outcomes are the only evidence
# that will ever say whether this edge is real, so systematically truncating
# them would quietly bias the very number being measured.
LOOP_END = os.environ.get("KRONOS_LOOP_END", "16:05")   # ET, just past the close

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
    if f["bar_of_day"][t] < MIN_BAR_OF_DAY:
        return 0
    if f["minute"][t] >= LAST_ENTRY_MINUTE:
        return 0
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
    # Anchor the day to ET, not UTC. A UTC rollover lands at 20:00 ET, which
    # is a different day from the session it is supposed to bookend.
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if s.get("day") != today:
        # New session: reset the daily counters but keep nothing stale.
        s = {"day": today, "alerted": [], "realized_r": 0.0,
             "consec_losses": 0, "open": None}
    s.setdefault("alerted", [])
    s.setdefault("realized_r", 0.0)
    s.setdefault("consec_losses", 0)
    s.setdefault("open", None)
    return s


def resolve_open(state: dict, tk: str, f: dict) -> None:
    """
    Mark the open signal against bars that printed after its entry.

    Stop is checked BEFORE target. When one 5-minute bar spans both levels the
    data cannot say which came first, and assuming the favourable order is the
    single biggest way a strategy flatters itself.

    This is also what finally makes the circuit breakers real: consec_losses
    and realized_r could never increment while nothing recorded an outcome.
    """
    op = state.get("open")
    if not op or op["ticker"] != tk:
        return

    long = op["side"] == "LONG"
    for i in range(f["n"]):
        if f["idx"][i].isoformat() <= op["entry_bar"]:
            continue
        hi, lo = f["h"][i], f["l"][i]
        hit_stop = lo <= op["stop"] if long else hi >= op["stop"]
        hit_target = hi >= op["target"] if long else lo <= op["target"]
        if hit_stop:
            close_open(state, -1.0, f"stopped at {op['stop']:.2f}")
            return
        if hit_target:
            close_open(state, TARGET_R, f"target {op['target']:.2f} hit")
            return


def close_open(state: dict, r: float, why: str) -> None:
    op = state["open"]
    state["realized_r"] += r
    state["consec_losses"] = 0 if r > 0 else state["consec_losses"] + 1
    print(f"[bot] CLOSED {op['side']} {op['ticker']} — {why} · {r:+.2f}R "
          f"· day {state['realized_r']:+.2f}R")
    post_issue(
        f"CLOSED {op['side']} {op['ticker']} · {r:+.2f}R · {why}",
        "\n".join([
            f"Entry {op['entry']:.2f} · stop {op['stop']:.2f} · "
            f"target {op['target']:.2f}", "",
            f"Result **{r:+.2f}R** ({why}).", "",
            f"_Day: {state['realized_r']:+.2f}R · "
            f"consecutive losses {state['consec_losses']}._",
            "_Stop is checked before target: when one bar spans both, the_",
            "_data cannot say which came first, so the loss is assumed._",
        ]))
    state["open"] = None


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


def post_issue(title: str, body: str) -> None:
    """
    Post the alert from inside the bot, not from a later workflow step.

    In loop mode the process lives for the whole session, so an alert that
    waits for the job to finish would reach the phone hours after the setup
    was tradeable. A late alert on an intraday signal is not a degraded
    alert, it is a wrong one.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
    if not (token and repo):
        print("[bot] no GITHUB_TOKEN/REPOSITORY — alert printed only")
        return
    payload = json.dumps({
        "title": title, "body": body, "labels": ["qqq-signal"],
        # Assignment notifies unconditionally; watching the repo does not.
        "assignees": [owner] if owner else [],
    }).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "kronos-bot"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[bot] alert posted: issue #{json.load(r)['number']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[bot] alert POST failed: {exc}")


def sweep(state: dict) -> int:
    """One pass over both tickers. Returns the number of new alerts."""
    # Only today's bars are tradeable. The lookback window reaches back a few
    # bars, and the alert ledger resets each session — without this, the first
    # run after the open would re-issue yesterday's closing signals as if they
    # were live, on a position that was supposed to be flat by that close.
    session_date = datetime.now(ET).date()

    found = 0
    for tk in TICKERS:
        try:
            df = load(tk)
        except Exception as exc:  # noqa: BLE001
            # One bad fetch must not end the session. The loop is the whole
            # day's coverage; killing it over a transient Yahoo error would
            # cost every remaining setup.
            print(f"{tk}: fetch failed ({exc})")
            continue
        if df is None:
            print(f"{tk}: no data")
            continue
        f = build(df)
        last = f["n"] - 1

        # Settle any open signal on this ticker before considering a new one.
        resolve_open(state, tk, f)

        # ONE POSITION AT A TIME.
        #
        # Every signal sizes to the full $570 — notional equals the account,
        # because SPY at $771 against $570 is a fraction of a share and the
        # buying-power cap binds on essentially every trade. So two live
        # signals are not two trades, they are the same money twice. The bot
        # has no broker connection and cannot see fills, so the only honest
        # discipline is to issue nothing further until the outstanding signal
        # has resolved.
        if state.get("open"):
            continue

        # Re-check the last few CLOSED bars. The newest bar is still forming,
        # so it is deliberately excluded — acting on a partial bar is acting on
        # a number that has not happened yet.
        for t in range(max(210, last - LOOKBACK_BARS), last):
            if f["idx"][t].date() != session_date:
                continue
            sig = signal_at(f, t)
            if sig == 0:
                continue
            key = f"{tk}:{f['idx'][t].isoformat()}"
            if key in state["alerted"]:
                continue

            entry = f["o"][t + 1] if t + 1 <= last else f["c"][t]
            risk_per_share = entry * STOP_PCT
            stop = entry - risk_per_share if sig > 0 else entry + risk_per_share
            target = (entry + risk_per_share * TARGET_R if sig > 0
                      else entry - risk_per_share * TARGET_R)

            shares = round(ACCOUNT * RISK_PCT / risk_per_share, 4)
            if shares * entry > ACCOUNT:      # buying-power cap
                shares = round(ACCOUNT / entry, 4)
            notional = shares * entry

            side = "LONG" if sig > 0 else "SHORT"
            head = (f"{side} {tk} @ {entry:.2f} · stop {stop:.2f} · "
                    f"target {target:.2f} · {shares:g}sh (${notional:,.0f}) · "
                    f"risk ${shares*risk_per_share:.2f}")
            detail = (f"bar {f['idx'][t]:%H:%M} ET · regime {regime(f, t)} · "
                      f"stop {STOP_PCT*100:.2f}% · R:R {TARGET_R}:1 · "
                      f"FLAT BY CLOSE")
            print(f"\nALERT: {head}\n       {detail}")
            state["alerted"].append(key)
            state["open"] = {
                "ticker": tk, "side": side, "entry": float(entry),
                "stop": float(stop), "target": float(target),
                "entry_bar": f["idx"][t].isoformat(),
            }
            write_state(state)
            post_issue(head, "\n".join([
                "```", f"ALERT: {head}", f"       {detail}", "```", "",
                "**Flat by the close — no overnight hold.**", "",
                "_VWAP pullback. Expectancy measured at +0.02 to +0.04R,_",
                "_which is inside the noise band — size stays at 0.5%._",
            ]))
            found += 1
            break     # one position at a time — stop scanning this ticker
    return found


def main() -> int:
    state = read_state()

    print(f"QQQ/SPY VWAP-pullback bot · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"account ${ACCOUNT:,.0f} · risk {RISK_PCT*100:.2f}%/trade "
          f"(${ACCOUNT*RISK_PCT:.2f}) · stop {STOP_PCT*100:.2f}% · "
          f"target {TARGET_R}R ({STOP_PCT*TARGET_R*100:.2f}%)")

    stop_reason = halted(state)
    if stop_reason:
        print(f"HALTED: {stop_reason}")
        print("No new signals will be issued today.")
        return 0

    if not LOOP:
        found = sweep(state)
        if not found:
            print("\nNo setups on closed bars.")
        write_state(state)
        print(f"\nday P&L {state['realized_r']:+.2f}R · consec losses "
              f"{state['consec_losses']} · alerts today {len(state['alerted'])}")
        return 0

    # Loop mode: one job covers the whole session.
    #
    # Workflow dispatch returns 403 for this token, so the only triggers are
    # cron and push — and cron fired zero of roughly forty slots the day this
    # was written. Twenty-one scheduled runs are twenty-one chances to be
    # dropped; one long-lived job needs the scheduler to work exactly once.
    end_h, end_m = (int(x) for x in LOOP_END.split(":"))
    stop_at = datetime.now(ET).replace(hour=end_h, minute=end_m,
                                       second=0, microsecond=0)
    print(f"LOOP until {stop_at:%H:%M} ET, every {SLEEP}s\n")

    total = 0
    while datetime.now(ET) < stop_at:
        try:
            total += sweep(state)
        except Exception as exc:  # noqa: BLE001
            print(f"[bot] sweep failed, continuing: {exc}")
        left = (stop_at - datetime.now(ET)).total_seconds()
        if left <= 0:
            break
        print(f"[{datetime.now(ET):%H:%M} ET] alerts today {len(state['alerted'])} "
              f"· {left/60:.0f} min left", flush=True)
        time.sleep(min(SLEEP, left))

    # The rule is flat by the close, so nothing may carry overnight. The exit
    # price is whatever the close was, which this process cannot know without
    # another fetch — so the position is cleared and left OUT of realized_r
    # rather than booked as a fabricated 0R. A made-up outcome would corrupt
    # the very statistics being gathered to judge whether the edge is real.
    if state.get("open"):
        op = state["open"]
        print(f"[bot] flat by close: {op['side']} {op['ticker']} from "
              f"{op['entry']:.2f} — outcome not measured")
        post_issue(
            f"FLAT BY CLOSE · {op['side']} {op['ticker']} from {op['entry']:.2f}",
            "\n".join([
                "Position closes at the bell per the rule. It hit neither "
                "stop nor target during the session.", "",
                f"Entry {op['entry']:.2f} · stop {op['stop']:.2f} · "
                f"target {op['target']:.2f}", "",
                "_Left out of the day's R total: the exit price is the close,_",
                "_which this run cannot read, and inventing a number would_",
                "_corrupt the statistics being gathered to test the edge._",
            ]))
        state["open"] = None

    write_state(state)
    print(f"\nSession done. {total} alert(s) this run · "
          f"{len(state['alerted'])} today · day P&L {state['realized_r']:+.2f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# run: 17:40:33Z

# run: 17:48:43Z
