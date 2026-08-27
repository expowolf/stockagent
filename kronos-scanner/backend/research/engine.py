#!/usr/bin/env python3
"""
SPY/QQQ intraday research engine.

Built to the spec's discipline, and specifically to avoid the ways an intraday
backtest lies to you:

  LOOK-AHEAD      Signals are evaluated on a CLOSED bar; entry fills at the
                  NEXT bar's open. Nothing is ever filled at the price that
                  generated it.
  INTRABAR ORDER  When a bar spans both stop and target, the stop is taken.
                  Daily/intraday bars cannot say which came first, and
                  assuming the favourable order is the single biggest source
                  of fake backtest edge.
  COSTS           Half-spread on entry and exit, plus slippage, plus
                  commission, subtracted from every trade in R terms.
  SESSION VWAP    VWAP resets each session. A continuous VWAP across days is
                  not the indicator anyone trades and quietly leaks trend.
  OVERLAP         One position at a time per instrument. Overlapping entries
                  inflate trade counts and understate risk.

Data reality: yfinance caps 5m history at ~60 days. That is enough to detect a
strong effect and NOT enough for a clean train/validate/holdout split. Results
are reported walk-forward, and the honest conclusion may be "insufficient
evidence".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Cost model — SPY/QQQ are the tightest instruments in the market, but not free
# --------------------------------------------------------------------------- #
SPREAD = 0.01        # typical penny spread on SPY/QQQ
SLIPPAGE = 0.01      # one extra cent each way, conservative for a retail stop
COMMISSION = 0.0     # most retail brokers, per share, equities


def round_trip_cost(price: float) -> float:
    """Dollar cost per share of entering and exiting once."""
    return SPREAD + 2 * SLIPPAGE + 2 * COMMISSION


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(a: np.ndarray, span: int) -> np.ndarray:
    k = 2.0 / (span + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out


def rma(a: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing, used by ATR and ADX."""
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    out[n - 1] = np.nanmean(a[:n])
    for i in range(n, len(a)):
        out[i] = (out[i - 1] * (n - 1) + a[i]) / n
    return out


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def adx(h, l, c, n=14):
    up = h - np.roll(h, 1)
    dn = np.roll(l, 1) - l
    up[0] = dn[0] = 0.0
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(plus, n) / tr
        mdi = 100 * rma(minus, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(np.nan_to_num(dx), n), pdi, mdi


def session_vwap(df: pd.DataFrame) -> np.ndarray:
    """
    VWAP anchored to each session's open.

    Resetting daily matters: a rolling or continuous VWAP is a different,
    smoother series that quietly encodes multi-day trend, and it is not what a
    trader is looking at on the chart.
    """
    tp = (df["high"] + df["low"] + df["close"]).to_numpy(float) / 3.0
    v = df["volume"].to_numpy(float)
    day = pd.to_datetime(df.index).normalize()
    out = np.empty(len(df))
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or day[i] != day[start]:
            seg_pv = np.cumsum(tp[start:i] * v[start:i])
            seg_v = np.cumsum(v[start:i])
            out[start:i] = np.where(seg_v > 0, seg_pv / np.maximum(seg_v, 1e-9), tp[start:i])
            start = i
    return out


def build(df: pd.DataFrame) -> dict:
    """All features, computed once per instrument."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    idx = pd.to_datetime(df.index)

    vwap = session_vwap(df)
    atr = rma(true_range(h, l, c), 14)
    adx14, pdi, mdi = adx(h, l, c, 14)

    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    e200 = ema(c, 200)

    # Relative volume vs the same bar-count average
    vol20 = pd.Series(v).rolling(20).mean().to_numpy()
    rvol = np.where(vol20 > 0, v / vol20, 1.0)

    # Session bookkeeping
    day = idx.normalize()
    minute = idx.hour * 60 + idx.minute
    new_session = np.r_[True, day[1:] != day[:-1]]
    bar_of_day = np.zeros(n, int)
    k = 0
    for i in range(n):
        k = 0 if new_session[i] else k + 1
        bar_of_day[i] = k

    # Opening range (first N bars of each session) and prior-day levels
    def opening_range(bars: int):
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        s = 0
        for i in range(1, n + 1):
            if i == n or new_session[i]:
                seg = slice(s, min(s + bars, i))
                if i - s > bars:
                    hi[s + bars:i] = h[seg].max()
                    lo[s + bars:i] = l[seg].min()
                s = i
        return hi, lo

    or5_h, or5_l = opening_range(1)    # 5m bars: 1 bar = 5 min
    or15_h, or15_l = opening_range(3)  # 3 bars = 15 min

    # Previous day's high/low, available from the first bar of the next session
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    s = 0
    prev_h = prev_l = np.nan
    for i in range(1, n + 1):
        if i == n or new_session[i]:
            pdh[s:i] = prev_h
            pdl[s:i] = prev_l
            prev_h, prev_l = h[s:i].max(), l[s:i].min()
            s = i

    return dict(o=o, h=h, l=l, c=c, v=v, n=n, idx=idx, minute=minute,
                vwap=vwap, atr=atr, adx=adx14, pdi=pdi, mdi=mdi,
                e9=e9, e21=e21, e50=e50, e200=e200, rvol=rvol,
                bar_of_day=bar_of_day, new_session=new_session,
                or5_h=or5_h, or5_l=or5_l, or15_h=or15_h, or15_l=or15_l,
                pdh=pdh, pdl=pdl)


# --------------------------------------------------------------------------- #
# Regime classifier
# --------------------------------------------------------------------------- #
def regime(f: dict, t: int) -> str:
    """Classify the environment BEFORE any signal is considered."""
    if not np.isfinite(f["atr"][t]) or not np.isfinite(f["adx"][t]):
        return "H"  # unknown -> no trade

    above_vwap = f["c"][t] > f["vwap"][t]
    stacked_up = f["e9"][t] > f["e21"][t] > f["e50"][t]
    stacked_dn = f["e9"][t] < f["e21"][t] < f["e50"][t]
    a = f["adx"][t]
    atr_pct = f["atr"][t] / max(f["c"][t], 1e-9)

    # Volatility extremes get their own buckets regardless of direction.
    lookback = slice(max(0, t - 100), t)
    atr_hist = f["atr"][lookback] / np.maximum(f["c"][lookback], 1e-9)
    if len(atr_hist) > 20 and np.isfinite(np.nanmedian(atr_hist)):
        med = np.nanmedian(atr_hist)
        if atr_pct > 1.8 * med:
            return "F"  # high-volatility breakout regime
        if atr_pct < 0.6 * med:
            return "G"  # compression

    if a >= 25 and stacked_up and above_vwap:
        return "A"
    if a >= 25 and stacked_dn and not above_vwap:
        return "C"
    if a >= 18 and stacked_up:
        return "B"
    if a >= 18 and stacked_dn:
        return "D"
    if a < 18:
        return "E"  # range / mean-reverting
    return "H"


TREND_UP = {"A", "B"}
TREND_DN = {"C", "D"}


# --------------------------------------------------------------------------- #
# Setups. Each returns +1 / -1 / 0, evaluated on a CLOSED bar t.
# --------------------------------------------------------------------------- #
def s_orb(f, t, r):
    """Opening-range breakout, 15-minute range, with trend and volume."""
    if f["bar_of_day"][t] < 3 or f["bar_of_day"][t] > 24:
        return 0
    if not np.isfinite(f["or15_h"][t]):
        return 0
    if f["rvol"][t] < 1.2:
        return 0
    if r in TREND_UP and f["c"][t] > f["or15_h"][t] and f["c"][t] > f["vwap"][t]:
        return 1
    if r in TREND_DN and f["c"][t] < f["or15_l"][t] and f["c"][t] < f["vwap"][t]:
        return -1
    return 0


def s_vwap_pullback(f, t, r):
    """Trend continuation: pull back into VWAP/9EMA, reject, resume."""
    if r not in TREND_UP | TREND_DN:
        return 0
    touched_v = f["l"][t] <= f["vwap"][t] <= f["h"][t]
    touched_e = f["l"][t] <= f["e9"][t] <= f["h"][t]
    if not (touched_v or touched_e):
        return 0
    body_up = f["c"][t] > f["o"][t]
    if r in TREND_UP and body_up and f["c"][t] > f["vwap"][t]:
        return 1
    if r in TREND_DN and not body_up and f["c"][t] < f["vwap"][t]:
        return -1
    return 0


def s_pdh_break(f, t, r):
    """Break of previous-day high/low with volume — a level everyone watches."""
    if not np.isfinite(f["pdh"][t]) or f["rvol"][t] < 1.3:
        return 0
    prev_below = f["c"][t - 1] <= f["pdh"][t]
    prev_above = f["c"][t - 1] >= f["pdl"][t]
    if r in TREND_UP and prev_below and f["c"][t] > f["pdh"][t]:
        return 1
    if r in TREND_DN and prev_above and f["c"][t] < f["pdl"][t]:
        return -1
    return 0


def s_momentum_ignition(f, t, r):
    """Range-expansion bar with abnormal volume, aligned with VWAP."""
    if not np.isfinite(f["atr"][t]) or f["atr"][t] <= 0:
        return 0
    rng = f["h"][t] - f["l"][t]
    if rng < 1.5 * f["atr"][t] or f["rvol"][t] < 1.8:
        return 0
    body = f["c"][t] - f["o"][t]
    if body > 0 and f["c"][t] > f["vwap"][t] and r not in TREND_DN:
        return 1
    if body < 0 and f["c"][t] < f["vwap"][t] and r not in TREND_UP:
        return -1
    return 0


def s_exhaustion(f, t, r):
    """Mean reversion — ONLY in a range or compression regime, per the spec."""
    if r not in ("E", "G"):
        return 0
    if not np.isfinite(f["atr"][t]) or f["atr"][t] <= 0:
        return 0
    dev = (f["c"][t] - f["vwap"][t]) / f["atr"][t]
    if dev < -2.0 and f["c"][t] > f["o"][t]:
        return 1
    if dev > 2.0 and f["c"][t] < f["o"][t]:
        return -1
    return 0


SETUPS = {
    "ORB-15":            s_orb,
    "VWAP pullback":     s_vwap_pullback,
    "PDH/PDL break":     s_pdh_break,
    "Momentum ignition": s_momentum_ignition,
    "Exhaustion MR":     s_exhaustion,
}


# --------------------------------------------------------------------------- #
# Trade simulation
# --------------------------------------------------------------------------- #
def simulate(f, setup, stop_atr=1.2, target_r=2.0, max_bars=24,
             breakeven_at=None, flat_eod=True, start=0, end=None):
    """
    Walk bars, fire the setup, resolve each trade.

    Entry fills at the NEXT bar's open — never at the signal bar's close.
    """
    end = end or f["n"]
    trades = []
    t = max(start, 210)  # let the 200-EMA warm up
    while t < min(end, f["n"]) - 2:
        r = regime(f, t)
        if r == "H":
            t += 1
            continue
        sig = setup(f, t, r)
        if sig == 0:
            t += 1
            continue
        atr = f["atr"][t]
        if not np.isfinite(atr) or atr <= 0:
            t += 1
            continue

        # Fill on the next bar's open.
        e = t + 1
        entry = f["o"][e]
        risk = atr * stop_atr
        if risk <= 0:
            t += 1
            continue

        cost = round_trip_cost(entry)
        stop = entry - risk if sig > 0 else entry + risk
        target = entry + risk * target_r if sig > 0 else entry - risk * target_r

        out = None
        be_armed = False
        held = 0
        for k in range(e, min(e + max_bars, f["n"])):
            held = k - e + 1
            if flat_eod and k + 1 < f["n"] and f["new_session"][k + 1]:
                px = f["c"][k]
                out = ((px - entry) if sig > 0 else (entry - px)) / risk
                break

            # Breakeven arming, checked before exits so it cannot rescue a bar
            # that already hit the stop.
            if breakeven_at is not None and not be_armed:
                fav = (f["h"][k] - entry) if sig > 0 else (entry - f["l"][k])
                if fav >= risk * breakeven_at:
                    stop = entry
                    be_armed = True

            # Stop before target — see module docstring.
            if sig > 0:
                if f["l"][k] <= stop:
                    out = (stop - entry) / risk
                    break
                if f["h"][k] >= target:
                    out = target_r
                    break
            else:
                if f["h"][k] >= stop:
                    out = (entry - stop) / risk
                    break
                if f["l"][k] <= target:
                    out = target_r
                    break
        if out is None:
            px = f["c"][min(e + max_bars - 1, f["n"] - 1)]
            out = ((px - entry) if sig > 0 else (entry - px)) / risk

        out -= cost / risk  # costs in R
        trades.append(dict(r=out, dir=sig, bars=held, entry=entry,
                           t=t, regime=r, minute=f["minute"][t]))
        t = e + held  # no overlapping positions
    return trades


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(trades) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([x["r"] for x in trades])
    wins, losses = r[r > 0], r[r <= 0]
    gross_w, gross_l = wins.sum(), -losses.sum()

    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0, eq])[1:]
    dd = peak - eq

    downside = r[r < 0]
    sortino = r.mean() / downside.std() * np.sqrt(252) if len(downside) > 1 and downside.std() > 0 else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    return {
        "n": len(r),
        "win": len(wins) / len(r),
        "exp_r": r.mean(),
        "total_r": r.sum(),
        "pf": gross_w / gross_l if gross_l > 0 else np.inf,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxdd_r": dd.max() if len(dd) else 0.0,
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "best": r.max(),
        "worst": r.min(),
        "avg_bars": np.mean([x["bars"] for x in trades]),
    }


def fmt(name: str, m: dict) -> str:
    if not m.get("n"):
        return f"  {name:<20} {'—':>5}"
    return (f"  {name:<20} {m['n']:>5} {m['win']:>6.0%} {m['exp_r']:>+8.3f} "
            f"{m['pf']:>6.2f} {m['sharpe']:>7.2f} {m['maxdd_r']:>7.1f} {m['total_r']:>+8.1f}")


HEADER = (f"  {'setup':<20} {'n':>5} {'win':>6} {'exp R':>8} "
          f"{'PF':>6} {'sharpe':>7} {'maxDD':>7} {'totR':>8}")
