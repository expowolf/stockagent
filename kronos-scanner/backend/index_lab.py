#!/usr/bin/env python3
"""
Index lab — candidate intraday strategies for QQQ / SPY.

Two ETFs instead of a wide universe changes the problem completely. There is no
screening to do; the edge has to come from timing, and the only honest way to
choose a rule is to measure it.

Design constraints from the user:
  - aggressive: take real risk to make real money
  - tight stop: a wrong trade must be cheap
  - QQQ and SPY only

Exits use the live journal's pessimistic convention: within a bar the stop is
assumed to hit before the target, because intrabar order is unknowable and
assuming otherwise is how a backtest flatters itself.

Run:  python index_lab.py
"""

from __future__ import annotations

import numpy as np
import yfinance as yf

TICKERS = ["QQQ", "SPY"]


def load(ticker: str, interval: str, period: str):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "levels"):
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower)
    return df.dropna(subset=["close"])


def feats(df):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)

    def rmean(a, w):
        out = np.full(n, np.nan)
        cs = np.cumsum(np.insert(np.nan_to_num(a), 0, 0.0))
        out[w:] = (cs[w:-1] - cs[:-w - 1]) / w
        return out

    def rmax(a, w):
        out = np.full(n, np.nan)
        for i in range(w, n):
            out[i] = a[i - w:i].max()
        return out

    def rmin(a, w):
        out = np.full(n, np.nan)
        for i in range(w, n):
            out[i] = a[i - w:i].min()
        return out

    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    atr = rmean(tr, 14)

    d = np.diff(c, prepend=c[0])
    rs_up = rmean(np.where(d > 0, d, 0.0), 14)
    rs_dn = rmean(np.where(d < 0, -d, 0.0), 14)
    rsi = np.where(rs_dn > 0, 100 - 100 / (1 + rs_up / rs_dn), 50.0)

    ema9 = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    k9, k21 = 2 / 10, 2 / 22
    ema9[0] = ema21[0] = c[0]
    for i in range(1, n):
        ema9[i] = c[i] * k9 + ema9[i - 1] * (1 - k9)
        ema21[i] = c[i] * k21 + ema21[i - 1] * (1 - k21)

    sd = np.full(n, np.nan)
    for i in range(20, n):
        sd[i] = c[i - 20:i].std()
    sma20 = rmean(c, 20)

    return dict(o=o, h=h, l=l, c=c, v=v, n=n, atr=atr, rsi=rsi,
                ema9=ema9, ema21=ema21, sma20=sma20, sd=sd,
                hi20=rmax(h, 20), lo20=rmin(l, 20),
                hi10=rmax(h, 10), lo10=rmin(l, 10),
                vol20=rmean(v, 20))


# --------------------------------------------------------------------------- #
# Candidates. Each returns +1 (long), -1 (short), or 0.
# Long-only is enforced later; shorts are measured to see if the signal has
# information in both directions.
# --------------------------------------------------------------------------- #
def opening_drive(f, t):
    """First bar of strength continuing — classic momentum ignition."""
    if not np.isfinite(f["atr"][t]) or not np.isfinite(f["hi10"][t]):
        return 0
    thrust = f["c"][t] - f["o"][t]
    if thrust > 0.6 * f["atr"][t] and f["c"][t] > f["hi10"][t] and f["v"][t] > 1.3 * f["vol20"][t]:
        return 1
    if -thrust > 0.6 * f["atr"][t] and f["c"][t] < f["lo10"][t] and f["v"][t] > 1.3 * f["vol20"][t]:
        return -1
    return 0


def ema_pullback(f, t):
    """Trend-following: 9>21 and price pulls back to the 9 then resumes."""
    if not np.isfinite(f["ema21"][t]) or not np.isfinite(f["atr"][t]):
        return 0
    up = f["ema9"][t] > f["ema21"][t]
    dn = f["ema9"][t] < f["ema21"][t]
    touched = f["l"][t] <= f["ema9"][t] <= f["h"][t]
    if up and touched and f["c"][t] > f["ema9"][t]:
        return 1
    if dn and touched and f["c"][t] < f["ema9"][t]:
        return -1
    return 0


def vwap_reversion(f, t):
    """Mean reversion: stretched 2+ SD from the 20-bar mean, snapping back."""
    if not np.isfinite(f["sd"][t]) or f["sd"][t] <= 0:
        return 0
    z = (f["c"][t] - f["sma20"][t]) / f["sd"][t]
    if z < -2.0 and f["c"][t] > f["o"][t]:
        return 1
    if z > 2.0 and f["c"][t] < f["o"][t]:
        return -1
    return 0


def range_breakout(f, t):
    """Break of the prior 20-bar range on volume."""
    if not np.isfinite(f["hi20"][t]):
        return 0
    if f["c"][t] > f["hi20"][t] and f["v"][t] > 1.2 * f["vol20"][t]:
        return 1
    if f["c"][t] < f["lo20"][t] and f["v"][t] > 1.2 * f["vol20"][t]:
        return -1
    return 0


def squeeze_break(f, t):
    """Volatility contraction then expansion — coiled spring."""
    if t < 40 or not np.isfinite(f["sd"][t]) or not np.isfinite(f["atr"][t]):
        return 0
    recent_sd = f["sd"][t]
    prior_sd = np.nanmean(f["sd"][t - 20:t])
    if not np.isfinite(prior_sd) or prior_sd <= 0:
        return 0
    coiled = recent_sd < 0.8 * prior_sd
    if not coiled:
        return 0
    if f["c"][t] > f["hi10"][t]:
        return 1
    if f["c"][t] < f["lo10"][t]:
        return -1
    return 0


CANDIDATES = {
    "opening drive":   opening_drive,
    "ema pullback":    ema_pullback,
    "2SD reversion":   vwap_reversion,
    "range breakout":  range_breakout,
    "squeeze break":   squeeze_break,
}


def simulate(rule, f, stop_atr, target_r, max_hold, allow_short=True):
    """Fire the rule bar by bar; resolve on later bars, stop before target."""
    rs = []
    t = 30
    n = f["n"]
    while t < n - 1:
        sig = rule(f, t)
        if sig == 0 or (sig < 0 and not allow_short):
            t += 1
            continue
        atr = f["atr"][t]
        if not np.isfinite(atr) or atr <= 0:
            t += 1
            continue

        entry = f["c"][t]
        risk = atr * stop_atr
        if sig > 0:
            stop, target = entry - risk, entry + risk * target_r
        else:
            stop, target = entry + risk, entry - risk * target_r

        out = None
        for k in range(t + 1, min(t + 1 + max_hold, n)):
            if sig > 0:
                if f["l"][k] <= stop:
                    out = -1.0; break
                if f["h"][k] >= target:
                    out = target_r; break
            else:
                if f["h"][k] >= stop:
                    out = -1.0; break
                if f["l"][k] <= target:
                    out = target_r; break
        if out is None:
            last = f["c"][min(t + max_hold, n - 1)]
            out = ((last - entry) if sig > 0 else (entry - last)) / risk
        rs.append(out)
        t += max_hold  # no overlapping positions
    return rs


def report(title, results):
    print(f"\n{title}")
    print(f"  {'strategy':<16} {'n':>5} {'win':>6} {'avg R':>7} {'total R':>8} {'expect':>8}")
    print("  " + "-" * 54)
    ranked = sorted(results.items(), key=lambda kv: -(np.mean(kv[1]) if kv[1] else -9))
    for name, rs in ranked:
        if not rs:
            print(f"  {name:<16} {0:>5}")
            continue
        a = np.array(rs)
        print(f"  {name:<16} {len(a):>5} {(a>0).mean():>5.0%} {a.mean():>7.3f} "
              f"{a.sum():>8.1f} {a.mean()*len(a)/max(len(a),1):>8.3f}")


def main() -> int:
    # 30-minute bars over 60 days: enough samples to separate signal from noise,
    # and a timeframe a swing-ish trader can actually act on.
    for interval, period, hold in (("30m", "60d", 6), ("1h", "180d", 5)):
        for tk in TICKERS:
            df = load(tk, interval, period)
            if df is None or len(df) < 100:
                print(f"{tk} {interval}: no data")
                continue
            f = feats(df)
            res = {n: simulate(r, f, stop_atr=1.0, target_r=2.0, max_hold=hold)
                   for n, r in CANDIDATES.items()}
            report(f"{tk} {interval} ({len(df)} bars) — stop 1xATR, target 2R, hold {hold}", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
