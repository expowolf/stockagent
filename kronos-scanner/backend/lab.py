#!/usr/bin/env python3
"""
Strategy lab.

Backtests candidate entry rules over the relay's history and reports what
actually matters: how often each fires, and what happened next.

Everything here is walk-forward in the trivial sense — each signal is judged
only on bars AFTER it fired — and exits use the same pessimistic convention as
the live journal: stop checked before target, because daily bars cannot say
which came first and assuming otherwise is how a backtest flatters itself.

Run:  python lab.py
"""

from __future__ import annotations

import numpy as np

from data.ingestion import load_universe, DEFAULT_UNIVERSE

HOLD = 5          # bars, matching the spec's 3-5 day swing
STOP_ADR = 1.0    # stop at 1x average daily range
TARGET_R = 3.0    # target at 3x the stop distance


def features(df):
    """Per-bar arrays the candidate rules read."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    if n < 130:
        return None

    prev = np.roll(c, 1)
    chg = np.where(prev > 0, c / prev - 1, 0.0)
    gap = np.where(prev > 0, o / prev - 1, 0.0)

    # rolling means without pandas overhead
    def roll_mean(a, w):
        out = np.full(len(a), np.nan)
        cs = np.cumsum(np.insert(a, 0, 0.0))
        out[w:] = (cs[w:-1] - cs[:-w - 1]) / w
        return out

    vol50 = roll_mean(v, 50)
    volr = np.where(vol50 > 0, v / vol50, 0.0)
    adr = roll_mean((h - l) / np.where(c > 0, c, np.nan), 20)

    sma20 = roll_mean(c, 20)
    sma50 = roll_mean(c, 50)

    # 20-day high/low excluding today
    hi20 = np.full(n, np.nan)
    lo20 = np.full(n, np.nan)
    for i in range(21, n):
        hi20[i] = h[i - 20:i].max()
        lo20[i] = l[i - 20:i].min()

    perf21 = np.full(n, np.nan)
    perf63 = np.full(n, np.nan)
    perf21[21:] = c[21:] / c[:-21] - 1
    perf63[63:] = c[63:] / c[:-63] - 1

    # RSI(14), Wilder
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = roll_mean(up, 14)
    ad = roll_mean(dn, 14)
    rsi = np.where(ad > 0, 100 - 100 / (1 + au / ad), 50.0)

    return dict(o=o, h=h, l=l, c=c, v=v, chg=chg, gap=gap, volr=volr, adr=adr,
                sma20=sma20, sma50=sma50, hi20=hi20, lo20=lo20,
                perf21=perf21, perf63=perf63, rsi=rsi, n=n)


# --------------------------------------------------------------------------- #
# Candidate entry rules. Each returns True to signal a long at bar t.
# --------------------------------------------------------------------------- #
def ep_strict(f, t):
    """The handwritten spec: gap + 10x volume on a neglected name."""
    if f["chg"][t] < 0.10 or f["gap"][t] < 0.03 or f["volr"][t] < 10:
        return False
    ph, pl = f["h"][t - 61:t].max(), f["l"][t - 61:t].min()
    mid = (ph + pl) / 2
    if mid <= 0 or (ph - pl) / mid > 0.45:
        return False
    return abs(f["c"][t - 1] / f["c"][t - 61] - 1) <= 0.25


def gap_and_go(f, t):
    """Looser cousin: a real gap on merely heavy volume, no neglect test."""
    return f["gap"][t] >= 0.03 and f["volr"][t] >= 3.0 and f["chg"][t] >= 0.04


def breakout_20d(f, t):
    """Close above the prior 20-day high on expanding volume, in an uptrend."""
    if np.isnan(f["hi20"][t]) or np.isnan(f["sma50"][t]):
        return False
    return (f["c"][t] > f["hi20"][t]
            and f["volr"][t] >= 2.0
            and f["c"][t] > f["sma50"][t])


def momentum_pullback(f, t):
    """Strong 3-month performer pulling back to its 20-day average."""
    if np.isnan(f["sma20"][t]) or np.isnan(f["perf63"][t]):
        return False
    near = abs(f["c"][t] / f["sma20"][t] - 1) < 0.03
    return (f["perf63"][t] >= 0.30
            and near
            and f["c"][t] > f["sma50"][t]
            and f["rsi"][t] < 65)


def oversold_bounce(f, t):
    """Short-term reversal: deeply oversold but still above the 50-day."""
    if np.isnan(f["sma50"][t]) or np.isnan(f["lo20"][t]):
        return False
    return (f["rsi"][t] < 30
            and f["c"][t] > f["sma50"][t]
            and f["volr"][t] >= 1.5)


CANDIDATES = {
    "EP (your spec)":      ep_strict,
    "gap-and-go":          gap_and_go,
    "20d breakout":        breakout_20d,
    "momentum pullback":   momentum_pullback,
    "oversold bounce":     oversold_bounce,
}


def simulate(rule, data):
    """Fire the rule across history; resolve each signal on later bars."""
    rs, n_sig, n_bars = [], 0, 0

    for tk, df in data.items():
        f = features(df)
        if f is None:
            continue
        n = f["n"]
        t = 70
        while t < n - 1:
            n_bars += 1
            try:
                fired = rule(f, t)
            except Exception:  # noqa: BLE001
                fired = False
            if not fired:
                t += 1
                continue

            adr = f["adr"][t]
            if not np.isfinite(adr) or adr <= 0:
                t += 1
                continue

            entry = f["c"][t]
            stop = entry * (1 - adr * STOP_ADR)
            target = entry * (1 + adr * STOP_ADR * TARGET_R)
            risk = entry - stop
            n_sig += 1

            outcome = None
            for k in range(t + 1, min(t + 1 + HOLD, n)):
                # Stop first — see module docstring.
                if f["l"][k] <= stop:
                    outcome = -1.0
                    break
                if f["h"][k] >= target:
                    outcome = TARGET_R
                    break
            if outcome is None:
                last = f["c"][min(t + HOLD, n - 1)]
                outcome = (last - entry) / risk
            rs.append(outcome)

            # Skip the hold window so overlapping signals on one name are not
            # counted as independent trades.
            t += HOLD
        # end per-ticker
    return rs, n_sig, n_bars


def main() -> int:
    data = load_universe(DEFAULT_UNIVERSE, period="2y")
    usable = {k: v for k, v in data.items() if v is not None and len(v) >= 130}
    if not usable:
        print("no usable history — is the relay populated?")
        return 2

    days = int(np.median([len(v) for v in usable.values()]))
    print(f"lab: {len(usable)} names, ~{days} bars each "
          f"({days/21:.0f} months), hold {HOLD}d, stop 1xADR, target {TARGET_R}R\n")
    print(f"{'strategy':<20} {'signals':>8} {'/month':>7} {'win':>6} "
          f"{'avg R':>7} {'total R':>8}")
    print("-" * 60)

    for name, rule in CANDIDATES.items():
        rs, n_sig, _ = simulate(rule, usable)
        if not rs:
            print(f"{name:<20} {0:>8} {0:>7.1f} {'—':>6} {'—':>7} {'—':>8}")
            continue
        arr = np.array(rs)
        per_month = len(arr) / (days / 21)
        print(f"{name:<20} {len(arr):>8} {per_month:>7.1f} "
              f"{(arr > 0).mean():>5.0%} {arr.mean():>7.2f} {arr.sum():>8.1f}")

    print("\nnotes:")
    print("  avg R is per trade; positive means the rule made money in this window.")
    print("  One window is not proof — regimes change, and none of this is")
    print("  out-of-sample against a period I have not already looked at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
