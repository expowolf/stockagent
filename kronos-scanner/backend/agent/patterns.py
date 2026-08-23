# agent/patterns.py
"""
Candlestick pattern detection — pure numpy, zero token cost.

This is the free tier of the funnel: every ticker is screened here so that
the paid tiers only ever see a handful of survivors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Pattern:
    name: str
    direction: str   # "bullish" | "bearish"
    strength: float  # 0..1
    bars_ago: int

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{self.name}({self.direction}, {self.strength:.2f}, -{self.bars_ago})"


def _parts(o, h, l, c):
    """Body, shadows and range for a candle (arrays or scalars)."""
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return body, rng, upper, lower


def _trend_before(closes: np.ndarray, idx: int, lookback: int = 5) -> str:
    """Crude prior-trend read used to qualify reversal patterns."""
    start = max(0, idx - lookback)
    if idx - start < 2:
        return "flat"
    seg = closes[start:idx]
    slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    scale = np.mean(np.abs(seg)) + 1e-12
    norm = slope / scale
    if norm > 0.002:
        return "up"
    if norm < -0.002:
        return "down"
    return "flat"


def detect(ohlc: dict, lookback_bars: int = 5) -> List[Pattern]:
    """
    Detect candlestick patterns in the last `lookback_bars` bars.

    `ohlc` maps "open"/"high"/"low"/"close" to equal-length sequences.
    Returns patterns found, most recent first.
    """
    o = np.asarray(ohlc["open"], dtype=float)
    h = np.asarray(ohlc["high"], dtype=float)
    l = np.asarray(ohlc["low"], dtype=float)
    c = np.asarray(ohlc["close"], dtype=float)

    n = len(c)
    if n < 5:
        return []

    found: List[Pattern] = []
    start = max(3, n - lookback_bars)

    for i in range(start, n):
        bars_ago = n - 1 - i
        body, rng, upper, lower = _parts(o[i], h[i], l[i], c[i])
        pbody, prng, pupper, plower = _parts(o[i - 1], h[i - 1], l[i - 1], c[i - 1])
        prior = _trend_before(c, i)

        bullish_bar = c[i] > o[i]
        prev_bullish = c[i - 1] > o[i - 1]

        # --- Doji: indecision, only meaningful after a run -------------------
        if body <= 0.1 * rng and prior in ("up", "down"):
            found.append(
                Pattern(
                    "doji",
                    "bullish" if prior == "down" else "bearish",
                    min(1.0, 0.45 + 0.3 * (1.0 - body / rng)),
                    bars_ago,
                )
            )

        # --- Hammer / Shooting star -----------------------------------------
        # Shadow limits are measured against the bar RANGE, not the body: with a
        # small body, a body-relative limit is unsatisfiable and the pattern can
        # never fire.
        if lower >= 2.0 * body and upper <= 0.15 * rng and prior == "down":
            found.append(Pattern("hammer", "bullish", min(1.0, 0.55 + 0.4 * (lower / rng)), bars_ago))

        if upper >= 2.0 * body and lower <= 0.15 * rng and prior == "up":
            found.append(Pattern("shooting_star", "bearish", min(1.0, 0.55 + 0.4 * (upper / rng)), bars_ago))

        # --- Engulfing --------------------------------------------------------
        if bullish_bar and not prev_bullish and c[i] >= o[i - 1] and o[i] <= c[i - 1] and body > pbody:
            found.append(Pattern("bullish_engulfing", "bullish", min(1.0, 0.6 + 0.25 * (body / (pbody + 1e-12) - 1)), bars_ago))

        if (not bullish_bar) and prev_bullish and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body > pbody:
            found.append(Pattern("bearish_engulfing", "bearish", min(1.0, 0.6 + 0.25 * (body / (pbody + 1e-12) - 1)), bars_ago))

        # --- Harami (inside bar) ---------------------------------------------
        if body < pbody * 0.6 and max(o[i], c[i]) <= max(o[i - 1], c[i - 1]) and min(o[i], c[i]) >= min(o[i - 1], c[i - 1]):
            found.append(
                Pattern(
                    "harami",
                    "bullish" if not prev_bullish else "bearish",
                    0.45,
                    bars_ago,
                )
            )

        # --- Piercing line / Dark cloud cover ---------------------------------
        pmid = (o[i - 1] + c[i - 1]) / 2.0
        if bullish_bar and not prev_bullish and o[i] < l[i - 1] and c[i] > pmid and c[i] < o[i - 1]:
            found.append(Pattern("piercing_line", "bullish", 0.6, bars_ago))

        if (not bullish_bar) and prev_bullish and o[i] > h[i - 1] and c[i] < pmid and c[i] > o[i - 1]:
            found.append(Pattern("dark_cloud_cover", "bearish", 0.6, bars_ago))

        # --- Marubozu (conviction bar) ---------------------------------------
        if body >= 0.9 * rng:
            found.append(
                Pattern(
                    "marubozu",
                    "bullish" if bullish_bar else "bearish",
                    min(1.0, 0.5 + 0.4 * (body / rng)),
                    bars_ago,
                )
            )

        # --- Morning / Evening star (3-bar) -----------------------------------
        if i >= 2:
            b2, r2, _, _ = _parts(o[i - 2], h[i - 2], l[i - 2], c[i - 2])
            mid2 = (o[i - 2] + c[i - 2]) / 2.0
            small_middle = pbody <= 0.5 * b2

            if c[i - 2] < o[i - 2] and small_middle and bullish_bar and c[i] > mid2:
                found.append(Pattern("morning_star", "bullish", 0.75, bars_ago))

            if c[i - 2] > o[i - 2] and small_middle and (not bullish_bar) and c[i] < mid2:
                found.append(Pattern("evening_star", "bearish", 0.75, bars_ago))

        # --- Three white soldiers / black crows -------------------------------
        if i >= 2:
            up3 = all(c[j] > o[j] for j in (i - 2, i - 1, i))
            rising = c[i] > c[i - 1] > c[i - 2]
            if up3 and rising:
                found.append(Pattern("three_white_soldiers", "bullish", 0.8, bars_ago))

            dn3 = all(c[j] < o[j] for j in (i - 2, i - 1, i))
            falling = c[i] < c[i - 1] < c[i - 2]
            if dn3 and falling:
                found.append(Pattern("three_black_crows", "bearish", 0.8, bars_ago))

    # Most recent first, strongest first within the same bar.
    found.sort(key=lambda p: (p.bars_ago, -p.strength))
    return found


def dedupe(patterns: List[Pattern]) -> List[Pattern]:
    """
    Keep only the most recent occurrence of each pattern name.

    Multi-bar patterns (three_white_soldiers, three_black_crows) are detected on
    every overlapping window, so the same run would otherwise be counted two or
    three times and swamp a fresher single-bar reversal.
    """
    best: dict = {}
    for p in patterns:
        prior = best.get(p.name)
        if prior is None or p.bars_ago < prior.bars_ago:
            best[p.name] = p
    out = list(best.values())
    out.sort(key=lambda p: (p.bars_ago, -p.strength))
    return out


def net_bias(patterns: List[Pattern], decay: float = 0.55) -> float:
    """
    Collapse detected patterns into one score in [-1, 1].

    Patterns are de-duplicated first, then weighted by recency. The decay is
    steep on purpose: a reversal printing today should outweigh the trend it
    reverses, which is the entire point of a reversal signal.
    """
    if not patterns:
        return 0.0

    score = 0.0
    weight_sum = 0.0
    for p in dedupe(patterns):
        w = (decay ** p.bars_ago) * p.strength
        score += w if p.direction == "bullish" else -w
        weight_sum += w

    if weight_sum == 0:
        return 0.0
    return float(np.clip(score / weight_sum, -1.0, 1.0))
