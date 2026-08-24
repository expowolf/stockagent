# agent/strategies/episodic_pivot.py
"""
Episodic Pivot + consolidation breakout.

Implemented from the handwritten spec. Two setups, both swing-held 3-5 days
with a stop no wider than average daily range:

SETUP A — Episodic Pivot
    A neglected stock is repriced by a catalyst. Three ingredients:
      1. a major news item
      2. a big gap up
      3. a huge increase in volume (~10x average daily)
    The edge is in the word NEGLECTED: the name must have been going sideways
    for weeks, months or years before the pivot. A stock that has already been
    running is explicitly excluded — by then the repricing has happened and you
    are buying someone else's move.

SETUP B — Consolidation breakout
    Small/mid cap in an uptrend that goes flat on REDUCED volume, drawn from
    the strongest 1-2% of the universe over 1, 3 and 6 months. Volume drying up
    inside a tight range is the tell: supply has been absorbed.

Stops come from the spec: never wider than average daily range. Targets are
derived at 3x that distance to satisfy the risk gate's minimum reward-to-risk;
the spec sets the stop and the hold period, not the target, so that multiple is
a stated assumption rather than something read off the page.

News is NOT checked here. `evaluate()` runs before the news watcher (which only
fetches for candidates, to control cost), so catalyst confirmation is delegated
to the paid tier via `prompt_hint()`. For Setup A that confirmation matters: a
gap on volume with no news is a different, weaker setup.
"""

from __future__ import annotations

import os
from typing import Optional

from ..strategy import MarketContext, Strategy, StrategySignal


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class EpisodicPivotStrategy(Strategy):
    name = "episodic-pivot"

    # --- Setup A thresholds -------------------------------------------------
    MIN_CHANGE = _f("EP_MIN_CHANGE", 0.10)        # "up 10%"
    MIN_GAP = _f("EP_MIN_GAP", 0.03)              # a real gap, not just a rally
    MIN_VOL_RATIO = _f("EP_MIN_VOL_RATIO", 10.0)  # "10 times average daily volume"
    # Alternative qualifier for early session: already traded this many normal
    # FULL days' volume. Immune to extrapolation error.
    MIN_VOL_VS_FULL_DAY = _f("EP_MIN_VOL_VS_FULL_DAY", 1.5)
    MAX_PRIOR_RANGE = _f("EP_MAX_PRIOR_RANGE", 0.45)   # sideways before the pivot
    MAX_PRIOR_TREND = _f("EP_MAX_PRIOR_TREND", 0.25)   # net progress before the pivot
    MIN_QUIET_BARS = _f("EP_MIN_QUIET_BARS", 20)       # neglected, not recently active
    MAX_RECENT_RUNUP = _f("EP_MAX_RUNUP", 0.40)        # "don't trade stocks up for a bit"

    # --- Setup B thresholds -------------------------------------------------
    MIN_PERF_RANK = _f("EP_MIN_PERF_RANK", 0.98)  # top 1-2% of the universe
    MAX_TIGHT_RANGE = _f("EP_MAX_TIGHT_RANGE", 0.15)   # flat consolidation
    MAX_VOL_CONTRACTION = _f("EP_MAX_VOL_CONTRACTION", 0.85)  # volume drying up

    # --- shared -------------------------------------------------------------
    MIN_DOLLAR_VOLUME = _f("EP_MIN_DOLLAR_VOLUME", 2_000_000)
    TARGET_R_MULTIPLE = _f("EP_TARGET_R", 3.0)
    STOP_ADR_MULT = _f("EP_STOP_ADR_MULT", 1.0)   # "stop no longer than ADR"

    # ------------------------------------------------------------------ gates
    def prefilter(self, ctx: MarketContext) -> bool:
        """
        FREE gate over the whole universe. Engines have not run.

        Kept slightly looser than `screen()` on purpose: it costs nothing to let
        a marginal name through here, but rejecting a real pivot at this stage
        means never seeing it at all.
        """
        if ctx.dollar_volume is not None and ctx.dollar_volume < self.MIN_DOLLAR_VOLUME:
            return False

        # Setup A: a big up move on expanding volume.
        if (ctx.change_pct or 0) >= self.MIN_CHANGE * 0.8 and (ctx.volume_ratio or 0) >= 4.0:
            return True

        # Setup B: strong trailing performance with volume drying up.
        strong = max(ctx.perf_1m or 0, ctx.perf_3m or 0, ctx.perf_6m or 0)
        if strong >= 0.25 and (ctx.volume_contraction or 99) <= 1.0:
            return True

        return False

    def screen(self, ctx: MarketContext) -> bool:
        """Full context. Only genuine setups reach a paid call from here."""
        return bool(self._setup_a(ctx) or self._setup_b(ctx))

    # ------------------------------------------------------------------ setups
    def _setup_a(self, ctx: MarketContext) -> Optional[dict]:
        """Episodic Pivot: gap + volume on a previously ignored name."""
        if (ctx.change_pct or 0) < self.MIN_CHANGE:
            return None
        if (ctx.gap_pct or 0) < self.MIN_GAP:
            return None
        # Two independent readings of the volume surge, either of which
        # qualifies. Early in the session the projection is noisy, but having
        # already traded well over a normal FULL day is unambiguous on its own.
        projected_ok = (ctx.volume_ratio or 0) >= self.MIN_VOL_RATIO
        already_ok = (ctx.volume_vs_full_day or 0) >= self.MIN_VOL_VS_FULL_DAY
        if not (projected_ok or already_ok):
            return None

        # NEGLECT — the core of the setup, and the part worth being strict
        # about. Two independent conditions, BOTH required:
        #   range  — it oscillated in a band rather than trending
        #   trend  — it made little net progress
        # Range alone is not enough: a steady grind higher keeps a tight
        # rolling range while being precisely the name the spec rules out.
        ranged = (
            (ctx.prior_range_pct is not None and ctx.prior_range_pct <= self.MAX_PRIOR_RANGE)
            or (ctx.bars_since_big_move is not None and ctx.bars_since_big_move >= self.MIN_QUIET_BARS)
        )
        went_nowhere = (
            ctx.prior_trend_pct is None
            or abs(ctx.prior_trend_pct) <= self.MAX_PRIOR_TREND
        )
        if not (ranged and went_nowhere):
            return None

        # "Don't trade stocks that have been up for a bit." Today's own gap is
        # excluded from this test — we are asking what it did BEFORE the pivot.
        # Check the longest horizon available, not just one month: a slow
        # multi-month advance is still "already up", and a 21-bar window is
        # blind to it.
        for horizon in (ctx.perf_1m, ctx.perf_3m, ctx.perf_6m):
            if horizon is None:
                continue
            if horizon - (ctx.change_pct or 0) > self.MAX_RECENT_RUNUP:
                return None

        vol = ctx.volume_ratio or 0
        conviction = min(1.0, 0.55 + 0.03 * (vol - self.MIN_VOL_RATIO) + 0.8 * max(0.0, (ctx.change_pct or 0) - self.MIN_CHANGE))
        # Early in the session the volume read is a projection; temper
        # conviction rather than pretending the number is settled.
        if ctx.volume_partial and (ctx.session_elapsed or 1) < 0.25:
            conviction *= 0.85
        return {
            "setup": "A",
            "conviction": round(min(conviction, 0.95), 2),
            "why": (
                f"EP: +{(ctx.change_pct or 0)*100:.0f}% gap {(ctx.gap_pct or 0)*100:.0f}% "
                f"on {vol:.0f}x vol{' (proj)' if ctx.volume_partial else ''}, quiet before"
            ),
        }

    def _setup_b(self, ctx: MarketContext) -> Optional[dict]:
        """Consolidation breakout from a top-ranked performer."""
        if ctx.perf_rank is None or ctx.perf_rank < self.MIN_PERF_RANK:
            return None

        # Flat, tight range near the highs.
        near_high = ctx.pct_from_20d_high is not None and ctx.pct_from_20d_high >= -0.08
        tight = ctx.prior_range_pct is not None and ctx.prior_range_pct <= self.MAX_TIGHT_RANGE

        # Volume drying up is what separates a base from a stall.
        drying = (ctx.volume_contraction or 99) <= self.MAX_VOL_CONTRACTION
        if not (near_high and tight and drying):
            return None

        conviction = min(0.9, 0.5 + 0.4 * (ctx.perf_rank - self.MIN_PERF_RANK) / max(1e-6, 1 - self.MIN_PERF_RANK))
        return {
            "setup": "B",
            "conviction": round(conviction, 2),
            "why": f"breakout: rank {ctx.perf_rank:.0%}, {(ctx.prior_range_pct or 0)*100:.0f}% base, vol {(ctx.volume_contraction or 0):.2f}x",
        }

    # ------------------------------------------------------------------ signal
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        found = self._setup_a(ctx) or self._setup_b(ctx)
        if not found:
            return StrategySignal(bias="neutral", conviction=0.0, rationale="no setup")

        entry = ctx.price

        # Stop straight from the spec: no wider than average daily range.
        adr = ctx.adr_pct
        if not adr or adr <= 0:
            return StrategySignal(
                bias="neutral", conviction=0.0,
                rationale="no ADR available — cannot size a stop",
            )

        stop_distance = adr * self.STOP_ADR_MULT
        invalidation = entry * (1.0 - stop_distance)
        target = entry * (1.0 + stop_distance * self.TARGET_R_MULTIPLE)

        return StrategySignal(
            bias="long",
            conviction=found["conviction"],
            rationale=found["why"],
            entry=round(entry, 2),
            invalidation=round(invalidation, 2),
            target=round(target, 2),
        )

    def prompt_hint(self) -> str:
        return (
            "Setup: Episodic Pivot / consolidation breakout, swing held 3-5 days.\n"
            "An Episodic Pivot REQUIRES a genuine catalyst — earnings, guidance, "
            "FDA, M&A, a contract. If the news field is absent or the headlines are "
            "routine coverage, analyst chatter or a recap, answer SKIP: a gap on "
            "volume with no catalyst is a weaker, different setup.\n"
            "Also SKIP if the name had already run up substantially before this "
            "move — the repricing has happened and the edge is gone."
        )
