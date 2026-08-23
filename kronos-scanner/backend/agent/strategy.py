# agent/strategy.py
"""
Strategy plug-point.

No trading strategy ships enabled here on purpose — this is the slot you drop
yours into. The agent's plumbing (data, patterns, quant engines, credit
governor, verdicts) is fully functional without one; only the final directional
call is deferred to whatever you implement.

To add yours:

    1. Subclass Strategy in a new file under agent/strategies/.
    2. Implement `screen()` (free, decides if a name deserves paid analysis)
       and `evaluate()` (produces the directional call).
    3. Point KRONOS_STRATEGY at it, e.g. KRONOS_STRATEGY=my_module:MyStrategy

`screen()` is the cost lever: anything it rejects never reaches a paid tier.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from .patterns import Pattern


@dataclass
class MarketContext:
    """Everything a strategy can see about one ticker. All computed for free."""

    ticker: str
    price: float
    # Candlestick evidence
    patterns: List[Pattern] = field(default_factory=list)
    pattern_bias: float = 0.0          # -1 (bearish) .. +1 (bullish)
    # Quant engine outputs
    ou_score: Optional[float] = None       # 0..100 mean-reversion pull
    ou_halflife: Optional[float] = None
    hmm_regime: Optional[str] = None       # Bull / Bear / HighVol / LowVol
    hmm_confidence: Optional[float] = None
    kalman_alpha: Optional[float] = None   # normalized drift
    evt_var_99: Optional[float] = None     # tail loss magnitude
    evt_expected_shortfall: Optional[float] = None
    # Structure
    pct_from_20d_high: Optional[float] = None
    pct_from_20d_low: Optional[float] = None
    atr_pct: Optional[float] = None        # ATR as fraction of price
    rsi_14: Optional[float] = None
    # Yahoo Finance news (populated only for candidates; free unless analysed)
    news_impact: Optional[str] = None      # "bullish" | "bearish" | "neutral"
    news_magnitude: float = 0.0            # 0..1 expected near-term price impact
    news_tradeable: bool = False
    news_note: str = ""
    headlines: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Compact one-line digest — this is what gets sent to the model."""
        bits = [f"{self.ticker} ${self.price:.2f}"]
        if self.patterns:
            bits.append("patterns=" + ",".join(f"{p.name}/{p.direction}" for p in self.patterns[:3]))
        bits.append(f"bias={self.pattern_bias:+.2f}")
        if self.ou_score is not None:
            bits.append(f"OU={self.ou_score:.0f}")
        if self.hmm_regime:
            bits.append(f"regime={self.hmm_regime}")
        if self.kalman_alpha is not None:
            bits.append(f"drift={self.kalman_alpha:+.4f}")
        if self.evt_var_99 is not None:
            bits.append(f"VaR99={self.evt_var_99*100:.2f}%")
        if self.rsi_14 is not None:
            bits.append(f"RSI={self.rsi_14:.0f}")
        if self.atr_pct is not None:
            bits.append(f"ATR={self.atr_pct*100:.2f}%")
        if self.news_impact and self.news_magnitude > 0:
            bits.append(f"news={self.news_impact}/{self.news_magnitude:.1f}")
        return " | ".join(bits)


@dataclass
class StrategySignal:
    """A strategy's directional read."""

    bias: str = "neutral"       # "long" | "short" | "neutral"
    conviction: float = 0.0     # 0..1
    rationale: str = ""
    entry: Optional[float] = None
    invalidation: Optional[float] = None
    target: Optional[float] = None

    @property
    def actionable(self) -> bool:
        return self.bias in ("long", "short") and self.conviction > 0.0


class Strategy(ABC):
    """Base class for a pluggable trading strategy."""

    name: str = "unnamed"

    def prefilter(self, ctx: MarketContext) -> bool:
        """
        FIRST gate, and the one that makes a large universe practical.

        Called with candlestick patterns and structure only — the quant engines
        have NOT run yet, because they cost ~167x more per ticker than pattern
        detection (50ms vs 0.3ms). Whatever this rejects never pays that cost,
        so scanning hundreds of names stays cheap.

        `ctx.ou_score`, `ctx.hmm_regime`, `ctx.kalman_alpha` and
        `ctx.evt_var_99` are all None here — do not read them.

        Default: require a real directional lean (|bias| >= 0.35), which passes
        roughly 45% of a broad universe and so halves engine work. Override
        this — a trend strategy in particular should decide here, from price
        structure alone, whether a name is even in the right regime to bother
        costing 50ms on.
        """
        return abs(ctx.pattern_bias) >= 0.35

    @abstractmethod
    def screen(self, ctx: MarketContext) -> bool:
        """
        SECOND gate. Full context — engines have run by now.

        Return True only if this name is worth spending tokens on. This is your
        primary cost control: the stricter this is, the fewer paid calls the
        agent makes. Aim to pass ~10-20% of the universe.
        """

    @abstractmethod
    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """Produce the directional call for a name that passed `screen()`."""

    def prompt_hint(self) -> str:
        """
        Optional extra instruction appended to the model prompt, letting you
        steer the paid tier without editing the agent.
        """
        return ""


class NeutralStrategy(Strategy):
    """
    Default placeholder. Deliberately takes NO directional view.

    It still exercises the whole pipeline: `screen()` promotes names showing
    genuine structural confluence, so you can watch the funnel work and see
    exactly what your strategy would receive. But `evaluate()` always abstains,
    so the agent will answer SKIP until you plug in real logic.
    """

    name = "neutral-placeholder"

    def screen(self, ctx: MarketContext) -> bool:
        # Promote only on agreement between an independent pattern read and
        # at least one quant engine — a neutral definition of "interesting".
        if abs(ctx.pattern_bias) < 0.35:
            return False

        agrees = 0
        if ctx.ou_score is not None and ctx.ou_score >= 60:
            agrees += 1
        if ctx.kalman_alpha is not None:
            drift_dir = 1 if ctx.kalman_alpha > 0 else -1
            pattern_dir = 1 if ctx.pattern_bias > 0 else -1
            if drift_dir == pattern_dir:
                agrees += 1
        if ctx.hmm_regime in ("Bull", "Bear"):
            agrees += 1
        return agrees >= 2

    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        return StrategySignal(
            bias="neutral",
            conviction=0.0,
            rationale=(
                "No strategy configured — structural candidate only. "
                "Set KRONOS_STRATEGY to enable directional calls."
            ),
        )


def load_strategy() -> Strategy:
    """
    Resolve the active strategy from KRONOS_STRATEGY ("module:ClassName").
    Falls back to NeutralStrategy when unset or on any import error.
    """
    spec = os.environ.get("KRONOS_STRATEGY", "").strip()
    if not spec:
        return NeutralStrategy()

    try:
        module_name, _, class_name = spec.partition(":")
        if not class_name:
            raise ValueError("expected format 'module:ClassName'")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        if not isinstance(instance, Strategy):
            raise TypeError(f"{spec} is not a Strategy subclass")
        return instance
    except Exception as exc:  # noqa: BLE001 - never let a bad plugin kill the agent
        print(f"[strategy] could not load '{spec}': {exc}. Using neutral placeholder.")
        return NeutralStrategy()
