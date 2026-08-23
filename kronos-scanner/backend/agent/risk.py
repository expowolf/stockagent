# agent/risk.py
"""
Risk gate.

This enforces well-established trade discipline and is deliberately separate
from strategy: it never decides *what* to trade, only whether a proposed trade
is structured survivably. Your entry logic stays entirely yours.

The rules here are the boring, consensus ones that keep accounts alive:

  1. No stop, no trade.       A position without a predefined invalidation is
                              not a trade, it is exposure.
  2. Fixed fractional risk.   Risk a small, constant fraction of the account
                              per trade (the "1% rule"), sized to the stop.
  3. Minimum reward-to-risk.  Don't pay more than you stand to make.
  4. Stop outside noise.      A stop inside average daily range is a donation
                              to market makers — you get wicked out and the
                              thesis never had a chance to play out.

Rule 4 is where the EVT and ATR engines earn their keep: they tell us what
"noise" actually measures on this instrument today.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .strategy import MarketContext, StrategySignal


@dataclass
class RiskAssessment:
    approved: bool
    reason: str
    rr: Optional[float] = None            # reward-to-risk ratio
    stop_pct: Optional[float] = None      # stop distance as fraction of entry
    shares: Optional[int] = None          # position size under the risk rule
    risk_amount: Optional[float] = None   # dollars at risk if stopped
    warnings: Optional[list] = None

    def line(self) -> str:
        """One-line summary for a notification."""
        if not self.approved:
            return f"blocked: {self.reason}"
        bits = []
        if self.rr is not None:
            bits.append(f"R:R {self.rr:.1f}")
        if self.stop_pct is not None:
            bits.append(f"risk {self.stop_pct*100:.1f}%")
        if self.shares:
            bits.append(f"{self.shares} sh")
        return " · ".join(bits)


@dataclass
class RiskPolicy:
    """Tunable, but the defaults are the conventional ones."""

    account_size: float = 10_000.0
    risk_per_trade_pct: float = 0.01   # 1% of account per trade
    min_rr: float = 1.5                # never pay 1 to make less than 1.5
    max_stop_pct: float = 0.10         # a >10% stop is a position-size problem
    min_stop_pct: float = 0.003        # a <0.3% stop is inside the spread
    min_stop_atr_mult: float = 0.5     # stop must clear half the daily range
    veto_inside_noise: bool = True

    @classmethod
    def from_env(cls) -> "RiskPolicy":
        def _f(name, default):
            try:
                return float(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(
            account_size=_f("KRONOS_ACCOUNT_SIZE", 10_000.0),
            risk_per_trade_pct=_f("KRONOS_RISK_PER_TRADE", 0.01),
            min_rr=_f("KRONOS_MIN_RR", 1.5),
            max_stop_pct=_f("KRONOS_MAX_STOP_PCT", 0.10),
            veto_inside_noise=os.environ.get("KRONOS_VETO_NOISE", "1") != "0",
        )

    # ------------------------------------------------------------------ assess
    def assess(self, sig: StrategySignal, ctx: MarketContext) -> RiskAssessment:
        """Evaluate a proposed trade. Returns approval plus sizing."""
        warnings: list = []

        entry = sig.entry if sig.entry else ctx.price
        stop = sig.invalidation

        # Rule 1 — no stop, no trade. Non-negotiable.
        if stop is None:
            return RiskAssessment(
                approved=False,
                reason="no stop loss defined",
                warnings=warnings,
            )

        if entry is None or entry <= 0:
            return RiskAssessment(approved=False, reason="no valid entry price")

        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return RiskAssessment(approved=False, reason="stop equals entry")

        # Stop on the correct side of entry for the direction.
        if sig.bias == "long" and stop >= entry:
            return RiskAssessment(approved=False, reason="long stop is above entry")
        if sig.bias == "short" and stop <= entry:
            return RiskAssessment(approved=False, reason="short stop is below entry")

        stop_pct = risk_per_share / entry

        # Rule 4 — stop must sit outside normal noise.
        if self.veto_inside_noise and ctx.atr_pct:
            floor = self.min_stop_atr_mult * ctx.atr_pct
            if stop_pct < floor:
                return RiskAssessment(
                    approved=False,
                    reason=f"stop inside noise ({stop_pct*100:.1f}% < {floor*100:.1f}% of ATR)",
                    stop_pct=stop_pct,
                    warnings=warnings,
                )

        # A stop tighter than the 99% daily tail is survivable but fragile.
        if ctx.evt_var_99 and stop_pct < ctx.evt_var_99:
            warnings.append(
                f"stop ({stop_pct*100:.1f}%) inside 99% daily tail "
                f"({ctx.evt_var_99*100:.1f}%) — elevated wick-out risk"
            )

        if stop_pct > self.max_stop_pct:
            return RiskAssessment(
                approved=False,
                reason=f"stop too wide ({stop_pct*100:.1f}% > {self.max_stop_pct*100:.0f}%)",
                stop_pct=stop_pct,
                warnings=warnings,
            )
        if stop_pct < self.min_stop_pct:
            return RiskAssessment(
                approved=False,
                reason=f"stop too tight ({stop_pct*100:.2f}%)",
                stop_pct=stop_pct,
                warnings=warnings,
            )

        # Rule 3 — reward must justify the risk.
        rr = None
        if sig.target:
            reward = abs(sig.target - entry)
            rr = reward / risk_per_share
            if rr < self.min_rr:
                return RiskAssessment(
                    approved=False,
                    reason=f"R:R {rr:.1f} below {self.min_rr:.1f} minimum",
                    rr=rr,
                    stop_pct=stop_pct,
                    warnings=warnings,
                )
        else:
            warnings.append("no target — R:R unverified")

        # Rule 2 — fixed fractional sizing off the stop, not off conviction.
        risk_budget = self.account_size * self.risk_per_trade_pct
        shares = int(risk_budget // risk_per_share)
        if shares < 1:
            return RiskAssessment(
                approved=False,
                reason=(
                    f"position size < 1 share at {self.risk_per_trade_pct*100:.0f}% risk "
                    f"(needs ${risk_per_share:.2f}/share)"
                ),
                rr=rr,
                stop_pct=stop_pct,
                warnings=warnings,
            )

        return RiskAssessment(
            approved=True,
            reason="risk checks passed",
            rr=rr,
            stop_pct=stop_pct,
            shares=shares,
            risk_amount=round(shares * risk_per_share, 2),
            warnings=warnings,
        )
