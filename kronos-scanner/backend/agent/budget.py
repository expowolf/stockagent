# agent/budget.py
"""
Credit governor.

Keeps an all-morning scanning session inside a fixed usage window (default 5h)
by pacing spend against elapsed time instead of burning budget front-loaded.

Two guarantees:
  1. A hard reserve is never touched, so the window cannot be exhausted.
  2. When budget runs out the agent does NOT stop — it degrades to Tier.NONE
     (pure-math verdicts), which costs nothing and still produces decisions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Tier(str, Enum):
    """Escalation tiers, cheapest first."""

    NONE = "none"    # pure numeric reasoning — zero tokens
    CHEAP = "cheap"  # small model, short structured prompt
    DEEP = "deep"    # larger model, only for genuine finalists


# Model per tier. NONE never calls the API.
TIER_MODEL = {
    Tier.CHEAP: "claude-haiku-4-5-20251001",
    Tier.DEEP: "claude-sonnet-5",
}

# Pre-flight cost estimates (input+output) used only for affordability checks.
# Real spend is recorded from API usage after each call.
TIER_EST_TOKENS = {
    Tier.NONE: 0,
    Tier.CHEAP: 1_100,
    Tier.DEEP: 3_500,
}


@dataclass
class BudgetState:
    window_started_at: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    throttled: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CreditGovernor:
    """
    Paces token spend across a rolling usage window.

    Args:
        token_budget: total tokens allowed in the window.
        window_hours: length of the usage window (Claude resets on 5h windows).
        reserve_pct:  fraction held back and never spent (safety margin).
        slack:        tolerance band around the time-proportional pace target
                      before throttling up or down.
        state_path:   where to persist state so restarts keep their pacing.
    """

    def __init__(
        self,
        token_budget: int = 400_000,
        window_hours: float = 5.0,
        reserve_pct: float = 0.20,
        slack: float = 0.08,
        state_path: Optional[str] = None,
    ) -> None:
        self.token_budget = int(token_budget)
        self.window_seconds = float(window_hours) * 3600.0
        self.reserve_pct = float(reserve_pct)
        self.slack = float(slack)
        self.state_path = Path(
            state_path
            or os.environ.get("KRONOS_BUDGET_STATE", "/tmp/kronos_budget.json")
        )
        self.state = self._load()

    # ----------------------------------------------------------------- state
    def _load(self) -> BudgetState:
        try:
            raw = json.loads(self.state_path.read_text())
            state = BudgetState(**raw)
        except Exception:  # noqa: BLE001 - any problem means start fresh
            return BudgetState()

        # Roll the window over if the previous one has expired.
        if time.time() - state.window_started_at >= self.window_seconds:
            return BudgetState()
        return state

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state.__dict__))
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            print(f"[budget] could not persist state: {exc}")

    def _roll_if_expired(self) -> None:
        if time.time() - self.state.window_started_at >= self.window_seconds:
            self.state = BudgetState()
            self._save()

    # ------------------------------------------------------------- accounting
    @property
    def spendable(self) -> int:
        """Budget minus the untouchable reserve."""
        return int(self.token_budget * (1.0 - self.reserve_pct))

    @property
    def spent(self) -> int:
        return self.state.total_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.spendable - self.spent)

    @property
    def elapsed_fraction(self) -> float:
        """How far through the window we are, in [0, 1]."""
        self._roll_if_expired()
        elapsed = time.time() - self.state.window_started_at
        return min(1.0, max(0.0, elapsed / self.window_seconds))

    @property
    def spent_fraction(self) -> float:
        if self.spendable <= 0:
            return 1.0
        return min(1.0, self.spent / self.spendable)

    @property
    def pace(self) -> float:
        """
        Spend pace relative to time.

        < 1.0 means under-spending (room to escalate).
        > 1.0 means over-spending (throttle down).
        """
        elapsed = self.elapsed_fraction
        if elapsed <= 1e-6:
            # At window start, treat any spend as ahead of pace.
            return 0.0 if self.spent == 0 else 2.0
        return self.spent_fraction / elapsed

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record real usage from an API response."""
        self._roll_if_expired()
        self.state.input_tokens += int(input_tokens)
        self.state.output_tokens += int(output_tokens)
        self.state.calls += 1
        self._save()

    def note_throttle(self) -> None:
        self.state.throttled += 1
        self._save()

    # -------------------------------------------------------------- decisions
    def can_afford(self, tier: Tier) -> bool:
        est = TIER_EST_TOKENS.get(tier, 0)
        if est == 0:
            return True
        return self.remaining >= est

    def allow(self, tier: Tier) -> bool:
        """Whether a specific tier is permitted right now."""
        if tier is Tier.NONE:
            return True
        if not self.can_afford(tier):
            return False
        return tier in self._permitted_tiers()

    def _permitted_tiers(self) -> set:
        """Tiers allowed at the current pace."""
        p = self.pace
        if p > 1.0 + self.slack:
            # Ahead of budget — numeric only until time catches up.
            return {Tier.NONE}
        if p < 1.0 - self.slack:
            # Behind budget — we can afford depth.
            return {Tier.NONE, Tier.CHEAP, Tier.DEEP}
        return {Tier.NONE, Tier.CHEAP}

    def plan_tier(self, desired: Tier) -> Tier:
        """
        Downgrade `desired` to the best tier currently affordable.
        Never raises, never blocks — worst case returns Tier.NONE.
        """
        self._roll_if_expired()
        permitted = self._permitted_tiers()

        for tier in (desired, Tier.CHEAP, Tier.NONE):
            if tier in permitted and self.can_afford(tier):
                if tier is not desired:
                    self.note_throttle()
                return tier
        self.note_throttle()
        return Tier.NONE

    def seconds_remaining(self) -> float:
        self._roll_if_expired()
        elapsed = time.time() - self.state.window_started_at
        return max(0.0, self.window_seconds - elapsed)

    def suggest_interval(self, planned_cycles: int = 36) -> float:
        """
        Suggested seconds between scans so the remaining window is covered
        by roughly `planned_cycles` more scans.
        """
        left = self.seconds_remaining()
        if planned_cycles <= 0:
            return 300.0
        return max(60.0, left / planned_cycles)

    # ------------------------------------------------------------------ views
    def snapshot(self) -> dict:
        """Human/API readable status — surfaced in the phone UI."""
        self._roll_if_expired()
        return {
            "spent_tokens": self.spent,
            "spendable_tokens": self.spendable,
            "remaining_tokens": self.remaining,
            "reserve_tokens": self.token_budget - self.spendable,
            "percent_used": round(100.0 * self.spent_fraction, 1),
            "percent_elapsed": round(100.0 * self.elapsed_fraction, 1),
            "pace": round(self.pace, 2),
            "status": self.status(),
            "calls": self.state.calls,
            "throttled_calls": self.state.throttled,
            "permitted_tiers": sorted(t.value for t in self._permitted_tiers()),
            "window_resets_in_minutes": round(self.seconds_remaining() / 60.0, 1),
        }

    def status(self) -> str:
        if self.remaining <= 0:
            return "exhausted-degraded-to-free"
        p = self.pace
        if p > 1.0 + self.slack:
            return "throttling"
        if p < 1.0 - self.slack:
            return "headroom"
        return "on-pace"
