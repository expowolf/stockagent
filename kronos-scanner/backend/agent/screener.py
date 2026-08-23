# agent/screener.py
"""
The cost funnel.

Stage 0  load cached OHLCV                      free
Stage 1  candlestick patterns + quant engines   free
Stage 2  strategy.screen()                      free
Stage 3  model call on survivors only           paid, governed

Only names surviving stages 1-2 ever reach a paid tier, so a 20-ticker sweep
typically costs one or two small calls rather than twenty.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .budget import CreditGovernor, Tier, TIER_MODEL
from .patterns import dedupe, detect, net_bias
from .strategy import MarketContext, Strategy, StrategySignal, load_strategy

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


@dataclass
class Verdict:
    """The only thing the phone actually shows: take it or don't."""

    ticker: str
    decision: str          # "TAKE" | "SKIP"
    confidence: float      # 0..1
    reason: str            # one line
    price: float
    tier: str              # which tier produced this
    tokens: int = 0
    entry: Optional[float] = None
    invalidation: Optional[float] = None
    target: Optional[float] = None
    at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _structure(df) -> dict:
    """Cheap structural context: range position, ATR, RSI."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    out: dict = {}
    if len(close) >= 20:
        win_h = float(np.max(high[-20:]))
        win_l = float(np.min(low[-20:]))
        price = float(close[-1])
        out["pct_from_20d_high"] = (price - win_h) / win_h if win_h else None
        out["pct_from_20d_low"] = (price - win_l) / win_l if win_l else None

    if len(close) >= 15:
        prev_close = close[-15:-1]
        tr = np.maximum.reduce([
            high[-14:] - low[-14:],
            np.abs(high[-14:] - prev_close),
            np.abs(low[-14:] - prev_close),
        ])
        atr = float(np.mean(tr))
        out["atr_pct"] = atr / float(close[-1]) if close[-1] else None

    if "rsi_14" in df.columns:
        val = df["rsi_14"].iloc[-1]
        out["rsi_14"] = float(val) if val == val else None  # NaN check

    return out


def build_context(ticker: str, df, signals=None) -> MarketContext:
    """Stage 1 — assemble everything a strategy needs. Costs nothing."""
    ohlc = {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
    }
    # Dedupe here too: the model is billed per token, so it must never be shown
    # the same multi-bar pattern three times.
    pats = dedupe(detect(ohlc, lookback_bars=5))

    ctx = MarketContext(
        ticker=ticker.upper(),
        price=float(df["close"].iloc[-1]),
        patterns=pats,
        pattern_bias=net_bias(pats),
        **_structure(df),
    )

    if signals is not None:
        ctx.ou_score = signals.ou_score
        ctx.ou_halflife = signals.ou_halflife
        ctx.hmm_regime = signals.hmm_regime
        ctx.hmm_confidence = signals.hmm_confidence
        ctx.kalman_alpha = signals.kalman_alpha
        ctx.evt_var_99 = signals.evt_var_99
        ctx.evt_expected_shortfall = signals.evt_expected_shortfall

    return ctx


_JSON_RE = re.compile(r"\{.*\}", re.S)

_SYSTEM = (
    "You are a trade-gate. You output ONLY strict JSON, no prose.\n"
    'Schema: {"decision":"TAKE"|"SKIP","confidence":0.0-1.0,"reason":"<=15 words"}\n'
    "Default to SKIP. TAKE only when the evidence is unambiguous and the "
    "reward clearly exceeds the risk to the invalidation level."
)


class Screener:
    """Runs the funnel over a universe and returns one verdict per ticker."""

    def __init__(
        self,
        governor: Optional[CreditGovernor] = None,
        strategy: Optional[Strategy] = None,
        max_paid_per_scan: int = 3,
    ) -> None:
        self.governor = governor or CreditGovernor()
        self.strategy = strategy or load_strategy()
        # Hard ceiling on paid calls per sweep. This bounds cost even if a
        # plugged-in strategy promotes far too many names — only the top-K by
        # conviction are paid for; the rest still get free-tier verdicts.
        self.max_paid_per_scan = max(0, int(max_paid_per_scan))
        self._client = None
        # Fingerprints of setups already reported, to avoid duplicate alerts.
        self._seen: Dict[str, str] = {}

    # ------------------------------------------------------------------ model
    def _client_or_none(self):
        if self._client is not None:
            return self._client
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or anthropic is None:
            return None
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _ask_model(self, ctx: MarketContext, sig: StrategySignal, tier: Tier):
        """Paid tier. Returns (decision, confidence, reason, tokens)."""
        client = self._client_or_none()
        if client is None:
            return None

        hint = self.strategy.prompt_hint()
        prompt = (
            f"{ctx.summary()}\n"
            f"strategy_bias={sig.bias} conviction={sig.conviction:.2f}\n"
            f"{('strategy_note=' + sig.rationale) if sig.rationale else ''}\n"
            f"{hint}\n"
            "Take this trade?"
        )

        try:
            msg = client.messages.create(
                model=TIER_MODEL[tier],
                max_tokens=120,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the scan
            print(f"[screener] model call failed for {ctx.ticker}: {exc}")
            return None

        usage = getattr(msg, "usage", None)
        tokens = 0
        if usage is not None:
            self.governor.record(usage.input_tokens, usage.output_tokens)
            tokens = usage.input_tokens + usage.output_tokens

        text = msg.content[0].text if msg.content else ""
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        decision = str(data.get("decision", "SKIP")).upper()
        if decision not in ("TAKE", "SKIP"):
            decision = "SKIP"
        conf = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))[:120]
        return decision, max(0.0, min(1.0, conf)), reason, tokens

    # ----------------------------------------------------------------- funnel
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _free_pass(self, ticker: str, df, signals=None):
        """
        Stages 0-2, all free. Returns (ctx, signal, verdict).

        A non-None verdict means the name resolved without paying and needs no
        further work.
        """
        ctx = build_context(ticker, df, signals)

        if not self.strategy.screen(ctx):
            return ctx, None, Verdict(
                ticker=ctx.ticker,
                decision="SKIP",
                confidence=0.0,
                reason="filtered: no structural setup",
                price=ctx.price,
                tier=Tier.NONE.value,
                at=self._now(),
            )

        sig = self.strategy.evaluate(ctx)

        # Strategy abstained — don't pay to second-guess it.
        if not sig.actionable:
            return ctx, sig, Verdict(
                ticker=ctx.ticker,
                decision="SKIP",
                confidence=0.0,
                reason=sig.rationale or "strategy abstained",
                price=ctx.price,
                tier=Tier.NONE.value,
                entry=sig.entry,
                invalidation=sig.invalidation,
                target=sig.target,
                at=self._now(),
            )

        return ctx, sig, None

    def _free_verdict(self, ctx: MarketContext, sig: StrategySignal) -> Verdict:
        """Numeric-only verdict — used when budget is spent or a call fails."""
        return Verdict(
            ticker=ctx.ticker,
            decision="TAKE" if sig.conviction >= 0.6 else "SKIP",
            confidence=sig.conviction,
            reason=(sig.rationale or "numeric verdict")[:110] + " [free-tier]",
            price=ctx.price,
            tier=Tier.NONE.value,
            entry=sig.entry,
            invalidation=sig.invalidation,
            target=sig.target,
            at=self._now(),
        )

    def _paid_verdict(self, ctx: MarketContext, sig: StrategySignal) -> Verdict:
        """Stage 3 — model call at whatever tier the governor permits."""
        desired = Tier.DEEP if sig.conviction >= 0.7 else Tier.CHEAP
        tier = self.governor.plan_tier(desired)

        if tier is not Tier.NONE:
            result = self._ask_model(ctx, sig, tier)
            if result is not None:
                decision, conf, reason, tokens = result
                return Verdict(
                    ticker=ctx.ticker,
                    decision=decision,
                    confidence=conf,
                    reason=reason,
                    price=ctx.price,
                    tier=tier.value,
                    tokens=tokens,
                    entry=sig.entry,
                    invalidation=sig.invalidation,
                    target=sig.target,
                    at=self._now(),
                )
        return self._free_verdict(ctx, sig)

    def evaluate_one(self, ticker: str, df, signals=None) -> Verdict:
        """Single-ticker path (used by the on-demand endpoint)."""
        ctx, sig, verdict = self._free_pass(ticker, df, signals)
        if verdict is not None:
            return verdict
        return self._paid_verdict(ctx, sig)

    def scan(self, data: Dict[str, object], signals_by_ticker: Optional[dict] = None) -> List[Verdict]:
        """
        Run the funnel across a universe.

        Every name gets the free stages. Only the top `max_paid_per_scan`
        actionable candidates, ranked by conviction, are paid for — so a sweep
        has a hard, predictable cost ceiling no matter how many names a
        strategy promotes.
        """
        signals_by_ticker = signals_by_ticker or {}
        out: List[Verdict] = []
        candidates = []  # (ctx, sig) awaiting a paid verdict

        for ticker, df in data.items():
            try:
                if df is None or len(df) < 30:
                    continue
                ctx, sig, verdict = self._free_pass(ticker, df, signals_by_ticker.get(ticker))
                if verdict is not None:
                    out.append(verdict)
                else:
                    candidates.append((ctx, sig))
            except Exception as exc:  # noqa: BLE001 - one bad name can't stop the sweep
                print(f"[screener] {ticker} failed: {exc}")

        # Highest conviction first, then spend down to the cap.
        candidates.sort(key=lambda pair: -pair[1].conviction)
        for i, (ctx, sig) in enumerate(candidates):
            if i < self.max_paid_per_scan:
                out.append(self._paid_verdict(ctx, sig))
            else:
                out.append(self._free_verdict(ctx, sig))

        # TAKEs first, then by confidence.
        out.sort(key=lambda v: (v.decision != "TAKE", -v.confidence))
        return out

    def new_takes(self, verdicts: List[Verdict]) -> List[Verdict]:
        """TAKE verdicts not already reported for the same setup."""
        fresh = []
        for v in verdicts:
            if v.decision != "TAKE":
                continue
            fingerprint = f"{v.decision}:{round(v.confidence, 1)}:{round(v.price, 2)}"
            if self._seen.get(v.ticker) == fingerprint:
                continue
            self._seen[v.ticker] = fingerprint
            fresh.append(v)
        return fresh
