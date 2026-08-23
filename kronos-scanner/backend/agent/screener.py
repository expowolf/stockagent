# agent/screener.py
"""
The cost funnel.

Three escalating gates, each far more expensive than the last, so cost tracks
the number of *interesting* names rather than universe size:

  A  patterns + strategy.prefilter()   ~0.3ms/ticker   every name
  B  quant engines + strategy.screen() ~95ms/ticker    stage-A survivors only
  C  model call                        tokens          top-K by conviction only

Stage B is ~300x the cost of stage A, which is the whole reason A exists: on a
570-name universe the prefilter drops well over half before any engine runs,
roughly halving sweep time.

Token cost is independent of universe size — only stage C spends, and it is
capped at `max_paid_per_scan` regardless of how many names were scanned.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .budget import CreditGovernor, Tier, TIER_MODEL
from .patterns import dedupe, detect, net_bias
from .risk import RiskAssessment, RiskPolicy
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
    invalidation: Optional[float] = None   # the stop loss
    target: Optional[float] = None
    # Risk-gate output
    rr: Optional[float] = None
    stop_pct: Optional[float] = None
    shares: Optional[float] = None
    notional: Optional[float] = None
    risk_amount: Optional[float] = None
    warnings: list = field(default_factory=list)
    at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def alert_text(self) -> str:
        """Notification body. The stop loss is the point of this message."""
        lines = [f"{self.decision} {self.ticker} @ ${self.price:.2f}"]
        if self.invalidation:
            lines.append(f"STOP ${self.invalidation:.2f}")
            if self.stop_pct:
                lines[-1] += f"  (-{self.stop_pct*100:.1f}%)"
        if self.target:
            lines.append(f"target ${self.target:.2f}" + (f"  R:R {self.rr:.1f}" if self.rr else ""))
        if self.shares:
            lines.append(
                f"size {self.shares:g} sh (${self.notional:,.0f})  risk ${self.risk_amount:,.2f}"
            )
        if self.reason:
            lines.append(self.reason)
        for w in self.warnings or []:
            lines.append(f"! {w}")
        return "\n".join(lines)


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


def _attach_signals(ctx: MarketContext, signals) -> None:
    """Copy quant-engine outputs onto a context built from patterns alone."""
    ctx.ou_score = signals.ou_score
    ctx.ou_halflife = signals.ou_halflife
    ctx.hmm_regime = signals.hmm_regime
    ctx.hmm_confidence = signals.hmm_confidence
    ctx.kalman_alpha = signals.kalman_alpha
    ctx.evt_var_99 = signals.evt_var_99
    ctx.evt_expected_shortfall = signals.evt_expected_shortfall


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
        risk: Optional[RiskPolicy] = None,
    ) -> None:
        self.governor = governor or CreditGovernor()
        self.strategy = strategy or load_strategy()
        # Risk discipline is applied to every TAKE, independent of strategy.
        self.risk = risk or RiskPolicy.from_env()
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

    def _apply_risk(self, v: Verdict, ctx: MarketContext, sig: StrategySignal) -> Verdict:
        """
        Gate every TAKE through risk discipline.

        A TAKE that cannot be structured survivably is downgraded to SKIP. In
        particular a signal with no stop loss can never become a TAKE, no
        matter how confident the strategy or the model is about direction.
        """
        if v.decision != "TAKE":
            return v

        ra: RiskAssessment = self.risk.assess(sig, ctx)
        v.warnings = list(ra.warnings or [])

        if not ra.approved:
            v.decision = "SKIP"
            v.reason = f"risk: {ra.reason}"
            v.confidence = 0.0
            return v

        v.rr = ra.rr
        v.stop_pct = ra.stop_pct
        v.shares = ra.shares
        v.notional = ra.notional
        v.risk_amount = ra.risk_amount
        return v

    def _free_verdict(self, ctx: MarketContext, sig: StrategySignal) -> Verdict:
        """Numeric-only verdict — used when budget is spent or a call fails."""
        return self._apply_risk(Verdict(
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
        ), ctx, sig)

    def _paid_verdict(self, ctx: MarketContext, sig: StrategySignal) -> Verdict:
        """Stage 3 — model call at whatever tier the governor permits."""
        desired = Tier.DEEP if sig.conviction >= 0.7 else Tier.CHEAP
        tier = self.governor.plan_tier(desired)

        if tier is not Tier.NONE:
            result = self._ask_model(ctx, sig, tier)
            if result is not None:
                decision, conf, reason, tokens = result
                return self._apply_risk(Verdict(
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
                ), ctx, sig)
        return self._free_verdict(ctx, sig)

    def evaluate_one(self, ticker: str, df, signals=None) -> Verdict:
        """Single-ticker path (used by the on-demand endpoint)."""
        ctx, sig, verdict = self._free_pass(ticker, df, signals)
        if verdict is not None:
            return verdict
        return self._paid_verdict(ctx, sig)

    def scan(
        self,
        data: Dict[str, object],
        signals_by_ticker: Optional[dict] = None,
        signal_fn=None,
    ) -> List[Verdict]:
        """
        Run the funnel across a universe.

        Three escalating gates, each more expensive than the last, so cost
        tracks the number of *interesting* names rather than universe size:

          A. patterns + prefilter   ~0.3ms/ticker   every name
          B. quant engines + screen ~50ms/ticker    prefilter survivors only
          C. model call             tokens          top-K by conviction only

        Pass `signal_fn(ticker, df)` to have engines computed lazily in stage B
        — that is what makes a several-hundred-name universe practical.
        A precomputed `signals_by_ticker` dict still works for small sweeps.
        """
        signals_by_ticker = signals_by_ticker or {}
        out: List[Verdict] = []
        survivors = []   # (ticker, df, ctx) that cleared the cheap gate
        candidates = []  # (ctx, sig) awaiting a paid verdict

        self.stats = {"scanned": 0, "prefiltered": 0, "engines_run": 0, "screened": 0}

        # --- Stage A: patterns only. Engines have NOT run. -------------------
        for ticker, df in data.items():
            try:
                if df is None or len(df) < 30:
                    continue
                self.stats["scanned"] += 1
                ctx = build_context(ticker, df, signals_by_ticker.get(ticker))

                if not self.strategy.prefilter(ctx):
                    self.stats["prefiltered"] += 1
                    out.append(Verdict(
                        ticker=ctx.ticker, decision="SKIP", confidence=0.0,
                        reason="no price-action signal", price=ctx.price,
                        tier=Tier.NONE.value, at=self._now(),
                    ))
                    continue
                survivors.append((ticker, df, ctx))
            except Exception as exc:  # noqa: BLE001 - one bad name can't stop the sweep
                print(f"[screener] {ticker} prefilter failed: {exc}")

        # --- Stage B: engines, then the full screen -------------------------
        for ticker, df, ctx in survivors:
            try:
                if signal_fn is not None and ticker not in signals_by_ticker:
                    sigs = signal_fn(ticker, df)
                    self.stats["engines_run"] += 1
                    if sigs is not None:
                        _attach_signals(ctx, sigs)

                if not self.strategy.screen(ctx):
                    out.append(Verdict(
                        ticker=ctx.ticker, decision="SKIP", confidence=0.0,
                        reason="filtered: no structural setup", price=ctx.price,
                        tier=Tier.NONE.value, at=self._now(),
                    ))
                    continue

                self.stats["screened"] += 1
                sig = self.strategy.evaluate(ctx)

                if not sig.actionable:
                    out.append(self._apply_risk(Verdict(
                        ticker=ctx.ticker, decision="SKIP", confidence=0.0,
                        reason=sig.rationale or "strategy abstained", price=ctx.price,
                        tier=Tier.NONE.value, entry=sig.entry,
                        invalidation=sig.invalidation, target=sig.target,
                        at=self._now(),
                    ), ctx, sig))
                    continue

                candidates.append((ctx, sig))
            except Exception as exc:  # noqa: BLE001
                print(f"[screener] {ticker} failed: {exc}")

        # --- Stage C: paid, top-K by conviction only ------------------------
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
