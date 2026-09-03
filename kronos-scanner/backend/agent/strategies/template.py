# agent/strategies/template.py
"""
Template for your strategy. Copy this file, rename it, fill in the two methods.

Enable it with:

    export KRONOS_STRATEGY=agent.strategies.my_strategy:MyStrategy

Nothing here expresses a trading opinion — the logic is intentionally left to
you. The plumbing (data, candlestick patterns, quant engines, credit
governance, verdicts, push alerts) is already done and calls into these hooks.

Everything on MarketContext is precomputed and FREE:

    ctx.price                    last close
    ctx.patterns                 [Pattern(name, direction, strength, bars_ago)]
    ctx.pattern_bias             -1..+1, recency-weighted, de-duplicated
    ctx.ou_score                 0..100 mean-reversion pull
    ctx.ou_halflife              days
    ctx.hmm_regime               "Bull" | "Bear" | "HighVol" | "LowVol"
    ctx.hmm_confidence           0..1
    ctx.kalman_alpha             normalized drift (+ up, - down)
    ctx.evt_var_99               99% tail loss, e.g. 0.032 == 3.2%
    ctx.evt_expected_shortfall   mean loss beyond VaR
    ctx.pct_from_20d_high        e.g. -0.04 == 4% below the 20-day high
    ctx.pct_from_20d_low
    ctx.atr_pct                  ATR as a fraction of price
    ctx.rsi_14
"""

from __future__ import annotations

from ..strategy import MarketContext, Strategy, StrategySignal


class MyStrategy(Strategy):
    name = "my-strategy"

    def screen(self, ctx: MarketContext) -> bool:
        """
        FREE. Return True only if this name deserves a paid look.

        This is your main cost lever — it runs on every ticker and costs
        nothing. Aim to pass roughly 10-20% of the universe. Anything you
        reject here never consumes a single token.

        (The agent also caps paid calls per sweep via max_paid_per_scan, so a
        loose screen degrades cost gracefully rather than exploding it.)
        """
        raise NotImplementedError("implement your screen")

    def evaluate(self, ctx: MarketContext) -> StrategySignal:
        """
        Produce the directional call for a name that passed `screen()`.

        Return conviction 0.0 to abstain. Conviction >= 0.7 requests the deeper
        model tier; the governor may still downgrade it if credits are tight.

        Populate entry/invalidation/target when you can — they render on the
        phone card and drive position sizing.
        """
        raise NotImplementedError("implement your evaluation")

    def prompt_hint(self) -> str:
        """
        Optional. Extra instruction appended to the model prompt, so you can
        steer the paid tier without touching agent internals.
        """
        return ""
