# agent/news_watch.py
"""
News watcher.

Reads Yahoo Finance headlines and asks what they imply for price, given the
live market data we already have for that name.

Cost discipline mirrors the price funnel, because news is the easiest place to
burn credits without noticing:

  free   fetch headlines (yfinance)
  free   recency filter, dedupe against already-seen article ids
  free   keyword materiality score
  paid   ONE call per ticker covering all its fresh headlines at once
  free   result cached by article id, so a repeat sweep costs nothing

Four separate guards keep this bounded: only candidates get news at all, only
material and unseen headlines qualify, headlines are batched per ticker rather
than per article, and a hard cap limits how many tickers may be analysed per
sweep. On top of that the CreditGovernor can still refuse.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from data.news import NewsItem, SeenArticles, fetch_news

from .budget import CreditGovernor, Tier, TIER_MODEL

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


_JSON_RE = re.compile(r"\{.*\}", re.S)

_SYSTEM = (
    "You judge how news affects a stock's near-term price. Output ONLY strict "
    "JSON, no prose.\n"
    'Schema: {"impact":"bullish"|"bearish"|"neutral","magnitude":0.0-1.0,'
    '"tradeable":true|false,"note":"<=15 words"}\n'
    "magnitude is expected near-term price impact, not how dramatic the "
    "headline sounds. Routine coverage, recaps and analyst chatter are "
    "neutral with low magnitude. Set tradeable only when the news is likely to "
    "move price beyond normal daily range."
)


@dataclass
class NewsRead:
    """What the watcher concluded about one ticker's news."""

    ticker: str
    impact: str = "neutral"      # bullish | bearish | neutral
    magnitude: float = 0.0       # 0..1 expected price impact
    tradeable: bool = False
    note: str = ""
    headlines: List[str] = field(default_factory=list)
    article_ids: List[str] = field(default_factory=list)
    tier: str = "none"
    tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.headlines:
            return ""
        return f"news {self.impact} {self.magnitude:.1f}" + (f" — {self.note}" if self.note else "")


class NewsWatcher:
    """Fetches, triages and (sparingly) analyses Yahoo Finance news."""

    def __init__(
        self,
        governor: Optional[CreditGovernor] = None,
        max_analyzed_per_scan: int = 2,
        min_materiality: float = 0.45,
        max_age_hours: float = 24.0,
        cache_path: Optional[str] = None,
    ) -> None:
        self.governor = governor or CreditGovernor()
        # Hard ceiling on paid news calls per sweep, independent of how many
        # tickers have fresh headlines.
        self.max_analyzed_per_scan = max(0, int(max_analyzed_per_scan))
        self.min_materiality = float(min_materiality)
        self.max_age_hours = float(max_age_hours)
        self.seen = SeenArticles()
        self._client = None
        self._cache_path = cache_path or os.environ.get(
            "KRONOS_NEWS_CACHE", "/tmp/kronos_news_cache.json"
        )
        self._cache: Dict[str, dict] = self._load_cache()
        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> dict:
        try:
            with open(self._cache_path) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_path, "w") as fh:
                json.dump(self._cache, fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[news] cache write failed: {exc}")

    @staticmethod
    def _cache_key(items: List[NewsItem]) -> str:
        return "|".join(sorted(i.id for i in items))

    # ------------------------------------------------------------------ model
    def _client_or_none(self):
        if self._client is not None:
            return self._client
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or anthropic is None:
            return None
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _ask(self, ticker: str, items: List[NewsItem], price_context: str, tier: Tier):
        client = self._client_or_none()
        if client is None:
            return None

        # One call for ALL of this ticker's headlines — batching here is worth
        # more than any prompt tuning.
        lines = "\n".join(f"- {i.headline()}" for i in items[:6])
        prompt = f"{ticker}\n{price_context}\n\nHeadlines:\n{lines}\n\nImpact on price?"

        try:
            msg = client.messages.create(
                model=TIER_MODEL[tier],
                max_tokens=150,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[news] model call failed for {ticker}: {exc}")
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

        impact = str(data.get("impact", "neutral")).lower()
        if impact not in ("bullish", "bearish", "neutral"):
            impact = "neutral"
        return {
            "impact": impact,
            "magnitude": max(0.0, min(1.0, float(data.get("magnitude", 0.0)))),
            "tradeable": bool(data.get("tradeable", False)),
            "note": str(data.get("note", ""))[:120],
            "tokens": tokens,
        }

    # ------------------------------------------------------------------ public
    def triage(self, ticker: str) -> List[NewsItem]:
        """Free stages: fetch, age-filter, dedupe, materiality threshold."""
        items = fetch_news(ticker, max_age_hours=self.max_age_hours)
        fresh = self.seen.filter_new(items)
        return [i for i in fresh if i.materiality >= self.min_materiality]

    def watch(self, tickers: List[str], price_context: Dict[str, str] = None) -> Dict[str, NewsRead]:
        """
        Run the news funnel over candidate tickers.

        `price_context` maps ticker -> a one-line price summary, so the model
        judges the headline against what the tape is actually doing.
        """
        price_context = price_context or {}
        self.stats = {"checked": 0, "with_news": 0, "analyzed": 0, "cached": 0, "throttled": 0}
        out: Dict[str, NewsRead] = {}

        # Free triage across every candidate.
        pending: List[tuple] = []
        for ticker in tickers:
            self.stats["checked"] += 1
            items = self.triage(ticker)
            if not items:
                continue
            self.stats["with_news"] += 1

            read = NewsRead(
                ticker=ticker.upper(),
                headlines=[i.title for i in items[:3]],
                article_ids=[i.id for i in items],
            )

            cached = self._cache.get(self._cache_key(items))
            if cached:
                self.stats["cached"] += 1
                read.impact = cached.get("impact", "neutral")
                read.magnitude = cached.get("magnitude", 0.0)
                read.tradeable = cached.get("tradeable", False)
                read.note = cached.get("note", "")
                read.tier = "cached"
                out[read.ticker] = read
                continue

            pending.append((ticker, items, read))

        # Most material first, then spend down to the cap.
        pending.sort(key=lambda p: -max(i.materiality for i in p[1]))

        for idx, (ticker, items, read) in enumerate(pending):
            if idx >= self.max_analyzed_per_scan:
                # Beyond the cap: keep the free keyword read, don't pay.
                read.note = "not analysed (per-sweep cap)"
                read.magnitude = round(max(i.materiality for i in items), 2)
                out[read.ticker] = read
                continue

            tier = self.governor.plan_tier(Tier.CHEAP)
            if tier is Tier.NONE:
                self.stats["throttled"] += 1
                read.note = "not analysed (credit throttle)"
                read.magnitude = round(max(i.materiality for i in items), 2)
                out[read.ticker] = read
                continue

            result = self._ask(ticker, items, price_context.get(ticker.upper(), ""), tier)
            if result is None:
                read.note = "analysis unavailable"
                read.magnitude = round(max(i.materiality for i in items), 2)
                out[read.ticker] = read
                continue

            self.stats["analyzed"] += 1
            read.impact = result["impact"]
            read.magnitude = result["magnitude"]
            read.tradeable = result["tradeable"]
            read.note = result["note"]
            read.tier = tier.value
            read.tokens = result["tokens"]

            self._cache[self._cache_key(items)] = {
                k: result[k] for k in ("impact", "magnitude", "tradeable", "note")
            }
            self.seen.mark(items)
            out[read.ticker] = read

        self._save_cache()
        return out
