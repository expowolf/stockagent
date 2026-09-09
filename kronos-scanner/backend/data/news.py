# data/news.py
"""
Yahoo Finance news feed.

Free and keyless via yfinance — the same source as the price data, so
headlines and candles stay consistent.

Fetching costs nothing. Everything here is deliberately free: normalization,
recency filtering, dedupe and a keyword materiality score all run locally so
that only genuinely new, genuinely material headlines ever reach a paid model
call downstream.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# Words that historically move a single name, weighted by how hard they hit.
# Purely a triage heuristic — it decides what is worth paying to analyse, not
# what the news means.
MATERIAL_KEYWORDS = {
    # hard catalysts
    "earnings": 1.0, "guidance": 1.0, "outlook": 0.8, "beats": 0.9, "misses": 0.9,
    "downgrade": 0.9, "upgrade": 0.9, "price target": 0.7,
    "acquisition": 1.0, "merger": 1.0, "buyout": 1.0, "takeover": 1.0,
    "bankruptcy": 1.0, "chapter 11": 1.0, "default": 0.9,
    "fda": 1.0, "approval": 0.8, "recall": 0.9, "halted": 1.0, "halt": 0.8,
    "investigation": 0.8, "lawsuit": 0.7, "sec filing": 0.6, "probe": 0.8,
    "ceo": 0.7, "resign": 0.8, "steps down": 0.8, "layoffs": 0.7,
    "split": 0.6, "dividend": 0.6, "buyback": 0.7, "offering": 0.8,
    "guidance cut": 1.0, "warns": 0.9, "surges": 0.7, "plunges": 0.8,
    "soars": 0.7, "tumbles": 0.8, "short seller": 0.9, "contract": 0.6,
    "partnership": 0.5, "breakthrough": 0.7, "delisting": 1.0,
}

_SEEN_PATH = Path(os.environ.get("KRONOS_NEWS_SEEN", "/tmp/kronos_news_seen.json"))
_MAX_SEEN = 4000


@dataclass
class NewsItem:
    id: str
    ticker: str
    title: str
    publisher: str
    published: Optional[str]      # ISO8601 UTC
    link: str = ""
    summary: str = ""
    materiality: float = 0.0
    age_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> str:
        """Compact form sent to the model — keep this cheap."""
        age = f"{self.age_hours:.0f}h" if self.age_hours is not None else "?"
        return f"[{age} · {self.publisher}] {self.title}"


def _parse_time(value) -> Optional[datetime]:
    """yfinance hands back epoch seconds or ISO strings depending on version."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _normalize(raw: dict, ticker: str) -> Optional[NewsItem]:
    """
    Accept both yfinance news shapes.

    Older releases return a flat dict (uuid/title/publisher/providerPublishTime).
    Newer ones nest everything under "content" with a different key for each
    field. Version drift here is the most likely thing to silently break the
    watcher, so handle both rather than assuming.
    """
    if not isinstance(raw, dict):
        return None

    content = raw.get("content") if isinstance(raw.get("content"), dict) else None

    if content:
        title = content.get("title") or ""
        summary = content.get("summary") or content.get("description") or ""
        provider = content.get("provider") or {}
        publisher = provider.get("displayName") if isinstance(provider, dict) else str(provider or "")
        published = content.get("pubDate") or content.get("displayTime")
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = canonical.get("url", "") if isinstance(canonical, dict) else str(canonical or "")
        art_id = raw.get("id") or content.get("id") or ""
    else:
        title = raw.get("title") or ""
        summary = raw.get("summary") or ""
        publisher = raw.get("publisher") or ""
        published = raw.get("providerPublishTime") or raw.get("pubDate")
        link = raw.get("link") or ""
        art_id = raw.get("uuid") or raw.get("id") or ""

    title = (title or "").strip()
    if not title:
        return None

    if not art_id:
        # Stable synthetic id so dedupe still works without one.
        art_id = hashlib.sha1(f"{ticker}:{title}".encode()).hexdigest()[:16]

    dt = _parse_time(published)
    age = None
    if dt is not None:
        age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0

    item = NewsItem(
        id=str(art_id),
        ticker=ticker.upper(),
        title=title,
        publisher=(publisher or "unknown").strip(),
        published=dt.isoformat() if dt else None,
        link=link or "",
        summary=(summary or "").strip()[:400],
        age_hours=age,
    )
    item.materiality = score_materiality(item)
    return item


def score_materiality(item: NewsItem) -> float:
    """
    Free triage score in [0, 1].

    Combines keyword hits with recency: a downgrade from 30 minutes ago matters
    far more than the same downgrade from two days ago.
    """
    text = f"{item.title} {item.summary}".lower()

    hits = [w for kw, w in MATERIAL_KEYWORDS.items() if kw in text]
    if not hits:
        base = 0.15 if re.search(r"\b(stock|shares|rally|selloff)\b", text) else 0.05
    else:
        # Strongest hit dominates; extras add a little.
        base = min(1.0, max(hits) + 0.1 * (len(hits) - 1))

    if item.age_hours is None:
        recency = 0.5
    elif item.age_hours <= 1:
        recency = 1.0
    elif item.age_hours <= 6:
        recency = 0.85
    elif item.age_hours <= 24:
        recency = 0.6
    elif item.age_hours <= 72:
        recency = 0.3
    else:
        recency = 0.1

    return round(min(1.0, base * recency), 3)


def fetch_news(ticker: str, max_age_hours: float = 48.0, limit: int = 10) -> List[NewsItem]:
    """Pull Yahoo Finance headlines for one ticker. Free; never raises."""
    if yf is None:
        return []

    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001 - news must never break a scan
        print(f"[news] {ticker}: {exc}")
        return []

    items: List[NewsItem] = []
    for raw in raw_items:
        item = _normalize(raw, ticker)
        if item is None:
            continue
        if item.age_hours is not None and item.age_hours > max_age_hours:
            continue
        items.append(item)

    items.sort(key=lambda i: (-i.materiality, i.age_hours if i.age_hours is not None else 999))
    return items[:limit]


def fetch_many(tickers: Iterable[str], max_age_hours: float = 48.0, limit: int = 10) -> dict:
    """Headlines for several tickers. Returns {ticker: [NewsItem]}."""
    out = {}
    for t in tickers:
        got = fetch_news(t, max_age_hours=max_age_hours, limit=limit)
        if got:
            out[t.upper()] = got
    return out


# --------------------------------------------------------------------------- #
# Seen-article tracking (so the same headline is never paid for twice)
# --------------------------------------------------------------------------- #
class SeenArticles:
    """Persistent set of article ids already analysed."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or _SEEN_PATH)
        self.ids: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            self.ids = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            self.ids = {}

    def _save(self) -> None:
        try:
            # Keep the newest entries only, so the file cannot grow forever.
            if len(self.ids) > _MAX_SEEN:
                newest = sorted(self.ids.items(), key=lambda kv: kv[1], reverse=True)
                self.ids = dict(newest[:_MAX_SEEN])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.ids))
        except Exception as exc:  # noqa: BLE001
            print(f"[news] could not persist seen set: {exc}")

    def is_new(self, item: NewsItem) -> bool:
        return item.id not in self.ids

    def mark(self, items: Iterable[NewsItem]) -> None:
        now = time.time()
        for i in items:
            self.ids[i.id] = now
        self._save()

    def filter_new(self, items: Iterable[NewsItem]) -> List[NewsItem]:
        return [i for i in items if self.is_new(i)]
