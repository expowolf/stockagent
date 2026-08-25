# agent/journal.py
"""
Trade journal.

The scanner was generating signals and forgetting them. Without a record there
is no way to answer the only question that matters — does this edge exist? —
and no basis for sizing up beyond a guess.

Every TAKE is recorded with its entry, stop and target. On later runs each open
entry is marked against subsequent bars: stop first if both were touched in the
same bar, because intrabar order is unknowable from daily data and assuming the
favourable sequence is how backtests lie.

Outcomes are expressed in R (multiples of the risk taken), which makes trades
comparable across position sizes and prices.

State lives in data/live/journal.json, committed by the Actions workflow, so it
survives runners and sessions alike.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _journal_path() -> Path:
    override = os.environ.get("KRONOS_JOURNAL")
    if override:
        return Path(override)
    here = Path(__file__).resolve().parent
    return here.parent.parent.parent / "data" / "live" / "journal.json"


@dataclass
class Entry:
    ticker: str
    opened_at: str          # ISO date of the signal
    entry: float
    stop: float
    target: Optional[float]
    shares: Optional[float] = None
    risk_amount: Optional[float] = None
    setup: str = ""
    conviction: float = 0.0
    # Filled in when resolved
    status: str = "open"    # open | target | stop | expired
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    r_multiple: Optional[float] = None
    bars_held: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Journal:
    """Append-only record of signals plus their resolved outcomes."""

    # The spec holds 3-5 days. Give it a little room, then stop counting: an
    # unresolved trade is not a winner in waiting, it is a flat one.
    MAX_BARS = int(os.environ.get("KRONOS_MAX_HOLD_BARS", "7"))

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or _journal_path())
        self.entries: List[Entry] = []
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
            self.entries = [Entry(**e) for e in raw.get("entries", [])]
        except Exception:  # noqa: BLE001 - missing or corrupt starts empty
            self.entries = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stats": self.stats(),
                "entries": [e.to_dict() for e in self.entries],
            }
            self.path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"[journal] write failed: {exc}")

    # --------------------------------------------------------------- record
    def has_open(self, ticker: str) -> bool:
        return any(e.ticker == ticker.upper() and e.status == "open" for e in self.entries)

    def record(self, verdict) -> bool:
        """
        Log a TAKE. Returns False if this ticker already has an open entry —
        one position per name, so a repeat signal does not double-count the
        same idea in the statistics.
        """
        if verdict.decision != "TAKE" or not verdict.invalidation:
            return False
        ticker = verdict.ticker.upper()
        if self.has_open(ticker):
            return False

        self.entries.append(Entry(
            ticker=ticker,
            opened_at=(verdict.at or datetime.now(timezone.utc).isoformat())[:10],
            entry=float(verdict.entry or verdict.price),
            stop=float(verdict.invalidation),
            target=float(verdict.target) if verdict.target else None,
            shares=verdict.shares,
            risk_amount=verdict.risk_amount,
            setup=(verdict.reason or "")[:60],
            conviction=float(verdict.confidence or 0.0),
        ))
        return True

    # --------------------------------------------------------------- resolve
    def update(self, frames: Dict[str, object]) -> List[Entry]:
        """
        Mark open entries against price action since they were opened.

        `frames` maps ticker -> OHLCV DataFrame. Returns entries closed by
        this call.
        """
        closed: List[Entry] = []

        for e in self.entries:
            if e.status != "open":
                continue
            df = frames.get(e.ticker)
            if df is None or len(df) == 0:
                continue

            try:
                after = df[df["date"].astype(str) > e.opened_at]
            except Exception:  # noqa: BLE001
                continue
            if len(after) == 0:
                continue

            e.bars_held = len(after)
            risk = abs(e.entry - e.stop) or 1e-9

            for _, bar in after.iterrows():
                low, high = float(bar["low"]), float(bar["high"])

                # Stop is checked FIRST. When a bar spans both levels, daily
                # data cannot say which came first, and assuming the target is
                # exactly the optimism that makes a backtest look better than
                # the account ever will.
                if low <= e.stop:
                    e.status, e.exit_price = "stop", e.stop
                    e.r_multiple = -1.0
                    break
                if e.target and high >= e.target:
                    e.status, e.exit_price = "target", e.target
                    e.r_multiple = round((e.target - e.entry) / risk, 2)
                    break
            else:
                # Neither level touched. Past the hold window, close it flat at
                # the last close rather than leaving it open indefinitely.
                if e.bars_held >= self.MAX_BARS:
                    last = float(after["close"].iloc[-1])
                    e.status, e.exit_price = "expired", last
                    e.r_multiple = round((last - e.entry) / risk, 2)

            if e.status != "open":
                e.closed_at = str(after["date"].iloc[min(e.bars_held, len(after)) - 1])[:10]
                closed.append(e)

        return closed

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        done = [e for e in self.entries if e.status != "open" and e.r_multiple is not None]
        openn = [e for e in self.entries if e.status == "open"]

        if not done:
            return {
                "total": len(self.entries), "open": len(openn), "closed": 0,
                "note": "no closed trades yet — edge unvalidated",
            }

        wins = [e for e in done if e.r_multiple > 0]
        rs = [e.r_multiple for e in done]
        expectancy = sum(rs) / len(rs)

        return {
            "total": len(self.entries),
            "open": len(openn),
            "closed": len(done),
            "wins": len(wins),
            "losses": len(done) - len(wins),
            "win_rate": round(len(wins) / len(done), 3),
            "expectancy_r": round(expectancy, 3),
            "total_r": round(sum(rs), 2),
            "best_r": round(max(rs), 2),
            "worst_r": round(min(rs), 2),
            # Kelly needs a real win rate; below ~20 samples it is noise, and
            # sizing off noise is how small accounts die.
            "sample_adequate": len(done) >= 20,
        }

    def summary_line(self) -> str:
        s = self.stats()
        if not s.get("closed"):
            return f"journal: {s['total']} signals, {s['open']} open, none closed yet"
        flag = "" if s["sample_adequate"] else " (sample too small to size on)"
        return (
            f"journal: {s['closed']} closed, {s['win_rate']:.0%} win, "
            f"{s['expectancy_r']:+.2f}R avg, {s['total_r']:+.1f}R total{flag}"
        )
