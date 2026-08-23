# agent/runner.py
"""
Background scan loop.

Paces itself from the credit governor so it can run all morning: the interval
stretches when budget is tight and tightens when there is headroom. Scanning
never stops outright — once budget is spent the funnel keeps producing
free-tier verdicts.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .budget import CreditGovernor
from .screener import Screener, Verdict


class AgentRunner:
    """Owns the scan thread and the latest results."""

    def __init__(
        self,
        universe: List[str],
        load_fn: Callable[[List[str]], Dict[str, object]],
        signal_fn: Optional[Callable[[str, object], object]] = None,
        governor: Optional[CreditGovernor] = None,
        screener: Optional[Screener] = None,
        on_take: Optional[Callable[[List[Verdict]], None]] = None,
        min_interval: float = 60.0,
    ) -> None:
        self.universe = [t.upper() for t in universe]
        self.load_fn = load_fn
        self.signal_fn = signal_fn
        self.governor = governor or CreditGovernor()
        self.screener = screener or Screener(governor=self.governor)
        self.on_take = on_take
        self.min_interval = min_interval

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.last_verdicts: List[Verdict] = []
        self.last_scan_at: Optional[str] = None
        self.scan_count = 0
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ scan
    def scan_once(self) -> List[Verdict]:
        """One full sweep. Safe to call directly (e.g. from an endpoint)."""
        data = self.load_fn(self.universe)

        signals = {}
        if self.signal_fn is not None:
            for ticker, df in data.items():
                try:
                    signals[ticker] = self.signal_fn(ticker, df)
                except Exception as exc:  # noqa: BLE001
                    print(f"[runner] signals failed for {ticker}: {exc}")

        verdicts = self.screener.scan(data, signals)

        with self._lock:
            self.last_verdicts = verdicts
            self.last_scan_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.scan_count += 1

        fresh = self.screener.new_takes(verdicts)
        if fresh and self.on_take:
            try:
                self.on_take(fresh)
            except Exception as exc:  # noqa: BLE001
                print(f"[runner] on_take callback failed: {exc}")

        return verdicts

    # ------------------------------------------------------------------ loop
    def _loop(self, planned_cycles: int) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
                self.error = None
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[runner] scan failed: {self.error}")

            interval = max(
                self.min_interval,
                self.governor.suggest_interval(planned_cycles=planned_cycles),
            )
            # Wait in slices so stop() is responsive.
            self._stop.wait(timeout=interval)

    def start(self, planned_cycles: int = 36) -> bool:
        """Begin scanning in the background. Returns False if already running."""
        if self.is_running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(planned_cycles,), daemon=True, name="kronos-agent"
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running:
            return False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return True

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        with self._lock:
            verdicts = list(self.last_verdicts)
            scanned_at = self.last_scan_at
            count = self.scan_count

        takes = [v for v in verdicts if v.decision == "TAKE"]
        return {
            "running": self.is_running,
            "universe_size": len(self.universe),
            "scans_completed": count,
            "last_scan_at": scanned_at,
            "takes": len(takes),
            "skips": len(verdicts) - len(takes),
            "strategy": self.screener.strategy.name,
            "budget": self.governor.snapshot(),
            "next_interval_seconds": round(
                max(self.min_interval, self.governor.suggest_interval()), 1
            ),
            "error": self.error,
        }
