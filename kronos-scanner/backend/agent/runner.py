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
from .schedule import MarketSession, Phase, session_from_env
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
        session: Optional[MarketSession] = None,
        base_interval: float = 300.0,
        respect_session: bool = True,
    ) -> None:
        self.universe = [t.upper() for t in universe]
        self.load_fn = load_fn
        self.signal_fn = signal_fn
        self.governor = governor or CreditGovernor()
        self.screener = screener or Screener(governor=self.governor)
        self.on_take = on_take
        self.min_interval = min_interval
        self.session = session or session_from_env()
        self.base_interval = base_interval
        # When True the loop only scans inside the configured session window.
        self.respect_session = respect_session
        self.sleeping_until_open = False

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
    def _next_interval(self) -> float:
        """
        Seconds to wait before the next sweep.

        Inside the session this is the phase-scaled cadence (fast at the open,
        slower into the taper), floored by whatever the credit governor thinks
        is sustainable for the time remaining.
        """
        cadence = self.session.cadence_seconds(self.base_interval)

        if self.respect_session:
            left = self.session.seconds_until_close()
            if left > 0:
                # Pace the governor over the rest of *this session*.
                planned = max(1, int(left / max(cadence, 1.0)))
                cadence = max(cadence, self.governor.suggest_interval(planned_cycles=planned))

        return max(self.min_interval, cadence)

    def _loop(self, planned_cycles: int) -> None:
        while not self._stop.is_set():
            # Outside the trading window: sleep until the open instead of
            # scanning a closed market.
            if self.respect_session and not self.session.is_active():
                wait = self.session.seconds_until_open()
                self.sleeping_until_open = True
                phase = self.session.phase()
                print(
                    f"[runner] {phase}: sleeping {wait/60:.0f}m until next open "
                    f"({self.session.tz_name})"
                )
                # Wake at least hourly so status stays fresh and stop() is responsive.
                self._stop.wait(timeout=min(wait, 3600.0))
                continue

            self.sleeping_until_open = False
            try:
                self.scan_once()
                self.error = None
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[runner] scan failed: {self.error}")

            # Wait in slices so stop() is responsive.
            self._stop.wait(timeout=self._next_interval())

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
            "sleeping_until_open": self.sleeping_until_open,
            "universe_size": len(self.universe),
            "scans_completed": count,
            "last_scan_at": scanned_at,
            "takes": len(takes),
            "skips": len(verdicts) - len(takes),
            "strategy": self.screener.strategy.name,
            "session": self.session.describe(),
            "budget": self.governor.snapshot(),
            "next_interval_seconds": round(self._next_interval(), 1),
            "error": self.error,
        }
