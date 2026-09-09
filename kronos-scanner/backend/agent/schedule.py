# agent/schedule.py
"""
Trading-session schedule.

Keeps the agent awake only during the hours you actually trade, in YOUR local
timezone, and tapers scan cadence as the session quiets down.

Defaults target a Mountain-Time morning session: 07:30-11:00 MT, which is the
09:30 ET open through the pre-lunch slowdown.

Cadence is phase-aware, which doubles as credit management: the open gets the
fastest sweeps because that is when setups actually appear, and the late
session slows down rather than burning budget on a quiet tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional, Set

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore


# --------------------------------------------------------------------------- #
# US market holidays (rule-derived, so this never goes stale)
# --------------------------------------------------------------------------- #
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    # last occurrence
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    """NYSE observes Saturday holidays on Friday, Sunday holidays on Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def market_holidays(year: int) -> Set[date]:
    """NYSE full-day closures for a given year."""
    return {
        _observed(date(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                 # MLK Day
        _nth_weekday(year, 2, 0, 3),                 # Presidents Day
        _easter(year) - timedelta(days=2),           # Good Friday
        _nth_weekday(year, 5, 0, -1),                # Memorial Day
        _observed(date(year, 6, 19)),                # Juneteenth
        _observed(date(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                 # Labor Day
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving
        _observed(date(year, 12, 25)),               # Christmas
    }


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class Phase:
    CLOSED = "closed"
    PREMARKET = "premarket"
    OPEN_BURST = "open-burst"   # first hour: fastest cadence
    CORE = "core"               # steady mid-session
    TAPER = "taper"             # winding down toward your stop time
    DONE = "done"               # session finished for today


# Scan-interval multipliers per phase. Lower = more frequent.
PHASE_CADENCE = {
    Phase.OPEN_BURST: 0.5,
    Phase.CORE: 1.0,
    Phase.TAPER: 1.8,
}


@dataclass
class MarketSession:
    """
    Your trading window, expressed in your own timezone.

    Defaults: 07:30-11:00 America/Denver (= 09:30-13:00 ET).
    """

    tz_name: str = "America/Denver"
    start: time = time(7, 30)
    end: time = time(11, 0)
    burst_minutes: int = 60      # length of the fast open phase
    taper_before_end: int = 60   # minutes before `end` that cadence eases off
    skip_holidays: bool = True
    extra_closed: Set[date] = field(default_factory=set)

    @property
    def tz(self):
        if ZoneInfo is None:
            return None
        try:
            return ZoneInfo(self.tz_name)
        except Exception:  # noqa: BLE001 - bad tz name shouldn't crash the agent
            return ZoneInfo("UTC")

    def now(self) -> datetime:
        return datetime.now(tz=self.tz)

    # ------------------------------------------------------------- day checks
    def is_trading_day(self, d: Optional[date] = None) -> bool:
        d = d or self.now().date()
        if d.weekday() >= 5:
            return False
        if d in self.extra_closed:
            return False
        if self.skip_holidays and d in market_holidays(d.year):
            return False
        return True

    def _at(self, d: date, t: time) -> datetime:
        return datetime.combine(d, t, tzinfo=self.tz)

    # ------------------------------------------------------------- phase/state
    def phase(self, now: Optional[datetime] = None) -> str:
        now = now or self.now()
        d = now.date()

        if not self.is_trading_day(d):
            return Phase.CLOSED

        open_at = self._at(d, self.start)
        close_at = self._at(d, self.end)

        if now < open_at:
            # Treat the hour before the open as premarket, otherwise closed.
            return Phase.PREMARKET if (open_at - now) <= timedelta(hours=1) else Phase.CLOSED
        if now >= close_at:
            return Phase.DONE

        if now < open_at + timedelta(minutes=self.burst_minutes):
            return Phase.OPEN_BURST
        if now >= close_at - timedelta(minutes=self.taper_before_end):
            return Phase.TAPER
        return Phase.CORE

    def is_active(self, now: Optional[datetime] = None) -> bool:
        """True only while we should actually be scanning."""
        return self.phase(now) in (Phase.OPEN_BURST, Phase.CORE, Phase.TAPER)

    def progress(self, now: Optional[datetime] = None) -> float:
        """How far through today's session, 0..1. Used to pace credit spend."""
        now = now or self.now()
        d = now.date()
        if not self.is_trading_day(d):
            return 1.0
        open_at = self._at(d, self.start)
        close_at = self._at(d, self.end)
        total = (close_at - open_at).total_seconds()
        if total <= 0:
            return 1.0
        elapsed = (now - open_at).total_seconds()
        return min(1.0, max(0.0, elapsed / total))

    def session_seconds(self) -> float:
        d = self.now().date()
        return (self._at(d, self.end) - self._at(d, self.start)).total_seconds()

    def seconds_until_open(self, now: Optional[datetime] = None) -> float:
        """Seconds until the next session opens (0 if already active)."""
        now = now or self.now()
        if self.is_active(now):
            return 0.0

        d = now.date()
        # Later today?
        if self.is_trading_day(d):
            open_at = self._at(d, self.start)
            if now < open_at:
                return (open_at - now).total_seconds()

        # Next trading day, up to a week and a half out.
        for i in range(1, 11):
            nd = d + timedelta(days=i)
            if self.is_trading_day(nd):
                return (self._at(nd, self.start) - now).total_seconds()
        return 24 * 3600.0

    def seconds_until_close(self, now: Optional[datetime] = None) -> float:
        now = now or self.now()
        if not self.is_active(now):
            return 0.0
        return max(0.0, (self._at(now.date(), self.end) - now).total_seconds())

    # ------------------------------------------------------------- cadence
    def cadence_seconds(self, base_interval: float, now: Optional[datetime] = None) -> float:
        """Scale a base scan interval by the current phase."""
        mult = PHASE_CADENCE.get(self.phase(now), 1.0)
        return max(30.0, base_interval * mult)

    # ------------------------------------------------------------- views
    def describe(self, now: Optional[datetime] = None) -> dict:
        now = now or self.now()
        ph = self.phase(now)
        return {
            "timezone": self.tz_name,
            "local_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "window": f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}",
            "phase": ph,
            "active": self.is_active(now),
            "trading_day": self.is_trading_day(now.date()),
            "session_progress_pct": round(100 * self.progress(now), 1),
            "minutes_until_open": round(self.seconds_until_open(now) / 60.0, 1),
            "minutes_until_close": round(self.seconds_until_close(now) / 60.0, 1),
            "cadence_multiplier": PHASE_CADENCE.get(ph, 1.0),
        }


def session_from_env() -> MarketSession:
    """Build a session from env vars, defaulting to the MT morning window."""
    import os

    def _time(var: str, default: time) -> time:
        raw = os.environ.get(var, "").strip()
        if not raw:
            return default
        try:
            hh, _, mm = raw.partition(":")
            return time(int(hh), int(mm or 0))
        except Exception:  # noqa: BLE001
            return default

    return MarketSession(
        tz_name=os.environ.get("KRONOS_TZ", "America/Denver"),
        start=_time("KRONOS_SESSION_START", time(7, 30)),
        end=_time("KRONOS_SESSION_END", time(11, 0)),
        burst_minutes=int(os.environ.get("KRONOS_BURST_MINUTES", "60")),
        taper_before_end=int(os.environ.get("KRONOS_TAPER_MINUTES", "60")),
    )
