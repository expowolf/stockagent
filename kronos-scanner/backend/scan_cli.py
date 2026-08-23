#!/usr/bin/env python3
# scan_cli.py
"""
Kronos scan, command line. No server, no deployment — this is what the
in-chat skill runs.

Output is deliberately terse: the whole point is that Python does the analysis
for free and hands back the fewest tokens the answer needs.

    python scan_cli.py              # TAKEs + one summary line
    python scan_cli.py --all        # every verdict
    python scan_cli.py --json       # machine readable
    python scan_cli.py --tickers NVDA,AAPL
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from agent import CreditGovernor, Screener, session_from_env
from agent.screener import Verdict


def _load(tickers: List[str], period: str) -> Dict[str, object]:
    """Fetch OHLCV. Returns {} when no data path is available."""
    from data.ingestion import load_universe

    return load_universe(tickers, period=period)


def _signals(data: Dict[str, object]) -> dict:
    from main import compute_signals

    out = {}
    for ticker, df in data.items():
        try:
            out[ticker] = compute_signals(ticker, df)
        except Exception:  # noqa: BLE001 - a bad name shouldn't stop the sweep
            pass
    return out


def _fmt(v: Verdict, wide: bool) -> str:
    line = f"{v.decision:<4} {v.ticker:<6} {v.price:>8.2f}  {v.confidence:>4.0%}"
    if v.entry and v.invalidation:
        line += f"  e{v.entry:.2f}/s{v.invalidation:.2f}"
        if v.target:
            line += f"/t{v.target:.2f}"
    if wide or v.decision == "TAKE":
        line += f"  {v.reason[:58]}"
    return line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kronos scan")
    ap.add_argument("--tickers", default="", help="comma separated; default = configured universe")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--all", action="store_true", help="show SKIPs too")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--budget", type=int, default=None, help="token budget for the window")
    ap.add_argument("--max-paid", type=int, default=3, help="paid calls per sweep")
    ap.add_argument("--ignore-session", action="store_true", help="scan even outside the window")
    args = ap.parse_args(argv)

    session = session_from_env()
    desc = session.describe()

    if args.tickers.strip():
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        import os

        from data.ingestion import DEFAULT_UNIVERSE

        raw = os.environ.get("KRONOS_UNIVERSE", ",".join(DEFAULT_UNIVERSE[:20]))
        universe = [t.strip().upper() for t in raw.split(",") if t.strip()]

    governor = CreditGovernor(
        token_budget=args.budget or 400_000,
        progress_fn=session.progress,
    )
    screener = Screener(governor=governor, max_paid_per_scan=args.max_paid)

    # Data. If egress is blocked this comes back empty — say so, never invent.
    try:
        data = _load(universe, args.period)
    except Exception as exc:  # noqa: BLE001
        data = {}
        err = str(exc)
    else:
        err = ""

    if not data:
        payload = {
            "ok": False,
            "error": err or "no market data available (provider unreachable)",
            "session": desc,
        }
        if args.as_json:
            print(json.dumps(payload))
        else:
            print(f"KRONOS  {desc['local_time']}  ·  {desc['phase']}")
            print(f"NO DATA — {payload['error']}")
            print("Nothing scanned. No verdicts produced.")
        return 2

    verdicts = screener.scan(data, _signals(data))
    takes = [v for v in verdicts if v.decision == "TAKE"]
    snap = governor.snapshot()

    if args.as_json:
        print(json.dumps({
            "ok": True,
            "session": desc,
            "budget": snap,
            "verdicts": [v.to_dict() for v in verdicts],
        }))
        return 0

    # --- compact human output ------------------------------------------------
    print(
        f"KRONOS  {desc['local_time']}  ·  {desc['phase']}"
        f"  ·  {len(data)} scanned  ·  credits {snap['percent_used']}% ({snap['status']})"
    )
    if not session.is_active() and not args.ignore_session:
        print(f"(outside {desc['window']} {desc['timezone']} — results still valid)")

    shown = verdicts if args.all else takes
    if not shown:
        print(f"\nNo TAKEs. {len(verdicts)} names screened, all SKIP.")
    else:
        print()
        for v in shown:
            print(_fmt(v, args.all))
        if not args.all:
            print(f"\n+{len(verdicts) - len(takes)} SKIP")

    paid = sum(1 for v in verdicts if v.tier != "none")
    print(f"\ncost: {paid} paid call(s), {snap['spent_tokens']:,} tokens used this window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
