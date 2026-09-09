"""
Reader for the GitHub Actions data relay.

The workflow at .github/workflows/kronos-data.yml runs on GitHub's runners
(which can reach Yahoo) every 15 minutes and commits data/live/universe.parquet
to the repo. This module reads that file via raw.githubusercontent.com —
which THIS environment can reach — so the sandbox can scan a live universe
without ever calling Yahoo directly.

Enable by setting KRONOS_RELAY_REPO=owner/repo (e.g. expowolf/stockagent).
Fetched frames are cached briefly so repeat sweeps within a cadence do not
re-download.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


CACHE_TTL_S = float(os.environ.get("KRONOS_RELAY_TTL_S", "600"))  # 10 min
_MEM: dict = {"at": 0.0, "frames": {}, "manifest": {}}


def _url(path: str) -> str:
    repo = os.environ.get("KRONOS_RELAY_REPO", "").strip()
    branch = os.environ.get("KRONOS_RELAY_BRANCH", "claude/trusting-archimedes-jfd2sw").strip()
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def configured() -> bool:
    return bool(os.environ.get("KRONOS_RELAY_REPO", "").strip()) and httpx is not None


def _get(path: str) -> bytes:
    resp = httpx.get(_url(path), timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def status() -> dict:
    """Manifest snapshot: fetched_at, loaded/requested, staleness."""
    if not configured():
        return {"configured": False}
    try:
        m = json.loads(_get("data/live/manifest.json").decode())
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "error": str(exc)[:120]}
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(m["fetched_at"])).total_seconds()
        m["age_seconds"] = round(age)
    except Exception:  # noqa: BLE001
        pass
    return {"configured": True, **m}


def load_universe(tickers=None) -> dict:
    """
    Return {ticker: DataFrame(open, high, low, close, volume)}.

    Reads the stacked universe parquet from the relay repo, then splits by
    ticker. Optionally filters to `tickers`.
    """
    if not configured():
        return {}

    now = time.time()
    if _MEM["frames"] and now - _MEM["at"] < CACHE_TTL_S:
        frames = _MEM["frames"]
    else:
        try:
            data = _get("data/live/universe.parquet")
        except Exception as exc:  # noqa: BLE001
            print(f"[relay] read failed: {exc}")
            return {}
        try:
            df = pd.read_parquet(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            print(f"[relay] parquet parse failed: {exc}")
            return {}
        frames = {tk: g.drop(columns=["ticker"]).reset_index(drop=True)
                  for tk, g in df.groupby("ticker", sort=False)}
        _MEM.update({"at": now, "frames": frames})

    if tickers:
        want = {t.upper() for t in tickers}
        return {t: frames[t] for t in frames if t in want}
    return frames
