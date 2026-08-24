#!/usr/bin/env python3
"""
Fetch the universe on a GitHub Actions runner and write it to data/live/.

Runs where Yahoo IS reachable. The output is committed to the repo and read
by the scanner via raw.githubusercontent.com from a sandbox where Yahoo is
NOT reachable — that is the whole point of this file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "live"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from data.ingestion import DEFAULT_UNIVERSE  # noqa: E402


def fetch(tickers: list, period: str = "6mo") -> dict:
    """yf.download batches — 150 tickers is a couple of requests."""
    frames: dict = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        try:
            raw = yf.download(
                " ".join(chunk), period=period, interval="1d",
                auto_adjust=True, progress=False, threads=True, group_by="ticker",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fetch] chunk failed: {exc}", file=sys.stderr)
            continue

        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all").rename(columns=str.lower).reset_index()
                df = df.rename(columns={"date": "date", "Date": "date"})
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
                keep = ["date", "open", "high", "low", "close", "volume"]
                df = df[[c for c in keep if c in df.columns]].dropna(subset=["close"])
                if not df.empty:
                    frames[t.upper()] = df
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {t}: {exc}", file=sys.stderr)
    return frames


def main() -> int:
    universe = [t.strip().upper() for t in os.environ.get(
        "KRONOS_UNIVERSE", ",".join(DEFAULT_UNIVERSE)
    ).split(",") if t.strip()]

    t0 = time.time()
    frames = fetch(universe)
    print(f"[fetch] {len(frames)}/{len(universe)} in {time.time()-t0:.1f}s")

    # One parquet file with all tickers stacked — small, fast to read.
    stacked = []
    for tk, df in frames.items():
        d = df.copy()
        d["ticker"] = tk
        stacked.append(d)
    if not stacked:
        print("[fetch] nothing fetched — aborting write", file=sys.stderr)
        return 2

    combined = pd.concat(stacked, ignore_index=True)
    combined.to_parquet(OUT / "universe.parquet", index=False, compression="zstd")

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loaded": len(frames),
        "requested": len(universe),
        "missing": sorted(set(universe) - set(frames)),
        "tickers": sorted(frames),
        "last_bar": max(df["date"].max().isoformat()[:10] for df in frames.values()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fetch] wrote universe.parquet ({(OUT / 'universe.parquet').stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
