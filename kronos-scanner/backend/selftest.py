# selftest.py
"""
Kronos end-to-end self-test.

Run this anywhere with normal internet access to verify the full pipeline:
network -> yfinance -> indicators -> all four engines -> signal output.

    python selftest.py             # default tickers
    python selftest.py AAPL NVDA   # specific tickers

Exit code 0 = all checks passed, 1 = something failed.
"""

import sys
import math
import traceback
from typing import List

CHECKS: List[tuple] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    icon = "PASS" if ok else "FAIL"
    line = f"[{icon}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def check_connectivity() -> bool:
    """
    Confirm we can actually complete an HTTPS request to the data provider.

    A raw TCP connect is NOT sufficient: behind a filtering proxy the socket
    opens but the CONNECT tunnel is refused (403), so we must issue a real
    request to know whether live data can flow.
    """
    host = "query1.finance.yahoo.com"
    url = f"https://{host}/v8/finance/chart/AAPL?range=5d&interval=1d"
    try:
        import httpx

        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (kronos-selftest)"},
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = ""
        if "CONNECT" in msg or "403" in msg or "tunnel" in msg.lower():
            hint = (
                " This looks like a proxy/egress policy denying the connection "
                "rather than the provider being down — live data cannot flow "
                "from this machine until that host is allowed."
            )
        return record("HTTPS to data provider", False, f"{type(exc).__name__}: {msg}.{hint}")

    if resp.status_code != 200:
        return record(
            "HTTPS to data provider",
            False,
            f"HTTP {resp.status_code} from {host} — blocked or rate-limited, not a code fault",
        )

    return record("HTTPS to data provider", True, f"{host} responded 200")


def check_fetch(ticker: str):
    """Fetch real OHLCV and sanity-check the frame."""
    from data.ingestion import load_ticker

    try:
        df = load_ticker(ticker, period="2y", use_cache=False)
    except Exception as exc:  # noqa: BLE001
        record(f"{ticker}: fetch", False, str(exc))
        return None

    if df is None or df.empty:
        record(f"{ticker}: fetch", False, "empty frame returned")
        return None

    rows = len(df)
    if rows < 200:
        record(f"{ticker}: fetch", False, f"only {rows} rows, expected ~500 for 2y")
        return None

    required = {"open", "high", "low", "close", "volume", "sma_20", "sma_50", "rsi_14"}
    missing = required - set(df.columns)
    if missing:
        record(f"{ticker}: fetch", False, f"missing columns: {sorted(missing)}")
        return None

    last_close = float(df["close"].iloc[-1])
    if not (last_close > 0) or math.isnan(last_close):
        record(f"{ticker}: fetch", False, f"bad close price: {last_close}")
        return None

    record(
        f"{ticker}: fetch",
        True,
        f"{rows} rows, last close ${last_close:,.2f} on {df['date'].iloc[-1]}",
    )
    return df


def check_engines(ticker: str, df) -> bool:
    """Run all four engines and validate every output is in a sane range."""
    from main import compute_signals

    try:
        sig = compute_signals(ticker, df)
    except Exception as exc:  # noqa: BLE001
        record(f"{ticker}: engines", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False

    ok = True

    # OU score must be a real number in [0, 100].
    if not (0.0 <= sig.ou_score <= 100.0) or math.isnan(sig.ou_score):
        ok = record(f"{ticker}: OU score", False, f"out of range: {sig.ou_score}") and ok
    else:
        ok = record(f"{ticker}: OU score", True, f"{sig.ou_score:.1f}/100 (half-life {sig.ou_halflife}d)") and ok

    # HMM must resolve to a real regime, not the Unknown fallback.
    if sig.hmm_regime in ("Unknown", None):
        ok = record(f"{ticker}: HMM regime", False, "fell back to Unknown — fit failed") and ok
    elif not (0.0 <= sig.hmm_confidence <= 1.0):
        ok = record(f"{ticker}: HMM regime", False, f"bad confidence {sig.hmm_confidence}") and ok
    else:
        ok = record(f"{ticker}: HMM regime", True, f"{sig.hmm_regime} @ {sig.hmm_confidence:.0%}") and ok

    # Kalman alpha must be finite.
    if sig.kalman_alpha is None or math.isnan(sig.kalman_alpha) or math.isinf(sig.kalman_alpha):
        ok = record(f"{ticker}: Kalman alpha", False, f"non-finite: {sig.kalman_alpha}") and ok
    else:
        ok = record(f"{ticker}: Kalman alpha", True, f"{sig.kalman_alpha:+.6f}") and ok

    # EVT VaR should be a positive loss magnitude, and ES >= VaR.
    if sig.evt_var_99 is None:
        ok = record(f"{ticker}: EVT tail risk", False, "GPD fit returned nothing") and ok
    elif not (0.0 < sig.evt_var_99 < 1.0):
        ok = record(f"{ticker}: EVT tail risk", False, f"implausible VaR: {sig.evt_var_99}") and ok
    elif sig.evt_expected_shortfall is not None and sig.evt_expected_shortfall < sig.evt_var_99:
        ok = record(
            f"{ticker}: EVT tail risk", False,
            f"ES ({sig.evt_expected_shortfall:.4f}) < VaR ({sig.evt_var_99:.4f}) — violates definition",
        ) and ok
    else:
        ok = record(
            f"{ticker}: EVT tail risk", True,
            f"VaR99 {sig.evt_var_99*100:.2f}% / ES {sig.evt_expected_shortfall*100:.2f}%",
        ) and ok

    return ok


def main() -> int:
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "NVDA", "KO"]

    print("=" * 62)
    print("KRONOS SELF-TEST — live data pipeline")
    print("=" * 62)

    if not check_connectivity():
        print("\nAborting: no network path to the data provider.")
        _summary()
        return 1

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        df = check_fetch(ticker)
        if df is not None:
            check_engines(ticker, df)

    return _summary()


def _summary() -> int:
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print("\n" + "=" * 62)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 62)
    if passed != total:
        print("\nFailed checks:")
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("All good — live data flows end to end through all four engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
