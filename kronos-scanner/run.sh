#!/usr/bin/env bash
# run.sh — Kronos, one command, zero API keys.
#
#   ./run.sh            one sweep of the full universe
#   ./run.sh check      10-second data reachability test
#   ./run.sh loop       sweep every 15 minutes until Ctrl-C
#   ./run.sh AAPL,NVDA  sweep specific tickers
#
# No signup, no keys, no server, no deployment. Yahoo Finance is keyless and is
# the only data source needed.

set -euo pipefail
cd "$(dirname "$0")/backend"

# ---- strategy + account ----------------------------------------------------
export KRONOS_STRATEGY="${KRONOS_STRATEGY:-agent.strategies.episodic_pivot:EpisodicPivotStrategy}"
export KRONOS_ACCOUNT_SIZE="${KRONOS_ACCOUNT_SIZE:-570}"
export KRONOS_RISK_PER_TRADE="${KRONOS_RISK_PER_TRADE:-0.04}"
export KRONOS_TZ="${KRONOS_TZ:-America/Denver}"
export KRONOS_SESSION_START="${KRONOS_SESSION_START:-07:30}"
export KRONOS_SESSION_END="${KRONOS_SESSION_END:-11:00}"

PY="${PYTHON:-python3}"

ensure_deps() {
  if ! $PY -c "import yfinance, pandas, numpy, scipy, hmmlearn" 2>/dev/null; then
    echo "installing dependencies (one time)..."
    $PY -m pip install -q -r requirements.txt
  fi
}

case "${1:-scan}" in
  check)
    ensure_deps
    echo "checking Yahoo Finance reachability..."
    $PY - <<'EOF'
import sys
try:
    import yfinance as yf
    df = yf.download("SPY", period="5d", interval="1d",
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        print("FAIL — Yahoo returned no rows. Check your connection.")
        sys.exit(1)
    close = float(df["Close"].iloc[-1])
    print(f"OK — SPY last close ${close:,.2f} over {len(df)} bars.")
    print("Data path works. No API key was used.")
except Exception as exc:
    print(f"FAIL — {exc}")
    sys.exit(1)
EOF
    ;;

  loop)
    ensure_deps
    echo "Kronos loop — sweeping every 15 min. Ctrl-C to stop."
    while true; do
      $PY scan_cli.py || true
      echo "--- next sweep in 15 min ---"
      sleep 900
    done
    ;;

  scan)
    ensure_deps
    exec $PY scan_cli.py
    ;;

  *)
    ensure_deps
    exec $PY scan_cli.py --tickers "$1" --ignore-session
    ;;
esac
