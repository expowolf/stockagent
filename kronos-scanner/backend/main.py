# main.py
"""FastAPI application exposing the Kronos quant engines and a daily scan."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.ingestion import DEFAULT_UNIVERSE, load_ticker, load_universe
from engines import EVTTailRisk, HMMRegimeDetector, KalmanTrendFilter, OUMeanReversion
from models import (
    DailyMarketData,
    QuantSignalOutputs,
    init_db,
    make_engine,
    make_session_factory,
)
from notify import push

_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

app = FastAPI(title="Kronos AI Stock Scanner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_Session = None


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SignalResponse(BaseModel):
    ticker: str
    date: datetime
    current_price: float
    ou_score: float
    ou_halflife: Optional[float]
    hmm_regime: str
    hmm_confidence: float
    kalman_alpha: float
    evt_var_99: Optional[float]
    evt_expected_shortfall: Optional[float]


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def compute_signals(ticker: str, df) -> SignalResponse:
    """Run all four engines on a single ticker's OHLCV frame."""
    prices = df["close"].to_numpy(dtype=float)
    current_price = float(prices[-1])
    last_date = df["date"].iloc[-1]
    if not isinstance(last_date, datetime):
        last_date = datetime.fromtimestamp(
            float(np.datetime64(last_date).astype("datetime64[s]").astype("int64"))
        )

    # 1) OU mean reversion
    ou = OUMeanReversion(window=252)
    ou_fit = ou.fit(prices)
    ou_score = ou.score(current_price)
    ou_halflife = ou.halflife if ou_fit else None

    # 2) HMM regime
    hmm = HMMRegimeDetector(window=252, n_states=4)
    hmm.fit(prices)
    regime, _probs, confidence = hmm.predict_regime(prices)

    # 3) Kalman trend / alpha
    kalman = KalmanTrendFilter(process_variance=0.01, measurement_variance=1.0)
    _trends, _resid, alpha = kalman.smooth_series(prices)

    # 4) EVT tail risk
    evt = EVTTailRisk(threshold_percentile=90, window=252)
    evt_fit = evt.fit(prices)
    if evt_fit:
        var99, es99, _risk = evt.calculate_var_es(tail_prob=0.01)
    else:
        var99, es99 = None, None

    return SignalResponse(
        ticker=ticker.upper(),
        date=last_date,
        current_price=current_price,
        ou_score=round(float(ou_score), 2),
        ou_halflife=round(float(ou_halflife), 2) if ou_halflife else None,
        hmm_regime=regime,
        hmm_confidence=round(float(confidence), 4),
        kalman_alpha=round(float(alpha), 6),
        evt_var_99=round(float(var99), 6) if var99 is not None else None,
        evt_expected_shortfall=round(float(es99), 6) if es99 is not None else None,
    )


def persist_signals(sig: SignalResponse) -> None:
    """Upsert a signal row (best-effort; ignores DB errors so scans don't fail)."""
    if _Session is None:
        return
    try:
        with _Session() as session:
            row = (
                session.query(QuantSignalOutputs)
                .filter_by(ticker=sig.ticker, date=sig.date)
                .one_or_none()
            )
            if row is None:
                row = QuantSignalOutputs(ticker=sig.ticker, date=sig.date)
                session.add(row)
            row.ou_score = sig.ou_score
            row.ou_halflife = sig.ou_halflife
            row.hmm_regime = sig.hmm_regime
            row.hmm_confidence = sig.hmm_confidence
            row.kalman_alpha = sig.kalman_alpha
            row.evt_var_99 = sig.evt_var_99
            row.evt_expected_shortfall = sig.evt_expected_shortfall
            session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[persist] skipped {sig.ticker}: {exc}")


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _startup() -> None:
    global _engine, _Session
    try:
        _engine = make_engine()
        init_db(_engine)
        _Session = make_session_factory(_engine)
    except Exception as exc:  # noqa: BLE001 - DB is optional for read-only use
        print(f"[startup] database unavailable, running without persistence: {exc}")
        _engine, _Session = None, None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "db": _Session is not None}


@app.get("/signals/{ticker}", response_model=SignalResponse)
def get_signal(ticker: str, period: str = "2y"):
    df = load_ticker(ticker, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    sig = compute_signals(ticker, df)
    persist_signals(sig)
    return sig


@app.get("/scan", response_model=List[SignalResponse])
def scan(
    tickers: Optional[List[str]] = Query(default=None),
    period: str = "2y",
    limit: int = 100,
):
    """Run a batch scan. Defaults to the built-in universe if none supplied."""
    universe = tickers or DEFAULT_UNIVERSE
    universe = universe[:limit]
    data = load_universe(universe, period=period)

    results: List[SignalResponse] = []
    for tkr, df in data.items():
        if df.empty or len(df) < 50:
            continue
        try:
            sig = compute_signals(tkr, df)
            persist_signals(sig)
            results.append(sig)
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] {tkr} failed: {exc}")

    # Rank by OU mean-reversion opportunity (descending).
    results.sort(key=lambda s: s.ou_score, reverse=True)
    return results


def _claude_analysis(ticker: str, alert_msg: str, sig: SignalResponse) -> str:
    """
    Ask Claude to interpret a TradingView alert in the context of Kronos signals.
    Returns a concise analysis string suitable for a push notification.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return f"{ticker} alert received. OU={sig.ou_score:.1f} | Regime={sig.hmm_regime} | VaR={sig.evt_var_99*100:.2f}%" if sig.evt_var_99 else f"{ticker} alert. OU={sig.ou_score:.1f} | Regime={sig.hmm_regime}"

    prompt = f"""TradingView fired an alert for {ticker}: "{alert_msg}"

Kronos quant signals as of {sig.date.strftime('%Y-%m-%d')}:
- Current price: ${sig.current_price:.2f}
- OU mean-reversion score: {sig.ou_score:.1f}/100 (half-life {sig.ou_halflife:.1f}d)
- HMM regime: {sig.hmm_regime} (confidence {sig.hmm_confidence:.0%})
- Kalman alpha (drift): {sig.kalman_alpha:+.4f}
- 99% VaR: {f'{sig.evt_var_99*100:.2f}%' if sig.evt_var_99 else 'n/a'}

In 2-3 sentences: what does this alert mean given these signals? Is this a high-conviction setup or noise? Be direct, no disclaimers."""

    try:
        msg = _anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[claude] analysis failed: {exc}")
        return f"{ticker} alert: {alert_msg}"


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
    """
    Receive a TradingView alert, run Kronos engines, ask Claude for context,
    then push a notification to your phone via ntfy.sh.

    TradingView alert message body (set this in your TV alert):
    {
      "ticker": "{{ticker}}",
      "price": {{close}},
      "volume": {{volume}},
      "message": "{{strategy.order.comment}}"
    }

    Required env vars:  NTFY_TOPIC, ANTHROPIC_API_KEY
    """
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        # TradingView can send plain-text alerts too.
        body = {"message": (await request.body()).decode()}

    ticker = str(body.get("ticker", body.get("symbol", "UNKNOWN"))).upper().strip()
    price = body.get("price") or body.get("close")
    alert_msg = str(body.get("message", body.get("alert", "Alert fired")))

    # Run Kronos engines on the alerted ticker.
    sig: Optional[SignalResponse] = None
    try:
        df = load_ticker(ticker, period="2y")
        if not df.empty:
            sig = compute_signals(ticker, df)
            persist_signals(sig)
    except Exception as exc:  # noqa: BLE001
        print(f"[webhook] signal computation failed for {ticker}: {exc}")

    if sig is None:
        # Can't enrich — send a bare alert.
        push(
            title=f"Kronos Alert: {ticker}",
            message=f"{alert_msg}\nPrice: {price or 'n/a'}\n(No Kronos data available)",
            priority="high",
            tags=["chart_increasing", "bell"],
        )
        return {"status": "sent", "ticker": ticker, "enriched": False}

    # Ask Claude for a quick interpretation.
    analysis = _claude_analysis(ticker, alert_msg, sig)

    # Decide notification priority from signal strength.
    if sig.ou_score > 75:
        priority = "urgent"
        tags = ["rotating_light", "chart_increasing"]
    elif sig.ou_score > 55:
        priority = "high"
        tags = ["bell", "chart_increasing"]
    else:
        priority = "default"
        tags = ["bell"]

    push_body = (
        f"{analysis}\n\n"
        f"OU {sig.ou_score:.1f} | {sig.hmm_regime} | "
        f"VaR {sig.evt_var_99*100:.2f}%" if sig.evt_var_99 else
        f"{analysis}\n\nOU {sig.ou_score:.1f} | {sig.hmm_regime}"
    )

    push(
        title=f"Kronos: {ticker} @ ${sig.current_price:.2f}",
        message=push_body,
        priority=priority,
        tags=tags,
    )

    return {
        "status": "sent",
        "ticker": ticker,
        "enriched": True,
        "ou_score": sig.ou_score,
        "regime": sig.hmm_regime,
        "analysis_preview": analysis[:120],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
