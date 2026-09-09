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


# --------------------------------------------------------------------------- #
# Trading agent
# --------------------------------------------------------------------------- #
from fastapi.responses import HTMLResponse  # noqa: E402

from agent import CreditGovernor, Screener, session_from_env  # noqa: E402
from agent.runner import AgentRunner  # noqa: E402

AGENT_UNIVERSE = [
    t.strip().upper()
    for t in os.environ.get("KRONOS_UNIVERSE", ",".join(DEFAULT_UNIVERSE)).split(",")
    if t.strip()
]

# Trading window in your local timezone (default: 07:30-11:00 America/Denver,
# i.e. the 09:30 ET open through the pre-lunch slowdown).
_session = session_from_env()

_governor = CreditGovernor(
    token_budget=int(os.environ.get("KRONOS_TOKEN_BUDGET", "400000")),
    window_hours=float(os.environ.get("KRONOS_WINDOW_HOURS", "5")),
    reserve_pct=float(os.environ.get("KRONOS_RESERVE_PCT", "0.20")),
    # Pace spend across the trading session, not raw wall-clock.
    progress_fn=_session.progress,
)
_screener = Screener(
    governor=_governor,
    max_paid_per_scan=int(os.environ.get("KRONOS_MAX_PAID_PER_SCAN", "3")),
)


def _agent_load(tickers):
    return load_universe(tickers, period="1y")


def _agent_signals(ticker, df):
    return compute_signals(ticker, df)


def _on_take(fresh):
    """Push new TAKE verdicts to the phone."""
    for v in fresh:
        push(
            title=f"TAKE {v.ticker} @ ${v.price:.2f}",
            message=f"{v.reason}\nconfidence {v.confidence:.0%} · {v.tier} tier",
            priority="high",
            tags=["chart_increasing"],
        )


_agent = AgentRunner(
    universe=AGENT_UNIVERSE,
    load_fn=_agent_load,
    signal_fn=_agent_signals,
    governor=_governor,
    screener=_screener,
    on_take=_on_take,
    session=_session,
    base_interval=float(os.environ.get("KRONOS_BASE_INTERVAL", "300")),
)


@app.get("/agent/session")
def agent_session():
    """Where we are in your trading day, in your timezone."""
    return _session.describe()


@app.get("/agent/status")
def agent_status():
    """Runner + credit status. This is what the phone header shows."""
    return _agent.status()


@app.get("/agent/scan")
def agent_scan():
    """Run one sweep now and return every verdict."""
    verdicts = _agent.scan_once()
    return {
        "scanned_at": _agent.last_scan_at,
        "budget": _governor.snapshot(),
        "verdicts": [v.to_dict() for v in verdicts],
    }


@app.get("/agent/verdicts")
def agent_verdicts():
    """Last sweep's results without re-running (costs nothing)."""
    return {
        "scanned_at": _agent.last_scan_at,
        "budget": _governor.snapshot(),
        "verdicts": [v.to_dict() for v in _agent.last_verdicts],
    }


@app.post("/agent/start")
def agent_start(planned_cycles: int = 36):
    started = _agent.start(planned_cycles=planned_cycles)
    return {"started": started, "already_running": not started, **_agent.status()}


@app.post("/agent/stop")
def agent_stop():
    return {"stopped": _agent.stop(), "running": _agent.is_running}


@app.get("/agent/ui", response_class=HTMLResponse)
def agent_ui():
    """Self-contained mobile page. Add to home screen and it behaves like an app."""
    return _AGENT_UI_HTML


_AGENT_UI_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Kronos</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:#09090b;color:#e4e4e7;font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     padding:16px 14px calc(28px + env(safe-area-inset-bottom));min-height:100vh}
header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}
h1{font-size:19px;letter-spacing:-.02em}
h1 span{color:#22d3ee}
.meta{font-size:11px;color:#71717a}
.bar{height:6px;background:#27272a;border-radius:99px;overflow:hidden;margin:6px 0 4px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#eab308,#ef4444);
       transition:width .4s}
.budget{background:#18181b;border:1px solid #27272a;border-radius:12px;padding:11px 13px;margin-bottom:14px}
.budget .row{display:flex;justify-content:space-between;font-size:11px;color:#a1a1aa}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:10px;font-weight:600;
      text-transform:uppercase;letter-spacing:.04em}
.on-pace{background:#052e16;color:#4ade80}.throttling{background:#422006;color:#fbbf24}
.headroom{background:#082f49;color:#38bdf8}.exhausted-degraded-to-free{background:#450a0a;color:#f87171}
.open-burst{background:#052e16;color:#4ade80}.core{background:#082f49;color:#38bdf8}
.taper{background:#422006;color:#fbbf24}.closed,.done,.premarket{background:#27272a;color:#a1a1aa}
.card{background:#18181b;border:1px solid #27272a;border-left-width:4px;border-radius:12px;
      padding:13px 14px;margin-bottom:9px;display:flex;gap:12px;align-items:center}
.card.take{border-left-color:#22c55e;background:#0b1f14}
.card.skip{border-left-color:#3f3f46;opacity:.62}
.tk{font-weight:700;font-size:16px;letter-spacing:-.01em;min-width:62px}
.px{font-size:11px;color:#a1a1aa;font-variant-numeric:tabular-nums}
.why{font-size:12px;color:#d4d4d8;margin-top:3px}
.lv{font-size:10px;color:#71717a;margin-top:3px;font-variant-numeric:tabular-nums}
.dec{margin-left:auto;text-align:right}
.dec b{display:block;font-size:15px}
.take .dec b{color:#4ade80}.skip .dec b{color:#71717a}
.conf{font-size:10px;color:#71717a}
button{width:100%;padding:13px;border:0;border-radius:11px;background:#22d3ee;color:#06373f;
       font-weight:700;font-size:15px;margin-bottom:9px}
button:disabled{opacity:.5}
button.ghost{background:#27272a;color:#d4d4d8}
.empty{text-align:center;color:#52525b;padding:34px 0;font-size:13px}
</style></head><body>
<header>
  <h1>KRONOS <span>agent</span></h1>
  <div class="meta" id="stamp">—</div>
</header>

<div class="budget">
  <div class="row"><span>Session</span><span id="sphase" class="pill">—</span></div>
  <div class="bar"><i id="sfill" style="width:0%;background:#22d3ee"></i></div>
  <div class="row"><span id="swin">—</span><span id="snext">—</span></div>
</div>

<div class="budget">
  <div class="row"><span>Credits</span><span id="bstat" class="pill">—</span></div>
  <div class="bar"><i id="bfill" style="width:0%"></i></div>
  <div class="row"><span id="btok">—</span><span id="bwin">—</span></div>
</div>

<button id="scan" onclick="scan()">Scan now</button>
<button class="ghost" id="toggle" onclick="toggleLoop()">Start auto-scan</button>

<div id="list"><div class="empty">No scan yet.</div></div>

<script>
let running=false;

function fmt(n){return n>=1000?(n/1000).toFixed(1)+'k':n}

function paint(d){
  const b=d.budget||{};
  document.getElementById('bfill').style.width=(b.percent_used||0)+'%';
  const s=document.getElementById('bstat');
  s.textContent=(b.status||'—').replace(/-/g,' ');
  s.className='pill '+(b.status||'');
  document.getElementById('btok')
    .textContent=fmt(b.spent_tokens||0)+' / '+fmt(b.spendable_tokens||0)+' tokens';
  document.getElementById('bwin')
    .textContent='resets in '+(b.window_resets_in_minutes||0)+'m';
  if(d.scanned_at)document.getElementById('stamp')
    .textContent=new Date(d.scanned_at).toLocaleTimeString();

  const v=d.verdicts||[];
  const el=document.getElementById('list');
  if(!v.length){el.innerHTML='<div class="empty">No verdicts.</div>';return}
  el.innerHTML=v.map(x=>{
    const t=x.decision==='TAKE';
    const lv=(x.entry&&x.invalidation)
      ?`<div class="lv">entry ${x.entry.toFixed(2)} · stop ${x.invalidation.toFixed(2)}`
        +(x.target?` · target ${x.target.toFixed(2)}`:'')+`</div>`:'';
    return `<div class="card ${t?'take':'skip'}">
      <div><div class="tk">${x.ticker}</div><div class="px">$${x.price.toFixed(2)}</div></div>
      <div style="flex:1"><div class="why">${x.reason||''}</div>${lv}</div>
      <div class="dec"><b>${x.decision}</b>
        <div class="conf">${Math.round((x.confidence||0)*100)}% · ${x.tier}</div></div>
    </div>`}).join('');
}

function paintSession(s){
  const p=document.getElementById('sphase');
  p.textContent=(s.phase||'—').replace(/-/g,' ');
  p.className='pill '+(s.phase||'');
  document.getElementById('sfill').style.width=(s.session_progress_pct||0)+'%';
  document.getElementById('swin').textContent=s.window+' '+(s.timezone||'').split('/').pop();
  document.getElementById('snext').textContent = s.active
    ? Math.round(s.minutes_until_close)+'m left · scan x'+s.cadence_multiplier
    : (s.minutes_until_open>90
        ? 'opens in '+(s.minutes_until_open/60).toFixed(1)+'h'
        : 'opens in '+Math.round(s.minutes_until_open)+'m');
}

async function load(){
  try{paint(await (await fetch('/agent/verdicts')).json())}catch(e){}
  try{paintSession(await (await fetch('/agent/session')).json())}catch(e){}
}
async function scan(){
  const b=document.getElementById('scan');
  b.disabled=true;b.textContent='Scanning…';
  try{paint(await (await fetch('/agent/scan')).json())}
  catch(e){alert('Scan failed: '+e)}
  b.disabled=false;b.textContent='Scan now';
}
async function toggleLoop(){
  const t=document.getElementById('toggle');
  await fetch(running?'/agent/stop':'/agent/start',{method:'POST'});
  running=!running;
  t.textContent=running?'Stop auto-scan':'Start auto-scan';
}
load();
setInterval(load,30000);
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
