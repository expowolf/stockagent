# KRONOS AI Stock Scanner

A local, free-tier quantitative stock scanner with embedded TradingView
visualization. Daily OHLCV from `yfinance`, four quant engines, a FastAPI
backend, and a Next.js frontend.

## Architecture

```
kronos-scanner/
├── backend/
│   ├── main.py                 FastAPI app (/health, /scan, /signals/{ticker})
│   ├── models.py               SQLAlchemy ORM + session helpers
│   ├── engines/
│   │   ├── ou_reversion.py     Ornstein-Uhlenbeck mean reversion
│   │   ├── hmm_regime.py       Hidden Markov regime detector
│   │   ├── kalman_trend.py     Kalman local-linear-trend filter
│   │   └── evt_risk.py         EVT (Peak-Over-Threshold) tail risk
│   ├── data/ingestion.py       yfinance loader + parquet caching + indicators
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── pages/index.tsx         Scan dashboard
│   ├── components/TradingViewChart.tsx
│   └── package.json
└── docker-compose.yml          PostgreSQL + FastAPI
```

## Quick start

### Backend (local)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload          # http://localhost:8000/docs
```

The database is optional: if `DATABASE_URL` is unreachable the API still
serves computed signals, it just skips persistence.

### Full stack (Docker)

```bash
docker compose up --build          # Postgres + backend on :8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                        # http://localhost:3000
```

Set `NEXT_PUBLIC_API_BASE` if the backend is not on `http://localhost:8000`.

## API

| Endpoint            | Description                                        |
|---------------------|----------------------------------------------------|
| `GET /health`       | Liveness + DB status                               |
| `GET /signals/{t}`  | Compute all engines for one ticker                 |
| `GET /scan`         | Batch scan a universe, ranked by OU score          |

## Engines

| Engine  | Method                          | Output                              |
|---------|---------------------------------|-------------------------------------|
| OU      | OLS discretization, half-life   | `ou_score` (0–100), `ou_halflife`   |
| HMM     | Gaussian HMM (Baum-Welch)       | regime label + confidence           |
| Kalman  | Local linear trend state-space  | normalized `kalman_alpha`           |
| EVT     | GPD over loss-tail exceedances  | 99% VaR + Expected Shortfall        |

## Trading agent

A phone-facing agent that sweeps ~20 tickers and answers one question per name:
**take the trade or not.** No journal, no report.

Open `/agent/ui` on your phone and add it to the home screen.

### Cost funnel

Every stage before the last is pure math and costs nothing, so a 20-ticker
sweep typically spends a couple of small calls rather than twenty.

| Stage | Work | Cost |
|-------|------|------|
| 0 | Load cached OHLCV | free |
| 1 | Candlestick patterns + OU/HMM/Kalman/EVT | free |
| 2 | `strategy.screen()` | free |
| 3 | Model call, top-K candidates only | paid, governed |

### Trading session

The agent runs on **your** clock. Defaults to **07:30–11:00 America/Denver** —
the 09:30 ET open through the pre-lunch slowdown.

| Phase | Local time | Cadence | Why |
|-------|-----------|---------|-----|
| `open-burst` | 07:30–08:30 | **150s** (2× faster) | Highest volatility, setups appear |
| `core` | 08:30–10:00 | 300s | Steady mid-session |
| `taper` | 10:00–11:00 | 540s (1.8× slower) | Tape quiets, stop spending on noise |

Outside the window the loop **sleeps until the next open** rather than scanning
a closed market. Weekends and all ten NYSE holidays are skipped automatically
(rule-derived, including Good Friday and observed dates — nothing to maintain).

Cadence tapering doubles as credit management: budget goes where the moves are.

Tune via env: `KRONOS_TZ`, `KRONOS_SESSION_START`, `KRONOS_SESSION_END`,
`KRONOS_BURST_MINUTES`, `KRONOS_TAPER_MINUTES`, `KRONOS_BASE_INTERVAL`.

### Credit governance

`agent/budget.py` paces spend against elapsed time rather than burning budget
front-loaded.

- **Time-proportional pacing** — over pace, it drops to free-tier; under pace,
  it permits the deeper model.
- **Untouchable reserve** (default 20%) — the window cannot be exhausted.
- **Hard per-sweep cap** (`max_paid_per_scan`, default 3) — bounds cost even if
  a strategy promotes every name.
- **Graceful degradation** — out of budget, the agent keeps producing verdicts
  from the numeric stages instead of stopping.
- **Adaptive interval** — the scan loop stretches or tightens to cover the
  remaining window.

Spend is paced against **session progress**, not wall-clock, so a 3.5h morning
uses its allowance properly instead of trickling it across a 5h window.

Measured over a full 07:30–11:00 MT session, every sweep hitting the paid cap
(worst case — real mornings surface far fewer candidates):

```
sweeps      : 49   (24 burst / 18 core / 7 taper)
tokens      : 153,700 / 320,000 spendable
budget used : 48%
reserve     : 80,000 untouched
status      : headroom
```

Tune via env: `KRONOS_TOKEN_BUDGET`, `KRONOS_WINDOW_HOURS`, `KRONOS_RESERVE_PCT`,
`KRONOS_MAX_PAID_PER_SCAN`, `KRONOS_UNIVERSE`.

### Plugging in a strategy

No strategy ships enabled — that slot is yours. Copy
`agent/strategies/template.py`, implement `screen()` and `evaluate()`, then:

```bash
export KRONOS_STRATEGY=agent.strategies.my_strategy:MyStrategy
```

Until then the agent runs end to end and answers SKIP, so you can watch the
funnel and the credit governor work before committing to any logic.

### Agent endpoints

| Endpoint | Description |
|---|---|
| `GET /agent/ui` | Mobile page (add to home screen) |
| `GET /agent/scan` | Run one sweep now |
| `GET /agent/verdicts` | Last results, no re-run (free) |
| `GET /agent/status` | Runner + session + credit status |
| `GET /agent/session` | Current phase in your timezone |
| `POST /agent/start` \| `/agent/stop` | Control the auto-scan loop |

## Roadmap

Phase 4 (composite scoring / ranking model) is implemented in a follow-up.
