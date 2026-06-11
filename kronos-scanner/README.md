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

## Roadmap

Phase 4 (composite scoring / ranking model) is implemented in a follow-up.
