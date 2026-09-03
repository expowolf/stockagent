# TradingView → this repo → the agent

Fifteen minutes of setup on your desktop. Do it once.

## The one thing to understand first

There is no way to plug TradingView directly into the Claude session on your
phone. Two hard constraints, both verified rather than assumed:

1. **The cloud session cannot reach TradingView.** This environment's network
   policy refuses it — the proxy answers `403 to CONNECT` for
   `scanner.tradingview.com`, `symbol-search.tradingview.com`, and
   `www.tradingview.com`.
2. **The TradingView MCP server is desktop-only by design.** It drives the real
   TradingView Desktop app over Chrome DevTools Protocol on `localhost:9222`.
   Nothing outside your machine can reach that, by definition.

So your desktop is the only place the two sides can meet. It reads the charts
and commits bars to this repo; the cloud session and the bot read the repo.
That is the same relay pattern that already solved the Yahoo problem here, and
it is why you can still drive everything from your phone.

**What you get:** real TradingView bars, your charts, your indicators — and a
phone-accessible agent. **What you give up:** the desktop has to be awake and
running for live data. If it sleeps, the relay stops and the loader refuses to
serve stale bars rather than pretend.

---

## Step 1 — Install the MCP server

```bash
git clone https://github.com/tradesdontlie/tradingview-mcp.git ~/tradingview-mcp
cd ~/tradingview-mcp
npm install
```

Verified here on Node v22: installs clean, **147 of 152 unit tests pass**. The
5 failures are all `pine check (server compile)`, which call TradingView's
compile server — blocked in the cloud sandbox where this was tested. They
should pass on your machine.

## Step 2 — Launch TradingView with debugging enabled

It must be started this way, or the bridge has nothing to attach to.

**Mac**
```bash
/Applications/TradingView.app/Contents/MacOS/TradingView --remote-debugging-port=9222
```

**Windows** — TradingView now ships as an MSIX package, so don't try to launch
the `.exe` directly:
```bat
cd %USERPROFILE%\tradingview-mcp
scripts\launch_tv_debug.bat
```

Confirm the bridge sees it:
```bash
cd ~/tradingview-mcp && node src/cli/index.js status
```

## Step 3 — Wire it into Claude Code on the desktop

Add to `~/.claude/.mcp.json` (merge into `mcpServers` if the file exists —
don't overwrite other servers):

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "node",
      "args": ["/Users/YOU/tradingview-mcp/src/server.js"]
    }
  }
}
```

Use your real path. This gives the **desktop** Claude 78 TradingView tools.
It does not affect the phone session — see the constraint above.

## Step 4 — Start the relay

From this repo, on the desktop:

```bash
node desktop/tv_relay.mjs
```

It pulls SPY and QQQ 5-minute bars every 60 seconds, writes them to
`data/live/tv/`, and pushes. Options:

```bash
KRONOS_TV_SYMBOLS=SPY,QQQ,IWM node desktop/tv_relay.mjs
KRONOS_TV_TF=15 node desktop/tv_relay.mjs
node desktop/tv_relay.mjs --once      # one pass
node desktop/tv_relay.mjs --no-push   # local only
```

**It drives your actual chart.** The relay switches symbol and timeframe on the
real window each pass, so don't plan to stare at that window while it runs — use
a second layout or a second monitor. It restores your original symbol and
timeframe on Ctrl-C.

## Step 5 — Confirm the cloud can see it

From anywhere, including your phone, ask me to run:

```bash
python kronos-scanner/backend/research/tv_source.py
```

You want lines like:

```
SPY_5: 500 bars · last 2026-09-01T14:05:00Z c=772.31 · written 1 min ago
```

`STALE` means the desktop relay stopped.

---

## How the strategy plugs in

`tv_source.load_tv(ticker)` returns exactly what the bot's own `load()` returns:
a DataFrame indexed by tz-aware `America/New_York` timestamps, columns
`open/high/low/close/volume`, regular session only, or `None`.

That means swapping a strategy from Yahoo to TradingView is a one-line change,
and your strategy code never has to know where bars came from.

Two deliberate behaviours worth knowing:

- **The forming bar is dropped by default.** TradingView's last bar is still
  being built. Acting on it means acting on a number that has not happened yet.
  Pass `drop_forming_bar=False` if you truly want it.
- **Stale data raises instead of returning.** If the relay has been silent for
  more than 10 minutes, `load_tv` raises `StaleRelay`. A missing feed returning
  `None` is fine — a *dead* feed silently serving old bars is how you get a
  confident signal computed on yesterday's tape.

## Status of the old bot

The QQQ/SPY VWAP-pullback bot is **paused**: schedule commented out and the
`.github/kick` push trigger unwired, so the morning routine cannot restart it.
`workflow_dispatch` still works for a deliberate manual run.
