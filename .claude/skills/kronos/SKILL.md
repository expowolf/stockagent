---
name: kronos
description: Scan the watchlist and answer take-the-trade-or-not for each name. Use when the user runs /kronos, asks for a scan, asks "any setups", "what do I take", or wants a morning market check. Runs locally via Python — no server, no deployment.
---

# kronos — take it or don't

One job: sweep ~20 tickers and say **TAKE** or **SKIP** for each. Nothing else.

No journal. No narrative. No commentary the user didn't ask for.

## Run it

```bash
cd kronos-scanner/backend && python scan_cli.py
```

Useful flags:

| Flag | When |
|------|------|
| `--all` | user wants SKIPs too (default shows only TAKEs) |
| `--tickers NVDA,AAPL` | user named specific symbols |
| `--json` | you need to post-process |
| `--ignore-session` | user asks outside 07:30–11:00 MT and wants it anyway |

## Report the output almost verbatim

The CLI is already terse on purpose. **Relay it; do not re-analyse it.**
Re-deriving in prose is exactly the token waste this design exists to avoid.

Good:

```
KRONOS  2026-08-24 08:14 MDT  ·  open-burst  ·  20 scanned  ·  credits 12% (on-pace)

TAKE NVDA    214.72   78%  e214.72/s208.10/t226.00  bullish engulfing + drift agree

+19 SKIP

cost: 1 paid call, 3,240 tokens used this window
```

Then stop. Add at most one short line if something genuinely needs flagging
(exit code 2, budget exhausted, session closed).

Do **not** add: a table of every SKIP, per-name reasoning, market outlook,
disclaimers, or a summary of what the numbers "mean".

## Exit codes

- `0` — scanned fine
- `2` — **no market data**. Say so plainly and stop. Never fill the gap with
  web-search quotes or remembered prices; a fabricated price on a trade
  decision is the one unacceptable failure. If the user wants a data
  workaround, offer it — don't silently substitute one.

## Scanning all morning

For a continuous session, this composes with `/loop`:

```
/loop 5m /kronos
```

The CLI is already session-aware (07:30–11:00 MT, phase-tapered cadence) and
credit-governed, so repeated invocations stay inside budget on their own.

## Phone alerts

New TAKEs push to the phone automatically when `NTFY_TOPIC` is set. Every
alert carries its **stop loss** in both the title and body, plus position size
under the 1% rule. Alerts are deduped, so `/loop` will not re-alert the same
setup.

If a TAKE appears but the summary says `notifications off`, tell the user once
that `NTFY_TOPIC` is unset — don't repeat it every scan.

## Risk gate

Every TAKE is gated on risk discipline before it can reach the user, whatever
the strategy says:

- **no stop loss = never a TAKE** (hard block, non-negotiable)
- stop must clear normal noise (>= 0.5x ATR)
- R:R at or above the configured minimum
- position sized off the stop, not off conviction

When a name is blocked, the reason is prefixed `risk:` — report it as-is. Do
not argue with the gate or suggest overriding it.

## Credit discipline

The Python does the analysis for free — candlestick detection and the
OU/HMM/Kalman/EVT engines cost zero tokens. Only 0–3 finalists per sweep ever
reach a model call, and the governor throttles those when spend runs ahead of
the session clock.

Your job is to keep the *chat* side just as cheap: run, relay, stop. If the
user asks a follow-up about one name, answer that name — don't re-scan.
