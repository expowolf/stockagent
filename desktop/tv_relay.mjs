#!/usr/bin/env node
/**
 * TradingView -> repo relay. Runs on the DESKTOP, next to TradingView.
 *
 * WHY THIS EXISTS
 *   The trading bot and the phone-accessible Claude session both run in the
 *   cloud, where tradingview.com is blocked by network policy (the proxy
 *   answers 403 to CONNECT). The TradingView MCP server talks to the desktop
 *   app over Chrome DevTools Protocol on localhost:9222 — reachable only from
 *   this machine.
 *
 *   So the desktop is the only place both sides can meet. This script reads
 *   bars off the real TradingView charts and commits them to the repo, which
 *   the cloud can read. Same relay pattern that already solved Yahoo here.
 *
 * WHAT IT DOES NOT DO
 *   It does not decide trades and it does not alert. It moves data. The
 *   strategy lives in the repo and is reviewed like code.
 *
 * USAGE
 *   node desktop/tv_relay.mjs                    # SPY,QQQ on 5m, loop
 *   KRONOS_TV_SYMBOLS=SPY,QQQ,IWM node desktop/tv_relay.mjs
 *   node desktop/tv_relay.mjs --once             # single pass, no loop
 *   node desktop/tv_relay.mjs --no-push          # write locally, don't push
 *
 * PREREQUISITE
 *   TradingView Desktop running with --remote-debugging-port=9222 and the
 *   tradingview-mcp checkout present (see SETUP-TRADINGVIEW.md).
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');

const TV_HOME = process.env.KRONOS_TV_HOME
  || join(process.env.HOME || process.env.USERPROFILE || '', 'tradingview-mcp');
const TV_CLI = join(TV_HOME, 'src', 'cli', 'index.js');

const SYMBOLS = (process.env.KRONOS_TV_SYMBOLS || 'SPY,QQQ')
  .split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
const TF = process.env.KRONOS_TV_TF || '5';
const BARS = Number(process.env.KRONOS_TV_BARS || 500);   // CLI hard cap is 500
const EVERY_MS = Number(process.env.KRONOS_TV_EVERY || 60) * 1000;
const OUT_DIR = join(REPO, 'data', 'live', 'tv');
const BRANCH = process.env.KRONOS_BRANCH || 'claude/trusting-archimedes-jfd2sw';

const ONCE = process.argv.includes('--once');
const NO_PUSH = process.argv.includes('--no-push');

const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}Z]`, ...a);

/** Run the tv CLI and parse its JSON. Throws with the CLI's own message. */
async function tv(args) {
  const { stdout } = await execFileAsync('node', [TV_CLI, ...args], {
    cwd: TV_HOME, timeout: 45_000, maxBuffer: 32 * 1024 * 1024,
  });
  const s = stdout.trim();
  if (!s) throw new Error(`tv ${args.join(' ')} returned nothing`);
  try {
    return JSON.parse(s);
  } catch {
    // The CLI prints human text on some errors; surface it rather than a
    // JSON.parse stack trace that says nothing about what actually failed.
    throw new Error(`tv ${args.join(' ')} -> non-JSON: ${s.slice(0, 200)}`);
  }
}

/**
 * TradingView bar times are UNIX seconds. Normalise defensively: a value that
 * looks like milliseconds is converted rather than silently producing dates in
 * 1970 or the year 55000, which would poison every downstream session filter.
 */
function toIsoUtc(t) {
  const n = Number(t);
  if (!Number.isFinite(n)) return null;
  const ms = n > 1e12 ? n : n * 1000;
  const d = new Date(ms);
  const y = d.getUTCFullYear();
  if (y < 2000 || y > 2100) return null;
  return d.toISOString();
}

async function pullSymbol(sym) {
  await tv(['symbol', sym]);
  await tv(['timeframe', TF]);
  // The chart needs a moment to load the new series; asking immediately
  // returns the PREVIOUS symbol's bars, which would silently mislabel data.
  await new Promise(r => setTimeout(r, 1200));

  const state = await tv(['state']);
  const got = String(state?.symbol ?? state?.data?.symbol ?? '').toUpperCase();
  if (got && !got.includes(sym)) {
    throw new Error(`chart shows ${got}, expected ${sym} — refusing to save mislabelled bars`);
  }

  const res = await tv(['ohlcv', '-n', String(BARS)]);
  const raw = res?.bars ?? res?.data?.bars;
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('no bars returned');

  const bars = [];
  for (const b of raw) {
    const iso = toIsoUtc(b.time);
    if (iso === null) continue;
    if (![b.open, b.high, b.low, b.close].every(v => Number.isFinite(Number(v)))) continue;
    bars.push({
      t: iso,
      o: Number(b.open), h: Number(b.high),
      l: Number(b.low), c: Number(b.close),
      v: Number(b.volume) || 0,
    });
  }
  if (!bars.length) throw new Error('every bar failed validation');
  bars.sort((a, b) => a.t.localeCompare(b.t));

  return {
    symbol: sym,
    timeframe: TF,
    source: 'tradingview-desktop',
    fetched_at: new Date().toISOString(),
    bar_count: bars.length,
    // The final bar is the one still forming. Flagged, not dropped: the
    // consumer decides, and a bot acting on a partial bar is acting on a
    // number that has not happened yet.
    last_bar_may_be_partial: true,
    bars,
  };
}

function writeIfChanged(sym, payload) {
  mkdirSync(OUT_DIR, { recursive: true });
  const path = join(OUT_DIR, `${sym}_${TF}.json`);
  const next = JSON.stringify(payload, null, 1);
  if (existsSync(path)) {
    try {
      // Compare bars only. fetched_at changes every pass, so including it
      // would make every single run look like a change and push constantly.
      const prev = JSON.parse(readFileSync(path, 'utf8'));
      if (JSON.stringify(prev.bars) === JSON.stringify(payload.bars)) return false;
    } catch { /* unreadable/corrupt -> rewrite it */ }
  }
  writeFileSync(path, next);
  return true;
}

async function git(args) {
  return execFileAsync('git', args, { cwd: REPO, timeout: 120_000 });
}

async function push(changed) {
  if (NO_PUSH || !changed.length) return;
  try {
    await git(['add', 'data/live/tv']);
    const { stdout } = await execFileAsync('git', ['diff', '--cached', '--name-only'], { cwd: REPO });
    if (!stdout.trim()) return;
    await git(['commit', '-m', `tv relay: ${changed.join(',')} @ ${new Date().toISOString().slice(0, 16)}Z`]);
    try { await git(['pull', '--rebase', '-q', 'origin', BRANCH]); } catch { /* offline is survivable */ }
    await git(['push', '-q', 'origin', `HEAD:${BRANCH}`]);
    log(`pushed ${changed.join(',')}`);
  } catch (e) {
    // A failed push must never end the relay. Bars keep updating locally and
    // the next pass retries; losing the day's feed over one network blip is a
    // far worse outcome than a late commit.
    log(`push failed (continuing): ${String(e.message).slice(0, 160)}`);
  }
}

async function pass() {
  const changed = [];
  for (const sym of SYMBOLS) {
    try {
      const payload = await pullSymbol(sym);
      if (writeIfChanged(sym, payload)) changed.push(sym);
      const last = payload.bars.at(-1);
      log(`${sym} ${payload.bar_count} bars, last ${last.t} c=${last.c}`);
    } catch (e) {
      log(`${sym} FAILED: ${String(e.message).slice(0, 200)}`);
    }
  }
  await push(changed);
}

async function main() {
  if (!existsSync(TV_CLI)) {
    console.error(`tradingview-mcp not found at ${TV_HOME}`);
    console.error('Clone it there, or set KRONOS_TV_HOME. See SETUP-TRADINGVIEW.md');
    process.exit(2);
  }
  try {
    const st = await tv(['status']);
    log('CDP status:', JSON.stringify(st).slice(0, 160));
  } catch (e) {
    console.error('Cannot reach TradingView Desktop over CDP (port 9222).');
    console.error('Launch it with --remote-debugging-port=9222 first.');
    console.error(String(e.message).slice(0, 300));
    process.exit(3);
  }

  // Put the user's chart back where they left it. This script drives the real
  // TradingView window; leaving it parked on the last relayed symbol would
  // quietly hijack their workspace.
  let original = null;
  try {
    const st = await tv(['state']);
    original = { symbol: st?.symbol ?? st?.data?.symbol, tf: st?.timeframe ?? st?.data?.timeframe };
  } catch { /* not fatal */ }
  const restore = async () => {
    if (original?.symbol) {
      try {
        await tv(['symbol', String(original.symbol)]);
        if (original.tf) await tv(['timeframe', String(original.tf)]);
        log(`restored chart to ${original.symbol} ${original.tf ?? ''}`);
      } catch { /* best effort */ }
    }
    process.exit(0);
  };
  process.on('SIGINT', restore);
  process.on('SIGTERM', restore);

  log(`relay up · ${SYMBOLS.join(',')} · ${TF}m · every ${EVERY_MS / 1000}s · out ${OUT_DIR}`);
  await pass();
  if (ONCE) return restore();
  setInterval(pass, EVERY_MS);
}

main().catch(e => { console.error(e); process.exit(1); });
