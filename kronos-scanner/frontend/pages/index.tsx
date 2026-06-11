// frontend/pages/index.tsx
'use client';

import React, { useEffect, useState } from 'react';
import { TradingViewChart } from '../components/TradingViewChart';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

interface Signal {
  ticker: string;
  date: string;
  current_price: number;
  ou_score: number;
  ou_halflife: number | null;
  hmm_regime: string;
  hmm_confidence: number;
  kalman_alpha: number;
  evt_var_99: number | null;
  evt_expected_shortfall: number | null;
}

export default function Home() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [selected, setSelected] = useState<string>('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runScan = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/scan?limit=24`);
      if (!res.ok) throw new Error(`Scan failed: ${res.status}`);
      const data: Signal[] = await res.json();
      setSignals(data);
      if (data.length > 0) setSelected(data[0].ticker);
    } catch (e: any) {
      setError(e.message ?? 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runScan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedSignal = signals.find((s) => s.ticker === selected);

  return (
    <div className="min-h-screen bg-black text-zinc-100 p-6">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold tracking-tight">
          KRONOS <span className="text-cyan-400">AI Scanner</span>
        </h1>
        <button
          onClick={runScan}
          disabled={loading}
          className="px-4 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-sm font-semibold"
        >
          {loading ? 'Scanning…' : 'Run Scan'}
        </button>
      </header>

      {error && (
        <div className="mb-4 p-3 rounded-md bg-red-950 border border-red-800 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Signal table */}
        <div className="lg:col-span-1 bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800 text-sm font-semibold text-zinc-400">
            Ranked Signals
          </div>
          <div className="max-h-[640px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-zinc-500 sticky top-0 bg-zinc-950">
                <tr>
                  <th className="text-left px-4 py-2">Ticker</th>
                  <th className="text-right px-2 py-2">OU</th>
                  <th className="text-left px-2 py-2">Regime</th>
                  <th className="text-right px-4 py-2">VaR</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s) => (
                  <tr
                    key={s.ticker}
                    onClick={() => setSelected(s.ticker)}
                    className={`cursor-pointer border-t border-zinc-900 hover:bg-zinc-900 ${
                      selected === s.ticker ? 'bg-zinc-900' : ''
                    }`}
                  >
                    <td className="px-4 py-2 font-mono font-semibold">{s.ticker}</td>
                    <td
                      className={`px-2 py-2 text-right font-mono ${
                        s.ou_score > 70 ? 'text-green-400' : 'text-zinc-300'
                      }`}
                    >
                      {s.ou_score.toFixed(1)}
                    </td>
                    <td className="px-2 py-2 text-blue-400">{s.hmm_regime}</td>
                    <td className="px-4 py-2 text-right font-mono text-orange-400">
                      {s.evt_var_99 != null ? `${(s.evt_var_99 * 100).toFixed(2)}%` : '—'}
                    </td>
                  </tr>
                ))}
                {signals.length === 0 && !loading && (
                  <tr>
                    <td colSpan={4} className="px-4 py-6 text-center text-zinc-600">
                      No signals yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Chart */}
        <div className="lg:col-span-2">
          <TradingViewChart
            ticker={selected}
            height={620}
            signals={
              selectedSignal
                ? {
                    ou_score: selectedSignal.ou_score,
                    hmm_regime: selectedSignal.hmm_regime,
                    evt_var: selectedSignal.evt_var_99 ?? 0,
                  }
                : undefined
            }
          />
        </div>
      </div>
    </div>
  );
}
