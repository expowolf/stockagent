// frontend/components/TradingViewChart.tsx
'use client';

import React, { useEffect, useRef, useState } from 'react';

interface TradingViewChartProps {
  ticker: string;
  height?: number;
  width?: string;
  signals?: {
    ou_score: number;
    hmm_regime: string;
    evt_var: number;
  };
}

export const TradingViewChart: React.FC<TradingViewChartProps> = ({
  ticker,
  height = 500,
  width = '100%',
  signals,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scriptLoaded, setScriptLoaded] = useState(false);

  // Load the TradingView tv.js script exactly once.
  useEffect(() => {
    if (window.TradingView) {
      setScriptLoaded(true);
      return;
    }
    const existing = document.getElementById('tradingview-tv-js');
    if (existing) {
      existing.addEventListener('load', () => setScriptLoaded(true));
      return;
    }
    const script = document.createElement('script');
    script.id = 'tradingview-tv-js';
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => setScriptLoaded(true);
    document.body.appendChild(script);
  }, []);

  // (Re)build the widget when the script is ready or the ticker changes.
  useEffect(() => {
    const container = containerRef.current;
    if (!scriptLoaded || !container || !window.TradingView) {
      return;
    }

    container.innerHTML = '';

    const widgetConfig = {
      autosize: true,
      symbol: `NASDAQ:${ticker}`,
      interval: 'D',
      timezone: 'Etc/UTC',
      theme: 'dark',
      style: '1',
      locale: 'en',
      toolbar_bg: '#09090b',
      enable_publishing: false,
      allow_symbol_change: true,
      studies: [
        'Volume@tv-basicstudies',
        'RSI@tv-basicstudies',
        'MACD@tv-basicstudies',
        'BB@tv-basicstudies',
      ],
      container_id: container.id,
      hide_side_toolbar: false,
      withdateranges: true,
    };

    try {
      // eslint-disable-next-line no-new
      new window.TradingView.widget(widgetConfig);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('TradingView widget initialization failed:', error);
    }

    return () => {
      if (container) {
        container.innerHTML = '';
      }
    };
  }, [scriptLoaded, ticker]);

  return (
    <div className="w-full bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800">
      {/* Kronos signal overlay */}
      {signals && (
        <div className="grid grid-cols-4 gap-2 p-3 bg-zinc-900 border-b border-zinc-800 text-xs font-mono">
          <div className="text-center">
            <div className="text-gray-400">OU Score</div>
            <div
              className={`text-lg font-bold ${
                signals.ou_score > 70 ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {signals.ou_score.toFixed(1)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-400">Regime</div>
            <div className="text-lg font-bold text-blue-400">{signals.hmm_regime}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-400">VaR (99%)</div>
            <div className="text-lg font-bold text-orange-400">
              {(signals.evt_var * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-400">Status</div>
            <div className="text-lg font-bold text-cyan-400">LIVE</div>
          </div>
        </div>
      )}

      <div
        id={`tradingview-chart-${ticker}`}
        ref={containerRef}
        style={{ height: `${height}px`, width }}
      />
    </div>
  );
};

// Declare TradingView global
declare global {
  interface Window {
    TradingView: any;
  }
}

export default TradingViewChart;
