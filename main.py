"""
main.py — QBOT CLI entry point
Usage:
  python main.py --mode backtest  --symbols AAPL MSFT BTC-USD --period 2y
  python main.py --mode paper     --symbols AAPL BTC-USD --poll 60
  python main.py --mode optimize  --symbols AAPL --period 5y
  python main.py --mode server    (starts the web UI + API)
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))


def run_backtest(symbols, period):
    from data.fetcher import DataFetcher
    from core.ensemble_model import EnsembleModel
    from backtest.engine import BacktestEngine, BacktestConfig
    import pandas as pd

    print(f"\n{'='*55}\n  QBOT — Backtest Mode\n{'='*55}")
    price_dict, signal_dict = {}, {}

    for sym in symbols:
        print(f"\n[{sym}] Fetching {period} of data...")
        df = DataFetcher.fetch(sym, period=period)
        if df.empty:
            print(f"  Skipping {sym} — no data"); continue
        split = int(len(df) * 0.70)
        print(f"[{sym}] Training ensemble on {split} bars...")
        model = EnsembleModel()
        model.fit(df.iloc[:split])
        signals = model.predict_signal(df.iloc[split:])
        price_dict[sym]  = df['close'].iloc[split:]
        signal_dict[sym] = signals

    if not price_dict:
        print("No data — using synthetic"); sym = 'SYN'
        from data.fetcher import DataFetcher
        df = DataFetcher.synthetic(n=600); split = int(len(df)*0.7)
        model = EnsembleModel(); model.fit(df.iloc[:split])
        price_dict[sym] = df['close'].iloc[split:]
        signal_dict[sym] = model.predict_signal(df.iloc[split:])

    cfg    = BacktestConfig()
    engine = BacktestEngine(cfg)
    result = (engine.run_single(price_dict[list(price_dict)[0]],
                                signal_dict[list(signal_dict)[0]],
                                list(price_dict)[0])
              if len(price_dict)==1
              else engine.run_portfolio(price_dict, signal_dict))

    m = result['metrics']
    print(f"\n{'='*55}\n  RESULTS\n{'='*55}")
    for k, v in m.items():
        print(f"  {k:<20} {v}")
    result['equity_curve'].to_csv('equity_curve.csv')
    if not result['trades'].empty:
        result['trades'].to_csv('trades.csv', index=False)
    with open('metrics.json','w') as f:
        json.dump(m, f, indent=2, default=str)
    print(f"\n  Saved: equity_curve.csv  trades.csv  metrics.json\n")


def run_optimize(symbol, period):
    from data.fetcher import DataFetcher
    from core.ensemble_model import EnsembleModel
    from backtest.engine import BacktestEngine

    print(f"\n{'='*55}\n  QBOT — Walk-Forward Optimization\n{'='*55}")
    df = DataFetcher.fetch(symbol, period=period)
    model = EnsembleModel(); model.fit(df.iloc[:int(len(df)*0.5)])
    signals = model.predict_signal(df)
    wf = BacktestEngine().walk_forward(df['close'], signals)
    print(wf.to_string(index=False))
    print(f"\n  Mean Sharpe: {wf['sharpe'].mean():.3f}  Std: {wf['sharpe'].std():.3f}")
    wf.to_csv('walk_forward.csv', index=False)


def run_paper(symbols, poll_interval):
    import time
    from data.fetcher import DataFetcher
    from core.ensemble_model import EnsembleModel
    from live.paper_engine import PaperEngine

    print(f"\n{'='*55}\n  QBOT — Paper Trading\n{'='*55}")
    paper = PaperEngine()
    models = {}
    for sym in symbols:
        df = DataFetcher.fetch(sym, period='1y')
        if not df.empty:
            m = EnsembleModel(); m.fit(df); models[sym] = (m, df)

    try:
        iteration = 0
        while True:
            iteration += 1
            from datetime import datetime
            print(f"\n[{datetime.now():%H:%M:%S}] Iteration #{iteration}")
            for sym in symbols:
                if sym not in models: continue
                model, _ = models[sym]
                fresh = DataFetcher.fetch(sym, period='1y', force_refresh=(iteration%10==0))
                if fresh.empty: continue
                price  = float(fresh['close'].iloc[-1])
                signal = int(model.predict_signal(fresh).iloc[-1])
                label  = {1:'LONG', -1:'SHORT', 0:'HOLD'}[signal]
                result = paper.on_bar(sym, price, signal)
                print(f"  {sym:<12} ${price:>10.2f}  {label:<6}  {result.get('action') or 'no action'}")
                models[sym] = (model, fresh)
            s = paper.summary()
            print(f"\n  Portfolio ${s['portfolio_value']:>12,.2f}  "
                  f"Return {s['total_return_pct']:+.2f}%  Trades {s['num_trades']}")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n\n  Stopped. Final summary:")
        import json; print(json.dumps(paper.summary(), indent=2, default=str))


def run_server():
    import api_server
    api_server.app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='QBOT Trading Bot')
    p.add_argument('--mode', choices=['backtest','paper','optimize','server'], default='server')
    p.add_argument('--symbols', nargs='+', default=['AAPL','MSFT','BTC-USD'])
    p.add_argument('--period',  default='2y')
    p.add_argument('--poll',    type=int, default=60)
    args = p.parse_args()

    if   args.mode == 'backtest': run_backtest(args.symbols, args.period)
    elif args.mode == 'optimize': run_optimize(args.symbols[0], args.period)
    elif args.mode == 'paper':    run_paper(args.symbols, args.poll)
    elif args.mode == 'server':   run_server()
