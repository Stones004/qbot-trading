# QBOT — Ensemble ML Trading Bot

Full-stack trading bot with XGBoost + LightGBM + meta-learner ensemble, real-time paper trading, walk-forward backtesting, and a terminal-style web UI.

---

## Quick start (local)

```bash
# 1. Clone / download this folder
cd qbot

# 2. Run the setup script (creates venv, installs deps, runs smoke test)
bash setup.sh

# 3. Start the server
source .venv/bin/activate
python api_server.py

# 4. Open the UI
open http://localhost:5000
```

---

## Deploy to Render (free tier)

### Step 1 — Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/qbot-trading.git
git branch -M main
git push -u origin main
```

### Step 2 — Create Render service

1. Go to **https://render.com** → sign up / log in
2. Click **New +** → **Web Service**
3. Click **Connect account** → authorise GitHub
4. Select your **qbot-trading** repo
5. Render automatically reads `render.yaml` — just click **Create Web Service**
6. First deploy takes ~3 minutes (installing XGBoost + LightGBM)
7. Your live URL: `https://qbot-trading.onrender.com`

### Step 3 — Open the UI

Visit `https://qbot-trading.onrender.com` — the full terminal UI is served directly.

> **Free tier note:** Render free services spin down after 15 min of inactivity.
> First request after spin-down takes ~30 seconds. Upgrade to Starter ($7/mo) for always-on.

---

## Project structure

```
qbot/
├── api_server.py          ← Flask REST + WebSocket API (entry point)
├── main.py                ← CLI entry point
├── requirements.txt
├── Procfile               ← for Railway / Heroku
├── render.yaml            ← Render deploy config
├── Dockerfile             ← Docker deploy
├── Makefile               ← dev shortcuts
├── setup.sh               ← one-command setup
│
├── core/
│   ├── ensemble_model.py  ← XGB + LGB + meta-learner
│   ├── feature_engineer.py ← 20 technical features
│   └── risk_manager.py    ← risk checks + position sizing
│
├── data/
│   ├── fetcher.py         ← yfinance + parquet cache + GBM fallback
│   └── cache/             ← auto-populated parquet files
│
├── backtest/
│   └── engine.py          ← backtest + metrics + walk-forward
│
├── live/
│   └── paper_engine.py    ← paper trading with state persistence
│
└── dashboard/
    └── index.html         ← full trading terminal UI
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serves the trading UI |
| `GET`  | `/api/settings` | Get current config |
| `POST` | `/api/settings` | Update config (persisted to settings.json) |
| `POST` | `/api/backtest` | Run full backtest, returns metrics + equity curve + trades |
| `POST` | `/api/signals`  | Get current ML signals for a symbol list |
| `GET`  | `/api/paper/summary` | Portfolio state + equity log + open positions |
| `POST` | `/api/paper/tick` | Feed a manual price tick |
| `POST` | `/api/paper/reset` | Reset paper portfolio |
| `WS`   | `/ws/paper` | Live WebSocket stream of paper trading ticks |

---

## CLI usage

```bash
# Run backtest (prints metrics, saves CSV)
python main.py --mode backtest --symbols AAPL MSFT BTC-USD --period 2y

# Walk-forward Sharpe optimisation
python main.py --mode optimize --symbols AAPL --period 5y

# Paper trading loop (polls every 60s)
python main.py --mode paper --symbols AAPL BTC-USD --poll 60

# Start web server
python main.py --mode server
```

---

## Makefile shortcuts

```bash
make install       # install deps
make run           # start server
make dev           # start with DEBUG=true
make backtest      # CLI backtest
make paper         # CLI paper trading
make test          # run tests
make clean         # remove __pycache__
make reset         # wipe paper state + cache
make docker-build  # build Docker image
make docker-run    # run in Docker
```
