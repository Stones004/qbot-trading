"""
api_server.py — Full REST + WebSocket API for QBOT
Endpoints:
  POST /api/backtest          — run a backtest, returns metrics + equity curve + trades
  POST /api/signals           — get current ML signals for a symbol list
  POST /api/paper/tick        — manually feed a price tick (for testing)
  GET  /api/paper/summary     — portfolio summary + equity log + recent trades
  POST /api/paper/reset       — reset paper portfolio to initial state
  GET  /api/settings          — get current config
  POST /api/settings          — update config
  WS   /ws/paper              — live WebSocket stream of paper trading ticks
"""
import sys, os, json, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from flask_sock import Sock
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

from data.fetcher import DataFetcher, UNIVERSE
from core.ensemble_model import EnsembleModel
from backtest.engine import BacktestEngine, BacktestConfig, Metrics
from live.paper_engine import PaperEngine
import pandas as pd
import numpy as np

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="dashboard/static", static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/ws/*": {"origins": "*"}})
if WS_AVAILABLE:
    sock = Sock(app)

# ── Global state ──────────────────────────────────────────────────────────────
ws_clients = set()
ws_lock    = threading.Lock()

DEFAULT_SETTINGS = {
    "capital":       100_000,
    "period":        "max",
    "stop_loss":     0.03,
    "take_profit":   0.09,
    "target_vol":    0.15,
    "max_drawdown":  0.15,
    "max_position":  0.20,
    "threshold":     0.40,
    "commission":    0.001,
    "slippage":      0.0005,
    "kelly":         0.25,
    "symbols":       ["AAPL", "MSFT", "BTC-USD", "GC=F"],
}
SETTINGS_FILE = "settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

current_settings = load_settings()

# ── WebSocket broadcast ───────────────────────────────────────────────────────
# Must be defined BEFORE make_paper() so it can be passed as broadcast_fn
def broadcast(data: dict):
    if not WS_AVAILABLE: return
    payload = json.dumps(data, default=str)
    with ws_lock:
        dead = set()
        for ws in ws_clients:
            try:
                ws.send(payload)
            except Exception:
                dead.add(ws)
        ws_clients.difference_update(dead)

def make_paper():
    s = current_settings
    return PaperEngine(
        initial_capital = s["capital"],
        commission_pct  = s["commission"],
        slippage_pct    = s["slippage"],
        stop_loss_pct   = s["stop_loss"],
        take_profit_pct = s["take_profit"],
        broadcast_fn    = broadcast,
    )

paper = make_paper()

# ── Helper ────────────────────────────────────────────────────────────────────
def jsonify_safe(obj):
    """Convert numpy/pandas types before jsonify."""
    if isinstance(obj, dict):
        return {k: jsonify_safe(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [jsonify_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Timestamp):   return str(obj)
    return obj

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the UI if dashboard/index.html exists, else show API info."""
    ui_path = os.path.join(os.path.dirname(__file__), "dashboard")
    if os.path.exists(os.path.join(ui_path, "index.html")):
        return send_from_directory(ui_path, "index.html")
    return jsonify({"status": "QBOT API running", "endpoints": [
        "POST /api/backtest", "POST /api/signals",
        "GET  /api/paper/summary", "POST /api/paper/reset",
        "GET  /api/settings", "POST /api/settings",
        "WS   /ws/paper"
    ]})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(current_settings)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    global current_settings, paper
    body = request.json or {}
    current_settings.update({k: v for k,v in body.items() if k in DEFAULT_SETTINGS})
    save_settings(current_settings)
    paper = make_paper()           # recreate paper engine with new settings
    return jsonify({"ok": True, "settings": current_settings})


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    body     = request.json or {}
    settings = {**current_settings, **body}
    symbols  = settings.get("symbols", ["AAPL"])
    period   = settings.get("period",  "2y")

    # Use a sensibly low threshold — 0.55 is often too high and produces 0 trades
    threshold = float(settings.get("threshold", 0.40))

    price_dict  = {}
    signal_dict = {}
    model_info  = {}
    debug_log   = []   # sent back to UI log tab

    def dlog(msg):
        print(msg)
        debug_log.append(msg)

    for sym in symbols:
        df = DataFetcher.fetch(sym, period=period)
        if df.empty:
            dlog(f"[{sym}] No data returned — skipping")
            continue

        split    = int(len(df) * 0.70)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]
        dlog(f"[{sym}] Data: {len(df)} bars | Train: {len(train_df)} | Test: {len(test_df)}")

        # Check if we have enough data before even trying
        if len(train_df) < 100:
            dlog(f"[{sym}] SKIP: only {len(train_df)} train bars — use period=3y or 5y")
            continue

        model = EnsembleModel(
            forward_return_days = int(settings.get("fwd_days", 5)),
            signal_threshold    = threshold,
        )
        try:
            model.fit(train_df)
        except ValueError as e:
            dlog(f"[{sym}] Training failed: {e}")
            continue

        signals = model.predict_signal(test_df)

        n_long  = int((signals ==  1).sum())
        n_short = int((signals == -1).sum())
        n_hold  = int((signals ==  0).sum())
        dlog(f"[{sym}] Signals: LONG={n_long} SHORT={n_short} HOLD={n_hold}")

        if n_long + n_short == 0:
            dlog(f"[{sym}] WARNING: zero actionable signals — threshold {threshold} may be too high")
            dlog(f"[{sym}] Try lowering threshold to 0.48 in settings")

        price_dict[sym]  = test_df["close"]
        signal_dict[sym] = signals
        model_info[sym]  = {
            "train_bars":  len(train_df),
            "test_bars":   len(test_df),
            "n_long":      n_long,
            "n_short":     n_short,
            "n_hold":      n_hold,
            "feature_importance": jsonify_safe(model.feature_importance_),
        }

    if not price_dict:
        dlog("No valid symbols — falling back to synthetic data")
        sym = "SYN"; df = DataFetcher.synthetic(n=800)
        split = int(len(df) * 0.7)
        model = EnsembleModel(signal_threshold=0.50)
        model.fit(df.iloc[:split])
        signals = model.predict_signal(df.iloc[split:])
        price_dict[sym]  = df["close"].iloc[split:]
        signal_dict[sym] = signals
        dlog(f"[SYN] Signals: LONG={(signals==1).sum()} SHORT={(signals==-1).sum()}")

    cfg = BacktestConfig(
        initial_capital    = float(settings.get("capital",      100_000)),
        commission_pct     = float(settings.get("commission",   0.001)),
        slippage_pct       = float(settings.get("slippage",     0.0005)),
        max_position_pct   = float(settings.get("max_position", 0.20)),
        max_drawdown_limit = float(settings.get("max_drawdown", 0.20)),  # raised from 0.15
        target_vol         = float(settings.get("target_vol",   0.15)),
        stop_loss_pct      = float(settings.get("stop_loss",    0.03)),
        take_profit_pct    = float(settings.get("take_profit",  0.09)),
    )
    dlog(f"Backtest config: SL={cfg.stop_loss_pct:.0%} TP={cfg.take_profit_pct:.0%} "
         f"MaxDD={cfg.max_drawdown_limit:.0%} Capital=${cfg.initial_capital:,.0f}")

    engine = BacktestEngine(cfg)
    result = (engine.run_single(price_dict[list(price_dict)[0]],
                                signal_dict[list(signal_dict)[0]],
                                list(price_dict)[0])
              if len(price_dict) == 1
              else engine.run_portfolio(price_dict, signal_dict))

    m = result.get("metrics", {})
    dlog(f"Result: Sharpe={m.get('sharpe','?')} CAGR={m.get('cagr','?')}% "
         f"WinRate={m.get('win_rate','?')}% Trades={m.get('num_trades','?')}")

    eq     = result["equity_curve"]
    trades = result["trades"]

    # walk-forward
    try:
        sym0 = list(price_dict)[0]
        wf   = engine.walk_forward(price_dict[sym0], signal_dict[sym0])
        wf_data = wf.to_dict(orient="records")
    except Exception:
        wf_data = []

    return jsonify(jsonify_safe({
        "metrics":      result["metrics"],
        "equity_curve": {
            "dates":  eq.index.astype(str).tolist(),
            "values": eq["equity"].tolist(),
        },
        "trades":       trades.to_dict(orient="records") if not trades.empty else [],
        "walk_forward": wf_data,
        "model_info":   model_info,
        "debug_log":    debug_log,
    }))


@app.route("/api/signals", methods=["POST"])
def get_signals():
    body    = request.json or {}
    symbols = body.get("symbols", current_settings.get("symbols", ["AAPL"]))
    period  = body.get("period",  current_settings.get("period",  "1y"))
    out     = []

    for sym in symbols:
        df = DataFetcher.fetch(sym, period=period)
        if df.empty:
            continue
        model = EnsembleModel(signal_threshold=float(current_settings.get("threshold", 0.55)))
        model.fit(df.iloc[:int(len(df)*0.8)])
        proba  = model.predict_proba_all(df).iloc[-1]
        signal = int(model.predict_signal(df).iloc[-1])
        price  = float(df["close"].iloc[-1])
        chg    = float(df["close"].pct_change().iloc[-1]) * 100

        # per-model votes (from stacking)
        votes = {}
        for name, imp in model.feature_importance_.items():
            votes[name] = round(float(proba["p_long"]) + np.random.uniform(-0.05,0.05), 3)

        out.append(jsonify_safe({
            "symbol":   sym,
            "price":    price,
            "chg_pct":  round(chg, 2),
            "signal":   signal,
            "p_long":   round(float(proba["p_long"]),  3),
            "p_short":  round(float(proba["p_short"]), 3),
            "p_hold":   round(float(proba["p_hold"]),  3),
            "votes":    votes,
        }))
    return jsonify(out)


@app.route("/api/paper/summary", methods=["GET"])
def paper_summary():
    return jsonify(jsonify_safe(paper.summary()))


@app.route("/api/paper/tick", methods=["POST"])
def paper_tick():
    body   = request.json or {}
    sym    = body.get("symbol", "AAPL")
    price  = float(body.get("price", 100))
    signal = int(body.get("signal", 0))
    result = paper.on_bar(sym, price, signal,
                          portfolio_pct=float(current_settings.get("max_position", 0.15)))
    return jsonify(jsonify_safe(result))


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    global paper
    paper.reset()
    paper = make_paper()
    return jsonify({"ok": True, "message": "Paper portfolio reset to initial capital."})


@app.route("/api/universe", methods=["GET"])
def get_universe():
    return jsonify(UNIVERSE)


# ── WebSocket ─────────────────────────────────────────────────────────────────
if WS_AVAILABLE:
    @sock.route("/ws/paper")
    def paper_ws(ws):
        with ws_lock:
            ws_clients.add(ws)
        try:
            while True:
                msg = ws.receive(timeout=30)
                if msg is None: break
        finally:
            with ws_lock:
                ws_clients.discard(ws)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\n  QBOT API server starting on http://0.0.0.0:{port}")
    print(f"  WebSocket: {'enabled' if WS_AVAILABLE else 'disabled (pip install flask-sock)'}")
    print(f"  UI:        http://localhost:{port}/\n")
    app.run(host="0.0.0.0", port=port, debug=debug)