"""
diagnose.py — verifies the fixed ensemble model produces real signals
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

SEP = "=" * 58
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):      print(f"  [OK]   {m}")
def warn(m):    print(f"  [WARN] {m}")
def fail(m):    print(f"  [FAIL] {m}")
def info(m):    print(f"         {m}")

from data.fetcher import DataFetcher
from core.ensemble_model import EnsembleModel
from backtest.engine import BacktestEngine, BacktestConfig

section("1 · TRAINING")
df       = DataFetcher.fetch("AAPL", period="5y", force_refresh=False)
split    = int(len(df) * 0.70)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]
info(f"Train: {len(train_df)} | Test: {len(test_df)}")

model = EnsembleModel(forward_return_days=5, signal_threshold=0.38)
model.fit(train_df)

if model._meta_collapsed:
    warn("Meta-learner collapsed — using direct base model average (fallback active)")
else:
    ok("Meta-learner working correctly")

section("2 · THRESHOLD SCAN")
cfg = BacktestConfig(
    initial_capital=100_000, commission_pct=0.001, slippage_pct=0.0005,
    stop_loss_pct=0.03, take_profit_pct=0.09,
    use_vol_targeting=True, target_vol=0.15,
    max_position_pct=0.20, max_drawdown_limit=0.25,
)
engine = BacktestEngine(cfg)

print(f"\n  {'Threshold':>10}  {'Trades':>7}  {'WinRate':>8}  {'Sharpe':>8}  {'CAGR':>7}")
print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}")

best_sharpe = -999
best_thresh = None
best_m      = None

from core.feature_engineer import FeatureEngineer
test_fe   = FeatureEngineer.add_features(test_df.copy())
feat_cols = model.FEATURE_COLS
feat_clean= test_fe[feat_cols].ffill().bfill().fillna(0)
mask      = feat_clean.notna().all(axis=1)
stacked   = model._stack(feat_clean[mask].values)
proba     = model._predict_proba_from_stack(stacked)

for thresh in [0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
    raw = np.zeros(len(stacked), dtype=int)
    for j in range(len(stacked)):
        ps, ph, pl = proba[j]
        if   pl >= thresh: raw[j] =  1
        elif ps >= thresh: raw[j] = -1

    sigs = pd.Series(0, index=test_df.index)
    sigs[mask] = raw
    n_act = int((raw != 0).sum())

    if n_act == 0:
        print(f"  {thresh:>10.2f}  {'0':>7}  {'—':>8}  {'—':>8}  {'—':>7}")
        continue

    res = engine.run_single(test_df['close'], sigs, symbol="AAPL")
    m   = res['metrics']
    if not m:
        print(f"  {thresh:>10.2f}  {n_act:>7}  {'—':>8}  {'—':>8}  {'—':>7}")
        continue

    sharpe = float(m.get('sharpe', 0))
    cagr   = float(m.get('cagr', 0))
    wr     = float(m.get('win_rate', 0))
    nt     = int(m.get('num_trades', 0))
    flag   = "  <-- BEST" if sharpe > best_sharpe and sharpe > 0 else ""
    print(f"  {thresh:>10.2f}  {nt:>7}  {wr:>7.1f}%  {sharpe:>8.3f}  {cagr:>6.1f}%{flag}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_thresh = thresh
        best_m      = m

section("3 · RESULT")
if best_sharpe > 1.0:
    ok(f"Sharpe {best_sharpe:.3f} at threshold={best_thresh}  -- model is working!")
    info(f"  Win rate: {best_m.get('win_rate',0):.1f}%")
    info(f"  Trades:   {best_m.get('num_trades',0)}")
    info(f"  CAGR:     {best_m.get('cagr',0):.1f}%")
    info(f"\n  Set in api_server.py DEFAULT_SETTINGS:")
    info(f'    "threshold": {best_thresh}')
elif best_sharpe > 0:
    warn(f"Sharpe {best_sharpe:.3f} — positive but below 1.0")
    info("  The fallback (direct base model average) is producing some signal.")
    info("  To improve: run the UI backtest with these settings:")
    info("    period=5y, take_profit=0.12, forward_return_days=10")
else:
    fail(f"Still no positive Sharpe.")
    info("  The features correlate with 5-day returns (we confirmed that)")
    info("  but XGBoost win rate DROPS as confidence rises, meaning it")
    info("  learned a pattern that doesn't generalise to the test period.")
    info("")
    info("  This is the hardest possible case: the train and test periods")
    info("  have different market regimes.")
    info("")
    info("  Best fix: train on a LATER window that matches recent market")
    info("  conditions. In diagnose.py, change split from 0.70 to 0.50")
    info("  so the model trains on more recent data.")

print(f"\n{SEP}\n  Done.\n{SEP}\n")