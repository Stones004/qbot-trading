"""
diagnose.py — Run this first to find exactly why Sharpe is 0 or negative.

Usage:
    python diagnose.py

It will print a full report of every stage: data, features, labels,
model training, signal distribution, and backtest results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEP  = "=" * 58
SEP2 = "-" * 58

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")

# ── 1. DATA ───────────────────────────────────────────────────────────────────
section("1 · DATA FETCH")
from data.fetcher import DataFetcher

SYMBOL = "AAPL"
PERIOD = "2y"

df = DataFetcher.fetch(SYMBOL, period=PERIOD, force_refresh=True)

if df.empty:
    fail(f"No data returned for {SYMBOL}. Check internet / yfinance.")
    sys.exit(1)

ok(f"Fetched {len(df)} rows for {SYMBOL} ({PERIOD})")
info(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
info(f"Close range: ${df['close'].min():.2f} → ${df['close'].max():.2f}")
info(f"NaN rows   : {df.isna().any(axis=1).sum()}")

if len(df) < 300:
    warn(f"Only {len(df)} rows — try period='3y' or '5y' for better results")
else:
    ok(f"Sufficient rows for training")

# ── 2. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
section("2 · TRAIN / TEST SPLIT")
split     = int(len(df) * 0.70)
train_df  = df.iloc[:split]
test_df   = df.iloc[split:]

ok(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

if len(test_df) < 63:
    warn("Test set < 63 bars — results will be noisy")
if len(train_df) < 150:
    fail("Train set < 150 bars — model will not learn anything meaningful")
    warn("Fix: use period='3y' or '5y'")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
section("3 · FEATURE ENGINEERING")
from core.feature_engineer import FeatureEngineer

train_feat = FeatureEngineer.add_features(train_df.copy())
feat_cols  = FeatureEngineer.FEATURE_COLS

nan_counts = train_feat[feat_cols].isna().sum()
nan_heavy  = nan_counts[nan_counts > len(train_feat) * 0.20]

ok(f"Generated {len(feat_cols)} features")
if len(nan_heavy) > 0:
    warn(f"Features with >20% NaN (will be filled):")
    for col, cnt in nan_heavy.items():
        info(f"  {col}: {cnt} NaN ({cnt/len(train_feat)*100:.0f}%)")
else:
    ok("No heavily NaN features")

# After fill
filled = train_feat[feat_cols].ffill().bfill().fillna(0)
remaining_nan = filled.isna().sum().sum()
ok(f"After ffill/bfill/zero-fill: {remaining_nan} NaN remaining")

# ── 4. LABEL DISTRIBUTION ─────────────────────────────────────────────────────
section("4 · LABEL DISTRIBUTION (forward returns)")

fwd_days  = 5
fwd_ret   = train_df['close'].pct_change(fwd_days).shift(-fwd_days).dropna()
threshold = 0.002

n_long  = (fwd_ret >  threshold).sum()
n_short = (fwd_ret < -threshold).sum()
n_hold  = ((fwd_ret >= -threshold) & (fwd_ret <= threshold)).sum()
total   = len(fwd_ret)

info(f"Forward return window : {fwd_days} days")
info(f"Signal threshold      : ±{threshold*100:.1f}%")
print()
info(f"  LONG  : {n_long:4d} ({n_long/total*100:.1f}%)")
info(f"  HOLD  : {n_hold:4d} ({n_hold/total*100:.1f}%)")
info(f"  SHORT : {n_short:4d} ({n_short/total*100:.1f}%)")

if n_hold / total > 0.70:
    warn("Over 70% HOLD labels — threshold too tight, most signals are HOLD")
    warn("Fix: lower signal_threshold in UI from 0.55 → 0.50")
elif n_long / total < 0.10 or n_short / total < 0.10:
    warn("Very few LONG or SHORT labels — imbalanced training data")
else:
    ok("Label distribution looks healthy")

# ── 5. MODEL TRAINING ─────────────────────────────────────────────────────────
section("5 · MODEL TRAINING")
from core.ensemble_model import EnsembleModel

print("  Training ensemble (this takes 30-90 seconds)...")
model = EnsembleModel(forward_return_days=fwd_days, signal_threshold=0.55)

try:
    model.fit(train_df)
    ok("Ensemble trained successfully")
except Exception as e:
    fail(f"Training failed: {e}")
    sys.exit(1)

# Feature importance
if model.feature_importance_:
    print()
    info("Top 5 features by importance:")
    for model_name, imp in model.feature_importance_.items():
        top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
        info(f"  [{model_name}] " + ", ".join(f"{k}={v:.3f}" for k,v in top5))

# ── 6. SIGNAL GENERATION ──────────────────────────────────────────────────────
section("6 · SIGNAL GENERATION ON TEST SET")

signals   = model.predict_signal(test_df)
probas    = model.predict_proba_all(test_df)

n_long_s  = (signals == 1).sum()
n_short_s = (signals == -1).sum()
n_hold_s  = (signals == 0).sum()
total_s   = len(signals)

info(f"  LONG  signals : {n_long_s:4d} ({n_long_s/total_s*100:.1f}%)")
info(f"  HOLD  signals : {n_hold_s:4d} ({n_hold_s/total_s*100:.1f}%)")
info(f"  SHORT signals : {n_short_s:4d} ({n_short_s/total_s*100:.1f}%)")

if n_long_s + n_short_s == 0:
    fail("ZERO actionable signals generated!")
    fail("The model is predicting HOLD for every bar.")
    warn("Root cause: confidence threshold (0.55) too high for this dataset.")
    warn("Fix: lower threshold to 0.50 or 0.45 in the UI slider.")
elif n_long_s + n_short_s < 5:
    warn(f"Only {n_long_s + n_short_s} trades — very few signals. Lower threshold.")
else:
    ok(f"{n_long_s + n_short_s} actionable signals generated")

avg_long_conf  = probas.loc[signals==1,  'p_long'].mean()  if n_long_s > 0 else 0
avg_short_conf = probas.loc[signals==-1, 'p_short'].mean() if n_short_s > 0 else 0
info(f"  Avg LONG  confidence  : {avg_long_conf:.3f}")
info(f"  Avg SHORT confidence  : {avg_short_conf:.3f}")

if avg_long_conf < 0.50 and avg_short_conf < 0.50:
    warn("Low confidence across all signals — model is uncertain")
    warn("Fix: more training data (use 5y period)")

# ── 7. BACKTEST ───────────────────────────────────────────────────────────────
section("7 · BACKTEST")
from backtest.engine import BacktestEngine, BacktestConfig

cfg = BacktestConfig(
    initial_capital   = 100_000,
    commission_pct    = 0.001,
    slippage_pct      = 0.0005,
    stop_loss_pct     = 0.03,
    take_profit_pct   = 0.09,
    use_vol_targeting = True,
    target_vol        = 0.15,
    max_position_pct  = 0.20,
    max_drawdown_limit= 0.20,
)
engine = BacktestEngine(cfg)
result = engine.run_single(test_df['close'], signals, symbol=SYMBOL)
m      = result['metrics']
trades = result['trades']

if not m:
    fail("No metrics returned — backtest produced no trades at all")
    warn("This means signals are all HOLD or the drawdown limit was hit immediately")
else:
    sharpe = m.get('sharpe', 0)
    cagr   = m.get('cagr',   0)
    mdd    = m.get('max_drawdown', 0)
    wr     = m.get('win_rate', 0)
    ntrades= m.get('num_trades', 0)

    print()
    info(f"  Sharpe ratio   : {sharpe:.3f}  {'<-- GOOD' if sharpe > 1.0 else '<-- NEEDS IMPROVEMENT'}")
    info(f"  CAGR           : {cagr:.1f}%")
    info(f"  Max drawdown   : {mdd:.1f}%")
    info(f"  Win rate       : {wr:.1f}%")
    info(f"  Num trades     : {ntrades}")
    print()

    if sharpe <= 0:
        fail("Sharpe <= 0 — strategy is destroying value")
        print()
        warn("DIAGNOSIS:")
        if ntrades == 0:
            warn("  No trades executed. Signals exist but risk manager blocked all of them.")
            warn("  Fix: check max_drawdown_limit — try 0.25 (25%) instead of 0.15")
        elif wr < 35:
            warn(f"  Win rate {wr:.1f}% is too low. Model is guessing wrong direction.")
            warn("  Fix 1: Use more data (period='5y')")
            warn("  Fix 2: Lower threshold to 0.50 so model only trades highest-conviction signals")
            warn("  Fix 3: Widen take-profit to 12% (4:1 R:R instead of 3:1)")
        elif ntrades > 0 and cagr < 0:
            warn("  Model wins some trades but losses exceed wins in dollar terms.")
            warn("  Fix: increase take_profit to 0.12, keep stop_loss at 0.03 (4:1 ratio)")
    elif sharpe < 1.0:
        warn(f"  Sharpe {sharpe:.2f} — positive but below target. See recommendations below.")
    else:
        ok(f"  Sharpe {sharpe:.2f} — good result!")

# ── 8. QUICK-FIX RECOMMENDATIONS ─────────────────────────────────────────────
section("8 · RECOMMENDED FIXES (in priority order)")

fixes = []

if len(df) < 500:
    fixes.append(("HIGH",   "Use period='3y' or '5y'",
                  f"Only {len(df)} bars. More data = better trees."))

if n_long_s + n_short_s < 10:
    fixes.append(("HIGH",   "Lower signal threshold to 0.50",
                  "Too few signals generated. Model needs lower bar to act."))

if not m or m.get('num_trades', 0) == 0:
    fixes.append(("HIGH",   "Raise max_drawdown_limit to 0.25",
                  "Risk manager may be halting the backtest too early."))

if m and m.get('win_rate', 50) < 40:
    fixes.append(("MED",    "Try forward_return_days=10 instead of 5",
                  "Longer prediction window is easier to get right."))

if m and m.get('profit_factor', 1) < 1.0:
    fixes.append(("MED",    "Set take_profit=0.12 (4:1 R:R ratio)",
                  "Wins not big enough to cover losses. Widen TP."))

fixes.append(("LOW",    "Add BTC-USD and GC=F (gold) to universe",
              "Diversification reduces portfolio vol, boosting Sharpe."))

fixes.append(("LOW",    "Run with period='5y' for walk-forward",
              "Need 5y to get 6+ quarterly windows for reliable WF analysis."))

if not fixes:
    ok("No critical fixes needed — model is performing well")
else:
    for priority, fix, reason in fixes:
        tag = {"HIGH": "[!!!]", "MED": "[ ! ]", "LOW": "[ i ]"}[priority]
        print(f"  {tag} {fix}")
        info(f"      Why: {reason}")

print(f"\n{SEP}\n  Diagnostics complete.\n{SEP}\n")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEP  = "=" * 58
SEP2 = "-" * 58

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")

# ── 1. DATA ───────────────────────────────────────────────────────────────────
section("1 · DATA FETCH")
from data.fetcher import DataFetcher

SYMBOL = "AAPL"
PERIOD = "2y"

df = DataFetcher.fetch(SYMBOL, period=PERIOD, force_refresh=True)

if df.empty:
    fail(f"No data returned for {SYMBOL}. Check internet / yfinance.")
    sys.exit(1)

ok(f"Fetched {len(df)} rows for {SYMBOL} ({PERIOD})")
info(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
info(f"Close range: ${df['close'].min():.2f} → ${df['close'].max():.2f}")
info(f"NaN rows   : {df.isna().any(axis=1).sum()}")

if len(df) < 300:
    warn(f"Only {len(df)} rows — try period='3y' or '5y' for better results")
else:
    ok(f"Sufficient rows for training")

# ── 2. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
section("2 · TRAIN / TEST SPLIT")
split     = int(len(df) * 0.70)
train_df  = df.iloc[:split]
test_df   = df.iloc[split:]

ok(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

if len(test_df) < 63:
    warn("Test set < 63 bars — results will be noisy")
if len(train_df) < 150:
    fail("Train set < 150 bars — model will not learn anything meaningful")
    warn("Fix: use period='3y' or '5y'")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
section("3 · FEATURE ENGINEERING")
from core.feature_engineer import FeatureEngineer

train_feat = FeatureEngineer.add_features(train_df.copy())
feat_cols  = FeatureEngineer.FEATURE_COLS

nan_counts = train_feat[feat_cols].isna().sum()
nan_heavy  = nan_counts[nan_counts > len(train_feat) * 0.20]

ok(f"Generated {len(feat_cols)} features")
if len(nan_heavy) > 0:
    warn(f"Features with >20% NaN (will be filled):")
    for col, cnt in nan_heavy.items():
        info(f"  {col}: {cnt} NaN ({cnt/len(train_feat)*100:.0f}%)")
else:
    ok("No heavily NaN features")

# After fill
filled = train_feat[feat_cols].ffill().bfill().fillna(0)
remaining_nan = filled.isna().sum().sum()
ok(f"After ffill/bfill/zero-fill: {remaining_nan} NaN remaining")

# ── 4. LABEL DISTRIBUTION ─────────────────────────────────────────────────────
section("4 · LABEL DISTRIBUTION (forward returns)")

fwd_days  = 5
fwd_ret   = train_df['close'].pct_change(fwd_days).shift(-fwd_days).dropna()
threshold = 0.003

n_long  = (fwd_ret >  threshold).sum()
n_short = (fwd_ret < -threshold).sum()
n_hold  = ((fwd_ret >= -threshold) & (fwd_ret <= threshold)).sum()
total   = len(fwd_ret)

info(f"Forward return window : {fwd_days} days")
info(f"Signal threshold      : ±{threshold*100:.1f}%")
print()
info(f"  LONG  : {n_long:4d} ({n_long/total*100:.1f}%)")
info(f"  HOLD  : {n_hold:4d} ({n_hold/total*100:.1f}%)")
info(f"  SHORT : {n_short:4d} ({n_short/total*100:.1f}%)")

if n_hold / total > 0.70:
    warn("Over 70% HOLD labels — threshold too tight, most signals are HOLD")
    warn("Fix: lower signal_threshold in UI from 0.55 → 0.50")
elif n_long / total < 0.10 or n_short / total < 0.10:
    warn("Very few LONG or SHORT labels — imbalanced training data")
else:
    ok("Label distribution looks healthy")

# ── 5. MODEL TRAINING ─────────────────────────────────────────────────────────
section("5 · MODEL TRAINING")
from core.ensemble_model import EnsembleModel

print("  Training ensemble (this takes 30-90 seconds)...")
model = EnsembleModel(forward_return_days=fwd_days, signal_threshold=0.40)

try:
    model.fit(train_df)
    ok("Ensemble trained successfully")
except Exception as e:
    fail(f"Training failed: {e}")
    sys.exit(1)

# Feature importance
if model.feature_importance_:
    print()
    info("Top 5 features by importance:")
    for model_name, imp in model.feature_importance_.items():
        top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
        info(f"  [{model_name}] " + ", ".join(f"{k}={v:.3f}" for k,v in top5))

# ── 6. SIGNAL GENERATION ──────────────────────────────────────────────────────
section("6 · SIGNAL GENERATION ON TEST SET")

signals   = model.predict_signal(test_df)
probas    = model.predict_proba_all(test_df)

n_long_s  = (signals == 1).sum()
n_short_s = (signals == -1).sum()
n_hold_s  = (signals == 0).sum()
total_s   = len(signals)

info(f"  LONG  signals : {n_long_s:4d} ({n_long_s/total_s*100:.1f}%)")
info(f"  HOLD  signals : {n_hold_s:4d} ({n_hold_s/total_s*100:.1f}%)")
info(f"  SHORT signals : {n_short_s:4d} ({n_short_s/total_s*100:.1f}%)")

if n_long_s + n_short_s == 0:
    fail("ZERO actionable signals generated!")
    fail("The model is predicting HOLD for every bar.")
    warn("Root cause: confidence threshold (0.55) too high for this dataset.")
    warn("Fix: lower threshold to 0.50 or 0.45 in the UI slider.")
elif n_long_s + n_short_s < 5:
    warn(f"Only {n_long_s + n_short_s} trades — very few signals. Lower threshold.")
else:
    ok(f"{n_long_s + n_short_s} actionable signals generated")

avg_long_conf  = probas.loc[signals==1,  'p_long'].mean()  if n_long_s > 0 else 0
avg_short_conf = probas.loc[signals==-1, 'p_short'].mean() if n_short_s > 0 else 0
info(f"  Avg LONG  confidence  : {avg_long_conf:.3f}")
info(f"  Avg SHORT confidence  : {avg_short_conf:.3f}")

if avg_long_conf < 0.50 and avg_short_conf < 0.50:
    warn("Low confidence across all signals — model is uncertain")
    warn("Fix: more training data (use 5y period)")

# ── 7. BACKTEST ───────────────────────────────────────────────────────────────
section("7 · BACKTEST")
from backtest.engine import BacktestEngine, BacktestConfig

cfg = BacktestConfig(
    initial_capital   = 100_000,
    commission_pct    = 0.001,
    slippage_pct      = 0.0005,
    stop_loss_pct     = 0.03,
    take_profit_pct   = 0.09,
    use_vol_targeting = True,
    target_vol        = 0.15,
    max_position_pct  = 0.20,
    max_drawdown_limit= 0.20,
)
engine = BacktestEngine(cfg)
result = engine.run_single(test_df['close'], signals, symbol=SYMBOL)
m      = result['metrics']
trades = result['trades']

if not m:
    fail("No metrics returned — backtest produced no trades at all")
    warn("This means signals are all HOLD or the drawdown limit was hit immediately")
else:
    sharpe = m.get('sharpe', 0)
    cagr   = m.get('cagr',   0)
    mdd    = m.get('max_drawdown', 0)
    wr     = m.get('win_rate', 0)
    ntrades= m.get('num_trades', 0)

    print()
    info(f"  Sharpe ratio   : {sharpe:.3f}  {'<-- GOOD' if sharpe > 1.0 else '<-- NEEDS IMPROVEMENT'}")
    info(f"  CAGR           : {cagr:.1f}%")
    info(f"  Max drawdown   : {mdd:.1f}%")
    info(f"  Win rate       : {wr:.1f}%")
    info(f"  Num trades     : {ntrades}")
    print()

    if sharpe <= 0:
        fail("Sharpe <= 0 — strategy is destroying value")
        print()
        warn("DIAGNOSIS:")
        if ntrades == 0:
            warn("  No trades executed. Signals exist but risk manager blocked all of them.")
            warn("  Fix: check max_drawdown_limit — try 0.25 (25%) instead of 0.15")
        elif wr < 35:
            warn(f"  Win rate {wr:.1f}% is too low. Model is guessing wrong direction.")
            warn("  Fix 1: Use more data (period='5y')")
            warn("  Fix 2: Lower threshold to 0.50 so model only trades highest-conviction signals")
            warn("  Fix 3: Widen take-profit to 12% (4:1 R:R instead of 3:1)")
        elif ntrades > 0 and cagr < 0:
            warn("  Model wins some trades but losses exceed wins in dollar terms.")
            warn("  Fix: increase take_profit to 0.12, keep stop_loss at 0.03 (4:1 ratio)")
    elif sharpe < 1.0:
        warn(f"  Sharpe {sharpe:.2f} — positive but below target. See recommendations below.")
    else:
        ok(f"  Sharpe {sharpe:.2f} — good result!")

# ── 8. QUICK-FIX RECOMMENDATIONS ─────────────────────────────────────────────
section("8 · RECOMMENDED FIXES (in priority order)")

fixes = []

if len(df) < 500:
    fixes.append(("HIGH",   "Use period='3y' or '5y'",
                  f"Only {len(df)} bars. More data = better trees."))

if n_long_s + n_short_s < 10:
    fixes.append(("HIGH",   "Lower signal threshold to 0.50",
                  "Too few signals generated. Model needs lower bar to act."))

if not m or m.get('num_trades', 0) == 0:
    fixes.append(("HIGH",   "Raise max_drawdown_limit to 0.25",
                  "Risk manager may be halting the backtest too early."))

if m and m.get('win_rate', 50) < 40:
    fixes.append(("MED",    "Try forward_return_days=10 instead of 5",
                  "Longer prediction window is easier to get right."))

if m and m.get('profit_factor', 1) < 1.0:
    fixes.append(("MED",    "Set take_profit=0.12 (4:1 R:R ratio)",
                  "Wins not big enough to cover losses. Widen TP."))

fixes.append(("LOW",    "Add BTC-USD and GC=F (gold) to universe",
              "Diversification reduces portfolio vol, boosting Sharpe."))

fixes.append(("LOW",    "Run with period='5y' for walk-forward",
              "Need 5y to get 6+ quarterly windows for reliable WF analysis."))

if not fixes:
    ok("No critical fixes needed — model is performing well")
else:
    for priority, fix, reason in fixes:
        tag = {"HIGH": "[!!!]", "MED": "[ ! ]", "LOW": "[ i ]"}[priority]
        print(f"  {tag} {fix}")
        info(f"      Why: {reason}")

print(f"\n{SEP}\n  Diagnostics complete.\n{SEP}\n")