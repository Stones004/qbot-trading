#!/usr/bin/env bash
# setup.sh — QBOT setup (Windows-safe, ASCII only)
set -e

echo ""
echo "  QBOT Trading Bot -- Setup"
echo "  --------------------------"
echo ""

# -- 1. Python version check --
echo "[1/5] Checking environment..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)
if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
  echo "  ERROR: Python 3.9+ required (found $python_version)"
  exit 1
fi
echo "  OK Python $python_version"
git --version &>/dev/null || { echo "  ERROR: git not installed"; exit 1; }
echo "  OK git $(git --version | awk '{print $3}')"

# -- 2. Create venv --
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "  Created .venv"
else
  echo "  .venv already exists"
fi
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
echo "  Activated .venv"

# -- 3. Install dependencies --
echo ""
echo "[3/5] Installing dependencies (this takes 2-3 min)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  All packages installed"

# -- 4. Create directories --
echo ""
echo "[4/5] Creating project structure..."
mkdir -p data/cache dashboard/static tests config
touch data/__init__.py 2>/dev/null || true
touch core/__init__.py 2>/dev/null || true
touch backtest/__init__.py 2>/dev/null || true
touch live/__init__.py 2>/dev/null || true
echo "  Directories ready"

# -- 5. Smoke test --
echo ""
echo "[5/5] Running smoke test..."
python3 -c "
import sys; sys.path.insert(0,'.')
from data.fetcher import DataFetcher
df = DataFetcher.synthetic(n=120)
print('    Synthetic data:', len(df), 'rows - OK')
from core.feature_engineer import FeatureEngineer
fe = FeatureEngineer.add_features(df)
print('    Features:', len(FeatureEngineer.FEATURE_COLS), 'cols - OK')
from backtest.engine import BacktestEngine
import pandas as pd, numpy as np
prices  = fe['close']
signals = pd.Series(np.random.choice([-1,0,1], len(prices)), index=prices.index)
result  = BacktestEngine().run_single(prices, signals)
print('    Backtest sharpe =', result['metrics'].get('sharpe','n/a'), '- OK')
"
echo "  Smoke test passed"

# -- Git init --
echo ""
echo "Initialising git..."
if [ ! -d ".git" ]; then
  git init -q
fi
git add -A
git commit -m "feat: initial QBOT setup" --allow-empty -q
echo "  Git ready"

echo ""
echo "========================================================"
echo "  SETUP COMPLETE"
echo "========================================================"
echo ""
echo "  Run locally:"
echo ""
echo "    Windows:  .venv\\Scripts\\activate"
echo "    Mac/Linux: source .venv/bin/activate"
echo "    Then:      python api_server.py"
echo "    Open:      http://localhost:5000"
echo ""
echo "  Deploy to Render:"
echo ""
echo "  Step 1 - Create GitHub repo at https://github.com/new"
echo "           Name: qbot-trading"
echo "           Leave empty (no README)"
echo ""
echo "  Step 2 - Push code:"
echo "    git remote add origin https://github.com/YOUR_NAME/qbot-trading.git"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "  Step 3 - Go to https://render.com"
echo "    New + > Web Service > Connect GitHub > qbot-trading"
echo "    render.yaml is detected automatically"
echo "    Click: Create Web Service"
echo "    Wait ~3 min. Live at: https://qbot-trading.onrender.com"
echo ""
echo "========================================================"
echo ""
