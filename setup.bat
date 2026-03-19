@echo off
chcp 65001 >nul 2>&1
setlocal

echo.
echo   QBOT Trading Bot -- Windows Setup
echo   -----------------------------------
echo.

REM -- Check Python --
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found.
    echo   Download from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version') do set PYVER=%%i
echo   OK Python %PYVER%

REM -- Check git --
git --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: git not found.
    echo   Download from https://git-scm.com/download/win
    pause
    exit /b 1
)
echo   OK git found

REM -- Create venv --
echo.
echo [2/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists
)
call .venv\Scripts\activate.bat
echo   Activated .venv

REM -- Install deps --
echo.
echo [3/5] Installing dependencies (this takes 2-3 min)...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo   ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo   All packages installed

REM -- Create dirs --
echo.
echo [4/5] Creating project structure...
if not exist "data\cache"      mkdir data\cache
if not exist "dashboard"       mkdir dashboard
if not exist "tests"           mkdir tests
if not exist "config"          mkdir config
type nul >> data\__init__.py
type nul >> core\__init__.py
type nul >> backtest\__init__.py
type nul >> live\__init__.py
echo   Directories ready

REM -- Smoke test --
echo.
echo [5/5] Running smoke test...
python -c "import sys; sys.path.insert(0,'.'); from data.fetcher import DataFetcher; df = DataFetcher.synthetic(n=100); print('    Data OK:', len(df), 'rows'); from core.feature_engineer import FeatureEngineer; fe = FeatureEngineer.add_features(df); print('    Features OK:', len(FeatureEngineer.FEATURE_COLS), 'cols')"
if errorlevel 1 (
    echo   WARNING: Smoke test had errors - check imports
) else (
    echo   Smoke test passed
)

REM -- Git --
echo.
echo Initialising git...
if not exist ".git" (
    git init -q
)
git add -A
git commit -m "feat: initial QBOT setup" --allow-empty -q 2>nul
echo   Git ready

echo.
echo ========================================================
echo   SETUP COMPLETE
echo ========================================================
echo.
echo   To run locally:
echo     1. Double-click: start.bat
echo     OR run in terminal:
echo        .venv\Scripts\activate
echo        python api_server.py
echo     2. Open: http://localhost:5000
echo.
echo   To deploy to Render:
echo     See README.md -- Step 2 and Step 3
echo.
echo ========================================================
echo.
pause
