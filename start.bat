@echo off
chcp 65001 >nul 2>&1
echo.
echo   Starting QBOT...
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo   ERROR: Virtual environment not found.
    echo   Run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo   Server starting on http://localhost:5000
echo   Press Ctrl+C to stop.
echo.

python api_server.py

pause
