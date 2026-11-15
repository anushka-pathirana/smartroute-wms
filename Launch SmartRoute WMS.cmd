@echo off
rem ===================================================================
rem  SmartRoute ML - Warehouse Management System launcher
rem  Double-click to start the app; the browser opens automatically.
rem  Close this window (or press Ctrl+C) to stop the app.
rem ===================================================================
title SmartRoute ML - WMS
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Please install Python 3.12+ and try again.
    pause
    exit /b 1
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages - first run only...
    python -m pip install -r requirements.txt
)

if not exist "models\footwear_routing_policy_xgb.joblib" (
    echo Trained models not found. Running the pipeline first - this takes about 25 minutes...
    python main.py
)

echo Starting SmartRoute ML at http://localhost:8501 ...
python -m streamlit run app\streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
pause
