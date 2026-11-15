@echo off
rem ===================================================================
rem  Warehouse 3D Viewer - double-click to open in your default browser.
rem  Works offline; no server needed. Rebuilds the data file if missing.
rem ===================================================================
title Warehouse 3D Viewer
cd /d "%~dp0"

if not exist "warehouse_data.js" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo warehouse_data.js is missing and Python was not found on PATH.
        pause
        exit /b 1
    )
    echo Building warehouse_data.js from the SQLite review database...
    python build_viewer.py
)

start "" "%~dp0index.html"
