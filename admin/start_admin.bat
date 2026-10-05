@echo off
title Traffic Guardian - Admin Explainability
echo ============================================
echo   Traffic Guardian - Admin Explainability
echo ============================================
echo.
cd /d "%~dp0"
echo Starting server on http://127.0.0.1:8643 ...
echo Press Ctrl+C to stop.
echo.
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8643
pause
