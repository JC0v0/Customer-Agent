@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto novenv
echo Starting dev app. Log: run_dev.log
".venv\Scripts\python.exe" app.py >> "%~dp0run_dev.log" 2>&1
echo.
echo App exited with code %ERRORLEVEL%
pause
exit /b 0

:novenv
echo Missing .venv. Please run: uv sync
pause
exit /b 1
