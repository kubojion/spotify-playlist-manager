@echo off
REM ── Spotify AI Playlist Manager ── Windows Launcher ──
REM Run this script from the project root directory.

cd /d "%~dp0\.."

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo Starting Spotify AI Playlist Manager...
echo.
python run.py
