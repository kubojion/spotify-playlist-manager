#!/usr/bin/env bash
# ── Spotify AI Playlist Manager ── Mac/Linux Launcher ──
# Run this script from the project root directory.

set -e
cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo
echo "Starting Spotify AI Playlist Manager..."
echo
python run.py
