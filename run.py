from src.app import app
import os
import sys
import webbrowser
from threading import Timer

# Ensure the 'src' directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def open_browser():
    """Opens the browser after a 1.5s delay to allow the server to start."""
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == '__main__':
    print('Starting Spotify AI Playlist Manager...')

    # 1. Schedule the browser to open in 1.5 seconds
    Timer(1.5, open_browser).start()

    # 2. Start the server (This blocks execution until you press Ctrl+C)
    app.run(host='127.0.0.1', port=5000, debug=True)
