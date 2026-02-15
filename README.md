# Spotify AI Playlist Manager

An AI-powered Spotify playlist manager that uses real web research and retrieval to find songs. Unlike simple recommendation engines, the AI actually searches the web, analyzes your music taste, and cross-references against Spotify's catalog to build playlists with genuine depth and variety.

You supply your own API keys. Nothing is stored externally.

---

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/42ceeb70-cf45-4bc1-b4b0-ecf3734d738d" width="100%">
  <br><br>
  <em>Preview of Main Window</em>
</div>
<br>


## What It Does

**Generate from your playlists** -- Select existing playlists. The AI analyzes your taste and builds a new playlist of songs you will actually like, pulling from the full Spotify catalog.

**Generate from a prompt** -- Describe what you want in plain language. The AI researches artists, subgenres, and deep cuts using web search, then resolves every track against Spotify.

**Manage playlists** -- Sort by release date, popularity, artist, or name. Chat with the AI about your library. Add AI-matched songs to existing playlists. Rename, delete, and edit playlists inline.

**Multi-model support** -- Choose between OpenAI models (GPT-5 Mini, GPT-5.2, GPT-5 Nano, and others) or Google Gemini models. Each brings different strengths.

**Other features** -- 30-second previews, album art, thumbs up/down ratings with iterative refinement, one-click save to Spotify, dark Spotify-inspired UI.

---

## Prerequisites

- Python 3.10 or newer
- A Spotify Developer account (free)
- An OpenAI API key, a Google Gemini API key, or both

---

## Installation & Usage

### Option A: Windows .exe

No installation required. Just download and run.

1.  Go to the **[Releases Page](../../releases)** on GitHub.
2.  Download the latest `SpotifyPlaylistManager.exe`.
3.  Double-click to run.
    * *Note: Windows Defender might flag this because it is not digitally signed. Click "More Info" > "Run Anyway".*
4.  The app will open in your browser automatically.

### Option B: Run from Source (Windows/Mac/Linux)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kubojion/spotify-playlist-manager.git](https://github.com/kubojion/spotify-playlist-manager.git)
    cd spotify-playlist-manager
    ```

2.  **Run the launcher:**
    * **Windows:** Double-click `scripts\start.bat`
    * **Mac/Linux:** Run `./scripts/start.sh` (you may need to run `chmod +x scripts/start.sh` first).

The script automatically creates a virtual environment, installs Python dependencies, and launches the app.
---

## First Time Setup (API Keys)

On the first launch, the app will open a setup screen in your browser. You will need to enter your keys:

### 1. Get Spotify Keys
1.  Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2.  Log in and click **"Create App"**.
3.  **Important:** In the app settings, find "Redirect URIs" and add exactly:
    `http://127.0.0.1:5000/callback`
4.  Copy the **Client ID** and **Client Secret**.

### 2. Get AI Keys (Choose One or Both)
* **OpenAI:** Go to [OpenAI API Keys](https://platform.openai.com/api-keys). (Requires a funded account).
* **Google Gemini:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey). (Free tier available).

Enter these keys into the app setup screen to begin.

---

## Project Structure

```
spotify-playlist-manager/
  src/                  Application source code
    app.py              Flask backend and API routes
    ai_client.py        Multi-provider AI client (OpenAI + Gemini)
    spotify_client.py   Spotify API wrapper
    config_manager.py   Configuration and secrets management
    static/             Frontend (HTML, CSS, JS)
  scripts/              Launcher scripts
    start.bat           Windows
    start.sh            macOS / Linux
  docker/               Docker setup (optional)
  run.py                Entry point
  requirements.txt      Python dependencies
  .env.example          Environment variable template
```

---

## Configuration

API keys can be provided in two ways:

1. **In-app setup** (recommended) -- Enter your keys in the browser on first launch. They are saved to `config.json` in the project root.

2. **Environment variables** -- Copy `.env.example` to `.env` and fill in your keys. Environment variables take priority over `config.json`.

---

## Notes

- The Spotify redirect URI must be `http://127.0.0.1:5000/callback`. Set this in both your Spotify Developer Dashboard and your configuration.
- The app runs locally on port 5000. It does not expose anything to the internet.
- `config.json` and `.env` are in `.gitignore` and will not be committed.
- This is a personal tool. You use your own API keys and interact only with your own Spotify account.

---

## License
MIT

