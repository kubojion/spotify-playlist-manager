"""
Configuration manager for Spotify AI Playlist Manager.
Handles loading/saving API keys and preferences to a local config.json file.
"""

import os
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
except ImportError:
    pass


# Prefer environment variables for secrets. This keeps keys out of git history
# and is the recommended deployment pattern.
ENV_MAP = {
    'spotify_client_id': 'SPOTIFY_CLIENT_ID',
    'spotify_client_secret': 'SPOTIFY_CLIENT_SECRET',
    'spotify_redirect_uri': 'SPOTIFY_REDIRECT_URI',
    'openai_api_key': 'OPENAI_API_KEY',
    'gemini_api_key': 'GEMINI_API_KEY',
}

CONFIG_FILE = os.path.join(_PROJECT_ROOT, 'config.json')


def load_config():
    """Load configuration from config.json."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config):
    """Save configuration to config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def is_configured():
    """Check if required API keys are present (Spotify required, at least one AI provider)."""
    spotify_ok = get_config_value('spotify_client_id') and get_config_value(
        'spotify_client_secret')
    ai_ok = get_config_value(
        'openai_api_key') or get_config_value('gemini_api_key')
    return bool(spotify_ok and ai_ok)


def get_config_value(key, default=None):
    """Get a single config value."""
    env_key = ENV_MAP.get(key)
    if env_key and os.environ.get(env_key):
        return os.environ.get(env_key)
    return load_config().get(key, default)


def set_config_value(key, value):
    """Set a single config value."""
    config = load_config()
    config[key] = value
    save_config(config)
