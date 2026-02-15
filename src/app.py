"""
Spotify AI Playlist Manager — Flask Backend
Serves the SPA and provides REST API endpoints for all features.

Architecture (post-overhaul):
- ALL generation modes are retrieval-grounded.  The LLM never "names songs from memory"
  as the primary pathway.
- Pipeline: web_search discovery → Spotify playlist retrieval → merge candidates →
  AI rerank/pick → Spotify resolve.
- OpenAI uses the Responses API with built-in web_search tool.
- Gemini falls back to standard generation + Spotify retrieval.
- Session-level tracking prevents repeated recommendations.
"""

import os
import json
import logging
import re
from collections import Counter
from flask import Flask, request, jsonify, redirect

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

from .config_manager import load_config, save_config, is_configured, get_config_value
from .spotify_client import SpotifyClient
from .ai_client import AIClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = os.urandom(24)

spotify: SpotifyClient | None = None
ai: AIClient | None = None


def init_clients():
    """Initialize Spotify and OpenAI clients from saved config."""
    global spotify, ai
    config = load_config()
    cid = get_config_value('spotify_client_id', '')
    csec = get_config_value('spotify_client_secret', '')
    oai = get_config_value('openai_api_key', '')
    gai = get_config_value('gemini_api_key', '')
    if cid and csec:
        spotify = SpotifyClient(cid, csec)
    if oai or gai:
        ai = AIClient(openai_api_key=oai or None, gemini_api_key=gai or None)
        # Apply safety settings from config
        ai.update_safety_settings(
            max_output_tokens=config.get('max_output_tokens', 0),
            max_tool_calls=config.get('max_tool_calls', 0),
            reasoning_effort=config.get('reasoning_effort', 'medium'),
        )


# Initialize on startup if already configured
if is_configured():
    init_clients()


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/status')
def api_status():
    """Check if app is configured and user is authenticated."""
    config = load_config()
    configured = is_configured()
    authenticated = False
    user = None

    if configured and spotify:
        try:
            authenticated = spotify.is_authenticated()
            if authenticated:
                u = spotify.get_current_user()
                user = {
                    'display_name': u.get('display_name', 'User'),
                    'id': u.get('id', ''),
                    'image': (u.get('images', [{}])[0].get('url') if u.get('images') else None),
                }
        except Exception as e:
            log.warning(f'Auth check failed: {e}')
            authenticated = False

    return jsonify({
        'configured': configured,
        'authenticated': authenticated,
        'user': user,
        'preferred_model': config.get('preferred_model', 'gpt-5-mini'),
        'max_output_tokens': config.get('max_output_tokens', 0),
        'max_tool_calls': config.get('max_tool_calls', 0),
        'reasoning_effort': config.get('reasoning_effort', 'medium'),
        'cost_preset': config.get('cost_preset', 'med'),
    })


@app.route('/api/setup', methods=['POST'])
def api_setup():
    """Save API keys."""
    data = request.json
    config = load_config()
    config['spotify_client_id'] = data.get('spotify_client_id', '').strip()
    config['spotify_client_secret'] = data.get(
        'spotify_client_secret', '').strip()
    config['openai_api_key'] = data.get('openai_api_key', '').strip()
    config['gemini_api_key'] = data.get('gemini_api_key', '').strip()
    save_config(config)
    init_clients()
    return jsonify({'success': True})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update app settings (model, keys)."""
    data = request.json
    config = load_config()
    if 'preferred_model' in data:
        config['preferred_model'] = data['preferred_model']
    if 'openai_api_key' in data and data['openai_api_key'].strip():
        config['openai_api_key'] = data['openai_api_key'].strip()
    if 'gemini_api_key' in data and data['gemini_api_key'].strip():
        config['gemini_api_key'] = data['gemini_api_key'].strip()
    if 'spotify_client_id' in data and data['spotify_client_id'].strip():
        config['spotify_client_id'] = data['spotify_client_id'].strip()
    if 'spotify_client_secret' in data and data['spotify_client_secret'].strip():
        config['spotify_client_secret'] = data['spotify_client_secret'].strip()
    # OpenAI safety / cost caps
    if 'max_output_tokens' in data:
        config['max_output_tokens'] = int(data['max_output_tokens'] or 0)
    if 'max_tool_calls' in data:
        config['max_tool_calls'] = int(data['max_tool_calls'] or 0)
    if 'reasoning_effort' in data:
        config['reasoning_effort'] = data['reasoning_effort']
    if 'cost_preset' in data:
        config['cost_preset'] = data['cost_preset']
    save_config(config)
    init_clients()
    return jsonify({'success': True})


@app.route('/api/auth/login')
def api_login():
    """Get Spotify authorization URL."""
    if not spotify:
        return jsonify({'error': 'Spotify not configured'}), 400
    return jsonify({'auth_url': spotify.get_auth_url()})


@app.route('/callback')
def callback():
    """Handle Spotify OAuth callback."""
    code = request.args.get('code')
    error = request.args.get('error')
    if error:
        return redirect('/?error=auth_denied')
    if code and spotify:
        try:
            spotify.handle_callback(code)
        except Exception as e:
            log.error(f'OAuth callback error: {e}')
            return redirect('/?error=auth_failed')
    return redirect('/')


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Clear Spotify auth token."""
    cache_path = '.spotify_cache'
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return jsonify({'success': True})


@app.route('/api/models')
def api_models():
    """Get available AI models (filtered by configured providers)."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400
    check = request.args.get('check_access', '').lower() in ('1', 'true')
    return jsonify({'models': ai.get_available_models(check_access=check)})


@app.route('/api/verify-keys')
def api_verify_keys():
    """Verify which API keys are valid and connected."""
    result = {
        'openai': {'configured': False, 'verified': False, 'error': None},
        'gemini': {'configured': False, 'verified': False, 'error': None},
        'spotify': {'configured': False, 'verified': False, 'error': None},
    }
    if ai:
        ai_status = ai.verify_keys()
        result['openai'] = ai_status.get('openai', result['openai'])
        result['gemini'] = ai_status.get('gemini', result['gemini'])
    if spotify:
        result['spotify']['configured'] = True
        try:
            if spotify.is_authenticated():
                result['spotify']['verified'] = True
            else:
                result['spotify']['error'] = 'Not authenticated'
        except Exception as e:
            result['spotify']['error'] = str(e)[:120]
    return jsonify(result)


@app.route('/api/playlists')
def api_playlists():
    """Get the user's Spotify playlists."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        playlists = spotify.get_playlists()
        result = []
        for p in playlists:
            if not p:
                continue
            images = p.get('images', [])
            result.append({
                'id': p['id'],
                'name': p.get('name', 'Untitled'),
                'track_count': p.get('tracks', {}).get('total', 0),
                'image': images[0]['url'] if images else None,
                'owner': p.get('owner', {}).get('display_name', ''),
                'description': p.get('description', ''),
            })
        return jsonify({'playlists': result})
    except Exception as e:
        log.error(f'Failed to fetch playlists: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/playlists/<playlist_id>/tracks')
def api_playlist_tracks(playlist_id):
    """Get tracks from a specific playlist."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        tracks = spotify.get_playlist_tracks(playlist_id)
        return jsonify({'tracks': tracks})
    except Exception as e:
        log.error(f'Failed to fetch tracks: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/playlists/<playlist_id>/rename', methods=['PUT'])
def api_rename_playlist(playlist_id):
    """Rename a playlist and/or update its description."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    name = data.get('name')
    description = data.get('description')
    if not name and description is None:
        return jsonify({'error': 'Provide name or description'}), 400
    try:
        spotify.rename_playlist(playlist_id, name=name,
                                description=description)
        return jsonify({'success': True})
    except Exception as e:
        log.error(f'Rename playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/playlists/<playlist_id>', methods=['DELETE'])
def api_delete_playlist(playlist_id):
    """Unfollow (delete) a playlist."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        spotify.delete_playlist(playlist_id)
        return jsonify({'success': True})
    except Exception as e:
        log.error(f'Delete playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/playlists/<playlist_id>/reorder', methods=['PUT'])
def api_reorder_playlist(playlist_id):
    """Reorder tracks in an existing playlist."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    track_uris = data.get('track_uris', [])
    if not track_uris:
        return jsonify({'error': 'No tracks provided'}), 400
    try:
        spotify.reorder_playlist_tracks(playlist_id, track_uris)
        return jsonify({'success': True})
    except Exception as e:
        log.error(f'Reorder playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL-GROUNDED GENERATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _retrieval_pipeline(prompt, target_count, model, extra_queries=None):
    """Search Spotify playlists for real tracks matching the prompt.

    Pipeline:
    1. AI extracts 8-12 search queries from the prompt
    2. Merge any extra queries (from web discovery)
    3. Search Spotify for public playlists matching each query
    4. Collect track candidates from those playlists  (NO popularity sort cap)
    5. AI picks the best tracks from the candidate pool
    6. Return resolved track objects

    Anti-popularity-bias: candidates are shuffled before sending to the model,
    not sorted by popularity.  The model sees pop scores but isn't biased by
    position.
    """
    # Step 1: Extract search queries
    query_result = ai.extract_search_queries(prompt, model)
    queries = query_result.get('queries', [])

    # Merge extra queries from web discovery
    if extra_queries:
        seen = {q.lower() for q in queries}
        for eq in extra_queries:
            if eq.lower() not in seen:
                queries.append(eq)
                seen.add(eq.lower())

    if not queries:
        return []

    log.info(f'Retrieval pipeline: {len(queries)} queries: {queries[:8]}')

    # Step 2 & 3: Search playlists and collect candidates
    def _tokens(s):
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    prompt_tokens = _tokens(prompt)

    def _playlist_relevance(pl, query):
        hay = f"{pl.get('name', '')} {pl.get('description', '')} {query}".lower()
        t = _tokens(hay)
        if not t or not prompt_tokens:
            return 0.0
        return len(prompt_tokens & t) / max(len(prompt_tokens), 1)

    seen_uris = set()
    candidates = []
    for q in queries[:12]:  # search up to 12 queries (was 8)
        playlists_found = spotify.search_playlists(q, limit=6)
        # Keep the top 3 most relevant playlists per query (was 2)
        playlists_scored = sorted(
            ((pl, _playlist_relevance(pl, q)) for pl in playlists_found),
            key=lambda x: x[1],
            reverse=True,
        )
        for pl, score in playlists_scored[:3]:
            tracks = spotify.get_playlist_track_candidates(pl['id'], limit=50)
            for t in tracks:
                uri = t.get('uri')
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    t['_source_playlist'] = pl.get('name', '')
                    t['_source_query'] = q
                    candidates.append(t)

    if not candidates:
        return {'songs': [], 'name': '', 'description': ''}

    log.info(
        f'Retrieval pipeline: collected {len(candidates)} unique candidates')

    # Step 4: AI picks best tracks (candidates are shuffled inside pick_from_candidates)
    pick_result = ai.pick_from_candidates(
        prompt, candidates, target_count, model)

    if not pick_result.get('playlist', {}).get('songs'):
        return {'songs': [], 'name': '', 'description': ''}

    ai_name = pick_result.get('playlist', {}).get('name', '')
    ai_desc = pick_result.get('playlist', {}).get('description', '')

    # Step 5: Resolve picked songs (they have URIs from candidates)
    picked = pick_result['playlist']['songs']
    resolved = []
    candidate_by_uri = {c['uri']: c for c in candidates}

    for song in picked:
        uri = song.get('uri', '')
        title = song.get('title', song.get('name', ''))
        artist = song.get('artist', '')

        if uri and uri in candidate_by_uri:
            track = spotify.get_track_by_uri(uri)
            if track:
                resolved.append(track)
                continue

        # Fallback: search normally
        track = spotify.search_track(title, artist)
        if track:
            resolved.append(track)

    return {'songs': resolved, 'name': ai_name, 'description': ai_desc}


def _web_discovery_pipeline(prompt, target_count, model):
    """Full retrieval-grounded pipeline with web search.

    1. Web search discovery: OpenAI searches the web for song recommendations
    2. Resolve web-discovered songs via Spotify search
    3. Spotify playlist retrieval using queries from web discovery + AI extraction
    4. Merge all candidates (web-found + retrieval)
    5. AI reranks and picks the final set
    6. Return resolved tracks

    This is the PRIMARY generation path for auto/thinking modes with OpenAI.
    """
    log.info(
        f'Web discovery pipeline: "{prompt[:60]}..." → {target_count} tracks')

    # ── Step 1: Web search discovery ──────────────────────────────────
    pool_target = max(target_count * 2, 40)
    web_result = ai.web_discover(prompt, count=pool_target, model=model)

    web_songs = web_result.get('songs', [])
    web_queries = web_result.get('search_queries', [])
    log.info(
        f'Web discovery found {len(web_songs)} songs, {len(web_queries)} queries')

    # ── Step 2: Resolve web-discovered songs via Spotify ──────────────
    web_resolved = []
    web_uris = set()
    for song in web_songs:
        title = song.get('title', '')
        artist = song.get('artist', '')
        track = spotify.search_track(title, artist)
        if track and track.get('uri'):
            if track['uri'] not in web_uris:
                web_uris.add(track['uri'])
                # Mark as web-discovered candidate
                web_resolved.append({
                    'title': track['name'],
                    'artist': track['artist'],
                    'uri': track['uri'],
                    'popularity': track.get('popularity', 0),
                    '_source_playlist': 'web_search',
                    '_source_query': 'web_discovery',
                })

    log.info(
        f'Web songs resolved on Spotify: {len(web_resolved)}/{len(web_songs)}')

    # ── Step 3: Spotify playlist retrieval ────────────────────────────
    retrieval_candidates = []
    try:
        query_result = ai.extract_search_queries(prompt, model)
        ai_queries = query_result.get('queries', [])
        all_queries = list(web_queries) + ai_queries

        # Deduplicate queries
        seen_q = set()
        deduped = []
        for q in all_queries:
            low = q.lower().strip()
            if low not in seen_q:
                seen_q.add(low)
                deduped.append(q)

        seen_uris = set(web_uris)  # Don't re-add web-found tracks
        for q in deduped[:12]:
            playlists_found = spotify.search_playlists(q, limit=5)
            for pl in playlists_found[:3]:
                tracks = spotify.get_playlist_track_candidates(
                    pl['id'], limit=50)
                for t in tracks:
                    uri = t.get('uri')
                    if uri and uri not in seen_uris:
                        seen_uris.add(uri)
                        t['_source_playlist'] = pl.get('name', '')
                        t['_source_query'] = q
                        retrieval_candidates.append(t)

        log.info(f'Retrieval added {len(retrieval_candidates)} candidates')
    except Exception as e:
        log.warning(f'Retrieval step failed: {e}')

    # ── Step 4: Merge all candidates ──────────────────────────────────
    all_candidates = web_resolved + retrieval_candidates
    log.info(f'Total candidate pool: {len(all_candidates)}')

    if not all_candidates:
        return []

    # ── Step 5: AI reranks and picks ──────────────────────────────────
    pick_result = ai.pick_from_candidates(
        prompt, all_candidates, target_count, model)

    if not pick_result.get('playlist', {}).get('songs'):
        # Fallback: return web_resolved directly if picking failed
        return _resolve_by_uris(web_resolved[:target_count])

    ai_name = pick_result.get('playlist', {}).get('name', '')
    ai_desc = pick_result.get('playlist', {}).get('description', '')
    picked = pick_result['playlist']['songs']

    # ── Step 6: Resolve final tracks ──────────────────────────────────
    candidate_by_uri = {c['uri']: c for c in all_candidates}
    resolved = []
    for song in picked:
        uri = song.get('uri', '')
        if uri and uri in candidate_by_uri:
            track = spotify.get_track_by_uri(uri)
            if track:
                resolved.append(track)
                continue
        # Fallback search
        track = spotify.search_track(
            song.get('title', ''), song.get('artist', ''))
        if track:
            resolved.append(track)

    log.info(
        f'Web discovery pipeline complete: {len(resolved)} tracks resolved')
    return {'songs': resolved, 'name': ai_name, 'description': ai_desc}


def _resolve_by_uris(candidates):
    """Resolve a list of candidate dicts with 'uri' field to full track objects."""
    resolved = []
    for c in candidates:
        uri = c.get('uri', '')
        if uri:
            track = spotify.get_track_by_uri(uri)
            if track:
                resolved.append(track)
                continue
        track = spotify.search_track(c.get('title', ''), c.get('artist', ''))
        if track:
            resolved.append(track)
    return resolved


def _resolve_songs(songs):
    """Search Spotify for each AI-recommended song and enrich with metadata."""
    resolved = []
    for song in songs:
        title = song.get('title', song.get('name', ''))
        artist = song.get('artist', '')

        track = spotify.search_track(title, artist)

        if not track and artist:
            track = spotify.search_track(title, '')

        if track:
            resolved.append(track)
        else:
            resolved.append({
                'id': None,
                'name': title,
                'artist': artist,
                'artists': [artist] if artist else [],
                'album': '',
                'album_art': None,
                'album_art_small': None,
                'release_date': '',
                'popularity': 0,
                'preview_url': None,
                'spotify_url': '',
                'uri': '',
                'not_found': True,
            })
    return resolved


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/generate/from-prompt', methods=['POST'])
def api_generate_from_prompt():
    """Generate a playlist from a text description.

    ALL modes are now retrieval-grounded:
    - 'quick': single-pass LLM generation + Spotify search resolve (fastest, lowest quality)
    - 'auto': web discovery + Spotify retrieval → AI rerank (balanced)
    - 'thinking': same as auto but with higher reasoning effort (slowest, best quality)
    """
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    prompt = data.get('prompt', '')
    size = data.get('size', 'medium')
    model = data.get('model', 'gpt-5-mini')
    mode = data.get('mode', 'auto')
    history = data.get('conversation_history', [])

    size_map = {'small': 15, 'medium': 30, 'large': 50}
    target_count = size_map.get(size, 30)

    try:
        # ── Quick mode: single-pass LLM + resolve ────────────────────
        if mode == 'quick':
            result = ai.generate_from_prompt(
                prompt, size, model, history, mode='quick')

            if result.get('status') == 'clarify':
                return jsonify(result)

            if result.get('status') == 'ready' and result.get('playlist'):
                songs = result['playlist'].get('songs', [])
                resolved = _resolve_songs(songs)
                found = [s for s in resolved if not s.get('not_found')]

                # If too many not found, try retrieval pipeline as backup
                if len(found) < target_count * 0.6:
                    try:
                        retrieval_result = _retrieval_pipeline(
                            prompt, target_count, model)
                        retrieval_songs = retrieval_result.get('songs', []) if isinstance(
                            retrieval_result, dict) else retrieval_result
                        if retrieval_songs:
                            found_uris = {s.get('uri')
                                          for s in found if s.get('uri')}
                            extra = [s for s in retrieval_songs
                                     if s.get('uri') and s['uri'] not in found_uris]
                            merged = list(found)
                            for s in extra:
                                if len(merged) >= target_count:
                                    break
                                merged.append(s)
                            if merged:
                                resolved = merged
                    except Exception as e:
                        log.warning(f'Quick mode retrieval backup failed: {e}')

                result['playlist']['songs'] = resolved
                # Track recommendations for session memory
                ai.record_recommended(resolved)
            # Attach debug info for Quick mode too
            result_data = result if isinstance(result, dict) else {}
            if ai._last_usage:
                result_data['_debug'] = {'usage': ai._last_usage}
            return jsonify(result_data)

        # ── Auto / Thinking mode: full retrieval-grounded pipeline ────
        # Both OpenAI (web_search tool) and Gemini (Google Search grounding)
        # now support web discovery — use the same pipeline for both.
        # Web discovery is best-effort: if it fails (400, timeout, etc.),
        # fall back to Spotify-only retrieval rather than returning nothing.
        # 'spotify-only' mode skips web search entirely.
        resolved = []
        playlist_name = ''
        playlist_desc = ''

        if mode != 'spotify-only':
            try:
                pipeline_result = _web_discovery_pipeline(
                    prompt, target_count, model)
                resolved = pipeline_result.get('songs', []) if isinstance(
                    pipeline_result, dict) else pipeline_result
                if isinstance(pipeline_result, dict):
                    playlist_name = pipeline_result.get('name', '')
                    playlist_desc = pipeline_result.get('description', '')
            except Exception as e:
                log.warning(f'Web discovery pipeline failed, falling back to '
                            f'retrieval-only: {e}')

        if not resolved:
            # Fallback 1 (or primary for spotify-only): Spotify retrieval
            try:
                pipeline_result = _retrieval_pipeline(
                    prompt, target_count, model)
                resolved = pipeline_result.get('songs', []) if isinstance(
                    pipeline_result, dict) else pipeline_result
                if isinstance(pipeline_result, dict) and not playlist_name:
                    playlist_name = pipeline_result.get('name', '')
                    playlist_desc = pipeline_result.get('description', '')
            except Exception as e:
                log.warning(f'Retrieval pipeline also failed: {e}')

        if not resolved:
            # Fallback 2: plain LLM generation + resolve
            result = ai.generate_from_prompt(
                prompt, size, model, history, mode='quick')
            if result.get('status') == 'ready' and result.get('playlist'):
                resolved = _resolve_songs(result['playlist'].get('songs', []))
                resolved = [s for s in resolved if not s.get('not_found')]

        # Build response
        ai.record_recommended(resolved)

        # Use AI-generated name if available, otherwise fallback
        final_name = playlist_name or (
            prompt[:50] if prompt else 'AI Playlist')
        final_desc = playlist_desc or f'Generated from: {prompt[:100]}'

        # Collect debug info if available
        debug_info = None
        if ai._last_usage:
            debug_info = {'usage': ai._last_usage}

        response_data = {
            'status': 'ready',
            'message': f'Found {len(resolved)} tracks matching your request.',
            'suggestions': [],
            'playlist': {
                'name': final_name,
                'description': final_desc,
                'songs': resolved,
            }
        }
        if debug_info:
            response_data['_debug'] = debug_info
        return jsonify(response_data)

    except Exception as e:
        log.error(f'Generate from prompt failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/from-playlists', methods=['POST'])
def api_generate_from_playlists():
    """Generate a playlist based on the user's existing playlists.

    ALL modes are retrieval-grounded:
    - Web search discovers songs matching the user's taste profile
    - Spotify retrieval finds real tracks from similar public playlists
    - AI reranks to pick the best matches
    """
    if not ai or not spotify:
        return jsonify({'error': 'Not configured'}), 400
    if not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    playlist_ids = data.get('playlist_ids', [])
    prompt = data.get('prompt', '')
    size = data.get('size', 'medium')
    model = data.get('model', 'gpt-5-mini')
    mode = data.get('mode', 'auto')
    history = data.get('conversation_history', [])

    if not playlist_ids:
        return jsonify({'error': 'No playlists selected'}), 400

    size_map = {'small': 15, 'medium': 30, 'large': 50}
    target_count = size_map.get(size, 30)

    try:
        # Fetch playlist summaries
        all_playlists = spotify.get_playlists()
        playlist_name_map = {p['id']: p.get(
            'name', 'Untitled') for p in all_playlists if p}

        summaries = []
        for pid in playlist_ids:
            summary = spotify.get_playlist_summary(pid)
            summary['name'] = playlist_name_map.get(pid, 'Unknown Playlist')
            summaries.append(summary)

        # Build context text for web discovery
        context_parts = []
        for ps in summaries:
            context_parts.append(
                f"Playlist: {ps['name']} ({ps['track_count']} tracks)")
            if ps.get('top_artists'):
                context_parts.append(
                    f"  Artists: {', '.join(ps['top_artists'][:8])}")
            if ps.get('sample_tracks'):
                context_parts.append(
                    f"  Tracks: {', '.join(ps['sample_tracks'][:10])}")
        playlist_context = '\n'.join(context_parts)

        # Build a combined prompt for retrieval
        taste_prompt = f"Based on taste: {playlist_context}"
        if prompt:
            taste_prompt += f"\n\nUser direction: {prompt}"

        # ── Quick mode ────────────────────────────────────────────────
        if mode == 'quick':
            result = ai.generate_from_playlists(
                summaries, prompt, size, model, history, mode='quick')

            if result.get('status') == 'clarify':
                return jsonify(result)

            if result.get('status') == 'ready' and result.get('playlist'):
                result['playlist']['songs'] = _resolve_songs(
                    result['playlist']['songs'])
                ai.record_recommended(result['playlist']['songs'])
            # Attach debug info for Quick mode too
            if ai._last_usage:
                result['_debug'] = {'usage': ai._last_usage}
            return jsonify(result)

        # ── Auto / Thinking mode ──────────────────────────────────────
        # Web discovery is best-effort: if it fails, continue with
        # Spotify-only retrieval rather than aborting.
        # 'spotify-only' mode skips web search entirely.
        resolved = []

        # Web discovery from playlists (best-effort, skipped in spotify-only)
        pool_target = max(target_count * 2, 40)
        web_songs = []
        web_queries = []
        if mode != 'spotify-only':
            try:
                web_result = ai.web_discover_from_playlists(
                    playlist_context, prompt, count=pool_target, model=model)
                web_songs = web_result.get('songs', [])
                web_queries = web_result.get('search_queries', [])
            except Exception as e:
                log.warning(f'Web discovery from playlists failed, continuing '
                            f'with retrieval-only: {e}')
        log.info(
            f'Playlist web discovery: {len(web_songs)} songs, {len(web_queries)} queries')

        # Resolve web songs
        web_resolved = []
        web_uris = set()
        for song in web_songs:
            track = spotify.search_track(
                song.get('title', ''), song.get('artist', ''))
            if track and track.get('uri') and track['uri'] not in web_uris:
                web_uris.add(track['uri'])
                web_resolved.append({
                    'title': track['name'],
                    'artist': track['artist'],
                    'uri': track['uri'],
                    'popularity': track.get('popularity', 0),
                    '_source_playlist': 'web_search',
                    '_source_query': 'web_discovery',
                })

        # Retrieval pipeline with combined queries
        retrieval_candidates = []
        try:
            all_queries = list(web_queries)
            # Add taste-based queries
            query_result = ai.extract_search_queries(taste_prompt, model)
            all_queries.extend(query_result.get('queries', []))

            seen_q = set()
            deduped = []
            for q in all_queries:
                low = q.lower().strip()
                if low not in seen_q:
                    seen_q.add(low)
                    deduped.append(q)

            seen_uris = set(web_uris)
            for q in deduped[:12]:
                playlists_found = spotify.search_playlists(q, limit=5)
                for pl in playlists_found[:3]:
                    tracks = spotify.get_playlist_track_candidates(
                        pl['id'], limit=50)
                    for t in tracks:
                        uri = t.get('uri')
                        if uri and uri not in seen_uris:
                            seen_uris.add(uri)
                            t['_source_playlist'] = pl.get('name', '')
                            t['_source_query'] = q
                            retrieval_candidates.append(t)
        except Exception as e:
            log.warning(f'Retrieval step failed: {e}')

        # Merge and pick
        all_candidates = web_resolved + retrieval_candidates
        playlist_name = ''
        playlist_desc = ''
        if all_candidates:
            pick_result = ai.pick_from_candidates(
                taste_prompt, all_candidates, target_count, model)
            playlist_name = pick_result.get('playlist', {}).get('name', '')
            playlist_desc = pick_result.get(
                'playlist', {}).get('description', '')
            if pick_result.get('playlist', {}).get('songs'):
                candidate_by_uri = {c['uri']: c for c in all_candidates}
                for song in pick_result['playlist']['songs']:
                    uri = song.get('uri', '')
                    if uri and uri in candidate_by_uri:
                        track = spotify.get_track_by_uri(uri)
                        if track:
                            resolved.append(track)
                            continue
                    track = spotify.search_track(
                        song.get('title', ''), song.get('artist', ''))
                    if track:
                        resolved.append(track)

        # Fallback if pipeline yielded nothing
        if not resolved:
            result = ai.generate_from_playlists(
                summaries, prompt, size, model, history, mode='quick')
            if result.get('status') == 'ready' and result.get('playlist'):
                resolved = _resolve_songs(result['playlist'].get('songs', []))
                resolved = [s for s in resolved if not s.get('not_found')]
                if not playlist_name:
                    playlist_name = result.get('playlist', {}).get('name', '')
                    playlist_desc = result.get(
                        'playlist', {}).get('description', '')

        ai.record_recommended(resolved)

        final_name = playlist_name or 'Based on your taste'
        final_desc = playlist_desc or prompt or 'Generated from your playlists'

        # Collect debug info if available
        debug_info = None
        if ai._last_usage:
            debug_info = {'usage': ai._last_usage}

        response_data = {
            'status': 'ready',
            'message': f'Found {len(resolved)} tracks based on your taste.',
            'suggestions': [],
            'playlist': {
                'name': final_name,
                'description': final_desc,
                'songs': resolved,
            }
        }
        if debug_info:
            response_data['_debug'] = debug_info
        return jsonify(response_data)

    except Exception as e:
        log.error(f'Generate from playlists failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/refine', methods=['POST'])
def api_refine():
    """Refine a generated playlist based on user feedback."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    current_songs = data.get('current_songs', [])
    new_prompt = data.get('prompt', '')
    model = data.get('model', 'gpt-5-mini')

    try:
        result = ai.refine_playlist(current_songs, {}, new_prompt, model)
        if result.get('status') == 'ready' and result.get('playlist'):
            result['playlist']['songs'] = _resolve_songs(
                result['playlist']['songs'])
            ai.record_recommended(result['playlist']['songs'])
        return jsonify(result)
    except Exception as e:
        log.error(f'Refine failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat with AI about playlists and music."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400

    data = request.json
    messages = data.get('messages', [])
    model = data.get('model', 'gpt-5-mini')

    # Build playlist context
    playlist_context = None
    if spotify and spotify.is_authenticated():
        try:
            pls = spotify.get_playlists()
            playlist_context = [
                {'name': p.get('name', ''), 'id': p['id'],
                 'tracks': p.get('tracks', {}).get('total', 0)}
                for p in pls if p
            ]
        except Exception:
            pass

    try:
        result = ai.chat(messages, playlist_context, model)

        # If AI wants to create a playlist, resolve songs
        if result.get('type') == 'create_playlist' and result.get('playlist', {}).get('songs'):
            result['playlist']['songs'] = _resolve_songs(
                result['playlist']['songs'])

        return jsonify(result)
    except Exception as e:
        log.error(f'Chat failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-playlist', methods=['POST'])
def api_save_playlist():
    """Save a generated playlist to the user's Spotify account."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    name = (data.get('name') or 'AI Generated Playlist').strip()[:100]
    description = (data.get(
        'description') or 'Generated by Spotify AI Playlist Manager').strip()[:300]
    track_uris = data.get('track_uris', [])

    if not track_uris:
        return jsonify({'error': 'No tracks to save'}), 400

    try:
        playlist = spotify.create_playlist(name, description)
        spotify.add_tracks_to_playlist(playlist['id'], track_uris)
        return jsonify({
            'success': True,
            'playlist_id': playlist['id'],
            'playlist_url': playlist.get('external_urls', {}).get('spotify', ''),
        })
    except Exception as e:
        log.error(f'Save playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/add-to-playlist', methods=['POST'])
def api_add_to_playlist():
    """Add tracks to an existing Spotify playlist."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    playlist_id = data.get('playlist_id')
    track_uris = data.get('track_uris', [])

    if not playlist_id:
        return jsonify({'error': 'No playlist specified'}), 400
    if not track_uris:
        return jsonify({'error': 'No tracks to add'}), 400

    try:
        spotify.add_tracks_to_playlist(playlist_id, track_uris)
        return jsonify({'success': True})
    except Exception as e:
        log.error(f'Add to playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/sort-playlist', methods=['POST'])
def api_sort_playlist():
    """Sort a playlist by a given criterion (preview only, no auto-save)."""
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    playlist_id = data.get('playlist_id')
    sort_by = data.get('sort_by', 'release_date')

    if not playlist_id:
        return jsonify({'error': 'No playlist selected'}), 400

    try:
        tracks = spotify.get_playlist_tracks(playlist_id)

        if sort_by == 'release_date':
            tracks.sort(key=lambda x: x.get('release_date', '0000'))
        elif sort_by == 'release_date_desc':
            tracks.sort(key=lambda x: x.get(
                'release_date', '0000'), reverse=True)
        elif sort_by == 'popularity':
            tracks.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        elif sort_by == 'popularity_asc':
            tracks.sort(key=lambda x: x.get('popularity', 0))
        elif sort_by == 'artist':
            tracks.sort(key=lambda x: x.get('artist', '').lower())
        elif sort_by == 'name':
            tracks.sort(key=lambda x: x.get('name', '').lower())

        return jsonify({
            'success': True,
            'tracks': tracks,
        })
    except Exception as e:
        log.error(f'Sort playlist failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/manage/sort-by-popularity', methods=['POST'])
def api_manage_sort_by_popularity():
    """Sort a playlist by popularity using AI knowledge."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    playlist_id = data.get('playlist_id')
    model = data.get('model', 'gpt-5-mini')

    if not playlist_id:
        return jsonify({'error': 'No playlist selected'}), 400

    try:
        tracks = spotify.get_playlist_tracks(playlist_id)
        track_summaries = [
            {'name': t.get('name', ''), 'artist': t.get('artist', '')} for t in tracks]

        result = ai.sort_by_popularity(track_summaries, model)

        # Map AI-sorted songs back to real track objects
        sorted_tracks = []
        for song in result.get('songs', []):
            name = song.get('title', song.get('name', ''))
            artist = song.get('artist', '')
            match = next(
                (t for t in tracks if t.get('name', '').lower() == name.lower()
                 and t.get('artist', '').lower() == artist.lower()),
                None
            )
            if not match:
                match = next(
                    (t for t in tracks if t.get('name', '').lower() == name.lower()),
                    None
                )
            if match:
                sorted_tracks.append(match)
                tracks = [t for t in tracks if t.get(
                    'uri') != match.get('uri')]

        sorted_tracks.extend(tracks)

        return jsonify({
            'success': True,
            'tracks': sorted_tracks,
            'message': result.get('message', 'Sorted by popularity'),
        })
    except Exception as e:
        log.error(f'AI sort by popularity failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/manage/add-songs', methods=['POST'])
def api_manage_add_songs():
    """Use AI to suggest songs to add to a playlist."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400
    if not spotify or not spotify.is_authenticated():
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    playlist_id = data.get('playlist_id')
    count = data.get('count', 10)
    model = data.get('model', 'gpt-5-mini')

    if not playlist_id:
        return jsonify({'error': 'No playlist selected'}), 400

    try:
        tracks = spotify.get_playlist_tracks(playlist_id)
        track_summaries = [
            {'name': t.get('name', ''), 'artist': t.get('artist', '')} for t in tracks]

        result = ai.add_songs(track_summaries, count, model)

        # Resolve AI suggestions via Spotify search
        new_songs = _resolve_songs(result.get('songs', []))

        return jsonify({
            'success': True,
            'songs': new_songs,
            'message': result.get('message', f'Added {count} new songs'),
        })
    except Exception as e:
        log.error(f'AI add songs failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/manage/chat', methods=['POST'])
def api_manage_chat():
    """Chat with AI about a specific playlist's songs."""
    if not ai:
        return jsonify({'error': 'No AI provider configured'}), 400

    data = request.json
    messages = data.get('messages', [])
    playlist_tracks = data.get('playlist_tracks', [])
    model = data.get('model', 'gpt-5-mini')

    try:
        result = ai.chat_about_playlist(messages, playlist_tracks, model)
        return jsonify(result)
    except Exception as e:
        log.error(f'Manage chat failed: {e}')
        return jsonify({'error': str(e)}), 500


def create_app():
    """Application factory for external runners."""
    return app
