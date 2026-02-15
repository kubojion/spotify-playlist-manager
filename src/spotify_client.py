"""
Spotify API client wrapper.
Handles authentication, playlist management, and track operations.
"""

import logging
import re
from difflib import SequenceMatcher

import spotipy
from spotipy.oauth2 import SpotifyOAuth

log = logging.getLogger(__name__)


class SpotifyClient:
    def __init__(self, client_id, client_secret, redirect_uri='http://127.0.0.1:5000/callback',
                 market='US'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        # Market for track relinking (auto-detected from user profile on login)
        self.market = market
        self._market_auto_detected = False
        self.scope = (
            'playlist-read-private '
            'playlist-read-collaborative '
            'playlist-modify-private '
            'playlist-modify-public '
            'user-read-private '
            'user-library-read'
        )
        self.auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=self.scope,
            cache_path='.spotify_cache',
            show_dialog=True
        )
        self.sp = None

    def get_auth_url(self):
        """Get the Spotify authorization URL."""
        return self.auth_manager.get_authorize_url()

    def handle_callback(self, code):
        """Exchange authorization code for access token."""
        token_info = self.auth_manager.get_access_token(code, as_dict=True)
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        return token_info

    def is_authenticated(self):
        """Check if we have a valid cached token."""
        token_info = self.auth_manager.get_cached_token()
        if token_info:
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            if not self._market_auto_detected:
                self._detect_user_market()
            return True
        return False

    def _detect_user_market(self):
        """Auto-detect market from user's Spotify account country."""
        try:
            user = self.sp.current_user()
            country = user.get('country')
            if country:
                self.market = country
                self._market_auto_detected = True
                log.info(f'Spotify market auto-detected: {country}')
        except Exception as e:
            log.warning(f'Could not detect user market, defaulting to '
                        f'{self.market}: {e}')

    def get_current_user(self):
        """Get the current user's profile."""
        if not self.sp:
            return None
        return self.sp.current_user()

    def get_playlists(self):
        """Get all of the current user's playlists."""
        if not self.sp:
            return []
        playlists = []
        results = self.sp.current_user_playlists(limit=50)
        playlists.extend(results['items'])
        while results.get('next'):
            results = self.sp.next(results)
            playlists.extend(results['items'])
        return playlists

    def get_playlist_tracks(self, playlist_id):
        """
        Get all tracks from a playlist with full metadata.
        Returns track data without genre fetching (fast).
        """
        if not self.sp:
            return []

        tracks = []
        results = self.sp.playlist_items(playlist_id, limit=100)
        tracks.extend(results['items'])
        while results.get('next'):
            results = self.sp.next(results)
            tracks.extend(results['items'])

        track_data = []
        for item in tracks:
            if not item or not item.get('track'):
                continue
            t = item['track']
            if not t or not t.get('id'):
                continue

            album = t.get('album', {})
            album_images = album.get('images', [])
            artists = t.get('artists', [])

            track_data.append({
                'id': t['id'],
                'name': t.get('name', 'Unknown'),
                'artist': artists[0]['name'] if artists else 'Unknown',
                'artists': [a['name'] for a in artists],
                'album': album.get('name', ''),
                'album_art': album_images[0]['url'] if album_images else None,
                'album_art_small': (
                    album_images[-1]['url'] if album_images else None
                ),
                'release_date': album.get('release_date', ''),
                'duration_ms': t.get('duration_ms', 0),
                'popularity': t.get('popularity', 0),
                'preview_url': t.get('preview_url'),
                'spotify_url': t.get('external_urls', {}).get('spotify', ''),
                'uri': t.get('uri', ''),
            })
        return track_data

    def get_playlist_summary(self, playlist_id, max_tracks=80):
        """
        Get a lightweight summary of a playlist for AI context.
        Only fetches track names and artists (fast, no extra API calls).
        """
        if not self.sp:
            return {}

        tracks = []
        results = self.sp.playlist_items(
            playlist_id, limit=100,
            fields='items(track(name,artists(name),album(name,release_date))),next'
        )
        tracks.extend(results['items'])
        while results.get('next') and len(tracks) < max_tracks:
            results = self.sp.next(results)
            tracks.extend(results['items'])

        from collections import Counter
        artist_counter = Counter()
        sample_tracks = []

        for item in tracks[:max_tracks]:
            if not item or not item.get('track'):
                continue
            t = item['track']
            if not t:
                continue
            artist_name = t['artists'][0]['name'] if t.get(
                'artists') else 'Unknown'
            track_name = t.get('name', 'Unknown')
            artist_counter[artist_name] += 1
            sample_tracks.append(f"{track_name} by {artist_name}")

        return {
            'track_count': len(tracks),
            'top_artists': [a for a, _ in artist_counter.most_common(10)],
            'sample_tracks': sample_tracks[:15],
        }

    # ─── Fuzzy matching helpers ────────────────────────────────────────────

    def _norm(self, s):
        """Normalize a string for fuzzy comparison: lowercase, strip parentheticals/remaster tags."""
        s = s.lower()
        # remove (feat. ...) / [Remastered] / (Live at ...)
        s = re.sub(r'\(.*?\)|\[.*?\]', '', s)
        # common "edition" words that should not drive matching
        s = re.sub(
            r'\b(remaster(ed)?|deluxe|bonus|version|edit|mix|live|mono|stereo|radio\s*edit|demo)\b',
            '',
            s,
            flags=re.I,
        )
        # normalise common punctuation/abbrev
        s = s.replace('&', 'and')
        # keep only alnums & spaces
        s = re.sub(r'[^a-z0-9 ]', '', s)
        return ' '.join(s.split())                        # collapse whitespace

    def _sim(self, a, b):
        """Similarity ratio (0..1) between two normalised strings."""
        return SequenceMatcher(None, self._norm(a), self._norm(b)).ratio()

    # ─── Track search ────────────────────────────────────────────────────

    def search_track(self, song_name, artist_name):
        """
        Search for a track on Spotify with scored matching.
        Returns the best matching track or None if no good match found.
        """
        if not self.sp or not song_name:
            return None

        # Thresholds balance precision vs recall.  The retrieval pipeline already
        # provides grounded candidates, so we can afford slightly relaxed matching
        # to avoid "exists but not found" failures.
        TITLE_THRESHOLD = 0.65 if artist_name else 0.72
        ARTIST_THRESHOLD = 0.55 if artist_name else 0.0
        KARAOKE_PENALTY = ['karaoke', 'tribute',
                           'in the style of', 'originally performed']

        def _score_candidate(item):
            """Score a Spotify search result against the target song."""
            t_name = item.get('name', '')
            t_artists = [a['name'] for a in item.get('artists', [])]
            title_score = self._sim(song_name, t_name)

            # If the caller provided an artist, compare against each credited artist and
            # take the best match (concatenating can artificially lower the similarity).
            if artist_name:
                artist_score = max((self._sim(artist_name, a)
                                   for a in t_artists), default=0.0)
            else:
                artist_score = 0.80

            # Penalise karaoke / tribute junk
            combined = (t_name + ' ' + ' '.join(t_artists)).lower()
            if any(k in combined for k in KARAOKE_PENALTY):
                # These are almost always wrong for a playlist request.
                return 0.0, 0.0

            return title_score, artist_score

        # ── Search strategy 1: structured query with artist ────────────
        if artist_name:
            # Quotes matter: without quotes, Spotify treats it as token soup.
            query = f'track:"{song_name}" artist:"{artist_name}"'
            try:
                results = self.sp.search(q=query, limit=10, type='track',
                                         market=self.market)
                items = results.get('tracks', {}).get('items', [])
                best, best_score = None, (-1, -1)
                for item in items:
                    ts, as_ = _score_candidate(item)
                    if ts >= TITLE_THRESHOLD and as_ >= ARTIST_THRESHOLD and (ts + as_) > sum(best_score):
                        best, best_score = item, (ts, as_)
                if best:
                    return self._format_track(best)
            except Exception:
                pass

        # ── Search strategy 2: broad text query ────────────────────────
        query = f'{song_name} {artist_name}'.strip()
        try:
            results = self.sp.search(q=query, limit=10, type='track',
                                     market=self.market)
            items = results.get('tracks', {}).get('items', [])
            best, best_score = None, (-1, -1)
            for item in items:
                ts, as_ = _score_candidate(item)
                if ts >= TITLE_THRESHOLD and as_ >= ARTIST_THRESHOLD and (ts + as_) > sum(best_score):
                    best, best_score = item, (ts, as_)
            if best:
                return self._format_track(best)
        except Exception:
            pass

        # ── Search strategy 3: title only (for instrumental/VA tracks) ─
        try:
            results = self.sp.search(
                q=f'track:"{song_name}"', limit=5, type='track',
                market=self.market)
            items = results.get('tracks', {}).get('items', [])
            best, best_score = None, -1
            for item in items:
                ts, _ = _score_candidate(item)
                if ts >= TITLE_THRESHOLD and ts > best_score:
                    best, best_score = item, ts
            if best:
                return self._format_track(best)
        except Exception:
            pass

        return None

    def get_track_by_uri(self, uri):
        """Fetch a track by Spotify URI (spotify:track:<id>) or plain ID."""
        if not self.sp or not uri:
            return None
        track_id = uri
        if uri.startswith('spotify:track:'):
            track_id = uri.split(':')[-1]
        if 'open.spotify.com/track/' in uri:
            track_id = uri.split('/track/')[-1].split('?')[0]
        try:
            t = self.sp.track(track_id, market=self.market)
            return self._format_track(t) if t else None
        except Exception:
            return None

    # ─── Playlist search (for retrieval pipeline) ────────────────────

    def search_playlists(self, query, limit=5):
        """Search Spotify for public playlists matching a query."""
        if not self.sp:
            return []
        try:
            results = self.sp.search(q=query, limit=limit, type='playlist')
            items = results.get('playlists', {}).get('items', [])
            return [
                {'id': p['id'], 'name': p.get('name', ''), 'owner': p.get('owner', {}).get('display_name', ''),
                 'tracks_total': p.get('tracks', {}).get('total', 0),
                 'description': p.get('description', '')}
                for p in items if p
            ]
        except Exception:
            return []

    def get_playlist_track_candidates(self, playlist_id, limit=80):
        """
        Fetch tracks from a playlist and return them as lightweight candidates.
        Returns list of {title, artist, uri, popularity}.
        """
        if not self.sp:
            return []
        try:
            results = self.sp.playlist_items(
                playlist_id, limit=min(limit, 100),
                fields='items(track(id,name,artists(name),album(name,images,release_date),popularity,preview_url,external_urls,uri)),next'
            )
            candidates = []
            for item in results.get('items', []):
                if not item or not item.get('track'):
                    continue
                t = item['track']
                if not t or not t.get('id'):
                    continue
                artists = t.get('artists', [])
                candidates.append({
                    'title': t.get('name', ''),
                    'artist': artists[0]['name'] if artists else '',
                    'uri': t.get('uri', ''),
                    'popularity': t.get('popularity', 0),
                })
                if len(candidates) >= limit:
                    break
            return candidates
        except Exception:
            return []

    def _format_track(self, t):
        """Format a Spotify track object into our standard format."""
        album = t.get('album', {})
        album_images = album.get('images', [])
        artists = t.get('artists', [])
        return {
            'id': t['id'],
            'name': t.get('name', 'Unknown'),
            'artist': artists[0]['name'] if artists else 'Unknown',
            'artists': [a['name'] for a in artists],
            'album': album.get('name', ''),
            'album_art': album_images[0]['url'] if album_images else None,
            'album_art_small': album_images[-1]['url'] if album_images else None,
            'release_date': album.get('release_date', ''),
            'duration_ms': t.get('duration_ms', 0),
            'popularity': t.get('popularity', 0),
            'preview_url': t.get('preview_url'),
            'spotify_url': t.get('external_urls', {}).get('spotify', ''),
            'uri': t.get('uri', ''),
        }

    def create_playlist(self, name, description='Generated by Spotify AI Playlist Manager'):
        """Create a new private playlist."""
        if not self.sp:
            return None
        user = self.sp.current_user()
        # Spotify enforces limits: name ≤ 100 chars, description ≤ 300 chars
        safe_name = (name or 'AI Playlist').strip()[:100]
        safe_desc = (description or '').strip()[:300]
        playlist = self.sp.user_playlist_create(
            user['id'], safe_name, public=False, description=safe_desc
        )
        return playlist

    def add_tracks_to_playlist(self, playlist_id, track_uris):
        """Add tracks to a playlist in batches of 100."""
        if not self.sp:
            return
        # Filter out None/empty URIs
        valid_uris = [u for u in track_uris if u]
        for i in range(0, len(valid_uris), 100):
            chunk = valid_uris[i:i + 100]
            self.sp.playlist_add_items(playlist_id, chunk)

    def rename_playlist(self, playlist_id, name=None, description=None):
        """Rename a playlist and/or update its description."""
        if not self.sp:
            return
        kwargs = {}
        if name is not None:
            kwargs['name'] = (name or 'Untitled').strip()[:100]
        if description is not None:
            kwargs['description'] = (description or '').strip()[:300]
        if kwargs:
            self.sp.playlist_change_details(playlist_id, **kwargs)

    def delete_playlist(self, playlist_id):
        """Unfollow (delete) a playlist."""
        if not self.sp:
            return
        self.sp.current_user_unfollow_playlist(playlist_id)

    def reorder_playlist_tracks(self, playlist_id, track_uris):
        """Replace all tracks in a playlist with the given URIs (reorder)."""
        if not self.sp:
            return
        valid_uris = [u for u in track_uris if u]
        # Replace all items (clears and sets)
        self.sp.playlist_replace_items(playlist_id, valid_uris[:100])
        # If more than 100, add the rest in batches
        for i in range(100, len(valid_uris), 100):
            chunk = valid_uris[i:i + 100]
            self.sp.playlist_add_items(playlist_id, chunk)
