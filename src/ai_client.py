"""
AI Client — Multi-provider AI for the Spotify Playlist Manager.

OpenAI  → Responses API  (web_search tool, structured outputs, reasoning.effort)
Gemini  → google-genai SDK (unchanged)

Architecture principles:
- The LLM is an intent extractor, query planner, and reranker — NOT a song database.
- Every generation mode is retrieval-grounded: the model never "names songs from memory"
  as the primary pathway.  Instead it uses web_search + Spotify retrieval.
- Structured Outputs (JSON Schema via text.format) guarantee parseable responses.
- Session-level tracking of previously recommended songs to avoid repeats.
"""

import json
import logging
import re
from openai import OpenAI

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

log = logging.getLogger(__name__)

# ─── Available models ────────────────────────────────────────────────────────

OPENAI_MODELS = [
    {
        'id': 'gpt-5.2',
        'name': 'GPT-5.2',
        'provider': 'openai',
        'description': 'Flagship model, expensive'
    },
    # {
    #     'id': 'gpt-5.2-pro',
    #     'name': 'GPT-5.2 Pro',
    #     'provider': 'openai',
    #     'description': 'Smarter, more precise responses (Pro plan)'
    # },
    {
        'id': 'gpt-5-mini',
        'name': 'GPT-5 Mini',
        'provider': 'openai',
        'description': 'Fast, cost-efficient version'
    },
    {
        'id': 'gpt-5-nano',
        'name': 'GPT-5 Nano',
        'provider': 'openai',
        'description': 'Fastest & cheapest variant'
    },
]

GEMINI_MODELS = [
    {
        'id': 'gemini-3-pro-preview',
        'name': 'Gemini 3 Pro',
        'provider': 'gemini',
        'description': 'Google flagship. Best for large context'
    },
    {
        'id': 'gemini-3-flash-preview',
        'name': 'Gemini 3 Flash',
        'provider': 'gemini',
        'description': 'Fast & efficient with built-in thinking'
    },
    {
        'id': 'gemini-2.5-flash',
        'name': 'Gemini 2.5 Flash',
        'provider': 'gemini',
        'description': 'Previous gen. fast and affordable'
    },
]

# ─── JSON Schemas for Structured Outputs ─────────────────────────────────────

SCHEMA_PLAYLIST = {
    "type": "json_schema",
    "name": "playlist_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ready", "clarify"]
            },
            "message": {
                "type": "string",
                "description": "Short message about the playlist or a clarification question"
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggestion bubbles when status is clarify (empty array when ready)"
            },
            "playlist": {
                "type": ["object", "null"],
                "description": "The playlist object (null when status is clarify)",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "songs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "artist": {"type": "string"}
                            },
                            "required": ["title", "artist"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["name", "description", "songs"],
                "additionalProperties": False
            }
        },
        "required": ["status", "message", "suggestions", "playlist"],
        "additionalProperties": False
    }
}

SCHEMA_SEARCH_QUERIES = {
    "type": "json_schema",
    "name": "search_queries",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "8-12 short Spotify search queries"
            }
        },
        "required": ["queries"],
        "additionalProperties": False
    }
}

SCHEMA_WEB_DISCOVERY = {
    "type": "json_schema",
    "name": "web_discovery",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "songs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "artist": {"type": "string"}
                    },
                    "required": ["title", "artist"],
                    "additionalProperties": False
                },
                "description": "Songs found via web research"
            },
            "search_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Spotify playlist search queries to find similar real tracks"
            }
        },
        "required": ["songs", "search_queries"],
        "additionalProperties": False
    }
}

SCHEMA_PICK_FROM_CANDIDATES = {
    "type": "json_schema",
    "name": "pick_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "playlist": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "songs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "artist": {"type": "string"},
                                "uri": {"type": "string"}
                            },
                            "required": ["title", "artist", "uri"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["name", "description", "songs"],
                "additionalProperties": False
            },
            "message": {"type": "string"}
        },
        "required": ["playlist", "message"],
        "additionalProperties": False
    }
}

SCHEMA_ADD_SONGS = {
    "type": "json_schema",
    "name": "add_songs_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "songs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "artist": {"type": "string"}
                    },
                    "required": ["title", "artist"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["message", "songs"],
        "additionalProperties": False
    }
}

SCHEMA_SORT = {
    "type": "json_schema",
    "name": "sort_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "songs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "artist": {"type": "string"}
                    },
                    "required": ["title", "artist"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["message", "songs"],
        "additionalProperties": False
    }
}

SCHEMA_CHAT = {
    "type": "json_schema",
    "name": "chat_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["message", "sort", "create_playlist"]
            },
            "message": {"type": "string"},
            "playlist_id": {"type": ["string", "null"]},
            "sort_by": {"type": ["string", "null"]},
            "new_name": {"type": ["string", "null"]},
            "playlist": {
                "type": ["object", "null"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "songs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "artist": {"type": "string"}
                            },
                            "required": ["title", "artist"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["name", "description", "songs"],
                "additionalProperties": False
            }
        },
        "required": ["type", "message", "playlist_id", "sort_by", "new_name", "playlist"],
        "additionalProperties": False
    }
}

# ─── System Prompts ──────────────────────────────────────────────────────────

PROMPT_WEB_DISCOVER = """\
You are SpotifyAI, a world-class music curator.

Your task: Use web search to discover songs that match the user's request.
Search the web for curated playlists, "best of" lists, music blog recommendations,
Reddit threads, RateYourMusic lists, and album reviews that match the described vibe.

STRATEGY:
- Search for multiple angles: genre lists, mood playlists, "if you like X" recommendations,
  decade-specific lists, underground/blog picks
- Look for REAL songs with correct titles and primary artists
- Mix well-known tracks with hidden gems and deep cuts
- Find songs from multiple eras within the target space
- Check multiple sources to cross-reference recommendations

After researching, return:
1. A list of songs you found with exact titles and primary artists
2. 8-12 short Spotify search queries that would find playlists containing similar music
   (e.g., "chill indie folk", "90s shoegaze essentials", "lo-fi study beats")

IMPORTANT: Only include songs you are confident actually exist. Use the exact
official title and primary credited artist name."""

PROMPT_WEB_DISCOVER_FROM_PLAYLISTS = """\
You are SpotifyAI, a world-class music curator.

The user has these playlists showing their taste:
{playlist_context}

Your task: Use web search to discover NEW songs they would love but probably haven't heard.
Search the web for recommendations based on their taste profile — look for "if you like X"
lists, similar artist recommendations, genre deep-dives, and curated playlists that match
their musical DNA.

STRATEGY:
- Analyze their taste: identify the genres, moods, and artists they gravitate toward
- Search for recommendations that extend their taste into adjacent territory
- Look for deeper cuts by artists similar to their favorites
- Find songs from the same scenes/movements they enjoy
- DO NOT recommend songs that are likely already in their playlists

After researching, return:
1. A list of NEW songs with exact titles and primary artists
2. 8-12 Spotify search queries to find playlists with similar undiscovered music

IMPORTANT: Only include songs you are confident actually exist."""

PROMPT_PICK_FROM_CANDIDATES = """\
You are SpotifyAI. The user wants a playlist based on this description:

"{prompt}"

Below is a pool of REAL Spotify tracks gathered from public playlists and web research.
Pick the best {count} tracks that match the user's request.
You may ONLY choose from the candidates below — do NOT invent new songs.

IMPORTANT RULES:
- Only choose tracks that clearly match the prompt's vibe/genre/mood
- When the prompt mentions specific examples, prioritize tracks stylistically close
- Prefer studio/original versions unless the prompt asks for live/remix
- Ensure variety: no artist more than twice, mix well-known with deeper cuts
- ALBUM DIVERSITY: avoid picking more than 1-2 songs from the same album. Spread picks across many different albums and artists.
- SHUFFLE THE ORDER: do NOT group songs by the same artist together. Alternate between different artists and styles so consecutive tracks feel varied. Imagine the listener has shuffle on.
- Consider song flow and energy progression
- EXCLUDE any songs from the "previously recommended" list below

{previously_recommended}

Candidates:
{candidates}"""

PROMPT_REFINE = """\
You are SpotifyAI, refining a previously generated playlist based on user feedback.

The user will provide:
- The current playlist with each song's status: "keep" (thumbs up), "remove" (thumbs down), or "neutral"
- Optional additional instructions for how to improve the playlist

Your job:
1. KEEP all "keep" songs in the playlist (do not remove them)
2. REPLACE all "remove" songs with better alternatives that fit the playlist's vibe
3. "neutral" songs can stay or be replaced based on the additional instructions
4. If additional instructions are given, adjust the overall direction
5. Maintain the same approximate playlist size
6. Only recommend REAL songs that exist on Spotify"""

PROMPT_CHAT = """\
You are SpotifyAI, a knowledgeable and friendly music assistant. You're chatting with \
a user about their Spotify library, music taste, and helping them organize their playlists.

You can:
1. Answer questions about their playlists and listening patterns
2. Suggest ways to reorganize or improve their library
3. Recommend music based on conversation
4. Help sort playlists by various criteria
5. Discuss music history, genres, artists, and connections
6. Create new playlists based on conversation

For regular conversation, set type to "message".
When the user asks to sort a playlist, set type to "sort" with the playlist_id and sort_by fields.
When you want to create a playlist, set type to "create_playlist" with a playlist object."""

PROMPT_ADD_SONGS = """\
You are SpotifyAI, a world-class music curator. The user has an existing playlist \
and wants you to suggest additional songs that would fit perfectly.

Given the list of songs currently in the playlist, analyze:
- The overall vibe, mood, and energy
- Genre patterns and sub-genres
- Era/decade tendencies
- Artist connections and similar artists
- The "glue" that holds the playlist together

Then suggest NEW songs that:
1. Are REAL songs that exist on Spotify (never invent)
2. Use the best-known official title and main credited artist
3. Are NOT already in the playlist
4. Include a mix of well-known and hidden gems
5. Would feel natural if shuffled with the existing tracks"""

PROMPT_SORT_BY_POPULARITY = """\
You are SpotifyAI, a music expert with deep knowledge of the music industry.
The user wants to sort their playlist by popularity/mainstream recognition.

Given a list of songs with their titles and artists, rank them from MOST popular/mainstream
to LEAST popular/mainstream. Consider:
- How well-known the artist is globally
- Whether the song was a charting hit
- Cultural impact and recognition
- Streaming numbers (estimate based on your knowledge)

IMPORTANT: Return ALL songs from the input, just reordered. Do not add or remove any songs."""

PROMPT_EXTRACT_QUERIES = """\
You are a music search assistant. Given a user's playlist request, \
generate 8-12 short search queries that would find relevant public Spotify playlists.

Think about: genres, moods, artists, eras, activities, instruments, and themes in the request.
If the user mentions example songs/artists, include at least 2 queries with those names.
Each query should be 2-5 words — the kind of thing someone would type into Spotify's search bar.

Return a JSON object with a single key "queries" containing an array of strings.
Example: {"queries": ["chill indie folk", "90s shoegaze essentials"]}"""


class AIClient:
    """Multi-provider AI client for music-related operations.

    OpenAI  → Responses API  (web_search, structured outputs, reasoning.effort)
    Gemini  → google-genai SDK (unchanged)
    """

    def __init__(self, openai_api_key=None, gemini_api_key=None):
        self.openai_client = None
        self.gemini_client = None
        # Session-level memory of previously recommended songs (prevents repeats)
        self._recommended_songs: set[str] = set()
        # OpenAI safety / cost caps (0 = unlimited)
        self.max_output_tokens = 0
        self.max_tool_calls = 0
        self.reasoning_effort = 'medium'  # low | medium | high
        # Debug: track last request token usage
        self._last_usage = None

        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        if gemini_api_key and HAS_GEMINI:
            self.gemini_client = genai.Client(api_key=gemini_api_key)

    def update_safety_settings(self, max_output_tokens=0, max_tool_calls=0,
                               reasoning_effort='medium'):
        """Update OpenAI cost/safety caps at runtime."""
        self.max_output_tokens = max_output_tokens or 0
        self.max_tool_calls = max_tool_calls or 0
        self.reasoning_effort = reasoning_effort or 'medium'

    # ─── Session memory ──────────────────────────────────────────────────

    def record_recommended(self, songs: list[dict]):
        """Track songs that have been recommended in this session."""
        for s in songs:
            key = f"{s.get('name', s.get('title', ''))}|{s.get('artist', '')}".lower(
            ).strip()
            if key and key != '|':
                self._recommended_songs.add(key)

    def get_previously_recommended_text(self) -> str:
        """Format previously recommended songs for prompt injection."""
        if not self._recommended_songs:
            return "Previously recommended: (none yet)"
        items = sorted(self._recommended_songs)[:100]
        return "Previously recommended (EXCLUDE these):\n" + \
               "\n".join(f"- {s.replace('|', ' by ')}" for s in items)

    def clear_session(self):
        """Clear session state (call when user starts a new generation session)."""
        self._recommended_songs.clear()

    # ─── Provider detection ──────────────────────────────────────────────

    def get_available_models(self, check_access=False):
        """Return models for configured providers only, with capability flags.

        Args:
            check_access: If True, query each provider's model list and mark
                          models that are not accessible as unavailable.
        """
        # Optionally probe provider model lists (/v1/models)
        openai_accessible = None
        gemini_accessible = None
        if check_access:
            if self.openai_client:
                try:
                    listing = self.openai_client.models.list()
                    openai_accessible = {m.id for m in listing.data}
                except Exception:
                    openai_accessible = set()
            if self.gemini_client:
                try:
                    listing = self.gemini_client.models.list()
                    gemini_accessible = {
                        m.name.replace('models/', '') for m in listing}
                except Exception:
                    gemini_accessible = set()

        models = []
        if self.openai_client:
            for m in OPENAI_MODELS:
                entry = {**m, 'supports_web_search': True}
                if openai_accessible is not None:
                    entry['available'] = m['id'] in openai_accessible
                models.append(entry)
        if self.gemini_client:
            for m in GEMINI_MODELS:
                entry = {
                    **m, 'supports_web_search': self._supports_google_search(m['id'])}
                if gemini_accessible is not None:
                    entry['available'] = m['id'] in gemini_accessible
                models.append(entry)
        return models

    def verify_keys(self):
        """Test each configured API key and return status."""
        status = {
            'openai': {'configured': bool(self.openai_client), 'verified': False, 'error': None},
            'gemini': {'configured': bool(self.gemini_client), 'verified': False, 'error': None},
        }
        if self.openai_client:
            try:
                self.openai_client.models.list()
                status['openai']['verified'] = True
            except Exception as e:
                status['openai']['error'] = str(e)[:120]
        if self.gemini_client:
            try:
                self.gemini_client.models.list(config={'page_size': 1})
                status['gemini']['verified'] = True
            except Exception as e:
                status['gemini']['error'] = str(e)[:120]
        return status

    def _get_provider(self, model):
        """Determine provider from model ID."""
        if model.startswith('gemini'):
            return 'gemini'
        return 'openai'

    # ─── OpenAI Responses API ────────────────────────────────────────────

    def _call_openai_responses(self, instructions, user_input, model,
                               schema=None, tools=None, reasoning_effort=None,
                               tool_choice=None, max_output_tokens=None,
                               max_tool_calls=None):
        """Call OpenAI Responses API.

        Args:
            instructions: System-level instructions string
            user_input: User message string or list of message dicts
            model: Model ID
            schema: JSON schema dict for structured outputs (text.format)
            tools: List of tool configs (e.g., [{"type": "web_search"}])
            reasoning_effort: "low", "medium", or "high" (for reasoning models)
            tool_choice: "auto" (default), "required" (force tool use), or "none"
            max_output_tokens: Cap total output tokens (visible + reasoning)
            max_tool_calls: Cap how many tool calls the model can make

        Returns:
            Parsed JSON dict from the model's response, with _web_search_used
            and _web_sources_count metadata when web_search tool was provided.
        """
        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
        }

        # Structured outputs via text.format
        if schema:
            kwargs["text"] = {"format": schema}

        # Tools (web_search, etc.)
        has_web_search = False
        if tools:
            kwargs["tools"] = tools
            has_web_search = any(t.get("type") == "web_search" for t in tools)
            if has_web_search:
                # Request sources so we can verify the search happened
                kwargs["include"] = ["web_search_call.action.sources"]
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # Reasoning effort for reasoning-capable models
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # Safety / cost caps
        if max_output_tokens and max_output_tokens > 0:
            kwargs["max_output_tokens"] = max_output_tokens
        if max_tool_calls and max_tool_calls > 0 and has_web_search:
            kwargs["max_tool_calls"] = max_tool_calls

        # Auto-truncate input to stay within the model's context window
        kwargs["truncation"] = "auto"

        response = self.openai_client.responses.create(**kwargs)

        # ── Inspect output items for web_search_call verification ────
        web_search_used = False
        web_sources = []
        if has_web_search and hasattr(response, 'output'):
            for item in response.output:
                if getattr(item, 'type', '') == 'web_search_call':
                    web_search_used = True
                    status = getattr(item, 'status', 'unknown')
                    action = getattr(item, 'action', None)
                    if action:
                        action_type = getattr(action, 'type', '')
                        queries = getattr(action, 'queries', []) or []
                        sources = getattr(action, 'sources', []) or []
                        web_sources.extend(sources)
                        log.info(f'Web search call: action={action_type}, '
                                 f'queries={queries[:3]}, '
                                 f'sources={len(sources)}, status={status}')
                    else:
                        log.info(f'Web search call: status={status}')

        if has_web_search:
            if web_search_used:
                log.info(
                    f'Web search CONFIRMED — {len(web_sources)} total sources')
            else:
                log.warning(
                    'Web search tool was provided but model did NOT use it')

        content = response.output_text
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            json_match = re.search(r'\{[\s\S]*\}', content or '')
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {'status': 'error',
                              'message': 'AI returned invalid JSON. Please try again.'}
            else:
                result = {'status': 'error',
                          'message': 'AI returned invalid JSON. Please try again.'}

        # Attach web search metadata for upstream logging
        if has_web_search:
            result['_web_search_used'] = web_search_used
            result['_web_sources_count'] = len(web_sources)

        # Token usage tracking
        usage_data = None
        if hasattr(response, 'usage') and response.usage:
            usage_data = {
                'input_tokens': getattr(response.usage, 'input_tokens', 0),
                'output_tokens': getattr(response.usage, 'output_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
            }
            result['_usage'] = usage_data
            self._last_usage = usage_data
            log.info(f'Token usage: in={usage_data["input_tokens"]}, '
                     f'out={usage_data["output_tokens"]}, '
                     f'total={usage_data["total_tokens"]}')

        return result

    # ─── Gemini ──────────────────────────────────────────────────────────

    def _call_gemini(self, system_prompt, messages, model, temperature=None,
                     google_search=False):
        """Call Google Gemini API using the google-genai SDK.

        Args:
            google_search: If True, enables Grounding with Google Search.
        """
        contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg['content'])]
            ))

        config_kwargs = {
            'system_instruction': system_prompt,
            'response_mime_type': 'application/json',
            'temperature': 0.3 if temperature is None else temperature,
        }
        if google_search:
            # SAFETY: Google Search + JSON mode in the same call causes a
            # Gemini 400.  If caller accidentally requests both, remove
            # response_mime_type and log a warning.
            log.warning('_call_gemini: google_search=True forces removal '
                        'of response_mime_type to avoid Gemini 400')
            del config_kwargs['response_mime_type']
            config_kwargs['tools'] = [
                types.Tool(google_search=types.GoogleSearch())
            ]

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.gemini_client.models.generate_content(
            model=model, contents=contents, config=config,
        )

        # Log grounding metadata if Google Search was used
        if google_search and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            gm = getattr(candidate, 'grounding_metadata', None)
            if gm:
                queries = getattr(gm, 'web_search_queries', []) or []
                chunks = getattr(gm, 'grounding_chunks', []) or []
                log.info(f'Gemini grounding: {len(queries)} queries, '
                         f'{len(chunks)} sources')
            else:
                log.warning('Gemini Google Search enabled but no grounding '
                            'metadata returned')

        content = response.text
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content or '')
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {'status': 'error',
                              'message': 'AI returned invalid JSON. Please try again.'}
            else:
                result = {'status': 'error',
                          'message': 'AI returned invalid JSON. Please try again.'}
        # Safety: ensure result is always a dict (model may return a JSON array)
        if isinstance(result, list):
            log.warning(
                f'Gemini returned a JSON array instead of object — wrapping')
            result = {'songs': result, 'search_queries': []}
        return result

    def _is_gemini_3(self, model):
        """Gemini 3 uses thinking_level; Gemini 2.5 uses thinking_budget."""
        return model.startswith('gemini-3')

    def _call_gemini_thinking(self, system_prompt, messages, model,
                              thinking_level='high', google_search=False):
        """Call Gemini with thinking/reasoning enabled.

        Args:
            google_search: If True, enables Grounding with Google Search.
        """
        contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg['content'])]
            ))

        thinking_kwargs = {}
        if self._is_gemini_3(model):
            # Gemini 3 Pro only supports 'low' and 'high' (not 'medium')
            if 'pro' in model and thinking_level == 'medium':
                thinking_kwargs['thinking_level'] = 'high'
            else:
                thinking_kwargs['thinking_level'] = thinking_level
        else:
            budget_map = {'low': 1024, 'medium': 4096, 'high': 12288}
            thinking_kwargs['thinking_budget'] = budget_map.get(
                thinking_level, 8192)

        config_kwargs = {
            'system_instruction': system_prompt,
            'response_mime_type': 'application/json',
            'thinking_config': types.ThinkingConfig(**thinking_kwargs),
        }
        if google_search:
            # SAFETY: Google Search + JSON mode in the same call causes a
            # Gemini 400.  Remove response_mime_type if tools are present.
            log.warning('_call_gemini_thinking: google_search=True forces '
                        'removal of response_mime_type to avoid Gemini 400')
            del config_kwargs['response_mime_type']
            config_kwargs['tools'] = [
                types.Tool(google_search=types.GoogleSearch())
            ]

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.gemini_client.models.generate_content(
            model=model, contents=contents, config=config,
        )

        content = response.text
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content or '')
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = None
            else:
                result = None
            if result is None:
                return {'status': 'error',
                        'message': 'AI returned invalid JSON. Please try again.'}
        # Safety: ensure result is always a dict (model may return a JSON array)
        if isinstance(result, list):
            log.warning(
                f'Gemini thinking returned a JSON array instead of object — wrapping')
            result = {'songs': result, 'search_queries': []}
        return result

    # ─── Gemini two-step grounded discovery ──────────────────────────────

    def _supports_google_search(self, model):
        """Check if the Gemini model supports Google Search grounding.

        As of 2026-02, only Gemini 2.5 and 2.0 stable models are supported.
        Gemini 3 preview models are NOT yet supported for Google Search.
        """
        if model.startswith('gemini-3'):
            return False
        return True

    def _gemini_grounded_discovery(self, prompt, system, model):
        """Two-step Gemini web search to avoid 'JSON mode + tools' 400.

        Step 1: Call Gemini WITH Google Search grounding but WITHOUT
                response_mime_type='application/json' (plain text output).
        Step 2: Send that text to a second Gemini call WITH JSON mode
                but WITHOUT tools, to parse into SCHEMA_WEB_DISCOVERY format.

        If the model doesn't support Google Search, falls back to a plain
        Gemini call (no web search).
        """
        # If model doesn't support Google Search, just do a plain generation
        if not self._supports_google_search(model):
            log.info(f'Model {model} does not support Google Search grounding, '
                     f'using plain generation instead')
            messages = [{'role': 'user', 'content': prompt}]
            result = self._call_gemini(system, messages, model,
                                       temperature=0.5, google_search=False)
            # Ensure it has the expected shape
            if 'songs' not in result:
                result = {'songs': [], 'search_queries': []}
            return result

        # ── Step 1: grounded search (plain text output) ──────────────
        contents = [types.Content(
            role='user',
            parts=[types.Part.from_text(text=prompt)]
        )]
        search_config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.5,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            # NO response_mime_type — let the model return free-form text
        )
        search_response = None
        try:
            search_response = self.gemini_client.models.generate_content(
                model=model, contents=contents, config=search_config,
            )
            search_text = search_response.text or ''
        except Exception as e:
            log.warning(f'Gemini grounded search failed: {e}')
            search_text = ''

        # Log grounding metadata
        if search_response and hasattr(search_response, 'candidates') and search_response.candidates:
            gm = getattr(search_response.candidates[0],
                         'grounding_metadata', None)
            if gm:
                queries = getattr(gm, 'web_search_queries', []) or []
                chunks = getattr(gm, 'grounding_chunks', []) or []
                log.info(f'Gemini grounding: {len(queries)} queries, '
                         f'{len(chunks)} sources')

        if not search_text.strip():
            log.warning('Gemini grounded search returned empty text')
            return {'songs': [], 'search_queries': []}

        # ── Step 2: parse into structured JSON (no tools) ────────────
        parse_instructions = (
            "You are a JSON extractor. The user will give you text containing "
            "song recommendations from a web search. Extract the songs and "
            "search queries into the exact JSON format specified.\n\n"
            "Return a JSON object with:\n"
            "- 'songs': array of objects with 'title' and 'artist' keys\n"
            "- 'search_queries': array of 8-12 short Spotify search query strings"
        )
        parse_messages = [{'role': 'user', 'content': (
            f"Extract song recommendations from this web search result:\n\n"
            f"{search_text}"
        )}]
        return self._call_gemini(
            parse_instructions, parse_messages, model, temperature=0.1,
            google_search=False,
        )

    # ─── Unified dispatch ────────────────────────────────────────────────

    def _call_ai(self, instructions, messages, model, schema=None,
                 tools=None, reasoning_effort=None, temperature=None,
                 google_search=False):
        """Unified AI call — routes to the correct provider.

        For OpenAI: uses Responses API with optional web_search + structured outputs.
        For Gemini: uses google-genai SDK with optional Google Search grounding.
        """
        provider = self._get_provider(model)

        if provider == 'gemini' and self.gemini_client:
            return self._call_gemini(instructions, messages, model,
                                     temperature=temperature,
                                     google_search=google_search)
        elif self.openai_client:
            return self._call_openai_responses(
                instructions=instructions,
                user_input=messages,
                model=model,
                schema=schema,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_output_tokens=self.max_output_tokens,
            )
        else:
            return {'status': 'error',
                    'message': 'No AI provider configured for this model.'}

    def _call_ai_reasoning(self, instructions, messages, model, effort='medium',
                           google_search=False):
        """Call with reasoning/thinking enabled.

        OpenAI: Responses API with reasoning.effort
        Gemini: thinking_config + optional Google Search grounding
        """
        provider = self._get_provider(model)
        if provider == 'gemini' and self.gemini_client:
            return self._call_gemini_thinking(instructions, messages, model,
                                              thinking_level=effort,
                                              google_search=google_search)
        elif self.openai_client:
            return self._call_openai_responses(
                instructions=instructions,
                user_input=messages,
                model=model,
                reasoning_effort=effort,
            )
        return {'status': 'error', 'message': 'No AI provider configured.'}

    # ─── Web Search Discovery ───────────────────────────────────────────

    def web_discover(self, prompt, count=50, model='gpt-5-mini'):
        """Use web search to discover real songs matching the user's request.

        OpenAI: Responses API with web_search tool (forced via tool_choice).
        Gemini: Google Search grounding tool.
        """
        provider = self._get_provider(model)

        if provider == 'openai' and self.openai_client:
            user_msg = (
                f"Find approximately {count} real songs matching this request: {prompt}\n\n"
                f"Search for curated playlists, 'best of' lists, music blogs, Reddit "
                f"recommendations, and review sites. Look for a mix of well-known tracks "
                f"and hidden gems. Cross-reference multiple sources."
            )

            result = self._call_openai_responses(
                instructions=PROMPT_WEB_DISCOVER,
                user_input=user_msg,
                model=model,
                schema=SCHEMA_WEB_DISCOVERY,
                tools=[{"type": "web_search"}],
                reasoning_effort=self.reasoning_effort or 'medium',
                max_output_tokens=self.max_output_tokens,
                max_tool_calls=self.max_tool_calls,
            )
            return result

        elif provider == 'gemini' and self.gemini_client:
            # Gemini: use Google Search grounding (two-step to avoid
            # "JSON mode + tools" 400 error)
            return self._gemini_grounded_discovery(
                prompt=(
                    f"Find approximately {count} real songs matching: {prompt}\n"
                    f"Search the web for curated playlists and recommendation lists. "
                    f"Also provide 8-12 Spotify search queries to find playlists "
                    f"with similar music."
                ),
                system=(
                    "You are SpotifyAI, a world-class music curator. "
                    "Use Google Search to find real songs matching the user's request. "
                    "Search for curated playlists, 'best of' lists, music blog "
                    "recommendations."
                ),
                model=model,
            )

        return {'songs': [], 'search_queries': []}

    def web_discover_from_playlists(self, playlist_context, prompt='',
                                    count=50, model='gpt-5-mini'):
        """Web search discovery based on user's existing playlists.

        OpenAI: web_search tool (forced). Gemini: Google Search grounding.
        """
        provider = self._get_provider(model)
        instructions = PROMPT_WEB_DISCOVER_FROM_PLAYLISTS.replace(
            '{playlist_context}', playlist_context)

        user_msg = (
            f"Based on my taste shown in the playlists above, find approximately "
            f"{count} new songs I would love.\n"
        )
        if prompt:
            user_msg += f"Additional direction: {prompt}\n"
        user_msg += (
            "Search for 'if you like X' recommendations, similar artist lists, "
            "genre deep-dives, and curated playlists. Find hidden gems I haven't heard."
        )

        if provider == 'openai' and self.openai_client:
            return self._call_openai_responses(
                instructions=instructions,
                user_input=user_msg,
                model=model,
                schema=SCHEMA_WEB_DISCOVERY,
                tools=[{"type": "web_search"}],
                reasoning_effort=self.reasoning_effort or 'medium',
                max_output_tokens=self.max_output_tokens,
                max_tool_calls=self.max_tool_calls,
            )
        elif provider == 'gemini' and self.gemini_client:
            # Two-step grounded discovery (avoids JSON mode + tools 400)
            return self._gemini_grounded_discovery(
                prompt=user_msg,
                system=instructions,
                model=model,
            )

        return {'songs': [], 'search_queries': []}

    # ─── Retrieval helpers (Spotify search query extraction & picking) ────

    def extract_search_queries(self, prompt, model='gpt-5-nano'):
        """Extract 8-12 Spotify playlist search queries from a user prompt."""
        messages = [{'role': 'user', 'content': prompt}]
        result = self._call_ai(
            PROMPT_EXTRACT_QUERIES, messages, model,
            schema=SCHEMA_SEARCH_QUERIES
        )
        # Handle Gemini returning array or non-standard format
        if isinstance(result, list):
            return {'queries': [str(q) for q in result]}
        if 'queries' not in result:
            # Try to find any string array in the result
            for val in result.values():
                if isinstance(val, list) and val and isinstance(val[0], str):
                    return {'queries': val}
            return {'queries': []}
        return result

    def pick_from_candidates(self, prompt, candidates, count=30,
                             model='gpt-5-mini'):
        """Pick the best tracks from a pool of real Spotify candidates.

        Uses structured output to guarantee parseable JSON with URIs.
        """
        # Shuffle to avoid always picking the most popular (anti-popularity bias)
        import random
        if len(candidates) > 400:
            random.shuffle(candidates)
            candidates = candidates[:400]
        else:
            candidates = list(candidates)
            random.shuffle(candidates)

        lines = []
        for c in candidates:
            src = c.get('_source_playlist', '')
            q = c.get('_source_query', '')
            parts = [f"pop:{c.get('popularity', 0)}"]
            if src:
                parts.append(f"src:{src}")
            if q:
                parts.append(f"q:{q}")
            meta = " | ".join(parts)
            lines.append(
                f"- {c['title']} — {c['artist']}  [{meta}]  uri:{c['uri']}")
        candidates_text = '\n'.join(lines)

        prev_text = self.get_previously_recommended_text()

        instructions = PROMPT_PICK_FROM_CANDIDATES \
            .replace('{prompt}', prompt) \
            .replace('{count}', str(count)) \
            .replace('{candidates}', candidates_text) \
            .replace('{previously_recommended}', prev_text)

        messages = [
            {'role': 'user',
             'content': f'Pick the best {count} tracks for: {prompt}'}
        ]

        result = self._call_ai(
            instructions, messages, model,
            schema=SCHEMA_PICK_FROM_CANDIDATES,
            reasoning_effort=self.reasoning_effort or 'medium',
        )

        # Normalize Gemini responses that may not match the OpenAI schema
        if 'playlist' not in result and 'songs' in result:
            result = {
                'playlist': {
                    'name': result.get('name', result.get('playlist_name', '')),
                    'description': result.get('description', ''),
                    'songs': result['songs'],
                },
                'message': result.get('message', ''),
            }
        elif 'playlist' not in result:
            # Last resort: look for any song-like array
            for val in result.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    if 'title' in val[0] or 'artist' in val[0]:
                        result = {
                            'playlist': {
                                'name': '',
                                'description': '',
                                'songs': val,
                            },
                            'message': '',
                        }
                        break

        return result

    # ─── Generation methods ──────────────────────────────────────────────

    @staticmethod
    def _normalize_playlist_result(result):
        """Normalize Gemini responses to match SCHEMA_PLAYLIST structure.

        Gemini may return {songs: [...]} instead of {status: 'ready',
        playlist: {songs: [...]}}. This ensures a consistent shape.
        """
        if not isinstance(result, dict):
            return result
        # Already has the expected structure
        if result.get('status') in ('ready', 'clarify'):
            return result
        # Has songs at top level
        if 'songs' in result and isinstance(result['songs'], list):
            return {
                'status': 'ready',
                'message': result.get('message', ''),
                'suggestions': [],
                'playlist': {
                    'name': result.get('name', result.get('playlist_name', '')),
                    'description': result.get('description', ''),
                    'songs': result['songs'],
                },
            }
        # Has playlist but no status
        if 'playlist' in result and isinstance(result.get('playlist'), dict):
            result.setdefault('status', 'ready')
            result.setdefault('message', '')
            result.setdefault('suggestions', [])
            return result
        return result

    def generate_from_prompt(self, prompt, size='medium', model='gpt-5-mini',
                             conversation_history=None, mode='auto'):
        """Generate a playlist from a text prompt (single-pass, model memory).

        This is now used as a FALLBACK or for 'quick' mode only.
        The primary path goes through the retrieval pipeline in app.py.
        """
        size_map = {'small': 15, 'medium': 30, 'large': 50}
        num_songs = size_map.get(size, 30)

        instructions = (
            "You are SpotifyAI, a world-class music curator. Generate a playlist "
            "based on the user's request. Only recommend REAL songs that actually "
            "exist on Spotify. Use the exact official title and primary credited artist.\n\n"
            "Mix well-known tracks with hidden gems. Consider song flow and energy.\n"
            "No artist more than 2-3 times unless specifically asked.\n"
            "ALBUM DIVERSITY: avoid picking more than 1-2 songs from the same album.\n"
            "SHUFFLE THE ORDER: do NOT group songs by the same artist together. "
            "Alternate between different artists and styles so consecutive tracks feel varied.\n\n"
            f"{self.get_previously_recommended_text()}"
        )

        messages = list(conversation_history or [])
        messages.append({
            'role': 'user',
            'content': f'Create a playlist with approximately {num_songs} songs: {prompt}'
        })

        if mode == 'quick':
            result = self._call_ai(instructions, messages, model,
                                   schema=SCHEMA_PLAYLIST,
                                   reasoning_effort='low')
            result = self._normalize_playlist_result(result)
            if result.get('status') == 'clarify':
                messages.append(
                    {'role': 'assistant', 'content': json.dumps(result)})
                messages.append({'role': 'user',
                                 'content': 'Generate the playlist directly without asking questions.'})
                result = self._call_ai(instructions, messages, model,
                                       schema=SCHEMA_PLAYLIST,
                                       reasoning_effort='low')
                result = self._normalize_playlist_result(result)
            return result

        # Auto/thinking: let the model decide
        result = self._call_ai(instructions, messages, model,
                               schema=SCHEMA_PLAYLIST)
        return self._normalize_playlist_result(result)

    def generate_from_playlists(self, playlist_summaries, prompt='',
                                size='medium', model='gpt-5-mini',
                                conversation_history=None, mode='auto'):
        """Generate a playlist based on user's existing playlists (single-pass fallback)."""
        size_map = {'small': 15, 'medium': 30, 'large': 50}
        num_songs = size_map.get(size, 30)

        context_parts = ["Here are the user's selected Spotify playlists:\n"]
        for ps in playlist_summaries:
            context_parts.append(
                f"**{ps['name']}** ({ps['track_count']} tracks)")
            if ps.get('top_artists'):
                context_parts.append(
                    f"   Top artists: {', '.join(ps['top_artists'][:8])}")
            if ps.get('sample_tracks'):
                context_parts.append(
                    f"   Sample tracks: {', '.join(ps['sample_tracks'][:10])}")
            context_parts.append('')

        instructions = (
            "You are SpotifyAI, a world-class music curator. Analyze the user's "
            "existing playlists to understand their taste, then generate a NEW playlist "
            "of songs they would love but probably haven't heard.\n\n"
            "Do NOT repeat songs from their existing playlists.\n"
            "Only recommend REAL songs. Use exact official titles and primary artists.\n"
            "Include a mix of familiar-feeling tracks and exciting discoveries.\n"
            "ALBUM DIVERSITY: avoid picking more than 1-2 songs from the same album.\n"
            "SHUFFLE THE ORDER: do NOT group songs by the same artist together. "
            "Alternate between different artists and styles so consecutive tracks feel varied.\n\n"
            f"{self.get_previously_recommended_text()}"
        )

        messages = list(conversation_history or [])
        user_msg = '\n'.join(context_parts)
        user_msg += f"\nGenerate approximately {num_songs} songs."
        if prompt:
            user_msg += f"\n\nUser's additional instructions: {prompt}"
        messages.append({'role': 'user', 'content': user_msg})

        if mode == 'quick':
            result = self._call_ai(instructions, messages, model,
                                   schema=SCHEMA_PLAYLIST,
                                   reasoning_effort='low')
            result = self._normalize_playlist_result(result)
            if result.get('status') == 'clarify':
                messages.append(
                    {'role': 'assistant', 'content': json.dumps(result)})
                messages.append({'role': 'user',
                                 'content': 'Generate the playlist directly.'})
                result = self._call_ai(instructions, messages, model,
                                       schema=SCHEMA_PLAYLIST,
                                       reasoning_effort='low')
                result = self._normalize_playlist_result(result)
            return result

        result = self._call_ai(instructions, messages, model,
                               schema=SCHEMA_PLAYLIST)
        return self._normalize_playlist_result(result)

    def refine_playlist(self, current_songs, feedback, new_prompt='',
                        model='gpt-5-mini'):
        """Refine a previously generated playlist based on user feedback."""
        songs_text = json.dumps(current_songs, indent=2)
        user_msg = f"Current playlist with user ratings:\n{songs_text}"
        if new_prompt:
            user_msg += f"\n\nAdditional instructions: {new_prompt}"

        messages = [{'role': 'user', 'content': user_msg}]
        return self._call_ai(PROMPT_REFINE, messages, model,
                             schema=SCHEMA_PLAYLIST)

    def chat(self, messages_history, playlist_context=None, model='gpt-5-mini'):
        """Chat with the AI about playlists and music."""
        instructions = PROMPT_CHAT
        if playlist_context:
            instructions += (
                f"\n\nThe user's Spotify library contains these playlists:\n"
                f"{json.dumps(playlist_context, indent=2)}"
            )
        return self._call_ai(instructions, messages_history, model,
                             schema=SCHEMA_CHAT)

    def add_songs(self, current_tracks, count=10, model='gpt-5-mini'):
        """Suggest songs to add to an existing playlist."""
        tracks_text = json.dumps(current_tracks, indent=2)
        messages = [{
            'role': 'user',
            'content': (
                f'Here is my current playlist ({len(current_tracks)} songs):\n\n'
                f'{tracks_text}\n\n'
                f'Suggest exactly {count} new songs that would fit perfectly.'
            )
        }]
        return self._call_ai(PROMPT_ADD_SONGS, messages, model,
                             schema=SCHEMA_ADD_SONGS)

    def sort_by_popularity(self, tracks, model='gpt-5-nano'):
        """Sort tracks by popularity using AI knowledge."""
        tracks_text = json.dumps(tracks, indent=2)
        messages = [{
            'role': 'user',
            'content': (
                f'Sort these {len(tracks)} songs from most popular '
                f'to least popular:\n\n{tracks_text}'
            )
        }]
        return self._call_ai(PROMPT_SORT_BY_POPULARITY, messages, model,
                             schema=SCHEMA_SORT)

    def chat_about_playlist(self, messages_history, playlist_tracks,
                            model='gpt-5-mini'):
        """Chat with AI about a specific playlist's content."""
        track_summary = [{'name': t.get('name', ''), 'artist': t.get(
            'artist', '')} for t in playlist_tracks[:100]]
        instructions = (
            PROMPT_CHAT +
            f"\n\nThe user is currently viewing a playlist with these tracks:\n"
            f"{json.dumps(track_summary, indent=2)}\n\n"
            f"Answer questions about these specific songs and artists. "
            f"Do NOT suggest creating playlists or sorting — just chat about the music."
        )
        return self._call_ai(instructions, messages_history, model,
                             schema=SCHEMA_CHAT)
