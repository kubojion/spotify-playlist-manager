/* ═══════════════════════════════════════════════════════════
   Spotify AI Playlist Manager — Vue 3 Application
   ═══════════════════════════════════════════════════════════ */

try {
    const { createApp, ref, reactive, computed, onMounted, watch, nextTick } = Vue;

    createApp({
        setup() {

            // ═══════════════════ STATE ═══════════════════

            // View management
            const view = ref('loading');       // 'loading' | 'setup' | 'login' | 'app'
            const feature = ref(null);         // 'from-playlists' | 'from-prompt' | 'manage'
            const showSettings = ref(false);

            // User
            const user = ref(null);

            // Models
            const models = ref([
                { id: 'gpt-5.2', name: 'GPT-5.2', provider: 'openai', description: 'Flagship — best quality', supports_web_search: true },
                // GPT-5.2 Pro commented out — very expensive
                // { id: 'gpt-5.2-pro', name: 'GPT-5.2 Pro', provider: 'openai', description: 'Smarter, more precise (Pro plan)', supports_web_search: true },
                { id: 'gpt-5-mini', name: 'GPT-5 Mini', provider: 'openai', description: 'Fast, cost-efficient GPT-5', supports_web_search: true },
                { id: 'gpt-5-nano', name: 'GPT-5 Nano', provider: 'openai', description: 'Fastest, cheapest GPT-5', supports_web_search: true },
            ]);
            const selectedModel = ref('gpt-5-mini');

            // Setup form
            const setupForm = reactive({
                spotifyClientId: '',
                spotifyClientSecret: '',
                openaiApiKey: '',
                geminiApiKey: '',
            });
            const setupLoading = ref(false);
            const setupError = ref('');

            // Settings form
            const settingsForm = reactive({
                model: 'gpt-5-mini',
                spotifyClientId: '',
                spotifyClientSecret: '',
                openaiApiKey: '',
                geminiApiKey: '',
                maxOutputTokens: 0,
                maxToolCalls: 0,
                reasoningEffort: 'medium',
                costPreset: 'med', // 'low' | 'med' | 'high' | 'custom'
            });

            // Cost presets
            const COST_PRESETS = {
                low:  { maxToolCalls: 1, maxOutputTokens: 5000, reasoningEffort: 'low', label: 'Low', cost: '≤$0.05' },
                med:  { maxToolCalls: 2, maxOutputTokens: 12000, reasoningEffort: 'medium', label: 'Med', cost: '≤$0.10' },
                high: { maxToolCalls: 3, maxOutputTokens: 25000, reasoningEffort: 'high', label: 'High', cost: '≤$0.30' },
            };

            // Key verification status
            const keyStatus = reactive({
                openai: { configured: false, verified: false, error: null },
                gemini: { configured: false, verified: false, error: null },
                spotify: { configured: false, verified: false, error: null },
            });
            const keyStatusLoading = ref(false);

            // Playlists
            const playlists = ref([]);
            const playlistsLoading = ref(false);
            const selectedPlaylists = ref([]);  // Feature 1: multi-select

            // Generation
            const prompt = ref('');
            const playlistSize = ref('medium');
            const generationMode = ref('auto'); // 'auto' | 'quick' | 'thinking'
            const isGenerating = ref(false);
            const generatingMessage = ref('Crafting your perfect playlist...');
            const conversationHistory = ref([]);

            // AI Bubbles
            const showBubbles = ref(false);
            const bubbleSuggestions = ref([]);
            const selectedBubbles = ref([]);
            const aiMessage = ref('');
            const customAnswer = ref('');
            const showCustomAnswerInput = ref(false);
            const hideOriginalPrompt = ref(false);

            // Chat-style conversation (NEW)
            const conversationMessages = ref([]);  // Array of {id, role, content, playlist, ratings, timestamp}
            let messageIdCounter = 0;
            const selectedMessageId = ref(null);  // Currently viewed playlist
            const maxPlaylistVersions = 5;

            // Generated playlist (right panel)
            const generatedPlaylist = ref(null);
            const songRatings = ref({});  // index -> 'up' | 'down' | null
            const showPlaylistPanel = ref(false);  // Control right panel visibility

            // Refine
            const showRefineInput = ref(false);
            const refinePrompt = ref('');
            const isRefining = ref(false);

            // Save
            const isSaving = ref(false);
            const savedPlaylistIds = ref(new Set()); // Track which message IDs have been saved

            // Debug mode
            const debugMode = ref(true);
            const lastDebugInfo = ref(null); // { usage: { input_tokens, output_tokens, total_tokens } }

            // Audio
            const audioPlayer = ref(null);
            const currentlyPlayingSong = ref(null);
            const currentlyPlayingIndex = ref(-1);
            const isPlaying = ref(false);
            const playbackProgress = ref(0);

            // Chat (Feature 3)
            const chatMessages = ref([]);
            const chatInput = ref('');
            const isChatLoading = ref(false);
            const chatContainer = ref(null);

            // Managed playlist (Feature 3)
            const managedPlaylist = ref(null);
            const manageAction = ref(null); // 'chat' | 'sort' | 'add-songs' | null
            const manageChatMessages = ref([]);
            const manageChatInput = ref('');
            const isManageChatLoading = ref(false);
            const manageSortedPlaylist = ref(null); // sorted result for right panel
            const manageAddedSongs = ref(null); // added songs result for right panel
            const isManageSorting = ref(false);
            const isManageAddingSongs = ref(false);
            const manageAddSongsCount = ref(10); // 5 | 10 | 15
            
            // Playlist sorting
            const playlistSortBy = ref('default'); // 'default', 'name', 'tracks'
            const showOnlyMyPlaylists = ref(false); // Filter by owner

            // Model dropdown
            const showModelDropdown = ref(false);

            // Per-feature state cache (preserves state across tab switches)
            const featureStateCache = {};

            // Chat history
            const chatHistories = ref(JSON.parse(localStorage.getItem('chatHistories') || '[]'));
            const currentChatId = ref(Date.now());
            const showHistoryFor = ref(null); // 'from-playlists' | 'from-prompt' | null
            const historyContextMenu = ref({ show: false, x: 0, y: 0, historyId: null });
            const renamingHistoryId = ref(null);
            const renamingHistoryName = ref('');

            // Playlist context menu (right-click on sidebar playlists)
            const playlistContextMenu = ref({ show: false, x: 0, y: 0, playlist: null });
            const isRenamingManagedPlaylist = ref(false);
            const editingPlaylistName = ref('');
            const editingPlaylistDesc = ref('');

            // Chat scroll
            const chatScrollContainer = ref(null);

            // Panel resize
            const leftPanelWidth = ref(320);  // wider default for playlists
            const rightPanelWidth = ref(420);
            let isResizingLeft = false;
            let isResizingRight = false;

            // Toasts
            const toasts = ref([]);
            let toastId = 0;

            // Sort options for manage
            const manageSortOptions = [
                { value: 'release_date', label: 'Oldest first' },
                { value: 'release_date_desc', label: 'Newest first' },
                { value: 'popularity', label: 'Popularity (Spotify)' },
                { value: 'name', label: 'Name A → Z' },
                { value: 'popularity_ai', label: 'Popularity (AI)' },
            ];


            // ═══════════════════ COMPUTED HELPERS ═══════════════════

            const playlistSelectionLabel = computed(() => {
                if (feature.value === 'from-playlists') {
                    const count = selectedPlaylists.value.length;
                    return count + ' playlist' + (count !== 1 ? 's' : '') + ' selected';
                }
                return 'Describe your perfect playlist';
            });

            const promptPlaceholder = computed(() => {
                if (feature.value === 'from-playlists') {
                    return 'Add instructions (optional)... e.g., "more upbeat", "70s rock vibes", "perfect for a road trip"';
                }
                return 'Describe your playlist... e.g., "symphonic rock with orchestra like Metallica S&M"';
            });

            const foundSongsCount = computed(() => {
                if (!generatedPlaylist.value) return 0;
                return generatedPlaylist.value.songs.filter(s => !s.not_found).length;
            });

            const notFoundSongsCount = computed(() => {
                if (!generatedPlaylist.value) return 0;
                return generatedPlaylist.value.songs.filter(s => s.not_found).length;
            });

            const isCurrentPlaylistSaved = computed(() => {
                return selectedMessageId.value && savedPlaylistIds.value.has(selectedMessageId.value);
            });
            
            const sortedPlaylists = computed(() => {
                let list = [...playlists.value];
                // Filter by owner if enabled
                if (showOnlyMyPlaylists.value && user.value) {
                    list = list.filter(p => p.owner === user.value.display_name);
                }
                if (playlistSortBy.value === 'name') {
                    return list.sort((a, b) => a.name.localeCompare(b.name));
                } else if (playlistSortBy.value === 'tracks') {
                    return list.sort((a, b) => b.track_count - a.track_count);
                } else if (playlistSortBy.value === 'name_desc') {
                    return list.sort((a, b) => b.name.localeCompare(a.name));
                } else if (playlistSortBy.value === 'tracks_asc') {
                    return list.sort((a, b) => a.track_count - b.track_count);
                } else if (playlistSortBy.value === 'owner') {
                    return list.sort((a, b) => a.owner.localeCompare(b.owner));
                }
                return list; // default order from Spotify
            });


            // ═══════════════════ API HELPERS ═══════════════════

            async function apiGet(url) {
                const res = await fetch(url);
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
                return data;
            }

            async function apiPost(url, body) {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
                return data;
            }

            async function apiPut(url, body) {
                const res = await fetch(url, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
                return data;
            }

            async function apiDelete(url) {
                const res = await fetch(url, { method: 'DELETE' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
                return data;
            }


            // ═══════════════════ SETUP & AUTH ═══════════════════

            async function checkStatus() {
                try {
                    const data = await apiGet('/api/status');
                    if (!data.configured) {
                        view.value = 'setup';
                    } else if (!data.authenticated) {
                        view.value = 'login';
                    } else {
                        user.value = data.user;
                        selectedModel.value = data.preferred_model || 'gpt-5-mini';
                        settingsForm.model = selectedModel.value;
                        // Load safety / cost settings
                        settingsForm.maxOutputTokens = data.max_output_tokens || 0;
                        settingsForm.maxToolCalls = data.max_tool_calls || 0;
                        settingsForm.reasoningEffort = data.reasoning_effort || 'medium';
                        settingsForm.costPreset = data.cost_preset || 'med';
                        view.value = 'app';
                        feature.value = 'from-prompt';
                        fetchPlaylists();
                        fetchModels(true);  // check access on initial load
                        verifyKeys();
                    }
                } catch (err) {
                    view.value = 'setup';
                }
            }

            async function submitSetup() {
                if (!setupForm.spotifyClientId || !setupForm.spotifyClientSecret || (!setupForm.openaiApiKey && !setupForm.geminiApiKey)) {
                    setupError.value = 'Spotify keys required, and at least one AI key (OpenAI or Gemini)';
                    return;
                }
                setupLoading.value = true;
                setupError.value = '';
                try {
                    await apiPost('/api/setup', {
                        spotify_client_id: setupForm.spotifyClientId,
                        spotify_client_secret: setupForm.spotifyClientSecret,
                        openai_api_key: setupForm.openaiApiKey,
                        gemini_api_key: setupForm.geminiApiKey,
                    });
                    addToast('Configuration saved!', 'success');
                    view.value = 'login';
                } catch (err) {
                    setupError.value = err.message;
                } finally {
                    setupLoading.value = false;
                }
            }

            async function loginSpotify() {
                try {
                    const data = await apiGet('/api/auth/login');
                    window.location.href = data.auth_url;
                } catch (err) {
                    addToast('Failed to start login: ' + err.message, 'error');
                }
            }

            async function logoutSpotify() {
                try {
                    await apiPost('/api/auth/logout', {});
                    addToast('Logged out', 'success');
                    closeSettings();
                    view.value = 'login';
                    user.value = null;
                } catch (err) {
                    addToast('Logout failed', 'error');
                }
            }

            function closeSettings() {
                showSettings.value = false;
                if (document.activeElement && typeof document.activeElement.blur === 'function') {
                    document.activeElement.blur();
                }
            }

            async function verifyKeys() {
                keyStatusLoading.value = true;
                try {
                    const data = await apiGet('/api/verify-keys');
                    if (data.openai) {
                        keyStatus.openai.configured = data.openai.configured;
                        keyStatus.openai.verified = data.openai.verified;
                        keyStatus.openai.error = data.openai.error;
                    }
                    if (data.gemini) {
                        keyStatus.gemini.configured = data.gemini.configured;
                        keyStatus.gemini.verified = data.gemini.verified;
                        keyStatus.gemini.error = data.gemini.error;
                    }
                    if (data.spotify) {
                        keyStatus.spotify.configured = data.spotify.configured;
                        keyStatus.spotify.verified = data.spotify.verified;
                        keyStatus.spotify.error = data.spotify.error;
                    }
                } catch {
                    // Silent fail
                } finally {
                    keyStatusLoading.value = false;
                }
            }

            function openSettings() {
                showSettings.value = true;
                settingsForm.model = selectedModel.value;
                verifyKeys();
            }


            // ═══════════════════ DATA FETCHING ═══════════════════

            async function fetchPlaylists() {
                playlistsLoading.value = true;
                try {
                    const data = await apiGet('/api/playlists');
                    playlists.value = data.playlists || [];
                } catch (err) {
                    addToast('Failed to load playlists: ' + err.message, 'error');
                } finally {
                    playlistsLoading.value = false;
                }
            }

            async function fetchModels(checkAccess = false) {
                try {
                    const url = checkAccess ? '/api/models?check_access=true' : '/api/models';
                    const data = await apiGet(url);
                    if (data.models && data.models.length) {
                        // Filter out unavailable models when access was checked
                        const available = checkAccess
                            ? data.models.filter(m => m.available !== false)
                            : data.models;
                        models.value = available.length ? available : data.models;
                        // If currently selected model is not in the available list, auto-select first available
                        if (!models.value.find(m => m.id === selectedModel.value)) {
                            selectedModel.value = models.value[0].id;
                            saveModelPreference();
                        }
                    }
                } catch {
                    // Use defaults
                }
            }

            // Check if current model supports web search
            const modelSupportsWebSearch = computed(() => {
                const m = models.value.find(model => model.id === selectedModel.value);
                return m ? (m.supports_web_search !== false) : true;
            });

            // Auto-switch away from web search mode if model doesn't support it
            watch(() => selectedModel.value, () => {
                if (!modelSupportsWebSearch.value && generationMode.value === 'thinking') {
                    generationMode.value = 'auto';
                }
            });

            function applyCostPreset(preset) {
                settingsForm.costPreset = preset;
                if (preset !== 'custom' && COST_PRESETS[preset]) {
                    const p = COST_PRESETS[preset];
                    settingsForm.maxToolCalls = p.maxToolCalls;
                    settingsForm.maxOutputTokens = p.maxOutputTokens;
                    settingsForm.reasoningEffort = p.reasoningEffort;
                }
            }


            // ═══════════════════ FEATURE SWITCHING ═══════════════════

            function setFeature(f) {
                if (feature.value === f) return;
                saveChatToHistory();

                // ── Save current feature state ──
                const cur = feature.value;
                if (cur === 'from-playlists' || cur === 'from-prompt') {
                    featureStateCache[cur] = {
                        conversationMessages: JSON.parse(JSON.stringify(conversationMessages.value)),
                        conversationHistory: JSON.parse(JSON.stringify(conversationHistory.value)),
                        selectedMessageId: selectedMessageId.value,
                        generatedPlaylist: generatedPlaylist.value ? JSON.parse(JSON.stringify(generatedPlaylist.value)) : null,
                        songRatings: { ...songRatings.value },
                        showPlaylistPanel: showPlaylistPanel.value,
                        showBubbles: showBubbles.value,
                        bubbleSuggestions: [...bubbleSuggestions.value],
                        selectedBubbles: [...selectedBubbles.value],
                        aiMessage: aiMessage.value,
                        hideOriginalPrompt: hideOriginalPrompt.value,
                        customAnswer: customAnswer.value,
                        showCustomAnswerInput: showCustomAnswerInput.value,
                        currentChatId: currentChatId.value,
                        savedPlaylistIds: new Set(savedPlaylistIds.value),
                    };
                } else if (cur === 'manage') {
                    featureStateCache['manage'] = {
                        managedPlaylist: managedPlaylist.value ? JSON.parse(JSON.stringify(managedPlaylist.value)) : null,
                        manageAction: manageAction.value,
                        manageChatMessages: JSON.parse(JSON.stringify(manageChatMessages.value)),
                        manageChatInput: manageChatInput.value,
                        manageSortedPlaylist: manageSortedPlaylist.value ? JSON.parse(JSON.stringify(manageSortedPlaylist.value)) : null,
                        manageAddedSongs: manageAddedSongs.value ? JSON.parse(JSON.stringify(manageAddedSongs.value)) : null,
                    };
                }

                feature.value = f;

                // ── Restore target feature state ──
                const cached = featureStateCache[f];
                if (cached && (f === 'from-playlists' || f === 'from-prompt')) {
                    conversationMessages.value = cached.conversationMessages;
                    conversationHistory.value = cached.conversationHistory;
                    selectedMessageId.value = cached.selectedMessageId;
                    generatedPlaylist.value = cached.generatedPlaylist;
                    songRatings.value = cached.songRatings;
                    showPlaylistPanel.value = cached.showPlaylistPanel;
                    showBubbles.value = cached.showBubbles;
                    bubbleSuggestions.value = cached.bubbleSuggestions;
                    selectedBubbles.value = cached.selectedBubbles;
                    aiMessage.value = cached.aiMessage;
                    hideOriginalPrompt.value = cached.hideOriginalPrompt;
                    customAnswer.value = cached.customAnswer;
                    showCustomAnswerInput.value = cached.showCustomAnswerInput;
                    currentChatId.value = cached.currentChatId;
                    savedPlaylistIds.value = cached.savedPlaylistIds;
                } else if (cached && f === 'manage') {
                    managedPlaylist.value = cached.managedPlaylist;
                    manageAction.value = cached.manageAction;
                    manageChatMessages.value = cached.manageChatMessages;
                    manageChatInput.value = cached.manageChatInput;
                    manageSortedPlaylist.value = cached.manageSortedPlaylist;
                    manageAddedSongs.value = cached.manageAddedSongs;
                } else {
                    // No cached state — reset to defaults
                    currentChatId.value = Date.now();
                    showBubbles.value = false;
                    bubbleSuggestions.value = [];
                    selectedBubbles.value = [];
                    aiMessage.value = '';
                    conversationHistory.value = [];
                    customAnswer.value = '';
                    showCustomAnswerInput.value = false;
                    hideOriginalPrompt.value = false;
                    conversationMessages.value = [];
                    selectedMessageId.value = null;
                    generatedPlaylist.value = null;
                    songRatings.value = {};
                    showPlaylistPanel.value = false;
                    if (f === 'manage') {
                        managedPlaylist.value = null;
                        chatMessages.value = [];
                        manageAction.value = null;
                        manageChatMessages.value = [];
                        manageChatInput.value = '';
                        manageSortedPlaylist.value = null;
                        manageAddedSongs.value = null;
                    }
                }
            }


            // ═══════════════════ PLAYLIST SELECTION (Feature 1) ═══════════════════

            function togglePlaylistSelection(id) {
                const idx = selectedPlaylists.value.indexOf(id);
                if (idx >= 0) {
                    selectedPlaylists.value.splice(idx, 1);
                } else {
                    selectedPlaylists.value.push(id);
                }
            }

            function isPlaylistSelected(id) {
                return selectedPlaylists.value.includes(id);
            }


            // ═══════════════════ GENERATION ═══════════════════

            async function generate() {
                if (isGenerating.value) return;

                // Validation
                if (feature.value === 'from-playlists' && !selectedPlaylists.value.length) {
                    addToast('Select at least one playlist', 'error');
                    return;
                }
                if (feature.value === 'from-prompt' && !prompt.value.trim() && !selectedBubbles.value.length && !customAnswer.value.trim()) {
                    addToast('Enter a prompt or select suggestions', 'error');
                    return;
                }

                isGenerating.value = true;
                const isThinking = generationMode.value !== 'quick';
                const isSpotifyOnly = generationMode.value === 'spotify-only';
                generatingMessage.value = feature.value === 'from-playlists'
                    ? (isSpotifyOnly ? 'Searching Spotify for songs matching your taste...'
                        : isThinking ? 'Searching the web & Spotify for songs matching your taste...' : 'Analyzing your playlists...')
                    : (isSpotifyOnly ? 'Searching Spotify for perfect tracks...'
                        : isThinking ? 'Searching the web & Spotify for perfect tracks...' : 'Crafting your perfect playlist...');

                // Build prompt with selected bubbles and custom answer
                let fullPrompt = prompt.value;
                if (selectedBubbles.value.length) {
                    fullPrompt = (fullPrompt ? fullPrompt + '. ' : '') +
                        'Focus on: ' + selectedBubbles.value.join(', ');
                }
                if (customAnswer.value.trim()) {
                    fullPrompt = (fullPrompt ? fullPrompt + '. ' : '') + customAnswer.value.trim();
                }

                // Show user message in chat immediately (before API call)
                if (fullPrompt.trim()) {
                    const userMsgId = ++messageIdCounter;
                    conversationMessages.value.push({
                        id: userMsgId,
                        role: 'user',
                        content: fullPrompt,
                        playlist: null,
                        ratings: {},
                        timestamp: Date.now(),
                    });
                    prompt.value = '';
                    customAnswer.value = '';
                    selectedBubbles.value = [];
                    showBubbles.value = false;
                    scrollChatToBottom();
                }

                try {
                    let result;
                    if (feature.value === 'from-playlists') {
                        generatingMessage.value = isThinking
                            ? 'Discovering songs and building candidate pool...'
                            : 'Reading your playlists and finding perfect songs...';
                        result = await apiPost('/api/generate/from-playlists', {
                            playlist_ids: selectedPlaylists.value,
                            prompt: fullPrompt,
                            size: playlistSize.value,
                            model: selectedModel.value,
                            mode: generationMode.value,
                            conversation_history: conversationHistory.value,
                        });
                    } else {
                        generatingMessage.value = isThinking
                            ? 'AI is reranking and curating the final playlist...'
                            : 'AI is curating songs just for you...';
                        result = await apiPost('/api/generate/from-prompt', {
                            prompt: fullPrompt,
                            size: playlistSize.value,
                            model: selectedModel.value,
                            mode: generationMode.value,
                            conversation_history: conversationHistory.value,
                        });
                    }

                    handleGenerationResult(result, fullPrompt);
                } catch (err) {
                    addToast('Generation failed: ' + err.message, 'error');
                } finally {
                    isGenerating.value = false;
                }
            }

            function handleGenerationResult(result, userPrompt) {
                // Build debug/usage info with model, mode, effort context
                const modelObj = models.value.find(m => m.id === selectedModel.value);
                const modelName = modelObj?.name || selectedModel.value;
                const provider = modelObj?.provider === 'gemini' ? 'Gemini' : 'OpenAI';
                const modeLabel = generationMode.value === 'quick' ? 'Quick'
                    : generationMode.value === 'thinking' ? 'Web Search'
                    : generationMode.value === 'spotify-only' ? 'Spotify'
                    : 'Auto';
                const effortLabel = settingsForm.reasoningEffort
                    ? settingsForm.reasoningEffort.charAt(0).toUpperCase() + settingsForm.reasoningEffort.slice(1)
                    : 'Medium';

                const debugInfo = result._debug ? {
                    ...result._debug,
                    modelName,
                    provider,
                    mode: modeLabel,
                    effort: effortLabel,
                } : {
                    usage: null,
                    modelName,
                    provider,
                    mode: modeLabel,
                    effort: effortLabel,
                };
                lastDebugInfo.value = debugInfo;
                
                if (result.status === 'clarify') {
                    // AI wants more info — show bubbles inline in chat
                    showBubbles.value = true;
                    bubbleSuggestions.value = result.suggestions || [];
                    aiMessage.value = result.message || 'Could you tell me more about what you\'re looking for?';
                    selectedBubbles.value = [];

                    // Add clarification to conversation messages (as assistant message)
                    const msgId = ++messageIdCounter;
                    conversationMessages.value.push({
                        id: msgId,
                        role: 'assistant',
                        content: result.message || 'Could you tell me more?',
                        playlist: null,
                        ratings: {},
                        timestamp: Date.now(),
                        debugInfo: debugInfo,
                    });

                    // Add to conversation history for API
                    conversationHistory.value.push(
                        { role: 'user', content: userPrompt },
                        { role: 'assistant', content: JSON.stringify(result) }
                    );
                } else if (result.status === 'ready' && result.playlist) {
                    // Playlist generated!
                    showBubbles.value = false;

                    const playlist = {
                        name: result.playlist.name || 'AI Playlist',
                        description: result.playlist.description || '',
                        songs: result.playlist.songs || [],
                        message: result.message || '',
                    };

                    // Add AI response to conversation (user message was already added in generate())
                    const aiMsgId = ++messageIdCounter;
                    conversationMessages.value.push({
                        id: aiMsgId,
                        role: 'assistant',
                        content: result.message || 'Here\'s your playlist!',
                        playlist: playlist,
                        ratings: {},
                        timestamp: Date.now(),
                        debugInfo: debugInfo,
                    });

                    // Limit to max versions
                    if (conversationMessages.value.length > maxPlaylistVersions * 2) {
                        conversationMessages.value = conversationMessages.value.slice(-maxPlaylistVersions * 2);
                    }

                    // Show this playlist
                    selectConversationMessage(aiMsgId);

                    // Clear state (prompt already cleared in generate())
                    showCustomAnswerInput.value = false;
                    conversationHistory.value = [];
                    showPlaylistPanel.value = true;
                    scrollChatToBottom();
                    saveChatToHistory();
                } else if (result.error) {
                    addToast('AI error: ' + result.error, 'error');
                } else {
                    addToast('Unexpected response from AI. Try again.', 'error');
                }
            }

            function selectConversationMessage(msgId) {
                const msg = conversationMessages.value.find(m => m.id === msgId);
                if (msg && msg.playlist) {
                    selectedMessageId.value = msgId;
                    generatedPlaylist.value = msg.playlist;
                    songRatings.value = { ...msg.ratings };
                    showPlaylistPanel.value = true;
                }
            }

            function startNewChat() {
                saveChatToHistory();
                currentChatId.value = Date.now();
                conversationMessages.value = [];
                conversationHistory.value = [];
                generatedPlaylist.value = null;
                songRatings.value = {};
                selectedMessageId.value = null;
                showPlaylistPanel.value = false;
                prompt.value = '';
                customAnswer.value = '';
                showBubbles.value = false;
                hideOriginalPrompt.value = false;
                showRefineInput.value = false;
                savedPlaylistIds.value = new Set();
            }

            function selectBubble(bubble) {
                const idx = selectedBubbles.value.indexOf(bubble);
                if (idx >= 0) {
                    selectedBubbles.value.splice(idx, 1);
                } else {
                    selectedBubbles.value.push(bubble);
                }
            }


            // ═══════════════════ RESULTS & REFINEMENT ═══════════════════

            function removeSong(index) {
                if (!generatedPlaylist.value) return;
                generatedPlaylist.value.songs.splice(index, 1);
                // Rebuild ratings for shifted indices
                const newRatings = {};
                Object.keys(songRatings.value).forEach(k => {
                    const ki = parseInt(k);
                    if (ki < index) newRatings[ki] = songRatings.value[ki];
                    else if (ki > index) newRatings[ki - 1] = songRatings.value[ki];
                });
                songRatings.value = newRatings;
                // Update the conversation message's playlist too
                if (selectedMessageId.value) {
                    const msg = conversationMessages.value.find(m => m.id === selectedMessageId.value);
                    if (msg && msg.playlist) {
                        msg.playlist.songs = [...generatedPlaylist.value.songs];
                        msg.ratings = { ...songRatings.value };
                    }
                }
            }

            function rateSong(index, rating) {
                if (songRatings.value[index] === rating) {
                    delete songRatings.value[index];
                    songRatings.value = { ...songRatings.value };  // trigger reactivity
                } else {
                    songRatings.value = { ...songRatings.value, [index]: rating };
                }
                
                // Save ratings to current message
                if (selectedMessageId.value) {
                    const msg = conversationMessages.value.find(m => m.id === selectedMessageId.value);
                    if (msg && msg.playlist) {
                        msg.ratings = { ...songRatings.value };
                    }
                }
            }

            async function refinePlaylist() {
                if (isRefining.value || !generatedPlaylist.value) return;
                isRefining.value = true;

                // Build current songs with ratings
                const currentSongs = generatedPlaylist.value.songs.map((song, idx) => ({
                    title: song.name,
                    artist: song.artist,
                    status: songRatings.value[idx] === 'up' ? 'keep' :
                        songRatings.value[idx] === 'down' ? 'remove' : 'neutral',
                }));

                try {
                    const result = await apiPost('/api/generate/refine', {
                        current_songs: currentSongs,
                        prompt: refinePrompt.value,
                        model: selectedModel.value,
                    });

                    if (result.status === 'ready' && result.playlist) {
                        const newPlaylist = {
                            name: result.playlist.name || generatedPlaylist.value.name,
                            description: result.playlist.description || generatedPlaylist.value.description,
                            songs: result.playlist.songs || [],
                            message: result.message || '',
                        };
                        
                        // Add user refinement request and AI response to conversation
                        const userMsgId = ++messageIdCounter;
                        conversationMessages.value.push({
                            id: userMsgId,
                            role: 'user',
                            content: refinePrompt.value || 'Refine the playlist',
                            playlist: null,
                            ratings: {},
                            timestamp: Date.now(),
                        });

                        const aiMsgId = ++messageIdCounter;
                        conversationMessages.value.push({
                            id: aiMsgId,
                            role: 'assistant',
                            content: result.message || 'Here\'s your refined playlist!',
                            playlist: newPlaylist,
                            ratings: {},
                            timestamp: Date.now(),
                        });

                        // Limit versions
                        if (conversationMessages.value.length > maxPlaylistVersions * 2) {
                            conversationMessages.value = conversationMessages.value.slice(-maxPlaylistVersions * 2);
                        }

                        // Show new playlist
                        selectConversationMessage(aiMsgId);
                        
                        showRefineInput.value = false;
                        refinePrompt.value = '';
                        addToast('Playlist refined!', 'success');
                    } else {
                        addToast('Refinement failed. Try again.', 'error');
                    }
                } catch (err) {
                    addToast('Refinement failed: ' + err.message, 'error');
                } finally {
                    isRefining.value = false;
                }
            }

            async function savePlaylistToSpotify() {
                if (isSaving.value || !generatedPlaylist.value) return;
                isSaving.value = true;

                const trackUris = generatedPlaylist.value.songs
                    .filter(s => s.uri && !s.not_found)
                    .map(s => s.uri);

                if (!trackUris.length) {
                    addToast('No valid tracks to save', 'error');
                    isSaving.value = false;
                    return;
                }

                try {
                    const result = await apiPost('/api/save-playlist', {
                        name: generatedPlaylist.value.name,
                        description: generatedPlaylist.value.description,
                        track_uris: trackUris,
                    });

                    if (result.success) {
                        savedPlaylistIds.value = new Set([...savedPlaylistIds.value, selectedMessageId.value]);
                        // Refresh playlists list
                        fetchPlaylists();
                    }
                } catch (err) {
                    addToast('Failed to save: ' + err.message, 'error');
                } finally {
                    isSaving.value = false;
                }
            }

            function discardPlaylist() {
                stopPreview();
                showPlaylistPanel.value = false;
                generatedPlaylist.value = null;
                songRatings.value = {};
                showRefineInput.value = false;
                refinePrompt.value = '';
                selectedMessageId.value = null;
            }


            // ═══════════════════ AUDIO PLAYBACK ═══════════════════

            function playPreview(song, index) {
                if (!song.preview_url) {
                    addToast('No preview available for this track', 'info');
                    return;
                }

                const player = audioPlayer.value;
                if (!player) return;

                // Toggle if same song
                if (currentlyPlayingIndex.value === index && isPlaying.value) {
                    player.pause();
                    isPlaying.value = false;
                    return;
                }

                // Play new song
                player.src = song.preview_url;
                player.volume = 0.5;
                player.play().catch(() => {
                    addToast('Unable to play preview', 'error');
                });
                currentlyPlayingSong.value = song;
                currentlyPlayingIndex.value = index;
                isPlaying.value = true;
            }

            function togglePlayback() {
                const player = audioPlayer.value;
                if (!player || !currentlyPlayingSong.value) return;
                if (isPlaying.value) {
                    player.pause();
                    isPlaying.value = false;
                } else {
                    player.play();
                    isPlaying.value = true;
                }
            }

            function stopPreview() {
                const player = audioPlayer.value;
                if (player) {
                    player.pause();
                    player.currentTime = 0;
                }
                currentlyPlayingSong.value = null;
                currentlyPlayingIndex.value = -1;
                isPlaying.value = false;
                playbackProgress.value = 0;
            }

            function onAudioEnded() {
                isPlaying.value = false;
                playbackProgress.value = 100;
            }

            function onTimeUpdate() {
                const player = audioPlayer.value;
                if (player && player.duration) {
                    playbackProgress.value = (player.currentTime / player.duration) * 100;
                }
            }


            // ═══════════════════ CHAT (Feature 3) ═══════════════════

            async function selectPlaylistForManage(playlist) {
                managedPlaylist.value = { ...playlist, tracks: [] };
                manageAction.value = null;
                manageChatMessages.value = [];
                manageChatInput.value = '';
                manageSortedPlaylist.value = null;
                manageAddedSongs.value = null;
                showPlaylistPanel.value = false;
                
                try {
                    const response = await apiGet(`/api/playlists/${playlist.id}/tracks`);
                    managedPlaylist.value.tracks = response.tracks || [];
                } catch (err) {
                    addToast('Failed to load playlist tracks', 'error');
                    console.error('Error loading tracks:', err);
                }
            }

            function setManageAction(action) {
                manageAction.value = action;
                manageChatMessages.value = [];
                manageChatInput.value = '';
                manageSortedPlaylist.value = null;
                manageAddedSongs.value = null;
                showPlaylistPanel.value = false;
            }

            async function sendManageChat() {
                const msg = manageChatInput.value.trim();
                if (!msg || isManageChatLoading.value || !managedPlaylist.value) return;

                manageChatMessages.value.push({ role: 'user', content: msg });
                manageChatInput.value = '';
                isManageChatLoading.value = true;
                scrollManageChatToBottom();

                const apiMessages = manageChatMessages.value.map(m => ({
                    role: m.role,
                    content: m.content,
                }));

                try {
                    const result = await apiPost('/api/manage/chat', {
                        messages: apiMessages,
                        playlist_tracks: managedPlaylist.value.tracks || [],
                        model: selectedModel.value,
                    });

                    manageChatMessages.value.push({
                        role: 'assistant',
                        content: result.message || JSON.stringify(result),
                    });
                } catch (err) {
                    manageChatMessages.value.push({
                        role: 'assistant',
                        content: 'Sorry, I encountered an error: ' + err.message,
                    });
                } finally {
                    isManageChatLoading.value = false;
                    scrollManageChatToBottom();
                }
            }

            function scrollManageChatToBottom() {
                nextTick(() => {
                    const el = document.getElementById('manage-chat-scroll');
                    if (el) el.scrollTop = el.scrollHeight;
                });
            }

            async function manageSortPlaylist(sortBy) {
                if (!managedPlaylist.value || isManageSorting.value) return;
                isManageSorting.value = true;

                try {
                    let result;
                    if (sortBy === 'popularity_ai') {
                        // AI-based popularity sort
                        result = await apiPost('/api/manage/sort-by-popularity', {
                            playlist_id: managedPlaylist.value.id,
                            model: selectedModel.value,
                        });
                    } else {
                        // Regular sort (oldest/newest/popularity/artist/name)
                        result = await apiPost('/api/sort-playlist', {
                            playlist_id: managedPlaylist.value.id,
                            sort_by: sortBy,
                        });
                    }

                    if (result.success || result.tracks) {
                        const sortLabels = { release_date: 'oldest first', release_date_desc: 'newest first', popularity: 'most popular first', popularity_asc: 'least popular first', artist: 'by artist', name: 'by name', popularity_ai: 'AI popularity' };
                        manageSortedPlaylist.value = {
                            name: managedPlaylist.value.name + ' (sorted)',
                            description: result.message || `Sorted ${sortLabels[sortBy] || sortBy}`,
                            songs: result.tracks || [],
                            playlistId: managedPlaylist.value.id,
                        };
                    } else {
                        addToast('Sort failed', 'error');
                    }
                } catch (err) {
                    addToast('Sort failed: ' + err.message, 'error');
                } finally {
                    isManageSorting.value = false;
                }
            }

            async function manageAddSongs() {
                if (!managedPlaylist.value || isManageAddingSongs.value) return;
                isManageAddingSongs.value = true;

                try {
                    const result = await apiPost('/api/manage/add-songs', {
                        playlist_id: managedPlaylist.value.id,
                        count: manageAddSongsCount.value,
                        model: selectedModel.value,
                    });

                    if (result.success || result.songs) {
                        manageAddedSongs.value = {
                            name: managedPlaylist.value.name,
                            description: result.message || `${manageAddSongsCount.value} new songs suggested`,
                            songs: result.songs || [],
                            playlistId: managedPlaylist.value.id,
                        };
                    } else {
                        addToast('Failed to add songs', 'error');
                    }
                } catch (err) {
                    addToast('Add songs failed: ' + err.message, 'error');
                } finally {
                    isManageAddingSongs.value = false;
                }
            }

            function removeManageAddedSong(index) {
                if (!manageAddedSongs.value) return;
                manageAddedSongs.value.songs.splice(index, 1);
            }

            async function saveManageResult(mode) {
                // mode: 'new' (create new playlist) | 'existing' (add to current) | 'update' (reorder in-place)
                const playlist = manageSortedPlaylist.value || manageAddedSongs.value;
                if (!playlist || isSaving.value) return;
                isSaving.value = true;

                const trackUris = playlist.songs
                    .filter(s => (s.uri || s.spotify_uri) && !s.not_found)
                    .map(s => s.uri || s.spotify_uri);

                if (!trackUris.length) {
                    addToast('No valid tracks to save', 'error');
                    isSaving.value = false;
                    return;
                }

                try {
                    if (mode === 'update' && playlist.playlistId) {
                        // Reorder existing playlist in-place
                        const result = await apiPut(`/api/playlists/${playlist.playlistId}/reorder`, { track_uris: trackUris });
                        if (result.success) {
                            addToast(`Updated "${managedPlaylist.value?.name || 'playlist'}" track order`, 'success');
                            // Refresh managed playlist tracks
                            const response = await apiGet(`/api/playlists/${playlist.playlistId}/tracks`);
                            if (managedPlaylist.value) {
                                managedPlaylist.value.tracks = response.tracks || [];
                            }
                            closeManageResult();
                        }
                    } else if (mode === 'existing' && playlist.playlistId) {
                        // Add songs to the existing playlist
                        const result = await apiPost('/api/add-to-playlist', {
                            playlist_id: playlist.playlistId,
                            track_uris: trackUris,
                        });
                        if (result.success) {
                            addToast(`Added ${trackUris.length} songs to "${playlist.name}"`, 'success');
                            // Refresh managed playlist tracks
                            const response = await apiGet(`/api/playlists/${playlist.playlistId}/tracks`);
                            if (managedPlaylist.value) {
                                managedPlaylist.value.tracks = response.tracks || [];
                            }
                            closeManageResult();
                        }
                    } else {
                        // Create a new playlist
                        const result = await apiPost('/api/save-playlist', {
                            name: playlist.name + (manageAddedSongs.value ? ' (expanded)' : ''),
                            description: playlist.description || '',
                            track_uris: trackUris,
                        });
                        if (result.success) {
                            addToast('Playlist saved to Spotify!', 'success');
                        }
                    }
                    fetchPlaylists();
                } catch (err) {
                    addToast('Failed to save: ' + err.message, 'error');
                } finally {
                    isSaving.value = false;
                }
            }

            function closeManageResult() {
                manageSortedPlaylist.value = null;
                manageAddedSongs.value = null;
            }

            function handleManageChatEnterKey(e) {
                if (e.shiftKey) return;
                e.preventDefault();
                sendManageChat();
            }

            function scrollChatToBottom() {
                nextTick(() => {
                    const el = chatScrollContainer.value;
                    if (el) {
                        el.scrollTop = el.scrollHeight;
                    }
                });
            }


            // ═══════════════════ CHAT HISTORY ═══════════════════

            function saveChatToHistory() {
                if (conversationMessages.value.length === 0) return;
                const existingIdx = chatHistories.value.findIndex(h => h.id === currentChatId.value);
                const chatEntry = {
                    id: currentChatId.value,
                    name: getChatAutoName(),
                    feature: feature.value,
                    messages: JSON.parse(JSON.stringify(conversationMessages.value)),
                    selectedPlaylists: [...selectedPlaylists.value],
                    timestamp: Date.now(),
                };
                if (existingIdx >= 0) {
                    chatEntry.name = chatHistories.value[existingIdx].name; // preserve renamed
                    chatHistories.value[existingIdx] = chatEntry;
                } else {
                    chatHistories.value.unshift(chatEntry);
                }
                if (chatHistories.value.length > 10) {
                    chatHistories.value = chatHistories.value.slice(0, 10);
                }
                localStorage.setItem('chatHistories', JSON.stringify(chatHistories.value));
            }

            function getChatAutoName() {
                // Use the name of the first generated playlist
                const firstPlaylist = conversationMessages.value.find(m => m.role === 'assistant' && m.playlist);
                if (firstPlaylist && firstPlaylist.playlist.name) {
                    return firstPlaylist.playlist.name;
                }
                // Fallback to first user message
                const firstUser = conversationMessages.value.find(m => m.role === 'user');
                if (firstUser) {
                    return firstUser.content.substring(0, 40) + (firstUser.content.length > 40 ? '...' : '');
                }
                return 'Untitled Chat';
            }

            function loadChatFromHistory(hist) {
                saveChatToHistory(); // save current first
                currentChatId.value = hist.id;
                feature.value = hist.feature;
                conversationMessages.value = JSON.parse(JSON.stringify(hist.messages));
                conversationHistory.value = [];
                showBubbles.value = false;
                selectedBubbles.value = [];
                prompt.value = '';
                showHistoryFor.value = null;
                savedPlaylistIds.value = new Set();
                // Restore selected playlists (for from-playlists feature)
                if (hist.selectedPlaylists && hist.selectedPlaylists.length) {
                    selectedPlaylists.value = [...hist.selectedPlaylists];
                }
                // Restore highest messageIdCounter
                const maxId = Math.max(0, ...conversationMessages.value.map(m => m.id));
                messageIdCounter = maxId;
                // Show the last playlist in conversation
                const lastPlaylistMsg = [...conversationMessages.value].reverse().find(m => m.playlist);
                if (lastPlaylistMsg) {
                    selectConversationMessage(lastPlaylistMsg.id);
                } else {
                    showPlaylistPanel.value = false;
                    generatedPlaylist.value = null;
                }
                nextTick(() => scrollChatToBottom());
            }

            function deleteChatHistory(id) {
                chatHistories.value = chatHistories.value.filter(h => h.id !== id);
                localStorage.setItem('chatHistories', JSON.stringify(chatHistories.value));
                historyContextMenu.value.show = false;
            }

            function showHistoryContextMenuFn(event, histId) {
                historyContextMenu.value = {
                    show: true,
                    x: event.clientX,
                    y: event.clientY,
                    historyId: histId,
                };
            }

            function startRenameChatHistory(id) {
                const hist = chatHistories.value.find(h => h.id === id);
                if (hist) {
                    renamingHistoryId.value = id;
                    renamingHistoryName.value = hist.name;
                }
                historyContextMenu.value.show = false;
            }

            function finishRenameChatHistory() {
                if (renamingHistoryId.value && renamingHistoryName.value.trim()) {
                    const hist = chatHistories.value.find(h => h.id === renamingHistoryId.value);
                    if (hist) {
                        hist.name = renamingHistoryName.value.trim();
                        localStorage.setItem('chatHistories', JSON.stringify(chatHistories.value));
                    }
                }
                renamingHistoryId.value = null;
                renamingHistoryName.value = '';
            }

            function toggleHistoryFor(featureType) {
                showHistoryFor.value = showHistoryFor.value === featureType ? null : featureType;
            }

            const historyForFeature = computed(() => {
                return (f) => chatHistories.value.filter(h => h.feature === f);
            });


            // ═══════════════════ PLAYLIST CONTEXT MENU ═══════════════════

            function showPlaylistContextMenuFn(event, playlist) {
                event.preventDefault();
                playlistContextMenu.value = {
                    show: true,
                    x: event.clientX,
                    y: event.clientY,
                    playlist: playlist,
                };
            }

            async function renamePlaylistFromMenu() {
                const p = playlistContextMenu.value.playlist;
                if (!p) return;
                playlistContextMenu.value.show = false;
                const newName = window.prompt('Rename playlist:', p.name);
                if (newName && newName.trim() && newName.trim() !== p.name) {
                    try {
                        await apiPut(`/api/playlists/${p.id}/rename`, { name: newName.trim() });
                        p.name = newName.trim();
                        addToast('Playlist renamed', 'success');
                    } catch (err) {
                        addToast('Rename failed: ' + err.message, 'error');
                    }
                }
            }

            async function deletePlaylistFromMenu() {
                const p = playlistContextMenu.value.playlist;
                if (!p) return;
                playlistContextMenu.value.show = false;
                if (!window.confirm(`Delete "${p.name}"? This will unfollow the playlist from your library.`)) return;
                try {
                    await apiDelete(`/api/playlists/${p.id}`);
                    playlists.value = playlists.value.filter(pl => pl.id !== p.id);
                    if (managedPlaylist.value && managedPlaylist.value.id === p.id) {
                        managedPlaylist.value = null;
                        manageAction.value = null;
                    }
                    addToast('Playlist deleted', 'success');
                } catch (err) {
                    addToast('Delete failed: ' + err.message, 'error');
                }
            }

            // Editable playlist name/description in manage panel
            function startEditManagedPlaylist() {
                if (!managedPlaylist.value) return;
                editingPlaylistName.value = managedPlaylist.value.name;
                editingPlaylistDesc.value = managedPlaylist.value.description || '';
                isRenamingManagedPlaylist.value = true;
            }

            async function finishEditManagedPlaylist() {
                if (!managedPlaylist.value || !isRenamingManagedPlaylist.value) return;
                const newName = editingPlaylistName.value.trim();
                const newDesc = editingPlaylistDesc.value.trim();
                if (!newName) {
                    isRenamingManagedPlaylist.value = false;
                    return;
                }
                const nameChanged = newName !== managedPlaylist.value.name;
                const descChanged = newDesc !== (managedPlaylist.value.description || '');
                if (nameChanged || descChanged) {
                    try {
                        const body = {};
                        if (nameChanged) body.name = newName;
                        if (descChanged) body.description = newDesc;
                        await apiPut(`/api/playlists/${managedPlaylist.value.id}/rename`, body);
                        managedPlaylist.value.name = newName;
                        managedPlaylist.value.description = newDesc;
                        // Update in sidebar list too
                        const pl = playlists.value.find(p => p.id === managedPlaylist.value.id);
                        if (pl) pl.name = newName;
                        addToast('Playlist updated', 'success');
                    } catch (err) {
                        addToast('Update failed: ' + err.message, 'error');
                    }
                }
                isRenamingManagedPlaylist.value = false;
            }

            function cancelEditManagedPlaylist() {
                isRenamingManagedPlaylist.value = false;
            }


            // ═══════════════════ SETTINGS ═══════════════════

            async function saveModelPreference() {
                try {
                    await apiPost('/api/settings', { preferred_model: selectedModel.value });
                } catch {
                    // Silent fail
                }
            }

            async function saveSettings() {
                const payload = { preferred_model: settingsForm.model };
                if (settingsForm.spotifyClientId) payload.spotify_client_id = settingsForm.spotifyClientId;
                if (settingsForm.spotifyClientSecret) payload.spotify_client_secret = settingsForm.spotifyClientSecret;
                if (settingsForm.openaiApiKey) payload.openai_api_key = settingsForm.openaiApiKey;
                if (settingsForm.geminiApiKey) payload.gemini_api_key = settingsForm.geminiApiKey;
                // Cost caps & preset
                payload.max_output_tokens = settingsForm.maxOutputTokens || 0;
                payload.max_tool_calls = settingsForm.maxToolCalls || 0;
                payload.reasoning_effort = settingsForm.reasoningEffort || 'medium';
                payload.cost_preset = settingsForm.costPreset || 'med';

                try {
                    await apiPost('/api/settings', payload);
                    // Don't change selectedModel — save preserves the currently active model
                    addToast('Settings saved!', 'success');
                    closeSettings();
                    // Clear form
                    settingsForm.spotifyClientId = '';
                    settingsForm.spotifyClientSecret = '';
                    settingsForm.openaiApiKey = '';
                    settingsForm.geminiApiKey = '';
                    // Refresh models in case a new provider was configured
                    fetchModels(true);
                } catch (err) {
                    addToast('Failed to save settings: ' + err.message, 'error');
                }
            }


            // ═══════════════════ MARKDOWN RENDERING ═══════════════════

            function renderMarkdown(text) {
                if (!text) return '';
                try {
                    return marked.parse(text);
                } catch {
                    return text;
                }
            }

            // ═══════════════════ FORMATTING HELPERS ═══════════════════

            function formatDuration(ms) {
                if (!ms || ms <= 0) return '';
                const totalSeconds = Math.floor(ms / 1000);
                const minutes = Math.floor(totalSeconds / 60);
                const seconds = totalSeconds % 60;
                return `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }


            // ═══════════════════ TOAST NOTIFICATIONS ═══════════════════

            function addToast(message, type = 'info') {
                const id = ++toastId;
                toasts.value.push({ id, message, type });
                setTimeout(() => {
                    toasts.value = toasts.value.filter(t => t.id !== id);
                }, 4000);
            }


            // ═══════════════════ PANEL RESIZE ═══════════════════

            function startResizeLeft(e) {
                isResizingLeft = true;
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            }

            function startResizeRight(e) {
                isResizingRight = true;
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            }

            function onResizeMove(e) {
                if (isResizingLeft) {
                    const newWidth = Math.max(180, Math.min(400, e.clientX));
                    leftPanelWidth.value = newWidth;
                } else if (isResizingRight) {
                    const newWidth = Math.max(280, Math.min(600, window.innerWidth - e.clientX));
                    rightPanelWidth.value = newWidth;
                }
            }

            function onResizeEnd() {
                if (isResizingLeft || isResizingRight) {
                    isResizingLeft = false;
                    isResizingRight = false;
                    document.body.style.cursor = '';
                    document.body.style.userSelect = '';
                }
            }


            // ═══════════════════ INPUT HELPERS ═══════════════════

            function handleEnterKey(e) {
                if (e.shiftKey) return; // Shift+Enter for newline
                e.preventDefault();
                generate();
            }

            function autoResizeTextarea(e) {
                const el = e.target;
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 150) + 'px';
            }


            // ═══════════════════ LIFECYCLE ═══════════════════

            onMounted(() => {
                // Remove pre-load fallback now that Vue has mounted
                const preLoad = document.getElementById('pre-load');
                if (preLoad) preLoad.remove();

                window.addEventListener('keydown', (event) => {
                    if (event.key === 'Escape' && showSettings.value) {
                        closeSettings();
                    }
                });

                // Close dropdowns on outside click
                document.addEventListener('click', (e) => {
                    historyContextMenu.value.show = false;
                    playlistContextMenu.value.show = false;
                    // Close model dropdown when clicking outside
                    if (showModelDropdown.value) {
                        const dropdown = document.getElementById('model-dropdown-container');
                        if (dropdown && !dropdown.contains(e.target)) {
                            showModelDropdown.value = false;
                        }
                    }
                });

                window.__forceCloseSettings = closeSettings;

                // Panel resize handlers
                document.addEventListener('mousemove', onResizeMove);
                document.addEventListener('mouseup', onResizeEnd);

                // Check URL for auth errors
                const params = new URLSearchParams(window.location.search);
                if (params.get('error')) {
                    setTimeout(() => addToast('Spotify authentication failed. Please try again.', 'error'), 500);
                    window.history.replaceState({}, '', '/');
                }
                checkStatus();
            });

            // Auto-scroll when new messages arrive or generation starts
            watch(() => conversationMessages.value.length, () => scrollChatToBottom());
            watch(isGenerating, () => scrollChatToBottom());


            // ═══════════════════ RETURN ═══════════════════

            return {
                // State
                view, feature, showSettings, user, models, selectedModel, showModelDropdown,
                setupForm, setupLoading, setupError,
                settingsForm, keyStatus, keyStatusLoading,
                playlists, playlistsLoading, selectedPlaylists, playlistSortBy, showOnlyMyPlaylists,
                prompt, playlistSize, generationMode, isGenerating, generatingMessage, conversationHistory,
                showBubbles, bubbleSuggestions, selectedBubbles, aiMessage, customAnswer, showCustomAnswerInput, hideOriginalPrompt,
                conversationMessages, selectedMessageId, showPlaylistPanel,
                generatedPlaylist, songRatings,
                showRefineInput, refinePrompt, isRefining,
                isSaving,
                debugMode, lastDebugInfo,
                audioPlayer, currentlyPlayingSong, currentlyPlayingIndex, isPlaying, playbackProgress,
                chatMessages, chatInput, isChatLoading, chatContainer, chatScrollContainer,
                managedPlaylist, manageAction, manageChatMessages, manageChatInput, isManageChatLoading,
                manageSortedPlaylist, manageAddedSongs, isManageSorting, isManageAddingSongs, manageAddSongsCount,
                toasts,
                quickChatPrompts: [], sortOptions: [], manageSortOptions,
                chatHistories, showHistoryFor, historyContextMenu,
                renamingHistoryId, renamingHistoryName,
                playlistContextMenu, isRenamingManagedPlaylist, editingPlaylistName, editingPlaylistDesc,
                leftPanelWidth, rightPanelWidth,
                COST_PRESETS,

                // Computed
                playlistSelectionLabel, promptPlaceholder,
                foundSongsCount, notFoundSongsCount, sortedPlaylists, historyForFeature,
                isCurrentPlaylistSaved, modelSupportsWebSearch,

                // Methods
                submitSetup, loginSpotify, logoutSpotify, closeSettings,
                setFeature, applyCostPreset,
                togglePlaylistSelection, isPlaylistSelected,
                generate, selectBubble, handleEnterKey, autoResizeTextarea,
                selectConversationMessage, startNewChat,
                rateSong, removeSong, refinePlaylist, savePlaylistToSpotify, discardPlaylist,
                playPreview, togglePlayback, stopPreview, onAudioEnded, onTimeUpdate,
                selectPlaylistForManage, setManageAction, sendManageChat, handleManageChatEnterKey,
                manageSortPlaylist, manageAddSongs, removeManageAddedSong,
                saveManageResult, closeManageResult,
                saveModelPreference, saveSettings, verifyKeys, openSettings,
                renderMarkdown, addToast, formatDuration,
                loadChatFromHistory, deleteChatHistory, toggleHistoryFor,
                showHistoryContextMenuFn, startRenameChatHistory, finishRenameChatHistory,
                showPlaylistContextMenuFn, renamePlaylistFromMenu, deletePlaylistFromMenu,
                startEditManagedPlaylist, finishEditManagedPlaylist, cancelEditManagedPlaylist,
                startResizeLeft, startResizeRight,
            };
        }
    }).mount('#app');

} catch (error) {
    console.error('Failed to initialize Vue app:', error);
    var preLoad = document.getElementById('pre-load');
    if (preLoad) preLoad.remove();
    document.body.innerHTML = '<div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#121212;color:#fff;font-family:sans-serif;padding:2rem;"><div style="max-width:600px;"><h1 style="color:#f00;font-size:2rem;margin-bottom:1rem;">Application Error</h1><p style="margin-bottom:1rem;">The application failed to start. Error details:</p><pre style="background:#000;padding:1rem;border-radius:0.5rem;overflow:auto;font-size:0.875rem;color:#1ed760;">' + error.toString() + '\n\n' + (error.stack || '') + '</pre><p style="margin-top:1rem;font-size:0.875rem;color:#999;">Press F12 to open developer console for more information.</p></div></div>';
}
