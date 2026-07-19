# Changelog

All notable changes to edge-router are documented here.

---

## 2026-07-18

### Added
- **Conversation history** — `/query` and `/query/stream` accept an optional `messages` array (`[{role, content}]`) so follow-ups ("scale that recipe for 20 people") resolve against prior turns. `main.py` adds a server-side session store (`_sessions`, `OrderedDict` keyed by `session_id`, max 50 sessions, oldest evicted first) as a fallback for clients that only send `session_id` with empty `messages`; client-sent messages win when present. Both sources are trimmed to the last 10 messages (5 turns) via `_session_context()` / `_trim_messages()` before use, and `_session_update()` appends each turn's exchange after the answer is known — wired into every return path in both endpoints, including token-accumulation in the raw-streaming `local_only_stream` path so session history is captured even when the SSE response is never assembled into one string elsewhere. `local_llm.py` flattens history into the prompt text for Ollama; `cloud_llms.py` already passed it through natively per provider.

### Changed
- **`OLLAMA_MODEL` switched from `gemma2:2b` to `llama3.2:3b`** in `.env`, matching the currently-requested default. Pulled via `docker exec ollama ollama pull llama3.2:3b` before switching, avoiding the 404 regression from the 2026-06-11 entry below (that time the env var was changed without the model being present on the instance).
- **`local_llm.py`: default formatting system prompt** — when the caller doesn't supply a `system` prompt, `query()` and `generate_stream()` now default to `_DEFAULT_FORMAT_SYSTEM_PROMPT` (passed via Ollama's native `system` field, not concatenated into the user prompt, so a small model doesn't echo it back) asking for blank lines between paragraphs/stanzas and one list item per line. This is a soft nudge, not a guarantee — verified against `llama3.2:3b` that it doesn't reliably change output for short responses; the actual "poem renders as a wall of text" bug this was investigated for turned out to be a frontend markdown-rendering issue (see pazlabs.io's changelog), not missing newlines from Ollama.

### Fixed
- **`sports_people` classifier matched bare sport names regardless of context** — `router.py`'s skill patterns included standalone `\bgolf\b`, `\btennis\b`, `\bsoccer\b`, etc., so "Write a poem about golf and the rain" scored one match each for `sports_people` (via "golf") and `creative` (via "poem"), and `sports_people` won the tie because it's listed first in `_PATTERNS`. `sports_people` has `SKILL_THRESHOLDS` fixed at `1.0` (always escalate) and maps straight to Grok, so a creative-writing prompt was silently shipped to a cloud provider instead of answered locally. Fixed with `_sport_pattern()`: generic sport names now only count toward `sports_people` when a real sports-context signal (`team`, `score`, `won`, `league`, a league acronym, etc.) also appears in the query, via a same-string lookahead. League acronyms (`nhl`, `nba`, `pga`, ...) and explicit role/event words (`player`, `championship`, `drafted`, ...) are unambiguous and unchanged.

### Fixed (code review follow-up)
- **`/query/stream` never escalated to cloud on low confidence** — the streaming endpoint only checked `realtime_classifier` (sports/news/price intent) and otherwise streamed unconditionally from Ollama, ignoring the skill classifier's `SKILL_THRESHOLDS` entirely. Queries classified into always-escalate skills (`math_data`, `current_events`, `sports_people`, threshold `1.0`) — e.g. "How many In-N-Out locations are there?", "What's the average humidity in Las Vegas in July?" — stayed local instead of escalating, and the SSE done-event's `confidence_score` was hardcoded `null` (rendered as 0% by the frontend).
- **`main.py`: `query_stream_endpoint` rewritten** to mirror `/query`'s routing logic exactly: local-only skills (`conversational`, `definition`, `creative`, `how_to`, `opinion_advice`, `language_task`) still stream live from Ollama since they never escalate; everything else runs `local_llm.query()` to completion first for a real confidence score, applies the local-first retry (general/coding), and escalates to the skill-matched cloud provider when `should_escalate` is true — then fake-streams whichever answer was chosen. Trades true incremental token streaming for escalatable skills in exchange for a routing decision that matches `/query`.

### Fixed (code review follow-up)
- **`router.py`: removed dead `route()` / `RouterResponse`** — an unused standalone implementation of the same local-first/escalate decision tree added when `SkillRouter` was built, never called from `main.py`. Its existence let the `/query/stream` routing bug above happen (nobody had one place to fix), and would have let the same drift happen again. Also dropped its now-unused imports (`dataclass`, `Any`, `config`, `local_llm`).
- **`main.py`: extracted `_resolve_query()`** — the local-only/confidence-check/local-first-retry/cloud-dispatch decision tree that `query_endpoint` and `query_stream_endpoint` each independently re-implemented (a second instance of the same class of bug fixed above) is now one function both endpoints call. `router.py`'s `route()` wasn't reused since it lacked the input/output token breakdown and per-branch structured logging `main.py` needs; the shared implementation now lives in `main.py` instead.
- **`main.py`: retry-path exception in `/query/stream` was unguarded** — the local-first-retry call inside the old `routed_stream()` could raise uncaught (e.g. malformed `messages` items), silently dropping the SSE connection with no error event, unlike the initial call three lines above it which was guarded. Fixed as a side effect of routing both the initial and retry calls through `_resolve_query()` behind a single try/except.
- **`main.py`: escalated-response SSE metadata was missing `model`, `prompt_tokens`, `completion_tokens`** — present on local/retry-success done-events but absent when the answer came from a cloud escalation, the one case a frontend most needs to know which model actually answered. `_stream_metadata()` now builds an identical shape for all three outcomes.
- **`main.py`: `/query/stream` never populated or read the response cache** — `/query` caches confident local answers and serves repeats instantly; the streaming endpoint never called `_cache_put`, so identical queries redid a full local inference every time regardless of which endpoint answered first. Both endpoints now share the same cache.
- **`main.py`: added SSE heartbeat during the blocking local-inference pass** — `/query/stream` now runs a full non-streaming local pass (see above) before the first byte; `_heartbeat_until_done()` sends an empty-token event every 15s while that runs, so a proxy or client idle-read timeout tuned to the old near-instant-first-byte behavior doesn't kill a request that's still legitimately in progress.
- **README.md** — fixed a self-contradicting claim ("a true Ollama stream if the local answer is accepted") added in the previous entry; local answers in escalatable skills are fake-streamed too, only local-only skills get true incremental streaming. Documented the heartbeat, the shared cache, and the shared `_resolve_query()` implementation.

---

## 2026-06-12

### Added
- **`realtime_classifier.py`** — pre-LLM intent classifier. Detects real-time queries (sports scores, stock prices, breaking news, live events) using temporal signal matching, topic keywords, sports entity lists, and regex patterns. Returns `IntentResult(is_realtime, confidence, signals, preferred_provider)`.
- **Realtime bypass in `/query`** — before running local Ollama, checks `classify_intent()`. If `is_realtime=True`, routes directly to `grok` (sports/scores) or `openai` (prices/news), bypassing local inference entirely. Response includes `realtime_intent: true` and `realtime_signals` list.
- **Realtime bypass in `/query/stream`** — same intent check at stream start. If realtime, calls cloud LLM, then fake-streams the response 4 chars/chunk to maintain streaming UX. Done event metadata includes full routing fields.

### Fixed
- **Ollama streaming done-event metadata** — the SSE `done` event emitted by `local_llm.generate_stream()` was missing fields required by the frontend: `routed_to`, `source`, `model_used`, `latency_ms`, `confidence_score`, and `tokens: {input, output, total}`. All fields now included. `latency_ms` is measured with `time.monotonic()` from before the HTTP call to Ollama.
- **`import time` added to `local_llm.py`** — was missing, required for `time.monotonic()` in the streaming path.


---

## 2026-06-11

### Added
- **System metrics collector** — background asyncio task (`_metrics_collector_loop`) starts at app startup via `asyncio.create_task` in the FastAPI lifespan. Samples CPU% (psutil), RAM used/total (psutil), and GPU% (tegrastats `GR3D_FREQ` parsed via subprocess) every 60 seconds. Stores rows in SQLite at `/app/data/metrics.db`. Prunes rows older than 7 days on each insert.
- **`GET /metrics/history?days=N`** — returns all `system_metrics` rows from the past N days (default 5), ordered ASC. Fields: `timestamp`, `cpu_pct`, `gpu_pct`, `ram_used_mb`, `ram_total_mb`.
- **`docker-compose.yml` data volume** — `/home/jeffpaz/edge-router/data:/app/data` bind mount so metrics DB survives container restarts.
- **Gemini `error_code` / `error_detail` fields** — `_probe_gemini` in `/providers/status` now parses Google API error JSON and returns `error_code` (HTTP status) and `error_detail` (human-readable message) on non-200 responses. Switched URL from `v1beta/models` to `v1/models`.
- **Billing `error_detail` passthrough** — `_billing_probe` now returns `{status, error_detail}` dict; `_get_billing_status` and `GET /billing` surface the error message when status is `"error"`.

### Changed
- **`GET /health` live model name** — was returning `config.OLLAMA_MODEL` (static env var). Now calls `GET /api/tags` and returns the first model's name from the live Ollama response. Falls back to `"No model loaded"` or `"Unavailable"` on failure.
- **`OLLAMA_MODEL` corrected to `gemma2:2b`** — `.env` had `llama3.2:3b` which doesn't exist on this Ollama instance, causing 404 on every inference call. Fixed to match the loaded model.
- **`GEMINI_MODEL` updated to `gemini-2.5-flash`** — `gemini-2.0-flash` is deprecated and returns a model-not-available error from the SDK. `gemini-2.5-flash` is current.
- **Streaming error handling in `generate_stream`** — client timeout reduced from 120s to 30s. Added `received_tokens` guard: if stream completes with no tokens, yields `{"error": true, "message": "Ollama returned an empty response", ...}` SSE event. Explicit `httpx.TimeoutException`, `HTTPStatusError`, and general `Exception` catches now yield structured `{"error": true, "message": "...", "provider": "ollama", "done": true}` SSE events instead of propagating silently.
- **`event_stream` error format in `/query/stream`** — outer exception handler now emits the same structured error SSE format (`error`, `message`, `provider`, `done`) instead of the previous `{done: true, metadata: {error: ...}}` shape.

---

## 2026-06-03

### Added
- **`GET /storage`** — disk usage for microSD (`/host_root`) and NVMe (`/host_mnt`) via psutil. Returns `{microsd: {label, total_gb, used_gb, free_gb}, nvme: {...}}`. Falls back to hardcoded values if psutil fails.
- **`GET /memory`** — system RAM stats via psutil. Returns `{total_gb, used_gb, available_gb, cached_gb}`. Falls back to hardcoded values.
- **Conversation history** — `/query` and `/query/stream` endpoints accept `messages: list` for multi-turn context; history is prepended to the prompt sent to Ollama.
- **Multi-signal confidence scoring** — local_llm.py combines token log-probabilities (Signal 1), hedging-language heuristic (Signal 2), self-rating via injected `CONFIDENCE: N` tag (Signal 3), hallucination-risk patterns (Signal 4), and query complexity pre-bias (Signal 5) into a weighted confidence score.
- **Skill-based routing** — router.py classifies queries into skills (coding, math_data, conversational, how_to, etc.) and routes to the appropriate cloud provider (Claude for coding, Gemini for math, Grok for current events, OpenAI for general). Per-skill confidence thresholds and context window sizes.
- **Provider health probes** — `GET /providers/status` makes lightweight model-list API calls to each provider (no tokens burned), cached 5 minutes. Returns status, model, latency, credits, and available models per provider.
- **Billing probes** — `GET /billing` makes 1-token API calls to each cloud provider to verify keys are active and quotas remain. Results cached 5 minutes.
- **Recent-queries in-memory buffer** — last 100 queries stored in a `deque` for `/stats` health visibility.
- **`GET /stats`** — returns query count, routing distribution, and latency stats from the in-memory buffer.
- **`POST /query/stream`** — SSE streaming endpoint. Streams Ollama tokens as `{"token": "...", "done": false}` events; final event includes routing metadata and token counts.
- **Ollama keep-alive** — warmup request on startup with `keep_alive: 10m`; keepalive re-sent after each inference to prevent model unload on Jetson.
- **`GET /jetson/containers`** — lists running Docker containers via Docker socket. Optional `X-Edge-Key` auth header.
- **Cloud LLM fallback chain** — on low confidence, routes to the skill-matched cloud provider. If that fails, tries remaining providers in priority order before returning an error.

### Infrastructure
- FastAPI with `asyncio` lifespan, structured JSON logging (`_JsonFormatter`), CORS middleware.
- Docker Compose stack: `edge-router` + `ollama` on `edge-net` bridge network. `ollama` uses `nvidia` runtime, model volume at `/mnt/models/ollama`.
- Host filesystem mounted read-only at `/host_root` and `/host_mnt`; Docker socket mounted for container listing.
