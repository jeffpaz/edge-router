# Changelog

All notable changes to edge-router are documented here.

---

## 2026-07-18

### Fixed
- **`/query/stream` never escalated to cloud on low confidence** — the streaming endpoint only checked `realtime_classifier` (sports/news/price intent) and otherwise streamed unconditionally from Ollama, ignoring the skill classifier's `SKILL_THRESHOLDS` entirely. Queries classified into always-escalate skills (`math_data`, `current_events`, `sports_people`, threshold `1.0`) — e.g. "How many In-N-Out locations are there?", "What's the average humidity in Las Vegas in July?" — stayed local instead of escalating, and the SSE done-event's `confidence_score` was hardcoded `null` (rendered as 0% by the frontend).
- **`main.py`: `query_stream_endpoint` rewritten** to mirror `/query`'s routing logic exactly: local-only skills (`conversational`, `definition`, `creative`, `how_to`, `opinion_advice`, `language_task`) still stream live from Ollama since they never escalate; everything else runs `local_llm.query()` to completion first for a real confidence score, applies the local-first retry (general/coding), and escalates to the skill-matched cloud provider when `should_escalate` is true — then fake-streams whichever answer was chosen. Trades true incremental token streaming for escalatable skills in exchange for a routing decision that matches `/query`.

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
