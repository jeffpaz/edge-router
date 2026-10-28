# edge-router

The AskJetson router on the Jetson Orin Nano (`192.168.11.50`). Queries go to a **local Ollama model** first (`llama3.2:3b`); if the response confidence falls below a configurable threshold, the query is escalated to a cloud provider (Claude, Gemini, Grok, or OpenAI). Private requests, personal documents and PazBrain questions go to the **Mac mini** instead (`192.168.11.60`, `qwen3:4b-instruct` and the PazRAG document index) and never reach a cloud provider.

It runs as a Docker Compose container next to an `ollama/ollama` container and is reached publicly through a Cloudflare tunnel at `api.pazlabs.io` (the pazlabs.io web chat). See [ARCHITECTURE.md](ARCHITECTURE.md) for how the three machines fit together and [CHANGELOG.md](CHANGELOG.md) for what changed.

```
POST /query  (or /query/stream, /ask/stream)
  │
  ├─ realtime_classifier ──is_realtime──► cloud LLM (Grok/OpenAI) ──► return
  │
  ├─ local-only skill (conversational/definition/creative/how_to/opinion_advice/language_task)
  │     └─ always local ──► return response
  │
  ├─ local Ollama ──confidence OK──► return response
  │
  └─ confidence low ───────────────► cloud LLM (skill-matched) ──► return response
```

`/query` and `/query/stream` share one routing implementation (`main.py`'s
`_resolve_query()`), so the two endpoints can't drift out of sync on which
queries escalate. Local-only skills are the one exception in `/query/stream`:
they skip `_resolve_query()` entirely and stream live from Ollama token-by-token,
since they never need a confidence check. Every other skill is resolved first
(local answer, local-first retry, or cloud escalation — whichever `_resolve_query()`
lands on), then the result is chunked into the same SSE shape either way. This
means local answers in escalatable skills (general, coding, math_data, current_events,
sports_people) arrive as a single completed block rather than token-by-token — the
trade for a routing decision that's guaranteed to match `/query`. A heartbeat event
is sent periodically while `_resolve_query()` is running so the connection doesn't
idle out on a slow local pass. Both endpoints also share the same response cache.

`/ask/stream` (used by the pazlabs.io web chat) also routes through `_resolve_query()`,
but plugs in `local_llm.query_streaming()` so every local pass, the retry included,
streams token by token. Alongside the tokens it sends live Jetson telemetry from a
per-request tegrastats sampler, and each Ollama call's own load/prefill/decode
timings. If the answer escalates to cloud, a `replace` event swaps the cloud
answer in for the local text. See [`POST /ask/stream`](#post-askstream).

Every Ollama call (all three endpoints, the retry pass, the keep-alive ping and the
startup warm-up) sends the same runner options (`num_ctx` 4096, `num_thread` 6) and
`keep_alive: -1`. Ollama reloads the model whenever a runner option differs from the
loaded runner, so this one shared set is what keeps follow-up queries from paying the
~6s load again. `num_predict` still varies by skill; it's a sampling option and never
triggers a reload.

## Tiered routing

Every request is classified on the Jetson (the regex skill classifier plus
`realtime_classifier`, no LLM call), then answered by one of three lanes chosen by
data sensitivity and job type (`tiers.py`):

| Lane | Host | Serves |
|---|---|---|
| `local` | Jetson Ollama (`OLLAMA_MODEL`) | local-only skills and confident local answers |
| `private` | Mac mini Ollama (`MINI_MODEL`) and its document index (`MINI_RAG_URL`) | private mode, personal documents and PazBrain questions, batch jobs (`POST /jobs/generate`), and the fallback when the cloud is down |
| `cloud` | Claude / Gemini / Grok / OpenAI | every other interactive query; live-data questions go to Grok with server-side `web_search` + `x_search` (Responses API) |

- **Fallback** for non-private requests, on a cloud timeout (`CLOUD_LANE_TIMEOUT_S`, whole provider chain) or every provider failing: cloud → mini → Jetson → a clear error. A live-data question answered by the mini or the Jetson comes back with `limitation: "live_data_unavailable"`. Fallback answers are never cached.
- **Every response** (`/query`, the final frame of `/query/stream` and `/ask/stream`, `/jobs/generate`) carries `tier`, `model`, `private`, `fallback_used` and `route_reason`.
- **Mini unset** (`MINI_OLLAMA_URL` or `MINI_MODEL` empty): the private lane is disabled, `/jobs/*` returns 503, and cloud-down fallback goes straight to the Jetson.
- **Runner options:** every call to the mini sends `num_ctx` = `MINI_NUM_CTX` (8192, the same as tripBuddy's calls) and `keep_alive: -1`, so the shared runner is never reloaded.
- **`GET /tiers/status`**: per-lane health. The two Ollama lanes are probed via `/api/tags` and cached 15 s; a failed mini call forces a fresh probe.
- **`POST /jobs/generate`** `{prompt, system?, format?, num_predict?, temperature?}`: always the mini, never cloud or Jetson, so it fails closed. Requires a private identity (`auth.private_identity`): a Firebase ID token for a verified email in `PRIVATE_ALLOWED_EMAILS`, or `PRIVATE_TOKEN`. Logs metadata only.

Tests: `pip install -r requirements-dev.txt && python -m pytest -q` (every Ollama and cloud call is faked).

### Private mode

`"private": true` on `/query`, `/query/stream` or `/ask/stream` (the web chat's
"Stay private / local" checkbox) keeps a request off every cloud provider:

- **Lane:** mini → Jetson → clear error. Live-data questions are answered locally with `limitation: "live_data_unavailable"`.
- **Second guard:** `privacy.private_scope()` marks the request, and every cloud call path calls `privacy.assert_cloud_allowed()`, which raises inside a private request.
- **Access:** requires `auth.private_identity()` (allow-listed verified email or `PRIVATE_TOKEN`), else 401/403.
- **Retention:** never cached, never stored in the server-side session, never logged with query or answer text.
- **Client duty:** the client must not send private turns as `messages` history on later non-private requests. The web chat filters them out.

### Personal documents

A question about the user's own records goes to the mini's document index (PazRAG `POST /rag/query` at `MINI_RAG_URL`, bearer `MINI_RAG_TOKEN`):

- **Detection (`personal_docs.py`):**
  - an explicit "in my documents / files" request
  - "my/our" + a document or personal-finance noun ("my 2023 W-2", "our mortgage rate", "my deductible")
  - first-person money or records verbs ("what did Acme pay me", "how much did I pay in property tax")

  General ("what is a W-2?") and how-to questions keep normal routing.
- **Private mode consults the documents first, whatever the wording** (`_private_with_docs()`). It uses the documents answer only when the index returns relevant, cited sources, and otherwise answers on the mini as before. Live-data, creative and language-task questions skip the lookup.
- **Always private:** it's forced private whatever the client sent, so it needs a private identity (401/403 otherwise) and follows every private-mode rule (no cache, no session store, no text in logs).
- **No fallback:** if the index is unset or down, the answer is a clear "unavailable", never the cloud or the Jetson.
- **Response:** `citations` (path, file_name, page, doc_type, year, source, score) in `/query` and in the final metadata of both streams, plus `limitation: "weak_evidence"` when the index found only weak matches.
- **Health:** `GET /tiers/status` reports the `docs` lane.

### PazBrain notes

Naming PazBrain ("Ask PazBrain …", "Use PazBrain …", "Search PazBrain for …", matched by `\bpaz ?brain\b`) is how the user asks to stay private and local. Such a question is forced private and goes to the same document index, which searches the PazBrain notes and the documents together. The word itself never narrows the search to the notes. Words like "notes", "wiki" or "journal" alone keep normal routing.

The notes reach the mini one way: `pazbrain/scripts/export_for_mini.py` renders the wiki (`~/pazbrain/wiki`) into one deterministic tar and pipes it over SSH with a key that, on the mini, can only run PazRAG's `scripts/pazbrain_receive.sh`. It skips unchanged exports and refuses an empty export or one with less than half the pages of the last push (`--force` overrides). The `pazbrain-mini-export.timer` (`pazbrain/systemd/`) runs it every 15 minutes once it's installed. Logs carry counts only.

## Project layout

```
edge-router/
├── main.py                FastAPI app: /query, /query/stream, /ask/stream, /health, monitoring endpoints
├── router.py              Skill classifier + routing logic
├── tiers.py               Routing lanes (Jetson / Mac mini / cloud), lane health, mini + document-index clients
├── privacy.py             Per-request private flag + the guard every cloud call checks
├── personal_docs.py       Personal-documents and PazBrain intent detection
├── local_llm.py           Ollama client with confidence scoring and SSE streaming
├── telemetry.py           Per-request telemetry for /ask/stream (tegrastats, mini metrics, KV estimate)
├── auth.py                Firebase ID token verification; private_identity() allow-list
├── trip_routes.py         tripBuddy trip-planning endpoint (+ providers_trip.py)
├── pazbrain/              PazBrain wiki API (/brain/api), read-only Logseq queries (/brain/query),
│                          host scripts (sync, Logseq ingest, export_for_mini.py) and systemd units
├── cloud_llms.py          Clients for Claude, Gemini, Grok, OpenAI
├── realtime_classifier.py Real-time intent detector (sports/stocks/news bypass)
├── config.py              API keys and settings from environment variables
├── docker-compose.yml     edge-router + ollama containers (how it runs on the Jetson)
├── deploy.sh              full rebuild and restart
├── tests/                 pytest; every Ollama, mini and cloud call is faked
└── requirements.txt
```

---


## Realtime bypass

Before sending a query to Ollama, `realtime_classifier.classify_intent()` checks whether the query requires live data (sports scores, stock prices, breaking news, live events). If it does, the local model is skipped and the query routes directly to:

- **Grok** — sports scores, match results, sports entities
- **OpenAI** — stock prices, financial data, breaking news

The `/query` response includes `realtime_intent: true` and a `realtime_signals` list when this path is taken. The `/query/stream` endpoint applies the same check and fake-streams the cloud response to maintain streaming UX.

---

## Conversation history

Both endpoints accept an optional `messages` array (`[{"role": "user"|"assistant", "content": "..."}]`) so follow-ups like "scale that recipe for 20 people" resolve against prior turns instead of being answered cold. Client-sent `messages` win when present; `main.py` also keeps a server-side session store (`_sessions`, keyed by `session_id`, max 50 sessions with oldest evicted first) as a fallback for clients that only send `session_id` with an empty `messages` array. Either source is trimmed to the last 10 messages (5 turns) before being used, to bound prompt size — see `_session_context()` / `_session_update()` in `main.py`.

For the local model, history is flattened into the prompt text (`local_llm.py`: `"Previous conversation:\n..."` prepended before the current question). Cloud providers receive it as a native messages array in each SDK's own format (Anthropic/OpenAI-style `role`/`content` objects, or Gemini's `parts` shape).

---

## Running on the Jetson

The Jetson (JetPack 6, Ubuntu 22, ARM64, 8 GB shared memory) runs two containers from
`~/edge-router/docker-compose.yml`:

- `ollama` (`ollama/ollama`, NVIDIA runtime, models in `/mnt/models/ollama`, `OLLAMA_KEEP_ALIVE=-1`).
- `edge-router` (built from this repo, uvicorn on `:8000`), reading `.env` and `/etc/pazbrain.env`.

The Cloudflare tunnel (`cloudflared` on the host) sends `api.pazlabs.io` to `localhost:8000`.
The `edge-router.service` systemd unit and `env.example` in this repo are from an earlier
bare-metal setup and aren't used.

```bash
cp .env.example .env && chmod 600 .env    # fill in keys; never commit .env
./deploy.sh                               # first time: full no-cache rebuild

# Updating to the latest main
cd ~/edge-router && git fetch -q && git checkout -q main && git pull --ff-only \
  && docker compose build edge-router && docker compose up -d --no-deps edge-router

docker logs -f edge-router                # JSON logs; metadata only for private requests
curl -s localhost:8000/health
curl -s localhost:8000/tiers/status
```

Tests run on any machine: `pip install -r requirements-dev.txt && python -m pytest -q`.

Host-side PazBrain jobs (outside the container, venv `/opt/pazbrain/venv`, scripts in
`~/pazbrain/bin`): `pazbrain-sync.timer` (Drive → wiki, every 15 minutes),
`pazbrain-query.service` (the read-only Logseq sidecar), and
`pazbrain-mini-export.timer` (wiki → mini, not installed yet).

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local model tag |
| `CONFIDENCE_THRESHOLD` | `0.70` | Min score [0–1] to accept local response |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model override |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Google model override |
| `GROK_MODEL` | `grok-3-mini` | xAI model override |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model override |
| `PAZBRAIN_QUERY_RATE_PER_MIN` | `30` | Per-caller rate limit for `POST /brain/query` |
| `GROK_SEARCH_MODEL` | `grok-4.20-0309-non-reasoning` | Live-data questions (Responses API with `web_search` + `x_search`) |
| `MINI_OLLAMA_URL`, `MINI_MODEL` | unset | The mini's Ollama (`http://192.168.11.60:11434`, `qwen3:4b-instruct`). Either unset disables the private lane |
| `MINI_NUM_CTX` | `8192` | Sent on every mini call; must match the mini daemon and tripBuddy |
| `MINI_TIMEOUT_S`, `CLOUD_LANE_TIMEOUT_S` | `120`, `60` | Mini call timeout; whole cloud chain before falling back |
| `MINI_RAG_URL`, `MINI_RAG_TOKEN` | unset | PazRAG on the mini (`http://192.168.11.60:8100`) and its bearer token. Either unset: documents questions get a clear "unavailable" |
| `MINI_RAG_TIMEOUT_S` | `90` | Document-index call timeout |
| `PRIVATE_ALLOWED_EMAILS` | unset | Verified Firebase emails allowed to use private mode, documents and `/jobs` |
| `PRIVATE_TOKEN` | unset | Static bearer token for scripts; both unset → nobody gets the private lane |
| `AUTH_ENABLED`, `AUTH_REQUIRED` | `false`, `false` | Verify and log Firebase tokens on public routes; `AUTH_REQUIRED` also rejects. Live: enabled, not required |

The Ollama runner options (`num_ctx` 4096, `num_thread` 6) and `keep_alive: -1` are fixed
in `local_llm.py` (`_RUNNER_OPTIONS`, `KEEP_ALIVE`), not read from the environment, so no
call can drift from the others. `config.OLLAMA_NUM_CTX` (and the `OLLAMA_NUM_CTX=512` in
`docker-compose.yml`) is not used by the local client: every request sets 4096.

Skill → cloud LLM mapping (fixed in `router.py`'s `_SKILL_MAP`):

| Skill | Provider | Model |
|---|---|---|
| `coding` | Claude | `claude-sonnet-4-20250514` |
| `math_data` | Gemini | `gemini-2.0-flash` |
| `current_events` | Grok | `GROK_MODEL` default |
| `sports_people` | Grok | `GROK_MODEL` default |
| `general` | OpenAI | `gpt-4o` |

`math_data`, `current_events`, and `sports_people` have their `SKILL_THRESHOLDS` set to `1.0` in `config.py` — they always escalate regardless of local confidence.

The `sports_people` classifier requires generic sport names (`golf`, `tennis`, `soccer`, `hockey`, `basketball`, `football`, `baseball`) to appear alongside an actual sports-context signal (`team`, `score`, `won`, `league`, a league acronym, etc.) elsewhere in the query — see `_sport_pattern()` in `router.py`. Without this, a bare sport name matched on its own regardless of context (e.g. "write a poem about golf" was misclassified as `sports_people` and escalated to Grok purely because it contained the word "golf", instead of being classified `creative` and answered locally).

## API

### `POST /query`

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | User query text |
| `force_cloud` | bool | no | Skip local model, go straight to cloud (default `false`) |
| `system` | string | no | System prompt forwarded to whichever LLM answers. Local model falls back to a default formatting-guidance prompt (blank lines between paragraphs/stanzas, one list item per line) when this is omitted — see `_DEFAULT_FORMAT_SYSTEM_PROMPT` in `local_llm.py`. |
| `session_id` | string | no | Client-chosen session identifier. Used to key the server-side conversation history fallback (see "Conversation history" above) and returned unchanged in the response. |
| `messages` | array | no | Prior turns as `[{"role": "user"|"assistant", "content": "..."}]`. Trimmed to the last 10 before use. |

Response:

| Field | Type | Description |
|---|---|---|
| `answer` | string | Model response |
| `source` | string | `"local"` or cloud provider name |
| `model_used` | string | Exact model ID that answered |
| `skill` | string | Classified query type |
| `latency_ms` | float | Total wall-clock time for the request |
| `confidence_score` | float\|null | Local model confidence; null when `force_cloud=true` or the realtime bypass answered |
| `tokens` | object | `{input, output, total}` token breakdown |
| `session_id` | string\|null | Echoes the request's `session_id` |
| `cache_hit` | bool | `true` when served from the in-memory answer cache instead of running inference |
| `realtime_intent` | bool | `true` when the realtime bypass (sports/news/price) answered instead of the normal local/cloud path |
| `realtime_signals` | array | Signal names that triggered the realtime bypass, if any |
| `routing_reason` | string\|null | Set for realtime-bypass responses; explains why that path was taken |


### `POST /query/stream`

Same request body as `/query`, same routing decision — see the diagram above. Returns
a Server-Sent Events (SSE) stream.

Token events: `data: {"token": "Hello", "done": false}`. While `_resolve_query()`
is running (the local-first/escalate decision), an empty-token event of the same
shape (`data: {"token": "", "done": false}`) may be sent periodically as a
heartbeat, so a proxy or client idle-read timeout doesn't kill the connection
during a slow local pass — safe to concatenate like any other token event.

Done event (final) — routing/confidence info is nested under `metadata`, and
its shape is identical regardless of which path answered (local, retry, or
cloud escalation):

```json
{
  "token": "",
  "done": true,
  "metadata": {
    "routed_to": "local",
    "source": "local",
    "model_used": "llama3.2:3b",
    "skill": "general",
    "confidence_score": 0.7967,
    "latency_ms": 1234,
    "tokens": { "input": 42, "output": 87, "total": 129 },
    "prompt_tokens": 42,
    "completion_tokens": 87
  }
}
```

For cloud-routed responses, `metadata.routed_to`/`metadata.source` are the provider
name (e.g. `"claude"`, `"grok"`, `"openai"`) and `metadata.model`/`model_used` are the
cloud model that answered — these fields are always present, whether the answer came
from local, a local-first retry, or a cloud escalation. `metadata.escalated: true` is
set when the route was a confidence-based escalation (as opposed to
`metadata.realtime_intent: true` for the realtime bypass, or `metadata.retry: true`
for a local-first retry that avoided escalation, or `metadata.cache_hit: true` when
the answer came from the shared response cache instead of running inference at all).
`confidence_score` is `null` only for realtime-bypass and force-cloud responses, where
no local inference ran.

On error, a single event is emitted instead: `data: {"error": true, "message": "...", "provider": "ollama", "done": true}`.

### `POST /ask/stream`

Same request body and routing as `/query/stream`, but the local passes stream live and
the response uses **named** SSE events (`event: <kind>` followed by `data: <json>`):

| Event | Data | When |
|---|---|---|
| `call` | `{call, label, t}` | An Ollama call starts; `label` is `local`, `retry` or `mini` |
| `token` | `{text}` | Live answer text |
| `metrics` | `{t, host, cpu, gpu, ram, tok, tps, kv, n}` | About 8/s for the whole request |
| `phases` | `{call, label, load, prefill, decode, search?, total, prompt_eval_count, eval_count, wall, t_start}` | Once per Ollama call, from its final chunk (times in seconds). A documents answer reports the mini's steps (`filters`, `search`, `facts`, `answer`) when it returns; `search` is embed + retrieve + rerank |
| `replace` | `{text, source?}` | Replace the answer text: a retry starts (`""`), the self-rating tag is stripped from the final local answer, or a cloud/cached answer arrives |
| `done` | `{metadata, wall}` | Final; `metadata` has the same shape as `/query/stream`'s done frame |
| `error` | `{message}` | Instead of `done` |

`metrics` fields:
- `t`: seconds since the request started.
- `host`: whose CPU / GPU / RAM this is: `jetson` (tegrastats) or `mini` (PazRAG `GET /sys/stream`, read once a call goes to the mini).
- `cpu`: mean of the online cores, %.
- `gpu`: `GR3D_FREQ` on the Jetson, the Apple GPU's utilisation (`ioreg`) on the mini, %.
- `ram`: MB used.
- `tok`: cumulative output tokens across all calls.
- `tps`: tokens over the trailing second.
- `kv`: KV-cache bytes for the current call, estimated as `(prompt tokens + tokens so far) × 2 × layers × kv_heads × head_dim × 2` (f16). Prompt tokens are counted at about 4 characters each until Ollama reports `prompt_eval_count`. `kv` is `null` for models missing from `telemetry.MODEL_SPECS`.
- `n`: tegrastats samples read.

Each request runs its own `tegrastats --interval 125` subprocess, which is killed when
the response ends, including when the client disconnects; the mini metrics stream is
closed the same way. Private answers from the mini stream live token by token;
documents answers arrive whole (their words don't stream yet).

### `POST /brain/query`

Read-only queries over the PazBrain Logseq graph. Requires the same auth as
`/brain/api` reads (a read token, a write token, or an allowlisted Firebase ID token).
The body is one of six operations, chosen by `op`:
- `search_page {q}`
- `search_block {q}`
- `show_page {slug, level 1-3}`
- `list_task {status?, limit ≤50}`
- `recent_updated {days ≤90}`
- `task_search {status, title?, days}`

Each caller is rate limited (`PAZBRAIN_QUERY_RATE_PER_MIN`, default 30/min; `429` with
`Retry-After` beyond that). The container has no logseq CLI. Requests are forwarded over
`/opt/pazbrain/run/brain-query.sock` to the host sidecar `pazbrain/scripts/brain_query.py`
(`pazbrain-query.service`). The sidecar validates again, runs only fixed CLI argument
lists (no shell, no write operations, no raw datalog), and caps result sizes.

### `GET /health`

Returns Ollama reachability, version, configured cloud API keys, and the confidence threshold.

### `GET /stats`

Returns in-memory routing history: total queries, breakdown by source/skill/model, average latency per model.

### `GET /tiers/status`

Health of each lane: `local` (Jetson Ollama), `private` (mini Ollama), `docs` (the mini's
document index) and `cloud` (configured providers outside a quota or billing window).
The Ollama lanes are probed via `/api/tags` and cached 15 s.

### `POST /jobs/generate`

`{prompt, system?, format?, num_predict?, temperature?}` → one call on the mini, never the
cloud or the Jetson (503 when the mini is unavailable). Needs a private identity. See
[Tiered routing](#tiered-routing).
