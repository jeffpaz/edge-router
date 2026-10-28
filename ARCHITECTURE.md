# edge-router architecture

edge-router is the AskJetson router. It runs on the Jetson Orin Nano and decides, per
request, which of three lanes answers: the Jetson itself, the Mac mini, or a cloud
provider. The choice depends on data sensitivity and job type, not on guessed difficulty.

## Machines

```
 browser (pazlabs.io/chat)
    │ HTTPS, Firebase ID token when signed in
    ▼
 Cloudflare tunnel ── api.pazlabs.io ──▶ Jetson Orin Nano  192.168.11.50   (8 GB, JetPack 6)
                                          ├─ edge-router container  :8000  (this repo)
                                          ├─ ollama container       :11434 llama3.2:3b
                                          ├─ PazBrain host jobs: Drive sync, Logseq sidecar,
                                          │   wiki → mini export
                                          └─ (Argus, unrelated)
                                                 │ LAN only
                                                 ▼
                                         Mac mini  192.168.11.60   (Apple M2, 8 GB, FileVault)
                                          ├─ Ollama :11434  qwen3:4b-instruct + bge-m3
                                          │   (shared with tripBuddy; num_ctx 8192)
                                          └─ PazRAG :8100   personal documents + PazBrain notes
                                              (bearer token; /rag/query, /sys/stream)

 cloud: Claude · Gemini · Grok (Responses API web_search / x_search) · OpenAI
```

The mini's services listen on its LAN address only and are never exposed through
Cloudflare. Personal documents reach the mini by an hourly one-way rsync from the main Mac
(`~/Documents` → `~/PrivateDocs`). PazBrain notes reach it by a one-way export from the
Jetson. Nothing on the mini can write back to the Jetson, Logseq or Drive.

## Lanes

| Lane | Answers | Never |
|---|---|---|
| `local`, the Jetson | Local-only skills (conversational, definition, creative, how-to, opinion, language) and confident local answers | — |
| `private`, the mini | `private: true` requests, personal-documents and PazBrain questions, `/jobs/*`, and cloud-down fallback | Calls a cloud provider |
| `cloud` | Every other interactive question; live-data questions go to Grok with server-side search | Sees a private request |

### A public request

```
classify (regex skills + realtime_classifier, no LLM)
  ├─ live data ─────────────▶ Grok web_search / x_search ─┐
  ├─ local-only skill ──────▶ Jetson                       │
  └─ otherwise ─────────────▶ Jetson ─ confident? ─ yes ──┤
                                         └─ no ─▶ skill-matched cloud provider
cloud down or timed out (CLOUD_LANE_TIMEOUT_S) ─▶ mini ─▶ Jetson ─▶ clear error
```

### A private request

A request is private when the client sends `private: true` (the web chat's "Stay
private / local" checkbox), when `personal_docs.py` detects a question about the user's
own records, or when it names PazBrain. Then:

1. **Who:** `auth.private_identity()` requires a verified Firebase email in
   `PRIVATE_ALLOWED_EMAILS` (or `PRIVATE_TOKEN` for scripts). Otherwise the request gets 401
   or 403 and no model runs. A source-IP check can't work: tunnel traffic arrives from a
   Docker bridge address.
2. **Where:** documents and PazBrain questions go to PazRAG `/rag/query`, with no fallback.
   Other private questions try the document index first, then the mini's model, then the
   Jetson, then a clear error. Live-data questions are answered locally with
   `limitation: "live_data_unavailable"`.
3. **Two guards against the cloud:** the private lane has no cloud client in it, and
   `privacy.private_scope()` makes every cloud call path raise `PrivateLeakError`.
4. **Nothing kept:** no answer cache, no server-side session, and no question or answer
   text in logs (metadata only). The web chat also leaves private turns out of later
   history.

## Runner options

Ollama reloads a model whenever runner options differ from the loaded runner. Every Jetson
call sends `num_ctx` 4096 and `num_thread` 6. Every mini call (router, PazRAG and tripBuddy)
sends `num_ctx` 8192. All of them send `keep_alive: -1`. That keeps one runner per host
loaded.

## Streaming and telemetry

`POST /ask/stream` (the web chat) streams named SSE events: `call`, `token`, `metrics`,
`phases`, `replace`, `done`.

- **Jetson answers:** a per-request `tegrastats --interval 125` sampler, killed when the
  response ends.
- **Mini answers:** the mini's own `GET /sys/stream` (CPU, GPU via `ioreg`, RAM). Private
  mini answers stream live with Ollama's load / prefill / decode. Documents answers report
  PazRAG's steps (filters, search, facts, answer) when they return.
- `metrics.host` says which machine a sample came from. The web panel shows it as a chip.

## PazBrain

- **Wiki and API:** the wiki lives on the Jetson (`~/pazbrain/wiki`, synced from Google
  Drive every 15 minutes). `/brain/api` serves it, and `/brain/query` runs allow-listed,
  read-only Logseq queries through a host sidecar.
- **Export to the mini:** `pazbrain/scripts/export_for_mini.py` renders the wiki (not the
  thinner Logseq graph) into a deterministic tar. It pushes the tar over SSH with a key
  that the mini restricts to `scripts/pazbrain_receive.sh` (`from=` the Jetson, `restrict`,
  forced command). It skips unchanged exports and refuses empty or much smaller ones.

## Code map

| File | Role |
|---|---|
| `main.py` | FastAPI app, endpoints, `_resolve_query()`, the private and docs lanes |
| `router.py` | Skill classifier and the skill → cloud provider map |
| `realtime_classifier.py` | Live-data detection |
| `personal_docs.py` | Personal-documents and PazBrain intent |
| `tiers.py` | Lane health; mini, document-index and mini-metrics clients |
| `privacy.py` | Per-request private flag and the cloud guard |
| `auth.py` | Firebase ID token verification, `private_identity()` |
| `local_llm.py` | Jetson Ollama client, confidence scoring, streaming |
| `cloud_llms.py` | Claude, Gemini, Grok, OpenAI clients |
| `telemetry.py` | `Run`: tegrastats, mini metrics, KV estimate, SSE events |
| `pazbrain/` | Wiki API, Logseq query routes, host scripts, systemd units |
| `trip_routes.py` | tripBuddy's planning endpoint |

## Related repos

- **PazRAG** (`github.com/jeffpaz/PazRAG`): the mini's document and notes index and its
  query service. See its `ARCHITECTURE.md`.
- **pazlabs.io**: the web chat (`/chat`), with the private checkbox, citations and the
  "This run" panel.
