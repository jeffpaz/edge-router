# edge-router

Routes LLM queries to a **local Ollama model** first. If the response confidence falls below a configurable threshold, the query is automatically escalated to a cloud provider (Claude, Gemini, Grok, or OpenAI).

```
POST /query  (or /query/stream)
  │
  ├─ realtime_classifier ──is_realtime──► cloud LLM (Grok/OpenAI) ──► return
  │
  ├─ local Ollama ──confidence OK──► return response
  │
  └─ confidence low ───────────────► cloud LLM ──► return response
```

## Project layout

```
edge-router/
├── main.py                FastAPI app, /query, /query/stream, /health endpoints
├── router.py              Skill classifier + routing logic
├── local_llm.py           Ollama client with confidence scoring and SSE streaming
├── cloud_llms.py          Clients for Claude, Gemini, Grok, OpenAI
├── realtime_classifier.py Real-time intent detector (sports/stocks/news bypass)
├── config.py              API keys and settings from environment variables
└── requirements.txt
```

---


## Realtime bypass

Before sending a query to Ollama, `realtime_classifier.classify_intent()` checks whether the query requires live data (sports scores, stock prices, breaking news, live events). If it does, the local model is skipped and the query routes directly to:

- **Grok** — sports scores, match results, sports entities
- **OpenAI** — stock prices, financial data, breaking news

The `/query` response includes `realtime_intent: true` and a `realtime_signals` list when this path is taken. The `/query/stream` endpoint applies the same check and fake-streams the cloud response to maintain streaming UX.

---

## Setup on Nvidia Jetson Orin Nano

### 1. Flash JetPack 6.x

Use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) or the Jetson Orin Nano SD card image.  
Verify CUDA is available after boot:

```bash
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama auto-detects the Jetson GPU via CUDA. Confirm:

```bash
ollama run llama3.2:3b "Hello from Jetson"
```

For heavier workloads (8 GB RAM Orin Nano) keep to 3B–7B models.  
Recommended: `llama3.2:3b`, `mistral:7b`, `phi3:mini`.

### 3. Create a Python virtual environment

```bash
sudo apt-get install -y python3-venv python3-pip
python3 -m venv ~/venvs/edge-router
source ~/venvs/edge-router/bin/activate
```

### 4. Clone and install dependencies

```bash
git clone <your-repo-url> ~/edge-router
cd ~/edge-router
pip install --upgrade pip
pip install -r requirements.txt
```

> **Jetson note:** If `httpx` or `uvicorn` wheels fail to build, install build tools first:
> ```bash
> sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
> ```

### 5. Configure environment variables

Copy and edit the example:

```bash
cat > ~/.env.edge-router << 'EOF'
# Local model
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Confidence threshold (0.0–1.0). Queries below this score are escalated.
CONFIDENCE_THRESHOLD=0.65

# Which cloud provider to use as fallback: claude | gemini | grok | openai
FALLBACK_PROVIDER=claude

# Cloud API keys (only the fallback provider key is required)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
XAI_API_KEY=xai-...
OPENAI_API_KEY=sk-...
EOF
```

Load before running:

```bash
set -a && source ~/.env.edge-router && set +a
```

### 6. Start the server

```bash
cd ~/edge-router
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

`--workers 1` is recommended on Orin Nano to avoid GPU memory contention between Ollama and multiple Python processes.

### 7. Test it

```bash
# Basic query — tries local model first, escalates to cloud if confidence is low
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?"}' | python3 -m json.tool

# Force cloud (skip local model entirely)
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a haiku about Jetson.", "force_cloud": true}' \
  | python3 -m json.tool

# Health — shows Ollama status and which cloud APIs are configured
curl http://localhost:8000/health

# Stats — routing history, avg latency per model
curl http://localhost:8000/stats
```

### 8. Run as a systemd service

```bash
# Create the env file from the template and fill in API keys
sudo mkdir -p /etc/edge-router
sudo cp ~/edge-router/env.example /etc/edge-router/env
sudo chmod 600 /etc/edge-router/env
sudo nano /etc/edge-router/env

# Install and enable the service
sudo cp ~/edge-router/edge-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edge-router
sudo systemctl start edge-router

# Verify
sudo systemctl status edge-router
sudo journalctl -u edge-router -f
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local model tag |
| `CONFIDENCE_THRESHOLD` | `0.70` | Min score [0–1] to accept local response |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model override |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Google model override |
| `GROK_MODEL` | `grok-3-mini` | xAI model override |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model override |

Skill → cloud LLM mapping (fixed in `router.py`):

| Skill | Provider | Model |
|---|---|---|
| `coding` | Claude | `claude-sonnet-4-20250514` |
| `math_data` | Gemini | `gemini-1.5-pro` |
| `current_events` | Grok | `grok-2-latest` |
| `general` | OpenAI | `gpt-4o` |

## API

### `POST /query`

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | User query text |
| `force_cloud` | bool | no | Skip local model, go straight to cloud (default `false`) |
| `system` | string | no | System prompt forwarded to whichever LLM answers |

Response:

| Field | Type | Description |
|---|---|---|
| `answer` | string | Model response |
| `source` | string | `"local"` or cloud provider name |
| `model_used` | string | Exact model ID that answered |
| `skill` | string | Classified query type |
| `latency_ms` | float | Total wall-clock time for the request |
| `local_confidence` | float\|null | Confidence score from local model; null when `force_cloud=true` |
| `tokens_used` | int | Total tokens consumed |


### `POST /query/stream`

Same request body as `/query`. Returns a Server-Sent Events (SSE) stream.

Each event is `data: <json>`. Token chunks: `{"chunk": "Hello"}`

Done event (final):

```json
{
  "done": true,
  "routed_to": "local",
  "source": "local",
  "model_used": "gemma2:2b",
  "skill": "general",
  "latency_ms": 1234,
  "confidence_score": null,
  "realtime_intent": false,
  "tokens": { "input": 42, "output": 87, "total": 129 },
  "prompt_tokens": 42,
  "completion_tokens": 87
}
```

For cloud-routed responses, `routed_to` is the provider name (e.g. `"claude"`, `"grok"`).

### `GET /health`

Returns Ollama reachability, version, configured cloud API keys, and the confidence threshold.

### `GET /stats`

Returns in-memory routing history: total queries, breakdown by source/skill/model, average latency per model.
