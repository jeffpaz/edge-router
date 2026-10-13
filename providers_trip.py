"""tripBuddy providers for the edge router — additive, used only by POST /trip/query.

Three destinations:
  local   Ollama on the Mac mini (TRIPBUDDY_OLLAMA_URL / TRIPBUDDY_OLLAMA_MODEL), JSON mode
  gemini  Google AI Studio free-tier key (TRIPBUDDY_GEMINI_API_KEY) via the google-genai SDK
  grok    xAI Responses API with the server-side web_search tool (reuses XAI_API_KEY)

Nothing here touches the shared router state in router.py: degraded tracking and
daily caps are tripBuddy-only, so pazlabs.io routing is unaffected.
"""
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

import config

_log = logging.getLogger("edge_router.trip")

# ---------------------------------------------------------------------------
# Settings (all optional except the keys; read from the container env / .env)
# ---------------------------------------------------------------------------

OLLAMA_URL       = os.getenv("TRIPBUDDY_OLLAMA_URL", "http://192.168.11.60:11434").rstrip("/")
OLLAMA_MODEL     = os.getenv("TRIPBUDDY_OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_CTX   = int(os.getenv("TRIPBUDDY_OLLAMA_NUM_CTX", "8192"))

GEMINI_API_KEY   = os.getenv("TRIPBUDDY_GEMINI_API_KEY", "")
GEMINI_MODELS    = [m.strip() for m in os.getenv(
    "TRIPBUDDY_GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash").split(",") if m.strip()]
GEMINI_GROUNDING = os.getenv("TRIPBUDDY_GEMINI_GROUNDING", "false").lower() == "true"

GROK_MODEL       = os.getenv("TRIPBUDDY_GROK_MODEL", "grok-4-fast")

GEMINI_DAILY_CAP = int(os.getenv("TRIPBUDDY_GEMINI_DAILY_CAP", "100"))
GROK_DAILY_CAP   = int(os.getenv("TRIPBUDDY_GROK_DAILY_CAP", "25"))

USAGE_DB         = os.getenv("TRIPBUDDY_USAGE_DB", "/app/data/tripbuddy_usage.db")

_DEGRADED_TTL = 1800.0  # 30 minutes, same policy as router.py but a separate table


class TripProviderError(Exception):
    def __init__(self, provider: str, message: str, status_code: int | None = None, quota: bool = False):
        self.provider = provider
        self.status_code = status_code
        self.quota = quota
        super().__init__(f"[{provider}] {message}")


@dataclass
class TripLLMResponse:
    answer: str
    provider: str          # "local" | "gemini" | "grok"
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    grounded: bool = False


# ---------------------------------------------------------------------------
# Daily caps — SQLite in the mounted data dir so counts survive restarts.
# Day boundary is UTC.
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(USAGE_DB, timeout=5)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage (day TEXT, provider TEXT, calls INTEGER, "
        "PRIMARY KEY (day, provider))"
    )
    return conn


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def usage_today() -> dict[str, int]:
    with _db_lock, _db() as conn:
        rows = conn.execute("SELECT provider, calls FROM usage WHERE day = ?", (_today(),)).fetchall()
    counts = {"gemini": 0, "grok": 0}
    counts.update({p: c for p, c in rows})
    return counts


def cap_for(provider: str) -> int:
    return {"gemini": GEMINI_DAILY_CAP, "grok": GROK_DAILY_CAP}.get(provider, 0)


def try_reserve(provider: str) -> bool:
    """Atomically count one call against today's cap. False when the cap is spent."""
    with _db_lock, _db() as conn:
        row = conn.execute(
            "SELECT calls FROM usage WHERE day = ? AND provider = ?", (_today(), provider)
        ).fetchone()
        used = row[0] if row else 0
        if used >= cap_for(provider):
            return False
        conn.execute(
            "INSERT INTO usage (day, provider, calls) VALUES (?, ?, 1) "
            "ON CONFLICT(day, provider) DO UPDATE SET calls = calls + 1",
            (_today(), provider),
        )
        return True


# ---------------------------------------------------------------------------
# tripBuddy-only degraded tracking (quota / billing / auth failures)
# ---------------------------------------------------------------------------

_degraded: dict[str, float] = {}


def is_degraded(provider: str) -> bool:
    expiry = _degraded.get(provider)
    if expiry is None:
        return False
    if time.time() < expiry:
        return True
    del _degraded[provider]
    return False


def mark_degraded(provider: str) -> None:
    _degraded[provider] = time.time() + _DEGRADED_TTL
    _log.warning("trip_provider_degraded", extra={"provider": provider, "ttl_s": _DEGRADED_TTL})


def is_configured(provider: str) -> bool:
    if provider == "gemini":
        return bool(GEMINI_API_KEY)
    if provider == "grok":
        return bool(config.XAI_API_KEY)
    return True


# ---------------------------------------------------------------------------
# Local — Ollama on the Mac mini
# ---------------------------------------------------------------------------

async def query_local(prompt: str, system: str, json_output: bool, json_schema: dict | None) -> TripLLMResponse:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.2},
    }
    if json_schema:
        body["format"] = json_schema
    elif json_output:
        body["format"] = "json"

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
            r.raise_for_status()
            d = r.json()
    except httpx.HTTPError as exc:
        raise TripProviderError("local", f"ollama unreachable or failed: {exc}") from exc
    return TripLLMResponse(
        answer=d.get("message", {}).get("content", ""),
        provider="local",
        model=OLLAMA_MODEL,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_tokens=d.get("prompt_eval_count", 0),
        output_tokens=d.get("eval_count", 0),
    )


# ---------------------------------------------------------------------------
# Gemini — google-genai SDK, AI Studio key with no billing attached
# ---------------------------------------------------------------------------

async def query_gemini(prompt: str, system: str, json_output: bool, json_schema: dict | None) -> TripLLMResponse:
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg: dict = {"system_instruction": system or None, "temperature": 0.3}
    if GEMINI_GROUNDING:
        # Structured output can't be combined with tools; JSON comes from the prompt instead.
        cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    elif json_schema:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = json_schema
    elif json_output:
        cfg["response_mime_type"] = "application/json"

    last: Exception | None = None
    for model in GEMINI_MODELS:
        t0 = time.perf_counter()
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
            usage = resp.usage_metadata
            return TripLLMResponse(
                answer=resp.text or "",
                provider="gemini",
                model=model,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                grounded=GEMINI_GROUNDING,
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code in (429, 401, 403):
                raise TripProviderError("gemini", str(exc)[:300], code, quota=True) from exc
            # 404 (model retired for this key) / 5xx (high demand): try the next model
            _log.warning("trip_gemini_model_failed", extra={"model": model, "status_code": code})
            last = exc
        except Exception as exc:  # network, SDK
            last = exc
    raise TripProviderError("gemini", f"all models failed: {last}"[:300])


# ---------------------------------------------------------------------------
# Grok — xAI Responses API with server-side web search
# ---------------------------------------------------------------------------

async def query_grok(prompt: str, system: str, json_output: bool, json_schema: dict | None) -> TripLLMResponse:
    sys_text = system
    if json_output or json_schema:
        sys_text = (sys_text + "\n\n" if sys_text else "") + "Respond with a single valid JSON object only."
    inputs = []
    if sys_text:
        inputs.append({"role": "system", "content": sys_text})
    inputs.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            r = await client.post(
                "https://api.x.ai/v1/responses",
                headers={"Authorization": f"Bearer {config.XAI_API_KEY}"},
                json={"model": GROK_MODEL, "input": inputs, "tools": [{"type": "web_search"}]},
            )
    except httpx.HTTPError as exc:
        raise TripProviderError("grok", f"connection failed: {exc}") from exc
    if r.status_code != 200:
        raise TripProviderError(
            "grok", r.text[:300], r.status_code, quota=r.status_code in (402, 403, 429)
        )
    d = r.json()
    text = ""
    for item in d.get("output", []):
        if item.get("type") == "message":
            text = "".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text")
    usage = d.get("usage", {})
    return TripLLMResponse(
        answer=text,
        provider="grok",
        model=d.get("model", GROK_MODEL),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        grounded=True,
    )


PROVIDERS = {"local": query_local, "gemini": query_gemini, "grok": query_grok}
