"""Routing lanes: LOCAL (Jetson Ollama), PRIVATE (Mac mini Ollama), CLOUD.

Lanes are chosen by data sensitivity and job type, not by guessed difficulty:

  LOCAL    every request is classified here (router.SkillRouter + realtime_classifier,
           no LLM call), and local-only skills / confident local answers are served here
  PRIVATE  batch jobs (POST /jobs/*), private data (later phases), and the fallback
           when the cloud is unreachable
  CLOUD    every other interactive query

Non-private fallback order on cloud timeout/5xx: cloud → mini → Jetson.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass

import httpx

import config
import router as _router

_log = logging.getLogger("edge_router.tiers")

LOCAL = "local"
PRIVATE = "private"
CLOUD = "cloud"

# Every call to the mini sends this one runner-option set, for the same reason as
# local_llm._RUNNER_OPTIONS: Ollama reloads the model whenever a runner option
# differs from the loaded runner. num_ctx matches tripBuddy's calls to the same host.
MINI_RUNNER_OPTIONS: dict = {"num_ctx": config.MINI_NUM_CTX}
KEEP_ALIVE = -1

HEALTH_TTL = 15.0  # seconds a lane health probe is reused


class MiniUnavailable(Exception):
    """The PRIVATE lane is disabled, unhealthy, or the call failed."""


class DocsUnavailable(Exception):
    """The personal-docs service (MINI_RAG_URL) is unset, down, or refused the call."""


def mini_enabled() -> bool:
    return bool(config.MINI_OLLAMA_URL and config.MINI_MODEL)


def docs_enabled() -> bool:
    return bool(config.MINI_RAG_URL and config.MINI_RAG_TOKEN)


@dataclass
class RouteInfo:
    """Routing fields included in every response."""
    tier: str | None        # LOCAL | PRIVATE | CLOUD; None when nothing could answer
    model: str
    private: bool = False
    fallback_used: bool = False
    route_reason: str = ""

    def fields(self) -> dict:
        return asdict(self)


def tier_of(source: str) -> str | None:
    """Map a response `source` ("local", "mini", a cloud provider, "error") to its tier."""
    if source == "local":
        return LOCAL
    if source == "mini":
        return PRIVATE
    if source == "error":
        return None
    return CLOUD


# ---------------------------------------------------------------------------
# Health  (Ollama /api/tags, cached HEALTH_TTL seconds per lane)
# ---------------------------------------------------------------------------

_health: dict[str, tuple[float, dict]] = {}


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


async def _probe_ollama(url: str, model: str) -> dict:
    t0 = time.perf_counter()
    try:
        async with _client(3.0) as client:
            r = await client.get(f"{url}/api/tags")
            r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        up = model in names or f"{model}:latest" in names
        result = {"up": up, "model": model}
        if not up:
            result["error"] = "model not pulled"
    except Exception as exc:
        result = {"up": False, "model": model, "error": type(exc).__name__}
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


async def lane_health(lane: str) -> dict:
    """Health of an Ollama lane (LOCAL or PRIVATE), cached HEALTH_TTL seconds."""
    if lane == PRIVATE and not mini_enabled():
        return {"up": False, "model": config.MINI_MODEL, "error": "disabled (MINI_OLLAMA_URL / MINI_MODEL unset)"}
    cached = _health.get(lane)
    if cached and time.monotonic() < cached[0]:
        return cached[1]
    if lane == PRIVATE:
        result = await _probe_ollama(config.MINI_OLLAMA_URL, config.MINI_MODEL)
    else:
        result = await _probe_ollama(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    _health[lane] = (time.monotonic() + HEALTH_TTL, result)
    return result


async def docs_health() -> dict:
    """The mini's document index service (GET /health), cached HEALTH_TTL seconds."""
    if not docs_enabled():
        return {"up": False, "error": "disabled (MINI_RAG_URL / MINI_RAG_TOKEN unset)"}
    cached = _health.get("docs")
    if cached and time.monotonic() < cached[0]:
        return cached[1]
    t0 = time.perf_counter()
    try:
        async with _client(3.0) as client:
            r = await client.get(f"{config.MINI_RAG_URL}/health")
            r.raise_for_status()
        body = r.json()
        result = {"up": True, "chunks": body.get("chunks"), "sources": body.get("sources", [])}
    except Exception as exc:
        result = {"up": False, "error": type(exc).__name__}
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _health["docs"] = (time.monotonic() + HEALTH_TTL, result)
    return result


async def docs_query(query: str) -> dict:
    """POST /rag/query on the mini. Returns {answer, citations, weak_evidence, model, ...}.
    Raises DocsUnavailable; the caller never falls back anywhere (no other host has the
    documents, and they must not leave the private tier). Never logs the query."""
    if not docs_enabled():
        raise DocsUnavailable("personal-docs lane disabled")
    try:
        async with _client(config.MINI_RAG_TIMEOUT_S) as client:
            r = await client.post(
                f"{config.MINI_RAG_URL}/rag/query",
                json={"query": query},
                headers={"Authorization": f"Bearer {config.MINI_RAG_TOKEN}"},
            )
            r.raise_for_status()
            return r.json()
    except (httpx.HTTPError, ValueError) as exc:
        _health.pop("docs", None)
        _log.warning("docs_call_failed", extra={"error_type": type(exc).__name__})
        raise DocsUnavailable(type(exc).__name__) from exc


def cloud_health() -> dict:
    """Configured cloud providers minus those in a quota/billing degraded window.
    No network probe here: GET /providers/status does that, cached 5 minutes."""
    keys = {
        "claude": config.ANTHROPIC_API_KEY,
        "gemini": config.GOOGLE_API_KEY,
        "grok":   config.XAI_API_KEY,
        "openai": config.OPENAI_API_KEY,
    }
    configured = [p for p, key in keys.items() if key]
    available = [p for p in configured if not _router._is_degraded(p)]
    return {"up": bool(available), "providers": available, "degraded": sorted(set(configured) - set(available))}


async def status() -> dict:
    local, mini, docs = await asyncio.gather(lane_health(LOCAL), lane_health(PRIVATE), docs_health())
    return {
        LOCAL:   {"enabled": True, "host": "jetson", **local},
        PRIVATE: {"enabled": mini_enabled(), "host": "mac-mini", **mini},
        "docs":  {"enabled": docs_enabled(), "host": "mac-mini", **docs},
        CLOUD:   {"enabled": True, **cloud_health()},
        "health_ttl_s": HEALTH_TTL,
    }


# ---------------------------------------------------------------------------
# PRIVATE lane call
# ---------------------------------------------------------------------------

@dataclass
class MiniResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


async def mini_generate(
    prompt: str,
    system: str = "",
    messages: list = [],
    *,
    num_predict: int = 600,
    temperature: float = 0.7,
    format: dict | str | None = None,
) -> MiniResult:
    """One non-streaming chat call to the mini. Raises MiniUnavailable instead of
    falling back anywhere; the caller decides what comes next.

    Never logs prompt or answer text: this lane carries private requests.
    """
    if not mini_enabled():
        raise MiniUnavailable("private lane disabled")
    health = await lane_health(PRIVATE)
    if not health["up"]:
        raise MiniUnavailable(f"mini unhealthy ({health.get('error', 'down')})")

    chat = [{"role": "system", "content": system}] if system else []
    chat += [{"role": m.get("role"), "content": m.get("content", "")} for m in messages if isinstance(m, dict)]
    chat.append({"role": "user", "content": prompt})
    body: dict = {
        "model": config.MINI_MODEL,
        "messages": chat,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {**MINI_RUNNER_OPTIONS, "num_predict": num_predict, "temperature": temperature},
    }
    if format is not None:
        body["format"] = format

    t0 = time.perf_counter()
    try:
        async with _client(config.MINI_TIMEOUT_S) as client:
            r = await client.post(f"{config.MINI_OLLAMA_URL}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        _health.pop(PRIVATE, None)  # re-probe on the next call
        _log.warning("mini_call_failed", extra={"error_type": type(exc).__name__})
        raise MiniUnavailable(type(exc).__name__) from exc

    return MiniResult(
        answer=(data.get("message") or {}).get("content", ""),
        model=config.MINI_MODEL,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
