import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import uuid
from collections import deque, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import cloud_llms
import config
import local_llm
import router as _router
from realtime_classifier import classify as classify_intent


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

_BASELINE_LOG_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        for k, v in record.__dict__.items():
            if k not in _BASELINE_LOG_ATTRS and not k.startswith("_"):
                doc[k] = v
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


def _configure_logging() -> None:
    root = logging.getLogger("edge_router")
    # Remove NullHandlers added by library-style imports in submodules
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]
    if root.handlers:
        return  # Real handler already installed
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


_configure_logging()
_log = logging.getLogger("edge_router.main")


# ---------------------------------------------------------------------------
# In-memory recent-queries store  (last 100, for /stats health checks)
# ---------------------------------------------------------------------------

class _StatsStore:
    """Bounded deque of the last 100 query records."""

    def __init__(self) -> None:
        self.total: int = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=100)

    def record(self, entry: dict[str, Any]) -> None:
        self.total += 1
        self._recent.append(entry)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_since_start": self.total,
            "recent_queries": list(self._recent),
        }


_stats = _StatsStore()


# ---------------------------------------------------------------------------
# Local answer cache  (in-memory, restart-cleared, max CACHE_MAX_SIZE entries)
# ---------------------------------------------------------------------------

_QUERY_CACHE: OrderedDict = OrderedDict()


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _cache_get(query: str) -> dict | None:
    key = _cache_key(query)
    entry = _QUERY_CACHE.get(key)
    if entry is not None:
        _log.info(
            "cache_hit query_preview=%.50s",
            query[:50],
            extra={"event": "cache_hit", "query_preview": query[:50]},
        )
        return entry
    _log.info(
        "cache_miss query_preview=%.50s",
        query[:50],
        extra={"event": "cache_miss", "query_preview": query[:50]},
    )
    return None


def _cache_put(query: str, data: dict, skill: str, confidence: float) -> None:
    threshold = config.SKILL_THRESHOLDS.get(skill, config.CONFIDENCE_THRESHOLD)
    if confidence < threshold:
        return
    key = _cache_key(query)
    if key in _QUERY_CACHE:
        del _QUERY_CACHE[key]
    _QUERY_CACHE[key] = data
    while len(_QUERY_CACHE) > config.CACHE_MAX_SIZE:
        _QUERY_CACHE.popitem(last=False)  # evict oldest
    _log.info(
        "cache_store skill=%s confidence=%.4f size=%d",
        skill,
        confidence,
        len(_QUERY_CACHE),
        extra={"event": "cache_store", "skill": skill, "confidence": confidence, "cache_size": len(_QUERY_CACHE)},
    )


# ---------------------------------------------------------------------------
# Conversation history  (in-memory, keyed by session_id, max 50 sessions,
# evict oldest; each session capped at the last 10 messages / 5 turns)
# ---------------------------------------------------------------------------

_SESSION_MAX_SESSIONS = 50
_SESSION_MAX_MESSAGES = 10  # 5 turns

_sessions: OrderedDict[str, list[dict]] = OrderedDict()


def _trim_messages(messages: list) -> list:
    """Keep only the most recent _SESSION_MAX_MESSAGES entries, oldest trimmed first."""
    return messages[-_SESSION_MAX_MESSAGES:] if messages else []


def _session_context(session_id: str | None, messages: list) -> list:
    """Trimmed conversation history for this turn.

    Client-sent messages win when present (the frontend already tracks its own
    history); server-stored session history is the fallback for clients that
    only send session_id.
    """
    if messages:
        return _trim_messages(messages)
    if session_id and session_id in _sessions:
        return _trim_messages(_sessions[session_id])
    return []


def _session_update(session_id: str | None, query: str, answer: str) -> None:
    """Append this turn to the session's history, evicting the oldest session past the cap."""
    if not session_id or not answer:
        return
    history = _sessions.get(session_id, [])
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    _sessions[session_id] = _trim_messages(history)
    _sessions.move_to_end(session_id)
    while len(_sessions) > _SESSION_MAX_SESSIONS:
        _sessions.popitem(last=False)


# ---------------------------------------------------------------------------
# Billing probe cache  (5-minute TTL — avoids burning quota on status checks)
# ---------------------------------------------------------------------------

_BILLING_TTL: float = 300.0  # seconds
_billing_cache: dict[str, dict] = {}


def _classify_billing_exc(exc: Exception) -> str:
    """Return 'quota' for billing/credit/rate-limit errors, 'error' for everything else."""
    err_str = str(exc).lower()
    quota_keywords = (
        "quota", "credit", "billing", "insufficient", "rate limit", "rate_limit",
        "payment", "balance", "permission", "does not have permission",
    )
    if any(kw in err_str for kw in quota_keywords):
        return "quota"
    for attr in ("status_code", "status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and code in (400, 402, 403, 429):
            return "quota"
    return "error"


async def _billing_probe(provider: str) -> dict:
    """Make one minimal API call (no retries) and return {status, error_detail}."""
    try:
        if provider == "claude":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
            await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "1"}],
            )

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            gmodel = genai.GenerativeModel(model_name=config.GEMINI_MODEL)
            await gmodel.generate_content_async("1")

        elif provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            await client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "1"}],
            )

        elif provider == "grok":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=config.XAI_API_KEY, base_url="https://api.x.ai/v1")
            await client.chat.completions.create(
                model=config.GROK_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "1"}],
            )

        else:
            return {"status": "error", "error_detail": None}

        return {"status": "ok", "error_detail": None}

    except Exception as exc:
        detail = str(exc)
        # Try to pull a cleaner message from HTTP exceptions
        for attr in ("message", "args"):
            val = getattr(exc, attr, None)
            if isinstance(val, str) and val:
                detail = val[:200]
                break
            elif isinstance(val, tuple) and val:
                detail = str(val[0])[:200]
                break
        return {"status": _classify_billing_exc(exc), "error_detail": detail}


async def _get_billing_status(provider: str) -> dict:
    """Return cached status if fresh; otherwise probe the provider."""
    now = time.time()
    cached = _billing_cache.get(provider)
    if cached and now < cached["_expires"]:
        return cached

    probe = await _billing_probe(provider)
    entry: dict = {
        "status":       probe["status"],
        "error_detail": probe["error_detail"],
        "checked_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_expires":     now + _BILLING_TTL,
    }
    _billing_cache[provider] = entry
    _log.info(
        "billing_check provider=%s status=%s",
        provider, probe["status"],
        extra={"provider": provider, "status": probe["status"]},
    )
    return entry


# ---------------------------------------------------------------------------
# System metrics  (SQLite collector — GPU/CPU/RAM every 60 s)
# ---------------------------------------------------------------------------

_METRICS_DB = "/app/data/metrics.db"


def _init_metrics_db() -> None:
    os.makedirs(os.path.dirname(_METRICS_DB), exist_ok=True)
    con = sqlite3.connect(_METRICS_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
            cpu_pct      REAL,
            gpu_pct      REAL,
            ram_used_mb  INTEGER,
            ram_total_mb INTEGER
        )
    """)
    con.commit()
    con.close()


def _sample_gpu_pct() -> float:
    """Run tegrastats for one line and parse GR3D_FREQ %."""
    tegrastats = "/host_root/usr/bin/tegrastats"
    if not os.path.exists(tegrastats):
        return 0.0
    try:
        proc = subprocess.Popen(
            [tegrastats],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        line = proc.stdout.readline()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _record_one_sample() -> None:
    import psutil
    cpu = psutil.cpu_percent(interval=1)
    vm  = psutil.virtual_memory()
    gpu = _sample_gpu_pct()
    ram_used  = vm.used  // (1024 * 1024)
    ram_total = vm.total // (1024 * 1024)
    con = sqlite3.connect(_METRICS_DB)
    con.execute(
        "INSERT INTO system_metrics (cpu_pct, gpu_pct, ram_used_mb, ram_total_mb) "
        "VALUES (?, ?, ?, ?)",
        (cpu, gpu, ram_used, ram_total),
    )
    con.execute("DELETE FROM system_metrics WHERE timestamp < datetime('now', '-7 days')")
    con.commit()
    con.close()


async def _metrics_collector_loop() -> None:
    _init_metrics_db()
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(None, _record_one_sample)
            _log.debug("metrics_collected")
        except Exception as exc:
            _log.warning("metrics_collect_error", extra={"error": str(exc)})
        await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info(
        "startup",
        extra={
            "ollama_url": config.OLLAMA_BASE_URL,
            "local_model": config.OLLAMA_MODEL,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "cloud_apis_set": {
                "claude": bool(config.ANTHROPIC_API_KEY),
                "gemini": bool(config.GOOGLE_API_KEY),
                "grok": bool(config.XAI_API_KEY),
                "openai": bool(config.OPENAI_API_KEY),
            },
        },
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/version")
            r.raise_for_status()
            _log.info("ollama_ready", extra={"version": r.json().get("version", "unknown")})
    except Exception as exc:
        _log.warning("ollama_unreachable", extra={"error": str(exc)})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={"model": config.OLLAMA_MODEL, "keep_alive": "10m"},
            )
        _log.info("ollama_warmup_sent")
    except Exception as exc:
        _log.warning("ollama_warmup_failed", extra={"error": str(exc)})

    asyncio.create_task(_metrics_collector_loop())

    yield

    _log.info("shutdown", extra={"total_queries": _stats.total})


app = FastAPI(
    title="edge-router",
    description=(
        "Runs every query through a local Ollama model first. "
        "Escalates to the skill-appropriate cloud LLM when local confidence is low."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — last resort, always returns 200 with JSON
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    _log.error(
        "unhandled_error path=%s method=%s error=%s",
        request.url.path,
        request.method,
        exc,
        extra={"path": request.url.path, "error": str(exc), "error_type": type(exc).__name__},
        exc_info=True,
    )
    return JSONResponse(
        status_code=200,
        content={
            "answer": "Sorry, I ran into an issue processing that request. Please try again.",
            "source": "error",
            "model_used": "none",
            "skill": "unknown",
            "latency_ms": 0,
            "confidence_score": 0,
            "tokens": {"input": 0, "output": 0, "total": 0},
            "session_id": None,
            "error": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text.")
    force_cloud: bool = Field(
        False,
        description="Skip the local model and send directly to the skill-matched cloud LLM.",
    )
    system: str = Field("", description="Optional system prompt forwarded to whichever LLM answers.")
    session_id: str | None = Field(None, description="Optional session identifier from the client.")
    messages: list = Field(default_factory=list, description="Conversation history.")


class TokenBreakdown(BaseModel):
    input: int
    output: int
    total: int


class QueryResponse(BaseModel):
    answer: str
    source: str                        # "local" | cloud provider name | "error"
    model_used: str
    skill: str
    latency_ms: float
    confidence_score: float | None     # local model confidence; None when force_cloud=True
    tokens: TokenBreakdown
    session_id: str | None
    cache_hit: bool = False
    realtime_intent: bool = False
    realtime_signals: list[str] = []
    routing_reason: str | None = None


# ---------------------------------------------------------------------------
# Shared local-first / escalate decision — used by both /query and /query/stream
# so the routing policy (local-only skills, confidence threshold, local-first
# retry, cloud dispatch) lives in exactly one place.
# ---------------------------------------------------------------------------

@dataclass
class _ResolvedQuery:
    source: str             # "local" | cloud provider name | "error"
    answer: str
    model: str
    confidence: float
    input_tokens: int
    output_tokens: int
    local_only: bool = False
    retry: bool = False
    escalated: bool = False
    cloud_latency_ms: float | None = None


async def _resolve_query(query: str, system: str, messages: list, skill: str, req_id: str = "") -> _ResolvedQuery:
    """Run the local model, then decide whether to return it or escalate to cloud.

    Local-only skills and confident local answers return immediately. general/coding
    get one local-first retry with a simplified query before escalating. Everything
    else escalates to the skill-matched cloud LLM via SkillRouter.dispatch(), which
    never raises.
    """
    local = await local_llm.query(query, system, messages, skill=skill)
    _log.info(
        "local_inference",
        extra={
            "req_id":         req_id,
            "skill":          skill,
            "model":          local["model"],
            "confidence":     local["confidence"],
            "should_escalate": local["should_escalate"],
            "signals":        local["signals"],
        },
    )

    if skill in _router.LOCAL_ONLY_SKILLS:
        _log.info(
            "local-only skill — no escalation",
            extra={"req_id": req_id, "skill": skill, "confidence": local["confidence"], "routed_to": "ollama"},
        )
        return _ResolvedQuery(
            source="local", answer=local["answer"], model=local["model"],
            confidence=local["confidence"],
            input_tokens=local["prompt_tokens"], output_tokens=local["completion_tokens"],
            local_only=True,
        )

    if not local["should_escalate"]:
        return _ResolvedQuery(
            source="local", answer=local["answer"], model=local["model"],
            confidence=local["confidence"],
            input_tokens=local["prompt_tokens"], output_tokens=local["completion_tokens"],
        )

    if config.RETRY_ENABLED and skill in ("general", "coding"):
        simplified = _router._simplify_query(query)
        if simplified != query:
            _log.info(
                "local_retry_attempt",
                extra={"req_id": req_id, "skill": skill, "simplified_preview": simplified[:60]},
            )
            retry_local = await local_llm.query(simplified, system, messages, skill=skill)
            if not retry_local["should_escalate"]:
                _log.info(
                    "local retry succeeded — escalation avoided",
                    extra={"req_id": req_id, "skill": skill, "confidence": retry_local["confidence"]},
                )
                return _ResolvedQuery(
                    source="local", answer=retry_local["answer"], model=retry_local["model"],
                    confidence=retry_local["confidence"],
                    input_tokens=retry_local["prompt_tokens"], output_tokens=retry_local["completion_tokens"],
                    retry=True,
                )
            _log.info(
                "local_retry_failed — escalating to cloud",
                extra={"req_id": req_id, "skill": skill, "retry_confidence": retry_local["confidence"]},
            )

    _log.info(
        "escalating_to_cloud",
        extra={
            "req_id":     req_id,
            "skill":      skill,
            "confidence": local["confidence"],
            "threshold":  config.SKILL_THRESHOLDS.get(skill, config.CONFIDENCE_THRESHOLD),
        },
    )
    cloud = await _router.skill_router.dispatch(
        query, skill, system, messages,
        local_answer=local["answer"],
        confidence=local["confidence"],
    )
    return _ResolvedQuery(
        source=cloud.provider, answer=cloud.answer, model=cloud.model,
        confidence=local["confidence"],
        input_tokens=cloud.input_tokens, output_tokens=cloud.output_tokens,
        escalated=True, cloud_latency_ms=cloud.latency_ms,
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse | JSONResponse:
    req_id = uuid.uuid4().hex[:8]
    t_wall = time.perf_counter()

    _log.info(
        "query_received",
        extra={
            "req_id": req_id,
            "session_id": req.session_id,
            "query_preview": req.query[:120],
            "force_cloud": req.force_cloud,
        },
    )

    # ── Cache check — return immediately on hit, skip all inference ──────────
    if not req.force_cloud:
        cached = _cache_get(req.query)
        if cached is not None:
            tokens = TokenBreakdown(**cached["tokens"])
            _stats.record({
                "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "session_id":       req.session_id,
                "source":           cached["source"],
                "model_used":       cached["model_used"],
                "skill":            cached["skill"],
                "latency_ms":       0.0,
                "confidence_score": cached["confidence_score"],
                "tokens":           tokens.model_dump(),
                "status":           "cache_hit",
            })
            _session_update(req.session_id, req.query, cached["answer"])
            return QueryResponse(
                answer=cached["answer"],
                source=cached["source"],
                model_used=cached["model_used"],
                skill=cached["skill"],
                latency_ms=0.0,
                confidence_score=cached["confidence_score"],
                tokens=tokens,
                session_id=req.session_id,
                cache_hit=True,
            )

    skill = _router.skill_router.classify(req.query)
    context_messages = _session_context(req.session_id, req.messages)

    try:
        # ── Realtime intent bypass — skips local LLM for sports/news/price ──────
        _rt = classify_intent(req.query)
        _routing_reason: str | None = None
        if _rt.is_realtime and not req.force_cloud:
            preferred = _rt.preferred_provider
            available = [p for p in ["grok", "openai", "claude"] if not _router._is_degraded(p)]
            selected = preferred if preferred in available else (available[0] if available else None)
            if selected:
                _routing_reason = f"realtime_intent signals={_rt.signals}"
                try:
                    rt_cloud = await cloud_llms.query(selected, req.query, req.system, messages=context_messages)
                    total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
                    tokens   = TokenBreakdown(
                        input=rt_cloud.input_tokens,
                        output=rt_cloud.output_tokens,
                        total=rt_cloud.input_tokens + rt_cloud.output_tokens,
                    )
                    _stats.record({
                        "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        "session_id":       req.session_id,
                        "source":           rt_cloud.provider,
                        "model_used":       rt_cloud.model,
                        "skill":            skill,
                        "latency_ms":       total_ms,
                        "confidence_score": None,
                        "tokens":           tokens.model_dump(),
                        "status":           "success",
                        "realtime_intent":  True,
                        "realtime_signals": _rt.signals,
                        "routing_reason":   _routing_reason,
                    })
                    _log.info(
                        "realtime_intent_routed provider=%s signals=%s",
                        rt_cloud.provider, _rt.signals,
                        extra={
                            "req_id":         req_id,
                            "provider":       rt_cloud.provider,
                            "skill":          skill,
                            "signals":        _rt.signals,
                            "routing_reason": _routing_reason,
                        },
                    )
                    _session_update(req.session_id, req.query, rt_cloud.answer)
                    return QueryResponse(
                        answer=rt_cloud.answer,
                        source=rt_cloud.provider,
                        model_used=rt_cloud.model,
                        skill=skill,
                        latency_ms=total_ms,
                        confidence_score=None,
                        tokens=tokens,
                        session_id=req.session_id,
                        realtime_intent=True,
                        realtime_signals=_rt.signals,
                        routing_reason=_routing_reason,
                    )
                except cloud_llms.ProviderError as rt_exc:
                    if _router._is_billing_error(rt_exc):
                        _router._mark_degraded(selected)
                    _log.warning(
                        "realtime_provider_failed provider=%s falling_through",
                        selected,
                        extra={"provider": selected, "error": str(rt_exc)},
                    )
                    # Fall through to normal local->cloud routing

        if req.force_cloud:
            cloud = await _router.skill_router.dispatch(req.query, skill, req.system, context_messages)
            total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
            tokens   = TokenBreakdown(
                input=cloud.input_tokens,
                output=cloud.output_tokens,
                total=cloud.input_tokens + cloud.output_tokens,
            )
            status = "fallback" if cloud.provider == "error" else "success"
            _stats.record({
                "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "session_id":       req.session_id,
                "source":           cloud.provider,
                "model_used":       cloud.model,
                "skill":            skill,
                "latency_ms":       total_ms,
                "confidence_score": None,
                "tokens":           tokens.model_dump(),
                "status":           status,
            })
            _log.info(
                "routed",
                extra={
                    "req_id":           req_id,
                    "session_id":       req.session_id,
                    "source":           cloud.provider,
                    "model":            cloud.model,
                    "skill":            skill,
                    "cloud_latency_ms": cloud.latency_ms,
                    "total_latency_ms": total_ms,
                    "tokens":           tokens.model_dump(),
                    "forced_cloud":     True,
                    "status":           status,
                },
            )
            _session_update(req.session_id, req.query, cloud.answer)
            return QueryResponse(
                answer=cloud.answer,
                source=cloud.provider,
                model_used=cloud.model,
                skill=skill,
                latency_ms=total_ms,
                confidence_score=None,
                tokens=tokens,
                session_id=req.session_id,
            )

        # ── Local-first / escalate — shared with /query/stream ───────────────
        result = await _resolve_query(req.query, req.system, context_messages, skill, req_id=req_id)
        total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
        tokens   = TokenBreakdown(
            input=result.input_tokens,
            output=result.output_tokens,
            total=result.input_tokens + result.output_tokens,
        )
        if result.source == "local":
            _cache_put(req.query, {
                "answer":           result.answer,
                "source":           "local",
                "model_used":       result.model,
                "skill":            skill,
                "confidence_score": result.confidence,
                "tokens":           tokens.model_dump(),
            }, skill, result.confidence)
            status = "retry_success" if result.retry else "success"
        else:
            status = "fallback" if result.source == "error" else "success"
        _stats.record({
            "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id":       req.session_id,
            "source":           result.source,
            "model_used":       result.model,
            "skill":            skill,
            "latency_ms":       total_ms,
            "confidence_score": result.confidence,
            "tokens":           tokens.model_dump(),
            "status":           status,
        })
        _log.info(
            "routed",
            extra={
                "req_id":           req_id,
                "session_id":       req.session_id,
                "source":           result.source,
                "model":            result.model,
                "skill":            skill,
                "cloud_latency_ms": result.cloud_latency_ms,
                "total_latency_ms": total_ms,
                "tokens":           tokens.model_dump(),
                "forced_cloud":     False,
                "status":           status,
            },
        )
        _session_update(req.session_id, req.query, result.answer)
        return QueryResponse(
            answer=result.answer,
            source=result.source,
            model_used=result.model,
            skill=skill,
            latency_ms=total_ms,
            confidence_score=result.confidence,
            tokens=tokens,
            session_id=req.session_id,
        )

    except httpx.HTTPError as exc:
        # Local Ollama unreachable
        _log.error("upstream_error", extra={"req_id": req_id, "error": str(exc)}, exc_info=True)
        total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
        _stats.record({
            "ts":       datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": req.session_id,
            "source":   "error",
            "skill":    skill,
            "latency_ms": total_ms,
            "status":   "error",
        })
        return JSONResponse(
            status_code=200,
            content={
                "answer": "I encountered an issue reaching the local model. Please try again.",
                "source": "error",
                "model_used": "none",
                "skill": skill,
                "latency_ms": total_ms,
                "confidence_score": None,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "session_id": req.session_id,
            },
        )
    except ValueError as exc:
        _log.error("bad_request", extra={"req_id": req_id, "error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("internal_error", extra={"req_id": req_id})
        total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
        _stats.record({
            "ts":       datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": req.session_id,
            "source":   "error",
            "skill":    skill,
            "latency_ms": total_ms,
            "status":   "error",
        })
        return JSONResponse(
            status_code=200,
            content={
                "answer": "Sorry, I ran into an issue processing that request. Please try again.",
                "source": "error",
                "model_used": "none",
                "skill": skill,
                "latency_ms": total_ms,
                "confidence_score": None,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "session_id": req.session_id,
            },
        )


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------

async def _fake_stream_text(answer: str, chunk_size: int = 4):
    """Chunk a pre-generated answer into SSE token events for UX continuity."""
    for i in range(0, max(len(answer), 1), chunk_size):
        yield f"data: {json.dumps({'token': answer[i:i+chunk_size], 'done': False})}\n\n"


async def _heartbeat_until_done(task: asyncio.Task, interval: float = 15.0):
    """Yield empty-token SSE events every `interval`s while `task` is still
    running, so a proxy/client idle-read timeout doesn't kill the connection
    during the blocking local-inference pass. Shaped identically to a real
    token event (empty string), so no client-side special-casing is needed.
    """
    while not task.done():
        await asyncio.wait({task}, timeout=interval)
        if not task.done():
            yield f"data: {json.dumps({'token': '', 'done': False})}\n\n"


def _stream_metadata(result: "_ResolvedQuery", skill: str, t_wall: float) -> dict:
    metadata = {
        "routed_to":         result.source,
        "source":            result.source,
        "model":             result.model,
        "model_used":        result.model,
        "skill":             skill,
        "confidence_score":  result.confidence,
        "latency_ms":        round((time.monotonic() - t_wall) * 1000),
        "tokens": {
            "input":  result.input_tokens,
            "output": result.output_tokens,
            "total":  result.input_tokens + result.output_tokens,
        },
        "prompt_tokens":     result.input_tokens,
        "completion_tokens": result.output_tokens,
    }
    if result.retry:
        metadata["retry"] = True
    if result.escalated:
        metadata["escalated"] = True
    return metadata


@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    """Stream tokens as Server-Sent Events, escalating to cloud on low local confidence."""
    req_id = uuid.uuid4().hex[:8]
    t_wall = time.monotonic()

    _log.info(
        "stream_query_received",
        extra={
            "req_id":        req_id,
            "session_id":    req.session_id,
            "query_preview": req.query[:120],
        },
    )

    # ── Cache check — return immediately on hit, skip all inference ──────────
    cached = _cache_get(req.query)
    if cached is not None:
        async def cached_stream():
            async for chunk in _fake_stream_text(cached["answer"]):
                yield chunk
            metadata = {
                "routed_to":         cached["source"],
                "source":            cached["source"],
                "model":             cached["model_used"],
                "model_used":        cached["model_used"],
                "skill":             cached["skill"],
                "confidence_score":  cached["confidence_score"],
                "latency_ms":        round((time.monotonic() - t_wall) * 1000),
                "tokens":            cached["tokens"],
                "prompt_tokens":     cached["tokens"]["input"],
                "completion_tokens": cached["tokens"]["output"],
                "cache_hit":         True,
            }
            yield f"data: {json.dumps({'token': '', 'done': True, 'metadata': metadata})}\n\n"
            _session_update(req.session_id, req.query, cached["answer"])
        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    skill = _router.skill_router.classify(req.query)
    context_messages = _session_context(req.session_id, req.messages)

    # Realtime intent — route to cloud and fake-stream the response for UX continuity
    _rt_s = classify_intent(req.query)
    if _rt_s.is_realtime:
        preferred = _rt_s.preferred_provider
        available = [p for p in ["grok", "openai", "claude"] if not _router._is_degraded(p)]
        selected  = preferred if preferred in available else (available[0] if available else None)
        if selected:
            _signals = _rt_s.signals
            async def realtime_cloud_stream():
                try:
                    cloud = await cloud_llms.query(selected, req.query, req.system, messages=context_messages)
                    async for chunk in _fake_stream_text(cloud.answer):
                        yield chunk
                    yield f"data: {json.dumps({'token': '', 'done': True, 'metadata': {'routed_to': cloud.provider, 'source': cloud.provider, 'model_used': cloud.model, 'skill': skill, 'confidence_score': None, 'latency_ms': cloud.latency_ms, 'tokens': {'input': cloud.input_tokens, 'output': cloud.output_tokens, 'total': cloud.input_tokens + cloud.output_tokens}, 'realtime_intent': True, 'realtime_signals': _signals}})}\n\n"
                    _session_update(req.session_id, req.query, cloud.answer)
                except Exception as exc:
                    if isinstance(exc, cloud_llms.ProviderError) and _router._is_billing_error(exc):
                        _router._mark_degraded(selected)
                    _log.warning("stream_realtime_failed provider=%s", selected, extra={"provider": selected, "error": str(exc)})
                    async for chunk in local_llm.generate_stream(req.query, req.system, context_messages, skill=skill):
                        yield chunk
            return StreamingResponse(
                realtime_cloud_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # Local-only skills never escalate regardless of confidence — stream Ollama live.
    if skill in _router.LOCAL_ONLY_SKILLS:
        async def local_only_stream():
            answer_parts: list[str] = []
            try:
                async for chunk in local_llm.generate_stream(
                    req.query, req.system, context_messages, skill=skill
                ):
                    if chunk.startswith("data: "):
                        try:
                            payload = json.loads(chunk[6:])
                            if payload.get("token"):
                                answer_parts.append(payload["token"])
                        except json.JSONDecodeError:
                            pass
                    yield chunk
                _session_update(req.session_id, req.query, "".join(answer_parts))
            except Exception as exc:
                _log.error("stream_error req_id=%s error=%s", req_id, exc)
                yield f"data: {json.dumps({'error': True, 'message': str(exc)[:200], 'provider': 'ollama', 'done': True})}\n\n"
        return StreamingResponse(
            local_only_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Accurate escalation path — shared decision logic with /query ─────────
    # Run the local model to completion first so the real confidence score (not a
    # guess) decides whether to escalate. This trades true incremental local
    # streaming for a routing decision that matches /query exactly. A heartbeat
    # keeps the connection alive while _resolve_query() runs.
    async def routed_stream():
        task = asyncio.ensure_future(_resolve_query(req.query, req.system, context_messages, skill, req_id=req_id))
        try:
            async for heartbeat in _heartbeat_until_done(task):
                yield heartbeat
            result = task.result()
        except Exception as exc:
            _log.error("stream_resolve_failed req_id=%s error=%s", req_id, exc)
            yield f"data: {json.dumps({'error': True, 'message': str(exc)[:200], 'provider': 'ollama', 'done': True})}\n\n"
            return

        async for chunk in _fake_stream_text(result.answer):
            yield chunk
        yield f"data: {json.dumps({'token': '', 'done': True, 'metadata': _stream_metadata(result, skill, t_wall)})}\n\n"
        _session_update(req.session_id, req.query, result.answer)

        if result.source == "local":
            _cache_put(req.query, {
                "answer":           result.answer,
                "source":           "local",
                "model_used":       result.model,
                "skill":            skill,
                "confidence_score": result.confidence,
                "tokens": {
                    "input":  result.input_tokens,
                    "output": result.output_tokens,
                    "total":  result.input_tokens + result.output_tokens,
                },
            }, skill, result.confidence)

    return StreamingResponse(
        routed_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    ollama_ok = False
    ollama_version: str | None = None
    ollama_model: str = "Unavailable"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/version")
            r.raise_for_status()
            ollama_ok = True
            ollama_version = r.json().get("version")
            tags = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            tags.raise_for_status()
            models = tags.json().get("models", [])
            ollama_model = models[0]["name"] if models else "No model loaded"
    except Exception as exc:
        _log.warning("health_check_ollama_fail", extra={"error": str(exc)})

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": {
            "reachable": ollama_ok,
            "url":       config.OLLAMA_BASE_URL,
            "model":     ollama_model,
            "version":   ollama_version,
        },
        "cloud_apis_configured": {
            "claude": bool(config.ANTHROPIC_API_KEY),
            "gemini": bool(config.GOOGLE_API_KEY),
            "grok":   bool(config.XAI_API_KEY),
            "openai": bool(config.OPENAI_API_KEY),
        },
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@app.get("/stats")
async def stats() -> dict:
    return _stats.snapshot()


# ---------------------------------------------------------------------------
# GET /billing
# ---------------------------------------------------------------------------

@app.get("/billing")
async def billing_endpoint() -> dict:
    """Check each cloud provider with a 1-token probe. Results cached 5 minutes."""
    providers = ["claude", "gemini", "openai", "grok"]
    results = await asyncio.gather(*[_get_billing_status(p) for p in providers])
    return {
        p: {
            "status":       r["status"],
            "checked_at":   r["checked_at"],
            **(({"error_detail": r["error_detail"]}) if r.get("error_detail") else {}),
        }
        for p, r in zip(providers, results)
    }


# ---------------------------------------------------------------------------
# GET /providers/status  — lightweight model-list checks, no tokens burned
# ---------------------------------------------------------------------------

_PROVIDERS_TTL: float = 300.0  # 5-minute cache
_providers_cache: dict[str, Any] = {}


async def _probe_ollama(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        lat = round((time.perf_counter() - t0) * 1000, 1)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "online",
            "model": config.OLLAMA_MODEL,
            "credits": "Local / Unlimited",
            "detail": ", ".join(models) if models else config.OLLAMA_MODEL,
            "models_available": models,
            "latency_ms": lat,
        }
    except Exception as exc:
        return {
            "status": "offline",
            "model": config.OLLAMA_MODEL,
            "credits": "N/A",
            "detail": str(exc)[:80],
            "models_available": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


async def _probe_anthropic(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    if not config.ANTHROPIC_API_KEY:
        return {"status": "unconfigured", "model": config.CLAUDE_MODEL, "credits": "No API key", "detail": "", "latency_ms": 0}
    try:
        r = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            timeout=8.0,
        )
        lat = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            return {"status": "online", "model": config.CLAUDE_MODEL, "credits": "API Key: Active", "detail": "", "latency_ms": lat}
        elif r.status_code in (401, 403):
            return {"status": "offline", "model": config.CLAUDE_MODEL, "credits": "API Key: Invalid", "detail": f"HTTP {r.status_code}", "latency_ms": lat}
        elif r.status_code == 429:
            return {"status": "degraded", "model": config.CLAUDE_MODEL, "credits": "Rate Limited", "detail": "HTTP 429", "latency_ms": lat}
        else:
            return {"status": "degraded", "model": config.CLAUDE_MODEL, "credits": f"HTTP {r.status_code}", "detail": r.text[:80], "latency_ms": lat}
    except Exception as exc:
        return {"status": "offline", "model": config.CLAUDE_MODEL, "credits": "Connection Error", "detail": str(exc)[:80], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def _probe_gemini(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    if not config.GOOGLE_API_KEY:
        return {"status": "unconfigured", "model": config.GEMINI_MODEL, "credits": "No API key", "detail": "", "latency_ms": 0}
    try:
        r = await client.get(
            f"https://generativelanguage.googleapis.com/v1/models?key={config.GOOGLE_API_KEY}",
            timeout=8.0,
        )
        lat = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            return {"status": "online", "model": config.GEMINI_MODEL, "credits": "API Key: Active", "detail": "", "latency_ms": lat}
        else:
            try:
                body = r.json()
                msg = body.get("error", {}).get("message", "Unknown error")
            except Exception:
                msg = r.text[:100]
            if r.status_code in (400, 401, 403):
                credits = "API Key: Invalid"
                status = "offline"
            elif r.status_code == 429:
                credits = "Rate Limited"
                status = "degraded"
            else:
                credits = f"HTTP {r.status_code}"
                status = "degraded"
            return {"status": status, "model": config.GEMINI_MODEL, "credits": credits, "detail": f"{r.status_code}: {msg}", "error_code": r.status_code, "error_detail": msg, "latency_ms": lat}
    except Exception as exc:
        return {"status": "offline", "model": config.GEMINI_MODEL, "credits": "Connection Error", "detail": str(exc)[:80], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def _probe_grok(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    if not config.XAI_API_KEY:
        return {"status": "unconfigured", "model": config.GROK_MODEL, "credits": "No API key", "detail": "", "latency_ms": 0}
    try:
        r = await client.get(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {config.XAI_API_KEY}"},
            timeout=8.0,
        )
        lat = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            return {"status": "online", "model": config.GROK_MODEL, "credits": "API Key: Active", "detail": "", "latency_ms": lat}
        elif r.status_code in (401, 403):
            return {"status": "offline", "model": config.GROK_MODEL, "credits": "API Key: Invalid", "detail": f"HTTP {r.status_code}", "latency_ms": lat}
        elif r.status_code == 429:
            return {"status": "degraded", "model": config.GROK_MODEL, "credits": "Rate Limited", "detail": "HTTP 429", "latency_ms": lat}
        else:
            return {"status": "degraded", "model": config.GROK_MODEL, "credits": f"HTTP {r.status_code}", "detail": r.text[:80], "latency_ms": lat}
    except Exception as exc:
        return {"status": "offline", "model": config.GROK_MODEL, "credits": "Connection Error", "detail": str(exc)[:80], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def _probe_openai(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    if not config.OPENAI_API_KEY:
        return {"status": "unconfigured", "model": config.OPENAI_MODEL, "credits": "No API key", "detail": "", "latency_ms": 0}
    try:
        r = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            timeout=8.0,
        )
        lat = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            credits = "API Key: Active"
            try:
                rb = await client.get(
                    "https://api.openai.com/v1/dashboard/billing/credit_grants",
                    headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                    timeout=5.0,
                )
                if rb.status_code == 200:
                    bd = rb.json()
                    granted = float(bd.get("total_granted") or 0)
                    used    = float(bd.get("total_used")    or 0)
                    credits = f"${granted - used:.2f} remaining"
            except Exception:
                pass
            return {"status": "online", "model": config.OPENAI_MODEL, "credits": credits, "detail": "", "latency_ms": lat}
        elif r.status_code in (401, 403):
            return {"status": "offline", "model": config.OPENAI_MODEL, "credits": "API Key: Invalid", "detail": f"HTTP {r.status_code}", "latency_ms": lat}
        elif r.status_code == 429:
            return {"status": "degraded", "model": config.OPENAI_MODEL, "credits": "Rate Limited", "detail": "HTTP 429", "latency_ms": lat}
        else:
            return {"status": "degraded", "model": config.OPENAI_MODEL, "credits": f"HTTP {r.status_code}", "detail": r.text[:80], "latency_ms": lat}
    except Exception as exc:
        return {"status": "offline", "model": config.OPENAI_MODEL, "credits": "Connection Error", "detail": str(exc)[:80], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/providers/status")
async def providers_status_endpoint() -> dict:
    """Check all provider connectivity with model-list endpoints. Cached 5 minutes."""
    now = time.time()
    cached = _providers_cache.get("data")
    if cached and now < cached.get("_expires", 0):
        return {k: v for k, v in cached.items() if k != "_expires"}

    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    async with httpx.AsyncClient() as client:
        ollama_r, claude_r, gemini_r, grok_r, openai_r = await asyncio.gather(
            _probe_ollama(client),
            _probe_anthropic(client),
            _probe_gemini(client),
            _probe_grok(client),
            _probe_openai(client),
        )

    # Overlay router degraded state so the UI reflects actual routing health
    for name, res in [("claude", claude_r), ("gemini", gemini_r), ("grok", grok_r), ("openai", openai_r)]:
        if res["status"] == "online" and _router._is_degraded(name):
            expiry = _router._degraded.get(name, 0)
            res["status"] = "degraded"
            res["degraded_until"] = datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(timespec="seconds")

    response: dict[str, Any] = {
        "providers": {
            "ollama": {"name": "Ollama", "emoji": "🟠", "checked_at": checked_at, **ollama_r},
            "claude": {"name": "Claude", "emoji": "🟣", "checked_at": checked_at, **claude_r},
            "gemini": {"name": "Gemini", "emoji": "🔵", "checked_at": checked_at, **gemini_r},
            "grok":   {"name": "Grok",   "emoji": "🟡", "checked_at": checked_at, **grok_r},
            "openai": {"name": "OpenAI", "emoji": "🟢", "checked_at": checked_at, **openai_r},
        },
        "checked_at": checked_at,
    }
    _providers_cache["data"] = {**response, "_expires": now + _PROVIDERS_TTL}
    _log.info(
        "providers_status_checked",
        extra={"statuses": {k: v["status"] for k, v in response["providers"].items()}},
    )
    return response


# ---------------------------------------------------------------------------
# GET /metrics/history
# ---------------------------------------------------------------------------

@app.get("/metrics/history")
async def metrics_history(days: int = 5) -> dict:
    """Return system_metrics rows for the past N days, ordered ASC."""
    def _query() -> list:
        if not os.path.exists(_METRICS_DB):
            return []
        con = sqlite3.connect(_METRICS_DB)
        rows = con.execute(
            """SELECT timestamp, cpu_pct, gpu_pct, ram_used_mb, ram_total_mb
               FROM system_metrics
               WHERE timestamp > datetime('now', ? || ' days')
               ORDER BY timestamp ASC""",
            (f"-{days}",),
        ).fetchall()
        con.close()
        return rows

    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(None, _query)
    return {
        "metrics": [
            {
                "timestamp":    r[0],
                "cpu_pct":      r[1],
                "gpu_pct":      r[2],
                "ram_used_mb":  r[3],
                "ram_total_mb": r[4],
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /storage
# ---------------------------------------------------------------------------

@app.get("/storage")
async def storage_endpoint() -> dict:
    import psutil

    def _disk(path: str) -> dict | None:
        try:
            u = psutil.disk_usage(path)
            return {
                "total_gb": round(u.total / 1e9, 1),
                "used_gb":  round(u.used  / 1e9, 1),
                "free_gb":  round(u.free  / 1e9, 1),
            }
        except Exception:
            return None

    sd = _disk("/host_root") or {"total_gb": 468.0, "used_gb": 42.0, "free_gb": 407.0}
    nv = _disk("/host_mnt") or {"total_gb": 1800.0, "used_gb": 28.0, "free_gb": 1772.0}

    return {
        "microsd": {"label": "SanDisk Extreme 512GB microSD", **sd},
        "nvme":    {"label": "Samsung 990 Pro 2TB NVMe",      **nv},
    }


# ---------------------------------------------------------------------------
# GET /memory
# ---------------------------------------------------------------------------

@app.get("/memory")
async def memory_endpoint() -> dict:
    import psutil

    try:
        vm = psutil.virtual_memory()
        return {
            "total_gb":     round(vm.total     / 1e9, 1),
            "used_gb":      round(vm.used       / 1e9, 1),
            "available_gb": round(vm.available  / 1e9, 1),
            "cached_gb":    round(getattr(vm, "cached", 0) / 1e9, 1),
        }
    except Exception:
        return {"total_gb": 7.4, "used_gb": 1.7, "available_gb": 3.7, "cached_gb": 2.0}


# ---------------------------------------------------------------------------
# GET /jetson/containers
# ---------------------------------------------------------------------------

_CONTAINERS_KEY = os.getenv("EDGE_CONTAINERS_KEY", "")


@app.get("/jetson/containers")
async def jetson_containers(request: Request) -> JSONResponse:
    """List Docker containers. Requires X-Edge-Key header when EDGE_CONTAINERS_KEY is set."""
    if _CONTAINERS_KEY:
        provided = request.headers.get("x-edge-key", "")
        if provided != _CONTAINERS_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds="/var/run/docker.sock"),
            timeout=10.0,
        ) as client:
            resp = await client.get("http://docker/v1.41/containers/json", params={"all": "true"})
            resp.raise_for_status()
            raw: list[dict] = resp.json()
        result = []
        for c in raw:
            name   = (c.get("Names") or ["/unknown"])[0].lstrip("/")
            image  = c.get("Image", "")
            status = c.get("Status", "")
            running = status.lower().startswith("up")
            port_parts: list[str] = []
            for p in (c.get("Ports") or []):
                pub  = p.get("PublicPort")
                priv = p.get("PrivatePort")
                if pub and priv:
                    port_parts.append(f"{pub}→{priv}" if pub != priv else str(priv))
                elif priv:
                    port_parts.append(str(priv))
            result.append({
                "name":    name,
                "image":   image,
                "status":  status,
                "running": running,
                "ports":   ", ".join(dict.fromkeys(port_parts)) or "—",
            })
        result.sort(key=lambda x: (not x["running"], x["name"].lower()))
        return JSONResponse(content=result)
    except Exception as exc:
        _log.error("jetson_containers: %s", exc)
        return JSONResponse(status_code=503, content={"error": f"Docker unavailable: {exc}"})


# ---------------------------------------------------------------------------
# Argus reverse proxy — routes /argus/* → http://127.0.0.1:8400/*
# The dedicated argus-api.pazlabs.io tunnel is not running (missing credentials).
# Proxying through the existing api.pazlabs.io tunnel is the workaround.
# ---------------------------------------------------------------------------

_HOP_BY_HOP = frozenset({"host", "content-length", "transfer-encoding", "connection"})


@app.api_route("/argus/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def argus_proxy(path: str, request: Request) -> Response:
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.request(
                method=request.method,
                url=f"http://host.docker.internal:8400/{path}",
                params=dict(request.query_params),
                headers=fwd_headers,
                content=await request.body(),
            )
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"error": "Argus backend unreachable"})
    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={"error": "Argus backend timeout"})
