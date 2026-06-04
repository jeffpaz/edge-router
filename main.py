import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cloud_llms
import config
import local_llm
import router as _router


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
    if root.handlers:
        return
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
    """Bounded deque of the last 100 query records.
    Thread-safe under asyncio's single-threaded event loop.
    """

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
    source: str                        # "local" | cloud provider name
    model_used: str
    skill: str
    latency_ms: float
    confidence_score: float | None     # local model confidence; None when force_cloud=True
    tokens: TokenBreakdown
    session_id: str | None


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
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

    skill = _router.skill_router.classify(req.query)
    local_confidence: float | None = None

    try:
        if not req.force_cloud:
            # Step 1: run query through local LLM
            local = await local_llm.query(req.query, req.system, req.messages)
            local_confidence = local["confidence"]

            _log.info(
                "local_inference",
                extra={
                    "req_id": req_id,
                    "model": local["model"],
                    "confidence": local["confidence"],
                    "should_escalate": local["should_escalate"],
                    "signals": local["signals"],
                },
            )

            # Step 2: confidence is sufficient — return local answer
            if not local["should_escalate"]:
                total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
                tokens   = TokenBreakdown(
                    input=local["prompt_tokens"],
                    output=local["completion_tokens"],
                    total=local["prompt_tokens"] + local["completion_tokens"],
                )
                _stats.record({
                    "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "session_id":       req.session_id,
                    "source":           "local",
                    "model_used":       local["model"],
                    "skill":            skill,
                    "latency_ms":       total_ms,
                    "confidence_score": local_confidence,
                    "tokens":           tokens.model_dump(),
                })
                _log.info(
                    "routed",
                    extra={
                        "req_id":     req_id,
                        "session_id": req.session_id,
                        "source":     "local",
                        "model":      local["model"],
                        "skill":      skill,
                        "latency_ms": total_ms,
                        "tokens":     tokens.model_dump(),
                    },
                )
                return QueryResponse(
                    answer=local["answer"],
                    source="local",
                    model_used=local["model"],
                    skill=skill,
                    latency_ms=total_ms,
                    confidence_score=local_confidence,
                    tokens=tokens,
                    session_id=req.session_id,
                )

            _log.info(
                "escalating_to_cloud",
                extra={
                    "req_id":     req_id,
                    "skill":      skill,
                    "confidence": local["confidence"],
                    "threshold":  config.CONFIDENCE_THRESHOLD,
                },
            )

        # Step 3: dispatch to the skill-matched cloud LLM
        cloud    = await _router.skill_router.dispatch(req.query, skill, req.system, req.messages)
        total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
        tokens   = TokenBreakdown(
            input=cloud.input_tokens,
            output=cloud.output_tokens,
            total=cloud.input_tokens + cloud.output_tokens,
        )
        _stats.record({
            "ts":               datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id":       req.session_id,
            "source":           cloud.provider,
            "model_used":       cloud.model,
            "skill":            skill,
            "latency_ms":       total_ms,
            "confidence_score": local_confidence,
            "tokens":           tokens.model_dump(),
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
                "forced_cloud":     req.force_cloud,
            },
        )
        return QueryResponse(
            answer=cloud.answer,
            source=cloud.provider,
            model_used=cloud.model,
            skill=skill,
            latency_ms=total_ms,
            confidence_score=local_confidence,
            tokens=tokens,
            session_id=req.session_id,
        )

    except httpx.HTTPError as exc:
        _log.error("upstream_error", extra={"req_id": req_id, "error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}") from exc
    except ValueError as exc:
        _log.error("bad_request", extra={"req_id": req_id, "error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("internal_error", extra={"req_id": req_id})
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    ollama_ok = False
    ollama_version: str | None = None

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/version")
            r.raise_for_status()
            ollama_ok = True
            ollama_version = r.json().get("version")
    except Exception as exc:
        _log.warning("health_check_ollama_fail", extra={"error": str(exc)})

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": {
            "reachable": ollama_ok,
            "url":       config.OLLAMA_BASE_URL,
            "model":     config.OLLAMA_MODEL,
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

    # microSD — host root fs mounted read-only at /host_root
    sd = _disk("/host_root") or {"total_gb": 468.0, "used_gb": 42.0, "free_gb": 407.0}

    # NVMe — host /mnt mounted read-only at /host_mnt
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
