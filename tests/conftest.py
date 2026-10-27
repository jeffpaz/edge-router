"""Test fixtures: every external call (Jetson Ollama, mini Ollama, cloud LLMs) is faked.

Run:  .venv/bin/python -m pytest -q
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

# Settings are read at import time, so set them before importing the app.
os.environ.update({
    "OLLAMA_BASE_URL": "http://jetson.test:11434",
    "OLLAMA_MODEL": "llama3.2:3b",
    "MINI_OLLAMA_URL": "http://mini.test:11434",
    "MINI_MODEL": "qwen3:4b-instruct",
    "PRIVATE_TOKEN": "test-private-token",
    "MINI_RAG_URL": "http://rag.test:8100",
    "MINI_RAG_TOKEN": "test-rag-token",
    "AUTH_ENABLED": "false",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cloud_llms  # noqa: E402
import config  # noqa: E402
import local_llm  # noqa: E402
import main  # noqa: E402
import router as _router  # noqa: E402
import tiers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@dataclass
class FakeCloud:
    """Stands in for cloud_llms.query. mode: "ok" | "fail" (5xx) | "slow"."""
    mode: str = "ok"
    calls: list = field(default_factory=list)

    async def __call__(self, provider, prompt, system="", model=None, messages=[], search=False):
        self.calls.append({"provider": provider, "search": search})
        if self.mode == "fail":
            raise cloud_llms.ProviderError(provider, "api_status_error", "upstream 503", 503)
        if self.mode == "slow":
            await asyncio.sleep(5)
        return cloud_llms.CloudResponse(
            answer=f"{provider} answer", provider=provider, model=model or f"{provider}-model",
            latency_ms=12.0, input_tokens=10, output_tokens=20,
        )


@dataclass
class FakeLocal:
    """Stands in for local_llm.query (the Jetson)."""
    answer: str = "jetson answer"
    confidence: float = 0.9
    should_escalate: bool = False
    calls: int = 0

    async def __call__(self, prompt, system="", messages=[], skill="general"):
        self.calls += 1
        return {
            "answer": self.answer, "confidence": self.confidence,
            "should_escalate": self.should_escalate, "model": config.OLLAMA_MODEL,
            "prompt_tokens": 5, "completion_tokens": 7, "signals": {},
        }


@dataclass
class FakeOllamaHosts:
    """httpx transport for tiers.py: answers /api/tags and /api/chat per host."""
    mini_up: bool = True
    mini_chat_ok: bool = True
    jetson_up: bool = True
    mini_answer: str = "mini answer"
    rag_up: bool = True
    rag_calls: list = field(default_factory=list)
    rag_reply: dict = field(default_factory=lambda: {
        "answer": "Box 1 wages were 85,432.10 [1].",
        "citations": [{"n": 1, "source": "docs", "path": "Taxes/2023/W2 Acme.pdf", "file_name": "W2 Acme.pdf",
                       "page": 1, "doc_type": "W2", "year": 2023, "score": 0.99}],
        "weak_evidence": False, "model": "qwen3:4b-instruct", "filters": {"doc_type": "W2", "year": 2023},
    })
    tags_calls: dict = field(default_factory=lambda: {"mini.test": 0, "jetson.test": 0})
    chat_bodies: list = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "rag.test":
            if not self.rag_up:
                raise httpx.ConnectError("down", request=request)
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "chunks": 42, "sources": ["docs"]})
            self.rag_calls.append({"auth": request.headers.get("authorization"), "body": json.loads(request.content)})
            return httpx.Response(200, json=self.rag_reply)
        up = self.mini_up if host == "mini.test" else self.jetson_up
        if request.url.path == "/api/tags":
            self.tags_calls[host] += 1
            if not up:
                raise httpx.ConnectError("down", request=request)
            model = config.MINI_MODEL if host == "mini.test" else config.OLLAMA_MODEL
            return httpx.Response(200, json={"models": [{"name": model}]})
        if request.url.path == "/api/chat" and host == "mini.test":
            self.chat_bodies.append(json.loads(request.content))
            if not (up and self.mini_chat_ok):
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": self.mini_answer},
                "prompt_eval_count": 11, "eval_count": 22,
            })
        return httpx.Response(404)


@dataclass
class Fakes:
    cloud: FakeCloud
    local: FakeLocal
    ollama: FakeOllamaHosts


@pytest.fixture
def fx(monkeypatch) -> Fakes:
    fakes = Fakes(FakeCloud(), FakeLocal(), FakeOllamaHosts())
    monkeypatch.setattr(cloud_llms, "query", fakes.cloud)
    monkeypatch.setattr(local_llm, "query", fakes.local)
    transport = httpx.MockTransport(fakes.ollama.handler)
    monkeypatch.setattr(tiers, "_client", lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout))
    for key in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(config, key, "test-key")
    main._QUERY_CACHE.clear()
    tiers._health.clear()
    _router._degraded.clear()
    return fakes


@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager, so the lifespan (Ollama warm-up, metrics
    # loop) never runs.
    return TestClient(main.app)


ROUTE_FIELDS = ("tier", "model", "private", "fallback_used", "route_reason")


def assert_route_fields(body: dict) -> None:
    missing = [k for k in ROUTE_FIELDS if k not in body]
    assert not missing, f"response missing route fields {missing}: {body}"
