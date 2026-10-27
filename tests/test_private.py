"""Phase 3 private mode: private requests never reach a cloud provider, on any endpoint,
are never cached or stored, and their text is never logged. Cloud clients are mocked
and every test asserts zero cloud calls."""
import asyncio
import json
import logging

import pytest

import cloud_llms
import config
import local_llm
import main
import privacy
import providers_trip
from conftest import assert_route_fields

# Captured at import, before the fx fixture swaps cloud_llms.query for a fake.
REAL_CLOUD_QUERY = cloud_llms.query

NEWS = "What's the latest news about the election today?"
GENERAL = "Write a python function to reverse a list"
SECRET = "zebra-ledger-7731"
PRIVATE = {"Authorization": "Bearer test-private-token"}


@pytest.fixture
def logs():
    """Every record the app logs, as text including all extra fields."""
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage() + " " + json.dumps(record.__dict__, default=str))

    handler = Capture()
    logger = logging.getLogger("edge_router")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def ask_private(client, query, path="/query", headers=PRIVATE, **extra):
    return client.post(path, json={"query": query, "private": True, **extra}, headers=headers)


def done_metadata(response) -> dict:
    """Final metadata from /query/stream (data frames) or /ask/stream (named events)."""
    event = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            payload = json.loads(line[6:])
            if event == "done" or payload.get("done"):
                return payload["metadata"]
    raise AssertionError(f"no done frame in {response.text[:300]}")


def assert_no_cloud(fx):
    assert fx.cloud.calls == [], f"cloud was called: {fx.cloud.calls}"


# ── /query ───────────────────────────────────────────────────────────────────

def test_private_general_question_uses_mini_with_zero_cloud_calls(client, fx):
    fx.local.should_escalate, fx.local.confidence = True, 0.1  # would escalate if not private
    body = ask_private(client, GENERAL).json()
    assert_route_fields(body)
    assert (body["tier"], body["source"], body["private"]) == ("private", "mini", True)
    assert body["fallback_used"] is False
    assert_no_cloud(fx)
    assert fx.local.calls == 0


def test_private_news_question_answers_locally_never_grok(client, fx):
    body = ask_private(client, NEWS).json()
    assert (body["tier"], body["private"]) == ("private", True)
    assert body["limitation"] == "live_data_unavailable"
    assert_no_cloud(fx)


def test_private_mini_down_falls_back_to_jetson_never_cloud(client, fx):
    fx.ollama.mini_up = False
    body = ask_private(client, NEWS).json()
    assert (body["tier"], body["source"], body["answer"]) == ("local", "local", "jetson answer")
    assert body["private"] is True and body["fallback_used"] is True
    assert body["limitation"] == "live_data_unavailable"
    assert_no_cloud(fx)


def test_private_everything_down_is_a_clear_error_never_cloud(client, fx):
    fx.ollama.mini_up = False
    fx.local.answer = ""
    body = ask_private(client, GENERAL).json()
    assert body["source"] == "error" and body["tier"] is None and body["private"] is True
    assert "never uses the cloud" in body["answer"]
    assert_no_cloud(fx)


def test_private_ignores_force_cloud(client, fx):
    body = ask_private(client, GENERAL, force_cloud=True).json()
    assert body["tier"] == "private"
    assert_no_cloud(fx)


def test_private_mini_unset_uses_jetson_never_cloud(client, fx, monkeypatch):
    monkeypatch.setattr(config, "MINI_OLLAMA_URL", "")
    body = ask_private(client, GENERAL).json()
    assert (body["tier"], body["private"]) == ("local", True)
    assert "private lane disabled" in body["route_reason"]
    assert_no_cloud(fx)


# ── Streams ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [GENERAL, NEWS])
def test_private_query_stream_zero_cloud_calls(client, fx, query):
    meta = done_metadata(ask_private(client, query, path="/query/stream"))
    assert_route_fields(meta)
    assert (meta["tier"], meta["private"]) == ("private", True)
    assert_no_cloud(fx)


@pytest.mark.parametrize("mini_up", [True, False])
def test_private_ask_stream_zero_cloud_calls(client, fx, monkeypatch, mini_up):
    async def fake_streaming(prompt, system, messages, skill, run, label):
        return {"answer": "jetson streamed", "confidence": 0.8, "should_escalate": False,
                "model": config.OLLAMA_MODEL, "prompt_tokens": 3, "completion_tokens": 4, "signals": {}}
    monkeypatch.setattr(local_llm, "query_streaming", fake_streaming)
    fx.ollama.mini_up = mini_up

    meta = done_metadata(ask_private(client, NEWS, path="/ask/stream"))
    assert meta["private"] is True
    assert meta["tier"] == ("private" if mini_up else "local")
    assert meta["limitation"] == "live_data_unavailable"
    assert_no_cloud(fx)


# ── Access ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/query", "/query/stream", "/ask/stream"])
@pytest.mark.parametrize("headers, status", [({}, 401), ({"Authorization": "Bearer wrong"}, 401)])
def test_private_requires_private_identity(client, fx, path, headers, status):
    r = ask_private(client, GENERAL, path=path, headers=headers)
    assert r.status_code == status
    assert r.json()["private"] is True
    assert_no_cloud(fx)
    assert fx.local.calls == 0 and fx.ollama.chat_bodies == []


def test_private_needs_an_allow_listed_email(client, fx, monkeypatch):
    async def fake_verify(token):
        return {"email": "stranger@example.com", "email_verified": True}
    monkeypatch.setattr(main.auth, "verify_token", fake_verify)
    monkeypatch.setattr(config, "PRIVATE_ALLOWED_EMAILS", frozenset({"jeffpaz@gmail.com"}))
    r = ask_private(client, GENERAL, headers={"Authorization": "Bearer a.b.c"})
    assert r.status_code == 403
    assert_no_cloud(fx)


# ── Nothing kept, nothing logged ─────────────────────────────────────────────

def test_private_answers_are_not_cached_or_stored_in_session(client, fx):
    fx.ollama.mini_up = False                  # Jetson answer, confident, local-only skill
    query = "What is a haiku?"
    ask_private(client, query, session_id="s-private")
    assert main._QUERY_CACHE == {}
    assert "s-private" not in main._sessions


@pytest.mark.parametrize("path", ["/query", "/query/stream", "/ask/stream"])
def test_private_text_is_never_logged(client, fx, logs, path):
    fx.ollama.mini_answer = f"the answer mentions {SECRET}"
    r = ask_private(client, f"Tell me about {SECRET} please", path=path)
    assert r.status_code == 200
    assert logs, "expected some metadata log records"
    leaked = [rec for rec in logs if SECRET in rec]
    assert not leaked, leaked[0][:300]


# ── Second guard: every cloud call path refuses inside a private request ─────

CLOUD_CALLS = [
    ("cloud_llms.query", lambda: REAL_CLOUD_QUERY("openai", "x")),
    ("query_claude", lambda: cloud_llms.query_claude("x")),
    ("query_gemini", lambda: cloud_llms.query_gemini("x")),
    ("query_grok", lambda: cloud_llms.query_grok("x")),
    ("query_grok search", lambda: cloud_llms.query_grok("x", search=True)),
    ("query_openai", lambda: cloud_llms.query_openai("x")),
    ("trip gemini", lambda: providers_trip.query_gemini("x", "", True, None)),
    ("trip grok", lambda: providers_trip.query_grok("x", "", True, None)),
]


@pytest.mark.parametrize("name, call", CLOUD_CALLS, ids=[n for n, _ in CLOUD_CALLS])
def test_every_cloud_call_path_raises_in_private_scope(name, call):
    async def attempt():
        with privacy.private_scope():
            await call()
    with pytest.raises(privacy.PrivateLeakError):
        asyncio.run(attempt())


def test_guard_is_off_for_normal_requests():
    privacy.assert_cloud_allowed("grok")  # no raise outside a private scope


def test_guard_catches_a_routing_bug(client, fx, monkeypatch):
    """If a bug sent a private request into the cloud lane, the guard still stops it."""
    provider_calls = []

    async def recording_provider(*args, **kwargs):
        provider_calls.append(args)
        raise AssertionError("cloud provider reached")

    monkeypatch.setattr(cloud_llms, "query", REAL_CLOUD_QUERY)
    monkeypatch.setattr(cloud_llms, "_PROVIDERS", {p: recording_provider for p in cloud_llms._PROVIDERS})

    async def buggy_private_lane(query, system, messages, skill, req_id="", **kwargs):
        return await main._cloud_lane(query, system, messages, skill, req_id, reason="bug")
    monkeypatch.setattr(main, "_private_lane", buggy_private_lane)

    # In production the global exception handler turns this into an error response;
    # the test client re-raises it instead.
    with pytest.raises(privacy.PrivateLeakError):
        ask_private(client, GENERAL)
    assert provider_calls == []
    assert_no_cloud(fx)
