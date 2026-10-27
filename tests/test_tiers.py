"""Phase 2 tiered routing: LOCAL (Jetson) / PRIVATE (mini) / CLOUD lanes and fallback."""
import json

import pytest

import auth
import config
import main
import tiers
from conftest import assert_route_fields

NEWS = "What's the latest news about the election today?"
CODING = "Write a python function to reverse a list"
TRIVIAL = "What is a haiku?"
JOB_HEADERS = {"Authorization": "Bearer test-private-token"}


def ask(client, query, **extra):
    r = client.post("/query", json={"query": query, **extra})
    assert r.status_code == 200
    body = r.json()
    assert_route_fields(body)
    return body


# ── Lane selection ───────────────────────────────────────────────────────────

def test_trivial_query_goes_local(client, fx):
    body = ask(client, TRIVIAL)
    assert body["skill"] == "definition"
    assert (body["tier"], body["source"]) == ("local", "local")
    assert body["route_reason"] == "local_only_skill:definition"
    assert body["fallback_used"] is False and body["private"] is False
    assert fx.cloud.calls == []
    assert fx.ollama.chat_bodies == []


def test_news_goes_cloud_grok_with_live_search(client, fx):
    body = ask(client, NEWS)
    assert body["realtime_intent"] is True
    assert (body["tier"], body["source"]) == ("cloud", "grok")
    assert fx.cloud.calls == [{"provider": "grok", "search": True}]
    assert body["route_reason"].startswith("realtime_intent")
    assert body["fallback_used"] is False
    assert fx.local.calls == 0 and fx.ollama.chat_bodies == []


def test_low_confidence_escalates_to_cloud(client, fx):
    fx.local.should_escalate, fx.local.confidence = True, 0.3
    body = ask(client, CODING)
    assert body["skill"] == "coding"
    assert (body["tier"], body["source"]) == ("cloud", "claude")
    assert body["route_reason"].startswith("escalated:coding")


# ── Fallback: cloud → mini → Jetson ──────────────────────────────────────────

def test_cloud_down_news_falls_back_to_mini(client, fx):
    fx.cloud.mode = "fail"
    body = ask(client, NEWS)
    assert (body["tier"], body["source"], body["model"]) == ("private", "mini", "qwen3:4b-instruct")
    assert body["fallback_used"] is True
    assert body["limitation"] == "live_data_unavailable"
    assert [c["provider"] for c in fx.cloud.calls] == ["grok", "openai", "claude"]
    assert "all cloud providers failed" in body["route_reason"]
    assert body["route_reason"].endswith("-> mini")


def test_cloud_down_escalation_falls_back_to_mini(client, fx):
    fx.cloud.mode = "fail"
    fx.local.should_escalate, fx.local.confidence = True, 0.3
    body = ask(client, CODING)
    assert (body["tier"], body["source"]) == ("private", "mini")
    assert body["fallback_used"] is True
    assert "limitation" not in body or body["limitation"] is None


def test_cloud_timeout_falls_back_to_mini(client, fx, monkeypatch):
    monkeypatch.setattr(config, "CLOUD_LANE_TIMEOUT_S", 0.05)
    fx.cloud.mode = "slow"
    body = ask(client, NEWS)
    assert (body["tier"], body["source"]) == ("private", "mini")
    assert "timed out" in body["route_reason"]


def test_cloud_and_mini_down_falls_back_to_jetson(client, fx):
    fx.cloud.mode = "fail"
    fx.ollama.mini_up = False
    body = ask(client, NEWS)
    assert (body["tier"], body["source"], body["answer"]) == ("local", "local", "jetson answer")
    assert body["fallback_used"] is True
    assert body["limitation"] == "live_data_unavailable"
    assert body["route_reason"].endswith("-> jetson")


def test_escalation_fallback_reuses_jetson_answer(client, fx):
    fx.cloud.mode = "fail"
    fx.ollama.mini_up = False
    fx.local.should_escalate, fx.local.confidence = True, 0.3
    body = ask(client, CODING)
    assert (body["tier"], body["answer"]) == ("local", "jetson answer")
    assert fx.local.calls == 1  # the escalation's local pass, not a second run


def test_everything_down_is_a_clear_error(client, fx):
    fx.cloud.mode = "fail"
    fx.ollama.mini_up = False
    fx.local.answer = ""
    body = ask(client, NEWS)
    assert body["source"] == "error" and body["tier"] is None
    assert "couldn't reach" in body["answer"]


def test_force_cloud_falls_back_too(client, fx):
    fx.cloud.mode = "fail"
    body = ask(client, CODING, force_cloud=True)
    assert (body["tier"], body["source"]) == ("private", "mini")
    assert body["route_reason"].startswith("force_cloud")


def test_fallback_answer_is_never_cached():
    result = main._ResolvedQuery(
        source="local", answer="a", model="m", confidence=0.99,
        input_tokens=1, output_tokens=1, fallback_used=True,
    )
    main._QUERY_CACHE.clear()
    main._cache_local_result("q", result, "definition")
    assert not main._QUERY_CACHE
    result.fallback_used = False
    main._cache_local_result("q", result, "definition")
    assert main._QUERY_CACHE


# ── Mini unset → PRIVATE lane disabled, fails closed ─────────────────────────

def test_mini_unset_disables_private_lane(client, fx, monkeypatch):
    monkeypatch.setattr(config, "MINI_OLLAMA_URL", "")
    status = client.get("/tiers/status").json()
    assert status["private"]["enabled"] is False and status["private"]["up"] is False

    fx.cloud.mode = "fail"
    body = ask(client, NEWS)
    assert (body["tier"], body["source"]) == ("local", "local")  # skips the mini entirely
    assert "private lane disabled" in body["route_reason"]
    assert fx.ollama.tags_calls["mini.test"] == 0 and fx.ollama.chat_bodies == []

    r = client.post("/jobs/generate", json={"prompt": "summarize"}, headers=JOB_HEADERS)
    assert r.status_code == 503
    assert r.json()["error"] == "private_lane_unavailable"


# ── Batch jobs: PRIVATE lane only ────────────────────────────────────────────

def test_job_runs_on_mini_with_fixed_runner_options(client, fx):
    schema = {"type": "object", "properties": {"year": {"type": "integer"}}}
    r = client.post("/jobs/generate", json={"prompt": "classify", "format": schema}, headers=JOB_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert_route_fields(body)
    assert (body["tier"], body["private"], body["answer"]) == ("private", True, "mini answer")
    assert fx.cloud.calls == [] and fx.local.calls == 0

    sent = fx.ollama.chat_bodies[-1]
    assert sent["model"] == "qwen3:4b-instruct"
    assert sent["keep_alive"] == -1
    assert sent["options"]["num_ctx"] == config.MINI_NUM_CTX == 8192
    assert sent["format"] == schema


def test_job_fails_closed_when_mini_errors(client, fx):
    fx.ollama.mini_chat_ok = False
    r = client.post("/jobs/generate", json={"prompt": "x"}, headers=JOB_HEADERS)
    assert r.status_code == 503
    assert fx.cloud.calls == [] and fx.local.calls == 0


@pytest.mark.parametrize("headers, status", [
    ({}, 401),
    ({"Authorization": "Bearer wrong"}, 401),
])
def test_job_requires_private_identity(client, fx, headers, status):
    r = client.post("/jobs/generate", json={"prompt": "x"}, headers=headers)
    assert r.status_code == status
    assert fx.ollama.chat_bodies == []


def test_job_firebase_email_must_be_allow_listed(client, fx, monkeypatch):
    async def fake_verify(token):
        return {"email": "someone@example.com", "email_verified": True}
    monkeypatch.setattr(auth, "verify_token", fake_verify)
    jwt_like = {"Authorization": "Bearer a.b.c"}

    monkeypatch.setattr(config, "PRIVATE_ALLOWED_EMAILS", frozenset())
    assert client.post("/jobs/generate", json={"prompt": "x"}, headers=jwt_like).status_code == 403

    monkeypatch.setattr(config, "PRIVATE_ALLOWED_EMAILS", frozenset({"someone@example.com"}))
    assert client.post("/jobs/generate", json={"prompt": "x"}, headers=jwt_like).status_code == 200


def test_job_unverified_email_is_forbidden(client, fx, monkeypatch):
    async def fake_verify(token):
        return {"email": "jeffpaz@gmail.com", "email_verified": False}
    monkeypatch.setattr(auth, "verify_token", fake_verify)
    monkeypatch.setattr(config, "PRIVATE_ALLOWED_EMAILS", frozenset({"jeffpaz@gmail.com"}))
    r = client.post("/jobs/generate", json={"prompt": "x"}, headers={"Authorization": "Bearer a.b.c"})
    assert r.status_code == 403


# ── Health ───────────────────────────────────────────────────────────────────

def test_tiers_status_reports_each_lane(client, fx):
    status = client.get("/tiers/status").json()
    assert status["local"]["up"] is True and status["local"]["model"] == config.OLLAMA_MODEL
    assert status["private"]["up"] is True and status["private"]["model"] == "qwen3:4b-instruct"
    assert status["cloud"]["up"] is True
    assert set(status["cloud"]["providers"]) == {"claude", "gemini", "grok", "openai"}


def test_lane_health_is_cached_15s(client, fx):
    for _ in range(3):
        client.get("/tiers/status")
    assert fx.ollama.tags_calls == {"mini.test": 1, "jetson.test": 1}
    tiers._health[tiers.PRIVATE] = (0.0, tiers._health[tiers.PRIVATE][1])  # expire it
    client.get("/tiers/status")
    assert fx.ollama.tags_calls["mini.test"] == 2


def test_failed_mini_call_forces_a_fresh_probe(client, fx):
    fx.ollama.mini_chat_ok = False
    client.post("/jobs/generate", json={"prompt": "x"}, headers=JOB_HEADERS)
    assert tiers.PRIVATE not in tiers._health


# ── Streaming metadata ───────────────────────────────────────────────────────

def _done_metadata(response) -> dict:
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            if payload.get("done"):
                return payload["metadata"]
    raise AssertionError("no done frame")


def test_stream_metadata_carries_route_fields(client, fx):
    meta = _done_metadata(client.post("/query/stream", json={"query": NEWS}))
    assert_route_fields(meta)
    assert (meta["tier"], meta["source"]) == ("cloud", "grok")

    fx.cloud.mode = "fail"
    main._QUERY_CACHE.clear()
    meta = _done_metadata(client.post("/query/stream", json={"query": NEWS}))
    assert (meta["tier"], meta["fallback_used"], meta["limitation"]) == ("private", True, "live_data_unavailable")
