"""Phase 5 personal-docs lane: questions about the user's documents are forced private,
go only to the mini's document index, carry citations, and never reach the cloud."""
import json

import pytest

import config
import personal_docs
from conftest import assert_route_fields
from test_private import PRIVATE, SECRET, done_metadata, logs  # noqa: F401  (logs is a fixture)

DOCS_Q = "What were my Box 1 wages on my 2023 W-2?"


def ask(client, query, path="/query", headers=PRIVATE, **extra):
    return client.post(path, json={"query": query, **extra}, headers=headers)


def assert_nothing_else_called(fx):
    assert fx.cloud.calls == [], fx.cloud.calls
    assert fx.local.calls == 0
    assert fx.ollama.chat_bodies == []  # the mini's general LLM lane


# ── intent ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    DOCS_Q,
    "how much was withheld on my w2 last year",
    "What's the interest rate on our mortgage?",
    "When does my homeowners insurance policy renew?",
    "Find the closing disclosure in my documents",
    "search my files for the Desert Sun lease",
    "what did my last pay stub say for PTO?",
    "Show me my 1099-DIV from Vanguard",
])
def test_personal_docs_questions_are_detected(q):
    assert personal_docs.is_personal_docs(q), q


@pytest.mark.parametrize("q", [
    "What is a W-2?",
    "How do I fill out a 1099?",
    "How do I file my taxes?",
    "Explain how escrow works for a mortgage",
    "What's the latest news about the election today?",
    "Write a poem about my dog",
    "What is a haiku?",
])
def test_other_questions_are_not(q):
    assert not personal_docs.is_personal_docs(q), q


# ── routing ──────────────────────────────────────────────────────────────────

def test_docs_question_goes_to_the_index_with_citations(client, fx):
    body = ask(client, DOCS_Q).json()              # note: private not set by the client
    assert_route_fields(body)
    assert (body["tier"], body["source"], body["private"]) == ("private", "mini", True)
    assert body["route_reason"] == "personal_docs" and body["skill"] == "personal_docs"
    assert body["citations"][0]["file_name"] == "W2 Acme.pdf"
    assert fx.ollama.rag_calls[0]["auth"] == "Bearer test-rag-token"
    assert fx.ollama.rag_calls[0]["body"] == {"query": DOCS_Q}
    assert_nothing_else_called(fx)


def test_docs_question_needs_a_private_identity_even_without_the_checkbox(client, fx):
    r = ask(client, DOCS_Q, headers={})
    assert r.status_code == 401 and r.json()["private"] is True
    assert fx.ollama.rag_calls == []
    assert_nothing_else_called(fx)


def test_docs_service_down_is_a_clear_error_never_cloud_or_jetson(client, fx):
    fx.ollama.rag_up = False
    body = ask(client, DOCS_Q).json()
    assert body["source"] == "error" and body["private"] is True
    assert "Nothing was sent to the cloud" in body["answer"]
    assert_nothing_else_called(fx)


def test_docs_lane_unset_is_a_clear_error(client, fx, monkeypatch):
    monkeypatch.setattr(config, "MINI_RAG_URL", "")
    body = ask(client, DOCS_Q).json()
    assert body["source"] == "error" and "disabled" in body["route_reason"]
    assert_nothing_else_called(fx)


def test_weak_evidence_becomes_a_limitation(client, fx):
    fx.ollama.rag_reply = {**fx.ollama.rag_reply, "weak_evidence": True}
    assert ask(client, DOCS_Q).json()["limitation"] == "weak_evidence"


def test_non_docs_questions_are_unchanged(client, fx):
    body = ask(client, "What is a haiku?", headers={}).json()
    assert (body["tier"], body["private"]) == ("local", False)
    assert fx.ollama.rag_calls == []


@pytest.mark.parametrize("path", ["/query/stream", "/ask/stream"])
def test_streams_carry_citations(client, fx, path):
    meta = done_metadata(ask(client, DOCS_Q, path=path))
    assert (meta["tier"], meta["private"], meta["route_reason"]) == ("private", True, "personal_docs")
    assert meta["citations"][0]["path"] == "Taxes/2023/W2 Acme.pdf"
    assert_nothing_else_called(fx)


@pytest.mark.parametrize("path", ["/query", "/query/stream", "/ask/stream"])
def test_docs_question_text_is_never_logged(client, fx, logs, path):  # noqa: F811
    q = f"What does my {SECRET} insurance policy cover?"
    fx.ollama.rag_reply = {**fx.ollama.rag_reply, "answer": f"It covers {SECRET} [1]."}
    assert personal_docs.is_personal_docs(q)
    assert ask(client, q, path=path).status_code == 200
    leaked = [rec for rec in logs if SECRET in rec]
    assert not leaked, leaked[0][:300]


def test_tiers_status_reports_the_docs_lane(client, fx):
    docs = client.get("/tiers/status").json()["docs"]
    assert (docs["enabled"], docs["up"], docs["chunks"]) == (True, True, 42)
