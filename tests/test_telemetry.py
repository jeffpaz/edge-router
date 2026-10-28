"""/ask/stream telemetry when the Mac mini answers: the mini's own metrics stream,
live mini tokens and phases, the documents lane's phases, and the Jetson again on
fallback. Every host is faked (conftest.py)."""
import asyncio
import json

import pytest

import config
import local_llm
import telemetry
from test_private import GENERAL, PRIVATE

DOCS_Q = "What were my 2023 W-2 wages from Acme?"


def events(response) -> list[tuple[str, dict]]:
    out, event = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[6:])))
    return out


def ask(client, query):
    return events(client.post("/ask/stream", json={"query": query, "private": True}, headers=PRIVATE))


def test_kv_estimate_covers_the_mini_model():
    assert telemetry.kv_bytes_per_token("qwen3:4b-instruct") == 2 * 36 * 8 * 128 * 2


def test_private_answer_on_the_mini_streams_live_with_mini_metrics(client, fx):
    fx.ollama.mini_answer = "the mini says hello"
    evs = ask(client, GENERAL)
    kinds = [k for k, _ in evs]
    assert ("call", "mini") in [(k, d.get("label")) for k, d in evs]
    assert "".join(d["text"] for k, d in evs if k == "token") == "the mini says hello"
    phases = [d for k, d in evs if k == "phases"]
    assert [(p["label"], p["load"], p["prefill"], p["decode"]) for p in phases] == [("mini", 0.5, 0.2, 0.9)]
    assert fx.ollama.chat_bodies[-1]["stream"] is True
    metrics = [d for k, d in evs if k == "metrics"]
    assert metrics[-1]["host"] == "mini" and metrics[-1]["tok"] == 4
    assert metrics[-1]["kv"] == (11 + 4) * telemetry.kv_bytes_per_token("qwen3:4b-instruct")
    assert all(m["cpu"] in (None, 42.0) for m in metrics if m["host"] == "mini")
    assert fx.ollama.sys_streams == [f"Bearer {config.MINI_RAG_TOKEN}"]
    assert kinds[-1] == "done" and evs[-1][1]["metadata"]["tier"] == "private"


def test_documents_lane_reports_the_minis_phases(client, fx):
    evs = ask(client, DOCS_Q)
    phases = [d for k, d in evs if k == "phases"]
    assert [p["label"] for p in phases] == ["filters", "search", "answer"]
    assert phases[1]["search"] == 1.5 and phases[1]["decode"] == 0
    assert phases[2]["prefill"] == 0.8 and phases[2]["eval_count"] == 30
    assert phases[0]["t_start"] <= phases[1]["t_start"] <= phases[2]["t_start"]
    metrics = [d for k, d in evs if k == "metrics"]
    assert metrics[-1]["host"] == "mini" and metrics[-1]["tok"] == 12 + 30
    assert fx.ollama.sys_streams                       # the mini's metrics were read
    assert evs[-1][0] == "done" and evs[-1][1]["metadata"]["citations"]


def test_mini_down_falls_back_to_jetson_metrics(client, fx, monkeypatch):
    async def fake_streaming(prompt, system, messages, skill, run, label):
        run.begin_call(label, 10)                      # what local_llm.query_streaming does
        return {"answer": "jetson streamed", "confidence": 0.8, "should_escalate": False,
                "model": config.OLLAMA_MODEL, "prompt_tokens": 3, "completion_tokens": 4, "signals": {}}
    monkeypatch.setattr(local_llm, "query_streaming", fake_streaming)
    fx.ollama.mini_up = False
    evs = ask(client, GENERAL)
    metrics = [d for k, d in evs if k == "metrics"]
    assert metrics[-1]["host"] == "jetson"
    assert evs[-1][1]["metadata"]["tier"] == "local"


def test_jobs_and_query_stay_non_streaming(client, fx):
    client.post("/query", json={"query": GENERAL, "private": True}, headers=PRIVATE)
    assert fx.ollama.chat_bodies[-1]["stream"] is False
    assert fx.ollama.sys_streams == []


def test_mini_sampler_failure_leaves_the_answer_alone(fx):
    async def broken():
        raise RuntimeError("no stream")
        yield  # pragma: no cover

    async def go():
        async with telemetry.Run("llama3.2:3b", mini_sampler=broken) as run:
            run.use_host("mini")
            await asyncio.sleep(0.3)
            return run.metrics()

    m = asyncio.run(go())
    assert m["host"] == "mini" and m["cpu"] is None


def test_closing_the_run_stops_the_mini_stream():
    state = {"open": False, "closed": False}

    async def endless():
        state["open"] = True
        try:
            while True:
                await asyncio.sleep(0.01)
                yield {"cpu": 1.0, "gpu": 2, "ram": 3}
        finally:
            state["closed"] = True

    async def go():
        async with telemetry.Run("llama3.2:3b", mini_sampler=endless) as run:
            run.use_host("mini")
            await asyncio.sleep(0.1)
            assert run.metrics()["cpu"] == 1.0
        await asyncio.sleep(0.05)                       # let the cancellation land

    asyncio.run(go())
    assert state == {"open": True, "closed": True}


@pytest.mark.parametrize("bad", [None, "x", [1, "y", {"label": "answer", "load": "slow"}]])
def test_remote_calls_tolerate_junk(bad):
    async def go():
        run = telemetry.Run("llama3.2:3b")
        run.remote_calls(bad, offset=0.0, model="qwen3:4b-instruct")
        return [run.events.get_nowait() for _ in range(run.events.qsize())]
    out = asyncio.run(go())
    assert all(e.startswith("event: phases") for e in out)
