"""tripBuddy's Gemini 429 handling: try the other model, then sideline Gemini only as long as the limit lasts."""
import asyncio

import pytest
from google.genai import errors

import providers_trip as pt


def _429(quota_id: str | None, delay: str | None = None) -> errors.ClientError:
    details = []
    if quota_id:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id, "quotaValue": "10"}]})
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "You exceeded your current quota", "details": details}})


def test_limit_scope_and_sideline_time():
    assert pt.gemini_limit(_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "34s")) == ("minute", 34.0)
    assert pt.gemini_limit(_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")) == ("minute", 60.0)
    scope, wait = pt.gemini_limit(_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert scope == "day" and 0 < wait <= 24 * 3600
    assert pt.gemini_limit(_429(None)) == ("unknown", pt._DEGRADED_TTL)


class _Models:
    def __init__(self, outcomes):
        self.outcomes, self.asked = outcomes, []

    async def generate_content(self, model, contents, config):
        self.asked.append(model)
        out = self.outcomes[model]
        if isinstance(out, Exception):
            raise out

        class R:
            text = '{"ok": true}'
            usage_metadata = None
        return R()


@pytest.fixture
def gemini(monkeypatch):
    def install(outcomes):
        models = _Models(outcomes)

        class Client:
            def __init__(self, **kw):
                self.aio = type("Aio", (), {"models": models})()
        monkeypatch.setattr("google.genai.Client", Client)
        monkeypatch.setattr(pt, "GEMINI_MODELS", list(outcomes))
        return models
    return install


def test_a_rate_limited_model_falls_back_to_the_next(gemini):
    models = gemini({"m1": _429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "20s"), "m2": "ok"})
    res = asyncio.run(pt.query_gemini("x", "", True, None))
    assert models.asked == ["m1", "m2"] and res.model == "m2"


def test_all_models_limited_sidelines_for_the_soonest(gemini):
    gemini({"m1": _429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "20s"),
            "m2": _429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")})
    with pytest.raises(pt.TripProviderError) as e:
        asyncio.run(pt.query_gemini("x", "", True, None))
    assert e.value.quota and e.value.status_code == 429 and e.value.retry_after == 20.0


def test_mark_degraded_uses_the_given_time(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pt.time, "time", lambda: now[0])
    pt.mark_degraded("gemini", 20.0)
    assert pt.is_degraded("gemini")
    now[0] += 21
    assert not pt.is_degraded("gemini")
