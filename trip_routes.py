"""POST /trip/query — tripBuddy's trip_planning intent. Additive to the edge router.

Mounted from main.py with `app.include_router(trip_routes.router)`. Existing routes,
classifiers, thresholds and provider state are not read-modified: this module has its
own sub-classifier, fallback chain, daily caps and degraded tracking.

Sub-intents and destinations:
  plan_edit            → local (Ollama on the Mac mini, JSON mode)
  place, directions    → gemini → grok → local
  event, factcheck     → grok → local

Callers authenticate with the X-TripBuddy-Key header (TRIPBUDDY_ROUTER_KEY). This path
is outside the Firebase-token middleware (which covers /query* only).
"""
import hmac
import logging
import os
import re
import time
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

import providers_trip as pt

_log = logging.getLogger("edge_router.trip")

router = APIRouter(prefix="/trip", tags=["tripBuddy"])

SubIntent = Literal["plan_edit", "place", "directions", "event", "factcheck", "expense"]

# ---------------------------------------------------------------------------
# Trip sub-classifier — only ever runs on /trip/query
# ---------------------------------------------------------------------------

# Imperative plan edits ("move the Alamo…", "add a dive bar…") win outright.
_PLAN_EDIT_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:add|move|remove|drop|delete|swap|switch|replace|put|push|pull|"
    r"shift|reorder|cancel|undo|revert|change|make|skip|keep|insert|schedule|"
    r"reschedule|rename|update|generate|regenerate|rebuild|summari[sz]e|write)\b",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[str, list[str]]] = [
    # Order is the tie-break priority.
    ("event", [
        r"\bconcerts?\b", r"\bshows?\b", r"\bgigs?\b", r"\btour\b", r"\bfestivals?\b",
        r"\bevents?\b", r"\btickets?\b", r"\bplaying\b", r"\bperforming\b", r"\bopening act\b",
        r"\bwhat night\b", r"\bwhat day\b", r"\bdoors\b", r"\bset ?times?\b", r"\blineup\b",
        r"\bvenue\b", r"\brodeo\b", r"\bgame\b",
    ]),
    ("factcheck", [
        r"\bconfirm\b", r"\bverify\b", r"\bis it true\b", r"\bfact[- ]?check\b",
        r"\bstill open\b", r"\bclosed (?:down|permanently)\b", r"\bdid .+ (?:close|move)\b",
        r"\bis .+ still\b",
    ]),
    ("directions", [
        r"\bhow (?:do|can|should|would) (?:we|i|you) get\b", r"\bfrom .+ to\b", r"\bdirections?\b",
        r"\bhow far\b", r"\bdistance\b", r"\bwalk(?:ing|able)?\b", r"\buber\b", r"\blyft\b",
        r"\bcab\b", r"\bdrive\b", r"\bparking\b", r"\bget there\b",
    ]),
    ("place", [
        r"\brestaurants?\b", r"\bbars?\b", r"\bdive\b", r"\bhonky[- ]?tonks?\b", r"\bbbq\b",
        r"\bbarbecue\b", r"\btacos?\b", r"\bbreakfast\b", r"\bbrunch\b", r"\blunch\b",
        r"\bdinner\b", r"\bcoffee\b", r"\bcaf[eé]s?\b", r"\bbrewer(?:y|ies)\b", r"\bdrinks?\b",
        r"\bfood\b", r"\beat\b", r"\bnear(?:by)?\b", r"\baddress\b", r"\bhours\b",
        r"\bopen (?:late|now|until)\b", r"\bneighbou?rhoods?\b", r"\bwhere (?:is|can|to|should)\b",
        r"\bbest\b", r"\bmuseums?\b", r"\bparks?\b", r"\bshops?\b", r"\bstores?\b",
        r"\bmarkets?\b", r"\bhat(?:ter)?s?\b",
    ]),
]

_DESTINATIONS: dict[str, list[str]] = {
    "plan_edit":  ["local"],
    "place":      ["gemini", "grok", "local"],
    "directions": ["gemini", "grok", "local"],
    "event":      ["grok", "local"],
    "factcheck":  ["grok", "local"],
    "expense":    ["local"],   # money stays on the home lab: never Gemini or Grok
}

# Expense privacy guard (tripBuddy Phase 6.4): a member's message that pairs a money amount with an
# expense verb ("Victor paid $48 for Uber", "I paid 212.40 for the Airbnb, split 4 ways") is routed
# to the local model only, whatever sub-intent or local_only flag the caller sent. Checked on the
# member's message (classify_text), or on the query when there is none.
_MONEY = re.compile(r"(\$\s?\d|\b\d+(?:[.,]\d{2})\b|\b\d+\s?(?:dollars|bucks|usd)\b)", re.I)
_EXPENSE_VERB = re.compile(r"\b(paid|pay|pays|paying|split|splitting|owe[sd]?|spent|spend|covered|cover|venmo(?:ed)?|"
                           r"reimburse[sd]?|chipped in|fronted|bought|charged|cost)\b", re.I)


def is_expense_text(text: str | None) -> bool:
    return bool(text) and bool(_MONEY.search(text)) and bool(_EXPENSE_VERB.search(text))


def classify_trip(query: str) -> tuple[str, dict[str, int]]:
    """Return (sub_intent, scores). Zero signals → plan_edit."""
    if _PLAN_EDIT_LEAD.search(query):
        return "plan_edit", {}
    text = query.lower()
    scores = {name: sum(1 for p in pats if re.search(p, text)) for name, pats in _PATTERNS}
    best, score = max(scores.items(), key=lambda kv: kv[1])  # first wins ties (priority order)
    return (best if score > 0 else "plan_edit"), scores


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TripQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The prompt (trip context + request).")
    system: str = Field("", description="Optional system prompt.")
    sub_intent: SubIntent | None = Field(None, description="Skip the classifier and force a sub-intent.")
    classify_text: str | None = Field(
        None, description="Text to classify when `query` carries a long context (e.g. the member's message)."
    )
    json_output: bool = Field(True, description="Ask the destination for a JSON object.")
    json_schema: dict | None = Field(None, description="Optional JSON schema for structured output.")
    local_only: bool = Field(False, description="Never call a cloud provider.")


class TripTokens(BaseModel):
    input: int
    output: int


class TripQueryResponse(BaseModel):
    answer: str
    routed_to: str                 # "local" | "gemini" | "grok"
    model_used: str
    sub_intent: str
    classifier_scores: dict[str, int]
    attempted: list[str]           # destinations tried, in order
    fallback_reason: str | None
    cap_notice: str | None
    grounded: bool
    latency_ms: float
    tokens: TripTokens
    usage_today: dict[str, int]
    caps: dict[str, int]


def _check_key(key: str | None) -> None:
    expected = os.getenv("TRIPBUDDY_ROUTER_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="tripBuddy routes are disabled (no TRIPBUDDY_ROUTER_KEY).")
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-TripBuddy-Key.")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/query", response_model=TripQueryResponse)
async def trip_query(
    req: TripQueryRequest,
    x_tripbuddy_key: str | None = Header(None, alias="X-TripBuddy-Key"),
) -> TripQueryResponse:
    _check_key(x_tripbuddy_key)
    t_wall = time.perf_counter()

    if req.sub_intent:
        sub_intent, scores = req.sub_intent, {}
    else:
        sub_intent, scores = classify_trip(req.classify_text or req.query)

    expense_guard = sub_intent == "expense" or is_expense_text(req.classify_text or req.query)
    chain = ["local"] if (req.local_only or expense_guard) else list(_DESTINATIONS[sub_intent])
    if expense_guard and not req.local_only:
        _log.info("trip_expense_guard", extra={"sub_intent": sub_intent})
    attempted: list[str] = []
    reasons: list[str] = []
    capped: list[str] = []

    for dest in chain:
        if dest != "local":
            if not pt.is_configured(dest):
                reasons.append(f"{dest}: not configured")
                continue
            if pt.is_degraded(dest):
                reasons.append(f"{dest}: degraded (recent quota/auth error)")
                continue
            if not pt.try_reserve(dest):
                capped.append(dest)
                reasons.append(f"{dest}: daily cap {pt.cap_for(dest)} reached")
                continue
        attempted.append(dest)
        try:
            res = await pt.PROVIDERS[dest](req.query, req.system, req.json_output, req.json_schema)
        except pt.TripProviderError as exc:
            if exc.quota:
                pt.mark_degraded(dest)
            reasons.append(f"{dest}: {'quota/auth ' if exc.quota else ''}error {exc.status_code or ''}".strip())
            _log.warning("trip_provider_failed", extra={"provider": dest, "error": str(exc)[:300]})
            continue

        cloud_capped = [p for p in ("gemini", "grok") if p in capped]
        cap_notice = None
        if cloud_capped and res.provider == "local":
            cap_notice = (
                "Daily cloud limit reached ("
                + ", ".join(f"{p} {pt.cap_for(p)}/day" for p in cloud_capped)
                + "). Answered by the local model instead."
            )
        total_ms = round((time.perf_counter() - t_wall) * 1000, 1)
        _log.info(
            "trip_routed",
            extra={
                "sub_intent": sub_intent, "routed_to": res.provider, "model": res.model,
                "attempted": attempted, "fallback_reason": "; ".join(reasons) or None,
                "latency_ms": total_ms,
            },
        )
        return TripQueryResponse(
            answer=res.answer,
            routed_to=res.provider,
            model_used=res.model,
            sub_intent=sub_intent,
            classifier_scores=scores,
            attempted=attempted,
            fallback_reason="; ".join(reasons) or None,
            cap_notice=cap_notice,
            grounded=res.grounded,
            latency_ms=total_ms,
            tokens=TripTokens(input=res.input_tokens, output=res.output_tokens),
            usage_today=pt.usage_today(),
            caps={"gemini": pt.GEMINI_DAILY_CAP, "grok": pt.GROK_DAILY_CAP},
        )

    _log.error("trip_all_destinations_failed", extra={"sub_intent": sub_intent, "reasons": reasons})
    raise HTTPException(
        status_code=503,
        detail={"error": "all destinations failed", "sub_intent": sub_intent, "reasons": reasons},
    )


@router.get("/usage")
async def trip_usage(x_tripbuddy_key: str | None = Header(None, alias="X-TripBuddy-Key")) -> dict:
    _check_key(x_tripbuddy_key)
    return {
        "usage_today": pt.usage_today(),
        "caps": {"gemini": pt.GEMINI_DAILY_CAP, "grok": pt.GROK_DAILY_CAP},
        "degraded": {p: pt.is_degraded(p) for p in ("gemini", "grok")},
        "local": {"url": pt.OLLAMA_URL, "model": pt.OLLAMA_MODEL},
        "gemini": {"models": pt.GEMINI_MODELS, "grounding": pt.GEMINI_GROUNDING},
        "grok": {"model": pt.GROK_MODEL},
    }
