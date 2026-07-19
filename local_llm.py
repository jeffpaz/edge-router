import asyncio
import json
import logging
import math
import re
import time

import httpx

import config

_log = logging.getLogger("edge_router.local_llm")


# ---------------------------------------------------------------------------
# Default formatting guidance — Ollama returns valid single-newline-separated
# text (soft line breaks), which markdown renderers collapse to spaces
# without CSS help. Nudging the model toward blank-line paragraph/stanza
# breaks reduces reliance on that CSS handling. Only used when the caller
# doesn't supply their own system prompt, so it never overrides one.
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Format your responses clearly: leave a "
    "blank line between paragraphs, put each line of a poem on its own line "
    "with a blank line between stanzas, and put each list or recipe step on "
    "its own line."
)


# ---------------------------------------------------------------------------
# Signal 1: Token log-probabilities  (Ollama /api/generate, logprobs=true)
# ---------------------------------------------------------------------------

def _score_from_logprobs(token_logprobs: list[float]) -> float:
    """Mean per-token probability → [0, 1]. Higher = model was less surprised."""
    if not token_logprobs:
        return 0.0
    avg = sum(token_logprobs) / len(token_logprobs)
    return min(1.0, max(0.0, math.exp(avg)))


# ---------------------------------------------------------------------------
# Signal 2: Hedging-language heuristic
# ---------------------------------------------------------------------------

# (phrase, penalty subtracted from 0.85 base)
_HEDGES: list[tuple[str, float]] = [
    ("i don't know",         0.55),
    ("i do not know",        0.55),
    ("i'm not sure",         0.45),
    ("i am not sure",        0.45),
    ("i'm not certain",      0.40),
    ("i am not certain",     0.40),
    ("not certain",          0.30),
    ("i cannot",             0.40),
    ("i can't",              0.40),
    ("i'm unable",           0.40),
    ("i am unable",          0.40),
    ("as an ai",             0.25),
    ("as a language model",  0.25),
    ("maybe",                0.15),
    ("perhaps",              0.10),
    ("i think",              0.15),
    ("i believe",            0.15),
    ("i'm not 100%",         0.30),
    ("may be incorrect",     0.30),
    ("might be wrong",       0.30),
    ("please verify",        0.20),
    ("consult a professional", 0.20),
    ("i'm not an expert",    0.25),
]


def _score_from_hedging(text: str) -> float:
    """Start at 0.85, subtract the single largest hedge penalty, adjust for length/content."""
    low = text.lower()
    penalty = max(
        (pen for phrase, pen in _HEDGES if phrase in low),
        default=0.0,
    )
    base = 0.85 - penalty

    words = len(text.split())
    if words < 5:
        base -= 0.45
    elif words < 15:
        base -= 0.20
    elif words < 30:
        base -= 0.10
    elif words < 50:
        base -= 0.05

    # Code blocks indicate a structured, detailed answer — reward them.
    if "```" in text:
        base += 0.15

    return min(1.0, max(0.0, base))


# ---------------------------------------------------------------------------
# Signal 3: Self-rating (injected into the model's own response, 1 call)
# ---------------------------------------------------------------------------

# Appended to every prompt so the model reports its own confidence.
_SELF_RATE_SUFFIX = (
    "\n\nAfter your answer, on a final line write exactly: "
    "CONFIDENCE: <number 1-10>"
)

_CONFIDENCE_RE = re.compile(
    r"confidence\s*:\s*(10|[1-9])",
    re.IGNORECASE,
)


def _parse_self_rating(text: str) -> tuple[str, float | None]:
    """
    Extract the CONFIDENCE tag the model wrote. Returns (clean_answer, score).
    Score is normalised to [0.0, 1.0] (1→0.0, 10→1.0). Returns None if absent.
    """
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return text, None

    rating = int(match.group(1))
    score = (rating - 1) / 9.0

    # Strip the confidence line so it doesn't appear in the returned answer.
    clean = _CONFIDENCE_RE.sub("", text)
    clean = re.sub(r"\n{2,}", "\n", clean).strip()
    return clean, score


# ---------------------------------------------------------------------------
# Signal 4: Hallucination-risk patterns in the answer
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, confidence penalty).
# Applied additively — a statistic-heavy answer accumulates multiple penalties.
_HALLUCINATION_SIGNALS: list[tuple[re.Pattern, float]] = [
    # "according to [Organization/Source]" alongside a percentage — fabricated citation
    (re.compile(r"according to [A-Z][a-zA-Z].*?\d+\.?\d*\s*%", re.DOTALL), 0.35),
    # Specific 4-digit year alongside a percentage — suspiciously precise historical stat
    (re.compile(r"(?:\b(19|20)\d{2}\b.*?\d+\.?\d*\s*%|\d+\.?\d*\s*%.*?\b(19|20)\d{2}\b)", re.DOTALL), 0.25),
    # "approximately X%" — sounds hedged but is still a fabricated number
    (re.compile(r"\bapproximately\s+\d+\.?\d*\s*%"), 0.20),
    # Any bare "N%" in the answer — LLMs invent specific statistics
    (re.compile(r"\d+\.?\d*\s*%"), 0.15),
]


def _hallucination_penalty(text: str) -> float:
    """Return the total confidence penalty for hallucination-risk patterns in text."""
    total = 0.0
    for pattern, penalty in _HALLUCINATION_SIGNALS:
        if pattern.search(text):
            total += penalty
    return total


# ---------------------------------------------------------------------------
# Signal 5: Query complexity pre-score bias
# ---------------------------------------------------------------------------

def _complexity_bias(query: str) -> float:
    """
    Compute a confidence adjustment from surface-level query complexity.
    Positive = query looks simple/local-answerable; negative = looks risky.
    Capped at ±0.25 before returning.
    """
    text = query.strip()
    lower = text.lower()
    words = text.split()

    bias = 0.0

    # Simple directive forms the local model handles well
    if lower.startswith(("how do i ", "what should i ")):
        bias += 0.20

    # Short queries are generally simpler
    if len(words) < 10:
        bias += 0.15

    # Detect proper nouns: capitalized words beyond the first word
    # (exclude single-char words like "I" which are always uppercase)
    inner_words = [w for w in words[1:] if len(w) > 1 and w.isalpha()]
    has_proper_noun = any(w[0].isupper() for w in inner_words)
    if not has_proper_noun:
        bias += 0.10

    # No 4-digit year reference
    if not re.search(r"\b\d{4}\b", text):
        bias += 0.10

    # No percentage / statistic signals
    if not re.search(r"%|percent|percentage|statistic", lower):
        bias += 0.10

    # Specific factual claim: proper noun alongside a number
    if has_proper_noun and re.search(r"\d+", text):
        bias -= 0.20

    # Citation / research signals — LLM likely to fabricate sources
    if re.search(r"according to|study shows|research says|data shows", lower):
        bias -= 0.25

    return max(-0.25, min(0.25, bias))


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def _combine(
    logprob_score: float | None,
    hedging_score: float,
    self_rating: float | None,
) -> float:
    """
    Weighted average. Logprobs are the most objective signal; self-rating is
    second; hedging is cheapest and least reliable.
    """
    if logprob_score is not None and self_rating is not None:
        return 0.50 * logprob_score + 0.30 * self_rating + 0.20 * hedging_score
    if logprob_score is not None:
        return 0.65 * logprob_score + 0.35 * hedging_score
    if self_rating is not None:
        return 0.55 * self_rating + 0.45 * hedging_score
    return hedging_score


# ---------------------------------------------------------------------------
# Per-skill generation parameters
# ---------------------------------------------------------------------------

_SKILL_NUM_CTX: dict[str, int] = {
    "conversational": 2048,
    "how_to":         2048,
    "definition":     2048,
    "opinion_advice": 2048,
    "creative":       2048,
    "language_task":  2048,
}
# Default (coding, general, anything else) → 4096

_SKILL_NUM_PREDICT: dict[str, int] = {
    "how_to":         600,
    "conversational": 500,
    "definition":     400,
    "opinion_advice": 500,
    "language_task":  500,
    "creative":       800,
    "coding":         800,
}
# Default → 300


# ---------------------------------------------------------------------------
# Keep-alive  (prevents model unload between queries on Jetson)
# ---------------------------------------------------------------------------

async def _send_keepalive() -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={"model": config.OLLAMA_MODEL, "keep_alive": "10m"},
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ollama call  (/api/generate supports logprobs; /api/chat does not)
# ---------------------------------------------------------------------------

async def _generate(prompt: str, system: str, skill: str = "general") -> tuple[str, list[float] | None, int, int]:
    num_ctx     = _SKILL_NUM_CTX.get(skill, 4096)
    num_predict = _SKILL_NUM_PREDICT.get(skill, 300)

    payload: dict = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx":        num_ctx,
            "num_predict":    num_predict,
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_thread":     6,
        },
        "logprobs": True,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    text = data.get("response", "")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    # Ollama ≥ 0.2 may return logprobs in one of several shapes.
    logprobs: list[float] | None = None
    raw = data.get("logprobs")
    if isinstance(raw, dict):
        vals = raw.get("token_logprobs") or raw.get("logprobs") or []
        logprobs = [v for v in vals if isinstance(v, (int, float))] or None
    elif isinstance(raw, list) and raw:
        logprobs = [v for v in raw if isinstance(v, (int, float))] or None

    asyncio.create_task(_send_keepalive())

    return text, logprobs, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def query(prompt: str, system: str = "", messages: list = [], skill: str = "general") -> dict:
    """
    Query the local Ollama model and return a confidence-scored result.

    Returns:
        answer            – model response with the self-rating tag stripped
        confidence        – float [0.0, 1.0] combined from all available signals
        should_escalate   – True when confidence < per-skill threshold
        model             – Ollama model tag used
        prompt_tokens     – tokens consumed by the prompt
        completion_tokens – tokens in the response
        signals           – individual signal values for debugging / logging
    """
    # Compute complexity bias from the raw query before building the full prompt
    bias = _complexity_bias(prompt)

    if messages:
        history_text = "Previous conversation:\n"
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"
        full_prompt = f"{history_text}\nCurrent question: {prompt}"
    else:
        full_prompt = prompt

    try:
        raw_text, logprobs, prompt_tokens, completion_tokens = await _generate(
            prompt=full_prompt + _SELF_RATE_SUFFIX,
            system=system or _DEFAULT_FORMAT_SYSTEM_PROMPT,
            skill=skill,
        )
    except httpx.HTTPStatusError as exc:
        _log.error(
            "Ollama returned HTTP %d — escalating to cloud. Body: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return {
            "answer": "",
            "confidence": 0.0,
            "should_escalate": True,
            "model": config.OLLAMA_MODEL,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "signals": {"logprobs": None, "hedging": 0.0, "self_rating": None, "hallucination_penalty": 0.0},
        }
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        _log.error("Ollama request failed — escalating to cloud. Error: %s", exc)
        return {
            "answer": "",
            "confidence": 0.0,
            "should_escalate": True,
            "model": config.OLLAMA_MODEL,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "signals": {"logprobs": None, "hedging": 0.0, "self_rating": None, "hallucination_penalty": 0.0},
        }

    answer, self_rating = _parse_self_rating(raw_text)

    logprob_score = _score_from_logprobs(logprobs) if logprobs else None
    hedging_score = _score_from_hedging(answer)
    confidence = _combine(logprob_score, hedging_score, self_rating)

    # Subtract hallucination-risk penalties based on answer content
    hal_penalty = _hallucination_penalty(answer)
    if hal_penalty > 0:
        confidence = max(0.0, confidence - hal_penalty)
        _log.info(
            "hallucination_penalty_applied penalty=%.2f confidence_after=%.4f",
            hal_penalty,
            confidence,
            extra={"hal_penalty": hal_penalty, "confidence_after": confidence},
        )

    # Boost: conversational queries get credit for being well within local capability
    if skill == "conversational":
        base_boost = 0.30

        has_stats = bool(re.search(r"\d+\.?\d*\s*%|\d+\s+(?:percent|out of|in \d+)", answer, re.IGNORECASE))
        stat_boost = 0.0 if has_stats else 0.20

        _PRACTICAL_WORDS = ("wash", "cut", "cook", "add", "place", "serve", "remove", "brush", "preheat", "grill")
        practical_boost = 0.10 if any(w in answer.lower() for w in _PRACTICAL_WORDS) else 0.0

        total_boost = base_boost + stat_boost + practical_boost
        before = confidence
        confidence = min(1.0, confidence + total_boost)
        _log.info(
            "conversational_boost applied base=+0.30 stat_boost=+%.2f practical_boost=+%.2f confidence %.4f → %.4f",
            stat_boost,
            practical_boost,
            before,
            confidence,
            extra={
                "skill": skill,
                "boost_base": base_boost,
                "boost_stat": stat_boost,
                "boost_practical": practical_boost,
                "confidence_before": before,
                "confidence_after": confidence,
            },
        )

    # Hard cap: math/data questions must always escalate to a cloud LLM
    if skill == "math_data":
        before = confidence
        confidence = min(confidence, 0.50)
        if confidence != before:
            _log.info(
                "math_data_cap applied confidence %.4f → 0.50",
                before,
                extra={"skill": skill, "confidence_before": before, "confidence_after": confidence},
            )

    # Hard cap: sports/people questions need real-time data — always escalate
    if skill in ("sports_people", "current_events"):
        before = confidence
        confidence = min(confidence, 0.40)
        if confidence != before:
            _log.info(
                "%s_cap applied confidence %.4f → 0.40",
                skill,
                before,
                extra={"skill": skill, "confidence_before": before, "confidence_after": confidence},
            )

    # Apply query complexity bias (computed before Ollama call, from raw query text)
    before_bias = confidence
    confidence = min(1.0, max(0.0, confidence + bias))
    _log.info(
        "complexity_bias skill=%s bias=%.4f confidence %.4f → %.4f",
        skill,
        bias,
        before_bias,
        confidence,
        extra={
            "skill": skill,
            "complexity_bias": round(bias, 4),
            "confidence_before_bias": round(before_bias, 4),
            "confidence_after_bias": round(confidence, 4),
        },
    )

    threshold = config.SKILL_THRESHOLDS.get(skill, config.CONFIDENCE_THRESHOLD)

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "should_escalate": confidence < threshold,
        "model": config.OLLAMA_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "signals": {
            "logprobs": round(logprob_score, 4) if logprob_score is not None else None,
            "hedging": round(hedging_score, 4),
            "self_rating": round(self_rating, 4) if self_rating is not None else None,
            "hallucination_penalty": round(hal_penalty, 4),
            "complexity_bias": round(bias, 4),
        },
    }


# ---------------------------------------------------------------------------
# Streaming generation  (for /query/stream SSE endpoint)
# ---------------------------------------------------------------------------

async def generate_stream(
    prompt: str,
    system: str = "",
    messages: list = [],
    skill: str = "general",
):
    """
    Async generator that streams Ollama tokens as SSE-formatted strings.

    Yields:
        'data: {"token": "...", "done": false}\\n\\n'  for each token
        'data: {"token": "", "done": true, "metadata": {...}}\\n\\n'  on completion
    """
    num_ctx     = _SKILL_NUM_CTX.get(skill, 4096)
    num_predict = _SKILL_NUM_PREDICT.get(skill, 300)

    if messages:
        history_text = "Previous conversation:\n"
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"
        full_prompt = f"{history_text}\nCurrent question: {prompt}"
    else:
        full_prompt = prompt

    payload: dict = {
        "model": config.OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "num_ctx":        num_ctx,
            "num_predict":    num_predict,
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_thread":     6,
        },
    }
    payload["system"] = system or _DEFAULT_FORMAT_SYSTEM_PROMPT

    received_tokens = False
    t_start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    done = chunk.get("done", False)
                    if done:
                        if not received_tokens:
                            yield f"data: {json.dumps({'error': True, 'message': 'Ollama returned an empty response', 'provider': 'ollama', 'done': True})}\n\n"
                            return
                        in_tok  = chunk.get("prompt_eval_count", 0)
                        out_tok = chunk.get("eval_count", 0)
                        lat_ms  = round((time.monotonic() - t_start) * 1000)
                        metadata = {
                            "routed_to":        "local",
                            "source":           "local",
                            "model":            config.OLLAMA_MODEL,
                            "model_used":       config.OLLAMA_MODEL,
                            "skill":            skill,
                            "confidence_score": None,
                            "latency_ms":       lat_ms,
                            "tokens": {
                                "input":  in_tok,
                                "output": out_tok,
                                "total":  in_tok + out_tok,
                            },
                            "prompt_tokens":     in_tok,
                            "completion_tokens": out_tok,
                        }
                        yield f"data: {json.dumps({'token': '', 'done': True, 'metadata': metadata})}\n\n"
                        asyncio.create_task(_send_keepalive())
                    else:
                        token = chunk.get("response", "")
                        if token:
                            received_tokens = True
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'error': True, 'message': 'Ollama timed out', 'provider': 'ollama', 'done': True})}\n\n"
    except httpx.HTTPStatusError as exc:
        msg = f"Ollama returned HTTP {exc.response.status_code}"
        if exc.response.status_code == 404:
            msg = f"Ollama model not found ({config.OLLAMA_MODEL}) — check that the model is loaded"
        yield f"data: {json.dumps({'error': True, 'message': msg, 'provider': 'ollama', 'done': True})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': True, 'message': f'Ollama is unreachable: {str(exc)[:120]}', 'provider': 'ollama', 'done': True})}\n\n"
