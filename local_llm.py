import logging
import math
import re

import httpx

import config

_log = logging.getLogger("edge_router.local_llm")


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
# Ollama call  (/api/generate supports logprobs; /api/chat does not)
# ---------------------------------------------------------------------------

async def _generate(prompt: str, system: str) -> tuple[str, list[float] | None, int, int]:
    payload: dict = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": config.OLLAMA_NUM_CTX},
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

    return text, logprobs, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def query(prompt: str, system: str = "", messages: list = []) -> dict:
    """
    Query the local Ollama model and return a confidence-scored result.

    Returns:
        answer          – model response with the self-rating tag stripped
        confidence      – float [0.0, 1.0] combined from all available signals
        should_escalate – True when confidence < config.CONFIDENCE_THRESHOLD
        model           – Ollama model tag used
        prompt_tokens   – tokens consumed by the prompt
        completion_tokens – tokens in the response
        signals         – individual signal values for debugging / logging
    """
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
            system=system,
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
            "signals": {"logprobs": None, "hedging": 0.0, "self_rating": None},
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
            "signals": {"logprobs": None, "hedging": 0.0, "self_rating": None},
        }

    answer, self_rating = _parse_self_rating(raw_text)

    logprob_score = _score_from_logprobs(logprobs) if logprobs else None
    hedging_score = _score_from_hedging(answer)
    confidence = _combine(logprob_score, hedging_score, self_rating)

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "should_escalate": confidence < config.CONFIDENCE_THRESHOLD,
        "model": config.OLLAMA_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "signals": {
            "logprobs": round(logprob_score, 4) if logprob_score is not None else None,
            "hedging": round(hedging_score, 4),
            "self_rating": round(self_rating, 4) if self_rating is not None else None,
        },
    }
