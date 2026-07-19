import logging
import re
import time
from datetime import datetime, timezone

import cloud_llms

# ---------------------------------------------------------------------------
# Logger  (NullHandler keeps this library-safe; callers configure output)
# ---------------------------------------------------------------------------

logging.getLogger("edge_router").addHandler(logging.NullHandler())
_log = logging.getLogger("edge_router.router")


# ---------------------------------------------------------------------------
# Degraded-provider tracking
# ---------------------------------------------------------------------------

_DEGRADED_TTL: float = 1800.0  # 30 minutes
_degraded: dict[str, float] = {}  # provider → wall-clock expiry (time.time())


def _is_billing_error(exc: cloud_llms.ProviderError) -> bool:
    """Return True for quota/billing/permission errors — warrants degrading the provider."""
    if exc.error_type == "rate_limit":
        return True
    if exc.error_type == "api_status_error":
        if exc.status_code in (429, 402, 403):
            return True
        if exc.status_code == 400:
            msg = str(exc).lower()
            return any(w in msg for w in ("credit", "billing", "quota", "payment", "balance"))
    return False


def _is_degraded(provider: str) -> bool:
    expiry = _degraded.get(provider)
    if expiry is None:
        return False
    if time.time() < expiry:
        return True
    del _degraded[provider]  # TTL expired — clean up silently
    return False


def _mark_degraded(provider: str) -> None:
    expiry = time.time() + _DEGRADED_TTL
    _degraded[provider] = expiry
    _log.warning(
        "provider_degraded provider=%s until=%s",
        provider,
        datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(timespec="seconds"),
        extra={
            "provider": provider,
            "degraded_until": datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(timespec="seconds"),
        },
    )


# ---------------------------------------------------------------------------
# SkillRouter
# ---------------------------------------------------------------------------

# Generic sport names (hockey, golf, tennis, ...) are ambiguous on their own —
# they show up just as often in creative/definition/how-to queries ("write a
# poem about golf") as in real sports questions. Require one of these
# sports-context signals to appear elsewhere in the query before the bare
# sport name counts toward the sports_people skill.
_SPORTS_CONTEXT = (
    r"(?:team|score|scored|won|game|match|tournament|player|coach|franchise|"
    r"league|roster|standings|championship|playoffs|season|drafted|traded|"
    r"signed|nhl|nba|nfl|mlb|nascar|pga|fifa|ufc)"
)


def _sport_pattern(word: str) -> str:
    """Match `word` only when a sports-context signal also appears in the query."""
    return (
        rf"\b{word}\b(?=.*\b{_SPORTS_CONTEXT}\b)"
        rf"|\b{_SPORTS_CONTEXT}\b(?=.*\b{word}\b)"
    )


class SkillRouter:
    """Classifies a query into a skill and dispatches to the right cloud LLM.

    Skill priority on ties (same match count): coding > math_data > current_events.
    """

    # Evaluated in order; first skill to outscore the rest wins ties naturally.
    _PATTERNS: list[tuple[str, list[str]]] = [
        ("coding", [
            r"\bcode\b",       r"\bdebug\b",      r"\bfunction\b",   r"\berror\b",
            r"\bpython\b",     r"\bjavascript\b",  r"\bsql\b",        r"\bbug\b",
            r"\bfix\b",        r"\bcompile\b",     r"\bsyntax\b",     r"\bscript\b",
            r"\bclass\b",      r"\bvariable\b",    r"\bloop\b",       r"\barray\b",
            r"\bapi\b",        r"\bendpoint\b",    r"\bgit\b",        r"\brefactor\b",
            r"\btraceback\b",  r"\bimport\b",      r"\btypeerror\b",  r"\brepository\b",
        ]),
        ("math_data", [
            r"\bcalculate\b",  r"\bstatistics?\b", r"\bdataset\b",    r"\bformula\b",
            r"\bgraph\b",      r"\baverage\b",     r"\bmean\b",       r"\bmedian\b",
            r"\bvariance\b",   r"\bregression\b",  r"\bprobability\b",r"\bequation\b",
            r"\bsolve\b",      r"\bintegral\b",    r"\bderivative\b", r"\bmatrix\b",
            r"\bchart\b",      r"\bplot\b",        r"\bcorrelation\b",r"\bstd\b",
            # Data / statistics signals — queries that need real numbers, not guesses
            r"\bpercent\b",    r"\bpercentage\b",  r"\d+\s*%",        r"\bhow many\b",
            r"\bhow much\b",   r"\bstatistic\b",   r"\bdata\b",       r"\bstudy\b",
            r"\bresearch\b",   r"\bsurvey\b",      r"\bdistribution\b",r"\bodds\b",
            r"\bratio\b",      r"\bproportion\b",  r"\bfraction\b",   r"\bhow often\b",
            r"\bhow frequently\b", r"\bwhat fraction\b", r"\bwhat share\b",
            r"\bpopulation\b", r"\bsample\b",      r"\bestimate\b",   r"\bapproximately how\b",
        ]),
        ("current_events", [
            r"\btoday\b",      r"\bnews\b",        r"\blatest\b",     r"\b202[4-9]\b",
            r"\bwho won\b",    r"\bbreaking\b",    r"\brecently\b",   r"\bcurrently\b",
            r"\bupdate\b",     r"\belection\b",    r"\bannounced\b",  r"\breleased\b",
            r"\bhappened\b",   r"\bright now\b",   r"\bthis week\b",  r"\bthis month\b",
            r"\bnet worth\b",  r"\bbiography\b",   r"\bfounded\b",    r"\bceo\b",
            r"how did \S+ make", r"who is [A-Z]",
        ]),
        ("sports_people", [
            r"\bnhl\b",        r"\bnba\b",         r"\bnfl\b",        r"\bmlb\b",
            r"\bnascar\b",     r"\bpga\b",         r"\bfifa\b",       r"\bufc\b",
            _sport_pattern("hockey"),     _sport_pattern("basketball"),
            _sport_pattern("football"),   _sport_pattern("baseball"),
            _sport_pattern("soccer"),     _sport_pattern("golf"),
            _sport_pattern("tennis"),
            r"\bteam\b",       r"\bplayer\b",      r"\bcoach\b",      r"\bfranchise\b",
            r"\bleague\b",     r"\broster\b",      r"\bstandings\b",  r"\bchampionship\b",
            r"\bplayoffs\b",   r"\bseason\b",      r"\bdrafted\b",    r"\btraded\b",
            r"\bsigned\b",
            r"\bwho owns\b",   r"\bwho plays\b",   r"\bwho played\b",
            r"\bwhich team\b", r"\bwhat team\b",
        ]),
        ("opinion_advice", [
            r"\bwhich is better\b", r"\bwhat do you think\b", r"\bpros and cons\b",
            r"\brecommend\b",       r"\badvice\b",          r"\bopinion\b",
            r"\bworth it\b",        r"\bcompare\b",         r"\bversus\b",
            r"\bvs\b",              r"\bdifference between\b",
        ]),
        ("definition", [
            r"\bwhat is\b",         r"\bwhat are\b",       r"\bexplain\b",
            r"\bdefine\b",          r"\bmeaning of\b",     r"\bwhat does\b",
            r"\bdescribe\b",        r"\btell me about\b",  r"\boverview of\b",
        ]),
        ("creative", [
            r"\bwrite me\b",        r"\bdraft\b",          r"\bbrainstorm\b",
            r"\bstory\b",           r"\bpoem\b",           r"\bcreative\b",
            r"\bimagine\b",         r"\bcome up with\b",   r"\bgenerate\b",
        ]),
        ("how_to", [
            r"how do i\b",          r"\bhow to\b",          r"\bsteps to\b",
            r"\bstep by step\b",    r"\binstructions for\b",r"\bguide me\b",
            r"\bwalk me through\b", r"\bhow should i\b",    r"\bwhat is the best way to\b",
        ]),
        ("language_task", [
            r"\bsummarize\b",       r"\brewrite\b",        r"\btranslate\b",
            r"\bfix grammar\b",     r"\brephrase\b",       r"\bsimplify\b",
            r"\bmake this shorter\b",r"\bedit this\b",     r"\bimprove this\b",
            r"\bclean up\b",        r"\bproofread\b",
        ]),
        ("conversational", [
            r"\brecipe\b",      r"\bcook\b",        r"\bfood\b",        r"\bmake\b",
            r"how do i\b",      r"what should i\b", r"\bhelp me\b",     r"\bidea\b",
            r"\bsuggest\b",     r"\btips\b",        r"\beasy\b",        r"\bsimple\b",
            r"\bbetter\b",      r"\bbest way\b",    r"\bwhat can\b",    r"should i\b",
            r"\bcan i\b",
        ]),
    ]

    # Skills that are always answered locally — cloud escalation is never attempted.
    # Evaluated after classification; log as "local-only skill — no escalation".
    LOCAL_ONLY_SKILLS: frozenset[str] = frozenset({
        "conversational",
        "definition",
        "creative",
        "how_to",
        "opinion_advice",
        "language_task",
    })

    # Skill → (cloud_llms provider key, model ID)
    # Local-only skills are intentionally absent — they never reach cloud dispatch.
    _SKILL_MAP: dict[str, tuple[str, str]] = {
        "coding":         ("claude", "claude-sonnet-4-20250514"),
        "math_data":      ("gemini", "gemini-2.0-flash"),
        "current_events":  ("grok",   None),
        "sports_people":   ("grok",   None),
        "general":        ("openai", "gpt-4o"),
    }

    def classify(self, prompt: str) -> str:
        """Return the skill label that best matches prompt, or 'general'."""
        text = prompt.lower()
        scores: dict[str, int] = {
            skill: sum(1 for pat in patterns if re.search(pat, text))
            for skill, patterns in self._PATTERNS
        }
        best_skill, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best_skill if best_score > 0 else "general"

    async def dispatch(
        self,
        query: str,
        skill: str,
        system: str = "",
        messages: list = [],
        local_answer: str | None = None,
        confidence: float | None = None,
    ) -> cloud_llms.CloudResponse:
        """Call the cloud LLM mapped to skill with fallback chain on error.

        Skips providers that are in the degraded set (quota/billing failures).
        Falls back: primary → openai → claude → grok → graceful message.
        Never raises; always returns a CloudResponse.
        """
        provider, model = self._SKILL_MAP.get(skill, self._SKILL_MAP["general"])

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        _log.info(
            "Routing to %s — skill=%s confidence=%s query_preview=%.50s",
            provider,
            skill,
            conf_str,
            query,
            extra={
                "provider": provider,
                "model": model,
                "skill": skill,
                "confidence": confidence,
                "query_preview": query[:50],
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )

        # Build fallback chain: primary first, then others (skipping duplicates)
        fallback_chain: list[tuple[str, str | None]] = [(provider, model)]
        for fallback_provider in ("openai", "claude", "grok"):
            if fallback_provider != provider:
                fallback_chain.append((fallback_provider, None))

        last_exc: Exception | None = None
        for attempt_provider, attempt_model in fallback_chain:
            # Skip providers known to be in a quota/billing failure window
            if _is_degraded(attempt_provider):
                expiry = _degraded.get(attempt_provider, 0)
                _log.info(
                    "Skipping %s — marked degraded until %s",
                    attempt_provider,
                    datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(timespec="seconds"),
                    extra={
                        "provider": attempt_provider,
                        "degraded_until": datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                continue

            try:
                return await cloud_llms.query(
                    attempt_provider, query, system, model=attempt_model, messages=messages
                )
            except cloud_llms.ProviderError as exc:
                if _is_billing_error(exc):
                    _mark_degraded(attempt_provider)
                _log.error(
                    "provider_failed provider=%s error_type=%s trying_next_fallback",
                    attempt_provider,
                    exc.error_type,
                    extra={
                        "provider": attempt_provider,
                        "error_type": exc.error_type,
                        "status_code": exc.status_code,
                        "error": str(exc),
                    },
                    exc_info=True,
                )
                last_exc = exc

        # All providers exhausted — return graceful message
        _log.error(
            "all_providers_failed skill=%s last_error=%s",
            skill,
            last_exc,
            extra={"skill": skill, "fallback_chain": [p for p, _ in fallback_chain]},
        )
        if local_answer is not None:
            msg = (
                "I encountered an issue reaching that service. "
                f"Here's what I know locally: {local_answer}"
            )
        else:
            msg = "Sorry, I ran into an issue reaching all available services. Please try again."
        return cloud_llms.CloudResponse(
            answer=msg,
            provider="error",
            model="none",
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
        )


# Module-level singleton — import and call directly if needed.
skill_router = SkillRouter()

# Convenient module-level alias so main.py can reference without going through the class.
LOCAL_ONLY_SKILLS: frozenset[str] = SkillRouter.LOCAL_ONLY_SKILLS


# ---------------------------------------------------------------------------
# Local-first retry helper
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"\b(?:please|could you|can you|i was wondering|i'?d like to know|would you mind)\b",
    re.IGNORECASE,
)


def _simplify_query(query: str) -> str:
    """Strip filler phrases and trim queries over 20 words to their core clause."""
    simplified = _FILLER_RE.sub("", query).strip()
    simplified = re.sub(r"\s{2,}", " ", simplified)
    words = simplified.split()
    if len(words) > 20:
        last_comma = simplified.rfind(",")
        if last_comma != -1:
            candidate = simplified[last_comma + 1:].strip()
            if candidate:
                simplified = candidate
        else:
            last_period = simplified.rfind(".")
            if last_period != -1 and last_period < len(simplified) - 1:
                candidate = simplified[last_period + 1:].strip()
                if candidate:
                    simplified = candidate
    return simplified or query
