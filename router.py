import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cloud_llms
import config
import local_llm

# ---------------------------------------------------------------------------
# Logger  (NullHandler keeps this library-safe; callers configure output)
# ---------------------------------------------------------------------------

logging.getLogger("edge_router").addHandler(logging.NullHandler())
_log = logging.getLogger("edge_router.router")


# ---------------------------------------------------------------------------
# SkillRouter
# ---------------------------------------------------------------------------

class SkillRouter:
    """Classifies a query into a skill and dispatches to the right cloud LLM.

    Skill priority on ties (same match count): coding > math_data > current_events.
    """

    # Evaluated in order; first skill to outscore the rest wins ties.
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
        ]),
        ("current_events", [
            r"\btoday\b",      r"\bnews\b",        r"\blatest\b",     r"\b202[4-9]\b",
            r"\bwho won\b",    r"\bbreaking\b",    r"\brecently\b",   r"\bcurrently\b",
            r"\bupdate\b",     r"\belection\b",    r"\bannounced\b",  r"\breleased\b",
            r"\bhappened\b",   r"\bright now\b",   r"\bthis week\b",  r"\bthis month\b",
        ]),
    ]

    # Skill → (cloud_llms provider key, model ID)
    _SKILL_MAP: dict[str, tuple[str, str]] = {
        "coding":         ("claude", "claude-sonnet-4-20250514"),
        "math_data":      ("gemini", "gemini-1.5-pro"),
        "current_events": ("grok",   "grok-2-latest"),
        "general":        ("openai", "gpt-4o"),
    }

    def classify(self, prompt: str) -> str:
        """Return the skill label that best matches prompt, or 'general'."""
        text = prompt.lower()
        # Score each non-general skill; first in _PATTERNS wins ties naturally.
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
    ) -> cloud_llms.CloudResponse:
        """Call the cloud LLM mapped to skill and log the routing decision."""
        provider, model = self._SKILL_MAP.get(skill, self._SKILL_MAP["general"])

        _log.info(
            "timestamp=%s skill=%s provider=%s model=%s query=%.80r",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            skill,
            provider,
            model,
            query,
        )

        return await cloud_llms.query(provider, query, system, model=model)


# Module-level singleton — import and call directly if needed.
skill_router = SkillRouter()


# ---------------------------------------------------------------------------
# RouterResponse
# ---------------------------------------------------------------------------

@dataclass
class RouterResponse:
    answer: str
    source: str          # "local" | cloud provider name
    model: str
    skill: str
    local_confidence: float
    tokens_used: int
    latency_ms: float    # cloud latency; -1 for local-only responses
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------

async def route(
    prompt: str,
    system: str = "",
    force_provider: str | None = None,
) -> RouterResponse:
    """Try local LLM first; escalate via SkillRouter when confidence is low.

    Args:
        prompt:         User query.
        system:         Optional system prompt forwarded to whichever LLM answers.
        force_provider: Bypass routing — send directly to "local", "claude",
                        "gemini", "grok", or "openai" (uses global config model).
    """
    skill = skill_router.classify(prompt)

    # ── Forced provider ──────────────────────────────────────────────────────
    if force_provider == "local":
        local = await local_llm.query(prompt, system)
        return RouterResponse(
            answer=local["answer"],
            source="local",
            model=local["model"],
            skill=skill,
            local_confidence=local["confidence"],
            tokens_used=local["prompt_tokens"] + local["completion_tokens"],
            latency_ms=-1,
            metadata={"forced": True, "signals": local["signals"]},
        )

    if force_provider in ("claude", "gemini", "grok", "openai"):
        # Uses global config model, not skill-specific model.
        cloud = await cloud_llms.query(force_provider, prompt, system)
        return RouterResponse(
            answer=cloud.answer,
            source=cloud.provider,
            model=cloud.model,
            skill=skill,
            local_confidence=-1.0,
            tokens_used=cloud.input_tokens + cloud.output_tokens,
            latency_ms=cloud.latency_ms,
            metadata={"forced": True},
        )

    # ── Normal path: local first ─────────────────────────────────────────────
    local = await local_llm.query(prompt, system)

    if not local["should_escalate"]:
        return RouterResponse(
            answer=local["answer"],
            source="local",
            model=local["model"],
            skill=skill,
            local_confidence=local["confidence"],
            tokens_used=local["prompt_tokens"] + local["completion_tokens"],
            latency_ms=-1,
            metadata={
                "threshold": config.CONFIDENCE_THRESHOLD,
                "signals": local["signals"],
            },
        )

    # ── Escalate: SkillRouter picks cloud LLM by skill ───────────────────────
    cloud = await skill_router.dispatch(prompt, skill, system)
    local_tokens = local["prompt_tokens"] + local["completion_tokens"]

    return RouterResponse(
        answer=cloud.answer,
        source=cloud.provider,
        model=cloud.model,
        skill=skill,
        local_confidence=local["confidence"],
        tokens_used=local_tokens + cloud.input_tokens + cloud.output_tokens,
        latency_ms=cloud.latency_ms,
        metadata={
            "threshold": config.CONFIDENCE_THRESHOLD,
            "escalated": True,
            "local_model": local["model"],
            "signals": local["signals"],
        },
    )
