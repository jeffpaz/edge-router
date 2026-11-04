"""PazBrain inside the router: the "brain" intent and lib/brain search (additive).

lib/brain (the FTS5 index the librarian keeps in /opt/pazbrain/.index/brain.db) is imported
straight from the /opt/pazbrain mount. It opens the index read-only.

Tiers (AGENTS.md "Visibility"): local-only pages never reach a cloud model or a cloud client.
  - tier "local": only for work that never leaves the Jetson and the Mac mini, i.e. private
    requests (the "Stay private / local" checkbox), which run inside privacy.private_scope().
  - tier "cloud": everything else, including LAN requests that might escalate to a cloud LLM.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

_LIB_DIR = "/opt/pazbrain/lib"
_log = logging.getLogger("edge_router.pazbrain")
_brain = None

# Questions about Jeff's own notes, projects and history. Checked before the other skills.
BRAIN_RE = re.compile(
    r"\bpazbrain\b|\bmy (?:second )?brain\b|\b(?:in|from|check|search) (?:the|my) brain\b"
    r"|\bmy (?:wiki|notes)\b|\bwhat do (?:i|we) know about\b|\bwhat did (?:i|we) (?:decide|say|note)\b"
    r"|\bremind me (?:about|what|who|when|why)\b|\bmy (?:projects?|hoa|board)\b",
    re.IGNORECASE,
)


def is_brain_query(text: str) -> bool:
    return bool(BRAIN_RE.search(text or ""))


def _lib():
    global _brain
    if _brain is None:
        if _LIB_DIR not in sys.path:
            sys.path.insert(0, _LIB_DIR)
        import brain  # /opt/pazbrain/lib/brain.py

        _brain = brain
    return _brain


def search(query: str, *, tier: str, limit: int = 5) -> list[dict[str, Any]]:
    """lib/brain.search with the tier applied twice: by the library and again here."""
    if tier not in ("local", "cloud"):
        raise ValueError(f"unknown tier {tier!r}")
    local = tier == "local"
    try:
        hits = _lib().search(query, limit=limit, include_local_only=local)
    except Exception as exc:  # a missing or busy index must never break a query
        _log.warning("brain_search_failed", extra={"error": f"{type(exc).__name__}: {exc}"[:200]})
        return []
    if not local:
        hits = [h for h in hits if h.get("visibility") == "cloud"]
    return hits[:limit]


def _clean(text: str) -> str:
    return " ".join((text or "").replace("[", "").replace("]", "").split())


def context(query: str, *, tier: str, limit: int = 5, max_chars: int = 1200) -> tuple[str, list[dict]]:
    """A compact notes block (the Jetson's Ollama runs with a 512-token window) and the hits."""
    hits = search(query, tier=tier, limit=limit)
    if not hits:
        return "", []
    lines = ["Notes from Jeff's PazBrain wiki, best match first. Use them if they answer the "
             "question, cite them as [[slug]], and say so plainly if they don't cover it:"]
    for h in hits:
        about = _clean(h.get("summary") or "") or _clean(h.get("snippet") or "")
        lines.append(f"- [[{h['slug']}]] {_clean(h.get('title') or h['slug'])}: {about}"[:260])
    return "\n".join(lines)[:max_chars], hits


def prompt_with_context(query: str, *, tier: str) -> tuple[str, list[dict]]:
    """The user's question with the top 5 hits in front of it; unchanged when nothing matches."""
    block, hits = context(query, tier=tier)
    if not block:
        return query, []
    return f"{block}\n\nQuestion: {query}", hits


def citations(hits: list[dict]) -> list[dict]:
    return [{"source": "pazbrain", "slug": h["slug"], "title": h.get("title") or h["slug"],
             "path": h.get("path", ""), "visibility": h.get("visibility", "")} for h in hits]
