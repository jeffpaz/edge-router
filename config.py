import os


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Legacy global threshold — fallback for skills not in SKILL_THRESHOLDS.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Per-skill confidence thresholds. A value of 1.0 means always escalate to cloud.
# Override any individual skill via env: SKILL_THRESHOLD_<SKILL_UPPER>=<float>
# e.g. SKILL_THRESHOLD_CODING=0.80
def _build_skill_thresholds() -> dict[str, float]:
    defaults: dict[str, float] = {
        "conversational":  0.40,
        "definition":      0.45,
        "creative":        0.40,
        "how_to":          0.45,
        "opinion_advice":  0.45,
        "language_task":   0.50,
        "general":         0.60,
        "coding":          0.75,
        "math_data":       1.0,   # always escalate
        "current_events":  1.0,   # always escalate
        "sports_people":   1.0,   # always escalate
    }
    return {
        skill: float(os.getenv(f"SKILL_THRESHOLD_{skill.upper()}", str(val)))
        for skill, val in defaults.items()
    }

SKILL_THRESHOLDS: dict[str, float] = _build_skill_thresholds()

# Maximum entries in the in-memory local answer cache (FIFO eviction on overflow).
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "200"))

# Enable local-first retry with a simplified query before escalating to cloud.
# Only applies to "general" and "coding" skills.
RETRY_ENABLED = os.getenv("RETRY_ENABLED", "true").lower() == "true"

# Which cloud LLM to fall back to when local confidence is low.
# Options: "claude", "gemini", "grok", "openai"
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "claude")
