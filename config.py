import os


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Minimum confidence score [0.0, 1.0] to use local LLM response.
# Below this threshold, the query is escalated to a cloud LLM.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Which cloud LLM to fall back to when local confidence is low.
# Options: "claude", "gemini", "grok", "openai"
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "claude")
