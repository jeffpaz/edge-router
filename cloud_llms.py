import asyncio
import logging
import random
import time
from dataclasses import dataclass

import config

logging.getLogger("edge_router").addHandler(logging.NullHandler())
_log = logging.getLogger("edge_router.cloud_llms")

# ---------------------------------------------------------------------------
# Backoff settings
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY  = 1.0
_MAX_DELAY   = 60.0


async def _backoff_sleep(attempt: int, provider: str) -> None:
    """Exponential backoff with full jitter before a rate-limit retry."""
    ceiling = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    delay = random.uniform(0, ceiling)
    _log.warning(
        "%s rate-limited — attempt %d/%d, retrying in %.1fs",
        provider, attempt + 1, _MAX_RETRIES, delay,
    )
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Raised when a cloud LLM call fails after all retries."""

    def __init__(self, provider: str, error_type: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.error_type = error_type
        self.status_code = status_code
        detail = f" (HTTP {status_code})" if status_code else ""
        super().__init__(f"[{provider}] {error_type}{detail}: {message}")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class CloudResponse:
    answer: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Provider functions
# ---------------------------------------------------------------------------

async def query_claude(prompt: str, system: str = "", model: str | None = None, messages: list = []) -> CloudResponse:
    import anthropic

    resolved = model or config.CLAUDE_MODEL
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    kwargs: dict = {
        "model": resolved,
        "max_tokens": 2048,
        "messages": [*messages, {"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    try:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                msg = await client.messages.create(**kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000
                return CloudResponse(
                    answer=msg.content[0].text,
                    provider="claude",
                    model=resolved,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                )
            except anthropic.RateLimitError:
                if attempt == _MAX_RETRIES:
                    raise
                await _backoff_sleep(attempt, "claude")
    except anthropic.RateLimitError as exc:
        _log.error(
            "provider_error provider=claude error_type=rate_limit",
            extra={"provider": "claude", "error_type": "rate_limit"},
            exc_info=True,
        )
        raise ProviderError("claude", "rate_limit", str(exc)) from exc
    except anthropic.APITimeoutError as exc:
        _log.error(
            "provider_error provider=claude error_type=timeout",
            extra={"provider": "claude", "error_type": "timeout"},
            exc_info=True,
        )
        raise ProviderError("claude", "timeout", str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        _log.error(
            "provider_error provider=claude error_type=connection_error",
            extra={"provider": "claude", "error_type": "connection_error"},
            exc_info=True,
        )
        raise ProviderError("claude", "connection_error", str(exc)) from exc
    except anthropic.APIStatusError as exc:
        _log.error(
            "provider_error provider=claude error_type=api_status status_code=%s",
            exc.status_code,
            extra={"provider": "claude", "error_type": "api_status_error", "status_code": exc.status_code},
            exc_info=True,
        )
        raise ProviderError("claude", "api_status_error", str(exc), exc.status_code) from exc
    except Exception as exc:
        _log.error(
            "provider_error provider=claude error_type=%s",
            type(exc).__name__,
            extra={"provider": "claude", "error_type": type(exc).__name__},
            exc_info=True,
        )
        raise ProviderError("claude", type(exc).__name__, str(exc)) from exc

    raise RuntimeError("unreachable")


async def query_gemini(prompt: str, system: str = "", model: str | None = None, messages: list = []) -> CloudResponse:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    resolved = model or config.GEMINI_MODEL
    genai.configure(api_key=config.GOOGLE_API_KEY)
    gmodel = genai.GenerativeModel(
        model_name=resolved,
        system_instruction=system or None,
    )
    chat_history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in messages
    ]

    try:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                chat = gmodel.start_chat(history=chat_history)
                response = await chat.send_message_async(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = response.usage_metadata
                return CloudResponse(
                    answer=response.text,
                    provider="gemini",
                    model=resolved,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=getattr(usage, "prompt_token_count", 0),
                    output_tokens=getattr(usage, "candidates_token_count", 0),
                )
            except ResourceExhausted:
                if attempt == _MAX_RETRIES:
                    raise
                await _backoff_sleep(attempt, "gemini")
    except ResourceExhausted as exc:
        _log.error(
            "provider_error provider=gemini error_type=rate_limit",
            extra={"provider": "gemini", "error_type": "rate_limit"},
            exc_info=True,
        )
        raise ProviderError("gemini", "rate_limit", str(exc)) from exc
    except Exception as exc:
        _log.error(
            "provider_error provider=gemini error_type=%s",
            type(exc).__name__,
            extra={"provider": "gemini", "error_type": type(exc).__name__},
            exc_info=True,
        )
        raise ProviderError("gemini", type(exc).__name__, str(exc)) from exc

    raise RuntimeError("unreachable")


async def query_grok(prompt: str, system: str = "", model: str | None = None, messages: list = []) -> CloudResponse:
    import openai
    from openai import AsyncOpenAI

    resolved = model or config.GROK_MODEL
    client = AsyncOpenAI(api_key=config.XAI_API_KEY, base_url="https://api.x.ai/v1")
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)
    api_messages.append({"role": "user", "content": prompt})

    try:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                completion = await client.chat.completions.create(model=resolved, messages=api_messages)
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = completion.usage
                return CloudResponse(
                    answer=completion.choices[0].message.content or "",
                    provider="grok",
                    model=resolved,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                )
            except openai.RateLimitError:
                if attempt == _MAX_RETRIES:
                    raise
                await _backoff_sleep(attempt, "grok")
    except openai.RateLimitError as exc:
        _log.error(
            "provider_error provider=grok error_type=rate_limit",
            extra={"provider": "grok", "error_type": "rate_limit"},
            exc_info=True,
        )
        raise ProviderError("grok", "rate_limit", str(exc)) from exc
    except openai.APITimeoutError as exc:
        _log.error(
            "provider_error provider=grok error_type=timeout",
            extra={"provider": "grok", "error_type": "timeout"},
            exc_info=True,
        )
        raise ProviderError("grok", "timeout", str(exc)) from exc
    except openai.APIConnectionError as exc:
        _log.error(
            "provider_error provider=grok error_type=connection_error",
            extra={"provider": "grok", "error_type": "connection_error"},
            exc_info=True,
        )
        raise ProviderError("grok", "connection_error", str(exc)) from exc
    except openai.APIStatusError as exc:
        _log.error(
            "provider_error provider=grok error_type=api_status status_code=%s",
            exc.status_code,
            extra={"provider": "grok", "error_type": "api_status_error", "status_code": exc.status_code},
            exc_info=True,
        )
        raise ProviderError("grok", "api_status_error", str(exc), exc.status_code) from exc
    except Exception as exc:
        _log.error(
            "provider_error provider=grok error_type=%s",
            type(exc).__name__,
            extra={"provider": "grok", "error_type": type(exc).__name__},
            exc_info=True,
        )
        raise ProviderError("grok", type(exc).__name__, str(exc)) from exc

    raise RuntimeError("unreachable")


async def query_openai(prompt: str, system: str = "", model: str | None = None, messages: list = []) -> CloudResponse:
    import openai
    from openai import AsyncOpenAI

    resolved = model or config.OPENAI_MODEL
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)
    api_messages.append({"role": "user", "content": prompt})

    try:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                completion = await client.chat.completions.create(model=resolved, messages=api_messages)
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = completion.usage
                return CloudResponse(
                    answer=completion.choices[0].message.content or "",
                    provider="openai",
                    model=resolved,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                )
            except openai.RateLimitError:
                if attempt == _MAX_RETRIES:
                    raise
                await _backoff_sleep(attempt, "openai")
    except openai.RateLimitError as exc:
        _log.error(
            "provider_error provider=openai error_type=rate_limit",
            extra={"provider": "openai", "error_type": "rate_limit"},
            exc_info=True,
        )
        raise ProviderError("openai", "rate_limit", str(exc)) from exc
    except openai.APITimeoutError as exc:
        _log.error(
            "provider_error provider=openai error_type=timeout",
            extra={"provider": "openai", "error_type": "timeout"},
            exc_info=True,
        )
        raise ProviderError("openai", "timeout", str(exc)) from exc
    except openai.APIConnectionError as exc:
        _log.error(
            "provider_error provider=openai error_type=connection_error",
            extra={"provider": "openai", "error_type": "connection_error"},
            exc_info=True,
        )
        raise ProviderError("openai", "connection_error", str(exc)) from exc
    except openai.APIStatusError as exc:
        _log.error(
            "provider_error provider=openai error_type=api_status status_code=%s",
            exc.status_code,
            extra={"provider": "openai", "error_type": "api_status_error", "status_code": exc.status_code},
            exc_info=True,
        )
        raise ProviderError("openai", "api_status_error", str(exc), exc.status_code) from exc
    except Exception as exc:
        _log.error(
            "provider_error provider=openai error_type=%s",
            type(exc).__name__,
            extra={"provider": "openai", "error_type": type(exc).__name__},
            exc_info=True,
        )
        raise ProviderError("openai", type(exc).__name__, str(exc)) from exc

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "claude": query_claude,
    "gemini": query_gemini,
    "grok":   query_grok,
    "openai": query_openai,
}


async def query(
    provider: str,
    prompt: str,
    system: str = "",
    model: str | None = None,
    messages: list = [],
) -> CloudResponse:
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from {list(_PROVIDERS)}")
    return await fn(prompt, system, model, messages)
