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
_BASE_DELAY  = 1.0   # seconds before first retry
_MAX_DELAY   = 60.0  # hard ceiling regardless of attempt number


async def _backoff_sleep(attempt: int, provider: str) -> None:
    """Exponential backoff with full jitter before a rate-limit retry."""
    # Full jitter: sleep uniform(0, min(cap, base * 2^attempt))
    # This avoids thundering-herd when many requests hit the limit simultaneously.
    ceiling = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    delay = random.uniform(0, ceiling)
    _log.warning(
        "%s rate-limited — attempt %d/%d, retrying in %.1fs",
        provider, attempt + 1, _MAX_RETRIES, delay,
    )
    await asyncio.sleep(delay)


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

async def query_claude(prompt: str, system: str = "", model: str | None = None) -> CloudResponse:
    import anthropic

    resolved = model or config.CLAUDE_MODEL
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    kwargs: dict = {
        "model": resolved,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

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

    raise RuntimeError("unreachable")  # satisfies type checkers


async def query_gemini(prompt: str, system: str = "", model: str | None = None) -> CloudResponse:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    resolved = model or config.GEMINI_MODEL
    genai.configure(api_key=config.GOOGLE_API_KEY)
    gmodel = genai.GenerativeModel(
        model_name=resolved,
        system_instruction=system or None,
    )

    for attempt in range(_MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = await gmodel.generate_content_async(prompt)
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

    raise RuntimeError("unreachable")


async def query_grok(prompt: str, system: str = "", model: str | None = None) -> CloudResponse:
    # xAI Grok exposes an OpenAI-compatible API.
    import openai
    from openai import AsyncOpenAI

    resolved = model or config.GROK_MODEL
    client = AsyncOpenAI(api_key=config.XAI_API_KEY, base_url="https://api.x.ai/v1")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(_MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            completion = await client.chat.completions.create(model=resolved, messages=messages)
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

    raise RuntimeError("unreachable")


async def query_openai(prompt: str, system: str = "", model: str | None = None) -> CloudResponse:
    import openai
    from openai import AsyncOpenAI

    resolved = model or config.OPENAI_MODEL
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(_MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            completion = await client.chat.completions.create(model=resolved, messages=messages)
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
) -> CloudResponse:
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from {list(_PROVIDERS)}")
    return await fn(prompt, system, model)
