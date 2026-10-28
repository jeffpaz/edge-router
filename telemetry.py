"""Per-request telemetry for POST /ask/stream, from whichever host is answering.

A `Run` owns one tegrastats subprocess for the life of a single request and
turns Ollama stream chunks into the three SSE event kinds the web panel draws:

  token    {text}
  metrics  {t, host, cpu, gpu, ram, tok, tps, kv}   ~8/s while the run is open
  phases   {call, label, load, prefill, decode, prompt_eval_count, eval_count, wall, ...}

`host` says whose CPU / GPU / RAM the sample is: "jetson" (tegrastats) or "mini".
Once a call goes to the Mac mini, the run also reads the mini's own metrics stream
(PazRAG GET /sys/stream, via the `mini_sampler` it was given) and reports those.
The documents lane is one request to the mini, so its phases arrive at the end
(`remote_calls`), including a `search` step (embed + retrieve + rerank).

plus `call` (an Ollama call started) and `replace` (swap the answer text).
Events go onto an asyncio.Queue that the endpoint drains into the response.
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import shutil
import signal
import time
from collections import deque

_log = logging.getLogger("edge_router.telemetry")

# Inside the container the host's binary is reachable through the read-only
# / bind mount (see docker-compose.yml); fall back to PATH for bare-metal runs.
_TEGRASTATS = next(
    (p for p in ("/host_root/usr/bin/tegrastats", shutil.which("tegrastats") or "") if p and os.path.exists(p)),
    None,
)

SAMPLE_INTERVAL_MS = 125

# ---------------------------------------------------------------------------
# KV-cache estimate
# ---------------------------------------------------------------------------

# (layers, kv_heads, head_dim) — from each model's GGUF metadata.
MODEL_SPECS: dict[str, tuple[int, int, int]] = {
    "gemma2:2b":   (26, 4, 256),
    "llama3.2:3b": (28, 8, 128),
    "qwen3:4b-instruct": (36, 8, 128),   # the mini
}

_F16_BYTES = 2


def kv_bytes_per_token(model: str) -> int | None:
    """K and V, every layer, f16. None when the model isn't in MODEL_SPECS."""
    spec = MODEL_SPECS.get(model)
    if spec is None:
        return None
    layers, kv_heads, head_dim = spec
    return 2 * layers * kv_heads * head_dim * _F16_BYTES


# ---------------------------------------------------------------------------
# tegrastats parsing
# ---------------------------------------------------------------------------

_RAM_RE  = re.compile(r"\bRAM (\d+)/(\d+)MB")
_CPU_RE  = re.compile(r"\bCPU \[([^\]]*)\]")
_GPU_RE  = re.compile(r"\bGR3D_FREQ (\d+)%")
_CORE_RE = re.compile(r"(\d+)%")


def parse_tegrastats(line: str) -> dict:
    """One tegrastats line → {cpu, gpu, ram}. Missing fields come back None.

    CPU is the mean of the online cores; offline cores print as "off" and are
    skipped rather than counted as 0%.
    """
    out: dict = {"cpu": None, "gpu": None, "ram": None}
    if m := _RAM_RE.search(line):
        out["ram"] = int(m.group(1))
    if m := _GPU_RE.search(line):
        out["gpu"] = int(m.group(1))
    if m := _CPU_RE.search(line):
        cores = [int(c) for c in _CORE_RE.findall(m.group(1))]
        if cores:
            out["cpu"] = round(sum(cores) / len(cores), 1)
    return out


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


# The Run of the /ask/stream request this task belongs to, so the mini lanes in
# tiers.py can stream into it without every lane passing it along. Set at the top
# of each pipeline task; None everywhere else (/query, /query/stream, jobs).
current_run: contextvars.ContextVar["Run | None"] = contextvars.ContextVar("current_run", default=None)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class Run:
    """Telemetry + event queue for one /ask/stream request.

    Use as `async with Run(model) as run:`; leaving the block always kills
    tegrastats and stops the metrics ticker, including when the client
    disconnects and the response generator is cancelled.
    """

    def __init__(self, model: str, mini_sampler=None) -> None:
        self.model = model
        self.host = "jetson"                  # whose metrics the panel is shown
        self._mini_sampler = mini_sampler     # () -> async iterator of {cpu, gpu, ram}
        self._mini_latest: dict = {"cpu": None, "gpu": None, "ram": None}
        self._mini_started = False
        self.events: asyncio.Queue[str | None] = asyncio.Queue()
        self.t0 = time.monotonic()
        self.samples = 0

        self._bpt = kv_bytes_per_token(model)
        self._latest: dict = {"cpu": None, "gpu": None, "ram": None}
        self._proc: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task] = []

        self.calls = 0
        self.tokens = 0                       # cumulative output tokens, all calls
        self._token_times: deque[float] = deque()
        self._call_prompt_tokens = 0          # estimate until Ollama reports the real count
        self._call_tokens = 0
        self._call_t0 = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Run":
        if _TEGRASTATS:
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    _TEGRASTATS, "--interval", str(SAMPLE_INTERVAL_MS),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,   # own process group → killpg reaches any children
                )
                self._tasks.append(asyncio.create_task(self._read_tegrastats()))
            except OSError as exc:
                _log.warning("tegrastats_start_failed", extra={"error": str(exc)})
        self._tasks.append(asyncio.create_task(self._tick()))
        return self

    async def __aexit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Synchronous on purpose: on client disconnect this runs inside a
        cancelled anyio scope, where any await would be cancelled before the
        process is signalled. asyncio's child watcher reaps the exit."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        _killpg(proc.pid, signal.SIGTERM)
        asyncio.get_running_loop().call_later(1.0, _killpg, proc.pid, signal.SIGKILL)

    async def _read_tegrastats(self) -> None:
        assert self._proc and self._proc.stdout
        async for raw in self._proc.stdout:
            parsed = parse_tegrastats(raw.decode(errors="replace"))
            self._latest = {k: v if v is not None else self._latest[k] for k, v in parsed.items()}
            self.samples += 1

    async def _read_mini(self) -> None:
        try:
            async for sample in self._mini_sampler():
                self._mini_latest = {k: sample.get(k) for k in ("cpu", "gpu", "ram")}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the panel goes blank; the answer is unaffected
            _log.warning("mini_metrics_failed", extra={"error_type": type(exc).__name__})

    def use_host(self, host: str) -> None:
        """Report `host`'s metrics from now on, starting the mini stream on first use."""
        self.host = host
        if host == "mini" and self._mini_sampler and not self._mini_started:
            self._mini_started = True
            self._tasks.append(asyncio.create_task(self._read_mini()))

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(SAMPLE_INTERVAL_MS / 1000)
            self.events.put_nowait(sse("metrics", self.metrics()))

    # ── metrics ──────────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def _tps(self, now: float) -> float:
        """Tokens over the trailing second (or since the first token, if sooner)."""
        while self._token_times and now - self._token_times[0] > 1.0:
            self._token_times.popleft()
        if not self._token_times:
            return 0.0
        window = min(1.0, max(now - self._token_times[0], 0.125))
        return round(len(self._token_times) / window, 1)

    def _kv(self) -> int | None:
        if self._bpt is None or self.calls == 0:
            return None
        return (self._call_prompt_tokens + self._call_tokens) * self._bpt

    def metrics(self) -> dict:
        now = time.monotonic()
        return {
            "t":   round(now - self.t0, 3),
            "host": self.host,
            **(self._mini_latest if self.host == "mini" else self._latest),
            "tok": self.tokens,
            "tps": self._tps(now),
            "kv":  self._kv(),
            "n":   self.samples,
        }

    # ── events from the Ollama call ──────────────────────────────────────────

    def begin_call(self, label: str, prompt_chars: int, *, host: str = "jetson", model: str | None = None) -> None:
        """An Ollama call is starting on `host`. Prompt tokens are guessed at ~4 chars
        each until the final chunk reports prompt_eval_count."""
        self.use_host(host)
        self._bpt = kv_bytes_per_token(model or self.model)
        self.calls += 1
        self._call_prompt_tokens = prompt_chars // 4
        self._call_tokens = 0
        self._call_t0 = time.monotonic()
        if self.calls > 1:
            self.replace("")
        self.events.put_nowait(sse("call", {"call": self.calls, "label": label, "t": round(self.elapsed(), 3)}))

    def token(self, text: str) -> None:
        """One Ollama chunk = one output token. `text` is what the client should
        see, which may be empty while the confidence tag is being held back."""
        self.tokens += 1
        self._call_tokens += 1
        self._token_times.append(time.monotonic())
        if text:
            self.events.put_nowait(sse("token", {"text": text}))

    def end_call(self, label: str, final_chunk: dict) -> None:
        """Forward Ollama's own timings (ns → s) for the phase bar."""
        def secs(key: str) -> float:
            return round((final_chunk.get(key) or 0) / 1e9, 4)

        self._call_prompt_tokens = final_chunk.get("prompt_eval_count") or 0
        self.events.put_nowait(sse("phases", {
            "call":              self.calls,
            "label":             label,
            "load":              secs("load_duration"),
            "prefill":           secs("prompt_eval_duration"),
            "decode":            secs("eval_duration"),
            "total":             secs("total_duration"),
            "prompt_eval_count": final_chunk.get("prompt_eval_count") or 0,
            "eval_count":        final_chunk.get("eval_count") or 0,
            "wall":              round(time.monotonic() - self._call_t0, 4),
            "t_start":           round(self._call_t0 - self.t0, 4),
        }))

    def remote_calls(self, steps: list, *, offset: float, model: str) -> None:
        """Phases for steps that ran inside one mini request (PazRAG's Answer.calls),
        which started `offset` seconds into this run. Ollama steps also count toward
        out tok and the KV estimate; `search` steps have only a duration."""
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            def num(key: str, step=step) -> float:
                v = step.get(key)
                return v if isinstance(v, (int, float)) else 0
            self.calls += 1
            if "search" not in step:
                self._bpt = kv_bytes_per_token(model)
                self.tokens += int(num("eval_count"))
                self._call_prompt_tokens, self._call_tokens = int(num("prompt_eval_count")), int(num("eval_count"))
            self.events.put_nowait(sse("phases", {
                "call":              self.calls,
                "label":             str(step.get("label") or "mini")[:20],
                "load":              num("load"),
                "prefill":           num("prefill"),
                "decode":            num("decode"),
                "search":            num("search"),
                "total":             num("wall"),
                "prompt_eval_count": int(num("prompt_eval_count")),
                "eval_count":        int(num("eval_count")),
                "wall":              num("wall"),
                "t_start":           round(offset + num("t_start"), 4),
            }))

    def replace(self, text: str, **extra) -> None:
        self.events.put_nowait(sse("replace", {"text": text, **extra}))

    def emit(self, event: str, data: dict) -> None:
        self.events.put_nowait(sse(event, data))
