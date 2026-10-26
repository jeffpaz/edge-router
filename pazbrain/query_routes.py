"""POST /brain/query: read-only queries over the PazBrain Logseq graph (additive; included by main.py).

The logseq CLI lives on the host, not in this container (a second CLI here would start its own
db-worker-node on the same graph), so requests are validated here and forwarded to the host sidecar
pazbrain/scripts/brain_query.py (pazbrain-query.service) over the unix socket in the /opt/pazbrain
mount. The sidecar validates again and is the only thing that runs the CLI. Only the six operations
below exist; nothing reachable from here can write to the graph or run raw datalog.

Same auth as the PazBrain reads: read token, write token, or an allowlisted Firebase ID token.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Annotated, Any, Literal, Optional, Union

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pazbrain.security import require_read

_log = logging.getLogger("edge_router.pazbrain")

_SOCKET = "/opt/pazbrain/run/brain-query.sock"
_TIMEOUT_S = 20.0  # sidecar caps each CLI call at 15s
_RATE_PER_MIN = int(os.getenv("PAZBRAIN_QUERY_RATE_PER_MIN", "30"))

Text = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[^-\x00-\x1f\x7f][^\x00-\x1f\x7f]*$",
                            description="1-100 chars, no control characters, must not start with '-'")]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,80}$")]
Status = Literal["backlog", "canceled", "doing", "done", "in-review", "todo"]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchPage(_Op):
    op: Literal["search_page"]
    q: Text


class SearchBlock(_Op):
    op: Literal["search_block"]
    q: Text


class ShowPage(_Op):
    op: Literal["show_page"]
    slug: Slug
    level: int = Field(default=2, ge=1, le=3, description="1 = page only, 2 = top-level blocks, 3 = one more level")


class ListTask(_Op):
    op: Literal["list_task"]
    status: Optional[Status] = None
    limit: int = Field(default=20, ge=1, le=50)


class RecentUpdated(_Op):
    op: Literal["recent_updated"]
    days: int = Field(default=7, ge=1, le=90)


class TaskSearch(_Op):
    op: Literal["task_search"]
    status: Status
    title: str = Field(default="", max_length=100, pattern=r"^([^-\x00-\x1f\x7f][^\x00-\x1f\x7f]*)?$")
    days: int = Field(default=0, ge=0, le=365, description="0 = any time")


BrainQuery = Annotated[
    Union[SearchPage, SearchBlock, ShowPage, ListTask, RecentUpdated, TaskSearch],
    Field(discriminator="op"),
]

router = APIRouter(tags=["pazbrain"])

# Sliding one-minute window per caller identity (same shape as pazbrain.security's write limit).
# The sidecar runs one CLI call at a time, so this keeps one caller from queueing out the others.
_hits: dict[str, deque[float]] = {}
_hits_lock = threading.Lock()


def _rate_limit(who: str) -> None:
    now = time.monotonic()
    with _hits_lock:
        window = _hits.setdefault(who, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= _RATE_PER_MIN:
            retry = max(1, int(60 - (now - window[0])) + 1)
            raise HTTPException(
                429,
                {"error": "rate_limited", "detail": f"Max {_RATE_PER_MIN} queries per minute."},
                {"Retry-After": str(retry)},
            )
        window.append(now)


@router.post("/brain/query", summary="Read-only PazBrain graph query (allow-listed operations)")
async def brain_query(body: BrainQuery, who: Annotated[str, Depends(require_read)]) -> dict[str, Any]:
    _rate_limit(who)
    try:
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=_SOCKET), timeout=_TIMEOUT_S) as client:
            resp = await client.post("http://brain-query/query", json=body.model_dump())
    except httpx.TimeoutException:
        raise HTTPException(504, {"error": "timeout", "detail": "PazBrain query timed out."})
    except httpx.HTTPError:
        raise HTTPException(503, {"error": "unavailable", "detail": "PazBrain query service is not running."})

    try:
        data = resp.json()
    except ValueError:
        data = {}
    _log.info("brain_query", extra={"op": body.op, "who": who, "status": resp.status_code})
    if resp.status_code != 200:
        status = resp.status_code if resp.status_code in (400, 404, 504) else 502
        raise HTTPException(status, {"error": data.get("error", "upstream_error"), "detail": data.get("detail", "")})
    return data
