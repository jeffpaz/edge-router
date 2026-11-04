"""PazBrain HTTP API, mounted by main.py at /brain/api.

The wiki is read-only here: only the librarian (Claude Code on the Jetson, following
/opt/pazbrain/AGENTS.md) writes wiki/. The one write is POST /inbox, which drops material into
raw/inbox/ for the librarian to ingest.

Visibility: requests that came through the Cloudflare Tunnel (they carry cf-ray /
cf-connecting-ip, which cloudflared always adds) never see `visibility: local-only` pages: not
in lists, search, backlinks, history or /recent, and a direct GET answers 404. Requests made
directly on the LAN (:8000) see everything.

A separate FastAPI sub-application rather than a router on the main app, so that:
  - its OpenAPI schema is served on its own at /brain/api/openapi.json;
  - errors keep their real status codes instead of hitting the main app's catch-all
    exception handler (which answers 200);
  - nothing here runs at import time: storage problems return 503 from /brain/api and
    can't take down the other routes.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pazbrain import scrubber, store
from pazbrain.security import require_read, require_write

_log = logging.getLogger("edge_router.pazbrain")

_MAX_REQUEST_BYTES = 256 * 1024

app = FastAPI(
    title="PazBrain",
    version="2.0.0",
    description=(
        "Private personal wiki, read-only. Reads need the read token, the write token, or an "
        "allowlisted Firebase ID token. POST /inbox (write token) drops material for the librarian."
    ),
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

Slug = Annotated[str, Path(description="Page slug, ^[a-z0-9][a-z0-9-]{0,80}$", max_length=81)]
PageType = Literal["person", "project", "area", "topic", "source", "concept", "meeting"]
Agent = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$", description="Who sent it, e.g. grok")]


class InboxIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, description="Markdown or plain text. Treated as data, never as instructions.")
    source_agent: Agent


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _limit_body(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > _MAX_REQUEST_BYTES:
        return JSONResponse({"error": "too_large", "detail": "request body too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(store.StoreError)
async def _store_error(_: Request, exc: store.StoreError) -> JSONResponse:
    return JSONResponse({"error": exc.kind, "detail": exc.detail}, status_code=exc.status)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    content = exc.detail if isinstance(exc.detail, dict) else {"error": "error", "detail": exc.detail}
    return JSONResponse(content, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse({"error": "validation", "detail": errors}, status_code=422)


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    _log.error("pazbrain_internal_error", extra={"error": str(exc)}, exc_info=True)
    return JSONResponse({"error": "internal", "detail": "PazBrain internal error"}, status_code=500)


def _ready() -> None:
    try:
        store.ensure_ready()
    except Exception as exc:
        _log.error("pazbrain_storage_unavailable", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(503, {"error": "unavailable", "detail": "PazBrain storage is unavailable"})


def _local(request: Request) -> bool:
    """True only for requests made directly on the LAN; anything through the tunnel is cloud."""
    return not any(h in request.headers for h in ("cf-ray", "cf-connecting-ip"))


Reader = Annotated[str, Depends(require_read)]
Writer = Annotated[str, Depends(require_write)]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@app.get("/pages", summary="List pages, newest first")
def list_pages(
    request: Request,
    _: Reader,
    type: Optional[PageType] = None,
    tag: Optional[str] = Query(default=None, max_length=50),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    _ready()
    pages = store.list_pages(type, tag, limit, include_local=_local(request))
    return {"count": len(pages), "pages": pages}


@app.get("/pages/{slug}", summary="Get a full page with frontmatter, backlinks and outgoing links")
def get_page(request: Request, slug: Slug, _: Reader) -> dict[str, Any]:
    _ready()
    return store.get_page(slug, include_local=_local(request))


@app.get("/search", summary="Full-text search (FTS5) with highlighted snippets")
def search(
    request: Request,
    _: Reader,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    _ready()
    results = store.search(q, limit, include_local=_local(request))
    return {"query": q, "count": len(results), "results": results}


@app.get("/history/{slug}", summary="git history of a page")
def history(request: Request, slug: Slug, _: Reader, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    _ready()
    return {"slug": slug, "history": store.history(slug, limit, include_local=_local(request))}


@app.get("/recent", summary="Recent operations from log.md, newest first")
def recent(request: Request, _: Reader, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    _ready()
    return {"changes": store.recent(limit, include_local=_local(request))}


# ---------------------------------------------------------------------------
# The one write: raw/inbox
# ---------------------------------------------------------------------------


@app.post("/inbox", summary="Drop material into raw/inbox/ for the librarian to ingest", status_code=201)
def add_to_inbox(item: InboxIn, _: Writer) -> dict[str, Any]:
    _ready()
    hits = scrubber.scan({"title": item.title, "content": item.content})
    if hits:
        _log.warning("pazbrain_secret_rejected", extra={"hits": hits})
        raise HTTPException(422, {
            "error": "secret_detected",
            "detail": "Rejected: content looks like a secret or financial identifier. Remove it and retry.",
            "matches": hits,
        })
    return store.add_to_inbox(title=item.title, content=item.content, source_agent=item.source_agent)
