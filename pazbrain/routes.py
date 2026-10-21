"""PazBrain HTTP API, mounted by main.py at /brain/api.

A separate FastAPI sub-application rather than a router on the main app, so that:
  - its OpenAPI schema is served on its own at /brain/api/openapi.json and the main
    app's schema is unchanged;
  - errors keep their real status codes instead of hitting the main app's catch-all
    exception handler (which answers 200);
  - nothing here runs at import time: the wiki is initialised on first request, so a
    storage problem returns 503 from /brain/api and can't take down the other routes.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Optional, Union

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
    version="1.0.0",
    description=(
        "Private personal wiki. Reads need the read token, the write token, or an "
        "allowlisted Firebase ID token; writes need the write token."
    ),
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

Slug = Annotated[str, Path(description="Page slug, ^[a-z0-9][a-z0-9-]{0,80}$", max_length=81)]
Category = Literal["people", "projects", "areas", "topics", "decisions", "journal", "inbox"]
Author = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$", description="Who made the change, e.g. grok")]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class Source(BaseModel):
    type: Literal["email", "file", "chat"]
    ref: str = Field(min_length=1, max_length=500, description="Message id, file path, or chat reference")
    date: Optional[str] = Field(default=None, max_length=40, description="ISO date of the source")


class PageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=200)
    category: Category
    tags: list[str] = Field(default_factory=list, max_length=50)
    sources: list[Source] = Field(default_factory=list, max_length=200)
    confidence: Literal["stated", "inferred"]
    summary: str = Field(max_length=300, description="One line")
    body: str = Field(default="", description="Markdown. Use [[slug]] or [[Page Title]] to link pages.")
    updated_by: Author
    base_updated: Optional[str] = Field(
        default=None, description="Required when replacing an existing page: its current `updated` value"
    )


class AppendSection(BaseModel):
    op: Literal["append_section"]
    heading: str = Field(min_length=1, max_length=200)
    content: str


class ReplaceSection(BaseModel):
    op: Literal["replace_section"]
    heading: str = Field(min_length=1, max_length=200)
    content: str


class AddTags(BaseModel):
    op: Literal["add_tags"]
    tags: list[str] = Field(min_length=1, max_length=50)


class AddSources(BaseModel):
    op: Literal["add_sources"]
    sources: list[Source] = Field(min_length=1, max_length=200)


PatchOp = Annotated[Union[AppendSection, ReplaceSection, AddTags, AddSources], Field(discriminator="op")]


class PatchIn(BaseModel):
    ops: list[PatchOp] = Field(min_length=1, max_length=20)
    updated_by: Author
    base_updated: str = Field(description="The page's current `updated` value")
    summary: Optional[str] = Field(default=None, max_length=300, description="Optionally replace the one-line summary")


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
    content: dict[str, Any] = {"error": exc.kind, "detail": exc.detail}
    if isinstance(exc, store.Conflict):
        content["current"] = exc.current
    return JSONResponse(content, status_code=exc.status)


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


def _scrub(payload: Any) -> None:
    hits = scrubber.scan(payload)
    if hits:
        _log.warning("pazbrain_secret_rejected", extra={"hits": hits})
        raise HTTPException(
            422,
            {
                "error": "secret_detected",
                "detail": "Write rejected: content looks like a secret or financial identifier. Remove it and retry.",
                "matches": hits,
            },
        )


Reader = Annotated[str, Depends(require_read)]
Writer = Annotated[str, Depends(require_write)]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@app.get("/pages", summary="List pages, newest first")
def list_pages(
    _: Reader,
    category: Optional[Category] = None,
    tag: Optional[str] = Query(default=None, max_length=50),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    _ready()
    pages = store.list_pages(category, tag, limit)
    return {"count": len(pages), "pages": pages}


@app.get("/pages/{slug}", summary="Get a full page with frontmatter, backlinks and outgoing links")
def get_page(slug: Slug, _: Reader) -> dict[str, Any]:
    _ready()
    return store.get_page(slug)


@app.get("/search", summary="Full-text search (FTS5) with highlighted snippets")
def search(
    _: Reader,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    _ready()
    results = store.search(q, limit)
    return {"query": q, "count": len(results), "results": results}


@app.get("/history/{slug}", summary="git history of a page")
def history(slug: Slug, _: Reader, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    _ready()
    return {"slug": slug, "history": store.history(slug, limit)}


@app.get("/recent", summary="Recent changes from log.md, newest first")
def recent(_: Reader, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    _ready()
    return {"changes": store.recent(limit)}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


@app.put("/pages/{slug}", summary="Create a page, or fully replace one (needs base_updated)")
def put_page(slug: Slug, page: PageIn, _: Writer) -> dict[str, Any]:
    _ready()
    store.check_slug(slug)
    data = page.model_dump()
    _scrub({k: data[k] for k in ("title", "tags", "sources", "summary", "body")})
    return store.put_page(slug, **data)


@app.patch("/pages/{slug}", summary="Patch a page: append/replace a section, add tags or sources")
def patch_page(slug: Slug, patch: PatchIn, _: Writer) -> dict[str, Any]:
    _ready()
    store.check_slug(slug)
    data = patch.model_dump()
    _scrub({"ops": data["ops"], "summary": data["summary"]})
    return store.patch_page(slug, **data)


@app.delete("/pages/{slug}", summary="Soft delete: move the page to _archive/")
def archive_page(
    slug: Slug,
    _: Writer,
    updated_by: str = Query(default="grok", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$"),
) -> dict[str, Any]:
    _ready()
    return store.archive_page(slug, updated_by=updated_by)
