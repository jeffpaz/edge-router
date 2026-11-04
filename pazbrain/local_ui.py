"""GET /brain/local: a LAN-only page for browsing all of PazBrain, local-only pages included.

The page is static HTML (pazbrain/static/brain_local.html, no third-party scripts). It reads
/brain/api with the PazBrain read token, which the browser asks for once and keeps in localStorage.
/brain/api already returns local-only pages only to LAN requests.

LAN-only: requests that came through the Cloudflare Tunnel carry cf-ray / cf-connecting-ip
(cloudflared always adds them, and a client can't remove them), and get a plain 404 here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["pazbrain"])
_PAGE = Path(__file__).with_name("static") / "brain_local.html"
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # Only this page's own inline code, and fetches to this origin.
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'",
}


@router.get("/brain/local", include_in_schema=False)
async def brain_local(request: Request) -> HTMLResponse:
    if any(h in request.headers for h in ("cf-ray", "cf-connecting-ip")):
        raise HTTPException(404)
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"), headers=_HEADERS)
