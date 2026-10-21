"""PazBrain auth: bearer tokens from /etc/pazbrain.env, Firebase sign-in for the web UI.

  writes  Authorization: Bearer $PAZBRAIN_WRITE_TOKEN
  reads   $PAZBRAIN_READ_TOKEN, the write token (the Grok bot reads before it writes),
          or a Firebase ID token for a verified email in PAZBRAIN_ALLOWED_EMAILS

Firebase tokens are checked by the edge-router's own auth.verify_token (Google's public
signing keys, RS256, aud/iss = the pazlabs project) — no firebase-admin service account.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from collections import deque

from fastapi import Header, HTTPException

import auth as firebase_auth

_log = logging.getLogger("edge_router.pazbrain")

_WRITE_TOKEN = os.getenv("PAZBRAIN_WRITE_TOKEN", "").strip()
_READ_TOKEN = os.getenv("PAZBRAIN_READ_TOKEN", "").strip()
_ALLOWED_EMAILS = {
    e.strip().lower() for e in os.getenv("PAZBRAIN_ALLOWED_EMAILS", "").split(",") if e.strip()
}
_WRITE_RATE_PER_MIN = int(os.getenv("PAZBRAIN_WRITE_RATE_PER_MIN", "60"))

_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def _matches(token: str, expected: str) -> bool:
    # An unset token must never match (compare_digest("", "") is True).
    return bool(expected) and hmac.compare_digest(token.encode(), expected.encode())


def _bearer(authorization: str) -> str:
    token = firebase_auth.bearer_token(authorization or "")
    if not token:
        raise HTTPException(401, {"error": "unauthorized", "detail": "Missing bearer token."}, _UNAUTHORIZED)
    return token


async def require_read(authorization: str = Header(default="")) -> str:
    """Returns the caller identity: 'token:read', 'token:write' or 'firebase:<email>'."""
    token = _bearer(authorization)
    if _matches(token, _READ_TOKEN):
        return "token:read"
    if _matches(token, _WRITE_TOKEN):
        return "token:write"
    if token.count(".") == 2:
        try:
            claims = await firebase_auth.verify_token(token)
        except firebase_auth.AuthError as exc:
            raise HTTPException(401, {"error": "unauthorized", "detail": exc.detail}, _UNAUTHORIZED)
        email = str(claims.get("email") or "").lower()
        if email and claims.get("email_verified") is True and email in _ALLOWED_EMAILS:
            return f"firebase:{email}"
        _log.warning("pazbrain_forbidden_email", extra={"email": email})
        raise HTTPException(403, {"error": "forbidden", "detail": "This account is not allowed to read PazBrain."})
    raise HTTPException(401, {"error": "unauthorized", "detail": "Invalid token."}, _UNAUTHORIZED)


# Sliding one-minute window per write token, keyed by a hash so tokens never sit in memory twice.
_hits: dict[str, deque[float]] = {}
_hits_lock = threading.Lock()


def _rate_limit(token: str) -> None:
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.monotonic()
    with _hits_lock:
        window = _hits.setdefault(key, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= _WRITE_RATE_PER_MIN:
            retry = max(1, int(60 - (now - window[0])) + 1)
            raise HTTPException(
                429,
                {"error": "rate_limited", "detail": f"Max {_WRITE_RATE_PER_MIN} writes per minute."},
                {"Retry-After": str(retry)},
            )
        window.append(now)


async def require_write(authorization: str = Header(default="")) -> str:
    token = _bearer(authorization)
    if not _matches(token, _WRITE_TOKEN):
        raise HTTPException(
            401, {"error": "unauthorized", "detail": "Writes require the PazBrain write token."}, _UNAUTHORIZED
        )
    _rate_limit(token)
    return "token:write"
