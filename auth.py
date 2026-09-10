"""Firebase ID token verification.

AskJetson (iOS) and, later, the pazlabs.io web app send a Firebase ID token as
`Authorization: Bearer <jwt>` with every request. This module verifies that JWT
against Google's public Secure Token signing keys.

A Firebase ID token is an RS256 JWT with:
  - iss  = https://securetoken.google.com/<project-id>
  - aud  = <project-id>
  - sub / user_id = the Firebase UID (non-empty)
  - exp  in the future, iat in the past
  - signed by one of the rotating certs published at Google's x509 endpoint

Nothing here reaches out to Firebase Admin / a service account — only the public
key set, cached and refreshed per its Cache-Control max-age.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.x509 import load_pem_x509_certificate

import config

_log = logging.getLogger("edge_router.auth")

# Public x509 certs for Firebase Secure Tokens. Response is {kid: pem-cert}, with
# a Cache-Control: max-age telling us how long the set is valid.
_CERTS_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/"
    "securetoken@system.gserviceaccount.com"
)
_FALLBACK_MAX_AGE = 3600.0


class AuthError(Exception):
    """Token missing, malformed, expired, or failing verification."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Signing-key cache
# ---------------------------------------------------------------------------

_keys: dict[str, RSAPublicKey] = {}
_keys_expiry: float = 0.0
_keys_lock = asyncio.Lock()


async def _public_key_for(kid: str) -> RSAPublicKey:
    global _keys_expiry

    now = time.time()
    if kid in _keys and now < _keys_expiry:
        return _keys[kid]

    async with _keys_lock:
        # Another coroutine may have refreshed while we waited.
        if kid in _keys and time.time() < _keys_expiry:
            return _keys[kid]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_CERTS_URL)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Serve a stale key if we still have one; the signature check is what
            # actually protects us, and Google keeps old certs valid past rotation.
            if kid in _keys:
                _log.warning("jwks_refresh_failed_using_stale", extra={"error": str(exc)})
                return _keys[kid]
            raise AuthError(f"Could not fetch token signing keys: {exc}") from exc

        certs: dict[str, str] = resp.json()
        _keys.clear()
        for k, pem in certs.items():
            try:
                _keys[k] = load_pem_x509_certificate(pem.encode()).public_key()  # type: ignore[assignment]
            except ValueError:
                continue

        _keys_expiry = time.time() + _parse_max_age(resp.headers.get("cache-control", ""))
        _log.info("jwks_refreshed", extra={"kids": list(_keys), "ttl_s": _keys_expiry - time.time()})

    key = _keys.get(kid)
    if key is None:
        raise AuthError("Token was signed with an unrecognised key.")
    return key


def _parse_max_age(cache_control: str) -> float:
    for part in cache_control.split(","):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                return max(float(part.split("=", 1)[1]), 60.0)
            except ValueError:
                break
    return _FALLBACK_MAX_AGE


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


async def verify_token(token: str) -> dict[str, Any]:
    """Return the decoded claims for a valid Firebase ID token, else raise AuthError."""
    project_id = config.FIREBASE_PROJECT_ID
    if not project_id:
        raise AuthError("Server auth is misconfigured (FIREBASE_PROJECT_ID unset).")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Malformed token: {exc}") from exc

    if header.get("alg") != "RS256":
        raise AuthError("Token must be signed with RS256.")
    kid = header.get("kid")
    if not kid:
        raise AuthError("Token header has no key id.")

    public_key = await _public_key_for(kid)

    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}") from exc

    if not claims.get("sub"):
        raise AuthError("Token subject is empty.")
    if claims.get("auth_time", 0) > time.time() + 60:
        raise AuthError("Token auth_time is in the future.")

    return claims


def bearer_token(authorization_header: str) -> str:
    """Pull the raw token out of an `Authorization: Bearer <token>` header value."""
    prefix = authorization_header[:7].lower()
    return authorization_header[7:].strip() if prefix == "bearer " else ""
