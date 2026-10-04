"""Internal service-to-service auth helpers (cross-repo contract C1).

Callee side: Cloud Run IAM (--no-allow-unauthenticated) does the real check;
`verify_internal_token` adds the optional shared-secret header check.
Caller side: `internal_auth_headers` attaches a Google ID token whose audience
is the callee's base URL, plus X-Internal-Token when configured.
"""
import hmac
import logging
import os
from urllib.parse import urlparse

from fastapi import HTTPException

logger = logging.getLogger("capsai.internal_auth")


def verify_internal_token(x_internal_token: str | None) -> None:
    expected = os.getenv("INTERNAL_API_TOKEN", "")
    if not expected:
        return
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def audience_for(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def internal_auth_headers(url: str) -> dict:
    headers = {}
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        token = google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience_for(url)
        )
        headers["Authorization"] = f"Bearer {token}"
    except Exception as e:  # local dev without metadata server / SA key
        logger.warning("Could not mint ID token for %s: %s", audience_for(url), e)
    internal = os.getenv("INTERNAL_API_TOKEN", "")
    if internal:
        headers["X-Internal-Token"] = internal
    return headers
