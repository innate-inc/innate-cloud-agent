# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

"""Token authentication via innate-auth service.

Forwards the robot's innate_service_key to innate-auth /v1/auth,
receives a JWT, and decodes it (without cryptographic verification)
to extract user identity and check the cloud_agent permission.
"""

import os
import json
import base64
import logging
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import requests
from packaging.version import Version

from src.constants_robots import MIN_CLIENT_VERSION

logger = logging.getLogger(__name__)

# A robot rejected for an out-of-date OS reconnects roughly once per second.
# Speaking the version warning on every attempt floods the robot's
# text-to-speech queue, so it is spoken at most once per this interval per
# robot token. The connection is still rejected on every attempt regardless.
VERSION_WARNING_INTERVAL_S = 3600.0  # once per hour
_last_version_warning: Dict[str, float] = {}


def should_speak_version_warning(token: str) -> bool:
    """Return True at most once per VERSION_WARNING_INTERVAL_S per token.

    This only rate-limits the *spoken* warning so reconnect storms don't spam
    the robot's TTS; it does not affect whether the connection is accepted.
    """
    now = time.monotonic()
    last = _last_version_warning.get(token)
    if last is not None and now - last < VERSION_WARNING_INTERVAL_S:
        return False
    _last_version_warning[token] = now
    # Opportunistically drop stale entries so the dict can't grow unbounded.
    if len(_last_version_warning) > 1000:
        for key, ts in list(_last_version_warning.items()):
            if now - ts >= VERSION_WARNING_INTERVAL_S:
                del _last_version_warning[key]
    return True


def compare_versions(
    client_version: str, min_version: str = MIN_CLIENT_VERSION
) -> Tuple[bool, str]:
    """
    Compare client version against minimum required version.

    Args:
        client_version: The client's semver version string (e.g., "1.2.3" or "1.2.3-dev")
        min_version: The minimum required version (defaults to MIN_CLIENT_VERSION)

    Returns:
        Tuple of (is_valid, message) where is_valid is True if client_version >= min_version
        Dev versions (containing "-dev") are always allowed.
    """
    try:
        client_ver = Version(client_version)
        client_version = client_version.replace(".", " point ")
        if "-dev" in client_version.lower():
            return True, f"Robot OS version {client_version} is a dev version (allowed)"

        min_ver = Version(min_version)

        if client_ver < min_ver:
            return False, (
                f"Robot OS version {client_version} is less than minimum required "
                f"version {min_version}. Please update your robot or modify code "
                "to switch to a dev version."
            )
        elif client_ver > min_ver:
            return True, (
                f"Robot OS version {client_version} is greater than minimum "
                f"version {min_version}"
            )
        else:
            return True, (
                f"Robot OS version {client_version} matches minimum "
                f"version {min_version}"
            )
    except Exception as e:
        return False, f"Invalid version format: {e}"


@dataclass
class AuthContext:
    """Authentication context for a validated token."""

    robot_special_token: str
    user_id: str
    innate_service_key: str


def _decode_jwt_payload_unsafe(token: str) -> Optional[Dict]:
    """Decode a JWT payload without verifying the signature.

    This is intentionally insecure — we trust innate-auth over the
    internal network and only need to read claims, not verify them.
    """
    parts = token.split(".")
    if len(parts) != 3:
        logger.error("JWT does not have 3 parts")
        return None
    try:
        # JWT base64url: add padding
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes)
    except Exception as e:
        logger.error(f"Failed to decode JWT payload: {e}")
        return None


class TokenAuthenticator:
    """Authenticates innate_service_key by calling innate-auth /v1/auth.

    On success, decodes the returned JWT (without crypto verification)
    to extract user_id, robot_special_token, and permissions.
    """

    _instance: Optional["TokenAuthenticator"] = None

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.auth_issuer_url: str = os.getenv("AUTH_ISSUER_URL", "").rstrip("/")

        # Check if authentication should be skipped (local dev)
        self.skip_auth: bool = os.getenv("SKIP_AUTH", "").lower() in (
            "true",
            "1",
            "yes",
        )
        if self.skip_auth:
            robot_token = os.getenv("DEFAULT_ROBOT_TOKEN")
            user_id = os.getenv("DEFAULT_USER_ID")
            service_key = os.getenv("DEFAULT_SERVICE_KEY")

            if not robot_token or not user_id or not service_key:
                raise ValueError(
                    "SKIP_AUTH is enabled but required environment variables are "
                    "not set. Please set DEFAULT_ROBOT_TOKEN, DEFAULT_USER_ID, "
                    "and DEFAULT_SERVICE_KEY"
                )

            self.default_auth_context = AuthContext(
                robot_special_token=robot_token,
                user_id=user_id,
                innate_service_key=service_key,
            )
            logger.info("SKIP_AUTH enabled — authentication will be bypassed")
        elif not self.auth_issuer_url:
            raise ValueError(
                "AUTH_ISSUER_URL must be set (e.g. http://auth:8080) "
                "unless SKIP_AUTH is enabled."
            )
        else:
            logger.info(f"Auth configured against {self.auth_issuer_url}")

        # Simple cache: {service_key: (AuthContext, expiry_timestamp)}
        self._cache: Dict[str, Tuple[AuthContext, float]] = {}
        self._cache_ttl: float = 300.0  # 5 minutes

        self._initialized = True

    def validate_service_key(self, service_key: str) -> Optional[AuthContext]:
        """Validate an innate_service_key by exchanging it for a JWT.

        Returns AuthContext on success, None on failure.
        """
        if self.skip_auth:
            logger.debug("SKIP_AUTH — returning default auth context")
            return self.default_auth_context

        if not service_key:
            logger.warning("Empty service key provided")
            return None

        # Check cache
        now = time.time()
        cached = self._cache.get(service_key)
        if cached is not None:
            ctx, expires_at = cached
            if now < expires_at:
                logger.debug(f"Cache hit for user {ctx.user_id}")
                return ctx
            del self._cache[service_key]

        # Call innate-auth /v1/auth
        try:
            resp = requests.post(
                f"{self.auth_issuer_url}/v1/auth",
                headers={
                    "Authorization": f"Bearer {service_key}",
                    "User-Agent": "innate-cloud-agent",
                },
                timeout=10,
            )
        except requests.RequestException as e:
            logger.error(f"Failed to reach innate-auth: {e}")
            return None

        if resp.status_code != 200:
            logger.warning(
                f"innate-auth returned {resp.status_code}: "
                f"{resp.text[:200] if resp.text else '(empty)'}"
            )
            return None

        body = resp.json()
        jwt_token: str = body.get("token", "")
        if not jwt_token:
            logger.error("innate-auth response missing 'token' field")
            return None

        # Decode JWT payload (no signature verification)
        claims = _decode_jwt_payload_unsafe(jwt_token)
        if claims is None:
            return None

        # Check cloud_agent permission
        perms = claims.get("perms", {})
        if not perms.get("cloud_agent", False):
            logger.warning(
                f"User {claims.get('sub', '?')} lacks cloud_agent permission"
            )
            return None

        ctx = AuthContext(
            robot_special_token=claims.get("rst", ""),
            user_id=claims.get("sub", ""),
            innate_service_key=service_key,
        )

        # Cache
        self._cache[service_key] = (ctx, now + self._cache_ttl)
        logger.info(f"Authenticated user {ctx.user_id} via innate-auth")
        return ctx

    def clear_cache(self) -> None:
        """Clear the token cache."""
        self._cache.clear()
        logger.info("Token cache cleared")


# Global authenticator instance
_authenticator: Optional[TokenAuthenticator] = None


def get_authenticator() -> TokenAuthenticator:
    """Get the global TokenAuthenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = TokenAuthenticator()
    return _authenticator
