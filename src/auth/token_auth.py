"""Token authentication against BigQuery user_management table."""

import os
import logging
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from google.cloud import bigquery
from packaging.version import Version

from src.constants_robots import MIN_CLIENT_VERSION

logger = logging.getLogger(__name__)


def compare_versions(client_version: str, min_version: str = MIN_CLIENT_VERSION) -> Tuple[bool, str]:
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
        # Allow any dev version without version checking
        client_ver = Version(client_version)
        client_version = client_version.replace(".", " point ")
        if "-dev" in client_version.lower():
            return True, f"Robot OS version {client_version} is a dev version (allowed)"
        
        min_ver = Version(min_version)
        
        if client_ver < min_ver:
            return False, f"Robot OS version {client_version} is less than minimum required version {min_version}. Please update your robot or modify code to switch to a dev version."
        elif client_ver > min_ver:
            return True, f"Robot OS version {client_version} is greater than minimum version {min_version}"
        else:
            return True, f"Robot OS version {client_version} matches minimum version {min_version}"
    except Exception as e:
        return False, f"Invalid version format: {e}"


@dataclass
class AuthContext:
    """Authentication context for a validated token."""

    robot_special_token: str
    user_id: str
    innate_service_key: str


class TokenAuthenticator:
    """
    Authenticates tokens by validating against BigQuery user_management table.
    Uses caching to reduce BigQuery queries.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.project_id = os.getenv("BIGQUERY_PROJECT_ID")
        self.dataset_id = os.getenv("BIGQUERY_AUTH_DATASET_ID", "innate_analytics")
        self.table_name = "user_management"

        # Token cache: {token: (auth_context, timestamp)}
        self._token_cache: Dict[str, tuple] = {}
        self._cache_ttl = 300  # 5 minutes

        # Check if authentication should be skipped
        self.skip_auth = os.getenv("SKIP_AUTH", "").lower() in ("true", "1", "yes")
        if self.skip_auth:
            robot_token = os.getenv("DEFAULT_ROBOT_TOKEN")
            user_id = os.getenv("DEFAULT_USER_ID")
            service_key = os.getenv("DEFAULT_SERVICE_KEY")
            
            if not robot_token or not user_id or not service_key:
                raise ValueError(
                    "SKIP_AUTH is enabled but required environment variables are not set. "
                    "Please set DEFAULT_ROBOT_TOKEN, DEFAULT_USER_ID, and DEFAULT_SERVICE_KEY"
                )
            
            self.default_auth_context = AuthContext(
                robot_special_token=robot_token,
                user_id=user_id,
                innate_service_key=service_key
            )
            logger.info("SKIP_AUTH enabled - authentication will be bypassed")

        if not self.project_id:
            logger.warning(
                "BIGQUERY_PROJECT_ID is not set. Token authentication will be disabled."
            )
            self.client = None
            self._initialized = True
            return

        try:
            self.client = bigquery.Client(project=self.project_id)
            self.full_table_id = (
                f"{self.project_id}.{self.dataset_id}.{self.table_name}"
            )
            logger.info(
                f"TokenAuthenticator initialized with table: {self.full_table_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client for auth: {e}")
            self.client = None

        self._initialized = True

    def validate_service_key(self, token: str) -> Optional[AuthContext]:
        """
        Validate service key against BigQuery user_management table.

        Args:
            token: The service key to validate (robot_special_token)

        Returns:
            AuthContext if valid, None otherwise
        """
        # Check if authentication should be skipped
        if self.skip_auth:
            logger.debug("SKIP_AUTH enabled - returning default auth context")
            return self.default_auth_context
        
        # return True
        if not self.client:
            # If BigQuery is not configured, allow any token for local development
            if os.getenv("SKIP_AUTH", "false").lower() == "true":
                logger.warning("SKIP_AUTH enabled - bypassing token validation")
                return AuthContext(
                    robot_special_token=token,
                    user_id="local_dev_user",
                    innate_service_key=token,
                )
            logger.warning("BigQuery client not initialized - cannot validate token")
            return None

        if not token:
            logger.warning("Empty token provided")
            return None

        # Check cache first
        current_time = time.time()
        if token in self._token_cache:
            auth_context, cached_time = self._token_cache[token]
            if current_time - cached_time < self._cache_ttl:
                logger.debug(f"Token cache hit for: {token[:10]}...")
                return auth_context
            else:
                # Cache expired
                del self._token_cache[token]

        # Query BigQuery
        try:
            auth_context = self._query_token(token)
            if auth_context:
                self._token_cache[token] = (auth_context, current_time)
                logger.info(f"Token validated for user: {auth_context.user_id}")
            return auth_context
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None

    def _query_token(self, key: str) -> Optional[AuthContext]:
        """
        Query BigQuery to validate key.

        Args:
            key: The service key to validate (and get robot_special_token)

        Returns:
            AuthContext if found, None otherwise
        """
        query = f"""
        SELECT 
            robot_special_token,
            user_id,
            innate_service_key
        FROM `{self.full_table_id}`
        WHERE innate_service_key = @key
        LIMIT 1
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("key", "STRING", key)]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                logger.warning(f"Token not found: {token[:10]}...")
                return None

            row = results[0]
            return AuthContext(
                robot_special_token=row.robot_special_token,
                user_id=row.user_id,
                innate_service_key=row.innate_service_key,
            )
        except Exception as e:
            logger.error(f"BigQuery query error: {e}")
            return None

    def clear_cache(self):
        """Clear the token cache."""
        self._token_cache.clear()
        logger.info("Token cache cleared")


# Global authenticator instance
_authenticator: Optional[TokenAuthenticator] = None


def get_authenticator() -> TokenAuthenticator:
    """Get the global TokenAuthenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = TokenAuthenticator()
    return _authenticator
