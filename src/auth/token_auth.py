"""Token authentication against BigQuery user_management table."""

import os
import logging
import time
from typing import Optional, Dict
from dataclasses import dataclass

from google.cloud import bigquery

logger = logging.getLogger(__name__)


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
        if not self.client:
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
