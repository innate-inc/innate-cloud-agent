import os
from datetime import datetime
from typing import Optional
import logging

from google.cloud import bigquery
from google.api_core import exceptions


class BigQueryDirectiveLogger:
    """
    Logs directive changes to a Google BigQuery table.
    """

    def __init__(self):
        """
        Initialize the BigQuery directive logger by reading configuration from
        environment variables.
        """
        project_id = os.getenv("BIGQUERY_PROJECT_ID")
        dataset_id = os.getenv("BIGQUERY_DATASET_ID")
        table_id = "directive_changes"  # Hardcoded table name

        self.logger = logging.getLogger(__name__)

        if not all([project_id, dataset_id]):
            self.logger.warning(
                "One or more BigQuery environment variables (BIGQUERY_PROJECT_ID, "
                "BIGQUERY_DATASET_ID) are not set. "
                "BigQuery directive logger will be disabled."
            )
            self.client = None
            return

        try:
            self.client = bigquery.Client(project=project_id)
            self.table_id = f"{project_id}.{dataset_id}.{table_id}"
            self._ensure_dataset_and_table_exist(dataset_id, table_id)
        except Exception as e:
            self.logger.error(
                f"Failed to initialize BigQuery client: {e}. Disabling logger."
            )
            self.client = None

    def _ensure_dataset_and_table_exist(self, dataset_id: str, table_id: str):
        """
        Ensures that the specified BigQuery dataset and table exist.
        If they don't, it creates them.
        """
        if not self.client:
            return

        dataset_ref = self.client.dataset(dataset_id)

        try:
            self.client.create_dataset(dataset_ref, timeout=30)
            self.logger.info(f"Created BigQuery dataset: {dataset_id}")
        except exceptions.Conflict:
            self.logger.info(f"BigQuery dataset '{dataset_id}' already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create or get dataset: {e}")
            raise

        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("directive_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("client_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("user_token", "STRING", mode="NULLABLE"),
        ]

        table_ref = dataset_ref.table(table_id)
        table = bigquery.Table(table_ref, schema=schema)

        try:
            self.client.create_table(table)
            self.logger.info(f"Created BigQuery table: {self.table_id}")
        except exceptions.Conflict:
            self.logger.info(f"BigQuery table '{self.table_id}' already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create or get table: {e}")
            raise

    def log_directive_change(
        self,
        directive_name: str,
        client_id: str,
        user_token: Optional[str] = None,
    ):
        """
        Log a directive change to BigQuery by streaming a row.

        Args:
            directive_name: Name of the new directive.
            client_id: The ID of the client.
            user_token: The user's token.
        """
        if not self.client:
            return

        row_to_insert = {
            "timestamp": datetime.utcnow().isoformat(),
            "directive_name": directive_name,
            "client_id": client_id,
            "user_token": user_token,
        }

        errors = self.client.insert_rows_json(self.table_id, [row_to_insert])
        if errors:
            self.logger.error(f"Encountered errors while inserting rows: {errors}")
