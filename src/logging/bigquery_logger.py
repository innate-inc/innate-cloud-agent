import os
from datetime import datetime
from typing import Dict, Any, List
import logging
import threading

from google.cloud import bigquery
from google.api_core import exceptions
from typing import Optional


class BigQueryLogger:
    """
    A unified logger for sending data to different Google BigQuery tables.
    This class is implemented as a singleton to ensure only one BigQuery client
    and one set of ensured tables exist per application instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.logger = logging.getLogger(__name__)
            self.project_id = os.getenv("BIGQUERY_PROJECT_ID")
            self.dataset_id = os.getenv("BIGQUERY_DATASET_ID")
            self._ensured_tables = set()

            self.schemas = {
                "token_metrics": [
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("model_name", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField(
                        "decision_making_time", "FLOAT", mode="NULLABLE"
                    ),
                    bigquery.SchemaField("input_tokens", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("output_tokens", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("total_tokens", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("tokens_per_second", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("total_processing_seconds", "FLOAT", mode="NULLABLE"),
                ],
                "directive_changes": [
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("directive_name", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("directive_text", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("client_id", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("user_token", "STRING", mode="NULLABLE"),
                ],
            }

            if not all([self.project_id, self.dataset_id]):
                self.logger.warning(
                    "BIGQUERY_PROJECT_ID or BIGQUERY_DATASET_ID are not set. "
                    "BigQuery logger will be disabled."
                )
                self.client = None
                self._initialized = True
                return

            try:
                self.client = bigquery.Client(project=self.project_id)
                self._ensure_dataset_exists()
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize BigQuery client: {e}. Disabling logger."
                )
                self.client = None

            self._initialized = True

    def _ensure_dataset_exists(self):
        if not self.client:
            return
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            self.client.create_dataset(dataset_ref, timeout=30)
            self.logger.info(f"Created BigQuery dataset: {self.dataset_id}")
        except exceptions.Conflict:
            self.logger.info(f"BigQuery dataset '{self.dataset_id}' already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create or get dataset: {e}")
            raise

    def _ensure_table_exists(self, table_id: str):
        if not self.client or table_id in self._ensured_tables:
            return

        schema = self.schemas.get(table_id)
        if not schema:
            self.logger.error(f"Schema for table '{table_id}' not found.")
            return

        full_table_id = f"{self.project_id}.{self.dataset_id}.{table_id}"
        table_ref = self.client.dataset(self.dataset_id).table(table_id)
        table = bigquery.Table(table_ref, schema=schema)

        try:
            self.client.create_table(table)
            self.logger.info(f"Created BigQuery table: {full_table_id}")
        except exceptions.Conflict:
            self.logger.info(f"BigQuery table '{full_table_id}' already exists.")
        except Exception as e:
            self.logger.error(f"Failed to create or get table: {e}")
            raise

        self._ensured_tables.add(table_id)

    def log(self, table_id: str, data: Dict[str, Any], logger: logging.Logger):
        if not self.client:
            return

        try:
            self._ensure_table_exists(table_id)

            full_table_id = f"{self.project_id}.{self.dataset_id}.{table_id}"
            errors = self.client.insert_rows_json(full_table_id, [data])
            logger.debug(f"Logged to BigQuery table {table_id}: {data}")
            if errors:
                logger.error(
                    f"Encountered errors while inserting rows into {table_id}: {errors}"
                )
        except Exception as e:
            logger.error(f"Failed to log to BigQuery table {table_id}: {e}")
