import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from google.cloud import bigquery
from google.api_core import exceptions


class BigQueryTokenLogger:
    """
    Logs token usage metrics to a Google BigQuery table.
    """

    def __init__(self):
        """
        Initialize the BigQuery token logger by reading configuration from
        environment variables.
        """
        project_id = os.getenv("BIGQUERY_PROJECT_ID")
        dataset_id = os.getenv("BIGQUERY_DATASET_ID")
        table_id = "token_metrics"  # Hardcoded table name

        self.logger = logging.getLogger(__name__)

        if not all([project_id, dataset_id]):
            self.logger.warning(
                "One or more BigQuery environment variables (BIGQUERY_PROJECT_ID, "
                "BIGQUERY_DATASET_ID) are not set. "
                "BigQuery logger will be disabled."
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
            bigquery.SchemaField("model_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("processing_time_seconds", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("input_tokens", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("output_tokens", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("total_tokens", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("tokens_per_second", "FLOAT", mode="NULLABLE"),
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

    def log_usage(
        self,
        model_name: str,
        processing_time_seconds: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """
        Log token usage metrics to BigQuery by streaming a row.

        Args:
            model_name: Name of the model used (e.g., "gemini-flash").
            processing_time_seconds: Time taken to process the request.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
        """
        if not self.client:
            return

        total_tokens = None
        tokens_per_second = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
            if processing_time_seconds > 0:
                tokens_per_second = total_tokens / processing_time_seconds

        row_to_insert = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "processing_time_seconds": processing_time_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
        }

        errors = self.client.insert_rows_json(self.table_id, [row_to_insert])
        if errors:
            self.logger.error(f"Encountered errors while inserting rows: {errors}")
