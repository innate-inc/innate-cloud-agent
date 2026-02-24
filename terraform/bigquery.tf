# ── BigQuery IAM ─────────────────────────────────────────────────────
# The agent creates its own dataset (agent_logs) at runtime via BigQueryLogger.
# Grant project-level access so it can create datasets and write data.

resource "google_project_iam_member" "bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.cloud_agent.email}"
}

resource "google_project_iam_member" "bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.cloud_agent.email}"
}
