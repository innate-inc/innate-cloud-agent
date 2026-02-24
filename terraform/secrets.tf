# ── Secret Manager secrets ──────────────────────────────────────────

# Agent-specific secrets
locals {
  secrets = toset([
    "CLOUD_AGENT_GEMINI_API_KEY",
  ])
}

resource "google_secret_manager_secret" "cloud_agent" {
  for_each = local.secrets

  secret_id = each.value
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "cloud_agent" {
  for_each = local.secrets

  secret_id = google_secret_manager_secret.cloud_agent[each.key].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_agent.email}"
}

# ── Shared secrets (created by other modules) ───────────────────────
# OPENAI_API_KEY is created by the service-proxy module; we just grant access.

resource "google_secret_manager_secret_iam_member" "openai_shared" {
  secret_id = "OPENAI_API_KEY"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_agent.email}"
}
