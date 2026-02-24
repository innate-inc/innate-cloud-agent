# ──────────────────────────────────────────────────────────────────────
# Cloud Agent module
# ──────────────────────────────────────────────────────────────────────

resource "google_service_account" "cloud_agent" {
  account_id   = "cloud-agent"
  display_name = "Cloud Agent (WebSocket brain server)"
}
