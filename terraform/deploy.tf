# ── Cloud Run service ────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "main" {
  name     = "innate-cloud-agent"
  location = var.region

  lifecycle { # ignore changes so we can update the image through github actions
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
      template[0].revision,
    ]
  }
  template {
    service_account = google_service_account.cloud_agent.email

    scaling {
      min_instance_count = 0
      max_instance_count = 100
    }

    timeout                          = "3600s"
    max_instance_request_concurrency = 80

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "8Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "AUTH_ISSUER_URL"
        value = var.auth_issuer_url
      }
      env {
        name  = "ROBOT_TYPE"
        value = "maurice_oak_d"
      }
      env {
        name  = "BIGQUERY_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "BIGQUERY_DATASET_ID"
        value = "agent_logs"
      }
      env {
        name  = "BAML_LOG"
        value = "off"
      }
      env {
        name  = "BAML_LOG_LEVEL"
        value = "off"
      }
      env {
        name  = "BAML_VERBOSE"
        value = "off"
      }
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "OPENAI_API_KEY"
            version = "latest"
          }
        }
      }
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.cloud_agent["CLOUD_AGENT_GEMINI_API_KEY"].secret_id
            version = "latest"
          }
        }
      }
    }
  }
  deletion_protection = false
}

# ── Public Access ────────────────────────────────────────────────────

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.main.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Domain Mapping ───────────────────────────────────────────────────

# resource "google_cloud_run_domain_mapping" "agent" {
#   location = var.region
#   name     = "agent-v1.innate.bot"
#
#   metadata {
#     namespace = var.project_id
#   }
#
#   spec {
#     route_name       = google_cloud_run_v2_service.main.name
#     certificate_mode = "AUTOMATIC"
#   }
# }
