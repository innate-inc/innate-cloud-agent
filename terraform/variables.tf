variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "auth_issuer_url" {
  description = "JWT issuer URL for Innate Auth"
  type        = string
}
