terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-04-732745575055-terraform-state"
    prefix = "prod"
  }
}
