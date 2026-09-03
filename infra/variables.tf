variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "eu-north-1"
}

variable "profile" {
  description = "Local AWS CLI profile to authenticate with."
  type        = string
  default     = "ninefin"
}

variable "bucket_name" {
  description = "S3 bucket for the data lake (corpus, artifacts, eval). Must be globally unique."
  type        = string
  default     = "frag-datalake-452575447847"
}

variable "domain_name" {
  description = "OpenSearch domain name (3-28 chars, lowercase)."
  type        = string
  default     = "frag-search"
}

variable "opensearch_instance_type" {
  description = "Free-tier-eligible single node. Do not scale this up without intent — only one node stays free."
  type        = string
  default     = "t3.small.search"
}

variable "opensearch_volume_gb" {
  description = "EBS gp3 size per node. Free tier covers 10 GB."
  type        = number
  default     = 10
}

variable "engine_version" {
  description = "OpenSearch engine version."
  type        = string
  default     = "OpenSearch_2.11"
}
