# --- OpenSearch domain ------------------------------------------------------
# A single free-tier node running dense k-NN and BM25 in one store. The app
# signs requests with SigV4 as the caller identity, so no username/password and
# no public credentials exist.

data "aws_caller_identity" "current" {}

resource "aws_opensearch_domain" "search" {
  domain_name    = var.domain_name
  engine_version = var.engine_version

  cluster_config {
    instance_type          = var.opensearch_instance_type
    instance_count         = 1     # one node only — more than one leaves free tier
    zone_awareness_enabled = false # single-AZ; multi-AZ needs >= 2 nodes
    # No dedicated master: a single-node domain does not need (or allow) one.
  }

  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = var.opensearch_volume_gb
  }

  # Encryption at rest with the AWS-managed key (no monthly KMS charge, unlike a
  # customer managed key). Node-to-node encryption and HTTPS are enforced.
  encrypt_at_rest {
    enabled = true
  }
  node_to_node_encryption {
    enabled = true
  }
  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  # Access is restricted to this account's caller identity; requests are SigV4
  # signed. This is an open-policy-with-IAM-principal domain (public endpoint,
  # but every call must be signed by the allowed principal).
  access_policies = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { AWS = data.aws_caller_identity.current.arn }
        Action    = "es:ESHttp*"
        Resource  = "arn:aws:es:${var.region}:${data.aws_caller_identity.current.account_id}:domain/${var.domain_name}/*"
      }
    ]
  })
}
