# --- Data lake bucket -------------------------------------------------------
# Holds raw/ source docs, corpus/ processed JSONL chunk records, artifacts/
# (Colab-trained models), and eval/ (MLflow artifacts, ablation CSVs).

resource "aws_s3_bucket" "datalake" {
  bucket = var.bucket_name

  # Portfolio data lake — safe to delete on teardown so `terraform destroy`
  # does not strand the bucket. Flip to false before storing anything you value.
  force_destroy = true
}

# Server-side encryption with the free S3-managed key (SSE-S3). A customer
# managed KMS key would add a monthly charge; the plan avoids that deliberately.
resource "aws_s3_bucket_server_side_encryption_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket                  = aws_s3_bucket.datalake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Expire noncurrent versions so re-ingest churn cannot silently grow past the
# 5 GB free-tier storage allowance.
resource "aws_s3_bucket_lifecycle_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {} # apply to every object in the bucket
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}
