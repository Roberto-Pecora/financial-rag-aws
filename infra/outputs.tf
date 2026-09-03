output "bucket_name" {
  description = "S3 data-lake bucket name (set as S3_BUCKET in .env)."
  value       = aws_s3_bucket.datalake.bucket
}

output "opensearch_endpoint" {
  description = "OpenSearch HTTPS endpoint (set as OPENSEARCH_ENDPOINT in .env)."
  value       = "https://${aws_opensearch_domain.search.endpoint}"
}

output "app_policy_arn" {
  description = "Least-privilege policy to attach to the dev user (then remove AdministratorAccess)."
  value       = aws_iam_policy.app.arn
}
