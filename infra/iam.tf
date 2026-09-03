# --- Least-privilege policy -------------------------------------------------
# Defines exactly the permissions the application needs: the data-lake bucket,
# HTTP access to the search domain, and Textract's async text-detection pair.
# Defined here as a managed policy; attach it to the dev user (and drop
# AdministratorAccess) once the resources exist. Attachment is intentionally
# left out so this config can be planned without mutating the user's policies.

resource "aws_iam_policy" "app" {
  name        = "frag-app-least-privilege"
  description = "S3 data lake + OpenSearch HTTP + Textract for the financial-rag-aws app."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "DataLakeBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.datalake.arn
      },
      {
        Sid      = "DataLakeObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.datalake.arn}/*"
      },
      {
        Sid      = "SearchHttp"
        Effect   = "Allow"
        Action   = "es:ESHttp*"
        Resource = "${aws_opensearch_domain.search.arn}/*"
      },
      {
        Sid    = "TextractAsync"
        Effect = "Allow"
        Action = [
          "textract:StartDocumentTextDetection",
          "textract:GetDocumentTextDetection"
        ]
        Resource = "*" # Textract actions are not resource-scoped
      }
    ]
  })
}
