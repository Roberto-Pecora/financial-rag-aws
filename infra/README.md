# infra

Terraform for the AWS data plane: an S3 data lake, a single free-tier OpenSearch
node, and a least-privilege IAM policy. The compute plane (model training,
serving, LLM calls) runs off AWS, so nothing here holds a GPU or an always-on
endpoint.

## What it creates

| Resource | Free-tier posture |
|---|---|
| S3 bucket (`raw/ corpus/ artifacts/ eval/`) | 5 GB free; SSE-S3 (no KMS charge); noncurrent versions expire after 7 days |
| OpenSearch domain | one `t3.small.search` node, single-AZ, 10 GB gp3 — 750 h/month free for 12 months |
| Encryption at rest | AWS-managed key (no monthly key charge) |
| IAM policy `frag-app-least-privilege` | S3 + `es:ESHttp*` + Textract only; defined, not auto-attached |

Only **one** OpenSearch node stays free. `instance_count` and `instance_type`
carry warnings for that reason — do not scale them without intent.

## Lifecycle

```bash
make infra-plan     # review before anything is created
make infra-up       # terraform apply  (creates billable-but-free resources)
# ... run the demo: ingest, index, eval ...
make infra-down     # terraform destroy — leaves no OpenSearch hours running
```

After `infra-up`, wire the outputs into `.env`:

```bash
cd infra
terraform output -raw bucket_name          # -> S3_BUCKET
terraform output -raw opensearch_endpoint  # -> OPENSEARCH_ENDPOINT
```

## Guardrails

- Credentials come from the local `ninefin` profile; no keys live in this config.
- OpenSearch bills per hour while it exists — `make infra-down` when not in use.
- Set a small budget alert in the console (Billing → Budgets) as a backstop.
- Tighten access last: attach `frag-app-least-privilege` to the dev user and
  remove `AdministratorAccess` once the resources exist.
