"""S3 data-lake I/O for corpus JSONL and artifacts.

The manifest job (`frag.jobs.ingest_manifest`) writes chunk records through a
sink callable; `s3_jsonl_sink` is the S3 implementation, so the same job targets
a local file in tests and S3 in production without changing its logic. The S3
client is injectable so this module is unit-tested with a fake and no AWS calls.

Layout under the bucket:
    raw/        source documents
    corpus/     processed JSONL chunk records
    artifacts/  Colab-trained model weights
    eval/       MLflow artifacts and ablation CSVs
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from typing import Any

Record = dict[str, Any]


def _client(client: Any | None = None):
    if client is not None:
        return client
    import boto3

    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE"),
        region_name=os.getenv("AWS_REGION", "eu-north-1"),
    )
    return session.client("s3")


def records_to_jsonl(records: Iterable[Record]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def upload_jsonl(
    records: Iterable[Record], bucket: str, key: str, client: Any | None = None
) -> int:
    """Write records as a single JSONL object to s3://bucket/key. Returns count."""
    records = list(records)
    body = records_to_jsonl(records).encode("utf-8")
    _client(client).put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/x-ndjson"
    )
    return len(records)


def download_jsonl(bucket: str, key: str, client: Any | None = None) -> list[Record]:
    """Read a JSONL object back into a list of records."""
    obj = _client(client).get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def s3_jsonl_sink(
    bucket: str, key: str, client: Any | None = None
) -> Callable[[Iterable[Record]], int]:
    """A `run_manifest` sink that uploads the corpus JSONL to S3."""

    def _sink(records: Iterable[Record]) -> int:
        return upload_jsonl(records, bucket, key, client=client)

    return _sink
