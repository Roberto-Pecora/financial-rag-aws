"""S3 data-lake I/O tests using a fake S3 client (no AWS calls)."""

from __future__ import annotations

import json

from frag.aws import s3_store
from frag.jobs import ingest_manifest


class _FakeS3:
    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.store[(Bucket, Key)] = Body
        return {}

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


def test_upload_and_download_roundtrip():
    client = _FakeS3()
    recs = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    n = s3_store.upload_jsonl(recs, "bkt", "corpus/c.jsonl", client=client)
    assert n == 2

    body = client.store[("bkt", "corpus/c.jsonl")].decode("utf-8")
    assert body.count("\n") == 2  # newline-delimited
    assert json.loads(body.splitlines()[0])["id"] == "1"

    back = s3_store.download_jsonl("bkt", "corpus/c.jsonl", client=client)
    assert back == recs


def test_s3_sink_wired_into_run_manifest(monkeypatch):
    client = _FakeS3()
    monkeypatch.setattr(
        ingest_manifest,
        "dispatch",
        lambda e: [{"id": "1", "text": "x", "metadata": {"ingest_path": "pdf_text"}}],
    )
    sink = s3_store.s3_jsonl_sink("bkt", "corpus/out.jsonl", client=client)
    stats = ingest_manifest.run_manifest([{}], sink=sink)

    assert stats["written"] == 1
    assert ("bkt", "corpus/out.jsonl") in client.store
