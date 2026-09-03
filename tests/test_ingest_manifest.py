"""Manifest ingestion: routing, idempotent dedupe, per-path stats, error isolation."""

from __future__ import annotations

import json

from frag.jobs import ingest_manifest


def _rec(rid: str, path: str = "pdf_text") -> dict:
    return {"id": rid, "text": "x", "metadata": {"ingest_path": path}}


def test_dispatch_routes_textract_supplied_text():
    entry = {
        "path": "s3://b/scan.pdf",
        "ingest_path": "textract",
        "text": "Total revenue: 5,678",
        "metadata": {"ticker": "TSLA"},
    }
    records = ingest_manifest.dispatch(entry)
    assert records
    assert all(r["metadata"]["ingest_path"] == "textract" for r in records)
    assert any("5,678" in r["text"] for r in records)


def test_dispatch_routes_sec_html():
    entry = {"url": "http://x/f.htm", "ingest_path": "sec_html", "text": "Revenue rose to 100."}
    records = ingest_manifest.dispatch(entry)
    assert records and records[0]["metadata"]["ingest_path"] == "ixhtml"


def test_run_manifest_dedupes_and_counts(monkeypatch):
    # Two entries whose extractors emit an overlapping record id -> deduped once.
    def fake_dispatch(entry):
        return {"a": [_rec("1"), _rec("2")], "b": [_rec("2"), _rec("3", "textract")]}[entry["k"]]

    monkeypatch.setattr(ingest_manifest, "dispatch", fake_dispatch)
    stats = ingest_manifest.run_manifest([{"k": "a"}, {"k": "b"}])

    assert stats["entries"] == 2
    assert stats["records"] == 3  # id "2" collapsed
    assert stats["per_ingest_path"] == {"pdf_text": 2, "textract": 1}


def test_run_manifest_isolates_errors(monkeypatch):
    def fake_dispatch(entry):
        if entry.get("bad"):
            raise ValueError("boom")
        return [_rec("ok")]

    monkeypatch.setattr(ingest_manifest, "dispatch", fake_dispatch)
    stats = ingest_manifest.run_manifest([{"path": "good"}, {"path": "z", "bad": True}])

    assert stats["records"] == 1
    assert len(stats["errors"]) == 1
    assert "boom" in stats["errors"][0]["error"]


def test_run_manifest_is_idempotent(monkeypatch):
    monkeypatch.setattr(ingest_manifest, "dispatch", lambda e: [_rec("1"), _rec("2")])
    run1 = ingest_manifest.run_manifest([{}])
    run2 = ingest_manifest.run_manifest([{}])
    assert run1["records"] == run2["records"] == 2


def test_jsonl_sink_and_load_roundtrip(tmp_path, monkeypatch):
    out = str(tmp_path / "corpus.jsonl")
    monkeypatch.setattr(ingest_manifest, "dispatch", lambda e: [_rec("1"), _rec("2")])
    stats = ingest_manifest.run_manifest([{}], sink=ingest_manifest.jsonl_file_sink(out))

    assert stats["written"] == 2
    lines = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert {r["id"] for r in lines} == {"1", "2"}


def test_load_manifest(tmp_path):
    mpath = str(tmp_path / "m.jsonl")
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"path": "a.pdf", "ingest_path": "pdf_text"}) + "\n")
        fh.write(json.dumps({"path": "b.pdf", "ingest_path": "pdf_table"}) + "\n")
    entries = ingest_manifest.load_manifest(mpath)
    assert len(entries) == 2 and entries[0]["path"] == "a.pdf"
