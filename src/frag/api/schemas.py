from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    id: str | None = None
    text: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = 8
    metadata_filter: dict[str, Any] | None = None


class EvalRequest(BaseModel):
    predictions: Any
    golden: Any
