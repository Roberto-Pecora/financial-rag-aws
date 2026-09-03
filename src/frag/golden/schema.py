from typing import Any

from pydantic import BaseModel


class GoldenExample(BaseModel):
    query: str
    gold_doc_ids: list[str]
    reference_answer: str
    task_type: str
    risk_level: str
    notes: str | None = None
    metadata_filter: dict[str, Any] | None = None
