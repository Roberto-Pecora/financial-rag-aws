"""Pydantic models for the LLM boundary — the JSON the actor and critic emit.

model_validate_json() does decode, shape-check, coercion and defaults in one
call, raising ValidationError for anything malformed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ActorResponse(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _as_stripped_str(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("citations", mode="before")
    @classmethod
    def _as_str_list(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(c).strip() for c in v]


class CriticResponse(BaseModel):
    overall_score: float = 0.0
    faithfulness_score: float = 0.0
    completeness_score: float = 0.0
    citation_score: float = 0.0
    issues: list[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def _as_str_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return [str(v)]
        return [str(i) for i in v]
