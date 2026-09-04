"""Agent tools: retrieval, graph lookup, and safe arithmetic.

Each tool pairs a Pydantic argument model with a callable, so the JSON schema
advertised to the model (native function-calling) and the argument validation are
one source of truth. Retrieval tools wrap stores that already return the
{text, metadata, score} contract, so citations flow through unchanged.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from frag.rag.untrusted import wrap_untrusted


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[..., str]

    def spec(self) -> dict[str, Any]:
        """The OpenAI/OpenRouter function-calling spec for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def call(self, arguments: dict[str, Any]) -> str:
        """Validate arguments against the model, then run. Raises ValidationError."""
        return self.run(**self.args_model.model_validate(arguments).model_dump())


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._by_name = {t.name: t for t in tools}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._by_name.values()]

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)


# -- argument models -------------------------------------------------------


class _RetrieveArgs(BaseModel):
    query: str
    top_k: int = 5


class _GraphArgs(BaseModel):
    query: str


class _CalcArgs(BaseModel):
    expression: str


# -- tool bodies -----------------------------------------------------------


def _cite(hits: list[dict[str, Any]]) -> str:
    """Render store hits as cited, untrusted-marked blocks the model can quote."""
    if not hits:
        return "No results."
    blocks = []
    for h in hits:
        meta = h.get("metadata", {})
        label = meta.get("doc_id") or meta.get("entity") or "doc"
        blocks.append(wrap_untrusted(label, h.get("text", "")))
    return "\n\n".join(blocks)


def make_retrieve_tool(store: Any) -> Tool:
    """retrieve(query, top_k) over any store with the search() contract."""

    def run(query: str, top_k: int = 5) -> str:
        return _cite(store.search(query=query, top_k=top_k))

    return Tool(
        name="retrieve",
        description="Search the filing/contract corpus for passages relevant to a query.",
        args_model=_RetrieveArgs,
        run=run,
    )


def make_graph_lookup_tool(retriever: Any) -> Tool:
    """graph_lookup(query) over the credit knowledge graph."""

    def run(query: str) -> str:
        return _cite(retriever.search(query=query))

    return Tool(
        name="graph_lookup",
        description=(
            "Look up structured credit facts (issuer, instrument, covenant, metric) and "
            "which documents contain them, from the knowledge graph."
        ),
        args_model=_GraphArgs,
        run=run,
    )


# Whitelisted AST nodes/operators for a safe calculator — no eval, no names.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_CMP_OPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_OPS:
        return _CMP_OPS[type(node.ops[0])](_eval_node(node.left), _eval_node(node.comparators[0]))
    raise ValueError("unsupported expression")


def safe_calc(expression: str) -> str:
    """Evaluate a whitelisted arithmetic/comparison expression over numbers."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


def make_calc_tool() -> Tool:
    """financial_calc(expression) — deterministic arithmetic/comparison, no eval."""
    return Tool(
        name="financial_calc",
        description=(
            "Evaluate an arithmetic or comparison over numbers you already have, e.g. "
            "'4200 / 1000' or '4.2 > 4'. Numbers only — no variable names."
        ),
        args_model=_CalcArgs,
        run=lambda expression: safe_calc(expression),
    )
