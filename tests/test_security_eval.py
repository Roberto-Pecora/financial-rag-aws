"""The injection eval: every committed case behaves as expected (100% defence)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_security", _ROOT / "scripts" / "eval_security.py"
)
eval_security = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_security)


def test_all_injection_cases_defended():
    cases = eval_security._load(_ROOT / "data" / "security" / "injection_cases.jsonl")
    assert cases  # the set is not empty
    for c in cases:
        assert eval_security._observe(c) == c["expected"], c["id"]
