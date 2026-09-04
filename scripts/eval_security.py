"""Score the guardrails against an injection eval set; report the defence rate.

    python scripts/eval_security.py --cases data/security/injection_cases.jsonl

All checks are deterministic and need no key. Prints a per-case table and the rate.
"""

from __future__ import annotations

import argparse
import json

from frag.agent.guardrails import screen_input, screen_output
from frag.rag.untrusted import wrap_untrusted


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _observe(case: dict) -> str:
    kind = case["kind"]
    if kind == "input":
        return "refused" if not screen_input(case["query"]).allowed else "allowed"
    if kind == "document":
        # A document is defended if wrapping neutralises any forged end-marker.
        wrapped = wrap_untrusted(case["id"], case["text"])
        forged = "<<<END_UNTRUSTED_DOC>>>" in case["text"]
        broke_out = wrapped.count("<<<END_UNTRUSTED_DOC>>>") > 1
        return "neutralised" if (forged and not broke_out) else "wrapped"
    if kind == "output":
        v = screen_output(case["answer"], case["citations"], set(case["allowed_labels"]))
        return "allowed" if v.allowed else "blocked"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/security/injection_cases.jsonl")
    args = ap.parse_args()

    cases = _load(args.cases)
    passed = 0
    print(f"{'case':28} {'kind':10} {'expected':12} {'observed':12} pass")
    for c in cases:
        observed = _observe(c)
        ok = observed == c["expected"]
        passed += ok
        print(f"{c['id']:28} {c['kind']:10} {c['expected']:12} {observed:12} {'✓' if ok else '✗'}")
    print(f"\nDefence rate: {passed}/{len(cases)} ({100 * passed / len(cases):.0f}%)")


if __name__ == "__main__":
    main()
