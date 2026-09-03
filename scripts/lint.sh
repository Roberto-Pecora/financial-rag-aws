#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v ruff >/dev/null 2>&1; then
  RUFF_CMD=(ruff)
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  if "$PYTHON_BIN" -m ruff --version >/dev/null 2>&1; then
    RUFF_CMD=("$PYTHON_BIN" -m ruff)
  else
    echo "Ruff is not installed in the current environment." >&2
    exit 1
  fi
fi

"${RUFF_CMD[@]}" check --fix . || true
"${RUFF_CMD[@]}" format . || true
"${RUFF_CMD[@]}" check . || true
