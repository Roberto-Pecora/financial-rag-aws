from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` to the first directory containing a ``.env``
    or ``pyproject.toml``, falling back to ``start`` if neither is found.

    Searching rather than hardcoding a parent depth keeps this correct if the
    package is moved within the tree (e.g. ``src/utils`` -> ``src/frag/utils``).
    """
    for candidate in (start, *start.parents):
        if (candidate / ".env").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


def configure_runtime() -> Path:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    dotenv_path = repo_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path
