"""Application version and git revision, for display in the header."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path


@lru_cache(maxsize=1)
def app_version() -> str:
    """The installed cuperiod version, or ``"?"`` if it cannot be resolved."""
    try:
        return _package_version("cuperiod")
    except PackageNotFoundError:  # pragma: no cover - source runs without metadata
        return "?"


@lru_cache(maxsize=1)
def git_short_hash() -> str | None:
    """The short git hash of the working tree, or None outside a git checkout."""
    repo = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def version_label() -> str:
    """A compact ``v<version> · <hash>`` label (hash omitted when unavailable)."""
    version = app_version()
    short_hash = git_short_hash()
    return f"v{version} · {short_hash}" if short_hash else f"v{version}"


__all__ = ["app_version", "git_short_hash", "version_label"]
