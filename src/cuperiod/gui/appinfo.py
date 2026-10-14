"""Application version and git revision, for display in the header.

The revision is read from cuPeriod's *own* source checkout, in pure Python. Two
earlier behaviours are deliberately gone. Running ``git`` by bare name executed any
``git.exe`` in the folder the GUI was launched from (Windows looks in the current
directory before ``PATH``, whatever ``cwd=`` says), and letting git search upward
from an installed package reported the commit of whatever unrelated repository
happened to enclose the virtualenv. A wheel install therefore shows no hash at all.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

#: A full object name: SHA-1 (40 hex) or SHA-256 (64 hex).
_OBJECT_NAME = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")

#: A symbolic ref HEAD may point at; anything else (``..``, absolute paths) is refused.
_REF_NAME = re.compile(r"refs/[A-Za-z0-9._/-]+")

#: Upper bound on what is read from any one git metadata file.
_MAX_READ = 1 << 20


@lru_cache(maxsize=1)
def app_version() -> str:
    """The installed cuperiod version, or ``"?"`` if it cannot be resolved."""
    try:
        return _package_version("cuperiod")
    except PackageNotFoundError:  # pragma: no cover - source runs without metadata
        return "?"


def checkout_root(module_file: str | Path | None = None) -> Path | None:
    """The cuPeriod source checkout this module was imported from, or None.

    ``module_file`` lives at ``<root>/src/cuperiod/gui/appinfo.py`` in a checkout (or
    an editable install of one); the root must hold ``src/cuperiod`` and a ``.git``
    entry. Nothing above the root is searched, so an installed package inside some
    other project's repository is not mistaken for a checkout.
    """
    module = Path(module_file if module_file is not None else __file__).resolve()
    parents = module.parents
    if len(parents) <= 3:
        return None
    root = parents[3]
    if not (root / "src" / "cuperiod").is_dir() or not (root / ".git").exists():
        return None
    return root


def _read_small(path: Path) -> str | None:
    """A small text file's stripped content, or None if unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_READ).strip()
    except OSError:
        return None


def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    """``(git_dir, common_dir)`` of a checkout, following a worktree's ``.git`` file."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        git_dir = dot_git
    elif dot_git.is_file():
        text = _read_small(dot_git)
        if text is None or not text.startswith("gitdir:"):
            return None
        git_dir = Path(text[len("gitdir:") :].strip())
        if not git_dir.is_absolute():
            git_dir = root / git_dir
        if not git_dir.is_dir():
            return None
    else:
        return None
    common = git_dir
    commondir = _read_small(git_dir / "commondir")
    if commondir:
        candidate = Path(commondir)
        common = candidate if candidate.is_absolute() else git_dir / candidate
    return git_dir, common


def _resolve_ref(ref: str, git_dir: Path, common: Path) -> str | None:
    """The object name a symbolic ref points at (loose ref first, then packed-refs)."""
    for base in (git_dir, common):
        loose = _read_small(base / ref)
        if loose:
            return loose.split()[0]
    packed = _read_small(common / "packed-refs")
    if packed is None:
        return None
    for line in packed.splitlines():
        if line.startswith(("#", "^")):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def read_git_revision(root: Path) -> str | None:
    """The 7-character HEAD revision of the checkout at ``root``, or None.

    Reads ``.git/HEAD`` directly: a detached object name, or a ``ref: refs/...`` line
    resolved through the loose ref and then ``packed-refs``; a worktree's ``gitdir:``
    file and its ``commondir`` are followed. Refs that leave ``refs/`` (``..``,
    absolute paths) and anything that is not a full hex object name give None, as does
    any unreadable file. Exotic layouts (reftable) simply show no revision.
    """
    dirs = _git_dirs(root)
    if dirs is None:
        return None
    git_dir, common = dirs
    head = _read_small(git_dir / "HEAD")
    if not head:
        return None
    if head.startswith("ref:"):
        ref = head[len("ref:") :].strip()
        if not _REF_NAME.fullmatch(ref) or ".." in ref.split("/"):
            return None
        name = _resolve_ref(ref, git_dir, common)
    else:
        name = head
    if name is None or not _OBJECT_NAME.fullmatch(name):
        return None
    return name[:7]


@lru_cache(maxsize=1)
def git_short_hash() -> str | None:
    """The short git hash of cuPeriod's source checkout, or None outside one.

    Never spawns a process (see the module docstring).
    """
    root = checkout_root()
    return read_git_revision(root) if root is not None else None


def version_label() -> str:
    """A compact ``v<version> · <hash>`` label (hash omitted when unavailable)."""
    version = app_version()
    short_hash = git_short_hash()
    return f"v{version} · {short_hash}" if short_hash else f"v{version}"


__all__ = [
    "app_version",
    "checkout_root",
    "git_short_hash",
    "read_git_revision",
    "version_label",
]
