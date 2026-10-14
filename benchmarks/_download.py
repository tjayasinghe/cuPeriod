"""Verified downloads for the benchmark data scripts.

Each download script pins the SHA-256 of every file it fetches. ``fetch`` streams
the file to ``<target>.part`` over a connection with a timeout, refuses a body
larger than ``max_bytes``, and moves it into place only when its digest matches.
A copy already on disk is checked against the same digest before it is reused.
So a substituted file from an on-path attacker, or one truncated by a dropped
connection, is never written to the cache, and a damaged cached copy is never
trusted. Without the digest, those files fed the committed result tables and
validation figures unchecked.

The digests were recorded from the copies behind the committed benchmark results
(trust on first use). If an archive revises a file in place, ``fetch`` stops with a
mismatch. Inspect the new file, then replace its digest in the script.

Standard library only, so it imports in every benchmark venv.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

#: Seconds a connection may stay silent before the transfer is abandoned.
TIMEOUT = 120.0
#: Ceiling on one response body; the largest benchmark file is ~1.2 MB.
MAX_BYTES = 64 * 1024 * 1024
_CHUNK = 1 << 16


class DownloadError(RuntimeError):
    """A transfer or a cached file did not match its pinned SHA-256."""


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    url: str,
    target: Path,
    sha256: str,
    *,
    timeout: float = TIMEOUT,
    max_bytes: int = MAX_BYTES,
    force: bool = False,
) -> bool:
    """Make ``target`` hold the file at ``url`` whose SHA-256 is ``sha256``.

    Parameters
    ----------
    url
        Where to get the file (HTTPS).
    target
        Final path; its folder must exist.
    sha256
        The pinned hex digest the file must have.
    timeout
        Seconds the connection may stay silent (connect and each read).
    max_bytes
        Larger bodies are refused without being kept.
    force
        Download again even when a verified copy is on disk.

    Returns
    -------
    bool
        True when the file was downloaded, False when the cached copy was kept.

    Raises
    ------
    DownloadError
        The cached copy or the downloaded body does not match ``sha256``, or the
        body is larger than ``max_bytes`` or shorter than its Content-Length.
        Nothing is left at ``target`` or ``<target>.part`` after a failed
        download; a mismatching cached copy is left for inspection.
    OSError
        The transfer itself failed (``urllib.error.URLError``, ``TimeoutError``).
    """
    want = sha256.strip().lower()
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        raise ValueError(f"not a hex SHA-256 digest: {sha256!r}")
    target = Path(target)
    if target.exists() and not force:
        got = sha256_of(target)
        if got != want:
            raise DownloadError(
                f"{target} is on disk but its SHA-256 is {got}, not the pinned "
                f"{want}. It is a truncated or altered copy: delete it and run again. "
                "If the archive has revised the file, check the new copy and update "
                "the pinned digest."
            )
        return False

    part = target.with_name(target.name + ".part")
    try:
        digest = hashlib.sha256()
        size = 0
        with (
            urllib.request.urlopen(url, timeout=timeout) as resp,
            part.open("wb") as out,
        ):
            declared = resp.headers.get("Content-Length")
            while chunk := resp.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise DownloadError(f"{url}: body exceeds {max_bytes} bytes")
                digest.update(chunk)
                out.write(chunk)
        # http.client returns a short body without complaint when the connection
        # drops mid-transfer; the digest would catch it, this names the cause.
        if declared is not None and declared.isdigit() and size != int(declared):
            raise DownloadError(
                f"{url}: received {size} of {declared} bytes (connection dropped)"
            )
        got = digest.hexdigest()
        if got != want:
            raise DownloadError(
                f"{url}: SHA-256 is {got}, not the pinned {want}. The file was not "
                "saved. It was altered in transit or revised by the archive."
            )
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return True
