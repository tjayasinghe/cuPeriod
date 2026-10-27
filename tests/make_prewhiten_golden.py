"""Write ``tests/data/prewhiten_golden.json`` from the current tree.

Run from the repository root::

    python tests/make_prewhiten_golden.py [--output PATH] [--force]

The golden pins the results of :func:`cuperiod.prewhiten` on the corpus of
``tests/prewhiten_corpus.py`` *before* the extraction loop is refactored, so it must
be written from a tree whose pre-whitening sources are the reference behaviour. It
records the commit, the digests of the pre-whitening sources and the numerical
environment's fingerprint next to the summaries; the test compares at ``1e-12`` when
the running fingerprint matches and at the looser cross-environment tolerances of
``tests/prewhiten_corpus.py`` otherwise.

An existing golden is never overwritten without ``--force``: regenerating it from a
tree whose behaviour has already moved would silently bless the drift it exists to
catch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep the run off any GPU: every corpus case names a CPU backend, but importing
# cuperiod probes devices.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, str(Path(__file__).parent))

import cuperiod as cup  # noqa: E402
import prewhiten_corpus as corpus  # noqa: E402


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def build() -> dict[str, object]:
    """Run the corpus and assemble the golden document."""
    cases = {}
    for case in corpus.CASES:
        result = corpus.run(case)
        cases[case.name] = {
            "case": corpus.describe(case),
            "summary": corpus.summarize(result),
        }
    return {
        "format": corpus.GOLDEN_FORMAT,
        "commit": _commit(),
        "cuperiod_version": cup.__version__,
        "sources": corpus.source_digest(),
        "fingerprint": corpus.fingerprint(),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=corpus.GOLDEN_PATH)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing golden file")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        print(f"{args.output} exists; pass --force to overwrite it", file=sys.stderr)
        return 1
    document = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {len(corpus.CASES)} cases ({len(text) / 1024:.0f} KiB) "
          f"to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
