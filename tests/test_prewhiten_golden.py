"""``cup.prewhiten`` reproduces the golden corpus recorded before the stepper refactor.

The golden (``tests/data/prewhiten_golden.json``, written by
``tests/make_prewhiten_golden.py``) holds, for every case of
``tests/prewhiten_corpus.py``, the full component table, the combinations, the stop
reason and counters, the fit statistics, samples of the residuals and summaries of the
stored spectra. Counts, labels, strings and flags must match exactly; numbers match
at ``1e-12`` relative when the numerical environment's fingerprint equals the one the
golden was written in, ``1e-9`` otherwise (a different BLAS or CPU moves the last bits).
The ``finufft`` cases are held to ``prewhiten_corpus.RTOL_FINUFFT`` at best, the
reproducibility that backend actually has.

A golden is never regenerated to make this pass: a failure here means the results of
``cup.prewhiten`` moved.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

import prewhiten_corpus as corpus


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    if not corpus.GOLDEN_PATH.exists():  # pragma: no cover - the file is committed
        pytest.fail(f"missing golden file {corpus.GOLDEN_PATH}")
    document: dict[str, Any] = json.loads(corpus.GOLDEN_PATH.read_text("utf-8"))
    assert document["format"] == corpus.GOLDEN_FORMAT
    return document


@pytest.fixture(scope="module")
def base_rtol(golden: dict[str, Any]) -> float:
    if golden["fingerprint"] == corpus.fingerprint():
        return corpus.RTOL_SAME_ENVIRONMENT
    return corpus.RTOL_OTHER_ENVIRONMENT


def test_every_case_is_recorded(golden: dict[str, Any]) -> None:
    assert sorted(golden["cases"]) == sorted(case.name for case in corpus.CASES)


def test_the_golden_stays_small() -> None:
    assert corpus.GOLDEN_PATH.stat().st_size < 1_000_000


def test_the_corpus_reaches_every_loop_exit(golden: dict[str, Any]) -> None:
    """The corpus is only a guard for the branches it actually takes."""
    reasons = [c["summary"]["stop_reason"] for c in golden["cases"].values()]
    for fragment in (
        "S/N", "FAP", "dBIC", "amplitude", "reached max_frequencies",
        "max_frequencies is 0", "no peak resolved", "too many candidates merged",
        "pruned",
    ):
        assert any(fragment in r for r in reasons), fragment
    summaries = [c["summary"] for c in golden["cases"].values()]
    # Blocked candidates: more attempts than accepted components plus one rejection.
    assert any(
        s["n_iterations"] > len(s["components"]) + s["n_pruned"] + 1 for s in summaries
    )
    assert {s["uncertainty_method"] for s in summaries} == {
        "covariance", "analytic", "bootstrap",
    }


def _close(actual: Any, expected: Any, rtol: float, scale: float) -> bool:
    if expected is None or actual is None:
        return actual is expected
    a, e = float(actual), float(expected)
    if math.isnan(e):
        return math.isnan(a)
    if math.isinf(e):
        return a == e
    return abs(a - e) <= rtol * max(abs(e), scale)


def _compare(
    actual: Any, expected: Any, rtol: float, path: str, scale: float,
    problems: list[str],
) -> None:
    """Recursive comparison: exact for non-floats, relative for floats."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            problems.append(f"{path}: keys {actual!r} != {expected!r}")
            return
        for key in expected:
            inner = scale
            if key == "residuals" and isinstance(expected[key], dict):
                # Residuals scatter about zero: their mean and the samples that happen
                # to sit near zero are held to the residual scatter, not to themselves.
                inner = max(scale, float(expected[key]["std"]))
            _compare(actual[key], expected[key], rtol, f"{path}.{key}", inner, problems)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problems.append(f"{path}: {actual!r} != {expected!r}")
            return
        # A list of numbers shares one absolute floor, a millionth of its largest
        # magnitude, so an entry that happens to sit near zero is not held to itself.
        numbers = [
            abs(x) for x in expected if isinstance(x, float) and math.isfinite(x)
        ]
        inner = max(numbers, default=0.0) * 1e-6 if numbers else scale
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, rtol, f"{path}[{i}]", max(scale, inner), problems)
        return
    if isinstance(expected, float) and not isinstance(expected, bool):
        if not _close(actual, expected, rtol, scale):
            problems.append(f"{path}: {actual!r} != {expected!r} (rtol {rtol:g})")
        return
    if actual != expected:
        problems.append(f"{path}: {actual!r} != {expected!r}")


@pytest.mark.parametrize("case", corpus.CASES, ids=lambda c: c.name)
def test_prewhiten_matches_the_golden(
    case: corpus.Case, golden: dict[str, Any], base_rtol: float
) -> None:
    recorded = golden["cases"][case.name]
    assert recorded["case"] == corpus.describe(case), "corpus definition changed"
    expected = recorded["summary"]
    actual = json.loads(json.dumps(corpus.summarize(corpus.run(case))))
    rtol = max(base_rtol, case.rtol_floor)
    problems: list[str] = []
    # Integers, strings and flags (counts, labels, stop reason, combinations, blend
    # flags) compare exactly inside _compare; floats relatively. Values that are zero
    # by construction (a residual mean, an offset of centred data) get an absolute
    # floor of 1e-300, i.e. none.
    _compare(actual, expected, rtol, case.name, 1e-300, problems)
    assert not problems, "\n".join(problems[:20])


def test_the_comparison_catches_a_drift(golden: dict[str, Any]) -> None:
    """The comparison itself is not vacuous: a 1e-10 move or a renamed label fails."""
    expected = golden["cases"]["triple-last"]["summary"]
    moved = json.loads(json.dumps(expected))
    moved["components"][0]["frequency"] *= 1.0 + 1e-10
    problems: list[str] = []
    _compare(moved, expected, corpus.RTOL_SAME_ENVIRONMENT, "x", 1e-300, problems)
    assert problems and "frequency" in problems[0]
    relabelled = json.loads(json.dumps(expected))
    relabelled["components"][0]["label"] = "F9"
    problems = []
    _compare(relabelled, expected, corpus.RTOL_OTHER_ENVIRONMENT, "x", 1e-300, problems)
    assert problems and "label" in problems[0]
