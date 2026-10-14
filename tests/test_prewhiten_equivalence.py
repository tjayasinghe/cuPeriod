"""cup.prewhiten after the switch: a session run and finalized, nothing else.

The golden corpus (tests/test_prewhiten_golden.py) holds its numbers to the
pre-session release; this file holds the public function to an explicitly driven
session (bitwise, numpy backend) and checks what the switch must not change around
it: writable result arrays, untouched inputs, and a light import.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.prewhiten.session import PreWhitenSession

CASES = [
    "triple-last", "triple-no-polish", "triple-fit-mean-false", "triple-bootstrap",
    "noerr-last", "pair-tight-guard", "combos-last", "nightly-last", "noise-last",
    "narrow-no-peak", "modulated-prune", "drift-budget",
]


def _summary(result: cup.PreWhitenResult) -> str:
    return json.dumps(pc.summarize(result), sort_keys=True)


@pytest.mark.parametrize("name", CASES)
def test_prewhiten_is_a_finalized_session(name: str) -> None:
    case = pc.case_by_name(name)
    data, settings, grid = pc.load(case)
    t, y, e = data
    payload = (t, y) if e is None else (t, y, e)
    public = cup.prewhiten(payload, settings=settings, grid=grid)
    session = PreWhitenSession(payload, settings=settings, grid=grid)
    session.run()
    driven = session.finalize()
    assert _summary(public) == _summary(driven)
    np.testing.assert_array_equal(public.residuals, driven.residuals)
    assert public.stop_reason == driven.stop_reason
    assert public.n_iterations == driven.n_iterations
    assert public.n_pruned == driven.n_pruned
    assert [c.label for c in public] == [c.label for c in driven]
    assert public.combinations == driven.combinations
    assert [c.blended for c in public] == [c.blended for c in driven]
    assert public.subsets is None and public.n_deleted == 0
    assert all(not c.has_constraint and c.flags == () for c in public)


def test_result_arrays_are_writable_and_inputs_untouched() -> None:
    data, settings, _ = pc.load(pc.case_by_name("triple-last"))
    t, y, e = data
    assert e is not None
    before = (t.copy(), y.copy(), e.copy())
    lc = cup.LightCurve.from_arrays(t, y, e)
    result = cup.prewhiten(lc, settings=settings)
    assert result.time is lc.time  # the coerced array itself, as in v1.3
    assert result.time.flags.writeable and result.residuals.flags.writeable
    assert t.flags.writeable and y.flags.writeable and e.flags.writeable
    for original, now in zip(before, (t, y, e), strict=True):
        np.testing.assert_array_equal(original, now)


def test_import_stays_light() -> None:
    code = (
        "import sys, cuperiod\n"
        "lazy = ('project', 'script', 'sampling', 'modelmap', 'variation')\n"
        "bad = [m for m in sys.modules if m.startswith('cuperiod.prewhiten.session.')"
        " and m.rsplit('.', 1)[-1] in lazy]\n"
        "qt = [m for m in sys.modules if m.split('.')[0] in "
        "('PySide6', 'pyqtgraph', 'PyQt5', 'PyQt6')]\n"
        "print(bad + qt)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_a_pure_run_never_builds_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    # The automatic path is the v1.3 call sequence: no model map is ever built.
    from cuperiod.prewhiten.session.kernel import SessionKernel

    original = SessionKernel.model_map

    def guarded(self: SessionKernel, state: object, **kw: object) -> object:
        mm = original(self, state, **kw)  # type: ignore[arg-type]
        assert mm is None, "the plain path built a model map"
        return mm

    monkeypatch.setattr(SessionKernel, "model_map", guarded)
    data, settings, _ = pc.load(pc.case_by_name("pair-tight-guard"))
    result = cup.prewhiten(data, settings=settings)
    assert result.n_components > 0
