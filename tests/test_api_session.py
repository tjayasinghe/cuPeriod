"""The public surface of interactive pre-whitening."""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest

import cuperiod as cup
import prewhiten_corpus as pc
from cuperiod.core.lightcurve import stack_bands
from cuperiod.prewhiten.session import PreWhitenSession, actions

EXPORTS = [
    "PreWhitenSession", "SessionSettings", "SessionSnapshot", "StepReport",
    "FitConstraints", "ComponentConstraint", "FrequencyTie", "FitStart",
    "ConstraintError", "Partition", "SessionError", "ProjectError",
    "SessionWarning", "VariationResult",
]


@pytest.mark.parametrize("name", EXPORTS)
def test_top_level_exports(name: str) -> None:
    # cup.prewhiten is the function; the subpackage is in sys.modules.
    pw = sys.modules["cuperiod.prewhiten"]
    assert name in cup.__all__ and name in pw.__all__
    assert getattr(cup, name) is getattr(pw, name)


def test_error_hierarchy() -> None:
    assert issubclass(cup.ProjectError, cup.SessionError)
    assert issubclass(cup.SessionError, cup.CuPeriodError)
    assert issubclass(cup.ConstraintError, ValueError)
    assert issubclass(cup.SessionWarning, UserWarning)


def test_a_typed_method_for_every_op() -> None:
    methods = {
        name for name, _ in inspect.getmembers(PreWhitenSession, inspect.isfunction)
    }
    assert set(actions.ARGS) <= methods
    for name in ("advance", "run", "finalize", "undo", "redo", "goto", "snapshot",
                 "restore", "save", "load", "replay", "to_script", "export_script",
                 "release", "result", "candidate", "variations", "spectrum",
                 "residual_spectrum", "window", "from_bands", "from_csv"):
        assert hasattr(PreWhitenSession, name), name


def test_prewhiten_signature_is_unchanged() -> None:
    params = list(inspect.signature(cup.prewhiten).parameters)
    assert params == ["data", "settings", "grid", "backend", "columns", "domain"]


def test_quick_tour() -> None:
    data, settings, _ = pc.load(pc.case_by_name("triple-last"))
    session = cup.PreWhitenSession(data, settings=settings)
    assert session.status == "searching" and session.components == ()
    first = session.step()
    assert first.kind == "accepted" and session.components[0].label == "F1"
    assert session.result().n_components == 1
    session.run()
    result = session.finalize()
    assert result == result and result.n_components == 3
    assert "searching" not in result.stop_reason
    assert repr(session).startswith("PreWhitenSession(")
    snapshot = session.snapshot()
    assert isinstance(snapshot, cup.SessionSnapshot)
    assert snapshot.creation.backend == session.backend == result.backend


def test_an_interactive_result_says_so() -> None:
    data, settings, _ = pc.load(pc.case_by_name("triple-last"))
    session = cup.PreWhitenSession(data, settings=settings)
    session.step()
    assert session.result().stop_reason == (
        "interactive session (not run to a stopping rule)"
    )


def test_from_bands_has_a_root_band_partition() -> None:
    rng = np.random.default_rng(5)
    bands = {}
    for name, zero in (("g", 15.0), ("r", 14.6)):
        t = np.sort(rng.uniform(0.0, 40.0, 150))
        y = zero + 0.05 * np.sin(2 * np.pi * 4.2 * t) + rng.normal(0, 0.005, t.size)
        bands[name] = cup.LightCurve.from_arrays(t, y, np.full(t.size, 0.005))
    mblc = cup.MultiBandLightCurve.from_light_curves(bands)
    settings = cup.PreWhitenSettings(backend="numpy", maximum_frequency=10.0,
                                     max_frequencies=2)
    session = PreWhitenSession.from_bands(mblc, settings=settings)
    (band,) = session.state.partitions
    assert band.name == "band" and set(band.labels) == {"g", "r"}
    lc, labels = stack_bands(mblc, centre=True)
    np.testing.assert_array_equal(session.data.value, lc.value)
    assert session.data.labels["band"].tolist() == labels.tolist()
    session.run()
    plain = cup.prewhiten(lc, settings=settings)
    stacked = session.finalize()
    np.testing.assert_array_equal(stacked.frequency, plain.frequency)
    assert session.snapshot().creation.partitions == (band,)
