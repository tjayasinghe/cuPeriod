"""Every documented settings field is read by the library, or deprecated loudly.

CORE-A-10 / CORE-B-10: ``n_peaks``, ``peak_separation_rayleigh``,
``downsample_points``, ``alias_freq_tolerance`` and ``harmonic_max`` were validated,
documented and environment-overridable but read by nothing, and ``BatchSettings``
(``CUPERIOD_BATCH_*``) still is. The peak fields are now wired through
``Periodogram.peak_defaults``; ``BatchSettings`` warns. This module keeps the next
dead field from slipping in unnoticed.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

import cuperiod as cup
from cuperiod.core import config
from cuperiod.core.result import PEAK_DEFAULT_FIELDS

SRC = Path(cup.__file__).resolve().parent

#: Models whose fields are deprecated as a whole (building one warns).
DEPRECATED_MODELS = {"BatchSettings"}


def _library_reads() -> set[str]:
    """Attribute names the library loads, outside the models and the GUI.

    Plus the fields read by name (``getattr``) to default a result's peak selection.
    The GUI is excluded: a field only the GUI's form reads changes no library result.
    """
    names: set[str] = set(PEAK_DEFAULT_FIELDS.values())
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("gui/") or rel == "core/config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                names.add(node.attr)
    return names


def _models() -> list[type]:
    return [
        getattr(config, name) for name in config.__all__ if name.endswith("Settings")
    ]


def test_every_settings_field_is_read_by_the_library() -> None:
    reads = _library_reads()
    dead = [
        f"{model.__name__}.{field}"
        for model in _models()
        if model.__name__ not in DEPRECATED_MODELS
        for field in model.model_fields
        if field not in reads
    ]
    assert dead == [], (
        "settings fields nothing reads (wire them, or retire them through "
        f"_retired_fields so setting one warns): {dead}"
    )


def test_batch_settings_is_deprecated_and_says_so_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning, match="BatchSettings is deprecated") as rec:
        cup.BatchSettings(workers=4)
    assert rec[0].filename == __file__
    for name in DEPRECATED_MODELS:  # every wholly deprecated model warns when built
        with pytest.warns(DeprecationWarning):
            getattr(config, name)()


@pytest.mark.parametrize(
    "run",
    [
        lambda: cup.batch_periodograms([], "GLS", workers=0),
        lambda: cup.batch_pdot([], workers=0),
        lambda: cup.batch_prewhiten([], workers=0),
    ],
    ids=["periodograms", "pdot", "prewhiten"],
)
def test_batch_environment_variables_are_reported_as_dead(
    monkeypatch: pytest.MonkeyPatch, run: object
) -> None:
    monkeypatch.setenv("CUPERIOD_BATCH_WORKERS", "8")
    with pytest.warns(DeprecationWarning, match="CUPERIOD_BATCH_WORKERS") as rec:
        run()  # type: ignore[operator]
    assert any(w.filename == __file__ for w in rec)  # attributed to the caller
    monkeypatch.delenv("CUPERIOD_BATCH_WORKERS")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*CUPERIOD_BATCH_")
        run()  # type: ignore[operator]


def test_the_retired_prewhiten_field_is_not_a_model_field() -> None:
    # Retired fields leave the model (so the meta-test above cannot see them) and
    # warn when passed; the vocabulary of retired names stays declared.
    assert "downsample_points" in cup.PreWhitenSettings._retired_fields
    assert "downsample_points" not in cup.PreWhitenSettings.model_fields
