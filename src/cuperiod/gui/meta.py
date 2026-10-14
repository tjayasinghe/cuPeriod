"""Method and backend introspection helpers for the GUI.

Thin, typed wrappers over the cuPeriod method registry (:mod:`cuperiod.methods.base`)
and backend probes (:mod:`cuperiod.core.backend`) so the controller and controls panel
can ask "what methods exist?", "what backends can this run here?", and "is this method
multiband / what's its natural domain?" — without importing registry internals widely.
"""

from __future__ import annotations

from typing import Final, TypeVar, cast

from pydantic_settings import BaseSettings

from cuperiod.core import backend as _backend
from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import GLSSettings, MHAOVSettings, PreWhitenSettings
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.core.grid import pseudo_nyquist_frequency
from cuperiod.core.result import PEAK_DEFAULT_FIELDS
from cuperiod.methods.base import get_method, method_names

_S = TypeVar("_S", bound=BaseSettings)

#: Box/transit methods where alias-diverse peak selection is the sensible default.
_ALIAS_DIVERSE_METHODS = frozenset({"BLS", "TLS"})

#: The GUI's frequency oversampling: denser than the library's 5 samples per peak, for
#: smoother, better-resolved spectra (the jagged look on a period axis at long
#: periods). It is the value the options form *shows* by default, so what is displayed
#: is what runs, and an explicit 5 (or anything else) is honoured as typed.
GUI_SAMPLES_PER_PEAK: Final = 10

#: Settings fields that only steer peak selection on a finished spectrum (the keys of
#: :data:`cuperiod.core.result.PEAK_DEFAULT_FIELDS`). The GUI applies them after the
#: compute, so they are left out of the cache key: editing one re-selects peaks from
#: the cached spectrum instead of recomputing it.
PEAK_ONLY_FIELDS: Final[frozenset[str]] = frozenset(PEAK_DEFAULT_FIELDS.values())


def gui_default_settings(model_cls: type[_S]) -> _S:
    """The settings the GUI starts a form from: the library's, plus GUI defaults.

    The model is instantiated normally, so documented ``CUPERIOD_<METHOD>_<FIELD>``
    environment overrides apply exactly as they do for :func:`cuperiod.periodogram`;
    an invalid override falls back to the class defaults instead of breaking the
    controls panel (pydantic's ``ValidationError`` and pydantic-settings'
    ``SettingsError`` are both ``ValueError``). On top of that, a ``samples_per_peak``
    left at a default below :data:`GUI_SAMPLES_PER_PEAK` is raised to it; one set in
    the environment is kept.
    """
    try:
        base = model_cls()
    except ValueError:
        base = cast(_S, model_cls.model_construct())
    if (
        "samples_per_peak" in model_cls.model_fields
        and "samples_per_peak" not in base.model_fields_set
        and getattr(base, "samples_per_peak", GUI_SAMPLES_PER_PEAK)
        < GUI_SAMPLES_PER_PEAK
    ):
        base = base.model_copy(update={"samples_per_peak": GUI_SAMPLES_PER_PEAK})
    return base


#: The GUI's auto grid reaches at least this short a period (days) for the fold-based
#: methods (PDM, CE, string-length), which pay a full fold of the light curve per
#: trial frequency. The frequency-domain analyses reach shorter periods — see
#: :func:`reaches_short_periods`.
MIN_PERIOD_FLOOR_DAYS = 0.1

#: Analyses whose cost per trial frequency is a trig sum / one NUFFT, so a wide band
#: is cheap and the auto grid can afford the same δ Scuti / HADS floor pre-whitening
#: uses (50 cycles/day — the bundled ASAS-SN HADS at P = 0.0898 d is the star the old
#: 10 c/d ceiling silently aliased).
_SHORT_PERIOD_SETTINGS: tuple[type[BaseSettings], ...] = (
    GLSSettings,
    MHAOVSettings,
    PreWhitenSettings,
)


def reaches_short_periods(settings: BaseSettings | type[BaseSettings]) -> bool:
    """Whether this analysis' auto band uses the short-period (50 c/d) floor."""
    cls = settings if isinstance(settings, type) else type(settings)
    return issubclass(cls, _SHORT_PERIOD_SETTINGS)


def method_display_names() -> list[str]:
    """All registered method names (canonical, sorted)."""
    return method_names()


def multiband_method_names() -> list[str]:
    """Only the methods that support multiband input (everything but TLS)."""
    return [name for name in method_names() if supports_multiband(name)]


def settings_class(name: str) -> type[BaseSettings]:
    """The pydantic settings class for ``name`` (drives the auto-generated form)."""
    return get_method(name).settings_cls


def default_settings(name: str) -> BaseSettings:
    """A default settings instance for ``name``."""
    return get_method(name).coerce_settings(None)


def supports_multiband(name: str) -> bool:
    """Whether ``name`` has a multi-band variant."""
    return get_method(name).supports_multiband


def objective_sense(name: str) -> str:
    """``"max"`` (peaks are maxima) or ``"min"`` (peaks are minima) for ``name``."""
    return get_method(name).objective_sense


def natural_domain(name: str) -> Domain | None:
    """The domain ``name`` forces (``Domain.FLUX`` for box/transit methods), or None."""
    return get_method(name).natural_domain


def alias_diverse_default(name: str) -> bool:
    """Whether alias-diverse peak selection is ``name``'s default (box searches)."""
    return get_method(name).name in _ALIAS_DIVERSE_METHODS


def auto_max_frequency(time: FloatArray, nyquist_factor: int = 5) -> float:
    """Auto maximum frequency: pseudo-Nyquist, floored so short periods are reached."""
    floor = 1.0 / MIN_PERIOD_FLOOR_DAYS
    return float(max(pseudo_nyquist_frequency(time, nyquist_factor), floor))


def prewhiten_backend_options() -> list[str]:
    """Backends the amplitude spectrum can actually run on this machine.

    Mirrors :func:`backend_options` for the pre-whitening analysis, which is not a
    registry method: each candidate is test-resolved so only usable entries are offered.
    """
    from cuperiod.prewhiten.spectrum import SPECTRUM_BACKENDS, resolve_spectrum_backend

    options: list[str] = []
    for backend in ("auto", "cpu", "gpu", "torch", *SPECTRUM_BACKENDS):
        if backend in options:
            continue
        try:
            resolve_spectrum_backend(backend)
        except BackendUnavailableError:
            continue
        options.append(backend)
    return options


def _torch_runs_on_gpu(resolved: str, settings: BaseSettings | None) -> bool:
    """Whether a resolved torch backend name will run on a GPU.

    ``torch:<device>`` names its device. Bare ``torch`` runs on ``settings.device`` when
    that pins one, else on the best device torch sees -- the CPU on a machine whose
    torch has no GPU, which the hint must say rather than "GPU".
    """
    if resolved.startswith("torch:"):
        return resolved != "torch:cpu"
    device = str(getattr(settings, "device", "auto"))
    if device != "auto":
        return device != "cpu"
    return _backend.torch_gpu_available()


def resolved_prewhiten_backend(
    backend: str, settings: BaseSettings | None = None
) -> tuple[str, bool] | None:
    """``(concrete_backend, is_gpu)`` for a pre-whitening backend request, or None.

    ``settings`` (optional) supplies the ``device`` a bare ``torch`` request runs on.
    """
    from cuperiod.prewhiten.spectrum import resolve_spectrum_backend

    try:
        resolved = resolve_spectrum_backend(backend)
    except BackendUnavailableError:
        return None
    if resolved == "torch" or resolved.startswith("torch:"):
        return resolved, _torch_runs_on_gpu(resolved, settings)
    return resolved, resolved == "cufinufft"


def resolved_backend(
    name: str, backend: str, settings: BaseSettings | None = None
) -> tuple[str, bool] | None:
    """The concrete backend ``backend`` resolves to for ``name`` and whether it's a GPU.

    Returns ``(resolved_name, is_gpu)``, or None if it cannot resolve here. Lets the UI
    show, e.g., ``auto -> cufinufft (GPU)`` so ``cpu``/``gpu`` are unambiguous.

    Resolved exactly as the run resolves it: ``auto`` defers to ``settings.backend``
    (the ``CUPERIOD_<METHOD>_BACKEND`` override included), and a bare ``torch`` runs on
    ``settings.device`` or else the best device torch sees. ``settings`` defaults to
    the method's environment-configured defaults.
    """
    method = get_method(name)
    if settings is None:
        try:
            settings = method.coerce_settings(None)
        except ValueError:  # an invalid environment override; the run reports it
            settings = None
    try:
        resolved = method.resolve_backend_for(backend, settings)
    except BackendUnavailableError:
        return None
    return resolved, method.is_gpu_backend(resolved, settings)


def backend_options(name: str) -> list[str]:
    """Backends that actually resolve for ``name`` on this machine (validated).

    Each candidate (``auto``/``cpu``/``gpu``/``torch`` and the method's concrete
    backends) is test-resolved through ``method.resolve_backend`` so only backends that
    really run are offered — the set is method-specific and reflects what is installed.
    """
    method = get_method(name)
    candidates = ["auto", "cpu", "gpu", "torch", *method.all_backends]
    options: list[str] = []
    for backend in candidates:
        if backend in options:
            continue
        try:
            method.resolve_backend(backend)
        except BackendUnavailableError:
            continue
        options.append(backend)
    return options


__all__ = [
    "GUI_SAMPLES_PER_PEAK",
    "MIN_PERIOD_FLOOR_DAYS",
    "PEAK_ONLY_FIELDS",
    "alias_diverse_default",
    "auto_max_frequency",
    "backend_options",
    "default_settings",
    "gui_default_settings",
    "method_display_names",
    "multiband_method_names",
    "natural_domain",
    "objective_sense",
    "prewhiten_backend_options",
    "reaches_short_periods",
    "resolved_backend",
    "resolved_prewhiten_backend",
    "settings_class",
    "supports_multiband",
]
