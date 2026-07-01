"""Method and backend introspection helpers for the GUI.

Thin, typed wrappers over the cuPeriod method registry (:mod:`cuperiod.methods.base`)
and backend probes (:mod:`cuperiod.core.backend`) so the controller and controls panel
can ask "what methods exist?", "what backends can this run here?", and "is this method
multiband / what's its natural domain?" — without importing registry internals widely.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.core.grid import pseudo_nyquist_frequency
from cuperiod.methods.base import get_method, method_names

#: Box/transit methods where alias-diverse peak selection is the sensible default.
_ALIAS_DIVERSE_METHODS = frozenset({"BLS", "TLS"})

#: The GUI's auto grid reaches at least this short a period (days) for freq methods.
MIN_PERIOD_FLOOR_DAYS = 0.1


def method_display_names() -> list[str]:
    """All registered method names (canonical, sorted)."""
    return method_names()


def multiband_method_names() -> list[str]:
    """Only the methods that support multiband input (GLS, BLS, MHAOV)."""
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


def resolved_backend(name: str, backend: str) -> tuple[str, bool] | None:
    """The concrete backend ``backend`` resolves to for ``name`` and whether it's a GPU.

    Returns ``(resolved_name, is_gpu)``, or None if it cannot resolve here. Lets the UI
    show, e.g., ``auto -> cufinufft (GPU)`` so ``cpu``/``gpu`` are unambiguous.
    """
    method = get_method(name)
    try:
        resolved = method.resolve_backend(backend)
    except BackendUnavailableError:
        return None
    return resolved, method.is_gpu_backend(resolved)


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
    "MIN_PERIOD_FLOOR_DAYS",
    "alias_diverse_default",
    "auto_max_frequency",
    "backend_options",
    "default_settings",
    "method_display_names",
    "multiband_method_names",
    "natural_domain",
    "objective_sense",
    "resolved_backend",
    "settings_class",
    "supports_multiband",
]
