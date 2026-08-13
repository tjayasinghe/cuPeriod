"""The method interface and registry.

Every periodogram is a :class:`PeriodogramMethod` subclass registered under an
uppercase name (``"GLS"``, ``"BLS"``, ...). The single-run, batch, and CLI layers only
ever go through this interface — ``default_grid`` to pick a trial grid, ``power`` to
compute a :class:`~cuperiod.core.result.Periodogram`, and ``make_engine`` to build a
reusable plan/kernel for batch throughput — so adding a method requires no changes to
the orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from pydantic_settings import BaseSettings

from cuperiod.core._arrayapi import TORCH_DEVICES
from cuperiod.core.backend import (
    available_backends,
    cuda_available,
    torch_available,
    torch_devices,
    torch_gpu_available,
)
from cuperiod.core.columns import Domain
from cuperiod.core.errors import BackendUnavailableError, UnknownMethodError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import ObjectiveSense, Periodogram


class PeriodogramMethod(ABC):
    """Abstract base for a periodogram method.

    Subclasses set the class attributes and implement :meth:`default_grid` and
    :meth:`power`. Multi-band-capable methods also set ``supports_multiband`` and
    implement :meth:`multiband_power`.
    """

    #: Registry name (canonical uppercase, e.g. ``"GLS"``).
    name: ClassVar[str]
    #: Whether peaks are maxima or minima of the statistic.
    objective_sense: ClassVar[ObjectiveSense]
    #: Whether the method has a multi-band variant.
    supports_multiband: ClassVar[bool] = False
    #: The domain inputs are converted to before computing. ``None`` means the method
    #: is domain-agnostic (magnitude or flux used as supplied); ``Domain.FLUX`` forces
    #: a flux conversion (box/transit methods, where a dip is the signal).
    natural_domain: ClassVar[Domain | None] = None
    #: The settings model class for this method.
    settings_cls: ClassVar[type[BaseSettings]]
    #: Best always-available CPU backend name.
    cpu_backend: ClassVar[str]
    #: Preferred CPU backend when its optional dependency is installed (the multicore
    #: ``numba`` kernels of the ``[fast]`` extra), or ``None`` to always use
    #: ``cpu_backend``.
    fast_cpu_backend: ClassVar[str | None] = None
    #: NVIDIA fast-path GPU backend name (cufinufft / cupy), or ``None`` if the method
    #: has no CUDA path.
    gpu_backend: ClassVar[str | None] = None
    #: Portable GPU backend name (``"torch"``) reaching AMD/Intel/Mac/CPU, or ``None``
    #: if the method has not been ported to the array-API path yet.
    portable_gpu_backend: ClassVar[str | None] = None
    #: Every backend this method can run.
    all_backends: ClassVar[tuple[str, ...]]

    # -- settings ----------------------------------------------------------------
    def coerce_settings(self, settings: BaseSettings | None) -> BaseSettings:
        """Return a settings instance of this method's type (default if ``None``)."""
        if settings is None:
            return self.settings_cls()
        if not isinstance(settings, self.settings_cls):
            raise TypeError(
                f"{self.name} expects {self.settings_cls.__name__}, "
                f"got {type(settings).__name__}"
            )
        return settings

    # -- backend resolution ------------------------------------------------------
    def resolve_backend(self, requested: str) -> str:
        """Resolve ``auto``/``cpu``/``gpu``/concrete to a runnable backend name.

        ``"auto"`` prefers the NVIDIA fast-path (``gpu_backend``) when a CUDA device is
        present, then the portable ``torch`` backend when torch sees a non-CPU device
        (AMD/Intel/Mac), and otherwise the proven CPU path. ``"gpu"`` is the same but
        raises when no GPU is available. A concrete ``"torch"`` / ``"torch:<device>"``
        request selects the portable path explicitly.

        Parameters
        ----------
        requested : str
            ``"auto"``, ``"cpu"``, ``"gpu"``, ``"torch"``, ``"torch:cpu|cuda|mps|xpu"``,
            or a concrete backend name belonging to this method.

        Returns
        -------
        str
            A concrete, available backend name (``"torch:<device>"`` kept as given).

        Raises
        ------
        BackendUnavailableError
            If GPU/torch was requested but is unavailable, or a named backend is unknown
            to this method or not importable here.
        """
        available = available_backends()
        if requested == "auto":
            if self.gpu_backend is not None and cuda_available():
                return self.gpu_backend
            if self.portable_gpu_backend is not None and torch_gpu_available():
                return self.portable_gpu_backend
            return self._best_cpu_backend(available)
        if requested == "cpu":
            return self._best_cpu_backend(available)
        if requested == "gpu":
            if self.gpu_backend is not None and cuda_available():
                return self.gpu_backend
            if self.portable_gpu_backend is not None and torch_gpu_available():
                return self.portable_gpu_backend
            raise BackendUnavailableError(
                f"{self.name}: no GPU backend available (need the [gpu] extra and a "
                "CUDA device, or the [torch] extra and a CUDA/ROCm/MPS/XPU device)"
            )
        if requested == "torch" or requested.startswith("torch:"):
            return self._resolve_torch(requested)
        if requested not in self.all_backends:
            raise BackendUnavailableError(
                f"{self.name}: unknown backend {requested!r}; "
                f"choose from {self.all_backends}"
            )
        if requested == self.gpu_backend and not cuda_available():
            raise BackendUnavailableError(
                f"{self.name}: backend {requested!r} needs a CUDA device"
            )
        if requested in {"finufft", "cufinufft", "numba", "astropy"} and (
            requested not in available
        ):
            raise BackendUnavailableError(
                f"{self.name}: backend {requested!r} is not installed"
            )
        return requested

    def _best_cpu_backend(self, available: set[str]) -> str:
        """``fast_cpu_backend`` when its dependency is present, else ``cpu_backend``."""
        if self.fast_cpu_backend is not None and self.fast_cpu_backend in available:
            return self.fast_cpu_backend
        return self.cpu_backend

    def _resolve_torch(self, requested: str) -> str:
        """Validate a concrete ``torch``/``torch:<device>`` request for this method."""
        if self.portable_gpu_backend != "torch":
            raise BackendUnavailableError(
                f"{self.name}: no portable 'torch' backend for this method"
            )
        if not torch_available():
            raise BackendUnavailableError(
                f"{self.name}: backend 'torch' needs the [torch] extra (pip install "
                "'cuperiod[torch]')"
            )
        if ":" in requested:
            device = requested.split(":", 1)[1]
            if device not in TORCH_DEVICES:
                raise BackendUnavailableError(
                    f"{self.name}: unknown torch device {device!r}; "
                    f"choose from {TORCH_DEVICES}"
                )
            if device not in torch_devices():
                raise BackendUnavailableError(
                    f"{self.name}: torch device {device!r} is not available here"
                )
        return requested

    def is_gpu_backend(self, backend: str) -> bool:
        """Whether ``backend`` runs on a GPU (NVIDIA fast-path or non-CPU torch)."""
        if backend == self.gpu_backend:
            return True
        if backend == "torch" or backend.startswith("torch:"):
            return backend != "torch:cpu"
        return False

    # -- compute -----------------------------------------------------------------
    @abstractmethod
    def default_grid(self, lc: LightCurve, settings: BaseSettings) -> GridSpec:
        """Build the default trial grid for ``lc`` under ``settings``."""

    @abstractmethod
    def power(
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: BaseSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        """Compute the periodogram on ``grid`` using ``backend``."""

    def multiband_power(
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: BaseSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        """Compute a multi-band periodogram. Override in multi-band methods.

        ``engine`` is the same reusable object :meth:`make_engine` builds for the
        single-band path; methods without a reusable plan simply ignore it.
        """
        raise NotImplementedError(f"{self.name} does not support multi-band input")

    def make_engine(self, backend: str, settings: BaseSettings) -> object | None:
        """Build a reusable plan/kernel engine for batch runs (``None`` for CPU)."""
        return None

    def estimate_device_bytes(self, n_points: int) -> int:
        """Estimate one worker's GPU footprint (bytes) for an ``n_points`` grid."""
        return 64 * 1024**2 + n_points * 16 * 6


# --- registry ----------------------------------------------------------------

_REGISTRY: dict[str, PeriodogramMethod] = {}


@dataclass(frozen=True)
class MethodInfo:
    """Introspection record for a registered method."""

    name: str
    objective_sense: str
    supports_multiband: bool
    natural_domain: str
    available_backends: tuple[str, ...]
    all_backends: tuple[str, ...]


def _normalize_name(name: str) -> str:
    """Canonical registry key: uppercase with non-alphanumerics removed.

    Lets ``"String-Length"``, ``"StringLength"`` and ``"STRINGLENGTH"`` (and ``"gls"`` /
    ``"GLS"``) all resolve to the same method.
    """
    return "".join(ch for ch in name if ch.isalnum()).upper()


def register(method: PeriodogramMethod) -> PeriodogramMethod:
    """Register ``method`` under its normalized name. Returns it (for decoration)."""
    _REGISTRY[_normalize_name(method.name)] = method
    return method


def get_method(name: str) -> PeriodogramMethod:
    """Look up a registered method by name.

    Matching ignores case and any non-alphanumeric characters, so ``"String-Length"``,
    ``"StringLength"`` and ``"STRINGLENGTH"`` all resolve to the same method.

    Raises
    ------
    UnknownMethodError
        If no method is registered under ``name``.
    """
    try:
        return _REGISTRY[_normalize_name(name)]
    except KeyError:
        raise UnknownMethodError(
            f"unknown method {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def method_names() -> list[str]:
    """Sorted names of all registered methods."""
    return sorted(_REGISTRY)


def list_methods() -> list[MethodInfo]:
    """Introspection records for every registered method."""
    available = available_backends()
    out: list[MethodInfo] = []
    for name in sorted(_REGISTRY):
        m = _REGISTRY[name]
        out.append(
            MethodInfo(
                name=name,
                objective_sense=m.objective_sense,
                supports_multiband=m.supports_multiband,
                natural_domain=str(m.natural_domain) if m.natural_domain else "any",
                available_backends=tuple(
                    b for b in m.all_backends if b in available or b == "numpy"
                ),
                all_backends=m.all_backends,
            )
        )
    return out


def registered_methods() -> Mapping[str, PeriodogramMethod]:
    """The live registry (read-only view)."""
    return dict(_REGISTRY)


__all__ = [
    "MethodInfo",
    "PeriodogramMethod",
    "get_method",
    "list_methods",
    "method_names",
    "register",
    "registered_methods",
]
