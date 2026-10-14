"""The method interface and registry.

Every periodogram is a :class:`PeriodogramMethod` subclass registered under an
uppercase name (``"GLS"``, ``"BLS"``, ...). The single-run, batch, and CLI layers only
ever go through this interface — ``default_grid`` to pick a trial grid, ``power`` to
compute a :class:`~cuperiod.core.result.Periodogram`, and ``make_engine`` to build a
reusable plan/kernel for batch throughput — so adding a method requires no changes to
the orchestration.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeVar, cast

from pydantic_settings import BaseSettings

from cuperiod.core._arrayapi import TORCH_DEVICES
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import (
    BACKEND_SELECTORS,
    available_backends,
    cpu_backends,
    cuda_available,
    normalize_backend,
    requested_backend,
    torch_available,
    torch_devices,
    torch_gpu_available,
)
from cuperiod.core.columns import Domain
from cuperiod.core.errors import BackendUnavailableError, UnknownMethodError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import ObjectiveSense, Periodogram

#: A single method's settings, a ``{method: settings}`` mapping, or ``None`` (defaults).
SettingsInput = BaseSettings | Mapping[str, BaseSettings] | None

_S = TypeVar("_S", bound=BaseSettings)

#: Per-class default-settings template, keyed by the environment it was built under.
_DEFAULT_SETTINGS: dict[
    type[BaseSettings], tuple[tuple[tuple[str, str], ...], BaseSettings]
] = {}

#: Concrete CPU backend names, resolvable without probing for a GPU.
_CPU_BACKEND_NAMES = frozenset({"finufft", "numba", "astropy", "numpy"})


def default_settings(cls: type[_S]) -> _S:
    """A default-constructed ``cls()``, without re-reading the environment every call.

    pydantic-settings re-parses the environment and validates every field on each
    construction (0.3-0.5 ms) — a visible share of a small periodogram when no
    settings are passed. A template is cached per class together with the
    ``<env_prefix>*`` variables it was built from, and a copy is returned while those
    variables are unchanged, so ``CUPERIOD_<METHOD>_<FIELD>`` overrides set, changed
    or removed mid-session still apply (and an invalid one still raises). Classes that
    read a dotenv or secrets directory bypass the cache.
    """
    config = cls.model_config
    if config.get("env_file") or config.get("secrets_dir"):
        return cls()
    prefix = str(config.get("env_prefix") or "").upper()
    env_key = tuple(
        sorted((k, v) for k, v in os.environ.items() if k.upper().startswith(prefix))
    )
    cached = _DEFAULT_SETTINGS.get(cls)
    if cached is not None and cached[0] == env_key:
        template = cast(_S, cached[1])
    else:
        template = cls()
        _DEFAULT_SETTINGS[cls] = (env_key, template)
    # Every settings field is a scalar or a tuple, so a shallow copy is independent.
    return template.model_copy()


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
            return default_settings(self.settings_cls)
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
        present *and* that backend's package is installed, then the portable ``torch``
        backend when torch sees a non-CPU device (AMD/Intel/Mac), and otherwise the
        proven CPU path. ``"gpu"`` is the same but raises when no GPU is available. A
        concrete ``"torch"`` / ``"torch:<device>"`` request selects the portable path
        explicitly. Selectors ignore case and surrounding whitespace (``"CPU"`` is
        ``"cpu"``), like method names.

        The installed-package check matters: ``cuda_available()`` only reports that
        cupy sees a device, while a method's fast path may need another library
        (GLS needs cufinufft). Without it, an environment with cupy and a GPU but no
        cufinufft would resolve ``"auto"`` to a backend that raises ``ImportError``
        deep in the compute path instead of quietly using the CPU.

        ``"cpu"`` and the concrete CPU names are answered from the CPU-only probe
        (:func:`~cuperiod.core.backend.cpu_backends`), so an explicit CPU request never
        imports cupy or touches the CUDA driver.

        Parameters
        ----------
        requested : str
            ``"auto"``, ``"cpu"``, ``"gpu"``, ``"torch"``, ``"torch:cpu|cuda|mps|xpu"``,
            or a concrete backend name belonging to this method.

        Returns
        -------
        str
            A concrete, available backend name (``"torch:<device>"`` kept as given,
            lower-cased).

        Raises
        ------
        BackendUnavailableError
            If GPU/torch was requested but is unavailable, or a named backend is unknown
            to this method or not importable here.
        """
        requested = normalize_backend(requested)
        if requested == "cpu":
            return self._best_cpu_backend(cpu_backends())
        if requested == "torch" or requested.startswith("torch:"):
            return self._resolve_torch(requested)
        if requested not in BACKEND_SELECTORS and requested not in self.all_backends:
            raise BackendUnavailableError(
                f"{self.name}: unknown backend {requested!r}; choose "
                f"{self._selector_help()}"
            )
        if requested in _CPU_BACKEND_NAMES:
            if requested != "numpy" and requested not in cpu_backends():
                raise BackendUnavailableError(
                    f"{self.name}: backend {requested!r} is not installed"
                )
            return requested
        available = available_backends()
        if requested == "auto":
            if self._cuda_path_ready(available):
                return str(self.gpu_backend)
            if self.portable_gpu_backend is not None and torch_gpu_available():
                return self.portable_gpu_backend
            return self._best_cpu_backend(available)
        if requested == "gpu":
            if self._cuda_path_ready(available):
                return str(self.gpu_backend)
            if self.portable_gpu_backend is not None and torch_gpu_available():
                return self.portable_gpu_backend
            extra = ""
            if self.gpu_backend is not None and cuda_available():
                extra = (
                    f" (a CUDA device is present, but {self.gpu_backend!r} is not "
                    "installed)"
                )
            raise BackendUnavailableError(
                f"{self.name}: no GPU backend available{extra} (need the [gpu] extra "
                "and a CUDA device, or the [torch] extra and a CUDA/ROCm/MPS/XPU "
                "device)"
            )
        if requested == self.gpu_backend and not cuda_available():
            raise BackendUnavailableError(
                f"{self.name}: backend {requested!r} needs a CUDA device"
            )
        if requested == "cufinufft" and requested not in available:
            raise BackendUnavailableError(
                f"{self.name}: backend {requested!r} is not installed"
            )
        return requested

    def _selector_help(self) -> str:
        """The valid selectors, for an unknown-backend message."""
        generic = ", ".join(repr(s) for s in BACKEND_SELECTORS)
        if self.portable_gpu_backend == "torch":
            generic += ", 'torch', 'torch:<device>'"
        return f"{generic}, or one of this method's backends {self.all_backends}"

    def resolve_backend_for(
        self, requested: str | None, settings: BaseSettings | None
    ) -> str:
        """Resolve a run's backend: the explicit request, else ``settings.backend``.

        The single place every driver (single-curve API, batch, period-change scans,
        multi-band FAP, LINCC adapter) should resolve through, so the documented
        ``backend`` settings field and its ``CUPERIOD_<METHOD>_BACKEND`` environment
        override take effect. ``None`` or ``"auto"`` defers to ``settings.backend``;
        any other request wins (see
        :func:`~cuperiod.core.backend.requested_backend`).

        Parameters
        ----------
        requested : str or None
            The caller's ``backend=`` argument.
        settings : settings model or None
            This method's settings.

        Returns
        -------
        str
            A concrete, available backend name.
        """
        return self.resolve_backend(requested_backend(requested, settings))

    def _cuda_path_ready(self, available: set[str]) -> bool:
        """Whether this method's NVIDIA fast path can actually run here.

        Both halves are needed: a CUDA device (``cuda_available``) *and* the
        backend's own package in ``available_backends()``.
        """
        if self.gpu_backend is None:
            return False
        return self.gpu_backend in available and cuda_available()

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

    def is_gpu_backend(
        self, backend: str, settings: BaseSettings | None = None
    ) -> bool:
        """Whether ``backend`` runs on a GPU (NVIDIA fast-path or non-CPU torch).

        A device-qualified ``"torch:<device>"`` is a GPU unless the device is ``cpu``.
        Bare ``"torch"`` runs on ``settings.device`` when that pins a device, else on
        the best device torch sees — so it is a GPU only when torch actually has one
        (a CPU-only torch install runs it on the CPU).
        """
        backend = normalize_backend(backend)
        if backend == self.gpu_backend:
            return True
        if backend.startswith("torch:"):
            return backend != "torch:cpu"
        if backend == "torch":
            device = str(getattr(settings, "device", "auto"))
            if device != "auto":
                return device != "cpu"
            return torch_gpu_available()
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

    def supports_power_rows(self, grid: GridSpec, backend: str) -> bool:
        """Whether :meth:`power_rows` has a batched path for ``grid`` on ``backend``.

        Lets a caller skip building the rows' times when the answer is no.
        """
        return False

    def power_rows(
        self,
        grid: GridSpec,
        lc: LightCurve,
        times: FloatArray,
        settings: BaseSettings,
        backend: str,
        engine: object | None = None,
    ) -> FloatArray | None:
        """The statistic of many re-timed copies of one light curve, or ``None``.

        Row ``r`` of the ``(n_rows, grid.size)`` result equals
        ``self.power(grid, replace(lc, time=times[r]), settings, backend,
        engine).power`` -- the values a period-derivative scan reads, in ascending
        frequency. ``lc`` must already be finite (``lc.finite()``) and ``times`` hold
        one finite time array per row, aligned with it. Methods with a batched
        device path override this (GLS on cufinufft, PDM on cupy) so the rows share
        uploads and one download; ``None`` -- the default -- tells the caller to loop
        over :meth:`power` instead.
        """
        return None

    def make_engine(self, backend: str, settings: BaseSettings) -> object | None:
        """Build a reusable plan/kernel engine for repeated calls on ``backend``.

        ``None`` when the method has none for that backend. GLS returns one for
        ``"cufinufft"`` and also for the CPU ``"finufft"`` backend (plan reuse for
        scans and batches). An engine is not thread-safe: use one per thread or
        process.
        """
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


def select_settings(
    methods: Sequence[PeriodogramMethod], settings: SettingsInput
) -> dict[str, BaseSettings]:
    """Pick each requested method's settings from a ``settings=`` argument.

    The one rule shared by :func:`cuperiod.periodogram` and the batch drivers:

    * ``None`` — every method gets its defaults.
    * A single settings object — the methods whose ``settings_cls`` it is use it and
      the others take defaults (so ``periodogram(lc, ["GLS", "BLS"],
      settings=GLSSettings(...))`` tunes GLS only). It must match at least one
      requested method: silently running defaults for a mismatched object hid typos.
    * A ``{method: settings}`` mapping — keys are matched like method names (case and
      separators ignored: ``"String-Length"`` is ``STRINGLENGTH``). A key naming no
      registered method raises, as do two keys naming the same method; keys for
      registered methods not in this run are ignored, so one shared mapping can serve
      several calls.

    Parameters
    ----------
    methods : sequence of PeriodogramMethod
        The methods of this run.
    settings : settings model, mapping, or None
        The caller's ``settings=`` argument.

    Returns
    -------
    dict
        ``{method.name: settings}`` for every requested method.

    Raises
    ------
    TypeError
        If a settings object (bare or a mapping value) is of the wrong class.
    UnknownMethodError
        If a mapping key names no registered method.
    ValueError
        If two mapping keys name the same method.
    """
    if settings is None:
        return {m.name: m.coerce_settings(None) for m in methods}
    if isinstance(settings, BaseSettings):
        if not any(isinstance(settings, m.settings_cls) for m in methods):
            expected = " or ".join(
                dict.fromkeys(m.settings_cls.__name__ for m in methods)
            )
            names = ", ".join(dict.fromkeys(m.name for m in methods))
            raise TypeError(
                f"settings: {names} expects {expected}, got {type(settings).__name__}"
                " (pass a {method: settings} mapping to tune several methods)"
            )
        return {
            m.name: (
                settings
                if isinstance(settings, m.settings_cls)
                else m.coerce_settings(None)
            )
            for m in methods
        }
    if not isinstance(settings, Mapping):
        raise TypeError(
            "settings must be a settings model, a {method: settings} mapping, or "
            f"None; got {type(settings).__name__}"
        )
    by_method: dict[str, tuple[str, BaseSettings | None]] = {}
    for key, value in settings.items():
        method = get_method(str(key))
        if method.name in by_method:
            raise ValueError(
                f"settings: keys {by_method[method.name][0]!r} and {key!r} both name "
                f"method {method.name}"
            )
        by_method[method.name] = (str(key), value)
    return {
        m.name: m.coerce_settings(by_method[m.name][1] if m.name in by_method else None)
        for m in methods
    }


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
    "SettingsInput",
    "default_settings",
    "get_method",
    "list_methods",
    "method_names",
    "register",
    "registered_methods",
    "select_settings",
]
