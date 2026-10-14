"""The numerical core a session's reducers code against: :class:`SessionKernel`.

The kernel owns the session's data, its spectrum engines (one per sampling, in a
small LRU) and the cache of :class:`~cuperiod.prewhiten.session.state.Derived`
companions, and it performs every fit with exactly the arguments
:func:`cuperiod.prewhiten` uses. A state with neither constraints nor data edits
("plain" — every state of an automatic run) takes the v1.3 code path call for call:
the same arrays, the same ``GridSpec`` object, the same keywords, so its numbers do
not move. Only a state that needs them reaches the sampling and model-map modules,
imported lazily.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.grid import GridSpec, check_grid_size
from cuperiod.prewhiten.engine import _zero_model_fit
from cuperiod.prewhiten.fit import (
    _ROUNDOFF_EPS,
    MultiSineFit,
    Refinement,
    fit_multisine,
)
from cuperiod.prewhiten.session.state import (
    CreationArgs,
    Derived,
    FitStats,
    PointMask,
    SessionData,
    SessionError,
    SessionState,
    UsedData,
)
from cuperiod.prewhiten.spectrum import (
    AmplitudeSpectrum,
    SpectrumEngine,
    _SpectrumReadout,
)

if TYPE_CHECKING:
    from cuperiod.prewhiten.session.modelmap import ModelMap

_TWO_PI = 2.0 * np.pi

#: Data spectra with subset zero points removed kept per sampling.
_OFFSET_SCANS = 4


@dataclass(frozen=True, eq=False)
class ExactUniformGrid(GridSpec):
    """A uniform frequency grid that reports exactly the ``(f0, df, nf)`` it was given.

    Rebuilding a grid from its parameters through its sample values loses the last
    bit of ``df`` (``(f0 + df) - f0 != df``), and the spectrum engine regrids from
    those parameters; a session restored or re-gridded from a stored ``(f0, df, nf)``
    must evaluate exactly the frequencies it did before.
    """

    params: tuple[float, float, int] = (0.0, 0.0, 0)

    @classmethod
    def of(cls, params: tuple[float, float, int]) -> ExactUniformGrid:
        """The grid ``f0 + df * arange(nf)``, checked before anything is allocated.

        Raises
        ------
        ValueError
            Unless ``f0 >= 0`` and ``df > 0`` are finite and ``nf >= 1``.
        GridTooLargeError
            When ``nf`` exceeds :data:`~cuperiod.core.grid.MAX_GRID_SIZE`: a
            ``(f0, df, nf)`` handed to the session constructor used to allocate
            whatever it asked for (``nf = 10**12`` is 8 TB), skipping the size guard
            every other grid builder applies.
        """
        f0, df, nf = float(params[0]), float(params[1]), int(params[2])
        if not (np.isfinite(f0) and f0 >= 0.0 and np.isfinite(df) and df > 0.0):
            raise ValueError(
                f"a uniform grid needs finite f0 >= 0 and df > 0, got f0={f0!r}, "
                f"df={df!r}"
            )
        if nf < 1:
            raise ValueError(f"a uniform grid needs nf >= 1 samples, got {nf}")
        check_grid_size(
            nf,
            maximum_frequency=f0 + df * (nf - 1),
            context="pre-whitening session grid",
        )
        values = f0 + df * np.arange(nf, dtype=np.float64)
        return cls(kind="frequency", values=values, uniform=True, params=(f0, df, nf))

    def uniform_frequency_params(self) -> tuple[float, float, int]:
        return self.params


@dataclass(eq=False)
class _Bundle:
    """One sampling's spectrum engine and what depends only on it."""

    key: str
    engine: SpectrumEngine
    used: UsedData
    weight_sum: float
    roundoff: float
    targets: dict[bool, tuple[float, FloatArray]] = field(default_factory=dict)
    scans: dict[Any, _SpectrumReadout] = field(default_factory=dict)
    spectra: dict[Any, AmplitudeSpectrum] = field(default_factory=dict)
    window: AmplitudeSpectrum | None = None


def _requested_backend(concrete: str) -> str:
    """The backend name to build another engine like one that ran ``concrete``.

    A torch engine reports ``torch:<device>``; the device comes from the settings.
    """
    return "torch" if concrete.startswith("torch") else concrete


class SessionKernel:
    """Data, engines, fits and the derived cache of one session.

    Build with :meth:`open`. Reducers receive the kernel, the current state and its
    :class:`Derived`, and return a new state through :meth:`commit_fit` (or
    ``dataclasses.replace`` for edits that fit nothing).
    """

    data: SessionData
    creation: CreationArgs

    def __init__(
        self,
        data: SessionData,
        creation: CreationArgs,
        *,
        grid: GridSpec | None = None,
        backend_request: str | None = None,
    ) -> None:
        self.data = data
        self.creation = creation
        self.t_ref = float(creation.t_ref)
        self.backend = creation.backend
        self.session_settings = creation.session_settings
        self._backend_request = backend_request or _requested_backend(creation.backend)
        self._grids: dict[tuple[float, float, int], GridSpec] = {}
        if grid is not None:
            self.register_grid(grid)
        self._bundles: OrderedDict[str, _Bundle] = OrderedDict()
        self._derived: OrderedDict[int, Derived] = OrderedDict()
        self._used_cache: OrderedDict[Any, UsedData] = OrderedDict()
        self._full = UsedData.full(data)
        self._serial = 0

    # -- construction ------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        data: SessionData,
        creation: CreationArgs,
        *,
        grid: GridSpec | None = None,
        engine: SpectrumEngine | None = None,
        backend_request: str | None = None,
    ) -> tuple[SessionKernel, SessionState, Derived]:
        """The v1.3 prologue: engine, offset-only fit, root state and its Derived.

        Parameters
        ----------
        data : SessionData
        creation : CreationArgs
        grid : GridSpec, optional
            The grid object the engine was (or is to be) built on; defaults to the
            exact grid of ``creation.grid``.
        engine : SpectrumEngine, optional
            An engine already built on the full data and ``grid`` (the constructor
            builds it first, in v1.3 order, so its errors surface first).
        backend_request : str, optional
            The backend name to build further engines with.
        """
        kernel = cls(data, creation, grid=grid, backend_request=backend_request)
        cfg = creation.settings
        root = SessionState(
            serial=kernel.next_serial(),
            components=(),
            offset=0.0,
            subset_offsets=None,
            stats=FitStats(0.0, 0.0, 0.0, data.n, 0, data.error is not None),
            mask=PointMask.none(data.n),
            partitions=tuple(creation.partitions),
            excluded=(),
            offsets_by=None,
            variation_by=None,
            weights_by=None,
            settings=cfg.model_copy(),
            grid=creation.grid,
            blocked=(),
            n_iterations=0,
            budget_origin=0,
            n_pruned=0,
            status="searching",
            stop_reason="",
            next_id=1,
        )
        if engine is not None:
            kernel._install(root, engine)
        fit = kernel.fit(root, np.zeros(0), refine="none", covariance=False)
        root = replace(
            root,
            offset=float(fit.offset) + kernel.zero_point(root),
            stats=FitStats.of(fit),
        )
        derived = Derived(
            fit=fit, scan=kernel.data_scan(root), sampling_key=kernel.sampling_key(root)
        )
        kernel.remember(root, derived)
        return kernel, root, derived

    def next_serial(self) -> int:
        """A fresh state serial."""
        self._serial += 1
        return self._serial

    def reserve_serials(self, at_least: int) -> None:
        """Make every later serial exceed ``at_least`` (after a restore)."""
        self._serial = max(self._serial, int(at_least))

    # -- grids and samplings -----------------------------------------------------------
    def register_grid(self, grid: GridSpec) -> tuple[float, float, int]:
        """Keep ``grid`` for states whose ``grid`` is its parameters; return them."""
        params = grid.uniform_frequency_params()
        key = (float(params[0]), float(params[1]), int(params[2]))
        self._grids[key] = grid
        return key

    def grid_spec(self, state: SessionState) -> GridSpec:
        """The grid object of ``state.grid`` (the original one when it is known)."""
        key = (float(state.grid[0]), float(state.grid[1]), int(state.grid[2]))
        grid = self._grids.get(key)
        if grid is None:
            grid = ExactUniformGrid.of(key)
            self._grids[key] = grid
        return grid

    def sampling_key(self, state: SessionState) -> str:
        """Identifies the engine a state needs: points, weights, grid and settings."""
        cfg = state.settings
        h = hashlib.sha1(state.mask.digest.encode())
        h.update(repr(tuple((n, tuple(s)) for n, s in state.excluded if s)).encode())
        if state.weights_by is not None:
            partition = state.partition(state.weights_by)
            h.update(repr((partition.name, partition.weights)).encode())
            h.update(partition.codes.tobytes())
        h.update(
            repr(
                (
                    tuple(state.grid),
                    cfg.normalization,
                    self._backend_request,
                    cfg.device,
                    cfg.precision,
                    cfg.nufft_eps,
                    cfg.direct_freq_batch,
                    self.t_ref,
                )
            ).encode()
        )
        return h.hexdigest()

    def used(self, state: SessionState) -> UsedData:
        """The points (weights, group codes) a fit of ``state`` uses."""
        if state.is_plain():
            return self._full
        key = (
            self.sampling_key(state),
            _partition_key(state, state.offsets_by),
            _partition_key(state, state.variation_by),
        )
        cached = self._used_cache.get(key)
        if cached is not None:
            self._used_cache.move_to_end(key)
            return cached
        from cuperiod.prewhiten.session.sampling import used_data

        used = used_data(self.data, state)
        self._used_cache[key] = used
        while len(self._used_cache) > 8:
            self._used_cache.popitem(last=False)
        return used

    def _engine_cache_size(self) -> int:
        on_gpu = self.backend == "cufinufft" or (
            self.backend.startswith("torch:") and not self.backend.endswith("cpu")
        )
        return 1 if on_gpu else max(1, int(self.session_settings.engine_cache))

    def _install(self, state: SessionState, engine: SpectrumEngine) -> _Bundle:
        """Wrap an engine for ``state``'s sampling into the LRU."""
        used = self.used(state)
        # The fit's weights: what its weighted RSS is a sum over (see stepper.attempt).
        weight = (
            np.ones(used.n) if used.error is None else 1.0 / (used.error * used.error)
        )
        if used.point_weights is not None:
            weight = weight * used.point_weights
        bundle = _Bundle(
            key=self.sampling_key(state),
            engine=engine,
            used=used,
            weight_sum=float(weight.sum()),
            roundoff=_ROUNDOFF_EPS
            * float(np.finfo(np.float64).eps)
            * float(np.max(np.abs(used.value))),
        )
        self._bundles[bundle.key] = bundle
        self._bundles.move_to_end(bundle.key)
        while len(self._bundles) > self._engine_cache_size():
            _, dropped = self._bundles.popitem(last=False)
            dropped.engine.release()
        return bundle

    def _bundle(self, state: SessionState) -> _Bundle:
        key = self.sampling_key(state)
        bundle = self._bundles.get(key)
        if bundle is not None:
            self._bundles.move_to_end(key)
            return bundle
        used = self.used(state)
        if used.n < 3 or not used.time.size:
            raise SessionError("too few points in use to build a spectrum")
        cfg = state.settings
        engine = SpectrumEngine(
            used.time,
            used.error_eff,
            grid=self.grid_spec(state),
            normalization=cfg.normalization,
            backend=self._backend_request,
            device=cfg.device,
            precision=cfg.precision,
            eps=cfg.nufft_eps,
            freq_batch=cfg.direct_freq_batch,
            t_ref=self.t_ref,
        )
        return self._install(state, engine)

    def engine(self, state: SessionState) -> SpectrumEngine:
        """The spectrum engine of ``state``'s sampling (LRU; rebuilt on a miss)."""
        return self._bundle(state).engine

    def release_engines(self) -> None:
        """Drop the device-side state of every cached engine."""
        for bundle in self._bundles.values():
            bundle.engine.release()

    # -- per-sampling quantities -------------------------------------------------------
    def baseline(self, state: SessionState) -> float:
        """Time span of the points in use (the Rayleigh resolution is its inverse)."""
        time = self.used(state).time
        return float(time.max() - time.min()) if time.size else 0.0

    def separation(self, state: SessionState) -> float:
        """The resolution guard between components (cycles/day)."""
        return state.settings.min_separation_rayleigh / self.baseline(state)

    def roundoff(self, state: SessionState) -> float:
        """Amplitudes at or below this are rounding noise of the data's scale."""
        return self._bundle(state).roundoff

    def weight_sum(self, state: SessionState) -> float:
        """Sum of the fit weights of the points in use (``N`` without error bars)."""
        return self._bundle(state).weight_sum

    def f_max(self, state: SessionState) -> float:
        """Top of the frequency grid."""
        return float(self.engine(state).frequency[-1])

    def bounds(
        self, state: SessionState, frequencies: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Per-frequency refinement box: at most one bound width from where it started.

        Without this a refinement started on a modest peak can slide down the wing of
        a much stronger neighbour and land on top of it, producing near-identical
        frequencies with enormous, mutually cancelling amplitudes.
        """
        engine = self.engine(state)
        f_min, f_max = float(engine.frequency[0]), float(engine.frequency[-1])
        grid_low, grid_high = max(0.5 * f_min, 1e-12), f_max
        max_shift = state.settings.refine_bound_rayleigh / self.baseline(state)
        low = np.maximum(frequencies - max_shift, grid_low)
        high = np.minimum(frequencies + max_shift, grid_high)
        return low, np.maximum(high, np.nextafter(low, np.inf))

    def zero_point(self, state: SessionState) -> float:
        """The weighted mean removed up front when ``fit_mean`` is off (else 0)."""
        return self._target(state)[0]

    def _target(self, state: SessionState) -> tuple[float, FloatArray]:
        """``(zero_point, value - zero_point)`` of the points in use.

        Without a free offset the fits and the spectrum must still share a zero point:
        the spectrum always works on the weighted-mean-subtracted values, so the fits
        do too, and the mean is added back to the reported offset.
        """
        bundle = self._bundle(state)
        fit_mean = bool(state.settings.fit_mean)
        cached = bundle.targets.get(fit_mean)
        if cached is None:
            used = bundle.used
            value, error = used.value, used.error
            weight = np.ones(used.n) if error is None else 1.0 / (error * error)
            if used.point_weights is not None:
                weight = weight * used.point_weights
            zero_point = (
                0.0 if fit_mean else float(np.dot(weight, value) / weight.sum())
            )
            target = value - zero_point if zero_point else value
            cached = (zero_point, target)
            bundle.targets[fit_mean] = cached
        return cached

    # -- fits --------------------------------------------------------------------------
    def model_map(
        self,
        state: SessionState,
        *,
        extra_free: int = 0,
        hold: Collection[int] = frozenset(),
    ) -> ModelMap | None:
        """``None`` for a plain state (the v1.3 fit); otherwise the model map."""
        if state.is_plain() and not hold:
            return None
        from cuperiod.prewhiten.session.modelmap import build

        return build(
            state,
            self.used(state),
            t_ref=self.t_ref,
            extra_free=extra_free,
            hold=hold,
            min_subset_points=self.session_settings.min_subset_points,
        )

    def fit(
        self,
        state: SessionState,
        frequencies: FloatArray,
        *,
        refine: Refinement,
        covariance: bool,
        mm: ModelMap | None = None,
    ) -> MultiSineFit:
        """Fit ``frequencies`` to the points ``state`` uses.

        ``fit_mean`` is ``settings.fit_mean`` for every call, as in v1.3; with no
        frequency and no free offset the model is empty (``y = 0`` on the centred
        values). Constraints are passed only when ``mm`` carries non-trivial ones, so
        a plain state's fit is the v1.3 call, keyword for keyword.
        """
        cfg = state.settings
        used = self._bundle(state).used
        _, target = self._target(state)
        freqs = np.asarray(frequencies, dtype=np.float64)
        constraints = mm.constraints if mm is not None else None
        if constraints is None:
            if freqs.size == 0:
                if not cfg.fit_mean:
                    return _zero_model_fit(used.time, target, used.error, self.t_ref)
                return fit_multisine(
                    used.time, target, used.error, np.zeros(0),
                    fit_mean=True, t_ref=self.t_ref, refine="none",
                    covariance=covariance,
                )
            if refine == "none":
                return fit_multisine(
                    used.time, target, used.error, freqs,
                    fit_mean=cfg.fit_mean, t_ref=self.t_ref, refine="none",
                    covariance=covariance,
                )
        return fit_multisine(
            used.time, target, used.error, freqs,
            fit_mean=cfg.fit_mean,
            t_ref=self.t_ref,
            refine=refine,
            sweeps=cfg.sweeps,
            frequency_bounds=self.bounds(state, freqs) if freqs.size else None,
            max_nfev=cfg.max_nfev,
            covariance=covariance,
            constraints=constraints,
        )

    def polish_refine(self, state: SessionState, n_components: int) -> Refinement:
        """The final polish policy: simultaneous up to ``max_simultaneous``."""
        return (
            "simultaneous"
            if n_components <= state.settings.max_simultaneous
            else "cyclic"
        )

    # -- spectra -----------------------------------------------------------------------
    def data_scan(self, state: SessionState) -> _SpectrumReadout:
        """Spectrum of the data in use, zero points removed, as the loop reads it."""
        bundle = self._bundle(state)
        key: Any = bool(state.settings.fit_mean)
        values: FloatArray | None = None
        if state.offsets_by is not None and state.subset_offsets is not None:
            # The bundle's own UsedData belongs to whichever state of this sampling
            # built the engine, and the zero-point role is not part of the sampling:
            # its group codes must come from this state.
            used = self.used(state)
            if used.offset_codes is not None and used.offset_subsets is not None:
                offsets = np.asarray(state.subset_offsets, dtype=np.float64)
                per_point = offsets[used.offset_subsets][used.offset_codes]
                key = ("offsets", state.offsets_by, state.subset_offsets)
                values = used.value - per_point
        cached = bundle.scans.get(key)
        if cached is None:
            if values is None:
                values = self._target(state)[1]
            cached = bundle.engine._scan(values)
            bundle.scans[key] = cached
            # Zero-point spectra are keyed by their offsets: keep only the newest few.
            stale = [k for k in bundle.scans if isinstance(k, tuple)][:-_OFFSET_SCANS]
            for old in stale:
                scan = bundle.scans.pop(old)
                bundle.spectra.pop(id(scan), None)
        return cached

    def data_spectrum(self, state: SessionState) -> AmplitudeSpectrum:
        """The full spectrum of :meth:`data_scan` (materialised once per sampling)."""
        scan = self.data_scan(state)
        if isinstance(scan, AmplitudeSpectrum):
            return scan
        bundle = self._bundle(state)
        key = id(scan)
        spectrum = bundle.spectra.get(key)
        if spectrum is None:
            spectrum = scan.to_spectrum()  # type: ignore[attr-defined]
            bundle.spectra[key] = spectrum
        return spectrum  # type: ignore[no-any-return]

    def window(self, state: SessionState) -> AmplitudeSpectrum:
        """The spectral window of the points in use."""
        bundle = self._bundle(state)
        if bundle.window is None:
            bundle.window = bundle.engine.window()
        return bundle.window

    # -- derived -----------------------------------------------------------------------
    def remember(self, state: SessionState, derived: Derived) -> None:
        """Cache ``derived`` for ``state`` (the newest ``derived_cache`` are kept)."""
        self._derived[state.serial] = derived
        self._derived.move_to_end(state.serial)
        while len(self._derived) > max(1, self.session_settings.derived_cache):
            self._derived.popitem(last=False)

    def forget(self) -> None:
        """Drop every cached Derived (tests use it to force rebuilds)."""
        self._derived.clear()

    def cached(self, state: SessionState) -> Derived | None:
        """The cached Derived of ``state``, if any."""
        return self._derived.get(state.serial)

    def derive(self, state: SessionState) -> Derived:
        """The Derived of ``state``: cached, or rebuilt from its parameters.

        With no active component (and no zero-point groups) the fit is the offset-only
        fit and the residual spectrum is the data spectrum itself. Otherwise the
        components are re-solved at their frequencies with ``refine="none"`` — for an
        unconstrained state this ends in the same design and solve as the fit that
        produced it, so it is bit-identical — and non-linear phases start from the
        stored values; any difference from the state is recorded as ``drift``.
        """
        derived = self._derived.get(state.serial)
        if derived is not None:
            self._derived.move_to_end(state.serial)
            return derived
        derived = self._rebuild(state)
        self.remember(state, derived)
        return derived

    def _rebuild(self, state: SessionState) -> Derived:
        mm = self.model_map(state)
        key = self.sampling_key(state)
        if not state.active and state.offsets_by is None:
            fit = self.fit(state, np.zeros(0), refine="none", covariance=False, mm=mm)
            return Derived(fit=fit, scan=self.data_scan(state), sampling_key=key)
        freqs = mm.frequencies if mm is not None else state.active_frequencies
        fit = self.fit(state, freqs, refine="none", covariance=False, mm=mm)
        scan = self.engine(state)._scan(fit.residuals)
        return Derived(fit=fit, scan=scan, sampling_key=key, drift=_drift(state, fit))

    def with_covariance(self, state: SessionState, derived: Derived) -> Derived:
        """``derived`` with a covariance (the fits of the loop skip it)."""
        if derived.fit.covariance.size:
            return derived
        mm = self.model_map(state)
        fit = self.fit(
            state, derived.fit.frequency, refine="none", covariance=True, mm=mm
        )
        out = Derived(
            fit=fit, scan=derived.scan, sampling_key=derived.sampling_key,
            drift=derived.drift,
        )
        if self._derived.get(state.serial) is derived:
            self._derived[state.serial] = out
        return out

    def commit_fit(
        self,
        state: SessionState,
        fit: MultiSineFit,
        *,
        mm: ModelMap | None = None,
        spectrum: _SpectrumReadout | None = None,
        **changes: Any,
    ) -> tuple[SessionState, Derived]:
        """A new state carrying ``fit`` (plus ``changes``) and its Derived.

        ``changes`` are applied first, so a new component passed in ``components``
        receives its fitted values here. Without a model map the fit's positions are
        the active components in table order.
        """
        current = replace(state, **changes) if changes else state
        if mm is None:
            active_ids = [c.id for c in current.active]
            if len(active_ids) != fit.n_components:
                raise SessionError(
                    f"fit of {fit.n_components} components for "
                    f"{len(active_ids)} active rows"
                )
            position = {cid: i for i, cid in enumerate(active_ids)}
            components = tuple(
                c
                if c.id not in position
                else replace(
                    c,
                    frequency=float(fit.frequency[position[c.id]]),
                    amplitude=float(fit.amplitude[position[c.id]]),
                    phase=float(fit.phase[position[c.id]]),
                )
                for c in current.components
            )
            offset, subset_offsets = float(fit.offset), current.subset_offsets
        else:
            from cuperiod.prewhiten.session.modelmap import apply

            update = apply(current, fit, mm)
            components, offset = update.components, update.offset
            subset_offsets = update.subset_offsets
        new = replace(
            current,
            serial=self.next_serial(),
            components=components,
            offset=offset + self.zero_point(current),
            subset_offsets=subset_offsets,
            stats=FitStats.of(fit),
        )
        scan = (
            spectrum
            if spectrum is not None
            else self.engine(new)._scan(fit.residuals)
        )
        derived = Derived(fit=fit, scan=scan, sampling_key=self.sampling_key(new))
        self.remember(new, derived)
        return new, derived

    def recalculate(
        self, state: SessionState, *, covariance: bool = False
    ) -> tuple[SessionState, Derived]:
        """Amplitudes, phases and zero points at held frequencies (Calculate).

        Parameters
        ----------
        state : SessionState
            The state to re-solve; its frequencies do not move.
        covariance : bool, default False
            Also solve for the parameter covariance, so the new Derived carries the
            errors a light view reads (fixed quantities get exactly 0). The explicit
            Calculate and Improve ask for it; the implicit Calculate after every
            edit does not, because it runs on each edit and its view only shows
            errors a covariance already at hand provides.
        """
        mm = self.model_map(state)
        if not state.active and state.offsets_by is None:
            fit = self.fit(
                state, np.zeros(0), refine="none", covariance=covariance, mm=mm
            )
            return self.commit_fit(state, fit, mm=mm, spectrum=self.data_scan(state))
        freqs = mm.frequencies if mm is not None else state.active_frequencies
        fit = self.fit(state, freqs, refine="none", covariance=covariance, mm=mm)
        return self.commit_fit(state, fit, mm=mm)


def _partition_key(state: SessionState, name: str | None) -> tuple[Any, ...] | None:
    """Content key of a role's partition (codes by digest, never by identity)."""
    if name is None:
        return None
    partition = state.partition(name)
    digest = hashlib.sha1(partition.codes.tobytes()).hexdigest()
    return (name, partition.labels, digest)


def _drift(state: SessionState, fit: MultiSineFit) -> float:
    """Largest relative difference between stored and rebuilt component values."""
    worst = 0.0
    for i, c in enumerate(state.active):
        if i >= fit.n_components:
            break
        pairs = (
            (c.frequency, float(fit.frequency[i]), abs(c.frequency)),
            (c.amplitude, float(fit.amplitude[i]), abs(c.amplitude)),
        )
        for stored, rebuilt, scale in pairs:
            if scale > 0.0 and np.isfinite(stored) and np.isfinite(rebuilt):
                worst = max(worst, abs(stored - rebuilt) / scale)
        dphi = (float(fit.phase[i]) - c.phase + np.pi) % _TWO_PI - np.pi
        if np.isfinite(dphi):
            worst = max(worst, abs(dphi) / _TWO_PI)
    return worst


__all__ = ["ExactUniformGrid", "SessionKernel"]
