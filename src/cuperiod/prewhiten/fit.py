"""Non-linear multi-sinusoid least squares — the "improve all" step of pre-whitening.

Period04's central operation is a simultaneous fit of every extracted component,

.. math::

    y(t) = c_0 + \\sum_k A_k \\sin\\!\\big(2\\pi f_k (t - t_{\\rm ref}) + \\phi_k\\big),

re-run after each new frequency so amplitudes and phases stay consistent as the solution
grows. This module implements that fit and the covariance it implies.

The model is *separable*: for fixed frequencies the amplitudes, phases and offset are
the solution of a linear least-squares problem. Only the frequencies are genuinely
non-linear, so the optimiser works on them alone by variable projection (Golub &
Pereyra 1973) with Kaufman's (1975) Jacobian — ``K`` free parameters instead of
``3K + 1``, and every linear parameter solved exactly at each step.

Three refinement policies are offered:

``"last"`` (the extraction loop's default)
    Optimise only the newest frequency against the residual of the others — one
    three-column sub-problem — while still re-solving *every* amplitude, phase and the
    offset jointly and exactly. Established frequencies move by far less than their own
    uncertainty when a new component is peeled off, so refining them at every iteration
    costs ``O(K)`` sub-problems to buy nothing; the final polish sweeps them all.
``"cyclic"``
    Block coordinate descent over all ``K`` frequencies: each is optimised in turn
    against the residual with its own component added back, then the linear coefficients
    are re-solved jointly. ``O(K N)`` per sweep, and because the model is additive every
    step is an exact block minimisation of the same objective, so the fit improves
    monotonically.
``"simultaneous"``
    Full variable projection over all ``K`` frequencies at once — including their mutual
    covariances, which matters for close pairs. The gold standard, at ``O(N K^2)`` per
    evaluation; used for the final polish by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten.spectrum import weighted_epoch

#: Frequency-refinement policy.
Refinement = Literal["none", "last", "cyclic", "simultaneous"]

#: Relative rank cutoff for the linear solves and the covariance pseudo-inverse.
_RCOND: Final = 1e-12

#: Widest design still solved through the normal equations (see :func:`_solve`).
_NORMAL_MAX_COLS: Final = 12

#: Convergence tolerances for the frequency optimiser. A frequency is worth knowing to
#: perhaps a hundredth of its uncertainty, which is ~1e-8 relative even for a long,
#: precise data set; 1e-11 leaves three orders of magnitude of margin and stops the
#: optimiser burning iterations chasing the last bits of round-off.
_FTOL: Final = 1e-11
_XTOL: Final = 1e-11
_GTOL: Final = 1e-11

#: Elements per transient block when accumulating the covariance normal matrix (8 MB).
_COV_BLOCK_ELEMS: Final = 1 << 20

_TWO_PI: Final = 2.0 * np.pi


def _as_weights(
    n: int, error: FloatArray | None
) -> tuple[FloatArray, FloatArray, bool]:
    """``(w, sqrt(w), weighted)`` for inverse-variance or uniform weighting."""
    if error is None:
        ones = np.ones(n, dtype=np.float64)
        return ones, ones, False
    dy = np.ascontiguousarray(np.asarray(error, dtype=np.float64))
    w = 1.0 / (dy * dy)
    return w, np.sqrt(w), True


def _design(dt: FloatArray, freqs: FloatArray, fit_mean: bool) -> FloatArray:
    """The ``(N, 2K + 1)`` linear design ``[cos_1, sin_1, ..., cos_K, sin_K, 1]``.

    Built one frequency at a time so no ``(N, K)`` angle matrix is ever materialised —
    the same total trigonometric work with bounded transient memory.
    """
    n = int(dt.size)
    k = int(freqs.size)
    ncol = 2 * k + (1 if fit_mean else 0)
    x = np.empty((n, max(ncol, 1)), dtype=np.float64)
    for j in range(k):
        theta = (_TWO_PI * float(freqs[j])) * dt
        x[:, 2 * j] = np.cos(theta)
        x[:, 2 * j + 1] = np.sin(theta)
    if fit_mean:
        x[:, -1] = 1.0
    return x[:, :ncol]


def _solve(design: FloatArray, rhs: FloatArray) -> Any:
    """Least-squares solution of ``design @ b = rhs`` (multi-RHS, rank-revealing).

    Narrow designs — every evaluation of the cyclic refinement's three-column
    sub-problem, which is where the extraction loop spends most of its time — go through
    the normal equations with a Cholesky factorisation. Forming ``X^T X`` squares the
    condition number, but a few trigonometric columns are conditioned like ``O(1)``
    so the squared condition number is still negligible, and a Gram-diagonal guard
    plus Cholesky's own positive-definiteness check catches the degenerate cases. That
    is several times faster than a QR of the tall matrix.

    Everything wider falls back to ``gelsy``, a column-pivoted QR: about the cost of a
    plain QR — far less than the SVD-based drivers — while still truncating
    rank-deficient directions, which matters when two trial frequencies drift close
    enough to make the design singular.
    """
    if design.shape[1] <= _NORMAL_MAX_COLS and design.shape[0] >= design.shape[1]:
        solution = _solve_normal(design, rhs)
        if solution is not None:
            return solution
    from scipy.linalg import lstsq

    return lstsq(
        design, rhs, cond=_RCOND, lapack_driver="gelsy", check_finite=False
    )[0]


def _solve_normal(design: FloatArray, rhs: FloatArray) -> Any:
    """Normal-equation solve, or ``None`` when the design looks degenerate."""
    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    gram = design.T @ design
    diagonal = np.diag(gram)
    if not np.all(np.isfinite(diagonal)):
        return None
    largest = float(diagonal.max()) if diagonal.size else 0.0
    if largest <= 0.0 or float(diagonal.min()) <= _RCOND * largest:
        return None
    try:
        factor = cho_factor(gram, lower=True, check_finite=False)
        solution = cho_solve(factor, design.T @ rhs, check_finite=False)
    except (LinAlgError, ValueError):
        return None
    return solution if np.all(np.isfinite(solution)) else None


def _split_linear(
    beta: FloatArray, k: int, fit_mean: bool
) -> tuple[FloatArray, FloatArray, float]:
    """Split a linear solution into ``(cos coefficients, sin coefficients, offset)``."""
    a = np.ascontiguousarray(beta[0 : 2 * k : 2])
    b = np.ascontiguousarray(beta[1 : 2 * k : 2])
    offset = float(beta[-1]) if fit_mean else 0.0
    return a, b, offset


class _VarPro:
    """Variable-projection residual/Jacobian for the frequencies of a multi-sine fit.

    ``fun`` and ``jac`` share one evaluation per parameter vector, so a Jacobian costs
    nothing extra once the residual has been formed at the same frequencies.
    """

    def __init__(
        self, dt: FloatArray, yw: FloatArray, sw: FloatArray, fit_mean: bool
    ) -> None:
        self._dt = dt
        self._yw = yw
        self._sw = sw
        self._fit_mean = fit_mean
        self._key: bytes | None = None
        self._residual: FloatArray = np.zeros(0, dtype=np.float64)
        self._jacobian: FloatArray = np.zeros((0, 0), dtype=np.float64)

    def _evaluate(self, freqs: FloatArray) -> None:
        key = np.ascontiguousarray(freqs, dtype=np.float64).tobytes()
        if key == self._key:
            return
        k = int(freqs.size)
        n = int(self._dt.size)
        design = _design(self._dt, freqs, self._fit_mean)
        design_w = design * self._sw[:, None]
        beta = _solve(design_w, self._yw)
        self._residual = self._yw - design_w @ beta
        a, b, _ = _split_linear(beta, k, self._fit_mean)
        # d/df_k of the k-th component: 2*pi*dt * (b_k cos - a_k sin).
        deriv = np.empty((n, k), dtype=np.float64)
        scaled_dt = (_TWO_PI * self._dt) * self._sw
        for j in range(k):
            deriv[:, j] = scaled_dt * (
                b[j] * design[:, 2 * j] - a[j] * design[:, 2 * j + 1]
            )
        self._jacobian = -(deriv - design_w @ _solve(design_w, deriv))
        self._key = key

    def fun(self, freqs: FloatArray) -> FloatArray:
        """Weighted residual vector at ``freqs`` (linear parameters projected out)."""
        self._evaluate(freqs)
        return self._residual

    def jac(self, freqs: FloatArray) -> FloatArray:
        """Kaufman variable-projection Jacobian at ``freqs``."""
        self._evaluate(freqs)
        return self._jacobian


def _normalize_bounds(
    bounds: tuple[Any, Any], k: int
) -> tuple[FloatArray, FloatArray]:
    """Broadcast scalar-or-array frequency bounds to two length-``k`` arrays.

    The upper bound is nudged above the lower one where they coincide: a collapsed box
    is legal input here (a frequency pinned by the caller) but the optimiser rejects it.
    """
    lo = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (k,)).copy()
    hi = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (k,)).copy()
    hi = np.maximum(hi, np.nextafter(lo, np.inf))
    return lo, hi


def _optimize_frequencies(
    dt: FloatArray,
    yw: FloatArray,
    sw: FloatArray,
    freqs: FloatArray,
    *,
    fit_mean: bool,
    bounds: tuple[Any, Any],
    scale: float,
    max_nfev: int,
) -> FloatArray:
    """Refine ``freqs`` by variable projection; never returns a worse solution."""
    from scipy.optimize import least_squares

    if freqs.size == 0:
        return freqs
    lo, hi = _normalize_bounds(bounds, int(freqs.size))
    start = np.clip(np.ascontiguousarray(freqs, dtype=np.float64), lo, hi)
    varpro = _VarPro(dt, yw, sw, fit_mean)
    start_cost = float(np.sum(varpro.fun(start) ** 2))
    try:
        result = least_squares(
            varpro.fun,
            start,
            jac=varpro.jac,
            bounds=(lo, hi),
            method="trf",
            x_scale=np.full(start.size, max(scale, float(np.finfo(float).tiny))),
            ftol=_FTOL,
            xtol=_XTOL,
            gtol=_GTOL,
            max_nfev=max_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
        return start
    improved = np.ascontiguousarray(np.asarray(result.x, dtype=np.float64))
    if not np.all(np.isfinite(improved)):  # pragma: no cover - defensive
        return start
    if 2.0 * float(result.cost) > start_cost:  # pragma: no cover - defensive
        return start
    return improved


def _cyclic_refine(
    dt: FloatArray,
    y: FloatArray,
    yw: FloatArray,
    sw: FloatArray,
    freqs: FloatArray,
    *,
    fit_mean: bool,
    bounds: tuple[FloatArray, FloatArray],
    scale: float,
    max_nfev: int,
    sweeps: int,
    indices: Sequence[int] | None = None,
) -> FloatArray:
    """Block coordinate descent over the frequencies (see the module docstring).

    ``indices`` restricts the sweep to a subset — the ``"last"`` policy passes just the
    newest frequency — while the linear parameters are always re-solved for *all* of
    them.
    """
    freqs = np.ascontiguousarray(freqs, dtype=np.float64).copy()
    k = int(freqs.size)
    targets = range(k) if indices is None else indices
    for _ in range(max(1, sweeps)):
        design = _design(dt, freqs, fit_mean)
        beta = _solve(design * sw[:, None], yw)
        model = design @ beta
        for j in targets:
            component = (
                float(beta[2 * j]) * design[:, 2 * j]
                + float(beta[2 * j + 1]) * design[:, 2 * j + 1]
            )
            partial = model - component
            target = (y - partial) * sw
            refined = _optimize_frequencies(
                dt, target, sw, freqs[j : j + 1],
                fit_mean=fit_mean,
                bounds=(bounds[0][j : j + 1], bounds[1][j : j + 1]),
                scale=scale,
                max_nfev=max_nfev,
            )
            freqs[j] = refined[0]
            # Re-solve just this component against its own target and fold it back into
            # the running model; the joint linear solve happens once per sweep (and once
            # more in the caller), so this stays O(N) per frequency.
            sub_design = _design(dt, freqs[j : j + 1], fit_mean)
            sub_beta = _solve(sub_design * sw[:, None], target)
            model = partial + sub_design @ sub_beta
            design[:, 2 * j] = sub_design[:, 0]
            design[:, 2 * j + 1] = sub_design[:, 1]
            beta[2 * j] = sub_beta[0]
            beta[2 * j + 1] = sub_beta[1]
    return freqs


def _normal_matrix(
    dt: FloatArray,
    sw: FloatArray,
    freqs: FloatArray,
    amps: FloatArray,
    phases: FloatArray,
    fit_mean: bool,
) -> FloatArray:
    """``J^T W J`` for the natural parameters ``[f, A, phi(, c0)]``.

    Accumulated in row blocks so the ``(N, 3K + 1)`` Jacobian is never materialised in
    full — a 100k-point curve with 100 frequencies would otherwise need a quarter of a
    gigabyte just to report uncertainties.
    """
    k = int(freqs.size)
    n = int(dt.size)
    ncol = 3 * k + (1 if fit_mean else 0)
    gram = np.zeros((ncol, ncol), dtype=np.float64)
    if ncol == 0 or n == 0:
        return gram
    block = max(1, _COV_BLOCK_ELEMS // max(ncol, 1))
    for start in range(0, n, block):
        stop = min(start + block, n)
        dt_b = dt[start:stop]
        jac = np.empty((stop - start, ncol), dtype=np.float64)
        for j in range(k):
            theta = _TWO_PI * float(freqs[j]) * dt_b + float(phases[j])
            cos_j = np.cos(theta)
            jac[:, j] = float(amps[j]) * cos_j * (_TWO_PI * dt_b)  # d/df
            jac[:, k + j] = np.sin(theta)  # d/dA
            jac[:, 2 * k + j] = float(amps[j]) * cos_j  # d/dphi
        if fit_mean:
            jac[:, -1] = 1.0
        jac *= sw[start:stop, None]
        gram += jac.T @ jac
    return gram


def _covariance(
    gram: FloatArray, *, rss: float, n_samples: int, n_parameters: int
) -> FloatArray:
    """``s^2 (J^T W J)^+`` with ``s^2 = RSS / (N - M)``.

    Scaling by the reduced chi-square is deliberate: it makes the uncertainties correct
    when no error bars were supplied (the noise level comes from the fit itself) and
    robust to the common case of survey error bars mis-scaled by a constant factor.
    """
    if gram.size == 0:
        return gram
    dof = n_samples - n_parameters
    scale = rss / dof if dof > 0 else float("nan")
    values, vectors = np.linalg.eigh(gram)
    cutoff = _RCOND * float(max(values.max(), 0.0))
    keep = values > cutoff
    inverse_values = np.where(keep, 1.0 / np.where(keep, values, 1.0), 0.0)
    covariance: FloatArray = (vectors * inverse_values) @ vectors.T
    return covariance * scale


@dataclass(frozen=True)
class MultiSineFit:
    """The fitted multi-sinusoid solution for one light curve.

    Attributes
    ----------
    frequency, amplitude, phase : numpy.ndarray
        The ``K`` components. ``phase`` is in radians for the convention
        ``A sin(2*pi*f*(t - t_ref) + phase)``, wrapped to ``[0, 2*pi)``.
    offset : float
        The fitted constant ``c_0`` (zero when ``fit_mean=False``).
    t_ref : float
        Epoch the phases are referenced to (days).
    residuals : numpy.ndarray
        ``value - model(time)`` at the input samples.
    covariance : numpy.ndarray
        Parameter covariance in the natural parameterisation, ordered
        ``[f_0..f_{K-1}, A_0..A_{K-1}, phi_0..phi_{K-1}(, c_0)]``. Empty when the caller
        asked for no covariance.
    rss : float
        Weighted residual sum of squares (equal to chi-square when errors were given).
    n_samples, n_parameters : int
        Sample and free-parameter counts (``M = 3K + 1``).
    weighted : bool
        Whether inverse-variance weights were used.
    """

    frequency: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    offset: float
    t_ref: float
    residuals: FloatArray
    covariance: FloatArray
    rss: float
    n_samples: int
    n_parameters: int
    weighted: bool
    fit_mean: bool = True

    @property
    def n_components(self) -> int:
        """Number of sinusoids in the solution."""
        return int(self.frequency.size)

    @property
    def chi2(self) -> float:
        """Chi-square (equal to :attr:`rss`; meaningful when errors were supplied)."""
        return self.rss

    @property
    def reduced_chi2(self) -> float:
        """Chi-square per degree of freedom."""
        dof = self.n_samples - self.n_parameters
        return self.rss / dof if dof > 0 else float("nan")

    @property
    def rms(self) -> float:
        """Unweighted RMS of the residuals, in the units of the input."""
        if self.residuals.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def bic(self) -> float:
        """Bayesian information criterion (lower is better).

        With errors the Gaussian log-likelihood is ``-chi2/2`` and
        ``BIC = chi2 + M ln N``. Without errors the noise scale is unknown and profiled
        out, giving ``BIC = N ln(RSS/N) + M ln N``. Either way only *differences*
        between solutions on one data set matter, which is all the stopping rule uses.
        """
        return self._criterion(float(np.log(self.n_samples)) if self.n_samples else 0.0)

    @property
    def aic(self) -> float:
        """Akaike information criterion (lower is better)."""
        return self._criterion(2.0)

    def _criterion(self, per_parameter: float) -> float:
        n, m = self.n_samples, self.n_parameters
        if n <= 0:
            return float("nan")
        penalty = m * per_parameter
        if self.weighted:
            return self.rss + penalty
        if self.rss <= 0.0:  # pragma: no cover - an exact fit
            return -float("inf")
        return n * float(np.log(self.rss / n)) + penalty

    # -- derived uncertainties ---------------------------------------------------
    def _sigma(self, block: int) -> FloatArray:
        k = self.n_components
        if self.covariance.size == 0 or k == 0:
            return np.full(k, np.nan, dtype=np.float64)
        diagonal = np.diag(self.covariance)[block * k : (block + 1) * k]
        return np.sqrt(np.clip(diagonal, 0.0, None))

    @property
    def frequency_error(self) -> FloatArray:
        """1-sigma frequency uncertainties from the covariance (cycles/day)."""
        return self._sigma(0)

    @property
    def amplitude_error(self) -> FloatArray:
        """1-sigma amplitude uncertainties from the covariance."""
        return self._sigma(1)

    @property
    def phase_error(self) -> FloatArray:
        """1-sigma phase uncertainties from the covariance (radians)."""
        return self._sigma(2)

    @property
    def offset_error(self) -> float:
        """1-sigma uncertainty on the constant term."""
        if self.covariance.size == 0 or not self.fit_mean:
            return float("nan")
        return float(np.sqrt(max(float(self.covariance[-1, -1]), 0.0)))

    # -- evaluation --------------------------------------------------------------
    def component(self, index: int, time: FloatArray) -> FloatArray:
        """The ``index``-th sinusoid evaluated at ``time``."""
        dt = np.asarray(time, dtype=np.float64) - self.t_ref
        theta = _TWO_PI * float(self.frequency[index]) * dt + float(self.phase[index])
        return float(self.amplitude[index]) * np.sin(theta)

    def model(self, time: FloatArray, *, include_offset: bool = True) -> FloatArray:
        """The full model evaluated at ``time``."""
        t = np.asarray(time, dtype=np.float64)
        out = np.full(t.shape, self.offset if include_offset else 0.0, dtype=np.float64)
        for k in range(self.n_components):
            out += self.component(k, t)
        return out


def fit_multisine(
    time: FloatArray,
    value: FloatArray,
    error: FloatArray | None,
    frequencies: FloatArray,
    *,
    fit_mean: bool = True,
    t_ref: float | None = None,
    refine: Refinement = "cyclic",
    sweeps: int = 1,
    frequency_bounds: tuple[Any, Any] | None = None,
    max_nfev: int = 200,
    covariance: bool = True,
) -> MultiSineFit:
    """Fit ``y = c0 + sum_k A_k sin(2 pi f_k (t - t_ref) + phi_k)``.

    Parameters
    ----------
    time, value : numpy.ndarray
        The light curve (finite points only).
    error : numpy.ndarray, optional
        1-sigma uncertainties; ``None`` fits unweighted.
    frequencies : numpy.ndarray
        Starting frequencies in cycles/day. May be empty (then only the offset is fit).
        Refinement is *local*: a start further than about half a Rayleigh width
        (``0.5/T``) from the true frequency can settle on a sidelobe, so seed it from a
        spectrum peak — which is what :func:`~cuperiod.prewhiten.prewhiten` does.
    fit_mean : bool, default True
        Include a free constant term.
    t_ref : float, optional
        Phase reference epoch; defaults to
        :func:`~cuperiod.prewhiten.spectrum.weighted_epoch`, which decorrelates each
        phase from its frequency and so minimises the reported phase uncertainty.
    refine : {"none", "last", "cyclic", "simultaneous"}, default "cyclic"
        Frequency-refinement policy (see the module docstring). ``"none"`` solves only
        the linear parameters at the given frequencies.
    sweeps : int, default 1
        Number of cyclic sweeps (ignored by the other policies).
    frequency_bounds : (lower, upper), optional
        Box the refined frequencies. Scalars apply to every frequency; arrays give each
        one its own box, which is how the extraction loop stops a refinement from
        wandering off its own peak and onto a neighbour's. Defaults to
        ``(f_min/2, 2 f_max)`` of the starting values.
    max_nfev : int, default 200
        Optimiser evaluation cap per non-linear solve.
    covariance : bool, default True
        Compute the parameter covariance (the source of the reported uncertainties).

    Returns
    -------
    MultiSineFit

    Examples
    --------
    >>> fit = fit_multisine(t, mag, err, [1.234, 2.468])      # doctest: +SKIP
    >>> fit.amplitude, fit.amplitude_error                    # doctest: +SKIP
    """
    t = np.ascontiguousarray(np.asarray(time, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if t.shape != y.shape:
        raise ValueError("time and value must have the same length")
    freqs = np.ascontiguousarray(np.asarray(frequencies, dtype=np.float64)).ravel()
    if freqs.size == 0 and not fit_mean:
        raise ValueError("fit_multisine needs at least one frequency or fit_mean=True")
    n = int(t.size)
    w, sw, weighted = _as_weights(n, error)
    ref = weighted_epoch(t, w) if t_ref is None else float(t_ref)
    dt = t - ref
    yw = sw * y
    baseline = float(t.max() - t.min()) if n else 0.0
    rayleigh = 1.0 / baseline if baseline > 0.0 else 1.0

    if frequency_bounds is not None:
        raw_bounds: tuple[Any, Any] = frequency_bounds
    elif freqs.size:
        low = max(0.5 * float(freqs.min()), 1e-12)
        raw_bounds = (low, max(2.0 * float(freqs.max()), low * (1.0 + 1e-9)))
    else:
        raw_bounds = (1e-12, 1.0)
    bounds = _normalize_bounds(raw_bounds, int(freqs.size))

    if freqs.size and refine == "simultaneous":
        freqs = _optimize_frequencies(
            dt, yw, sw, freqs,
            fit_mean=fit_mean, bounds=bounds, scale=rayleigh, max_nfev=max_nfev,
        )
    elif freqs.size and refine in {"cyclic", "last"}:
        freqs = _cyclic_refine(
            dt, y, yw, sw, freqs,
            fit_mean=fit_mean, bounds=bounds, scale=rayleigh,
            max_nfev=max_nfev, sweeps=1 if refine == "last" else sweeps,
            indices=(int(freqs.size) - 1,) if refine == "last" else None,
        )

    design = _design(dt, freqs, fit_mean)
    beta = _solve(design * sw[:, None], yw)
    a, b, offset = _split_linear(beta, int(freqs.size), fit_mean)
    amplitude = np.hypot(a, b)
    phase = np.mod(np.arctan2(a, b), _TWO_PI)
    residuals = y - design @ beta
    rss = float(np.dot(w, residuals * residuals))
    n_parameters = int(3 * freqs.size + (1 if fit_mean else 0))

    if covariance:
        gram = _normal_matrix(dt, sw, freqs, amplitude, phase, fit_mean)
        cov = _covariance(
            gram, rss=rss, n_samples=n, n_parameters=n_parameters
        )
    else:
        cov = np.zeros((0, 0), dtype=np.float64)
    return MultiSineFit(
        frequency=freqs,
        amplitude=amplitude,
        phase=phase,
        offset=offset,
        t_ref=ref,
        residuals=residuals,
        covariance=cov,
        rss=rss,
        n_samples=n,
        n_parameters=n_parameters,
        weighted=weighted,
        fit_mean=fit_mean,
    )


__all__ = ["MultiSineFit", "Refinement", "fit_multisine"]
