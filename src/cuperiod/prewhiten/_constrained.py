"""The general multi-sine kernel: fixed values, frequency ties, subsets and weights.

:func:`~cuperiod.prewhiten.fit.fit_multisine` dispatches here whenever it receives a
non-trivial :class:`~cuperiod.prewhiten.constraints.FitConstraints`; an unconstrained
call never reaches this module, so the v1.3 kernel and every result it produces are
untouched.

The model, for samples ``i`` with ``tau_i = t_i - t_ref`` and weights
``w_i = omega_i / sigma_i^2``,

.. math::

    y_i \\sim o_{g_o(i)} + \\sum_k A_{k, a_k(i)}
    \\sin\\big(2\\pi f_k \\tau_i + \\phi_{k, p_k(i)}\\big),
    \\qquad f = f_0 + M\\theta,

is still separable. Every amplitude and zero point that is free enters linearly; the
non-linear vector is ``alpha = [theta; psi]`` — the free *independent* frequencies
(ties are flattened into the integer matrix ``M``) and only those phases a constraint
makes non-linear (a fixed amplitude, or an amplitude shared across subsets whose phase
is per subset, or vice versa). Variable projection with Kaufman's Jacobian optimises
``alpha`` while every linear parameter is solved exactly at each step, exactly as the
v1.3 kernel does for the free case (where ``M = I`` and ``psi`` is empty).

Each component falls into one of three classes (spec table 1.3): a **pair** (free
amplitude and phase: ``cos``/``sin`` columns, per subset when both vary), a **signed**
sinusoid (a free amplitude multiplying ``sin(th + phi)`` with the phase fixed or
non-linear), or a **known** term (fixed amplitude, phase fixed or non-linear).

Designs that contain a zero-point indicator or a per-subset column are solved by a
column-pivoted QR (``gelsy``) rather than the normal equations: next to a low-frequency
component such columns make the design ill-conditioned enough that forming ``X^T X``
loses half the digits of the solution. On large designs, zero points per group never
become columns: they are projected out of the data and of every other column
(weighted group centring, exact by Frisch-Waugh-Lovell), so a fit with a zero point
per night costs about the same per evaluation as one with a single constant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, get_args

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.prewhiten import fit as _fit
from cuperiod.prewhiten.constraints import (
    ComponentConstraint,
    FitConstraints,
    FitDetail,
    FitStart,
)
from cuperiod.prewhiten.spectrum import weighted_epoch

_TWO_PI: Final = 2.0 * np.pi

#: Relative rank cutoff of the pivoted-QR solves of designs with partial-support
#: columns (zero-point indicators, per-subset columns).
_GELSY_COND: Final = 1e-13

#: ``|R_ii| / |R_00|`` above which a column of the natural-parameter design counts
#: towards its rank.
_RANK_TOL: Final = 1e-13

#: A parameter with at least this weight in an eigen-direction the covariance
#: pseudo-inverse drops is unidentifiable: its errors are reported NaN.
_DEGENERATE_WEIGHT: Final = 1e-3

#: A root this many box widths from an edge of its refinement box is "at_bound".
_AT_BOUND: Final = 1e-9

#: Gauss-Newton polish of fits with non-linear phases: step cap and the scaled step
#: below which it stops.
_POLISH_STEPS: Final = 6
_POLISH_TOL: Final = 1e-13

#: Rows per block when accumulating the natural-parameter normal matrix.
_COV_BLOCK_ELEMS: Final = 1 << 20

#: Size ``N x S_o`` of the zero-point indicator block from which the group zero
#: points are projected out of every solve (_Groups) instead of solved as columns.
#: Below it the block is cheap and the dense solve is kept: the two agree to
#: round-off, and small fits keep the exact arithmetic the oracle tests pinned.
_PROJECT_GROUPS_ELEMENTS: Final = 1 << 16


def _lstsq(design_w: FloatArray, rhs: FloatArray, full_support: bool) -> FloatArray:
    """Least squares through the shared fast solver only when every column is global.

    ``fit._solve`` (normal equations + Cholesky for narrow designs) is accurate for
    trigonometric columns of resolved frequencies and the single constant. A column
    that is non-zero on part of the data only — a zero point per subset, a per-subset
    amplitude — can make the design ill-conditioned next to a low frequency, and the
    normal equations would lose ``cond^2`` of accuracy, so those designs go to the
    rank-revealing pivoted QR directly.
    """
    shape = (0,) if rhs.ndim == 1 else (0, rhs.shape[1])
    if design_w.shape[1] == 0:
        return np.zeros(shape, dtype=np.float64)
    if full_support:
        out: FloatArray = np.asarray(_fit._solve(design_w, rhs), dtype=np.float64)
        return out
    from scipy.linalg import lstsq

    solution: FloatArray = lstsq(
        design_w, rhs, cond=_GELSY_COND, lapack_driver="gelsy", check_finite=False
    )[0]
    return solution


@dataclass(frozen=True)
class _Comp:
    """One component's class and its slice of the non-linear phase vector ``psi``."""

    index: int
    kind: str  # "pair" | "signed" | "known"
    amp_vary: bool
    phase_vary: bool
    amp_fixed: float | None
    phase_fixed: float | None
    psi_start: int
    n_psi: int

    @property
    def full_support(self) -> bool:
        return self.kind != "known" and not self.amp_vary

    @property
    def phase_free(self) -> bool:
        """True when the phase is a free parameter (pair, or a non-linear phase)."""
        return self.phase_fixed is None

    def n_columns(self, n_subsets: int) -> int:
        if self.kind == "pair":
            return 2 * n_subsets if self.amp_vary else 2
        if self.kind == "signed":
            return n_subsets if self.amp_vary else 1
        return 0


def _classify(
    constraints: tuple[ComponentConstraint, ...], n_subsets: int
) -> list[_Comp]:
    """The class of every component (spec table 1.3) and its ``psi`` layout."""
    comps: list[_Comp] = []
    psi = 0
    for k, c in enumerate(constraints):
        amp_vary, phase_vary = c.varies_amplitude, c.varies_phase
        amp = None if c.amplitude is None else float(c.amplitude)
        phase = None if c.phase is None else float(c.phase)
        if amp_vary and phase_vary:
            kind, n_psi = "pair", 0
        elif amp_vary:
            kind, n_psi = "signed", (0 if phase is not None else 1)
        elif phase_vary:
            kind = "signed" if amp is None else "known"
            n_psi = n_subsets
        elif amp is None:
            kind, n_psi = ("pair", 0) if phase is None else ("signed", 0)
        else:
            kind, n_psi = "known", (0 if phase is not None else 1)
        comps.append(_Comp(k, kind, amp_vary, phase_vary, amp, phase, psi, n_psi))
        psi += n_psi
    return comps


class _Groups:
    """The zero-point groups, projected out of every solve (Frisch-Waugh-Lovell).

    A zero point per group is an indicator column ``sqrt(w) 1_g`` of the weighted
    design. Those columns have disjoint support, so the projection onto their span
    is a weighted group mean: removing it from the data and from every other column
    and solving the rest gives exactly the component coefficients and residuals of
    the full least-squares problem, and the zero points are then the weighted group
    means of what the components leave. Every solve costs ``O(N m)`` for the
    projection plus a QR of the ``N x m`` component design, instead of a QR of an
    ``N x (m + S_o)`` design whose indicator block alone is ``N x S_o``.
    """

    def __init__(self, codes: IntArray, w: FloatArray, sw: FloatArray, n: int) -> None:
        from scipy.sparse import csr_matrix

        size = int(codes.size)
        self.codes = codes
        self.n = int(n)
        self.sw = sw
        self.total: FloatArray = np.asarray(
            np.bincount(codes, weights=w, minlength=self.n), dtype=np.float64
        )
        # summer @ z = sum_{i in g} sqrt(w_i) z_i: for a weighted column z = sqrt(w) x
        # that is sum_{i in g} w_i x_i.
        self.summer = csr_matrix(
            (sw, (codes, np.arange(size))), shape=(self.n, size), dtype=np.float64
        )

    def sums(self, weighted: FloatArray) -> FloatArray:
        """``D^T z`` of weighted columns ``z`` (``(S,)`` or ``(S, m)``)."""
        out: FloatArray = np.asarray(self.summer @ weighted, dtype=np.float64)
        return out

    def means(self, weighted: FloatArray) -> FloatArray:
        """Weighted group means ``sum_g w x / W_g`` of weighted columns ``sw x``."""
        sums = self.sums(weighted)
        out: FloatArray = (
            sums / self.total if sums.ndim == 1 else sums / self.total[:, None]
        )
        return out

    def project(self, weighted: FloatArray) -> FloatArray:
        """``M_D z``: weighted columns with their weighted group means removed."""
        means = self.means(weighted)
        if weighted.ndim == 1:
            out: FloatArray = weighted - self.sw * means[self.codes]
            return out
        out = weighted - self.sw[:, None] * means[self.codes]
        return out


@dataclass
class _Setup:
    """Everything about the data and the model layout that no iteration changes."""

    dt: FloatArray
    y: FloatArray
    w: FloatArray
    sw: FloatArray
    comps: list[_Comp]
    matrix: FloatArray  # M (K, P)
    constant: FloatArray  # f0 (K,)
    vcodes: IntArray | None
    n_v: int
    vmasks: list[FloatArray]
    ocodes: IntArray | None
    n_o: int
    offsets: str  # "none" | "single" | "groups"
    groups: _Groups | None = None  # offsets == "groups", projected (large N x S_o)

    @property
    def n_theta(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n_psi(self) -> int:
        return sum(c.n_psi for c in self.comps)

    def frequencies(self, theta: FloatArray) -> FloatArray:
        out: FloatArray = self.constant + self.matrix @ theta
        return out

    def phase_of(self, comp: _Comp, psi: FloatArray) -> float | FloatArray:
        """Per-point (or scalar) phase of a signed/known component."""
        if comp.phase_fixed is not None:
            return comp.phase_fixed
        if comp.n_psi == 0:
            return 0.0
        segment = psi[comp.psi_start : comp.psi_start + comp.n_psi]
        if comp.phase_vary:
            assert self.vcodes is not None
            out: FloatArray = segment[self.vcodes]
            return out
        return float(segment[0])


class _Problem:
    """Variable-projection residual and Kaufman Jacobian of a (sub-)problem.

    ``comps`` are the components in play (all of them, or the block ``D_j`` of one
    root), ``active_theta`` the global indices of the free frequencies being optimised
    and ``psi_active`` whether the non-linear phases of ``comps`` are; everything else
    is held at the values in ``theta`` / ``psi``. ``offsets`` chooses the zero-point
    columns: ``"single"`` (one constant), ``"groups"`` (one per offset subset) or
    ``"none"``.
    Residuals are weighted, ``sqrt(w) (target - eta0 - Phi beta)``.
    """

    def __init__(
        self,
        setup: _Setup,
        comps: Sequence[_Comp],
        target: FloatArray,
        theta: FloatArray,
        psi: FloatArray,
        *,
        active_theta: Sequence[int],
        psi_active: bool,
        offsets: str,
    ) -> None:
        self.s = setup
        self.comps = list(comps)
        self.target = target
        self.theta = np.array(theta, dtype=np.float64, copy=True)
        self.psi = np.array(psi, dtype=np.float64, copy=True)
        self.active_theta = [int(j) for j in active_theta]
        self.psi_comps = [c for c in self.comps if c.n_psi] if psi_active else []
        self.psi_active = bool(self.psi_comps)
        self.n_psi_active = sum(c.n_psi for c in self.psi_comps)
        self.offsets = offsets
        # Group zero points projected out of every solve rather than solved as
        # indicator columns (see _Groups).
        self.projected = offsets == "groups" and setup.groups is not None
        self.full_support = all(c.full_support for c in self.comps) and (
            offsets != "groups"
        )
        starts = []
        position = 0
        for c in self.comps:
            starts.append(position)
            position += c.n_columns(setup.n_v)
        self.col_start = starts
        self.n_comp_cols = position
        self.nfev = 0
        self._key: bytes | None = None
        self._jac_key: bytes | None = None
        self._design = np.zeros((0, 0))
        self._design_w = np.zeros((0, 0))
        self._eta = np.zeros(0)
        self._beta = np.zeros(0)
        self._residual = np.zeros(0)
        self._jacobian = np.zeros((0, 0))
        self._cos: list[FloatArray] = []
        self._sin: list[FloatArray] = []

    # -- packing ---------------------------------------------------------------------
    def pack(self) -> FloatArray:
        parts = [self.theta[self.active_theta]]
        parts.extend(
            self.psi[c.psi_start : c.psi_start + c.n_psi] for c in self.psi_comps
        )
        return np.concatenate(parts)

    def unpack(self, alpha: FloatArray) -> None:
        p = len(self.active_theta)
        if p:
            self.theta[self.active_theta] = alpha[:p]
        position = p
        for c in self.psi_comps:
            self.psi[c.psi_start : c.psi_start + c.n_psi] = alpha[
                position : position + c.n_psi
            ]
            position += c.n_psi

    # -- evaluation ------------------------------------------------------------------
    def _evaluate(self, alpha: FloatArray) -> None:
        alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        key = alpha.tobytes()
        if key == self._key:
            return
        self.unpack(alpha)
        s = self.s
        n = int(s.dt.size)
        freqs = s.frequencies(self.theta)
        n_off = {"none": 0, "single": 1, "groups": s.n_o}[self.offsets]
        if self.projected:
            n_off = 0  # projected out, never columns (_Groups)
        design = np.empty((n, self.n_comp_cols + n_off), dtype=np.float64, order="F")
        eta = np.zeros(n, dtype=np.float64)
        cosines: list[FloatArray] = []
        sines: list[FloatArray] = []
        for comp, start in zip(self.comps, self.col_start, strict=True):
            # The same expression as fit._design, so the free case builds identical
            # columns.
            theta = (_TWO_PI * float(freqs[comp.index])) * s.dt
            if comp.kind == "pair":
                c, sn = np.cos(theta), np.sin(theta)
                if comp.amp_vary:
                    for g, mask in enumerate(s.vmasks):
                        design[:, start + 2 * g] = c * mask
                        design[:, start + 2 * g + 1] = sn * mask
                else:
                    design[:, start] = c
                    design[:, start + 1] = sn
            else:
                arg = theta + s.phase_of(comp, self.psi)
                c, sn = np.cos(arg), np.sin(arg)
                if comp.kind == "signed":
                    if comp.amp_vary:
                        for g, mask in enumerate(s.vmasks):
                            design[:, start + g] = sn * mask
                    else:
                        design[:, start] = sn
                else:
                    assert comp.amp_fixed is not None
                    eta += comp.amp_fixed * sn
            cosines.append(c)
            sines.append(sn)
        if self.offsets == "single":
            design[:, -1] = 1.0
        elif self.offsets == "groups" and not self.projected:
            assert s.ocodes is not None
            block = design[:, self.n_comp_cols :]
            block[:] = 0.0
            block[np.arange(n), s.ocodes] = 1.0
        design_w = design * s.sw[:, None]
        rhs = s.sw * (self.target - eta)
        if self.projected:
            # Frisch-Waugh-Lovell: the component coefficients and the residuals of
            # the projected problem are those of the full one; the zero points are
            # the weighted group means of what the components leave.
            groups = s.groups
            assert groups is not None
            rhs_p = groups.project(rhs)
            if design_w.shape[1]:
                design_w = groups.project(design_w)
            coef = _lstsq(design_w, rhs_p, self.full_support)
            self._residual = rhs_p - design_w @ coef if coef.size else rhs_p
            left = s.sw * (self.target - eta - design @ coef) if coef.size else rhs
            beta = np.concatenate([coef, groups.means(left)])
        else:
            beta = _lstsq(design_w, rhs, self.full_support)
            self._residual = rhs - design_w @ beta if beta.size else rhs
        # With groups, design_w is the projected component design (what the Kaufman
        # Jacobian is taken against); design is never projected.
        self._design, self._design_w = design, design_w
        self._eta, self._beta = eta, beta
        self._cos, self._sin = cosines, sines
        self._key = key
        self.nfev += 1

    def fun(self, alpha: FloatArray) -> FloatArray:
        self._evaluate(alpha)
        return self._residual

    def amplitude_per_point(self, comp: _Comp, position: int) -> FloatArray | float:
        """``A[k, a_k(i)]`` of a signed or known component."""
        if comp.kind == "known":
            assert comp.amp_fixed is not None
            return comp.amp_fixed
        start = self.col_start[position]
        if comp.amp_vary:
            assert self.s.vcodes is not None
            out: FloatArray = self._beta[start : start + self.s.n_v][self.s.vcodes]
            return out
        return float(self._beta[start])

    def quadrature(self, position: int) -> FloatArray:
        """``D_k = d comp_k / d phase`` per point at the current solution."""
        comp = self.comps[position]
        c, sn = self._cos[position], self._sin[position]
        if comp.kind == "pair":
            start = self.col_start[position]
            if comp.amp_vary:
                assert self.s.vcodes is not None
                a = self._beta[start : start + 2 * self.s.n_v : 2][self.s.vcodes]
                b = self._beta[start + 1 : start + 2 * self.s.n_v : 2][self.s.vcodes]
                out: FloatArray = b * c - a * sn
                return out
            a0, b0 = float(self._beta[start]), float(self._beta[start + 1])
            out = b0 * c - a0 * sn
            return out
        out = self.amplitude_per_point(comp, position) * c
        return out

    def contribution(self, position: int) -> FloatArray:
        """Component ``position``'s model values at the current solution."""
        comp = self.comps[position]
        c, sn = self._cos[position], self._sin[position]
        if comp.kind == "pair":
            start = self.col_start[position]
            if comp.amp_vary:
                assert self.s.vcodes is not None
                a = self._beta[start : start + 2 * self.s.n_v : 2][self.s.vcodes]
                b = self._beta[start + 1 : start + 2 * self.s.n_v : 2][self.s.vcodes]
                out: FloatArray = a * c + b * sn
                return out
            out = float(self._beta[start]) * c + float(self._beta[start + 1]) * sn
            return out
        out = self.amplitude_per_point(comp, position) * sn
        return out

    def offset_contribution(self) -> FloatArray:
        n = int(self.s.dt.size)
        if self.offsets == "none" or self._beta.size == 0:
            return np.zeros(n, dtype=np.float64)
        if self.projected:
            assert self.s.ocodes is not None
            grouped: FloatArray = self._beta[self.n_comp_cols :][self.s.ocodes]
            return grouped
        out: FloatArray = self._design[:, self.n_comp_cols :] @ self._beta[
            self.n_comp_cols :
        ]
        return out

    def linear_model(self) -> FloatArray:
        """Every linear term (components and zero points) at the current solution."""
        n = int(self.s.dt.size)
        if self._beta.size == 0:
            return np.zeros(n, dtype=np.float64)
        if not self.projected:
            out: FloatArray = self._design @ self._beta
            return out
        coef = self._beta[: self.n_comp_cols]
        out = self.offset_contribution()
        if coef.size:
            out = self._design @ coef + out
        return out

    def jac(self, alpha: FloatArray) -> FloatArray:
        self._evaluate(alpha)
        if self._jac_key == self._key:
            return self._jacobian
        s = self.s
        n = int(s.dt.size)
        quad = [self.quadrature(p) for p in range(len(self.comps))]
        columns: list[FloatArray] = []
        two_pi_dt = _TWO_PI * s.dt
        for j in self.active_theta:
            col = np.zeros(n, dtype=np.float64)
            for position, comp in enumerate(self.comps):
                m = float(s.matrix[comp.index, j])
                if m != 0.0:
                    col += m * quad[position]
            columns.append(two_pi_dt * col)
        if self.psi_active:
            for position, comp in enumerate(self.comps):
                if comp.n_psi == 0:
                    continue
                if comp.phase_vary:
                    for mask in s.vmasks:
                        columns.append(quad[position] * mask)
                else:
                    columns.append(quad[position])
        if not columns:
            self._jacobian = np.zeros((n, 0))
        else:
            g_w = np.column_stack(columns) * s.sw[:, None]
            if self.projected:
                # M_full = M_[M_D Phi] M_D (Frisch-Waugh-Lovell); _design_w is the
                # projected component design.
                assert s.groups is not None
                g_w = s.groups.project(g_w)
            if self._design_w.shape[1]:
                coef = _lstsq(self._design_w, g_w, self.full_support)
                self._jacobian = -(g_w - self._design_w @ coef)
            else:
                self._jacobian = -g_w
        self._jac_key = self._key
        return self._jacobian


def _optimize(
    problem: _Problem,
    lower: FloatArray,
    upper: FloatArray,
    x_scale: FloatArray,
    max_nfev: int,
) -> FloatArray:
    """Run the trust-region solver on ``problem``; never returns a worse point."""
    from scipy.optimize import least_squares

    start = problem.pack()
    if start.size == 0:
        problem.fun(start)
        return start
    start = np.clip(start, lower, upper)
    start_cost = float(np.sum(problem.fun(start) ** 2))
    try:
        result = least_squares(
            problem.fun,
            start,
            jac=problem.jac,
            bounds=(lower, upper),
            method="trf",
            x_scale=x_scale,
            ftol=_fit._FTOL,
            xtol=_fit._XTOL,
            gtol=_fit._GTOL,
            max_nfev=max_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
        problem.fun(start)
        return start
    best = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(best)) or 2.0 * float(result.cost) > start_cost:
        problem.fun(start)
        return start
    if problem.psi_active:
        best = _polish(problem, best, lower, upper, x_scale)
    problem.fun(best)
    return best


def _polish(
    problem: _Problem,
    x: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    x_scale: FloatArray,
) -> FloatArray:
    """A few Gauss-Newton steps from the trust-region answer.

    The trust region stops once the cost stalls at ``ftol``, which leaves a
    non-linear phase ~1e-4 sigma short of its optimum. A rebuild of a stored solution
    (``FitConstraints.start``) must land back on the stored values to ~1e-9, so both
    the original fit and the rebuild finish on the exact stationary point: Kaufman's
    gradient is exact, so the Gauss-Newton fixed point is the true optimum. Only fits
    with non-linear phases are polished; the unconstrained-equivalent path stays
    step-for-step that of the v1.3 kernel.
    """
    cost = float(np.sum(problem.fun(x) ** 2))
    for _ in range(_POLISH_STEPS):
        jac = problem.jac(x)
        residual = problem.fun(x)
        delta = np.linalg.lstsq(jac, -residual, rcond=None)[0]
        trial = np.clip(x + delta, lower, upper)
        if not np.all(np.isfinite(trial)):
            break
        trial_cost = float(np.sum(problem.fun(trial) ** 2))
        if not trial_cost <= cost:
            break
        x, cost = trial, trial_cost
        if float(np.max(np.abs(delta) / x_scale)) < _POLISH_TOL:
            break
    problem.fun(x)
    return x


@dataclass
class _State:
    """The running solution of a block schedule."""

    theta: FloatArray
    psi: FloatArray
    contrib: FloatArray  # (K, N)
    offset_model: FloatArray
    nfev: int = 0


class _Solver:
    """The refine policies (spec 1.5) as schedules of exact block minimisations."""

    def __init__(
        self,
        setup: _Setup,
        lower: FloatArray,
        upper: FloatArray,
        scale: float,
        max_nfev: int,
        trace: Callable[[float], None] | None,
    ) -> None:
        self.s = setup
        self.lower = lower  # per root
        self.upper = upper
        self.scale = scale
        self.max_nfev = max_nfev
        self.trace = trace

    def _bounds(self, active: Sequence[int], q: int) -> tuple[
        FloatArray, FloatArray, FloatArray
    ]:
        lower = np.concatenate([self.lower[list(active)], np.full(q, -np.inf)])
        upper = np.concatenate([self.upper[list(active)], np.full(q, np.inf)])
        tiny = float(np.finfo(float).tiny)
        x_scale = np.concatenate(
            [np.full(len(active), max(self.scale, tiny)), np.ones(q)]
        )
        return lower, upper, x_scale

    def full(
        self,
        theta: FloatArray,
        psi: FloatArray,
        *,
        active: Sequence[int],
        psi_active: bool,
    ) -> _Problem:
        return _Problem(
            self.s, self.s.comps, self.s.y, theta, psi,
            active_theta=active, psi_active=psi_active, offsets=self.s.offsets,
        )

    def _record(self, state: _State) -> None:
        if self.trace is None:
            return
        model = state.offset_model + state.contrib.sum(axis=0)
        r = self.s.y - model
        self.trace(float(np.dot(self.s.w, r * r)))

    def _adopt(self, problem: _Problem, state: _State) -> None:
        """Take the full problem's current solution as the running model."""
        for position in range(len(problem.comps)):
            state.contrib[position] = problem.contribution(position)
        state.offset_model = problem.offset_contribution()
        state.theta = problem.theta.copy()
        state.psi = problem.psi.copy()

    def close(self, state: _State) -> _Problem:
        """One joint linear solve; VarPro over ``psi`` (warm start) when it exists."""
        problem = self.full(
            state.theta, state.psi, active=(), psi_active=self.s.n_psi > 0
        )
        lower, upper, x_scale = self._bounds((), problem.n_psi_active)
        _optimize(problem, lower, upper, x_scale, self.max_nfev)
        state.nfev += problem.nfev
        self._adopt(problem, state)
        self._record(state)
        return problem

    def block(self, state: _State, j: int) -> None:
        """Exact minimisation over ``theta_j`` and what it touches.

        The block is ``theta_j`` with the linear columns and the non-linear phases of
        every component whose frequency depends on it (plus the single constant).
        Freeing those phases too keeps this an exact block minimisation of the same
        objective, and it is what makes the sweeps converge for a component with
        per-subset phases: its frequency and its phases are nearly degenerate (a
        frequency step is a phase drift across subsets), and alternating between them
        zig-zags.
        """
        s = self.s
        members = [c for c in s.comps if s.matrix[c.index, j] != 0.0]
        others = [c.index for c in s.comps if s.matrix[c.index, j] == 0.0]
        partial = state.offset_model + state.contrib[others].sum(axis=0)
        offsets = "single" if s.offsets == "single" else "none"
        problem = _Problem(
            s, members, s.y - partial, state.theta, state.psi,
            active_theta=(j,), psi_active=True, offsets=offsets,
        )
        lower, upper, x_scale = self._bounds((j,), problem.n_psi_active)
        _optimize(problem, lower, upper, x_scale, self.max_nfev)
        state.nfev += problem.nfev
        state.theta = problem.theta.copy()
        state.psi = problem.psi.copy()
        for position, comp in enumerate(members):
            state.contrib[comp.index] = problem.contribution(position)
        if offsets == "single":
            state.offset_model = state.offset_model + problem.offset_contribution()
        self._record(state)

    def run(
        self, refine: str, sweeps: int, theta: FloatArray, psi: FloatArray
    ) -> tuple[FloatArray, FloatArray, int, list[int]]:
        """Run a policy; returns ``(theta, psi, nfev, roots moved)``."""
        s = self.s
        k = len(s.comps)
        p = s.n_theta
        if refine == "simultaneous" and p + s.n_psi > 0:
            problem = self.full(theta, psi, active=range(p), psi_active=s.n_psi > 0)
            lower, upper, x_scale = self._bounds(range(p), problem.n_psi_active)
            _optimize(problem, lower, upper, x_scale, self.max_nfev)
            return problem.theta, problem.psi, problem.nfev, list(range(p))
        n = int(s.dt.size)
        state = _State(
            theta=theta.copy(), psi=psi.copy(),
            contrib=np.zeros((k, n), dtype=np.float64),
            offset_model=np.zeros(n, dtype=np.float64),
        )
        self.close(state)
        if refine == "none" or p == 0:
            return state.theta, state.psi, state.nfev, []
        if refine == "last":
            roots = [j for j in range(p) if s.matrix[k - 1, j] != 0.0] if k else []
            passes = 1
        else:
            roots = list(range(p))
            passes = max(1, int(sweeps))
        if not roots:
            return state.theta, state.psi, state.nfev, []
        for _ in range(passes):
            for j in roots:
                self.block(state, j)
            self.close(state)
        return state.theta, state.psi, state.nfev, roots


def _relaxed_start(
    setup: _Setup, theta: FloatArray, psi: FloatArray
) -> FloatArray:
    """Starting phases by linear relaxation (spec 1.4).

    Every component with a non-linear phase is solved, at the start frequencies, as
    free ``(a, b)`` pairs — one per variation subset when anything varies — and its
    phases are read off ``z_s = b_s + i a_s``.
    """
    s = setup
    relaxed: list[_Comp] = []
    for c in s.comps:
        if c.n_psi == 0:
            relaxed.append(c)
        else:
            per_subset = c.amp_vary or c.phase_vary
            relaxed.append(_Comp(c.index, "pair", per_subset, per_subset, None, None,
                                 0, 0))
    view = _Setup(**{**s.__dict__, "comps": relaxed})
    problem = _Problem(
        view, relaxed, s.y, theta, np.zeros(0), active_theta=(), psi_active=False,
        offsets=s.offsets,
    )
    problem.fun(np.zeros(0))
    out = psi.copy()
    weights = _subset_weights(s)
    for position, c in enumerate(s.comps):
        if c.n_psi == 0:
            continue
        start = problem.col_start[position]
        groups = s.n_v if (c.amp_vary or c.phase_vary) else 1
        a = problem._beta[start : start + 2 * groups : 2]
        b = problem._beta[start + 1 : start + 2 * groups : 2]
        z = b + 1j * a
        if c.amp_vary:  # common phase, per-subset amplitude: the axial mean
            phi0 = 0.5 * float(np.angle(np.sum(weights * z * z)))
            if float(np.sum(weights * np.real(z * np.exp(-1j * phi0)))) < 0.0:
                phi0 += np.pi
            out[c.psi_start] = phi0
            continue
        sign = 1.0 if c.amp_fixed is None or c.amp_fixed >= 0.0 else -1.0
        out[c.psi_start : c.psi_start + c.n_psi] = np.angle(sign * z)
    return np.mod(out, _TWO_PI)


def _subset_weights(setup: _Setup) -> FloatArray:
    """``W_s = sum_{i in s} w_i`` over the variation subsets."""
    if setup.vcodes is None:
        return np.zeros(0)
    out: FloatArray = np.asarray(
        np.bincount(setup.vcodes, weights=setup.w, minlength=setup.n_v),
        dtype=np.float64,
    )
    return out


def _start_phases(
    setup: _Setup, theta: FloatArray, start: FitStart | None
) -> FloatArray:
    psi = np.zeros(setup.n_psi, dtype=np.float64)
    if setup.n_psi == 0:
        return psi
    if start is None:
        return _relaxed_start(setup, theta, psi)
    phase = np.asarray(start.phase, dtype=np.float64)
    subset = (
        None if start.subset_phase is None
        else np.asarray(start.subset_phase, dtype=np.float64)
    )
    for c in setup.comps:
        if c.n_psi == 0:
            continue
        if c.phase_vary:
            values = np.full(c.n_psi, float(phase[c.index]))
            if subset is not None:
                row = subset[c.index]
                values = np.where(np.isfinite(row), row, values)
            psi[c.psi_start : c.psi_start + c.n_psi] = values
        else:
            psi[c.psi_start] = float(phase[c.index])
    if not np.all(np.isfinite(psi)):
        return _relaxed_start(setup, theta, np.zeros_like(psi))
    return np.mod(psi, _TWO_PI)


# -- natural parameters ---------------------------------------------------------------


@dataclass
class _Natural:
    """Per-component natural values at the solution (signed amplitudes)."""

    amp: list[FloatArray]  # (S_v,) or (1,) per component
    phase: list[FloatArray]
    offsets: FloatArray  # (1,), (S_o,) or (0,)


def _natural(setup: _Setup, problem: _Problem) -> _Natural:
    """Amplitudes and phases off the final linear solution, signs canonicalised."""
    s = setup
    weights = _subset_weights(s)
    amps: list[FloatArray] = []
    phases: list[FloatArray] = []
    beta = problem._beta
    for position, c in enumerate(s.comps):
        start = problem.col_start[position]
        if c.kind == "pair":
            groups = s.n_v if c.amp_vary else 1
            a = beta[start : start + 2 * groups : 2]
            b = beta[start + 1 : start + 2 * groups : 2]
            amp = np.hypot(a, b)
            phase = np.arctan2(a, b)
        else:
            if c.kind == "known":
                assert c.amp_fixed is not None
                amp = np.array([c.amp_fixed])
            elif c.amp_vary:
                amp = beta[start : start + s.n_v].copy()
            else:
                amp = beta[start : start + 1].copy()
            if c.phase_fixed is not None:
                phase = np.array([c.phase_fixed])
            else:
                phase = problem.psi[c.psi_start : c.psi_start + c.n_psi].copy()
            # Sign canonicalisation (spec 1.4): only where the phase is free can a
            # flip of every amplitude be absorbed by phase + pi.
            if c.kind == "signed" and c.phase_fixed is None:
                flip = (
                    float(np.sum(weights * amp)) < 0.0 if c.amp_vary
                    else float(amp[0]) < 0.0
                )
                if flip:
                    amp = -amp
                    phase = phase + np.pi
        amps.append(np.asarray(amp, dtype=np.float64))
        phases.append(np.mod(np.asarray(phase, dtype=np.float64), _TWO_PI))
    n_comp = problem.n_comp_cols
    offsets = beta[n_comp:].copy() if s.offsets != "none" else np.zeros(0)
    return _Natural(amps, phases, offsets)


def _broadcast(values: FloatArray, n_v: int) -> FloatArray:
    return np.full(n_v, float(values[0])) if values.size == 1 else values


def _headline(
    setup: _Setup, comp: _Comp, amp: FloatArray, phase: FloatArray
) -> tuple[float, float, complex]:
    """Headline ``(A_k, phi_k, zbar)`` of a component (spec 1.6)."""
    if not (comp.amp_vary or comp.phase_vary):
        a, p = float(amp[0]), float(phase[0])
        return a, p, a * complex(np.exp(1j * p))
    weights = _subset_weights(setup)
    total = float(weights.sum())
    a_s = _broadcast(amp, setup.n_v)
    p_s = _broadcast(phase, setup.n_v)
    if comp.phase_fixed is not None:
        mean = float(np.dot(weights, a_s) / total)
        return mean, float(phase[0]), mean * complex(np.exp(1j * float(phase[0])))
    zbar = complex(np.sum(weights * a_s * np.exp(1j * p_s)) / total)
    return abs(zbar), float(np.mod(np.angle(zbar), _TWO_PI)), zbar


@dataclass
class _Layout:
    """Free natural parameters in covariance order and their bookkeeping."""

    names: list[str]
    kinds: list[tuple[str, int, int]]  # ("theta", j, -1) / ("amp"|"phase", k, g) / ...


def _layout(setup: _Setup, free_index: IntArray) -> _Layout:
    names: list[str] = []
    kinds: list[tuple[str, int, int]] = []
    for j in range(setup.n_theta):
        names.append(f"f[{int(free_index[j])}]")
        kinds.append(("theta", j, -1))
    for c in setup.comps:
        k = c.index
        if c.amp_fixed is None:
            if c.amp_vary:
                for g in range(setup.n_v):
                    names.append(f"A[{k}|{g}]")
                    kinds.append(("amp", k, g))
            else:
                names.append(f"A[{k}]")
                kinds.append(("amp", k, -1))
        if c.phase_fixed is None:
            if c.phase_vary:
                for g in range(setup.n_v):
                    names.append(f"phi[{k}|{g}]")
                    kinds.append(("phase", k, g))
            else:
                names.append(f"phi[{k}]")
                kinds.append(("phase", k, -1))
    if setup.offsets == "single":
        names.append("offset")
        kinds.append(("offset", -1, -1))
    elif setup.offsets == "groups":
        for g in range(setup.n_o):
            names.append(f"offset[{g}]")
            kinds.append(("offset", -1, g))
    return _Layout(names, kinds)


def _jacobian_rows(
    setup: _Setup,
    layout: _Layout,
    natural: _Natural,
    freqs: FloatArray,
    shift: float,
    rows: slice,
    columns: Sequence[int],
) -> FloatArray:
    """Weighted natural-parameter Jacobian of ``rows`` for the layout ``columns``."""
    s = setup
    dt = s.dt[rows]
    dt_c = dt - shift
    vc = s.vcodes[rows] if s.vcodes is not None else None
    oc = s.ocodes[rows] if s.ocodes is not None else None
    sins: list[FloatArray] = []
    quads: list[FloatArray] = []
    lever: list[FloatArray] = []
    for c in s.comps:
        f = float(freqs[c.index])
        amp = natural.amp[c.index]
        phase = natural.phase[c.index]
        a_i = amp[vc] if amp.size > 1 else float(amp[0])
        p_i = phase[vc] if phase.size > 1 else float(phase[0])
        if c.phase_free:
            arg = _TWO_PI * f * dt_c + (p_i + _TWO_PI * f * shift)
            lever.append(_TWO_PI * dt_c)
        else:
            arg = _TWO_PI * f * dt + p_i
            lever.append(_TWO_PI * dt)
        sins.append(np.sin(arg))
        quads.append(a_i * np.cos(arg))
    jac = np.zeros((dt.size, len(columns)), dtype=np.float64)
    for out_col, col in enumerate(columns):
        kind, a, g = layout.kinds[col]
        if kind == "theta":
            for c in s.comps:
                m = float(s.matrix[c.index, a])
                if m != 0.0:
                    jac[:, out_col] += m * lever[c.index] * quads[c.index]
        elif kind == "amp":
            jac[:, out_col] = sins[a] if g < 0 else sins[a] * (vc == g)
        elif kind == "phase":
            jac[:, out_col] = quads[a] if g < 0 else quads[a] * (vc == g)
        elif g < 0:
            jac[:, out_col] = 1.0
        else:
            assert oc is not None
            jac[:, out_col] = oc == g
    jac *= s.sw[rows, None]
    return jac


def _row_group_sums(groups: _Groups, rows: slice, weighted: FloatArray) -> FloatArray:
    """``D^T z`` restricted to the points ``rows`` (``weighted`` holds those rows)."""
    from scipy.sparse import csr_matrix

    codes = groups.codes[rows]
    summer = csr_matrix(
        (groups.sw[rows], (codes, np.arange(codes.size))),
        shape=(groups.n, codes.size),
        dtype=np.float64,
    )
    out: FloatArray = np.asarray(summer @ weighted, dtype=np.float64)
    return out


def _natural_gram(
    setup: _Setup,
    layout: _Layout,
    natural: _Natural,
    theta: FloatArray,
    shift: float,
    with_r: bool,
) -> tuple[FloatArray, FloatArray | None]:
    """``J^T W J`` of the natural parameters about the epoch ``t_ref + shift``.

    Free phases are parameterised at the shifted epoch (``phi_c = phi + 2 pi f
    shift``), where each is decorrelated from its frequency; a fixed phase stays at
    ``t_ref``. Accumulated in row blocks, with the triangular factor of the weighted
    design (for its rank) built alongside by a tall-skinny QR.

    The zero point of a group has the weighted indicator as its column, so its
    blocks are known in closed form: ``D^T D = diag(W_g)`` and ``D^T J`` are group
    sums, and the triangular factor of ``[D, J]`` is ``[[diag(sqrt W), D^T J /
    sqrt W], [0, R(M_D J)]]`` with ``R(M_D J)`` from a second pass over the
    group-centred columns. Nothing of size ``N x S_o`` is formed.
    """
    from scipy.linalg import qr

    s = setup
    n = int(s.dt.size)
    n_p = len(layout.names)
    gram = np.zeros((n_p, n_p), dtype=np.float64)
    r_factor: FloatArray | None = np.zeros((0, n_p)) if with_r else None
    if n_p == 0 or n == 0:
        return gram, r_factor
    freqs = s.frequencies(theta)
    groups = s.groups
    columns = [
        col
        for col, (kind, _, g) in enumerate(layout.kinds)
        if groups is None or not (kind == "offset" and g >= 0)
    ]
    n_c = len(columns)
    block = max(1, _COV_BLOCK_ELEMS // max(n_c, 1))
    gram_c = np.zeros((n_c, n_c), dtype=np.float64)
    cross = np.zeros((groups.n if groups is not None else 0, n_c), dtype=np.float64)
    r_c: FloatArray | None = np.zeros((0, n_c)) if with_r else None
    for start in range(0, n, block):
        rows = slice(start, min(start + block, n))
        jac = _jacobian_rows(s, layout, natural, freqs, shift, rows, columns)
        gram_c += jac.T @ jac
        if groups is not None:
            cross += _row_group_sums(groups, rows, jac)
        elif r_c is not None:
            stacked = np.vstack([r_c, jac])
            r_c = qr(stacked, mode="r", check_finite=False)[0][: min(
                n_c, stacked.shape[0]
            )]
    if groups is None:
        return gram_c, r_c
    offsets = {g: col for col, (kind, _, g) in enumerate(layout.kinds)
               if kind == "offset" and g >= 0}
    off = np.array([offsets[g] for g in range(groups.n)], dtype=np.int64)
    idx = np.asarray(columns, dtype=np.int64)
    gram[np.ix_(idx, idx)] = gram_c
    gram[np.ix_(off, idx)] = cross
    gram[np.ix_(idx, off)] = cross.T
    gram[off, off] = groups.total
    if r_c is None:
        return gram, None
    means = cross / groups.total[:, None]
    for start in range(0, n if n_c else 0, block):
        rows = slice(start, min(start + block, n))
        jac = _jacobian_rows(s, layout, natural, freqs, shift, rows, columns)
        jac -= groups.sw[rows, None] * means[groups.codes[rows]]
        stacked = np.vstack([r_c, jac])
        r_c = qr(stacked, mode="r", check_finite=False)[0][: min(
            n_c, stacked.shape[0]
        )]
    root = np.sqrt(groups.total)
    top = np.zeros((groups.n, n_p), dtype=np.float64)
    top[:, idx] = cross / root[:, None]
    top[np.arange(groups.n), off] = root
    bottom = np.zeros((r_c.shape[0], n_p), dtype=np.float64)
    bottom[:, idx] = r_c
    r_factor = np.vstack([top, bottom])
    return gram, r_factor


def _pseudo_inverse(gram: FloatArray) -> tuple[FloatArray, np.ndarray[Any, Any]]:
    """Jacobi-scaled eigen pseudo-inverse (as ``fit._covariance``) + degenerate mask.

    A parameter whose column is identically zero, or that carries weight in an
    eigen-direction below the cutoff, is unidentifiable: its row and column are NaN.
    """
    if gram.size == 0:
        return gram.copy(), np.zeros(0, dtype=bool)
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    live = norms > 0.0
    norms = np.where(live, norms, 1.0)
    values, vectors = np.linalg.eigh(gram / np.outer(norms, norms))
    cutoff = _fit._RCOND * float(max(values.max(), 0.0))
    keep = values > cutoff
    inverse_values = np.where(keep, 1.0 / np.where(keep, values, 1.0), 0.0)
    inverse: FloatArray = ((vectors * inverse_values) @ vectors.T) / np.outer(
        norms, norms
    )
    degenerate = ~live
    if not np.all(keep):
        dropped = np.abs(vectors[:, ~keep]) > _DEGENERATE_WEIGHT
        degenerate = degenerate | np.any(dropped, axis=1)
    if np.any(degenerate):
        inverse[degenerate, :] = np.nan
        inverse[:, degenerate] = np.nan
    return inverse, degenerate


def _masked_product(left: FloatArray, middle: FloatArray) -> FloatArray:
    """``left @ middle @ left.T`` where a NaN spreads only through non-zero weights."""
    nan = np.isnan(middle)
    clean = np.where(nan, 0.0, middle)
    out: FloatArray = left @ clean @ left.T
    touched = (np.abs(left) > 0.0).astype(np.float64)
    spread = touched @ nan.astype(np.float64) @ touched.T
    out[spread > 0.0] = np.nan
    return out


def _rank(r_factor: FloatArray | None, gram: FloatArray) -> tuple[int, float]:
    """Rank and condition of the column-scaled design from its triangular factor."""
    from scipy.linalg import qr

    n_p = gram.shape[0]
    if r_factor is None or n_p == 0 or r_factor.size == 0:
        return 0, float("nan")
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    scaled = r_factor / np.where(norms > 0.0, norms, 1.0)
    # Pivoted QR of R is pivoted QR of the design: an orthogonal factor on the left
    # changes no column norm the pivoting looks at.
    r_piv = qr(scaled, mode="r", pivoting=True, check_finite=False)[0]
    diagonal = np.abs(np.diag(r_piv))
    if diagonal.size == 0 or diagonal[0] == 0.0:
        return 0, float("inf")
    rank = int(np.count_nonzero(diagonal > _RANK_TOL * diagonal[0]))
    return rank, float(diagonal[0] / diagonal[rank - 1])


def fit_constrained(
    time: FloatArray,
    value: FloatArray,
    error: FloatArray | None,
    frequencies: FloatArray,
    *,
    constraints: FitConstraints,
    fit_mean: bool = True,
    t_ref: float | None = None,
    refine: str = "cyclic",
    sweeps: int = 1,
    frequency_bounds: tuple[Any, Any] | None = None,
    max_nfev: int = 200,
    covariance: bool = True,
    trace: Callable[[float], None] | None = None,
) -> _fit.MultiSineFit:
    """The constrained multi-sine fit behind ``fit_multisine(..., constraints=...)``.

    Parameters are those of :func:`~cuperiod.prewhiten.fit.fit_multisine`. ``trace``
    (testing aid) receives the weighted RSS of the running model after the initial
    solve, after every block and after every close of a block schedule.

    Raises
    ------
    ValueError
        For the input errors :func:`~cuperiod.prewhiten.fit.fit_multisine` rejects.
    ConstraintError
        For constraints the data cannot honour (see
        :meth:`FitConstraints.validate <cuperiod.prewhiten.constraints.FitConstraints.\
validate>`).
    """
    t = np.ascontiguousarray(np.asarray(time, dtype=np.float64)).ravel()
    y = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).ravel()
    if t.shape != y.shape:
        raise ValueError("time and value must have the same length")
    freqs = np.ascontiguousarray(np.asarray(frequencies, dtype=np.float64)).ravel()
    if refine not in get_args(_fit.Refinement):
        raise ValueError(
            f"unknown refine policy {refine!r}; choose from "
            f"{get_args(_fit.Refinement)}"
        )
    _fit._check_inputs(t, y, error, freqs, "none")
    n = int(t.size)
    k = int(freqs.size)
    if k == 0 and not fit_mean:
        raise ValueError("fit_multisine needs at least one frequency or fit_mean=True")
    constraints.validate(freqs, n, fit_mean=fit_mean, t_ref=t_ref)
    comps_c = constraints.expanded(k)

    w0, _, weighted = _fit._as_weights(n, error)
    if constraints.point_weights is not None:
        omega = np.asarray(constraints.point_weights, dtype=np.float64)
        w = w0 * (omega / float(np.mean(omega)))
    else:
        w = w0.copy()
    sw = np.sqrt(w)
    ref = weighted_epoch(t, w) if t_ref is None else float(t_ref)
    dt = t - ref
    baseline = float(t.max() - t.min()) if n else 0.0
    rayleigh = 1.0 / baseline if baseline > 0.0 else 1.0

    resolved = constraints.resolve(freqs)
    theta = resolved.theta(freqs)
    if refine != "none" and np.any(theta <= 0.0):
        raise ValueError(
            "frequencies to be refined must be positive (pass refine='none' to fit "
            "fixed frequencies as given)"
        )
    predicted = resolved.frequencies(theta)
    if frequency_bounds is not None:
        raw_bounds: tuple[Any, Any] = frequency_bounds
    elif k:
        low = max(0.5 * float(predicted.min()), 1e-12)
        raw_bounds = (low, max(2.0 * float(predicted.max()), low * (1.0 + 1e-9)))
    else:
        raw_bounds = (1e-12, 1.0)
    lo, hi = _fit._normalize_bounds(raw_bounds, k)
    root_lo, root_hi = lo[resolved.free_index], hi[resolved.free_index]

    vcodes: IntArray | None = None
    n_v = 0
    if constraints.variation_groups is not None:
        vcodes = np.asarray(constraints.variation_groups).astype(np.int64)
        n_v = int(vcodes.max()) + 1 if vcodes.size else 0
    ocodes: IntArray | None = None
    n_o = 0
    if constraints.offset_groups is not None:
        ocodes = np.asarray(constraints.offset_groups).astype(np.int64)
        n_o = int(ocodes.max()) + 1 if ocodes.size else 0
    offsets = "groups" if ocodes is not None else ("single" if fit_mean else "none")
    vmasks = (
        [(vcodes == g).astype(np.float64) for g in range(n_v)]
        if vcodes is not None else []
    )
    setup = _Setup(
        dt=dt, y=y, w=w, sw=sw, comps=_classify(comps_c, n_v),
        matrix=resolved.matrix, constant=resolved.constant,
        vcodes=vcodes, n_v=n_v, vmasks=vmasks, ocodes=ocodes, n_o=n_o,
        offsets=offsets,
        groups=(
            _Groups(ocodes, w, sw, n_o)
            if ocodes is not None and n * n_o >= _PROJECT_GROUPS_ELEMENTS
            else None
        ),
    )

    if refine != "none":
        theta = np.clip(theta, root_lo, root_hi)
    psi = _start_phases(setup, theta, constraints.start)
    solver = _Solver(setup, root_lo, root_hi, rayleigh, max_nfev, trace)
    theta, psi, nfev, moved = solver.run(refine, sweeps, theta, psi)

    final = solver.full(theta, psi, active=(), psi_active=False)
    final.fun(np.zeros(0))
    natural = _natural(setup, final)
    model = final._eta + final.linear_model()
    residuals = np.ascontiguousarray(y - model, dtype=np.float64)
    rss = float(np.dot(w, residuals * residuals))
    freqs_out = setup.frequencies(theta)

    layout = _layout(setup, resolved.free_index)
    n_p = len(layout.names)
    headline = [
        _headline(setup, c, natural.amp[c.index], natural.phase[c.index])
        for c in setup.comps
    ]
    amplitude = np.array([h[0] for h in headline], dtype=np.float64)
    phase = np.mod(np.array([h[1] for h in headline], dtype=np.float64), _TWO_PI)
    if offsets == "single":
        offset = float(natural.offsets[0])
    elif offsets == "groups":
        assert ocodes is not None
        w_o = np.bincount(ocodes, weights=w, minlength=n_o)
        offset = float(np.dot(w_o, natural.offsets) / w_o.sum())
    else:
        offset = 0.0

    cov_p = np.zeros((0, 0), dtype=np.float64)
    cov_q = np.zeros((0, 0), dtype=np.float64)
    degenerate = np.zeros(n_p, dtype=bool)
    rank, condition = -1, float("nan")
    if covariance:
        total = float(w.sum())
        shift = float(np.dot(w, dt) / total) if total > 0.0 and n else 0.0
        gram, r_factor = _natural_gram(setup, layout, natural, theta, shift, True)
        inverse, degenerate = _pseudo_inverse(gram)
        dof = n - n_p
        cov_p = inverse * (rss / dof if dof > 0 else float("nan"))
        cov_p = _shift_phases(cov_p, layout, setup, -_TWO_PI * shift)
        rank, condition = _rank(r_factor, gram)
        transform = _headline_transform(setup, layout, natural, headline, fit_mean)
        cov_q = _masked_product(transform, cov_p)

    detail = _detail(
        setup, constraints, layout, natural, cov_p, degenerate, headline,
        freqs_out, lo, hi, resolved, moved, root_lo, root_hi, theta, refine,
        t, baseline, rank, condition, nfev,
    )
    return _fit.MultiSineFit(
        frequency=freqs_out,
        amplitude=amplitude,
        phase=phase,
        offset=offset,
        t_ref=ref,
        residuals=residuals,
        covariance=cov_q,
        rss=rss,
        n_samples=n,
        n_parameters=n_p,
        weighted=weighted,
        fit_mean=fit_mean,
        detail=detail,
    )


def _shift_phases(
    cov: FloatArray, layout: _Layout, setup: _Setup, lever: float
) -> FloatArray:
    """Carry free phases from the centred epoch back to ``t_ref``.

    ``phi_ref = phi_c - 2 pi shift f_k`` with ``f_k = f0_k + M_k theta``: row and
    column updates over the non-zero ``M`` entries only, so an undetermined
    parameter's NaN does not spread through ``0 * NaN``.
    """
    if cov.size == 0 or lever == 0.0:
        return cov
    theta_col = {j: col for col, (kind, j, _) in enumerate(layout.kinds)
                 if kind == "theta"}
    free_phase = {c.index for c in setup.comps if c.phase_free}
    targets: list[tuple[int, list[tuple[int, float]]]] = []
    for col, (kind, k, _) in enumerate(layout.kinds):
        if kind != "phase" or k not in free_phase:
            continue
        terms = [
            (theta_col[j], float(setup.matrix[k, j]))
            for j in range(setup.n_theta) if setup.matrix[k, j] != 0.0
        ]
        if terms:
            targets.append((col, terms))
    out = cov.copy()
    for col, terms in targets:
        for source, m in terms:
            out[col, :] += lever * m * out[source, :]
    for col, terms in targets:
        for source, m in terms:
            out[:, col] += lever * m * out[:, source]
    return out


def _headline_transform(
    setup: _Setup,
    layout: _Layout,
    natural: _Natural,
    headline: list[tuple[float, float, complex]],
    fit_mean: bool,
) -> FloatArray:
    """``T`` of ``C_q = T C_p T^T`` for ``q = (f, A, phi[, c0])`` (spec 1.6)."""
    s = setup
    k_count = len(s.comps)
    n_q = 3 * k_count + (1 if fit_mean else 0)
    out = np.zeros((n_q, len(layout.kinds)), dtype=np.float64)
    index = {key: col for col, key in enumerate(layout.kinds)}
    weights = _subset_weights(s)
    total = float(weights.sum()) if weights.size else 0.0
    for c in s.comps:
        k = c.index
        for j in range(s.n_theta):
            m = float(s.matrix[k, j])
            if m != 0.0:
                out[k, index[("theta", j, -1)]] = m
        a_row, p_row = k_count + k, 2 * k_count + k
        if not (c.amp_vary or c.phase_vary):
            if ("amp", k, -1) in index:
                out[a_row, index[("amp", k, -1)]] = 1.0
            if ("phase", k, -1) in index:
                out[p_row, index[("phase", k, -1)]] = 1.0
            continue
        amp_s = _broadcast(natural.amp[k], s.n_v)
        phase_s = _broadcast(natural.phase[k], s.n_v)
        if c.phase_fixed is not None:
            # Signed weighted mean of the per-subset amplitudes; the phase is fixed.
            for g in range(s.n_v):
                out[a_row, index[("amp", k, g)]] = weights[g] / total
            continue
        a_k, phi_k, _ = headline[k]
        rot = np.exp(-1j * phi_k)
        derivs: list[tuple[int, complex]] = []
        unit = weights * np.exp(1j * phase_s) / total
        if c.amp_vary:
            derivs += [(index[("amp", k, g)], complex(unit[g])) for g in range(s.n_v)]
        elif ("amp", k, -1) in index:
            derivs.append((index[("amp", k, -1)], complex(np.sum(unit))))
        if c.phase_vary:
            derivs += [
                (index[("phase", k, g)], complex(1j * unit[g] * amp_s[g]))
                for g in range(s.n_v)
            ]
        elif ("phase", k, -1) in index:
            derivs.append((index[("phase", k, -1)], complex(1j * np.sum(unit * amp_s))))
        for col, dz in derivs:
            rotated = rot * dz
            out[a_row, col] = rotated.real
            out[p_row, col] = rotated.imag / a_k if a_k != 0.0 else np.nan
    if fit_mean:
        c0 = n_q - 1
        if s.offsets == "single":
            out[c0, index[("offset", -1, -1)]] = 1.0
        elif s.offsets == "groups":
            assert s.ocodes is not None
            w_o = np.bincount(s.ocodes, weights=s.w, minlength=s.n_o)
            for g in range(s.n_o):
                out[c0, index[("offset", -1, g)]] = w_o[g] / w_o.sum()
    return out


def _detail(
    setup: _Setup,
    constraints: FitConstraints,
    layout: _Layout,
    natural: _Natural,
    cov_p: FloatArray,
    degenerate: np.ndarray[Any, Any],
    headline: list[tuple[float, float, complex]],
    freqs: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    resolved: Any,
    moved: list[int],
    root_lo: FloatArray,
    root_hi: FloatArray,
    theta: FloatArray,
    refine: str,
    t: FloatArray,
    baseline: float,
    rank: int,
    condition: float,
    nfev: int,
) -> FitDetail:
    """Per-subset tables, flags and diagnostics of a constrained fit."""
    s = setup
    k_count = len(s.comps)
    sigma = (
        np.sqrt(np.clip(np.diag(cov_p), 0.0, None)) if cov_p.size
        else np.full(len(layout.kinds), np.nan)
    )
    if cov_p.size:
        sigma = np.where(np.isnan(np.diag(cov_p)), np.nan, sigma)
    index = {key: col for col, key in enumerate(layout.kinds)}
    weights = _subset_weights(s)

    sub_amp = sub_amp_err = sub_phase = sub_phase_err = None
    if s.vcodes is not None:
        sub_amp = np.full((k_count, s.n_v), np.nan)
        sub_amp_err = np.full((k_count, s.n_v), np.nan)
        sub_phase = np.full((k_count, s.n_v), np.nan)
        sub_phase_err = np.full((k_count, s.n_v), np.nan)
        for c in s.comps:
            k = c.index
            sub_amp[k] = _broadcast(natural.amp[k], s.n_v)
            sub_phase[k] = _broadcast(natural.phase[k], s.n_v)
            for g in range(s.n_v):
                if c.amp_vary:
                    sub_amp_err[k, g] = sigma[index[("amp", k, g)]]
                if c.phase_vary:
                    sub_phase_err[k, g] = sigma[index[("phase", k, g)]]
    sub_off = sub_off_err = None
    if s.offsets == "groups":
        sub_off = natural.offsets.copy()
        sub_off_err = np.array(
            [sigma[index[("offset", -1, g)]] for g in range(s.n_o)], dtype=np.float64
        )

    rms = np.empty(k_count, dtype=np.float64)
    for c in s.comps:
        k = c.index
        if (c.amp_vary or c.phase_vary) and weights.size:
            amp_s = _broadcast(natural.amp[k], s.n_v)
            rms[k] = float(np.sqrt(np.dot(weights, amp_s * amp_s) / weights.sum()))
        else:
            rms[k] = abs(float(natural.amp[k][0]))

    # -- flags --------------------------------------------------------------------
    flags: list[set[str]] = [set() for _ in range(k_count)]
    owner: dict[int, int] = {}
    for col, (kind, a, _) in enumerate(layout.kinds):
        if kind in ("amp", "phase"):
            owner[col] = a
        elif kind == "theta":
            owner[col] = -1 - a  # resolved below through M
    for bad in np.flatnonzero(degenerate).tolist():
        who = owner.get(int(bad))
        if who is None:
            continue
        if who >= 0:
            flags[who].add("degenerate")
        else:
            j = -1 - who
            for c in s.comps:
                if s.matrix[c.index, j] != 0.0:
                    flags[c.index].add("degenerate")
    status = resolved.status
    span_v = _spans(t, s.vcodes, s.n_v) if s.vcodes is not None else None
    span_o = _spans(t, s.ocodes, s.n_o) if s.ocodes is not None else None
    median_offset_span = (
        float(np.median(span_o)) if span_o is not None and span_o.size else None
    )
    for c in s.comps:
        k = c.index
        f = float(freqs[k])
        if status[k] == "fixed":
            flags[k].add("fixed_f")
        elif status[k] == "tied":
            flags[k].add("tied")
            if not np.any(s.matrix[k] != 0.0):
                flags[k].add("fixed_f")
            if not (lo[k] <= f <= hi[k]) or f <= 0.0:
                flags[k].add("out_of_band")
            others = np.delete(freqs, k)
            if baseline > 0.0 and others.size and float(
                np.min(np.abs(others - f))
            ) < 0.5 / baseline:
                flags[k].add("unresolved")
        if c.amp_fixed is not None:
            flags[k].add("fixed_a")
        if c.phase_fixed is not None:
            flags[k].add("fixed_phi")
        if headline[k][0] < 0.0:
            flags[k].add("negative_amplitude")
        if median_offset_span is not None and abs(f) * median_offset_span < 0.5:
            flags[k].add("low_frequency")
        if c.phase_vary and np.any(s.matrix[k] != 0.0):
            flags[k].add("weak_frequency")
        if (c.amp_vary or c.phase_vary) and span_v is not None and np.any(
            span_v * abs(f) < 1.0
        ):
            flags[k].add("short_subset")
    if refine != "none":
        for j in moved:
            width = float(root_hi[j] - root_lo[j])
            value = float(theta[j])
            if width > 0.0 and min(value - root_lo[j], root_hi[j] - value) <= (
                _AT_BOUND * width
            ):
                flags[int(resolved.free_index[j])].add("at_bound")

    return FitDetail(
        constraints=constraints,
        parameter_names=tuple(layout.names),
        parameter_covariance=cov_p,
        flags=tuple(frozenset(f) for f in flags),
        rms_amplitude=rms,
        subset_amplitude=sub_amp,
        subset_amplitude_error=sub_amp_err,
        subset_phase=sub_phase,
        subset_phase_error=sub_phase_err,
        subset_offset=sub_off,
        subset_offset_error=sub_off_err,
        rank=rank,
        n_columns=len(layout.names),
        condition=condition,
        nfev=nfev,
    )


def _spans(t: FloatArray, codes: IntArray, n_groups: int) -> FloatArray:
    first = np.full(n_groups, np.inf)
    last = np.full(n_groups, -np.inf)
    np.minimum.at(first, codes, t)
    np.maximum.at(last, codes, t)
    out: FloatArray = last - first
    return out


__all__ = ["fit_constrained"]
