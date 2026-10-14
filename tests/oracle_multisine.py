"""A brute-force reference for constrained multi-sine fits.

Independent of :mod:`cuperiod.prewhiten.constraints` and of the variable-projection
kernel in ``_constrained.py``: it works on plain arrays and optimises **every** free
natural parameter at once with ``scipy.optimize.least_squares`` and a dense,
analytic Jacobian.

The model, with ``tau = t - t_ref`` and ``f = constant + matrix @ theta`` (ties
substituted by the caller),

    y_i = o[g_o(i)] + sum_k A[k, a_k(i)] sin(2 pi f_k tau_i + phi[k, p_k(i)])

has the free parameters, in this order (the order of ``FitDetail.parameter_names``):

    p = [theta; for each component: its free amplitudes, then its free phases; offsets]

where a component's amplitude (phase) is ``"common"`` (one value), ``"subset"`` (one
per variation subset) or a number (fixed, not a parameter).

Why analytic: scipy's default ``"3-point"`` Jacobian uses a relative step, which on
long baselines makes the frequency columns — and with them the oracle's own sigmas —
wrong by 5e-4 (122 d) up to order one (3660 d). The analytic Jacobian is checked
against central differences with absolute steps (:func:`jacobian_error`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TWO_PI = 2.0 * np.pi
Mode = Any  # "common" | "subset" | float


@dataclass
class OracleSpec:
    """The data and the model layout.

    ``offsets`` is ``"none"``, ``"single"`` or an ``(N,)`` array of zero-point codes;
    ``vcodes`` the ``(N,)`` variation-subset codes (``None`` when nothing varies).
    """

    t: np.ndarray
    y: np.ndarray
    w: np.ndarray
    t_ref: float
    matrix: np.ndarray
    constant: np.ndarray
    amp: Sequence[Mode]
    phase: Sequence[Mode]
    vcodes: np.ndarray | None = None
    offsets: Any = "single"
    _layout: list[tuple[str, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tau = np.asarray(self.t, dtype=np.float64) - self.t_ref
        self.n_v = int(self.vcodes.max()) + 1 if self.vcodes is not None else 1
        self.ocodes = None if isinstance(self.offsets, str) else np.asarray(
            self.offsets, dtype=np.int64
        )
        self.n_o = int(self.ocodes.max()) + 1 if self.ocodes is not None else 0
        layout: list[tuple[str, int, int]] = []
        for j in range(self.matrix.shape[1]):
            layout.append(("f", j, -1))
        for k in range(len(self.amp)):
            for name, mode in (("A", self.amp[k]), ("phi", self.phase[k])):
                if mode == "common":
                    layout.append((name, k, -1))
                elif mode == "subset":
                    layout.extend((name, k, g) for g in range(self.n_v))
        if self.ocodes is not None:
            layout.extend(("o", -1, g) for g in range(self.n_o))
        elif self.offsets == "single":
            layout.append(("o", -1, -1))
        self._layout = layout

    @property
    def n_params(self) -> int:
        return len(self._layout)

    @property
    def layout(self) -> list[tuple[str, int, int]]:
        return list(self._layout)

    @property
    def baseline(self) -> float:
        return float(np.ptp(self.t))

    # -- unpacking -------------------------------------------------------------------
    def values(self, p: np.ndarray) -> tuple[np.ndarray, list[Any], list[Any], Any]:
        """Frequencies and per-point amplitudes, phases and offset."""
        k_count = len(self.amp)
        n = self.tau.size
        theta = np.zeros(self.matrix.shape[1])
        amps: list[Any] = [None] * k_count
        phases: list[Any] = [None] * k_count
        subset_amp: dict[int, np.ndarray] = {}
        subset_phase: dict[int, np.ndarray] = {}
        offset: Any = np.zeros(n)
        offsets = np.zeros(max(self.n_o, 1))
        for value, (kind, a, g) in zip(p, self._layout, strict=True):
            if kind == "f":
                theta[a] = value
            elif kind == "A":
                if g < 0:
                    amps[a] = value
                else:
                    subset_amp.setdefault(a, np.zeros(self.n_v))[g] = value
            elif kind == "phi":
                if g < 0:
                    phases[a] = value
                else:
                    subset_phase.setdefault(a, np.zeros(self.n_v))[g] = value
            elif g < 0:
                offset = np.full(n, value)
            else:
                offsets[g] = value
        if self.ocodes is not None:
            offset = offsets[self.ocodes]
        for k in range(k_count):
            if k in subset_amp:
                amps[k] = subset_amp[k][self.vcodes]
            elif amps[k] is None:
                amps[k] = float(self.amp[k])
            if k in subset_phase:
                phases[k] = subset_phase[k][self.vcodes]
            elif phases[k] is None:
                phases[k] = float(self.phase[k])
        freqs = self.constant + self.matrix @ theta
        return freqs, amps, phases, offset

    # -- model and Jacobian ----------------------------------------------------------
    def model(self, p: np.ndarray) -> np.ndarray:
        freqs, amps, phases, offset = self.values(p)
        out = np.array(offset, dtype=np.float64, copy=True)
        for k in range(len(self.amp)):
            out += amps[k] * np.sin(TWO_PI * freqs[k] * self.tau + phases[k])
        return out

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """Dense analytic ``d model / d p``, ``(N, n_p)``."""
        freqs, amps, phases, _ = self.values(p)
        n = self.tau.size
        jac = np.zeros((n, self.n_params))
        args = [TWO_PI * freqs[k] * self.tau + phases[k] for k in range(len(self.amp))]
        for col, (kind, a, g) in enumerate(self._layout):
            if kind == "f":
                total = np.zeros(n)
                for k in range(len(self.amp)):
                    m = self.matrix[k, a]
                    if m:
                        total += m * TWO_PI * self.tau * amps[k] * np.cos(args[k])
                jac[:, col] = total
                continue
            if kind == "o":
                jac[:, col] = 1.0 if g < 0 else (self.ocodes == g).astype(float)
                continue
            mask = np.ones(n) if g < 0 else (self.vcodes == g).astype(float)
            if kind == "A":
                jac[:, col] = mask * np.sin(args[a])
            else:
                jac[:, col] = mask * amps[a] * np.cos(args[a])
        return jac

    def residual(self, p: np.ndarray) -> np.ndarray:
        return np.sqrt(self.w) * (self.y - self.model(p))

    def rss(self, p: np.ndarray) -> float:
        r = self.residual(p)
        return float(r @ r)

    def covariance(self, p: np.ndarray) -> np.ndarray:
        """``s^2 (J^T W J)^-1`` with ``s^2 = RSS / (N - n_p)``; columns unit-scaled."""
        jac = self.jacobian(p) * np.sqrt(self.w)[:, None]
        gram = jac.T @ jac
        norms = np.sqrt(np.diag(gram))
        inverse = np.linalg.inv(gram / np.outer(norms, norms)) / np.outer(norms, norms)
        dof = self.tau.size - self.n_params
        return inverse * self.rss(p) / dof

    def gauss_newton_step(self, p: np.ndarray) -> np.ndarray:
        jac = self.jacobian(p) * np.sqrt(self.w)[:, None]
        return np.linalg.lstsq(jac, self.residual(p), rcond=None)[0]


def jacobian_error(spec: OracleSpec, p: np.ndarray) -> float:
    """Worst column error of the analytic Jacobian against central differences.

    Absolute steps ``1e-6 / T`` for frequencies and ``1e-6 max(1, |p|)`` otherwise;
    the error of a column is ``|J_analytic - J_cd| / |J_analytic|``.
    """
    analytic = spec.jacobian(p)
    worst = 0.0
    for col, (kind, _, _) in enumerate(spec.layout):
        h = 1e-6 / spec.baseline if kind == "f" else 1e-6 * max(1.0, abs(p[col]))
        up, down = p.copy(), p.copy()
        up[col] += h
        down[col] -= h
        numeric = (spec.model(up) - spec.model(down)) / (2.0 * h)
        scale = float(np.linalg.norm(analytic[:, col]))
        if scale > 0.0:
            error = float(np.linalg.norm(numeric - analytic[:, col])) / scale
            worst = max(worst, error)
    return worst


@dataclass
class OracleResult:
    params: np.ndarray
    rss: float
    covariance: np.ndarray
    sigma: np.ndarray
    status: int
    step_sigma: float  # largest remaining Gauss-Newton step, in sigma


def solve(spec: OracleSpec, truth: np.ndarray, *, seed: int = 0) -> OracleResult:
    """Optimise every free parameter from the truth plus 5 sigma.

    The perturbation is ``+-5 sigma`` per parameter (seeded signs), sigma from the
    analytic covariance at the truth: never the answer of the solver under test, where
    the oracle would stop after one step and prove nothing.
    """
    from scipy.optimize import least_squares

    sigma0 = np.sqrt(np.diag(spec.covariance(truth)))
    rng = np.random.default_rng(seed)
    start = truth + 5.0 * sigma0 * rng.choice([-1.0, 1.0], size=truth.size)
    result = least_squares(
        spec.residual,
        start,
        jac=lambda p: -spec.jacobian(p) * np.sqrt(spec.w)[:, None],
        method="trf",
        x_scale="jac",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        max_nfev=2000,
    )
    params = np.asarray(result.x, dtype=np.float64)
    cov = spec.covariance(params)
    sigma = np.sqrt(np.diag(cov))
    step = spec.gauss_newton_step(params)
    return OracleResult(
        params=params,
        rss=spec.rss(params),
        covariance=cov,
        sigma=sigma,
        status=int(result.status),
        step_sigma=float(np.max(np.abs(step) / sigma)),
    )
