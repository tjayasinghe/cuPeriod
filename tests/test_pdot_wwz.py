"""WWZ: parity with Foster's formulas, ridge recovery, N_eff masking, guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.errors import InsufficientDataError
from synth import synthetic_multiband_sine, synthetic_period_change, synthetic_sine

PERIOD = 0.6234

#: Foster's decay constant in the VARTOOLS convention (the function's default).
DEFAULT_C = 1.0 / (8.0 * math.pi**2)

#: The long-period-variable convention: a 3.56-period window, 4.5% resolution.
LPV_C = 0.001


def _reference_wwz(
    t: np.ndarray, y: np.ndarray, tau_val: float, f_val: float, c: float
) -> tuple[float, float, float]:
    """One WWZ cell straight from Foster (1996) — the algorithm of record.

    Explicit loops and a generic least-squares solve, written from the paper
    rather than from the implementation: weights ``exp(-c*omega^2*(t-tau)^2)``
    with ``omega`` *angular*, the three trial functions ``1``, ``cos``, ``sin``,
    the weighted inner product ``<f|g> = sum(w f g)/sum(w)``, and

        WWZ = (N_eff - 3) * V_y / (2 * (V_x - V_y)),   WWA = sqrt(y_2^2 + y_3^2)

    with ``V_x``/``V_y`` the weighted variations of the data and of the model
    evaluated at the data times. Returns ``(wwz, wwa, n_eff)``.
    """
    omega = 2.0 * math.pi * f_val
    n = int(t.size)
    w = np.empty(n)
    phi = np.empty((3, n))
    for a in range(n):
        w[a] = math.exp(-c * omega**2 * (t[a] - tau_val) ** 2)
        phi[0, a] = 1.0
        phi[1, a] = math.cos(omega * (t[a] - tau_val))
        phi[2, a] = math.sin(omega * (t[a] - tau_val))
    sw = float(w.sum())

    def inner(f: np.ndarray, g: np.ndarray) -> float:
        return float((w * f * g).sum() / sw)

    s = np.array([[inner(phi[i], phi[j]) for j in range(3)] for i in range(3)])
    b = np.array([inner(phi[i], y) for i in range(3)])
    coef = np.linalg.lstsq(s, b, rcond=None)[0]
    model = coef[0] * phi[0] + coef[1] * phi[1] + coef[2] * phi[2]

    n_eff = sw * sw / float((w * w).sum())
    v_x = inner(y, y) - inner(phi[0], y) ** 2
    v_y = inner(model, model) - inner(phi[0], model) ** 2
    wwz = (n_eff - 3.0) * v_y / (2.0 * (v_x - v_y))
    return wwz, math.hypot(float(coef[1]), float(coef[2])), n_eff


def steady() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant-period sine over a year: the null case for period wander."""
    return synthetic_sine(n=900, span=360.0, period=PERIOD, noise=0.02, seed=3)


def gapped() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same curve with a 70-day hole, for the N_eff mask."""
    t, mag, err = steady()
    keep = ~((t - t[0] >= 150.0) & (t - t[0] < 220.0))
    return t[keep], mag[keep], err[keep]


# --- parity with the transparent reference -----------------------------------


def test_grid_matches_the_reference_cell_by_cell() -> None:
    """Six scattered (tau, f) cells reproduce Foster's formulas exactly."""
    t, mag, _ = synthetic_sine(n=150, span=40.0, period=1.3, noise=0.02, seed=5)
    freq = np.array([0.5, 1.0 / 1.3, 0.9])
    taus = float(t.min()) + np.array([10.0, 25.0])
    res = cup.wwz((t, mag), frequency=freq, tau=taus, c=DEFAULT_C)
    assert res.shape == (2, 3)
    for i, tau_val in enumerate(taus):
        for j, f_val in enumerate(freq):
            ref_z, ref_a, ref_n = _reference_wwz(
                t, mag, float(tau_val), float(f_val), DEFAULT_C
            )
            assert res.wwz[i, j] == pytest.approx(ref_z, rel=1e-9)
            assert res.wwa[i, j] == pytest.approx(ref_a, rel=1e-9)
            assert res.n_eff[i, j] == pytest.approx(ref_n, rel=1e-9)


# --- a constant period gives a flat ridge ------------------------------------


def test_constant_period_ridge_is_flat() -> None:
    """No wander planted: every tau lands on f0, and the trend fit says zero."""
    t, mag, err = steady()
    res = cup.wwz((t, mag, err), c=LPV_C)
    assert res.shape == (50, res.frequency.size)
    assert res.n_samples == t.size
    assert res.baseline == pytest.approx(float(t.max() - t.min()))

    f_true = 1.0 / PERIOD
    peak_f = res.peak_frequency()
    valid = np.isfinite(peak_f)
    assert np.count_nonzero(valid) == 50
    # Well inside the window's own 4.5% frequency resolution at c = 0.001.
    assert np.all(np.abs(peak_f[valid] / f_true - 1.0) < 2e-2)
    assert np.allclose(res.peak_period()[valid], 1.0 / peak_f[valid])

    wwa = res.peak_wwa()
    assert np.all(np.abs(wwa[valid] / 0.4 - 1.0) < 0.15)

    pdot_over_p, trend_err = res.period_trend()
    assert trend_err > 0.0
    assert abs(pdot_over_p) < 5.0 * trend_err


def test_default_grid_is_geometric_and_bounded() -> None:
    """The automatic axis steps a quarter of the (constant) local resolution."""
    t, mag, err = steady()
    res = cup.wwz((t, mag, err), c=LPV_C)
    ratio = res.frequency[1:] / res.frequency[:-1]
    assert np.allclose(ratio, 1.0 + math.sqrt(2.0 * LPV_C) / 4.0)
    assert res.frequency[0] == pytest.approx(2.0 / res.baseline)
    assert res.frequency.size <= 2000
    assert res.meta["wwz"]["grid_capped"] is False
    assert res.meta["wwz"]["c"] == LPV_C
    assert np.all(np.diff(res.tau) > 0.0)
    assert res.tau[0] == pytest.approx(float(t.min()))
    assert res.tau[-1] == pytest.approx(float(t.max()))

    # A tiny c asks for a resolution the cap cannot afford: the range is kept
    # and the geometric step widened, exactly as the sliding map coarsens.
    tiny = cup.wwz((t, mag, err), c=1e-5, tau=res.tau[:1])
    assert tiny.meta["wwz"]["grid_capped"] is True
    assert tiny.frequency.size == 2000
    assert tiny.frequency[0] == pytest.approx(res.frequency[0])
    step = tiny.frequency[1:] / tiny.frequency[:-1]
    assert np.allclose(step, step[0])
    assert step[0] > 1.0 + math.sqrt(2.0 * 1e-5) / 4.0


# --- a drifting period drifts ------------------------------------------------


def test_drifting_period_is_recovered_by_the_ridge() -> None:
    """A 3%-per-baseline drift shows up in the ridge with the right size and sign."""
    t, mag, err, truth = synthetic_period_change(
        n=2000, span=1500.0, period=2.0, pdot=4.0e-5, noise=0.03, seed=2
    )
    assert truth["pdot_over_p"] == pytest.approx(2.0e-5)
    res = cup.wwz((t, mag, err), c=LPV_C)
    pdot_over_p, trend_err = res.period_trend()
    assert pdot_over_p > 0.0
    assert 0.5 < pdot_over_p / truth["pdot_over_p"] < 2.0
    assert trend_err < 0.5 * pdot_over_p
    # A lengthening period is a falling frequency, first valid tau to last.
    peak_f = res.peak_frequency()
    valid = np.flatnonzero(np.isfinite(peak_f))
    assert peak_f[valid[-1]] < peak_f[valid[0]]
    assert res.peak_period()[valid[-1]] > res.peak_period()[valid[0]]


# --- the N_eff mask ----------------------------------------------------------


def test_sparse_cells_are_masked_and_n_eff_explains_why() -> None:
    """A tau in a data gap loses its short-window cells; n_eff stays finite."""
    t, mag, err = gapped()
    freq = np.geomspace(0.02, 2.0, 60)
    taus = float(t.min()) + np.array([20.0, 100.0, 185.0, 300.0])
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus)
    in_gap = 2
    # At f >= 0.15 c/d the window is days wide and the nearest datum 35 d away.
    assert np.all(np.isnan(res.wwz[in_gap, -10:]))
    assert np.all(np.isnan(res.wwa[in_gap, -10:]))
    assert np.all(np.isfinite(res.n_eff))  # the mask's own diagnostic survives
    assert np.all(res.n_eff[in_gap, -10:] <= 3.0)
    for row in (0, 1, 3):
        assert np.count_nonzero(np.isnan(res.wwz[row])) < np.count_nonzero(
            np.isnan(res.wwz[in_gap])
        )


def test_min_neff_only_tightens_the_mask() -> None:
    """min_neff moves the NaN threshold; it does not change what was computed."""
    t, mag, err = gapped()
    freq = np.geomspace(0.02, 2.0, 60)
    taus = float(t.min()) + np.array([20.0, 100.0, 185.0, 300.0])
    loose = cup.wwz((t, mag, err), frequency=freq, tau=taus, min_neff=3.0)
    tight = cup.wwz((t, mag, err), frequency=freq, tau=taus, min_neff=20.0)
    n_loose = int(np.count_nonzero(np.isnan(loose.wwz)))
    n_tight = int(np.count_nonzero(np.isnan(tight.wwz)))
    assert n_tight > n_loose > 0
    assert np.array_equal(loose.n_eff, tight.n_eff)
    assert np.all(np.isnan(tight.wwz[tight.n_eff <= 20.0]))
    kept = tight.n_eff > 20.0
    assert np.array_equal(tight.wwz[kept], loose.wwz[kept])
    assert tight.meta["wwz"]["min_neff"] == 20.0


# --- c is the window --------------------------------------------------------


def test_c_sets_the_window_width() -> None:
    """sigma_t/P = 1/(2 pi sqrt(2c)): one period by default, halved when c x4."""
    t, mag, err = steady()
    freq = np.geomspace(0.5, 3.0, 40)
    taus = float(t.min()) + np.array([60.0, 180.0, 300.0])
    base = cup.wwz((t, mag, err), frequency=freq, tau=taus)
    assert base.c == pytest.approx(DEFAULT_C)
    assert base.sigma_t_over_p == pytest.approx(1.0)
    assert base.sigma_f_over_f == pytest.approx(math.sqrt(2.0 * DEFAULT_C))

    wider = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=0.25 * DEFAULT_C)
    narrower = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=4.0 * DEFAULT_C)
    assert narrower.sigma_t_over_p == pytest.approx(0.5 * base.sigma_t_over_p)
    assert wider.sigma_t_over_p == pytest.approx(2.0 * base.sigma_t_over_p)
    # A wider window in time holds more points: n_eff scales with it.
    assert np.median(wider.n_eff) > np.median(base.n_eff) > np.median(narrower.n_eff)


# --- guards and plumbing -----------------------------------------------------


def test_multiband_input_is_rejected() -> None:
    bands = synthetic_multiband_sine(band_points=(300, 250, 200), span=400.0)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {name: cup.LightCurve.from_arrays(*arrays) for name, arrays in bands.items()}
    )
    with pytest.raises(ValueError, match="pool or pick a band"):
        cup.wwz(mb)


def test_too_few_points_raises() -> None:
    t, mag, err = synthetic_sine(n=15, span=40.0)
    with pytest.raises(InsufficientDataError, match="finite points"):
        cup.wwz((t, mag, err))


def test_bad_c_and_n_tau_raise() -> None:
    t, mag, err = steady()
    with pytest.raises(ValueError, match="c must be positive"):
        cup.wwz((t, mag, err), c=0.0)
    with pytest.raises(ValueError, match="n_tau must be"):
        cup.wwz((t, mag, err), n_tau=0)
    with pytest.raises(ValueError, match="finite and positive"):
        cup.wwz((t, mag, err), frequency=np.array([0.5, -1.0]))
    with pytest.raises(ValueError, match="tau values must be finite"):
        cup.wwz((t, mag, err), tau=np.array([np.nan]))


def test_explicit_axes_are_used_verbatim() -> None:
    t, mag, err = steady()
    freq = np.linspace(1.0, 2.5, 37)
    taus = float(t.min()) + np.linspace(10.0, 350.0, 9)
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus, n_tau=999)
    assert np.array_equal(res.frequency, freq)
    assert np.array_equal(res.tau, taus)
    assert res.shape == (9, 37)
    assert np.allclose(res.period, 1.0 / freq)
    assert res.meta["wwz"]["explicit_frequency"] is True
    assert res.meta["wwz"]["explicit_tau"] is True
    assert res.meta["wwz"]["n_tau"] == 9  # n_tau= is ignored when tau= is given


def test_period_trend_needs_three_valid_taus() -> None:
    """Two rows have no residuals to fit; the trend refuses rather than guesses."""
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.array([100.0, 200.0])
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=LPV_C)
    with pytest.raises(InsufficientDataError, match="at least 3 valid taus"):
        res.period_trend()


def test_to_dict_round_trip() -> None:
    """to_dict carries the ridge always, the 2-D maps on request."""
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.linspace(20.0, 340.0, 8)
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=LPV_C)
    plain = res.to_dict()
    assert set(plain) == {
        "c",
        "sigma_t_over_p",
        "sigma_f_over_f",
        "n_tau",
        "n_frequency",
        "n_samples",
        "baseline",
        "tau",
        "peak_frequency",
        "peak_period",
        "peak_wwa",
    }
    assert plain["n_tau"] == 8
    assert plain["n_frequency"] == 30
    assert len(plain["peak_frequency"]) == 8
    assert plain["c"] == LPV_C

    full = res.to_dict(include_maps=True)
    assert set(full) - set(plain) == {"frequency", "wwz", "wwa", "n_eff"}
    assert np.asarray(full["wwz"]).shape == res.wwz.shape
    assert np.asarray(full["n_eff"]).shape == res.n_eff.shape
    assert len(full["frequency"]) == 30
