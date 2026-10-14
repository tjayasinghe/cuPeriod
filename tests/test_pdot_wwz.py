"""WWZ: parity with Foster's formulas, ridge recovery, N_eff masking, guards."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_numba
from cuperiod.core.errors import BackendUnavailableError, InsufficientDataError
from synth import synthetic_multiband_sine, synthetic_period_change, synthetic_sine

PERIOD = 0.6234

#: The module itself (``cuperiod.pdot.wwz`` the attribute is the function).
wwz_module = importlib.import_module("cuperiod.pdot.wwz")

#: Foster's decay constant in the VARTOOLS convention (the function's default).
DEFAULT_C = 1.0 / (8.0 * math.pi**2)

#: The long-period-variable convention: a 3.56-period window, 4.5% resolution.
LPV_C = 0.001

#: Both CPU backends; the numba tier is skipped where the [fast] extra is missing.
BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param("numba", id="numba", marks=requires_numba),
]


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_grid_matches_the_reference_cell_by_cell(backend: str) -> None:
    """Six scattered (tau, f) cells reproduce Foster's formulas exactly."""
    t, mag, _ = synthetic_sine(n=150, span=40.0, period=1.3, noise=0.02, seed=5)
    freq = np.array([0.5, 1.0 / 1.3, 0.9])
    taus = float(t.min()) + np.array([10.0, 25.0])
    res = cup.wwz((t, mag), frequency=freq, tau=taus, c=DEFAULT_C, backend=backend)
    assert res.shape == (2, 3)
    assert res.meta["wwz"]["backend"] == backend
    # The statistic is exactly offset-invariant, so the reference runs on the
    # centred values the implementation itself uses (raw magnitudes would only
    # add the reference's own round-off to the comparison).
    centred = mag - float(np.mean(mag))
    for i, tau_val in enumerate(taus):
        for j, f_val in enumerate(freq):
            ref_z, ref_a, ref_n = _reference_wwz(
                t, centred, float(tau_val), float(f_val), DEFAULT_C
            )
            assert res.wwz[i, j] == pytest.approx(ref_z, rel=1e-9)
            assert res.wwa[i, j] == pytest.approx(ref_a, rel=1e-9)
            assert res.n_eff[i, j] == pytest.approx(ref_n, rel=1e-9)


@requires_numba
def test_numba_and_numpy_backends_agree() -> None:
    """The multicore kernel is the vectorized path, cell for cell, mask for mask."""
    t, mag, err = gapped()
    freq = np.geomspace(0.02, 3.0, 90)
    taus = float(t.min()) + np.linspace(0.0, 360.0, 23)
    slow = cup.wwz((t, mag, err), frequency=freq, tau=taus, backend="numpy")
    fast = cup.wwz((t, mag, err), frequency=freq, tau=taus, backend="numba")
    assert fast.meta["wwz"]["backend"] == "numba"
    assert np.array_equal(np.isnan(slow.wwz), np.isnan(fast.wwz))
    peak = float(np.nanmax(slow.wwz))
    np.testing.assert_allclose(fast.wwz, slow.wwz, rtol=1e-8, atol=1e-10 * peak)
    np.testing.assert_allclose(fast.wwa, slow.wwa, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(fast.n_eff, slow.n_eff, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fast.peak_frequency, slow.peak_frequency, rtol=1e-9)


def test_backend_selection() -> None:
    """'auto'/'cpu' take the fastest CPU tier; anything else is refused by name."""
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 12)
    taus = float(t.min()) + np.array([100.0, 200.0])
    auto = cup.wwz((t, mag, err), frequency=freq, tau=taus)
    expected = "numba" if wwz_module._numba_available() else "numpy"
    assert auto.meta["wwz"]["backend"] == expected
    cpu = cup.wwz((t, mag, err), frequency=freq, tau=taus, backend=" CPU ")
    assert cpu.meta["wwz"]["backend"] == expected
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        cup.wwz((t, mag, err), frequency=freq, tau=taus, backend="gpu")


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_truncation_is_lossless(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping points far under the Gaussian changes nothing but round-off.

    The cut is anchored on the *nearest* datum, so a tau inside a gap (where
    every weight is tiny but the statistic is still defined) keeps its cells
    and its n_eff exactly as the full sum gives them.
    """
    t, mag, err = gapped()
    freq = np.geomspace(0.02, 3.0, 90)
    taus = float(t.min()) + np.linspace(0.0, 360.0, 23)
    cut = cup.wwz((t, mag, err), frequency=freq, tau=taus, backend=backend)
    monkeypatch.setattr(wwz_module, "_WINDOW_CUT", math.inf)
    full = cup.wwz((t, mag, err), frequency=freq, tau=taus, backend=backend)
    assert np.array_equal(np.isnan(cut.wwz), np.isnan(full.wwz))
    np.testing.assert_allclose(cut.n_eff, full.n_eff, rtol=1e-9, atol=1e-12)
    peak = float(np.nanmax(full.wwz))
    finite = np.isfinite(full.wwz)
    assert np.all(np.abs(cut.wwz[finite] - full.wwz[finite]) <= 1e-9 * peak)


# --- a constant period gives a flat ridge ------------------------------------


def test_constant_period_ridge_is_flat() -> None:
    """No wander planted: every tau lands on f0, and the trend fit says zero."""
    t, mag, err = steady()
    res = cup.wwz((t, mag, err), c=LPV_C)
    assert res.shape == (50, res.frequency.size)
    assert res.n_samples == t.size
    assert res.baseline == pytest.approx(float(t.max() - t.min()))

    f_true = 1.0 / PERIOD
    peak_f = res.peak_frequency
    valid = np.isfinite(peak_f)
    assert np.count_nonzero(valid) == 50
    # Well inside the window's own 4.5% frequency resolution at c = 0.001.
    assert np.all(np.abs(peak_f[valid] / f_true - 1.0) < 2e-2)
    assert np.allclose(res.peak_period[valid], 1.0 / peak_f[valid])

    wwa = res.peak_wwa
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
    peak_f = res.peak_frequency
    valid = np.flatnonzero(np.isfinite(peak_f))
    assert peak_f[valid[-1]] < peak_f[valid[0]]
    assert res.peak_period[valid[-1]] > res.peak_period[valid[0]]


def test_descending_frequency_axis_gives_the_same_ridge() -> None:
    """``frequency=1/periods`` (a descending axis) refines the ridge correctly.

    The parabolic offset is measured in index steps; on a descending axis a
    step to the next index is a step *down* in frequency, so an unsigned grid
    spacing moved every refined peak the wrong way.
    """
    t, mag, _ = synthetic_sine(
        n=1500, span=3000.0, period=100.37, amp=0.5, noise=0.05, seed=11
    )
    descending = 1.0 / np.linspace(80.0, 130.0, 41)
    taus = float(t.min()) + np.linspace(300.0, 2700.0, 25)
    desc = cup.wwz((t, mag), frequency=descending, tau=taus)
    asc = cup.wwz((t, mag), frequency=descending[::-1], tau=taus)
    np.testing.assert_allclose(desc.wwz[:, ::-1], asc.wwz, rtol=1e-9)
    np.testing.assert_allclose(desc.peak_frequency, asc.peak_frequency, rtol=1e-9)
    # And the refinement improves on the raw cells rather than undoing them.
    cells = np.array([asc.frequency[np.nanargmax(row)] for row in asc.wwz])
    truth = 1.0 / 100.37
    refined_err = np.sqrt(np.mean((1.0 / desc.peak_frequency - 1.0 / truth) ** 2))
    cell_err = np.sqrt(np.mean((1.0 / cells - 1.0 / truth) ** 2))
    assert refined_err < cell_err


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


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_bad_c_raises(bad: float) -> None:
    """c must be a positive *finite* number: NaN and inf used to slip through."""
    t, mag, err = steady()
    with pytest.raises(ValueError, match="c must be positive"):
        cup.wwz((t, mag, err), c=bad)
    with pytest.raises(ValueError, match="c must be positive"):
        cup.wwz((t, mag, err), c=bad, frequency=np.array([1.0, 1.5]))


@pytest.mark.parametrize("bad", [0.0, 2.9, -5.0, math.nan])
def test_min_neff_below_the_algebraic_floor_raises(bad: float) -> None:
    """Below N_eff = 3 the statistic turns negative; NaN would mask everything."""
    t, mag, err = steady()
    with pytest.raises(ValueError, match="min_neff"):
        cup.wwz((t, mag, err), min_neff=bad)


def test_bad_n_tau_and_axes_raise() -> None:
    t, mag, err = steady()
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


def test_explicit_axes_are_copied() -> None:
    """The frozen result owns its axes: editing the caller's arrays changes nothing."""
    t, mag, err = steady()
    freq = np.linspace(1.0, 2.5, 37)
    taus = float(t.min()) + np.linspace(10.0, 350.0, 9)
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus)
    assert not np.shares_memory(res.frequency, freq)
    assert not np.shares_memory(res.tau, taus)
    ridge = res.peak_frequency.copy()
    freq *= 2.0
    taus += 100.0
    assert res.frequency[0] == pytest.approx(1.0)
    assert res.tau[0] == pytest.approx(float(t.min()) + 10.0)
    assert np.array_equal(res.peak_frequency, ridge, equal_nan=True)


# --- inputs ------------------------------------------------------------------


def test_masked_values_in_a_tuple_are_dropped() -> None:
    """A numpy mask is honoured: placeholders under it never reach the map."""
    t, mag, _ = steady()
    bad = np.zeros(t.size, dtype=bool)
    bad[::10] = True
    masked = np.ma.MaskedArray(np.where(bad, 99.99, mag), mask=bad)
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.linspace(20.0, 340.0, 8)
    res = cup.wwz((t, masked), frequency=freq, tau=taus)
    clean = cup.wwz((t[~bad], mag[~bad]), frequency=freq, tau=taus)
    assert res.n_samples == int(np.count_nonzero(~bad))
    assert np.array_equal(res.wwz, clean.wwz, equal_nan=True)


def test_unusable_errors_do_not_drop_points() -> None:
    """WWZ ignores the errors, so a blank or zero error must not cost the point.

    Visual estimates in AAVSO-style data carry no uncertainty; dropping them
    (the weighted methods' ``finite()`` convention) silently discarded most of
    such a light curve. Rows with a non-finite time or value still go.
    """
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.linspace(20.0, 340.0, 8)
    holey = err.copy()
    holey[::3] = np.nan
    holey[1::3] = 0.0
    res = cup.wwz((t, mag, holey), frequency=freq, tau=taus)
    plain = cup.wwz((t, mag), frequency=freq, tau=taus)
    assert res.n_samples == t.size
    assert np.array_equal(res.wwz, plain.wwz, equal_nan=True)
    for blank in (np.full(t.size, np.nan), np.zeros(t.size)):
        assert cup.wwz((t, mag, blank), frequency=freq, tau=taus).n_samples == t.size
    nan_value = mag.copy()
    nan_value[5] = np.nan
    assert cup.wwz((t, nan_value, err), frequency=freq, tau=taus).n_samples == (
        t.size - 1
    )


@pytest.mark.parametrize("level", [0.0, 12.0, 13.37])
def test_constant_series_gives_an_empty_map(level: float) -> None:
    """No variation, no ridge: every cell is masked and n_eff still explains it."""
    t, _, _ = steady()
    freq = np.geomspace(0.05, 3.0, 40)
    taus = float(t.min()) + np.linspace(0.0, 360.0, 10)
    res = cup.wwz((t, np.full(t.size, level)), frequency=freq, tau=taus)
    assert np.all(np.isnan(res.wwz))
    assert np.all(np.isnan(res.wwa))
    assert np.all(np.isnan(res.peak_frequency))
    assert np.all(np.isfinite(res.n_eff))


def test_a_constant_offset_does_not_change_the_map() -> None:
    """The constant trial function makes WWZ offset-invariant, so it must be.

    Raw values used to enter the sums, where ``<x|x> - y.b`` cancels terms of
    size mean^2: at a mean 10^5 times the scatter the low-Z cells were wrong by
    tens of percent and some came out negative.
    """
    t, mag, _ = steady()
    freq = np.geomspace(0.05, 3.0, 80)
    taus = float(t.min()) + np.linspace(0.0, 360.0, 20)
    base = cup.wwz((t, mag), frequency=freq, tau=taus)
    shifted = cup.wwz((t, mag + 1.0e5), frequency=freq, tau=taus)
    assert np.array_equal(np.isnan(base.wwz), np.isnan(shifted.wwz))
    peak = float(np.nanmax(base.wwz))
    np.testing.assert_allclose(shifted.wwz, base.wwz, rtol=1e-6, atol=1e-9 * peak)
    np.testing.assert_allclose(shifted.wwa, base.wwa, rtol=1e-6, atol=1e-12)
    assert float(np.nanmin(shifted.wwz)) >= 0.0


# --- the ridge trend's error bar ----------------------------------------------


def _lpv(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A constant-period long-period variable: 400 points over 4000 d, P = 120 d."""
    t, mag, _ = synthetic_sine(
        n=400, span=4000.0, period=120.0, amp=0.3, noise=0.15, seed=seed
    )
    return t, mag


def test_period_trend_error_does_not_shrink_with_more_taus() -> None:
    """Overlapping windows are not independent: n_tau is a grid knob, not data.

    At c = 0.001 a window is 3.56 periods wide, so 25 or 200 taus over the run
    sample the same handful of independent windows; the quoted error used to
    fall as 1/sqrt(n_tau) while the true scatter stayed put.
    """
    t, mag = _lpv(0)
    freq = np.geomspace(1.0 / 150.0, 1.0 / 100.0, 40)
    coarse = cup.wwz((t, mag), frequency=freq, n_tau=25, c=LPV_C).period_trend()
    fine = cup.wwz((t, mag), frequency=freq, n_tau=200, c=LPV_C).period_trend()
    assert 0.8 < fine[1] / coarse[1] < 1.25


def test_period_trend_error_is_calibrated_on_constant_periods() -> None:
    """Null ensemble at the LPV convention: the pulls have roughly unit scatter.

    With the independent-points error, half of these constant-period stars
    showed a >3 sigma 'drift' (pull rms ~4-8).
    """
    freq = np.geomspace(1.0 / 150.0, 1.0 / 100.0, 30)
    pulls = []
    for seed in range(16):
        t, mag = _lpv(100 + seed)
        rate, err = cup.wwz((t, mag), frequency=freq, c=LPV_C).period_trend()
        pulls.append(rate / err)
    assert float(np.sqrt(np.mean(np.square(pulls)))) < 2.5


def test_period_trend_error_is_infinite_without_independent_windows() -> None:
    """Taus packed inside one window's width carry no slope information."""
    t, mag = _lpv(0)
    freq = np.geomspace(1.0 / 150.0, 1.0 / 100.0, 30)
    taus = float(t.min()) + np.linspace(1900.0, 2100.0, 9)
    rate, err = cup.wwz((t, mag), frequency=freq, tau=taus, c=LPV_C).period_trend()
    assert math.isfinite(rate)
    assert err == math.inf


def test_period_trend_needs_three_valid_taus() -> None:
    """Two rows have no residuals to fit; the trend refuses rather than guesses."""
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.array([100.0, 200.0])
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=LPV_C)
    with pytest.raises(InsufficientDataError, match="at least 3 valid taus"):
        res.period_trend()


def test_ridge_accessors_match_the_sliding_periodogram() -> None:
    """WWZResult and SlidingPeriodogram spell the ridge the same way.

    Both expose ``peak_frequency``/``peak_period`` as arrays (not methods) and
    take ``include_map=`` in ``to_dict``, so code written against one map works
    on the other.
    """
    t, mag, err = steady()
    freq = np.geomspace(1.0, 2.5, 30)
    taus = float(t.min()) + np.linspace(20.0, 340.0, 8)
    res = cup.wwz((t, mag, err), frequency=freq, tau=taus, c=LPV_C)
    slide = cup.sliding_periodogram((t, mag, err), period=PERIOD)
    for ridge in (res, slide):
        assert isinstance(ridge.peak_frequency, np.ndarray)
        assert isinstance(ridge.peak_period, np.ndarray)
        assert np.isfinite(float(np.nanmedian(ridge.peak_period)))
        assert "frequency" in ridge.to_dict(include_map=True)
    assert isinstance(res.peak_wwa, np.ndarray)
    # The cached ridge is read-only, so it cannot drift from the map.
    with pytest.raises(ValueError):
        res.peak_frequency[0] = 1.0


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

    full = res.to_dict(include_map=True)
    assert set(full) - set(plain) == {"frequency", "wwz", "wwa", "n_eff"}
    assert np.asarray(full["wwz"]).shape == res.wwz.shape
    assert np.asarray(full["n_eff"]).shape == res.n_eff.shape
    assert len(full["frequency"]) == 30
