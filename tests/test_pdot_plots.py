"""Period-change diagnostic figures: inputs, the drift-corrected fold, labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from synth import synthetic_period_change, synthetic_sine

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402

from cuperiod.pdot.plots import (  # noqa: E402
    _as_arrays,
    pdot_figure,
    plot_light_curve,
    plot_phase_fold,
)

#: Noise of the drifting test curve, the floor a sharp fold's scatter sits on.
NOISE = 0.03


def _axes_titled(fig: Figure, prefix: str) -> Any:
    """The (single) panel whose title starts with ``prefix``."""
    hits = [ax for ax in fig.axes if ax.get_title().startswith(prefix)]
    assert len(hits) == 1, [ax.get_title() for ax in fig.axes]
    return hits[0]


def _fold_points(ax: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(phase, value)`` of the scatter a fold panel drew."""
    offsets = np.asarray(ax.collections[-1].get_offsets(), dtype=np.float64)
    return offsets[:, 0], offsets[:, 1]


def _harmonic_rms(phase: np.ndarray, value: np.ndarray, n_harmonics: int = 4) -> float:
    """Scatter of a fold about its own least-squares Fourier series."""
    cols = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2.0 * np.pi * k * phase), np.sin(2.0 * np.pi * k * phase)]
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, value, rcond=None)
    return float(np.sqrt(np.mean((value - design @ coef) ** 2)))


def _drifting() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, mag, err, _ = synthetic_period_change(
        n=2500, span=1500.0, period=0.5, pdot=2.0e-7, noise=NOISE, seed=0
    )
    return t, mag, err


# --- inputs ------------------------------------------------------------------


def test_masked_samples_are_not_plotted() -> None:
    """A numpy mask is honoured by every input path, tuple and direct panel alike."""
    t, mag, err = synthetic_sine(n=800, span=300.0, seed=2)
    bad = np.zeros(t.size, dtype=bool)
    bad[::10] = True
    masked = np.ma.MaskedArray(np.where(bad, 99.99, mag), mask=bad)
    kept = int(np.count_nonzero(~bad))

    curve = _as_arrays((t, masked, err))
    assert curve is not None
    assert curve[0].size == kept
    assert float(np.max(curve[1])) < 50.0

    fig = pdot_figure(light_curve=(t, masked, err))
    ax = _axes_titled(fig, "light curve")
    assert ax.get_title() == f"light curve ({kept} points)"
    assert max(ax.get_ylim()) < 50.0

    direct = Figure().add_subplot()
    plot_light_curve(direct, t, masked, t_offset=0.0)
    assert direct.get_title() == f"light curve ({kept} points)"


# --- the drift-corrected fold -------------------------------------------------


def test_fold_uses_the_tracks_own_ephemeris() -> None:
    """A slightly-off input period is corrected by the track, so the fold is sharp.

    phase_track absorbs an error in its reference period into the O-C linear
    term; folding at the *input* period with only the quadratic rate left a
    0.3-cycle linear phase drift across this run (fold rms 0.13 vs noise 0.03).
    """
    t, mag, err = _drifting()
    for period in (0.5 * (1.0 + 1e-4), 0.5):
        track = cup.phase_track((t, mag, err), period=period)
        fig = pdot_figure(light_curve=(t, mag, err), track=track)
        phase, value = _fold_points(_axes_titled(fig, "fold at P"))
        assert phase.size == t.size
        assert _harmonic_rms(phase, value) < 1.1 * NOISE


# --- plot_phase_fold ----------------------------------------------------------


def test_unknown_model_raises() -> None:
    """A misspelt model is an error, not a silently unwarped fold."""
    t, mag, _ = _drifting()
    ax = Figure().add_subplot()
    with pytest.raises(ValueError, match="unknown period-evolution model"):
        plot_phase_fold(ax, t, mag, period=0.5, pdot=4e-7, model="constant-pdt")


def test_singular_rate_draws_an_unwarped_fold_and_says_so() -> None:
    """A rate that drives the period through zero falls back, visibly."""
    t, mag, _ = _drifting()
    ax = Figure().add_subplot()
    plot_phase_fold(ax, t, mag, period=0.5, pdot=-1.0)
    phase, _ = _fold_points(ax)
    expected = np.mod((t - float(np.mean(t))) / 0.5, 1.0)
    np.testing.assert_allclose(phase, expected, rtol=0.0, atol=1e-9)
    assert "unwarped" in ax.get_title()

    warped = Figure().add_subplot()
    plot_phase_fold(warped, t, mag, period=0.5, pdot=2.0e-7)
    assert "unwarped" not in warped.get_title()
    assert "dP/dt" in warped.get_title()


def test_fold_ylabel_defaults_to_magnitude() -> None:
    t, mag, _ = _drifting()
    ax = Figure().add_subplot()
    plot_phase_fold(ax, t, mag, period=0.5)
    assert ax.get_ylabel() == "magnitude"
    flux = Figure().add_subplot()
    plot_phase_fold(flux, t, mag, period=0.5, ylabel="flux")
    assert flux.get_ylabel() == "flux"


def test_flux_light_curve_labels_both_panels_flux() -> None:
    """Neither panel calls a flux a magnitude, and neither flips its axis."""
    t, mag, _ = _drifting()
    flux = 10.0 ** (-0.4 * (mag - 13.0))
    lc = cup.LightCurve(t, flux, None, domain="flux")
    track = cup.phase_track(lc, period=0.5)
    fig = pdot_figure(light_curve=lc, track=track)
    for prefix in ("light curve", "fold at P"):
        ax = _axes_titled(fig, prefix)
        assert ax.get_ylabel() == "flux"
        lo, hi = ax.get_ylim()
        assert lo < hi  # not inverted
    mag_track = cup.phase_track((t, mag), period=0.5)
    mag_fig = pdot_figure(light_curve=(t, mag), track=mag_track)
    for prefix in ("light curve", "fold at P"):
        ax = _axes_titled(mag_fig, prefix)
        assert ax.get_ylabel() == "magnitude"
        lo, hi = ax.get_ylim()
        assert lo > hi  # magnitudes: brighter upwards
