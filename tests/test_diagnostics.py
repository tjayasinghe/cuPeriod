"""Alias diagnostics: measured window peaks, alias scoring, both objective senses."""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.grid import GridSpec
from cuperiod.core.result import Periodogram
from cuperiod.diagnostics import CLASSIC_WINDOW_FREQUENCIES, alias_diagnostics
from synth import synthetic_multiband_sine

#: Planted frequency (cycles/day); its daily aliases at 1.7 and 3.7 are in the grid.
F_TRUE = 2.7


def nightly_light_curve(
    *,
    n_nights: int = 90,
    per_night: int = 6,
    night_length: float = 0.3,
    noise: float = 0.01,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A sine observed only in nightly windows — the classic 1 c/d alias generator."""
    rng = np.random.default_rng(seed)
    nights = np.repeat(np.arange(n_nights, dtype=float), per_night)
    time = np.sort(nights + rng.uniform(0.0, night_length, nights.size)) + 2458000.0
    mag = (
        12.0
        + 0.3 * np.sin(2 * np.pi * F_TRUE * time)
        + rng.normal(0.0, noise, time.size)
    )
    return time, mag, np.full(time.size, noise)


def make_grid(nf: int = 4001, f_max: float = 6.0) -> GridSpec:
    """A small explicit grid spanning the daily aliases and the second harmonic."""
    return GridSpec(kind="frequency", values=np.linspace(0.05, f_max, nf), uniform=True)


def find_candidate(report: cup.AliasReport, target: float, tol: float = 0.05):
    """The first candidate predicted within ``tol`` of ``target``, or None."""
    for candidate in report.candidates:
        if abs(candidate.frequency - target) <= tol:
            return candidate
    return None


@pytest.fixture(scope="module")
def nightly() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return nightly_light_curve()


@pytest.fixture(scope="module")
def nightly_gls(nightly: tuple[np.ndarray, ...]) -> Periodogram:
    pg = cup.periodogram(nightly, "GLS", grid=make_grid(), backend="cpu")
    assert isinstance(pg, Periodogram)
    return pg


@pytest.fixture(scope="module")
def nightly_report(
    nightly: tuple[np.ndarray, ...], nightly_gls: Periodogram
) -> cup.AliasReport:
    return alias_diagnostics(nightly_gls, nightly, backend="cpu")


# --- measured spectral window -------------------------------------------------


def test_window_finds_the_daily_peak(nightly_report: cup.AliasReport) -> None:
    """Nightly sampling puts the strongest non-DC window peak at 1 cycle/day."""
    peaks = nightly_report.window_peaks
    assert peaks
    assert min(abs(p.frequency - 1.0) for p in peaks) < 0.02
    # Sorted strongest-first and ratioed to that leading peak.
    assert peaks[0].amplitude_ratio == pytest.approx(1.0)
    assert all(p.amplitude_ratio <= 1.0 + 1e-12 for p in peaks)
    assert all(np.isfinite(p.amplitude) for p in peaks)
    # The DC lobe is excluded, and the grid stops at the default 5.5 c/d.
    assert all(0.02 < p.frequency <= 5.5 for p in peaks)


def test_rayleigh_is_one_over_baseline(
    nightly_report: cup.AliasReport, nightly_gls: Periodogram
) -> None:
    assert nightly_report.rayleigh == pytest.approx(1.0 / nightly_gls.baseline)


# --- alias candidates ---------------------------------------------------------


def test_daily_alias_is_a_serious_competitor(nightly_report: cup.AliasReport) -> None:
    """The +/-1 c/d aliases are matched in the periodogram and score well."""
    assert nightly_report.best_frequency == pytest.approx(F_TRUE, rel=2e-3)
    upper = find_candidate(nightly_report, F_TRUE + 1.0)
    lower = find_candidate(nightly_report, F_TRUE - 1.0)
    assert upper is not None and lower is not None
    for candidate in (upper, lower):
        assert candidate.matched
        assert np.isfinite(candidate.power)
        assert candidate.score > 0.2
        assert candidate.separation_rayleigh < 3.0
        assert "m=" in candidate.kind and not candidate.is_harmonic
    # Sparse nightly sampling of a single sine is exactly the ambiguous case.
    assert nightly_report.ambiguous is True
    # Sorted by score, descending.
    scores = [c.score for c in nightly_report.candidates]
    assert scores == sorted(scores, reverse=True)


def test_harmonic_candidate_is_reported(nightly_report: cup.AliasReport) -> None:
    """A pure sine still gets its 2f entry — listed, but never called ambiguous."""
    harmonics = [c for c in nightly_report.candidates if c.kind == "harmonic n=2"]
    assert len(harmonics) == 1
    assert harmonics[0].frequency == pytest.approx(
        2.0 * nightly_report.best_frequency
    )
    assert harmonics[0].is_harmonic


def test_candidates_stay_inside_the_grid(
    nightly_report: cup.AliasReport, nightly_gls: Periodogram
) -> None:
    """Out-of-range predictions (e.g. 3*f_true = 8.1 c/d) are dropped silently."""
    low, high = nightly_gls.frequency[0], nightly_gls.frequency[-1]
    assert all(low <= c.frequency <= high for c in nightly_report.candidates)
    assert not any(c.kind == "harmonic n=3" for c in nightly_report.candidates)
    # Nothing is a restatement of the frequency being diagnosed.
    rayleigh = nightly_report.rayleigh
    assert all(
        abs(c.frequency - nightly_report.best_frequency) >= rayleigh
        for c in nightly_report.candidates
    )


def test_explicit_frequency_is_diagnosed(
    nightly: tuple[np.ndarray, ...], nightly_gls: Periodogram
) -> None:
    """``frequency=`` overrides the periodogram's own best peak."""
    report = alias_diagnostics(
        nightly_gls, nightly, frequency=F_TRUE / 2.0, backend="cpu"
    )
    assert report.best_frequency == pytest.approx(F_TRUE / 2.0)
    assert np.isfinite(report.best_power)
    harmonic = next(c for c in report.candidates if c.kind == "harmonic n=2")
    assert harmonic.frequency == pytest.approx(F_TRUE)
    # The real peak, reached here as the harmonic, beats the half-frequency.
    assert harmonic.score > 1.0


# --- no light curve: the classic suspects -------------------------------------


def test_classic_suspects_without_data(nightly_gls: Periodogram) -> None:
    """Without a light curve the textbook alias spacings stand in for the window."""
    report = alias_diagnostics(nightly_gls)
    assert [p.frequency for p in report.window_peaks] == [
        f for _label, f in CLASSIC_WINDOW_FREQUENCIES
    ]
    assert all(np.isnan(p.amplitude) for p in report.window_peaks)
    assert all(np.isnan(p.amplitude_ratio) for p in report.window_peaks)
    assert report.candidates
    sidereal = find_candidate(report, F_TRUE + 1.0)
    assert sidereal is not None
    assert sidereal.kind.startswith("sidereal-day-alias m=+1")
    assert sidereal.score > 0.2
    # The yearly spacing is unresolvable on a 90-day baseline: no candidate for it.
    assert not any("year" in c.kind for c in report.candidates)


# --- minimized statistics ------------------------------------------------------


def test_min_sense_periodogram(nightly: tuple[np.ndarray, ...]) -> None:
    """PDM (smaller theta is better) is scored on the same 0-1 scale, no crash."""
    pg = cup.periodogram(nightly, "PDM", grid=make_grid(1001), backend="cpu")
    assert isinstance(pg, Periodogram)
    assert pg.objective_sense == "min"
    report = alias_diagnostics(pg, nightly, backend="cpu")
    assert report.best_frequency == pytest.approx(F_TRUE, rel=5e-3)
    assert np.isfinite(report.best_power)
    assert report.candidates
    matched = [c for c in report.candidates if c.matched]
    assert matched
    assert all(np.isfinite(c.score) and np.isfinite(c.power) for c in matched)
    assert all(c.score >= 0.0 for c in report.candidates)
    alias = find_candidate(report, F_TRUE - 1.0)
    assert alias is not None and alias.score > 0.2


def test_summary_is_printable(nightly_report: cup.AliasReport) -> None:
    text = nightly_report.summary()
    assert text
    assert len(text.splitlines()) >= 4
    assert f"{nightly_report.best_period:.8g}" in text
    assert "AMBIGUOUS" in text
    assert nightly_report.candidates[0].kind in text


# --- other inputs and guards ---------------------------------------------------


def test_multiband_input(nightly_gls: Periodogram) -> None:
    """A MultiBandLightCurve's bands are stacked into one sampling for the window."""
    bands = synthetic_multiband_sine(
        band_points=(80, 60), amplitudes=(0.30, 0.22), offsets=(15.0, 14.2),
        period=1.0 / F_TRUE,
    )
    mblc = cup.MultiBandLightCurve.from_light_curves(
        {n: cup.LightCurve.from_arrays(t, m, e) for n, (t, m, e) in bands.items()}
    )
    pg = cup.periodogram(mblc, "GLS", grid=make_grid(), backend="cpu")
    assert isinstance(pg, Periodogram)
    report = alias_diagnostics(pg, mblc, backend="cpu")
    assert report.best_frequency == pytest.approx(F_TRUE, rel=2e-3)
    assert report.window_peaks
    assert report.candidates
    assert np.isfinite(report.rayleigh)
    # Stacking both bands gives the full 180-day baseline, not one band's.
    assert report.rayleigh == pytest.approx(1.0 / pg.baseline)


def test_empty_periodogram_raises() -> None:
    empty = Periodogram.from_spectrum(
        method="GLS",
        backend="finufft",
        frequency=np.empty(0),
        power=np.empty(0),
        objective_sense="max",
        n_samples=0,
        baseline=10.0,
    )
    with pytest.raises(ValueError, match="empty"):
        alias_diagnostics(empty)
