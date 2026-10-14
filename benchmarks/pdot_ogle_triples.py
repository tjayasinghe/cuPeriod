"""OGLE bulge triples: recovering published light-time orbits from raw photometry.

Hong et al. (2022, AJ 163, 157) measured the eclipse timing variations of
14,127 contact binaries in the OGLE-III + OGLE-IV bulge fields (2001-2015):
one time of minimum per observing season from Wilson-Devinney modelling of the
seasonal light curve, then a parabola (13,716 stars), a sinusoid (307) or both
(104) fitted to the timings. Their sinusoidal systems are hierarchical triples
seen through the light-travel-time effect (LTTE), with outer periods of 5-14
years — comparable to or shorter than the 15-year baseline. That is the regime
in which a period change is *not* monotonic, and a parabola fitted to the O-C
alone returns a rate that is a baseline-weighted average of a wobble.

This benchmark runs cuPeriod's automated chain on six of those stars and asks:

* does :func:`cuperiod.phase_track` + :func:`cuperiod.ltte_search` recover the
  published outer period from the raw survey photometry, with no per-season
  binary modelling;
* what does the quadratic ephemeris alone report for the same O-C — i.e. how
  large a spurious "dP/dt" a cyclic variation buys when it is not modelled;
* and what the joint fit's *secular* rate is once the cycle is removed.

The stars (OGLE-BLG-ECL ids; inner period, and the outer period of Hong et al.
with its error, all from their Table 3; one has an independent value from
Hajdu et al. 2019, MNRAS 485, 2562):

    061372  P = 0.505810 d   P2 = 5.7 +- 0.1 yr
    246583  P = 0.403251 d   P2 = 9.8 +- 0.7 yr
    252457  P = 0.633032 d   P2 = 6.7 +- 0.6 yr
    289126  P = 0.445851 d   P2 = 6.2 +- 0.2 yr   (Hajdu+ 2019: 5.7 +- 0.2 yr)
    299343  P = 0.303691 d   P2 = 5.6 +- 0.1 yr
    299686  P = 0.444770 d   P2 = 10.2 +- 0.5 yr

They were chosen by hand for brightness (I = 13.3-15.9) and for outer periods
the baseline covers at least once; nothing about their O-C was inspected first.

Recipe: a 40-day running-median detrend (which also absorbs the OGLE-III/IV
zero-point offset), 40 phase-tracking segments (20-260 points each), the
template order BIC-selected up to 8 harmonics, and the default outer-period
scan (four median segment spacings up to one full baseline). The published
periods are used as the reference period; nothing else is taken from the
paper.

A gotcha this benchmark found: a contact binary with near-equal minima has a
template that is almost perfectly symmetric under a half-period shift, so a
segment's best template shift lands on either branch by noise and the O-C
jumped by P/2 between segments — on four of the six stars, for up to two
thirds of the segments (``report_ogle_triples.py`` counts them). ``phase_track``
now unwraps by the template's symmetry period (``symmetry_order``), which is
what turned the hour-scale "amplitudes" of the first attempt into the
few-hundred-second orbits below.

Data: run ``download_ogle_triples.py`` first (public OGLE photometry). Writes
results/pdot_ogle_triples.parquet and figures/ogle_triples_oc.png.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from _common import DATA, FIGURES, RESULTS  # noqa: E402

import cuperiod as cup  # noqa: E402

#: OGLE-BLG-ECL id -> (inner period [d], Hong+ 2022 P2 [yr], its error, other P2).
STARS: dict[str, tuple[float, float, float, str]] = {
    "061372": (0.505810, 5.7, 0.1, ""),
    "246583": (0.403251, 9.8, 0.7, ""),
    "252457": (0.633032, 6.7, 0.6, ""),
    "289126": (0.445851, 6.2, 0.2, "Hajdu+2019: 5.7 +- 0.2"),
    "299343": (0.303691, 5.6, 0.1, ""),
    "299686": (0.444770, 10.2, 0.5, ""),
}

DETREND_WIDTH = 40.0
N_SEGMENTS = 40
MAX_HARMONICS = 8
DAYS_PER_YEAR = 365.25


def load(star: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OGLE-III + OGLE-IV I-band photometry of one star, time-sorted (HJD)."""
    parts = []
    for phase in ("ogle3", "ogle4"):
        path = DATA / "ogle_triples" / f"OGLE-BLG-ECL-{star}_{phase}.dat"
        if not path.exists():
            raise SystemExit(
                f"{path} not found -- run 'python benchmarks/download_ogle_triples.py'"
            )
        parts.append(np.loadtxt(path))
    data = np.vstack(parts)
    data = data[np.argsort(data[:, 0])]
    return data[:, 0] + 2450000.0, data[:, 1], data[:, 2]


def analyze(star: str) -> tuple[dict[str, object], cup.PhaseTrackResult, cup.LTTEFit]:
    period, p2_pub, p2_err_pub, other = STARS[star]
    t, mag, err = load(star)
    flat = cup.detrend(t, mag, DETREND_WIDTH)
    settings = cup.PdotSettings(n_segments=N_SEGMENTS, max_harmonics=MAX_HARMONICS)
    track = cup.phase_track((t, flat, err), period=period, settings=settings)
    res = track.ltte_search(settings=settings)
    best = res.best
    row = {
        "star": f"OGLE-BLG-ECL-{star}",
        "period_d": period,
        "n_points": int(t.size),
        "baseline_yr": float((t.max() - t.min()) / DAYS_PER_YEAR),
        "n_segments": track.n_segments,
        "n_harmonics": track.template.n_harmonics,
        "symmetry_order": track.symmetry_order,
        "wrap_risk": bool(track.wrap_risk),
        # The parabola alone, as an O-C analysis without a cycle would report.
        "quad_pdot_d_per_yr": track.pdot_d_per_yr,
        "quad_pdot_err_d_per_yr": track.pdot_err_scaled * DAYS_PER_YEAR,
        "quad_chi2_reduced": track.quadratic_fit.chi2_reduced,
        "quad_f_test_p": track.f_test_p,
        # The joint light-time orbit.
        "ltte_p2_yr": best.outer_period_yr,
        "ltte_p2_err_yr": best.outer_period_err_yr * best.error_inflation,
        "ltte_amplitude_s": best.amplitude_s,
        "ltte_amplitude_err_s": best.amplitude_err_s * best.error_inflation,
        "ltte_delta_bic": res.delta_bic,
        "ltte_f_test_p": res.f_test_p,
        "ltte_n_cycles": res.n_cycles,
        "ltte_at_grid_edge": bool(best.at_grid_edge),
        "ltte_chi2_reduced": best.chi2_reduced,
        "ltte_mass_function_msun": best.mass_function,
        "secular_pdot_d_per_yr": best.pdot_d_per_yr,
        "secular_pdot_err_d_per_yr": best.pdot_err_scaled * DAYS_PER_YEAR,
        # The literature.
        "published_p2_yr": p2_pub,
        "published_p2_err_yr": p2_err_pub,
        "published_other": other,
        "p2_agreement_sigma": abs(best.outer_period_yr - p2_pub)
        / np.hypot(best.outer_period_err_yr * best.error_inflation, p2_err_pub),
    }
    return row, track, best


def figure(tracks: dict[str, tuple[cup.PhaseTrackResult, cup.LTTEFit]]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from cuperiod.pdot.plots import plot_oc

    fig = Figure(figsize=(15.0, 8.0), dpi=130, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3).ravel()
    for ax, (star, (track, best)) in zip(axes, tracks.items(), strict=True):
        plot_oc(ax, track, ltte=best)
        p2, e_p2 = STARS[star][1], STARS[star][2]
        ax.set_title(
            f"OGLE-BLG-ECL-{star}: P2 = {best.outer_period_yr:.1f} yr "
            f"(Hong+ 2022: {p2} +- {e_p2})",
            fontsize=10,
        )
    fig.suptitle(
        "OGLE bulge contact binaries with a third body: O-C from phase_track, "
        "light-time orbit from ltte_search"
    )
    fig.savefig(FIGURES / "ogle_triples_oc.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    rows = []
    fits: dict[str, tuple[cup.PhaseTrackResult, cup.LTTEFit]] = {}
    for star in STARS:
        row, track, best = analyze(star)
        rows.append(row)
        fits[star] = (track, best)
        print(
            f"{row['star']}  n={row['n_points']:5d}  {row['baseline_yr']:.1f} yr  "
            f"segments={row['n_segments']} (sym {row['symmetry_order']})\n"
            f"    parabola alone : dP/dt = {row['quad_pdot_d_per_yr']:+.2e} +- "
            f"{row['quad_pdot_err_d_per_yr']:.1e} d/yr   "
            f"(chi2r {row['quad_chi2_reduced']:.1f}, p = {row['quad_f_test_p']:.1e})\n"
            f"    light-time orbit: P2 = {row['ltte_p2_yr']:.2f} +- "
            f"{row['ltte_p2_err_yr']:.2f} yr   published {row['published_p2_yr']} +- "
            f"{row['published_p2_err_yr']} {row['published_other']}\n"
            f"                      A = {row['ltte_amplitude_s']:.0f} +- "
            f"{row['ltte_amplitude_err_s']:.0f} s   dBIC = {row['ltte_delta_bic']:.0f}"
            f"   cycles = {row['ltte_n_cycles']:.2f}   f(m) = "
            f"{row['ltte_mass_function_msun']:.4f} Msun   "
            f"chi2r {row['ltte_chi2_reduced']:.1f}\n"
            f"    secular dP/dt  : {row['secular_pdot_d_per_yr']:+.2e} +- "
            f"{row['secular_pdot_err_d_per_yr']:.1e} d/yr"
        )
    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS / "pdot_ogle_triples.parquet", index=False)

    within = table["p2_agreement_sigma"] <= 2.0
    offset = np.abs(table["ltte_p2_yr"] - table["published_p2_yr"])
    median_offset = float(np.median(offset / table["published_p2_yr"]))
    print(
        f"\n{int(within.sum())}/{len(table)} outer periods within 2 sigma of the "
        f"published value (median |P2 - P2_pub|/P2_pub = {median_offset:.1%})"
    )
    quad_snr = np.abs(table["quad_pdot_d_per_yr"]) / table["quad_pdot_err_d_per_yr"]
    joint_snr = np.abs(table["secular_pdot_d_per_yr"]) / table[
        "secular_pdot_err_d_per_yr"
    ]
    print(
        f"{int((quad_snr >= 3.0).sum())}/{len(table)} stars where the parabola alone "
        f"claims a >= 3 sigma dP/dt; {int((joint_snr >= 3.0).sum())}/{len(table)} once "
        "the cycle is fitted jointly"
    )
    print(f"-> {RESULTS / 'pdot_ogle_triples.parquet'}")
    if not args.no_figure:
        figure(fits)
        print(f"-> {FIGURES / 'ogle_triples_oc.png'}")


if __name__ == "__main__":
    main()
