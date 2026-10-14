"""V1309 Sco: reproducing a published period-decay law on the binary caught merging.

V1309 Sco erupted in September 2008 as a red nova. Because it sits in the OGLE
bulge fields it is the one stellar merger with dense photometry of the
*progenitor*, and Tylenda et al. (2011, A&A 528, A114) used that archive to show
the progenitor was a contact binary spiralling in to the merger. Their Eq. (1),
fitted to periods measured in subsamples of about 50 observations each, is

    P = 1.4456 * exp(15.29 / (t - t0)),    t0 = JD 2455233.5,           (1)

with ``P`` in days and ``t`` in Julian dates. ``t0`` lies about 520 days *after*
the outburst, so every pre-outburst point has ``t - t0 < 0`` and the period
falls ever faster as the singularity is approached. They state (their Sect. 6.2)
that the orbital period decreased by 1.2% during 6 years. Pejcha (2014, ApJ 788,
22) -- not Tylenda et al., who quote no rate -- is the source of the widely
repeated decay timescale ``P/|dP/dt|`` falling from about 1000 yr to about
170 yr; differentiating Eq. (1) gives the closed form ``P/|dP/dt| = (t-t0)^2/b``.

That makes this a demanding end-to-end test of the v1.3 tools, and specifically
of the two-step recipe the literature actually uses -- measure P per interval,
then fit a law to the track:

* the decay is singular, so a constant-dP/dt model is knowingly wrong and the
  tools must say so rather than quietly returning a number;
* the star brightens by ~0.8 mag as it spirals in, which swamps an undetrended
  periodogram with 1 c/d aliases;
* OGLE's formal errors understate the scatter about tenfold, which breaks any
  order-selection criterion that trusts them;
* the light-curve shape itself evolves, so no fixed template fits and the formal
  timing errors are optimistic.

Data: the authors' own supplementary photometry, mirrored by OGLE at
https://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/ (I-band, 2001-2009). Run
``download_v1309sco.py`` first. All analysis is restricted to the quiescent
pre-merger window; the violent phase began in March 2008 (HJD ~2454527).

Writes results/pdot_v1309sco.parquet, results/pdot_v1309sco_profile.parquet and
figures/v1309sco_pdot.png.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from _common import DATA, FIGURES, RESULTS  # noqa: E402

import cuperiod as cup  # noqa: E402

#: The authors' supplementary I-band photometry (see download_v1309sco.py).
PHOTOMETRY = DATA / "v1309sco" / "V1309_SCO_OIII.dat"

#: Quiescent window: eclipses present, before the March 2008 merger brightening.
WINDOW = (2452100.0, 2454300.0)

#: Running-median detrending width in days: far longer than the ~1.43 d period,
#: far shorter than the multi-year secular brightening.
DETREND_WIDTH = 40.0

#: Tylenda et al. (2011) Eq. (1): (P0 in days, b in days, t0 in JD).
TYLENDA_EQ1 = (1.4456, 15.29, 2455233.5)

#: Their Sect. 6.2: "the orbital period decreased by 1.2% during 6 years".
PUBLISHED_DECREASE_PERCENT = -1.2

#: Pejcha (2014): P/|dP/dt| fell from ~1000 yr to ~170 yr in under 6 years.
PUBLISHED_TIMESCALE_YR = (1000.0, 170.0)

#: Subsample size of the published analysis ("subsamples of ~50 data points").
POINTS_PER_INTERVAL = 50

#: Interval sizes used for the systematic-sensitivity check.
INTERVAL_SIZES = (35, 40, 50, 60)

SEC_PER_YR = 86400.0 * 365.25


def tylenda_period(time: np.ndarray) -> np.ndarray:
    """Eq. (1) evaluated at ``time`` (JD): the published period law."""
    p0, b, t0 = TYLENDA_EQ1
    return p0 * np.exp(b / (np.asarray(time, dtype=float) - t0))


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The OGLE-III I-band light curve, restricted to the quiescent window."""
    if not PHOTOMETRY.exists():
        raise SystemExit(
            f"{PHOTOMETRY} not found -- run 'python benchmarks/download_v1309sco.py'"
        )
    t, mag, err = np.loadtxt(PHOTOMETRY, unpack=True)
    keep = (t >= WINDOW[0]) & (t < WINDOW[1])
    return t[keep], mag[keep], err[keep]


def report_profile(t, flat, err, points_per_interval):  # type: ignore[no-untyped-def]
    """Measure the P(t) track, rank the laws, and compare with Eq. (1)."""
    prof = cup.period_profile(
        (t, flat, err), period=TYLENDA_EQ1[0],
        points_per_interval=points_per_interval,
    )
    fits = prof.compare(
        laws=("constant", "linear", "exponential", "singular", "pejcha")
    )
    return prof, fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    t, mag, err = load()
    flat = cup.detrend(t, mag, DETREND_WIDTH)
    span = t.max() - t.min()
    print(
        f"V1309 Sco OGLE-III I-band: n={t.size}, HJD {t.min():.0f}-{t.max():.0f} "
        f"({span / 365.25:.2f} yr), detrended with a {DETREND_WIDTH:.0f} d "
        "running median"
    )
    print(
        "Published: P = 1.4456*exp(15.29/(t-2455233.5)) d (Tylenda+2011 Eq. 1);"
        "\n           -1.2% over 6 yr (their Sect. 6.2); P/|dP/dt| ~1000 -> ~170 yr"
        " (Pejcha 2014)\n"
    )

    # -- the published recipe: periods per subsample, then a law over the track ---
    prof, fits = report_profile(t, flat, err, POINTS_PER_INTERVAL)
    lo, hi = prof.points[0].t_start, prof.points[-1].t_end
    print(
        f"period_profile: {prof.n_intervals} intervals of ~{POINTS_PER_INTERVAL}"
        f" points, {prof.n_harmonics} harmonics"
        f"{'  [some at window edge!]' if prof.any_at_window_edge else ''}"
    )
    print(f"\n{'law':<12} {'BIC':>10} {'d(BIC)':>9} {'rms (d)':>9}  parameters")
    best = fits[0]
    for f in fits:
        params = "  ".join(
            f"{k}={v:.4g}+-{f.errors[k]:.2g}" for k, v in f.params.items()
        )
        print(
            f"{f.law:<12} {f.bic:10.1f} {f.bic - best.bic:9.1f} {f.rms:9.5f}  {params}"
        )
    print(
        "\n(The singular law's b and t0 are individually degenerate -- they"
        " trade off strongly -- while the P(t) they predict over the observed"
        " span is not,\n which is the comparison below. 'pejcha' reaches a"
        " lower BIC here but with its exponent pinned against a bound and"
        " unconstrained\n (delta = 8 +- 360), so it is ranked last: a fit"
        " sitting on a constraint has not found an interior optimum, and its"
        " BIC\n does not mean what it normally would.)"
    )

    # -- the headline comparison ---------------------------------------------------
    fit = prof.fit("singular")
    grid = np.linspace(lo, hi, 400)
    max_dev = float(np.max(np.abs(fit.period_at(grid) - tylenda_period(grid))))
    resid = prof.period - tylenda_period(prof.t_mid)
    print("our singular fit vs the published Eq. (1):")
    for name, theirs in zip(("period", "b", "t0"), TYLENDA_EQ1, strict=True):
        print(
            f"  {name:<7} {fit.params[name]:14.4f} +- {fit.errors[name]:<12.4g}"
            f" published {theirs:12.4f}"
        )
    print(
        f"  largest difference between the two P(t) curves over the observed"
        f" span: {max_dev:.5f} d ({100 * max_dev / TYLENDA_EQ1[0]:.2f}% of P)"
    )
    print(
        f"  measured points scatter about Eq. (1) by {resid.std():.5f} d"
        f" ({100 * resid.std() / TYLENDA_EQ1[0]:.2f}% of P),"
        f" mean offset {100 * resid.mean() / TYLENDA_EQ1[0]:+.3f}%"
    )
    print(
        f"  that scatter is {resid.std() / np.median(prof.period_err):.0f}x the"
        " formal period errors: the light-curve shape evolves and the period\n"
        "  really does wander on months timescales (Tylenda+2011 Sect. 5.1), so"
        " fit_period_law rescales them before quoting parameters."
    )

    print("\ntotal fractional period change over the observed baseline:")
    print(f"  singular law         : {100 * fit.fractional_change(lo, hi):+.2f} %")
    print(f"  endpoint intervals   : {100 * prof.total_fractional_change():+.2f} %")
    print(
        f"  published Eq. (1)    : "
        f"{100 * (tylenda_period(hi) / tylenda_period(lo) - 1):+.2f} %"
    )
    print(f"  published statement  : {PUBLISHED_DECREASE_PERCENT:+.1f} %"
          "  (Tylenda+2011 Sect. 6.2)")

    first, last = fit.timescale_yr_at([lo, hi])
    e_first, e_last = (
        (np.array([lo, hi]) - TYLENDA_EQ1[2]) ** 2 / TYLENDA_EQ1[1] / 365.25
    )
    print("\ndecay timescale P/|dP/dt| (yr), start -> end of the baseline:")
    print(f"  singular law         : {first:7.0f} -> {last:6.0f}")
    print(f"  published Eq. (1)    : {e_first:7.0f} -> {e_last:6.0f}")
    print(f"  published statement  : {PUBLISHED_TIMESCALE_YR[0]:7.0f} ->"
          f" {PUBLISHED_TIMESCALE_YR[1]:6.0f}  (Pejcha 2014)")

    # -- how much does the interval size matter? -----------------------------------
    print("\nsensitivity to the interval size (the method's own systematic):")
    print(f"  {'points/interval':>15} {'n':>4} {'H':>2} {'total change':>13}"
          f" {'timescale (yr)':>18} {'d(BIC) vs linear':>17}")
    rows = []
    for size in INTERVAL_SIZES:
        p, f_all = report_profile(t, flat, err, size)
        f = p.fit("singular")
        a, b = p.points[0].t_start, p.points[-1].t_end
        ts_a, ts_b = f.timescale_yr_at([a, b])
        linear = next((x for x in f_all if x.law == "linear"), None)
        gap = (linear.bic - f.bic) if linear is not None else float("nan")
        print(
            f"  {size:15d} {p.n_intervals:4d} {p.n_harmonics:2d}"
            f" {100 * f.fractional_change(a, b):12.2f} %"
            f" {ts_a:8.0f} -> {ts_b:5.0f} {gap:17.1f}"
        )
        rows.append({
            "points_per_interval": size,
            "n_intervals": p.n_intervals,
            "n_harmonics": p.n_harmonics,
            "law": f.law,
            "period": f.params["period"],
            "b": f.params["b"],
            "t0": f.params["t0"],
            "total_fractional_change": f.fractional_change(a, b),
            "timescale_first_yr": float(ts_a),
            "timescale_last_yr": float(ts_b),
            "delta_bic_vs_linear": gap,
            "chi2_reduced": f.chi2_reduced,
        })

    # -- what a single constant rate would have said -------------------------------
    search = cup.pdot_search(
        (t, flat, err), period=TYLENDA_EQ1[0], backend="cpu",
        settings=cup.PdotSettings(window_rayleigh=60.0, max_cycle_drift=12.0),
    )
    track = cup.phase_track(
        (t, flat, err), period=search.best.period, pdot=search.best.pdot,
        settings=cup.PdotSettings(harmonics=4, n_segments=10),
    )
    single = track.pdot * span / track.period
    print("\nfor contrast, forcing a single constant dP/dt over the whole baseline:")
    sigma = track.pdot_err_scaled * SEC_PER_YR
    print(
        f"  pdot_search P = {search.best.period:.5f} d,"
        f" dP/dt = {track.pdot_s_per_yr:.1f} +- {sigma:.1f}"
        f" s/yr  (F-test p = {track.f_test_p:.1e})"
    )
    print(f"  implied total change : {100 * single:+.2f} %  <- overshoots")
    print(
        f"  ridge correlation between P and dP/dt in the search map:"
        f" {search.best.ridge_correlation:+.2f}"
    )
    ladder = {}
    w = 1.0 / track.delay_err**2
    dt = track.t_mid - float(np.dot(w, track.t_mid) / w.sum())
    for deg in (2, 3, 4):
        coef = np.polyfit(dt, track.delay, deg, w=1.0 / track.delay_err)
        ladder[deg] = float(np.std(track.delay - np.polyval(coef, dt)) * 24 * 60)
    print(
        "  O-C residual rms vs ephemeris degree (2 = constant dP/dt): "
        + ", ".join(f"deg {d}: {v:.0f} min" for d, v in ladder.items())
    )
    print(
        f"  chi2_r = {track.quadratic_fit.chi2_reduced:.0f}: the drift is"
        " overwhelmingly real, but a parabola does not describe it."
    )

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(RESULTS / "pdot_v1309sco.parquet", index=False)
    pd.DataFrame(prof.to_table()).assign(
        tylenda_period=tylenda_period(prof.t_mid)
    ).to_parquet(RESULTS / "pdot_v1309sco_profile.parquet", index=False)
    print(f"\nresults -> {RESULTS / 'pdot_v1309sco.parquet'}")

    if not args.no_figure:
        figure(t, mag, err, search, track, prof, fit)


def figure(t, mag, err, search, track, prof, fit):  # type: ignore[no-untyped-def]
    """The six-panel diagnostic, with the published law overlaid."""
    from cuperiod.pdot.plots import pdot_figure

    fig = pdot_figure(
        light_curve=(t, mag, err),
        search=search,
        track=track,
        profile=prof,
        law=fit,
        title="V1309 Sco (OGLE-III I band) — orbital decay before the 2008 merger",
    )
    # Overlay Eq. (1) on the P(t) panel for a direct visual comparison.
    for ax in fig.axes:
        if ax.get_ylabel() == "P (d)":
            grid = np.linspace(prof.t_mid.min(), prof.t_mid.max(), 300)
            ax.plot(
                grid - 2450000.0, tylenda_period(grid), "--", color="k", lw=1.2,
                label="Tylenda+2011 Eq. (1)",
            )
            ax.legend(fontsize=7)
            break
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "v1309sco_pdot.png", dpi=140)
    print(f"figure  -> {FIGURES / 'v1309sco_pdot.png'}")


if __name__ == "__main__":
    main()
