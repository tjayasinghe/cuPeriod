"""V1309 Sco: the period-change tools on the binary that was caught merging.

V1309 Sco erupted in September 2008 as a red nova. Because it sits in the OGLE
bulge fields it is the one stellar merger with dense photometry of the
*progenitor*, and Tylenda et al. (2011, A&A 528, A114) used that archive to show
the progenitor was a contact binary whose orbital period was decaying at an
accelerating rate — the first direct observation of a binary spiralling in to a
merger. Their result, against which this script is a blind check: an orbital
period near 1.4 d, and an orbital-decay timescale P/|dP/dt| shortening from
roughly 1000 yr in 2001 to roughly 170 yr in 2007.

That makes it an unusually demanding real-data test of the v1.3 tools:

* the drift is enormous (~5 days of accumulated O-C over the baseline, several
  orbital cycles), so the cycle-count ambiguity is real rather than theoretical;
* the decay is *exponential*, so a constant-dP/dt model is knowingly wrong and
  the tools must say so rather than quietly returning a number;
* the star brightens by ~0.8 mag over the baseline as it spirals in, which
  swamps an undetrended periodogram;
* the light-curve shape itself evolves, so a fixed template cannot fit and the
  formal timing errors are optimistic.

Data: the authors' own supplementary photometry, mirrored by OGLE at
ftp://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/ (I-band, 2001-2009). Run
``download_v1309sco.py`` first. All analysis here is restricted to the quiescent
pre-merger window; the violent phase began in March 2008 (HJD ~2454527).

Writes results/pdot_v1309sco.parquet and figures/v1309sco_pdot.png.
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

#: Tylenda et al. (2011): "the orbital period decreased by 1.2% during 6 years".
PUBLISHED_DECREASE_PERCENT = -1.2

#: Their quoted orbital period (their Sect. 5.3.2, used for the Kepler-law masses).
PUBLISHED_PERIOD = 1.43

SEC_PER_YR = 86400.0 * 365.25


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The OGLE-III I-band light curve, restricted to the quiescent window."""
    if not PHOTOMETRY.exists():
        raise SystemExit(
            f"{PHOTOMETRY} not found — run 'python benchmarks/download_v1309sco.py'"
        )
    t, mag, err = np.loadtxt(PHOTOMETRY, unpack=True)
    keep = (t >= WINDOW[0]) & (t < WINDOW[1])
    return t[keep], mag[keep], err[keep]


def detrend(t: np.ndarray, y: np.ndarray, width: float = DETREND_WIDTH) -> np.ndarray:
    """Remove the secular brightening with a running median."""
    out = np.empty_like(y)
    for i, ti in enumerate(t):
        out[i] = y[i] - np.median(y[np.abs(t - ti) < 0.5 * width])
    return out


def measure(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    seed_period: float,
    label: str,
    window_rayleigh: float = 25.0,
    n_segments: int = 6,
) -> dict[str, float | str]:
    """Search then track one interval; returns a row of the summary table."""
    dm = detrend(t, mag)
    res = cup.pdot_search(
        (t, dm, err),
        period=seed_period,
        backend="cpu",
        settings=cup.PdotSettings(
            window_rayleigh=window_rayleigh, max_cycle_drift=12.0
        ),
    )
    best = res.best
    track = cup.phase_track(
        (t, dm, err),
        period=best.period,
        pdot=best.pdot,
        settings=cup.PdotSettings(harmonics=4, n_segments=n_segments),
    )
    fit = track.quadratic_fit
    # The light-curve shape evolves, so a fixed template leaves a poor chi2 and
    # the formal timing errors are optimistic; inflate rather than quote them.
    inflate = float(np.sqrt(max(1.0, fit.chi2_reduced)))
    sigma = fit.pdot_err * inflate
    return {
        "interval": label,
        "n": int(t.size),
        "baseline": float(t.max() - t.min()),
        "t_ref": best.t_ref,
        "period": best.period,
        "pdot_search": best.pdot,
        "pdot": track.pdot,
        "pdot_err": sigma,
        "pdot_s_per_yr": track.pdot * SEC_PER_YR,
        "pdot_err_s_per_yr": sigma * SEC_PER_YR,
        "timescale_yr": abs(best.period / track.pdot) / 365.25,
        "f_test_p": track.f_test_p,
        "chi2_reduced": fit.chi2_reduced,
        "power": best.power,
        "zero_pdot_power": best.zero_pdot_power,
        "wrap_risk": bool(track.wrap_risk),
    }


def figure(t: np.ndarray, mag: np.ndarray, err: np.ndarray, rows: list) -> None:
    """Light curve, O-C diagram, and the shortening decay timescale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dm = detrend(t, mag)
    res = cup.pdot_search(
        (t, dm, err), period=1.4263, backend="cpu",
        settings=cup.PdotSettings(window_rayleigh=60.0, max_cycle_drift=12.0),
    )
    best = res.best
    track = cup.phase_track(
        (t, dm, err), period=best.period, pdot=best.pdot,
        settings=cup.PdotSettings(harmonics=4, n_segments=10),
    )
    fit = track.quadratic_fit

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    ax.plot(t - 2450000, mag, ".", ms=2, color="0.35")
    ax.invert_yaxis()
    ax.set_xlabel("HJD - 2450000")
    ax.set_ylabel("I (mag)")
    ax.set_title("V1309 Sco (OGLE-III), quiescent window")

    ax = axes[0, 1]
    im = ax.pcolormesh(
        res.frequency, res.pdot_over_p * SEC_PER_YR * res.best.period,
        res.power, shading="auto", cmap="viridis",
    )
    ax.plot(best.frequency, best.pdot_s_per_yr, "r+", ms=12)
    ax.set_xlabel("frequency (c/d)")
    ax.set_ylabel("dP/dt (s/yr)")
    ax.set_title("pdot_search map")
    fig.colorbar(im, ax=ax, label="GLS power")

    ax = axes[1, 0]
    dt = track.t_mid - fit.t_ref
    ax.errorbar(track.t_mid - 2450000, track.delay * 24.0,
                yerr=track.delay_err * 24.0 * np.sqrt(fit.chi2_reduced),
                fmt="o", ms=4, color="C0")
    grid = np.linspace(dt.min(), dt.max(), 200)
    ax.plot(grid + fit.t_ref - 2450000, fit.predict(grid + fit.t_ref) * 24.0,
            "-", color="C3", label="quadratic (constant dP/dt)")
    ax.set_xlabel("HJD - 2450000")
    ax.set_ylabel("O-C (hours)")
    ax.set_title("phase_track O-C")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    piece = [r for r in rows if r["interval"] != "2001-2007 (all)"]
    whole = [r for r in rows if r["interval"] == "2001-2007 (all)"][0]
    # Period against time: each interval's own constant-rate solution, and the
    # single global one that has to average an accelerating decay.
    for r in piece:
        span = np.array([-0.5, 0.5]) * r["baseline"]
        ax.plot(
            (r["t_ref"] + span) - 2450000,
            (r["period"] + r["pdot"] * span),
            "-", lw=2.5, color="C0",
            label="piecewise constant rate" if r is piece[0] else None,
        )
    span = np.array([-0.5, 0.5]) * whole["baseline"]
    ax.plot((whole["t_ref"] + span) - 2450000,
            whole["period"] + whole["pdot"] * span,
            "--", color="C3", label="single constant rate")
    ax.axhline(PUBLISHED_PERIOD, color="k", ls=":", alpha=0.6,
               label=f"Tylenda+2011: {PUBLISHED_PERIOD} d")
    ax.set_xlabel("HJD - 2450000")
    ax.set_ylabel("P (d)")
    ax.set_title("orbital period vs time")
    ax.legend(fontsize=7)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "v1309sco_pdot.png", dpi=140)
    print(f"\nfigure -> {FIGURES / 'v1309sco_pdot.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    t, mag, err = load()
    print(
        f"V1309 Sco OGLE-III I-band: n={t.size}, "
        f"HJD {t.min():.1f}-{t.max():.1f} ({(t.max()-t.min())/365.25:.2f} yr)"
    )
    print("Published (Tylenda et al. 2011): P ~ 1.4 d; P/|dP/dt| ~1000 yr (2001)"
          " -> ~170 yr (2007)\n")

    rows = [
        measure(t, mag, err, seed_period=1.4263, label="2001-2007 (all)",
                window_rayleigh=60.0, n_segments=10),
    ]
    for lo, hi, label, seed in [
        (2452100, 2453300, "2001-2004", 1.4450),
        (2453300, 2454300, "2004-2007", 1.4300),
    ]:
        m = (t >= lo) & (t < hi)
        rows.append(measure(t[m], mag[m], err[m], seed_period=seed, label=label))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / "pdot_v1309sco.parquet", index=False)

    print(f"{'interval':18s} {'n':>4s} {'P (d)':>9s} {'dP/dt (s/yr)':>18s}"
          f" {'P/|Pdot| (yr)':>13s} {'power':>14s}")
    for r in rows:
        print(
            f"{r['interval']:18s} {r['n']:4d} {r['period']:9.5f}"
            f" {r['pdot_s_per_yr']:10.1f} +-{r['pdot_err_s_per_yr']:5.1f}"
            f" {r['timescale_yr']:13.0f}"
            f"   {r['power']:.3f} vs {r['zero_pdot_power']:.3f}"
        )
    print(
        "\n(power: at the best cell vs anywhere on the constant-period row;"
        " dP/dt errors are\n formal values inflated by sqrt(chi2_r) because the"
        " light-curve shape itself evolves.)"
    )

    # The headline comparison. A constant rate cannot represent an accelerating
    # decay, so the global fit overshoots; letting each half carry its own rate
    # recovers the published total.
    whole = rows[0]
    piece = rows[1:]
    total_whole = whole["pdot"] * whole["baseline"]
    total_piece = sum(r["pdot"] * r["baseline"] for r in piece)
    print("\ntotal fractional period change over the ~6-year baseline:")
    print(f"  single constant rate : {100*total_whole/whole['period']:+.2f} %")
    print(f"  piecewise (2 rates)  : {100*total_piece/whole['period']:+.2f} %")
    print(f"  published            : {PUBLISHED_DECREASE_PERCENT:+.1f} %"
          "  (Tylenda et al. 2011)")

    # Is a constant rate adequate? Let the O-C answer.
    dm = detrend(t, mag)
    best = cup.pdot_search(
        (t, dm, err), period=1.4263, backend="cpu",
        settings=cup.PdotSettings(window_rayleigh=60.0, max_cycle_drift=12.0),
    ).best
    track = cup.phase_track(
        (t, dm, err), period=best.period, pdot=best.pdot,
        settings=cup.PdotSettings(harmonics=4, n_segments=10),
    )
    dt = track.t_mid - track.quadratic_fit.t_ref
    print("\nO-C residual rms vs ephemeris order (2 = constant dP/dt):")
    for deg in (2, 3, 4):
        coef = np.polyfit(dt, track.delay, deg, w=1.0 / track.delay_err)
        r = track.delay - np.polyval(coef, dt)
        print(f"  degree {deg}: {r.std()*24*60:7.1f} min")
    print(
        f"  chi2_r={track.quadratic_fit.chi2_reduced:.0f},"
        f" F-test p={track.f_test_p:.1e} -> the drift is overwhelmingly real,"
        "\n  but the falling residuals say a single constant rate does not"
        " describe it:\n  the decay is exponential (Tylenda et al. 2011, their"
        " Fig. 2)."
    )
    if not args.no_figure:
        figure(t, mag, err, rows)


if __name__ == "__main__":
    main()
