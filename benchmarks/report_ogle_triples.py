"""Figures and tables for REPORT_ogle_triples.md (the Hong et al. 2022 triples).

Runs the same chain as ``pdot_ogle_triples.py`` (detrend, ``phase_track``,
``ltte_search``) on the six OGLE bulge contact binaries and renders what the
report needs to show:

* ``fig1_folded.png``   the folded light curves and their templates — the
  near-equal minima behind the half-period symmetry;
* ``fig2_oc.png``       the O-C diagrams with the parabola alone, the joint
  parabola + light-time orbit, and the residuals of the joint fit;
* ``fig3_scan.png``     the outer-period scan (the floating-polynomial
  periodogram) against the published outer periods;
* ``fig4_summary.png``  recovered vs published outer periods, and the rate a
  parabola alone reports vs the secular rate of the joint fit;
* ``fig5_symmetry.png`` the star whose O-C slipped by half a period when the
  template's symmetry was not accounted for, before and after;
* ``fig6_simulation.png`` the injection-recovery numbers of
  ``pdot_ltte_recovery.py`` for context.

It also writes ``results/ogle_triples_tables.md`` (the Markdown tables the
report embeds) and prints them. Needs the photometry from
``download_ogle_triples.py`` and, for fig 6, ``results/pdot_ltte_recovery.parquet``.
"""

from __future__ import annotations

import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

warnings.filterwarnings("ignore")
from _common import FIGURES, RESULTS  # noqa: E402
from pdot_ogle_triples import (  # noqa: E402
    DAYS_PER_YEAR,
    DETREND_WIDTH,
    MAX_HARMONICS,
    N_SEGMENTS,
    STARS,
    analyze,
    load,
)

import cuperiod as cup  # noqa: E402
from cuperiod.pdot import track as _track_module  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
    }
)

FIG = FIGURES / "ogle_triples"
FIG.mkdir(parents=True, exist_ok=True)
TABLES = RESULTS / "ogle_triples_tables.md"

#: I-band magnitudes from Hong et al. (2022), VizieR J/AJ/163/157/table3.
I_MAG = {
    "061372": 15.793,
    "246583": 13.928,
    "252457": 14.508,
    "289126": 15.381,
    "299343": 15.370,
    "299686": 13.314,
}
#: Independent outer period of ECL-289126 (Hajdu et al. 2019, MNRAS 485, 2562).
HAJDU_289126 = (5.7, 0.2)
#: Assumed total mass of the contact binary for the minimum third-body mass.
INNER_MASS = 1.5
HOURS = 24.0
T_OFFSET = 2450000.0
SCALE = 1e-7  # rates are quoted in units of 1e-7 d/yr
#: Where the star labels of fig 4 go (two stars share a point).
LABEL_OFFSET = {
    "061372": (-68, 6),
    "299343": (6, -13),
    "252457": (6, 4),
    "289126": (6, 4),
}
#: Tick marks for the logarithmic outer-period axis of fig 3 (years).
SCAN_TICKS = (0.3, 0.5, 1, 2, 3, 5, 7, 10, 14)


def short(star: str) -> str:
    return f"ECL-{star}"


def pm(value: float, err: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    if not np.isfinite(err):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {err:.{digits}f}"


def run_star(star: str) -> dict:
    """The benchmark chain plus the extras the report shows."""
    period = STARS[star][0]
    settings = cup.PdotSettings(n_segments=N_SEGMENTS, max_harmonics=MAX_HARMONICS)
    row, track, best = analyze(star)
    res = track.ltte_search(settings=settings)
    res_ecc = track.ltte_search(settings=settings, eccentric=True)
    # A scan opened to half a cycle, for the star at the grid edge.
    wide = cup.PdotSettings(
        n_segments=N_SEGMENTS, max_harmonics=MAX_HARMONICS, ltte_min_cycles=0.5
    )
    res_wide = track.ltte_search(settings=wide)
    # Hong et al.'s model: a sinusoid on a *linear* ephemeris.
    res_linear = track.ltte_search(settings=settings, degree=1)

    # The first attempt: unwrapping by the full period, as if the template had
    # no half-period symmetry. Reproduced by switching the detection off.
    t, mag, err = load(star)
    flat = cup.detrend(t, mag, DETREND_WIDTH)
    original = _track_module._symmetry_order
    _track_module._symmetry_order = lambda template: 1
    try:
        naive = cup.phase_track((t, flat, err), period=period, settings=settings)
    finally:
        _track_module._symmetry_order = original
    diff = naive.delay - track.delay
    branch = np.abs(((diff + 0.5 * period) % period) - 0.5 * period) > 0.25 * period
    n_slipped = int(branch.sum())

    inflation = best.error_inflation
    floor = cup.detectable_pdot(
        period, float(np.median(track.delay_err)) * inflation, times=track.t_mid
    )
    return {
        "star": star,
        "row": row,
        "track": track,
        "naive": naive,
        "n_slipped": n_slipped,
        "res": res,
        "best": best,
        "res_ecc": res_ecc,
        "res_wide": res_wide,
        "res_linear": res_linear,
        "floor_d_per_yr": floor.sigma_pdot_d_per_yr,
        "t": t,
        "flat": flat,
        "err": err,
    }


# --- figures -------------------------------------------------------------------
def fig_folded(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    dots = {"ms": 2, "color": "0.55", "alpha": 0.35, "rasterized": True}
    for ax, run in zip(axes.ravel(), runs, strict=True):
        star, track = run["star"], run["track"]
        period = STARS[star][0]
        phase = ((run["t"] - track.t_ref) / period) % 1.0
        ax.plot(phase, run["flat"], ".", **dots)
        ax.plot(phase + 1.0, run["flat"], ".", **dots)
        grid = np.linspace(0.0, 2.0, 600)
        ax.plot(grid, track.template(grid * period), "-", color="C3", lw=1.4)
        lo, hi = np.percentile(run["flat"], [0.5, 99.5])
        pad = 0.08 * (hi - lo)
        ax.set_ylim(hi + pad, lo - pad)
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("phase (published period)")
        ax.set_ylabel("I (detrended)")
        ax.set_title(
            f"{short(star)}   P = {period:.6f} d   I = {I_MAG[star]:.1f}   "
            f"n = {run['t'].size}\n"
            f"template: {track.template.n_harmonics} harmonics, "
            f"fold symmetry {track.symmetry_order}",
            fontsize=9,
        )
    fig.suptitle(
        "OGLE-III + OGLE-IV I-band photometry folded on the published period, "
        "with the phase_track template"
    )
    fig.savefig(FIG / "fig1_folded.png")
    plt.close(fig)


def fig_oc(runs: list[dict]) -> None:
    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[3.0, 1.1, 3.0, 1.1])
    for i, run in enumerate(runs):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[2 * r, c])
        axr = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        star, track, best, row = run["star"], run["track"], run["best"], run["row"]
        zero = float(best.coefficients[0])
        t = track.t_mid
        oc = (track.delay - zero) * HOURS
        yerr = track.delay_err * best.error_inflation * HOURS
        grid = np.linspace(t.min() - 60.0, t.max() + 60.0, 600)
        ax.errorbar(t - T_OFFSET, oc, yerr=yerr, fmt="o", ms=3.5, color="C0", zorder=3)
        ax.plot(
            grid - T_OFFSET,
            (track.quadratic_fit.predict(grid) - zero) * HOURS,
            "-",
            color="C3",
            lw=1.3,
            label="parabola alone (constant dP/dt)",
        )
        ax.plot(
            grid - T_OFFSET,
            (best.predict(grid) - zero) * HOURS,
            "-",
            color="C2",
            lw=1.6,
            label="parabola + light-time orbit",
        )
        ax.plot(
            grid - T_OFFSET,
            (best.secular_part(grid) - zero) * HOURS,
            "--",
            color="C2",
            lw=1.0,
            label="its secular part",
        )
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper left")
        ax.set_ylabel("O-C (h)")
        p2 = pm(row["ltte_p2_yr"], row["ltte_p2_err_yr"])
        amp = pm(row["ltte_amplitude_s"], row["ltte_amplitude_err_s"], 0)
        ax.set_title(
            f"{short(star)}:  P2 = {p2} yr  (published {row['published_p2_yr']} ± "
            f"{row['published_p2_err_yr']})\nA = {amp} s   "
            f"ΔBIC = +{row['ltte_delta_bic']:.0f}   χ²ᵣ {row['ltte_chi2_reduced']:.1f}",
            fontsize=9,
        )
        plt.setp(ax.get_xticklabels(), visible=False)
        resid = (track.delay - best.predict(t)) * HOURS
        axr.errorbar(t - T_OFFSET, resid, yerr=yerr, fmt="o", ms=2.5, color="0.4")
        axr.axhline(0.0, color="C2", lw=1.0)
        axr.set_ylabel("resid (h)")
        axr.set_xlabel("HJD - 2450000")
    fig.suptitle(
        "O-C from phase_track (40 segments) with the parabola alone and the joint "
        "fit of ltte_search; residuals about the joint fit"
    )
    fig.savefig(FIG / "fig2_oc.png")
    plt.close(fig)


def fig_scan(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.0), layout="constrained")
    for ax, run in zip(axes.ravel(), runs, strict=True):
        star, res, best, row = run["star"], run["res"], run["best"], run["row"]
        p2_yr = res.outer_period / DAYS_PER_YEAR
        ax.plot(p2_yr, res.delta_chi2, "-", color="C0", lw=1.0)
        pub, e_pub = STARS[star][1], STARS[star][2]
        ax.axvspan(
            pub - e_pub, pub + e_pub, color="0.6", alpha=0.45, label="published ± 1σ"
        )
        ax.axvline(best.outer_period_yr, color="C2", lw=1.4, label="ltte_search best")
        if star == "289126":
            ax.axvline(
                HAJDU_289126[0], color="C1", ls="--", lw=1.2, label="Hajdu+ 2019"
            )
        ax.set_xscale("log")
        lo, hi = float(p2_yr.min()), float(p2_yr.max())
        ax.set_xlim(lo, hi)
        ticks = [v for v in SCAN_TICKS if lo <= v <= hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("trial outer period (yr)")
        ax.set_ylabel("Δχ²  (polynomial + sinusoid vs polynomial)")
        edge = "   (at the scan's long end)" if best.at_grid_edge else ""
        ax.set_title(
            f"{short(star)}:  peak {best.outer_period_yr:.2f} yr,  "
            f"ΔBIC = +{row['ltte_delta_bic']:.0f}{edge}",
            fontsize=9,
        )
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "The outer-period scan: chi-squared improvement of parabola + sinusoid over "
        "the parabola, per trial period (scan ends at one baseline)"
    )
    fig.savefig(FIG / "fig3_scan.png")
    plt.close(fig)


def fig_summary(runs: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for run in runs:
        star, row = run["star"], run["row"]
        ax1.errorbar(
            row["published_p2_yr"],
            row["ltte_p2_yr"],
            xerr=row["published_p2_err_yr"],
            yerr=row["ltte_p2_err_yr"],
            fmt="o",
            color="C2",
            capsize=2,
        )
        ax1.annotate(
            short(star),
            (row["published_p2_yr"], row["ltte_p2_yr"]),
            xytext=LABEL_OFFSET.get(star, (6, -10)),
            textcoords="offset points",
            fontsize=8,
        )
        if star == "289126":
            ax1.errorbar(
                HAJDU_289126[0],
                row["ltte_p2_yr"],
                xerr=HAJDU_289126[1],
                yerr=row["ltte_p2_err_yr"],
                fmt="s",
                mfc="none",
                color="C1",
                capsize=2,
                label="ECL-289126 vs Hajdu+ 2019",
            )
    lim = (3.0, 18.0)
    ax1.plot(lim, lim, "-", color="0.5", lw=1.0, label="1:1")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("published outer period (yr), Hong et al. 2022")
    ax1.set_ylabel("ltte_search outer period (yr)")
    ax1.set_title("Outer periods: raw photometry vs Wilson-Devinney timings")
    ax1.legend(fontsize=8, loc="upper left")

    x = np.arange(len(runs))
    quad = np.array([r["row"]["quad_pdot_d_per_yr"] for r in runs]) / SCALE
    quad_e = np.array([r["row"]["quad_pdot_err_d_per_yr"] for r in runs]) / SCALE
    sec = np.array([r["row"]["secular_pdot_d_per_yr"] for r in runs]) / SCALE
    sec_e = np.array([r["row"]["secular_pdot_err_d_per_yr"] for r in runs]) / SCALE
    ax2.errorbar(
        x - 0.15,
        quad,
        yerr=quad_e,
        fmt="o",
        color="C3",
        capsize=3,
        label="parabola alone",
    )
    ax2.errorbar(
        x + 0.15,
        sec,
        yerr=sec_e,
        fmt="o",
        color="C2",
        capsize=3,
        label="secular rate, cycle fitted jointly",
    )
    ax2.axhline(0.0, color="0.5", lw=1.0)
    for xi, (q, qe) in enumerate(zip(quad, quad_e, strict=True)):
        if abs(q) >= 3.0 * qe:
            ax2.annotate(
                f"{abs(q) / qe:.1f}σ",
                (xi - 0.15, q + np.sign(q) * qe),
                xytext=(0, 6 if q > 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="C3",
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([short(r["star"]) for r in runs], rotation=20)
    ax2.set_ylabel("dP/dt  (10⁻⁷ d/yr)")
    ax2.set_title(
        "The rate a parabola reports, and what is left once the cycle is removed"
    )
    ax2.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / "fig4_summary.png")
    plt.close(fig)


def fig_symmetry(run: dict) -> None:
    star, period = run["star"], STARS[run["star"]][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, track, title in (
        (ax1, run["naive"], "unwrapped by the full period (first attempt)"),
        (ax2, run["track"], "unwrapped by the template's symmetry period P/2"),
    ):
        fit = track.quadratic_fit
        zero = float(fit.coefficients[0])
        t = track.t_mid
        inflate = float(np.sqrt(max(1.0, fit.chi2_reduced)))
        ax.errorbar(
            t - T_OFFSET,
            (track.delay - zero) * HOURS,
            yerr=track.delay_err * inflate * HOURS,
            fmt="o",
            ms=3.5,
            color="C0",
            zorder=3,
        )
        grid = np.linspace(t.min() - 60.0, t.max() + 60.0, 400)
        ax.plot(
            grid - T_OFFSET,
            (fit.predict(grid) - zero) * HOURS,
            "-",
            color="C3",
            lw=1.3,
            label="parabola",
        )
        ax.set_xlabel("HJD - 2450000")
        ax.set_ylabel("O-C (h)")
        rate_err = fit.pdot_err_scaled * DAYS_PER_YEAR / SCALE
        rate = pm(fit.pdot_d_per_yr / SCALE, rate_err, 1)
        flag = "   [wrap risk flagged]" if track.wrap_risk else ""
        ax.set_title(
            f"{short(star)}: {title}\nparabola: dP/dt = {rate} ×10⁻⁷ d/yr,  "
            f"χ²ᵣ = {fit.chi2_reduced:.0f}{flag}",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper left")
    ax1.axhline(0.5 * period * HOURS, color="0.5", ls=":", lw=1.0)
    ax1.text(
        0.99,
        0.02,
        f"{run['n_slipped']} of {run['track'].n_segments} segments landed on the "
        f"other branch (P/2 = {0.5 * period * HOURS:.2f} h)",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(FIG / "fig5_symmetry.png")
    plt.close(fig)


def fig_simulation() -> pd.DataFrame | None:
    path = RESULTS / "pdot_ltte_recovery.parquet"
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    planted = table[table["planted"]]
    bins = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5))
    labels, rows = [], []
    for lo, hi in bins:
        sel = planted[(planted["ratio"] >= lo) & (planted["ratio"] < hi)]
        labels.append(f"{lo:.2f}–{hi:.2f}\n(n = {len(sel)})")
        rows.append(
            {
                "bin": f"{lo:.2f}–{hi:.2f}",
                "n": len(sel),
                "parabola alone ≥ 3σ": sel["quad_spurious_3sigma"].mean(),
                "cycle detected (ΔBIC > 10)": sel["ltte_detected"].mean(),
                "P2 within 10%": sel["period_recovered"].mean(),
                "joint secular ≥ 3σ": sel["secular_spurious_3sigma"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    keys = [
        "parabola alone ≥ 3σ",
        "cycle detected (ΔBIC > 10)",
        "P2 within 10%",
        "joint secular ≥ 3σ",
    ]
    colors = ["C3", "C0", "C2", "C8"]
    width = 0.2
    x = np.arange(len(bins))
    for k, (key, color) in enumerate(zip(keys, colors, strict=True)):
        offset = x + (k - 1.5) * width
        ax.bar(offset, 100.0 * summary[key], width, color=color, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("planted outer period / baseline")
    ax.set_ylabel("per cent of planted stars")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Synthetic light-time orbits with no secular drift (pdot_ltte_recovery.py): "
        "the parabola's false rate, and the joint fit's recovery"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(FIG / "fig6_simulation.png")
    plt.close(fig)
    nulls = table[~table["planted"]]
    summary.attrs["nulls"] = {
        "n": int(len(nulls)),
        "cycle_false": float(nulls["ltte_detected"].mean()),
        "parabola_false": float(nulls["quad_spurious_3sigma"].mean()),
        "median_seconds": float(table["seconds"].median()),
    }
    return summary


# --- tables --------------------------------------------------------------------
def tables(runs: list[dict], sim: pd.DataFrame | None) -> str:
    out = []
    out.append("### Table 1 — the sample and the tracking\n")
    out.append(
        "| Star | I | P (d) | points | baseline (yr) | segments | template harmonics "
        "| fold symmetry |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        row, star = run["row"], run["star"]
        out.append(
            f"| {short(star)} | {I_MAG[star]:.1f} | {row['period_d']:.6f} | "
            f"{row['n_points']} | {row['baseline_yr']:.1f} | {row['n_segments']} | "
            f"{row['n_harmonics']} | {row['symmetry_order']} |"
        )

    out.append("\n### Table 2 — the light-time orbits\n")
    out.append(
        "| Star | P2 (yr), this work | P2 (yr), Hong et al. | other | agreement | "
        "A (s) | cycles covered | ΔBIC | χ²ᵣ (joint) |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        row, best = run["row"], run["best"]
        other = "—"
        if run["star"] == "289126":
            other = f"{HAJDU_289126[0]} ± {HAJDU_289126[1]} (Hajdu+ 2019)"
        edge = " (scan edge)" if best.at_grid_edge else ""
        p2 = pm(row["ltte_p2_yr"], row["ltte_p2_err_yr"]) + edge
        amp = pm(row["ltte_amplitude_s"], row["ltte_amplitude_err_s"], 0)
        out.append(
            f"| {short(run['star'])} | {p2} | "
            f"{row['published_p2_yr']} ± {row['published_p2_err_yr']} | {other} | "
            f"{row['p2_agreement_sigma']:.1f}σ | {amp} | "
            f"{row['ltte_n_cycles']:.2f} | +{row['ltte_delta_bic']:.0f} | "
            f"{row['ltte_chi2_reduced']:.1f} |"
        )

    out.append("\n### Table 3 — the rate a parabola reports, and the secular rate\n")
    out.append(
        "| Star | parabola alone, dP/dt (10⁻⁷ d/yr) | S/N | χ²ᵣ (parabola) | "
        "F-test p (quadratic vs linear) | secular dP/dt (10⁻⁷ d/yr) | S/N | "
        "floor from the timing errors (10⁻⁷ d/yr) |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        row = run["row"]
        q = row["quad_pdot_d_per_yr"] / SCALE
        qe = row["quad_pdot_err_d_per_yr"] / SCALE
        s = row["secular_pdot_d_per_yr"] / SCALE
        se = row["secular_pdot_err_d_per_yr"] / SCALE
        out.append(
            f"| {short(run['star'])} | {q:+.2f} ± {qe:.2f} | {abs(q) / qe:.1f} | "
            f"{row['quad_chi2_reduced']:.1f} | {row['quad_f_test_p']:.1e} | "
            f"{s:+.2f} ± {se:.2f} | {abs(s) / se:.1f} | "
            f"{run['floor_d_per_yr'] / SCALE:.2f} |"
        )

    out.append("\n### Table 4 — what the orbits imply\n")
    out.append(
        "| Star | a₁₂ sin i (au) | f(m) (M☉) | "
        f"m₃ sin i ≥ (M☉, for M₁₂ = {INNER_MASS} M☉) |"
    )
    out.append("| --- | --- | --- | --- |")
    for run in runs:
        best = run["best"]
        out.append(
            f"| {short(run['star'])} | {best.a_sini_au:.3f} | "
            f"{best.mass_function:.4f} | {best.minimum_third_mass(INNER_MASS):.2f} |"
        )

    out.append(
        "\n### Table 5 — other models: a linear ephemeris, an eccentric orbit, "
        "a scan opened to half a cycle\n"
    )
    out.append(
        "| Star | P2 (yr), sinusoid on a linear ephemeris | e | "
        "P2 (yr), eccentric fit | A (s), eccentric fit | "
        "ΔBIC eccentric − circular | P2 (yr), scan to half a cycle |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        best, ecc, wide = run["best"], run["res_ecc"].best, run["res_wide"].best
        lin = run["res_linear"].best
        lin_txt = pm(lin.outer_period_yr, lin.outer_period_err_yr * lin.error_inflation)
        if lin.at_grid_edge:
            lin_txt += " (scan edge)"
        inflate = ecc.error_inflation
        wide_txt = pm(
            wide.outer_period_yr, wide.outer_period_err_yr * wide.error_inflation
        )
        if wide.at_grid_edge:
            wide_txt += " (scan edge)"
        out.append(
            f"| {short(run['star'])} | {lin_txt} | "
            f"{pm(ecc.eccentricity, ecc.eccentricity_err * inflate)} | "
            f"{pm(ecc.outer_period_yr, ecc.outer_period_err_yr * inflate)} | "
            f"{pm(ecc.amplitude_s, ecc.amplitude_err_s * inflate, 0)} | "
            f"{ecc.bic - best.bic:+.1f} | {wide_txt} |"
        )

    out.append("\n### Table 6 — the half-period slips of the first attempt\n")
    out.append(
        "| Star | segments on the other branch | parabola dP/dt then (10⁻⁷ d/yr) | "
        "χ²ᵣ then | wrap risk flagged then |"
    )
    out.append("| --- | --- | --- | --- | --- |")
    for run in runs:
        naive = run["naive"]
        rate_err = naive.pdot_err_scaled * DAYS_PER_YEAR / SCALE
        rate = pm(naive.pdot_d_per_yr / SCALE, rate_err, 1)
        out.append(
            f"| {short(run['star'])} | {run['n_slipped']} / {naive.n_segments} | "
            f"{rate} | {naive.quadratic_fit.chi2_reduced:.0f} | "
            f"{'yes' if naive.wrap_risk else 'no'} |"
        )

    if sim is not None:
        out.append(
            "\n### Table 7 — synthetic light-time orbits (pdot_ltte_recovery.py)\n"
        )
        out.append(
            "| P2 / baseline | n | parabola alone ≥ 3σ | cycle detected (ΔBIC > 10) | "
            "P2 within 10% | joint secular ≥ 3σ |"
        )
        out.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in sim.iterrows():
            out.append(
                f"| {r['bin']} | {r['n']} | {100 * r['parabola alone ≥ 3σ']:.0f}% | "
                f"{100 * r['cycle detected (ΔBIC > 10)']:.0f}% | "
                f"{100 * r['P2 within 10%']:.0f}% | "
                f"{100 * r['joint secular ≥ 3σ']:.0f}% |"
            )
        nulls = sim.attrs["nulls"]
        out.append(
            f"\nNulls (no orbit, n = {nulls['n']}): cycle falsely detected on "
            f"{100 * nulls['cycle_false']:.0f}%, parabola ≥ 3σ on "
            f"{100 * nulls['parabola_false']:.0f}%; median "
            f"{nulls['median_seconds']:.2f} s per star for tracking + search."
        )
    return "\n".join(out) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = [run_star(star) for star in STARS]
    fig_folded(runs)
    fig_oc(runs)
    fig_scan(runs)
    fig_summary(runs)
    worst = max(runs, key=lambda r: r["n_slipped"])
    fig_symmetry(worst)
    sim = fig_simulation()
    text = tables(runs, sim)
    TABLES.write_text(text, encoding="utf-8")
    print(text)
    for run in runs:
        res = run["res"]
        lo = res.outer_period.min() / DAYS_PER_YEAR
        hi = res.outer_period.max() / DAYS_PER_YEAR
        print(
            f"{short(run['star'])}: scan {lo:.2f}-{hi:.2f} yr, "
            f"{res.outer_period.size} trials; pdot_change = "
            f"{res.pdot_change * DAYS_PER_YEAR / SCALE:+.2f} x1e-7 d/yr; "
            f"omega = {run['best'].omega:.2f}; slipped {run['n_slipped']}"
        )
    print(f"-> {FIG}  and  {TABLES}")


if __name__ == "__main__":
    main()
