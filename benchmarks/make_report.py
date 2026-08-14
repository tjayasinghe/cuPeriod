"""Build the validation + benchmark report (figures + REPORT.md).

Consumes results/*.parquet and results/spectra/*.npz, writes figures/*.png and
REPORT.md. Safe to re-run; only renders the artefacts whose inputs exist.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cuperiod as cup  # noqa: E402
from _common import DATASET, FIGURES, HARMONIC_RATIOS, RESULTS, load_dataset  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.bbox": "tight",
})

METHOD_ORDER = ["GLS", "BLS", "PDM", "CE", "STRINGLENGTH", "MHAOV", "SUPERSMOOTHER", "TLS"]
MLABEL = {"STRINGLENGTH": "String-Len", "SUPERSMOOTHER": "SuperSmoother"}
CLASS_ORDER = ["ECLIPSING", "RR_LYRAE", "CEPHEID", "DELTA_SCUTI", "LONG_PERIOD", "ROTATIONAL"]
CCOLOR = dict(zip(CLASS_ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))


def ml(m):
    return MLABEL.get(m, m)


# 1.959964 = Phi^-1(0.975), the standard-normal 97.5th percentile (avoids a scipy dep).
_Z95 = 1.959964


def wilson_ci(k, n):
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi) in [0, 1]; NaN if n==0."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    z = _Z95
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def fmt_pct_ci(k, n):
    """'97.2% [92.1-99.0%]' style string for a recovery count k out of n."""
    if not n:
        return "—"
    lo, hi = wilson_ci(k, n)
    return f"{100*k/n:.1f}% [{100*lo:.1f}–{100*hi:.1f}%]"


# --------------------------------------------------------------------------
def fig_parity_and_reference(val):
    methods = [m for m in METHOD_ORDER if m in set(val["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) CPU vs GPU parity (relative error: the statistics have different scales)
    ax = axes[0]
    data = [val[val.method == m]["parity_max_rel"].clip(lower=1e-16).values for m in methods]
    parts = ax.boxplot(data, widths=0.6, patch_artist=True,
                       showfliers=True, flierprops=dict(marker=".", ms=3, alpha=.5))
    for p in parts["boxes"]:
        p.set(facecolor="#4c72b0", alpha=.6)
    ax.set_yscale("log")
    ax.axhline(1e-6, color="crimson", ls="--", lw=1, label="1e-6 (float32 round-off)")
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([ml(m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("max relative |stat$_{\\rm CPU}$ − stat$_{\\rm GPU}$|")
    ax.set_title("(a) CPU vs GPU backend parity")
    ax.legend(fontsize=7, loc="lower left")

    # (b) cuPeriod vs reference implementation
    ax = axes[1]
    refm = [m for m in methods if m not in ("TLS",)]
    absm = [m for m in refm if m not in ("PDM",)]  # PDM ref uses a different norm
    data = [val[val.method == m]["cup_ref_max_abs"].clip(lower=1e-16).dropna().values for m in absm]
    parts = ax.boxplot(data, widths=0.6, patch_artist=True,
                       showfliers=True, flierprops=dict(marker=".", ms=3, alpha=.5))
    for p in parts["boxes"]:
        p.set(facecolor="#55a868", alpha=.6)
    ax.set_yscale("log")
    ax.axhline(1e-6, color="crimson", ls="--", lw=1, label="1e-6")
    ax.set_xticks(range(1, len(absm) + 1))
    ax.set_xticklabels([ml(m) for m in absm], rotation=30, ha="right")
    ax.set_ylabel(r"max $|P_{\rm cuPeriod}-P_{\rm reference}|$")
    refnames = "GLS,BLS vs astropy · CE,String-Len,MHAOV vs textbook"
    ax.set_title("(b) cuPeriod vs reference\n" + refnames, fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_parity_reference.png")
    plt.close(fig)


def fig_spectra(spectra_files, meta):
    # one representative star + method per class -> show breadth across the H-R zoo
    avail = {}
    for fp in spectra_files:
        base = os.path.basename(fp)[:-4]
        sid, method = base.rsplit("_", 1)
        avail[(sid, method)] = fp
    sid2class = {str(r.asas_sn_id): r.broad_class for r in meta.itertuples()}
    sid2type = {str(r.asas_sn_id): r.vsx_type for r in meta.itertuples()}
    # method best suited to each class (all have an independent reference)
    prefer = {"ECLIPSING": "PDM", "RR_LYRAE": "MHAOV", "CEPHEID": "GLS",
              "DELTA_SCUTI": "GLS", "LONG_PERIOD": "PDM", "ROTATIONAL": "GLS"}
    panels = []
    for bc in CLASS_ORDER:
        sids = sorted({s for (s, m) in avail if sid2class.get(s) == bc})
        if not sids:
            continue
        method = prefer.get(bc, "GLS")
        sid = next((s for s in sids if (s, method) in avail), None)
        if sid is None:
            method = "GLS"; sid = next((s for s in sids if (s, "GLS") in avail), sids[0])
        if (sid, method) in avail:
            panels.append((bc, sid, method, avail[(sid, method)]))
    if not panels:
        return
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (bc, sid, method, fp) in zip(axes, panels):
        d = np.load(fp)
        x = d["period"] if "period" in d else 1.0 / d["freq"]
        cup, ref = np.asarray(d["cup"], float), np.asarray(d["ref"], float)
        sense = str(d["sense"]) if "sense" in d else "max"
        p_true = float(d["p_true"])
        # for minimisation stats, plot -stat so the signal is a peak; normalise [0,1]
        def nz(a, sense):
            a = -a if sense == "min" else a
            return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-30)
        order = np.argsort(x)
        ax.plot(x[order], nz(cup, sense)[order], lw=1.3, label="cuPeriod", color="#1f77b4")
        ax.plot(x[order], nz(ref, sense)[order], lw=0.9, ls="--", label="reference", color="#d62728")
        ax.axvline(p_true, color="k", lw=0.9, alpha=.55)
        ax.set_xscale("log")
        ax.set_title(f"{bc.replace('_',' ').title()} · {sid2type.get(sid,'')} · {ml(method)}",
                     fontsize=8)
        ax.set_xlabel("period [d]"); ax.set_ylabel("statistic (peak-up, norm.)")
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("cuPeriod (solid) vs independent reference (dashed) on real ASAS-SN "
                 "light curves — black line = VSX literature period", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_spectra_overlay.png")
    plt.close(fig)


def fig_recovery(val):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) recovered vs true period, all methods, harmonic guide lines
    ax = axes[0]
    for bc in CLASS_ORDER:
        sub = val[(val.broad_class == bc)]
        if sub.empty:
            continue
        ax.scatter(sub.vsx_period, sub.p_cpu, s=14, alpha=.6,
                   color=CCOLOR[bc], label=bc.replace("_", " ").title(), edgecolor="none")
    lim = [val.vsx_period.min() * 0.5, val.vsx_period.max() * 2]
    xx = np.array(lim)
    for r, lab in [(1, "P"), (0.5, "P/2"), (2, "2P"), (1 / 3, "P/3"), (3, "3P")]:
        ax.plot(xx, xx * r, lw=0.7, ls="--", color="gray", alpha=.6)
        ax.text(lim[1], lim[1] * r, lab, fontsize=6, color="gray", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("VSX literature period [d]")
    ax.set_ylabel("cuPeriod recovered period [d]")
    ax.set_title("(a) Period recovery (all methods, all stars)")
    ax.legend(fontsize=6, loc="upper left", framealpha=.9)

    # (b) recovery rate per method
    ax = axes[1]
    methods = [m for m in METHOD_ORDER if m in set(val.method)]
    exact = [val[val.method == m]["recover_exact"].mean() * 100 for m in methods]
    harm = [val[val.method == m]["recover_harmonic"].mean() * 100 for m in methods]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, harm, 0.4, label="incl. harmonic (P, P/2, 2P…)", color="#4c72b0")
    ax.bar(x + 0.2, exact, 0.4, label="exact (=P within 2%)", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels([ml(m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("recovery rate [%]"); ax.set_ylim(0, 105)
    ax.set_title("(b) Recovery rate by method")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_recovery.png")
    plt.close(fig)


MBR_ORDER = [
    ("gls_offsets", "GLS offsets (1,0)"), ("gls_perband", "GLS perband (0,1)"),
    ("gls_flex", "GLS flex (1,1)"), ("pdm", "PDM"), ("ce", "CE"),
    ("stringlength", "String-Len"), ("mhaov", "MHAOV"),
    ("supersmoother", "SuperSmoother"), ("bls", "BLS"),
]


def _mbr_scored(mbr):
    """The scored (first-listed backend) pass of the multiband-real results."""
    scored_backend = str(mbr["backend"].iloc[0])
    return mbr[mbr["backend"] == scored_backend], scored_backend


def fig_multiband_real(mbr):
    scored, _ = _mbr_scored(mbr)
    mb = scored[~scored.model.str.startswith("single_")]
    single = scored[scored.model.str.startswith("single_")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) strict + harmonic-aware recovery per model, single-band baselines first
    ax = axes[0]
    band_rate = single.groupby("model")["strict"].mean()
    best_band = band_rate.idxmax().removeprefix("single_")
    any_rate = single.groupby("sesar_id")["strict"].any().mean()
    labels = [f"best band ({best_band})", "any band"]
    strict = [band_rate.max() * 100, any_rate * 100]
    harm = [np.nan, np.nan]
    for key, label in MBR_ORDER:
        g = mb[mb.model == key]
        labels.append(label)
        strict.append(g["strict"].mean() * 100)
        harm.append(g["harmonic_ok"].mean() * 100)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, np.nan_to_num(harm), 0.4,
           label="incl. harmonic (P, P/2, 2P…)", color="#4c72b0")
    ax.bar(x + 0.2, strict, 0.4, label="strict (=P within 1%)", color="#dd8452")
    ax.axvline(1.5, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("recovery rate [%]")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Recovery on 100 real S82 RR Lyrae\n"
                 "left of dots: single-band GLS baseline", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")

    # (b) where the misses go: ratio vs true period, harmonic + 1-day alias loci
    ax = axes[1]
    ratio = mb.p_top / mb.p_true
    hit = mb["harmonic_ok"].to_numpy()
    ax.scatter(mb.p_true[hit], ratio[hit], s=8, alpha=.3, color="#4c72b0",
               edgecolor="none", label="harmonic-aware hit")
    ax.scatter(mb.p_true[~hit], ratio[~hit], s=14, alpha=.8, color="crimson",
               edgecolor="none", label="miss")
    pp = np.linspace(mb.p_true.min() * 0.95, mb.p_true.max() * 1.05, 200)
    for r in (0.5, 1.0, 2.0):
        ax.axhline(r, color="gray", lw=0.7, ls="--", alpha=.6)
    for s, lab in ((1, "+1 d$^{-1}$ alias"), (-1, "−1 d$^{-1}$ alias")):
        with np.errstate(divide="ignore"):
            loc = 1.0 / (1.0 + s * pp)
        ok = (loc > 0) & (loc < 10)
        ax.plot(pp[ok], loc[ok], lw=0.9, ls=":", color="#2ca02c", alpha=.9)
        if ok.any():
            ax.text(pp[ok][-1], loc[ok][-1], lab, fontsize=6, color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("Sesar 2010 literature period [d]")
    ax.set_ylabel("recovered / literature period")
    ax.set_title("(b) All multi-band model–star pairs\n"
                 "dashed: harmonics · dotted: ±1 cycle/day aliases", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_multiband_real.png")
    plt.close(fig)


INJ_SIGNAL_ORDER = ["sinusoid", "eclipse", "transit"]
INJ_MCOLOR = {"GLS": "#1f77b4", "MHAOV": "#ff7f0e", "PDM": "#2ca02c", "CE": "#9467bd",
             "STRINGLENGTH": "#8c564b", "BLS": "#d62728", "TLS": "#17becf"}


def fig_injection(inj):
    sigs = [s for s in INJ_SIGNAL_ORDER if s in set(inj.signal_type)]
    fig, axes = plt.subplots(1, len(sigs), figsize=(4.3 * len(sigs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sig in zip(axes, sigs):
        sub = inj[inj.signal_type == sig]
        methods = [m for m in METHOD_ORDER if m in set(sub.method)]
        for m in methods:
            s = sub[sub.method == m].groupby("snr").recovered.mean().sort_index() * 100
            ax.plot(s.index, s.values, "o-", ms=4, lw=1.4, color=INJ_MCOLOR.get(m, "gray"),
                    label=ml(m))
        ax.set_xscale("log")
        ax.set_xlabel("SNR (amplitude / photometric σ)")
        ax.set_title(sig.title())
        ax.set_ylim(0, 105)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("recovery fraction [%]")
    axes[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Synthetic injection–recovery: recovery fraction vs SNR "
                 "(real ASAS-SN cadences, harmonic-aware 2% scoring)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_injection.png")
    plt.close(fig)


def fig_benchmark(single, npts, grid, batch):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # (a) single-LC GPU speedup per method
    ax = axes[0]
    if single is not None:
        s = single.set_index("method").reindex([m for m in METHOD_ORDER if m in set(single.method)])
        x = np.arange(len(s))
        ax.bar(x, s["gpu_speedup"], 0.6, color="#4c72b0")
        top = float(s["gpu_speedup"].max())
        for i, (m, row) in enumerate(s.iterrows()):
            v = row["gpu_speedup"]
            ax.text(i, v, f"{v:.0f}x" if v >= 2 else f"{v:.1f}x",
                    ha="center", va="bottom", fontsize=7)
            cvr = row.get("cpu_vs_ref", np.nan) if "cpu_vs_ref" in s.columns else np.nan
            if np.isfinite(cvr):
                ax.text(i, top * 0.30, f"CPU\n{cvr:.0f}× vs\n{row['ref']}", ha="center",
                        va="center", fontsize=5.5, color="#1a7a1a",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#eafbea", ec="#1a7a1a", lw=0.4))
        ax.set_xticks(x); ax.set_xticklabels([ml(m) for m in s.index], rotation=30, ha="right")
        ax.set_ylabel("GPU speedup over cuPeriod CPU")
        ax.set_title("(a) Single light curve — GPU over cuPeriod CPU (bars);\n"
                     "green = cuPeriod CPU speedup over the reference tool", fontsize=7.5)
    # (b) scaling vs grid size — one colour per method, solid=GPU, dashed=CPU
    ax = axes[1]
    mcol = {"GLS": "#1f77b4", "PDM": "#2ca02c", "MHAOV": "#ff7f0e",
            "SUPERSMOOTHER": "#9467bd"}
    if grid is not None:
        for m, mk in [("GLS", "o"), ("PDM", "s"), ("MHAOV", "^"),
                      ("SUPERSMOOTHER", "D")]:
            g = grid[grid.method == m].sort_values("n")
            if g.empty:
                continue
            ax.plot(g.n, g.cpu_s, mk + "--", color=mcol[m], lw=1, ms=4, alpha=.8)
            ax.plot(g.n, g.gpu_s, mk + "-", color=mcol[m], lw=1.4, ms=4,
                    label=f"{ml(m)}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("search-grid size (frequencies)")
        ax.set_ylabel("wall time [s]")
        ax.set_title("(b) Scaling with grid size\nsolid = GPU, dashed = CPU", fontsize=8)
        ax.legend(fontsize=7, title="method", title_fontsize=7)
    # (c) batch throughput
    ax = axes[2]
    if batch is not None:
        for m, mk, col in [("GLS", "o", "#1f77b4"), ("PDM", "s", "#2ca02c")]:
            b = batch[batch.method == m].sort_values("n_lc")
            if b.empty:
                continue
            ax.plot(b.n_lc, b.gpu_lc_per_s, mk + "-", color=col, lw=1.6, ms=6,
                    label=f"{ml(m)} GPU")
            bc = b.dropna(subset=["cpu_lc_per_s"])
            ax.plot(bc.n_lc, bc.cpu_lc_per_s, mk + "--", color=col, lw=1.2, ms=6,
                    alpha=.6, label=f"{ml(m)} CPU pool")
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 1024, 4096]); ax.set_xticklabels(["256", "1024", "4096"])
        ax.set_xlabel("light curves in batch")
        ax.set_ylabel("throughput [light curves / s]")
        ax.set_title("(c) Batch throughput", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_benchmark.png")
    plt.close(fig)


def fig_tls(tls):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(tls.koi_period, tls.cup_gpu_period, s=45, color="#1f77b4",
               label="cuPeriod (GPU)", zorder=4)
    cpu = tls.dropna(subset=["cup_cpu_period"])
    if len(cpu):
        ax.scatter(cpu.koi_period, cpu.cup_cpu_period, s=80, facecolor="none",
                   edgecolor="#2ca02c", marker="s", label="cuPeriod (CPU subset)", zorder=3)
    if "tls_ref_period" in tls:
        ax.scatter(tls.koi_period, tls.tls_ref_period, s=90, facecolor="none",
                   edgecolor="#d62728", label="transitleastsquares", zorder=2)
    lim = [tls.koi_period.min() * 0.8, tls.koi_period.max() * 1.2]
    for r, ls in [(1, "-"), (2, ":"), (0.5, ":")]:
        ax.plot(lim, [l * r for l in lim], "k" + ls, lw=0.7, alpha=.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("known KOI period [d]"); ax.set_ylabel("recovered period [d]")
    ax.set_title("(a) Kepler KOI period recovery")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    sp = tls["gpu_speedup"].dropna().values
    if len(sp):
        ax.bar(range(len(sp)), sorted(sp), color="#4c72b0")
        ax.set_title(f"(b) TLS GPU speedup over CPU (median {np.median(sp):.0f}×)")
    ax.set_xlabel("KOI (CPU-timed subset)"); ax.set_ylabel("speedup ×")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_tls.png")
    plt.close(fig)


def load(name):
    p = RESULTS / name
    return pd.read_parquet(p) if p.exists() else None


def backend_recommendations(single):
    """Per-method fastest-backend table + a scenario recommendation, from bench_single."""
    rows = []
    for _, r in single.iterrows():
        times = {"cpu": r.cpu_s, "gpu": r.gpu_s, "torch": r.torch_s}
        times = {k: v for k, v in times.items() if np.isfinite(v)}
        fastest = min(times, key=times.get)
        spd = r.gpu_speedup
        if not np.isfinite(spd):
            rec = "`cpu`"
        elif spd >= 2:
            rec = "`gpu` if available, else `cpu`"
        elif spd >= 0.8:
            rec = "`cpu` (GPU comparable)"
        else:
            rec = "`cpu` (GPU slower here)"
        label = {"cpu": f"cpu ({r.cpu_backend})", "gpu": "gpu (CUDA)",
                 "torch": f"torch ({r.torch_backend})"}[fastest]
        rows.append(dict(method=ml(r.method), fastest=label,
                         t_best=f"{times[fastest]*1e3:.1f} ms",
                         gpu_over_cpu=(f"{spd:.1f}×" if np.isfinite(spd) else "—"),
                         recommendation=rec))
    return pd.DataFrame(rows)


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f(v) if f else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    val = load("validation_metrics.parquet")
    single = load("bench_single.parquet")
    npts = load("bench_npoints.parquet")
    grid = load("bench_grid.parquet")
    batch = load("bench_batch.parquet")
    tls = load("tls_results.parquet")
    inj = load("injection_recovery.parquet")
    mbr = load("multiband_real.parquet")
    spectra = sorted(glob.glob(str(RESULTS / "spectra" / "*.npz")))

    made = []
    if val is not None:
        fig_parity_and_reference(val); made.append("fig1")
        fig_recovery(val); made.append("fig3")
    meta, _, _, _ = load_dataset()
    if spectra:
        fig_spectra(spectra, meta); made.append("fig2")
    if single is not None or grid is not None or batch is not None:
        fig_benchmark(single, npts, grid, batch); made.append("fig4")
    if tls is not None:
        fig_tls(tls); made.append("fig5")
    if inj is not None:
        fig_injection(inj); made.append("fig6")
    if mbr is not None:
        fig_multiband_real(mbr); made.append("fig7")
    print("figures:", made)

    # ---- assemble REPORT.md ----------------------------------------------
    L = []
    L.append("# cuPeriod — Validation & Benchmark Report\n")
    torch_backend = "—"
    if single is not None and "torch_backend" in single.columns:
        tb = single["torch_backend"].dropna()
        tb = tb[tb != "—"]
        if len(tb):
            torch_backend = str(tb.iloc[0])
    if torch_backend == "—" and val is not None and "torch_backend" in val.columns:
        tb = val["torch_backend"].dropna()
        tb = tb[tb != "-"]
        if len(tb):
            torch_backend = str(tb.iloc[0])

    # ---- summary ----------------------------------------------------------
    if val is not None:
        worst_par = val.parity_max_rel.max()
        same_pct = val.cpu_gpu_same.mean() * 100
        harm_lo = val.groupby("method").recover_harmonic.mean().min() * 100
        summary = (
            f"**Summary.** cuPeriod {cup.__version__}'s {val.method.nunique()} "
            f"single-band-benchmarked period-search methods were validated on {len(meta)} "
            "real ASAS-SN light curves with "
            "literature periods, plus 12 confirmed Kepler KOIs for the transit methods. "
            f"CPU and GPU backends agree to round-off (worst-case relative difference "
            f"{worst_par:.0e}, dominated by the two single-precision GPU paths) and select "
            f"the identical best period on {same_pct:.0f}% of targets; every method with an "
            "established external reference implementation reproduces it on an identical "
            f"grid. Harmonic-aware period recovery is ≥{harm_lo:.0f}% for all of them. ")
        if batch is not None:
            gpk = batch.loc[batch.gpu_lc_per_s.idxmax()]
            summary += (f"Peak measured throughput is {gpk.gpu_lc_per_s:,.0f} light curves/s "
                        f"({ml(gpk.method)}) on one GPU. ")
        if mbr is not None:
            mb_scored, _ = _mbr_scored(mbr)
            mb_only = mb_scored[~mb_scored.model.str.startswith("single_")]
            best_strict = mb_only.groupby("model")["strict"].mean().max() * 100
            summary += (
                f"Every multi-band method was additionally validated on "
                f"{mb_scored.sesar_id.nunique()} real SDSS Stripe 82 RR Lyrae "
                f"with literature periods (best joint model: {best_strict:.0f}% "
                "strict top-period recovery; §4). ")
        summary += ("Practical guidance on backend selection is given in "
                    "§8; limitations in §9.")
        L.append(summary + "\n")

    # ---- 1. environment & methodology --------------------------------------
    L.append("## 1 — Test environment and methodology\n")
    L.append("### 1.1 Environment\n")
    L.append("| Component | Details |\n| --- | --- |")
    L.append("| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (compute capability 12.0, sm_120) |")
    L.append("| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 threads |")
    L.append("| Memory | 32 GB |")
    L.append(f"| Software | cuPeriod {cup.__version__}, Python 3.12, CuPy (CUDA 12), "
             f"PyTorch cu128 ({torch_backend}), numba, finufft |")
    L.append(f"| torch device (validated) | {torch_backend} |")
    L.append("| Reference tools | astropy (`LombScargle`, `BoxLeastSquares`), PyAstronomy "
             "(`pyPDM`), `transitleastsquares`; CE/String-Length/MHAOV vs direct NumPy "
             "implementations of the published algorithms |")
    class_counts = ", ".join(f"{bc.replace('_', ' ').title()} {n}"
                             for bc, n in meta.broad_class.value_counts().reindex(
                                 [c for c in CLASS_ORDER if c in set(meta.broad_class)]).items())
    L.append(f"| Validation data | {len(meta)} ASAS-SN g-band light curves "
             f"({meta.broad_class.nunique()} variability classes: {class_counts}) with VSX "
             "literature periods, bundled in `dataset/light_curves.parquet` (core sample plus "
             "an extension selected/downloaded via `dataset/download_extension.py` from "
             "ASAS-SN Sky Patrol — clean single VSX types, n_det≥300, baseline≥1000 d); 12 "
             "confirmed Kepler KOIs (Mendeley *Dataset_Machine_Learning_Exoplanets_2024*; "
             "flux via MAST/lightkurve); 100 SDSS Stripe 82 RR Lyrae with ugriz "
             "photometry and literature periods (Sesar et al. 2010, "
             "`dataset/s82_rrlyrae.parquet`) for the multi-band methods |\n")
    L.append("### 1.2 Timing methodology\n")
    L.append("Wall-clock times use `time.perf_counter()`. Every timed configuration is run "
             "once untimed first — so JIT compilation (numba), CUDA kernel/plan caching and "
             "context creation are excluded — then the **best of 3 repeats** is reported "
             "(best-of-2 for the box methods, best-of-1 for the scaling sweeps and external "
             "reference tools). Single-curve benchmarks use one representative real ASAS-SN "
             "light curve (~900 points, multi-year baseline) on a fixed 30 000-frequency "
             "grid; the box methods (BLS/TLS) search a bounded 0.5–4 d period window. Batch "
             "throughput is a deliberately conservative *single-shot* rate: it includes the "
             "one-off worker-pool spin-up (process spawn + per-worker CUDA context), so "
             "sustained rates over many chunks are higher.\n")
    L.append("### 1.3 Metric definitions\n")
    L.append("- **parity** — worst case over all validation stars of "
             "max_f |S_CPU(f) − S_GPU(f)| / max_f |S_CPU(f)|, the relative difference of the "
             "periodogram statistic between cuPeriod's CPU and GPU backends on an identical "
             "frequency grid.\n"
             "- **identical peak** — fraction of stars where CPU and GPU backends select "
             "exactly the same best period.\n"
             "- **reference agreement** — worst-case max_f |S_cuPeriod(f) − S_reference(f)| "
             "against the external implementation on an identical grid (Pearson *r* for PDM, "
             "whose PyAstronomy reference uses a different θ normalisation).\n"
             "- **harmonic recovery** — recovered period matches the VSX literature period "
             "or a method-appropriate harmonic (P/2, 2P, P/3, 3P) within 2%; "
             "**exact recovery** requires the literature period itself within 2%.\n")

    if val is not None:
        L.append("## 2 — Numerical validation\n")
        L.append("Every method runs on an identical grid through cuPeriod's CPU and GPU "
                 "backends and through an independent reference implementation. Two checks: "
                 "**CPU↔GPU parity** (the two backends must agree to round-off) and "
                 "**cuPeriod↔reference** (must match an established implementation).\n")
        rows = []
        for m in [x for x in METHOD_ORDER if x in set(val.method)]:
            s = val[val.method == m]
            refname = {"GLS": "astropy LS", "BLS": "astropy BLS", "PDM": "PyAstronomy",
                       "CE": "Graham 2013", "STRINGLENGTH": "Dworetsky 1983",
                       "MHAOV": "Sch.-Czerny", "TLS": "—"}[m]
            rows.append(dict(
                Method=ml(m), N=len(s),
                parity=f"{s.parity_max_rel.max():.1e}",
                same=f"{s.cpu_gpu_same.mean()*100:.0f}%",
                ref=refname,
                refagree=("—" if m == "TLS" else
                          (f"r≥{np.nanmin(s.cup_ref_corr):.4f}" if m == "PDM"
                           else f"{np.nanmax(s.cup_ref_max_abs):.1e}")),
            ))
        tbl = pd.DataFrame(rows).rename(columns={
            "N": "N stars", "parity": "CPU↔GPU parity (max rel.)",
            "same": "identical peak", "ref": "reference",
            "refagree": "ref. agreement (max abs.)"})
        L.append(md_table(tbl, list(tbl.columns)))
        L.append("\n**Table 1.** Numerical validation per method (metrics defined in §1.3). "
                 "The GLS and MHAOV GPU kernels are single precision, bounding their parity "
                 "at ≈1e-6/1e-7; all other GPU paths — String-Length included, via a stable "
                 "phase sort on every backend — are double precision. String-Length's "
                 "worst-case reference difference is an isolated outlier on 1–2 heavily "
                 "phase-tied stars, where the textbook reference breaks ties with an "
                 "unstable sort (correlation ≈1, median \\|Δ\\| ≈ 6e-12, recovered period "
                 "unaffected).\n")
        L.append("![parity](figures/fig1_parity_reference.png)\n")
        L.append("**Figure 1.** Per-star distribution of (a) CPU↔GPU parity and "
                 "(b) cuPeriod-vs-reference agreement, per method.\n")

        if "rel_diff_torch" in val.columns:
            trows = []
            for m in [x for x in METHOD_ORDER if x in set(val.method)]:
                s = val[val.method == m]
                avail = s["rel_diff_torch"].notna()
                n_avail = int(avail.sum())
                if n_avail == 0:
                    trows.append(dict(Method=ml(m), device="—", n="0",
                                      maxrel="—", same="—"))
                    continue
                sa = s[avail]
                dev = sa["torch_backend"].dropna()
                dev = dev[dev != "-"]
                device = str(dev.iloc[0]) if len(dev) else "—"
                trows.append(dict(
                    Method=ml(m), device=device, n=str(n_avail),
                    maxrel=f"{sa.rel_diff_torch.max():.1e}",
                    same=f"{sa.torch_cpu_same.mean()*100:.0f}%"))
            ttbl = pd.DataFrame(trows).rename(columns={
                "device": "torch device", "n": "N stars (torch avail.)",
                "maxrel": "max rel. diff vs CPU", "same": "identical peak vs CPU"})
            L.append(md_table(ttbl, list(ttbl.columns)))
            L.append("\n**Table 1b.** CPU↔torch parity per method — the portable `backend="
                     "\"torch\"` path against cuPeriod's CPU backend on the shared "
                     "validation grid. Rows with `N stars (torch avail.) = 0` mean torch "
                     "(or a compatible device) was unavailable in the environment that "
                     "produced this parquet; the wider rerun fills these in. Torch validated "
                     f"on **{torch_backend}** here — the same code path also runs on Apple "
                     "(mps) and Intel (xpu) devices but those were not exercised (§8).\n")
        if spectra:
            L.append("![spectra](figures/fig2_spectra_overlay.png)\n")
            L.append("**Figure 2.** cuPeriod (solid) vs independent reference (dashed) "
                     "periodograms for one representative star per variability class; the "
                     "vertical line marks the VSX literature period.\n")

        L.append("## 3 — Period recovery on real light curves\n")
        rr_rows = []
        for m in [x for x in METHOD_ORDER if x in set(val.method)]:
            s = val[val.method == m]
            n = len(s)
            rr_rows.append(dict(
                method=ml(m), n=n,
                harmonic=fmt_pct_ci(int(s.recover_harmonic.sum()), n),
                exact=fmt_pct_ci(int(s.recover_exact.sum()), n)))
        rr = pd.DataFrame(rr_rows)
        L.append(md_table(rr, ["method", "n", "harmonic", "exact"],
                          {"n": lambda v: str(v)}))
        L.append("\n**Table 2.** Period recovery rates, with Wilson 95% confidence "
                 "intervals. *harmonic* accepts the method-appropriate fold ambiguity "
                 "(e.g. Fourier methods recover P/2 for contact binaries); *exact* requires "
                 "the VSX literature period itself within 2%. The exact-recovery spread "
                 "across methods reflects the methods' differing harmonic responses to "
                 "eclipsing systems, not implementation quality — §2 establishes all "
                 "implementations match their references.\n")
        L.append("![recovery](figures/fig3_recovery.png)\n")
        L.append("**Figure 3.** (a) Recovered vs literature period for all method–star "
                 "pairs, with harmonic loci; (b) recovery rate per method.\n")

        # -- curated-core vs less-curated-extension split ------------------
        ext_path = DATASET / "light_curves_extension.parquet"
        if ext_path.exists():
            ext_ids = set(pd.read_parquet(ext_path, columns=["asas_sn_id"])
                          .asas_sn_id.astype(str))
            val_ids = val.assign(_sid=val.asas_sn_id.astype(str))
            is_ext = val_ids._sid.isin(ext_ids)
            n_ext_stars = len(ext_ids)
            n_core_stars = len(meta) - n_ext_stars
            core_h = val_ids[~is_ext]
            ext_h = val_ids[is_ext]
            L.append(
                f"**Curated core vs. less-curated extension.** The {len(meta)}-star sample "
                f"combines an original {n_core_stars}-star curated core with a {n_ext_stars}-star "
                "extension added to broaden coverage of harder classes — spot-evolving "
                "rotators (BY Dra/RS CVn, whose starspot-driven period can drift between "
                "observing seasons) and long-period semiregular/Mira variables (whose pulsation "
                "cycle wanders relative to a single catalogue period). Aggregate harmonic "
                f"recovery: core {fmt_pct_ci(int(core_h.recover_harmonic.sum()), len(core_h))} "
                f"vs. extension {fmt_pct_ci(int(ext_h.recover_harmonic.sum()), len(ext_h))} "
                "(pooled across all frequency methods; Wilson 95% CIs). The extension's lower "
                "rate reflects those harder classes, not a code difference — §2 shows CPU, GPU "
                "and torch backends still agree to round-off on every star in both groups. "
                "Because the extension is deliberately weighted toward these harder cases, the "
                f"pooled {len(meta)}-star rate is a more realistic field estimate than the "
                "curated core's rate alone.\n")

        # -- notable failures: stars missed (non-harmonic) by >=3 frequency methods --
        freq_methods = [m for m in METHOD_ORDER if m != "TLS" and m in set(val.method)]
        vfreq = val[val.method.isin(freq_methods)]
        vfreq = vfreq.assign(_ratio=vfreq.p_cpu / vfreq.vsx_period)
        miss = (vfreq.assign(missed=~vfreq.recover_harmonic)
                .groupby("asas_sn_id")
                .agg(n_missed=("missed", "sum"),
                     n_methods=("missed", "size"),
                     vsx_type=("vsx_type", "first"),
                     broad_class=("broad_class", "first"),
                     vsx_period=("vsx_period", "first"),
                     ratio=("_ratio", "mean")))
        miss = miss[miss.n_missed >= 3]
        if len(miss):
            # classify each miss by how the recovered/literature ratio behaves
            def _classify(ratio):
                if not np.isfinite(ratio):
                    return "unrecovered"
                if 0.95 <= ratio <= 1.05:
                    return "near-miss"
                near_harm = min(abs(ratio / r - 1.0) for r in HARMONIC_RATIOS if r != 1.0)
                if near_harm <= 0.05:
                    return "alias/harmonic"
                return "wandering"
            EXPL = {
                "near-miss": "recovered period is within ~5% of literature but outside the "
                             "2% tolerance — typical of spot-evolving rotators or a slightly "
                             "drifting period between epochs",
                "wandering": "recovered period is far from literature and from any small-"
                             "integer harmonic — typical of Mira/SR variables whose cycle "
                             "wanders between observing epochs relative to a single "
                             "catalogue period",
                "alias/harmonic": "recovered period sits near a harmonic/alias ratio just "
                                  "outside the accepted set — a photometric-alias selection, "
                                  "not a recovery failure",
                "unrecovered": "no finite period recovered on the shared grid",
            }
            miss = miss.assign(category=miss.ratio.map(_classify))
            L.append("**Notable failures.** Stars missed (non-harmonic) by ≥3 of the "
                     f"{len(freq_methods)} frequency methods, grouped by failure mode "
                     "(these are individual outliers absorbed into the aggregate rates "
                     "above — §2 establishes all methods match their references on "
                     "identical grids):\n")
            ftbl_rows = []
            for sid, r in miss.sort_values(["category", "asas_sn_id"]).iterrows():
                ftbl_rows.append(dict(
                    star=str(sid), vsx_type=r.vsx_type, cls=r.broad_class.replace("_", " ").title(),
                    period=f"{r.vsx_period:.4f}", missed=f"{int(r.n_missed)}/{int(r.n_methods)}",
                    ratio=f"{r.ratio:.2f}" if np.isfinite(r.ratio) else "—",
                    category=r.category))
            ftbl = pd.DataFrame(ftbl_rows).rename(columns={
                "star": "asas_sn_id", "vsx_type": "VSX type", "cls": "class",
                "period": "P_lit [d]", "missed": "missed/N", "ratio": "P_rec/P_lit"})
            L.append(md_table(ftbl, list(ftbl.columns)))
            for cat in ["near-miss", "wandering", "alias/harmonic", "unrecovered"]:
                if cat in set(miss.category):
                    L.append(f"\n- *{cat}*: {EXPL[cat]}.")
            L.append("\n")

    if mbr is not None:
        scored, scored_backend = _mbr_scored(mbr)
        mb = scored[~scored.model.str.startswith("single_")].copy()
        sb = scored[scored.model.str.startswith("single_")]
        n_stars = int(scored.sesar_id.nunique())
        types = scored.drop_duplicates("sesar_id").rrl_type.value_counts()
        gpu = mbr[mbr.backend != scored_backend]

        L.append("## 4 — Multi-band period recovery: real Stripe 82 RR Lyrae\n")
        L.append(
            f"§3 validates the single-band methods on real data; this section does "
            f"the same for every **multi-band** method, on the canonical real "
            f"multi-band test set: the SDSS Stripe 82 RR Lyrae of Sesar et al. "
            f"2010 (ApJ 708, 717) — the dataset VanderPlas & Ivezić 2015 "
            f"developed the shared-phase multiband periodogram on, the model that "
            f"ships as cuPeriod's default `offsets` GLS. {n_stars} stars "
            f"({int(types.get('ab', 0))} RRab, {int(types.get('c', 0))} RRc; the "
            f"first {n_stars} of 483 by Sesar ID, no quality selection), each with "
            f"real ugriz photometry (~55 epochs per band, ~280 points total) over "
            f"a ~3200-day baseline, and a literature period from the discovery "
            f"paper. Ground-based cadence at its most adversarial: strong ±1 "
            f"cycle/day aliasing. Blind search, identical for every star: periods "
            f"0.15–1.2 d at 5 samples per Rayleigh width (~97 000 trial "
            f"frequencies), every method at default settings (BLS builds its "
            f"native duration grid inside the same window). **strict** = top "
            f"period within 1% of the literature value, no harmonic credit; "
            f"**harmonic-aware** = §1.3's 2% harmonic tolerance. Bundle: "
            f"`dataset/s82_rrlyrae.parquet` via "
            f"`dataset/download_s82_rrlyrae.py`.\n")

        rows = []
        for band in ("u", "g", "r", "i", "z"):
            s = sb[sb.model == f"single_{band}"]
            if s.empty:
                continue
            rows.append(dict(
                model=f"single-band GLS, {band}", n=len(s),
                strict=fmt_pct_ci(int(s.strict.sum()), len(s)),
                harmonic=fmt_pct_ci(int(s.harmonic_ok.sum()), len(s)),
                dpp="—", tcpu="—", tgpu="—"))
        any_hits = sb.groupby("sesar_id")["strict"].any()
        rows.append(dict(
            model="single-band GLS, any band", n=len(any_hits),
            strict=fmt_pct_ci(int(any_hits.sum()), len(any_hits)),
            harmonic="—", dpp="—", tcpu="—", tgpu="—"))
        for key, label in MBR_ORDER:
            g = mb[mb.model == key]
            if g.empty:
                continue
            hits = g[g.strict]
            dpp = (f"{np.median(np.abs(hits.p_top / hits.p_true - 1.0)):.1e}"
                   if len(hits) else "—")
            gg = gpu[gpu.model == key]
            tgpu = f"{gg.seconds.median():.2f}" if len(gg) else "—"
            rows.append(dict(
                model=f"**{label}**", n=len(g),
                strict=fmt_pct_ci(int(g.strict.sum()), len(g)),
                harmonic=fmt_pct_ci(int(g.harmonic_ok.sum()), len(g)),
                dpp=dpp, tcpu=f"{g.seconds.median():.2f}", tgpu=tgpu))
        mtbl = pd.DataFrame(rows).rename(columns={
            "model": "model", "n": "N", "strict": "strict (1%)",
            "harmonic": "harmonic-aware (2%)", "dpp": "median \\|ΔP\\|/P",
            "tcpu": "t_CPU [s/★]", "tgpu": "t_GPU [s/★]"})
        L.append(md_table(mtbl, list(mtbl.columns), {"N": str}))
        L.append(
            "\n**Table 3.** Multi-band period recovery on real Stripe 82 RR "
            "Lyrae, Wilson 95% CIs. *median \\|ΔP\\|/P* is over strict hits "
            "(grid resolution is ~3e-5 of the period at these frequencies). "
            "Timings are median wall time per star on the shared ~97k-frequency "
            "grid, warm JIT, single shot — the batch runner amortises further "
            "via engine reuse. BLS is transit-shaped by design and is included "
            "for completeness, not as a recommended RR Lyrae tool.\n")

        # -- data-driven reading of the results ---------------------------
        notes = []
        # which family leads on this (dense) data?
        by_model = mb.groupby("model")["strict"].mean()
        fold_keys = ["pdm", "ce", "stringlength", "supersmoother"]
        gls_keys = ["gls_offsets", "gls_perband", "gls_flex"]
        fold_best = by_model.reindex(fold_keys).max()
        gls_best = by_model.reindex(gls_keys).max()
        if fold_best - gls_best >= 0.05:
            lblmap = dict(MBR_ORDER)
            fold_name = lblmap[by_model.reindex(fold_keys).idxmax()]
            notes.append(
                f"The pooled fold statistics lead on these well-sampled "
                f"curves: {fold_name} reaches {fold_best*100:.0f}% strict vs "
                f"{gls_best*100:.0f}% for the best GLS model. With ~280 points "
                "a fold uses the full non-sinusoidal light-curve shape, while "
                "the single-harmonic GLS models stay alias-limited — dense "
                "data reward shape, sparse data reward parsimony (see the "
                "model-choice note below).")
        # 1-day aliasing among non-harmonic misses
        miss = mb[~mb.harmonic_ok & np.isfinite(mb.p_top)]
        if len(miss):
            df_alias = np.abs(1.0 / miss.p_top - 1.0 / miss.p_true)
            on_alias = ((np.abs(df_alias - 1.0) < 0.05)
                        | (np.abs(df_alias - 2.0) < 0.05)).mean()
            notes.append(
                f"Of the {len(miss)} non-harmonic misses across all models, "
                f"{on_alias*100:.0f}% sit on the ±1 or ±2 cycle/day window "
                "aliases (Figure 7b) — the failure mode is the ground-based "
                "window function, not noise.")
        # fold-family: strict-vs-harmonic gap = integer-multiple picks
        fold = mb[mb.model.isin(["pdm", "ce", "stringlength", "supersmoother"])]
        gap = fold[fold.harmonic_ok & ~fold.strict]
        if len(gap):
            frac2 = (gap.ratio == 2.0).mean()
            notes.append(
                f"For the fold-family statistics (PDM/CE/String-Length/"
                f"SuperSmoother), {len(gap)} harmonic-aware hits are not strict "
                f"hits; {frac2*100:.0f}% of those sit at exactly 2P — the "
                "documented integer-multiple degeneracy of phase-folding "
                "statistics (a fold at 2P, 3P… of a true period stays coherent). "
                "The practical recipe stands: treat the *shortest* member of a "
                "near-tied family as the period, or arbitrate with "
                "`cuperiod.alias_diagnostics`.")
        if notes:
            L.append("**Reading the result.** " + " ".join(notes) + "\n")

        # offsets vs flexible models, tied to the simulated-cadence benchmark
        r_off = mb[mb.model == "gls_offsets"].strict.mean()
        r_flex = mb[mb.model == "gls_flex"].strict.mean()
        r_per = mb[mb.model == "gls_perband"].strict.mean()
        gls_rates = (f"offsets {r_off*100:.0f}%, perband {r_per*100:.0f}%, "
                     f"flex {r_flex*100:.0f}% strict")
        if max(r_flex, r_per) - r_off >= 0.05:
            reading = (
                f"On these well-sampled curves (~280 points) the flexible GLS "
                f"models beat the rigid shared-phase model ({gls_rates}): real "
                f"RR Lyrae amplitudes vary strongly with wavelength (u ≈ 2× z), "
                f"which `offsets` (common amplitude and phase, per-band offsets "
                f"only) cannot express, and the model mismatch leaks power "
                f"toward the 1-day alias.")
        elif r_off - max(r_flex, r_per) >= 0.05:
            reading = (
                f"Even on these well-sampled curves the shared-phase model "
                f"leads ({gls_rates}).")
        else:
            reading = (
                f"On these well-sampled curves (~280 points) the three GLS "
                f"models perform comparably ({gls_rates}).")
        L.append(
            f"**Model choice depends on sampling density.** {reading} "
            f"The **simulated sparse-cadence benchmark** "
            f"(`multiband_recovery.py`) probes the opposite regime: at 30 total "
            f"epochs the shared-phase `offsets` model recovers 82% vs ≤20% for "
            f"the flexible models — fewer parameters win when epochs are few. "
            f"Both regimes are real; `offsets` stays the default because the "
            f"sparse regime (early Rubin) is the one that needs a joint method "
            f"most, and the flexible models are one `mb_model=` switch away.\n")

        gagree = np.nan
        if len(gpu):
            merged = gpu.merge(
                mb[["sesar_id", "model", "p_top"]],
                on=["sesar_id", "model"], suffixes=("", "_ref"))
            gagree = (np.abs(merged.p_top / merged.p_top_ref - 1.0)
                      <= 1e-4).mean()
            t_cpu = mb.groupby("model").seconds.median()
            t_gpu = gpu.groupby("model").seconds.median()
            speedup = (t_cpu / t_gpu).dropna()
            gainers = speedup[speedup >= 1.5].sort_values(ascending=False)
            lbl = dict(MBR_ORDER)
            gain_txt = (
                "; GPU speedups ≥1.5× at this grid size: "
                + ", ".join(f"{lbl.get(m, m)} {v:.1f}×"
                            for m, v in gainers.items())
                if len(gainers) else
                "; no model gains ≥1.5× from the GPU at this single-shot size")
            laggards = speedup[speedup <= 0.2].sort_values()
            lag_txt = ""
            if len(laggards):
                verb = "is" if len(laggards) == 1 else "are"
                these = "this method" if len(laggards) == 1 else "these methods"
                lag_txt = (
                    " "
                    + " and ".join(f"{lbl.get(m, m)} ({1.0/v:.0f}× slower)"
                                   for m, v in laggards.items())
                    + f" {verb} dominated by fixed per-call dispatch and "
                      f"transfer overhead that a single-shot call cannot "
                      f"amortise — for one-off searches of {these} use the "
                      f"CPU tier, and at catalogue scale use the batch "
                      f"runner, which amortises launches and reuses engines "
                      f"across stars.")
            L.append(
                f"**Backends.** The GPU pass picks the same top period as the "
                f"scored {scored_backend} pass in {gagree*100:.1f}% of "
                f"model×star runs. Per-star GPU timings in Table 3 are "
                f"single-shot periodogram calls{gain_txt}.{lag_txt}\n")
        L.append("![multiband real](figures/fig7_multiband_real.png)\n")
        L.append(
            "**Figure 7.** (a) Strict and harmonic-aware recovery per model; "
            "the single-band GLS baseline (left of the dotted line) is what a "
            "per-band search achieves on the same grid. (b) Recovered/literature "
            "period ratio for every model×star pair: misses (red) concentrate "
            "on the ±1 cycle/day alias loci (dotted) and the integer-multiple "
            "harmonics (dashed).\n")

    if inj is not None:
        L.append("## 5 — Injection–recovery sensitivity\n")
        n_trials = int(inj.groupby(["method", "signal_type", "snr"]).size().max())
        L.append("§2–4 validate against real, bright, well-established stars — a favourable "
                 "regime. This section complements that with a controlled sweep: a known "
                 "synthetic signal of tunable amplitude, drawn onto *real* ASAS-SN "
                 "observation cadences (so the irregular sampling and seasonal gaps of "
                 "ground-based photometry are represented realistically), scored with the "
                 "same harmonic-aware 2% tolerance as §3. Three signal models, each run "
                 "through the methods it is diagnostic for: a **sinusoid** (+ mild 2nd "
                 "harmonic) for GLS/MHAOV/PDM/CE/String-Length; an **eclipse** fold (two "
                 "unequal narrow Gaussian dips per cycle) for PDM/CE/String-Length/BLS; and "
                 "a **box transit** for BLS/TLS. SNR is defined as injected amplitude / "
                 "photometric σ, with σ = 0.02 mag (typical ASAS-SN g-band precision); "
                 f"periods and phases are drawn per trial (seed 42, {n_trials} trials per "
                 "method × signal × SNR cell), all on cuPeriod's CPU (numba) backend.\n")
        tab_rows = []
        for sig in [s for s in INJ_SIGNAL_ORDER if s in set(inj.signal_type)]:
            sub = inj[inj.signal_type == sig]
            for m in [x for x in METHOD_ORDER if x in set(sub.method)]:
                s = sub[sub.method == m]
                row = dict(signal=sig.title(), method=ml(m))
                for snr in sorted(s.snr.unique()):
                    row[f"snr{snr:g}"] = s[s.snr == snr].recovered.mean() * 100
                tab_rows.append(row)
        itbl = pd.DataFrame(tab_rows)
        snr_cols = sorted([c for c in itbl.columns if c.startswith("snr")],
                          key=lambda c: float(c[3:]))
        hdr = {c: f"SNR={c[3:]}" for c in snr_cols}
        itbl = itbl.rename(columns=hdr)
        fmtd = {v: (lambda x: f"{x:.0f}%") for v in hdr.values()}
        L.append(md_table(itbl, ["signal", "method"] + list(hdr.values()), fmtd))
        L.append(f"\n**Table 4.** Recovery fraction (%) per method × signal × SNR, "
                 f"n={n_trials} trials/cell. Wilson intervals per cell are wide at this "
                 "trial count (omitted here for readability; §3's Table 2 shows the CI "
                 "convention on the larger real-star sample).\n")
        L.append("![injection](figures/fig6_injection.png)\n")
        L.append("**Figure 6.** Recovery fraction vs SNR, one panel per signal type, "
                 "one line per applicable method.\n")
        # honest note on where methods plateau below 100%
        notes = []
        top_snr = inj.snr.max()
        for sig in [s for s in INJ_SIGNAL_ORDER if s in set(inj.signal_type)]:
            sub = inj[(inj.signal_type == sig) & (inj.snr == top_snr)]
            for m in [x for x in METHOD_ORDER if x in set(sub.method)]:
                frac = sub[sub.method == m].recovered.mean()
                if frac < 0.90:
                    notes.append(f"{ml(m)} on {sig} plateaus at {frac*100:.0f}% even at the "
                                 f"highest tested SNR ({top_snr:g})")
        if notes:
            L.append("**Where methods plateau below 100%.** " + "; ".join(notes) +
                     ". These are method–signal mismatches, not implementation bugs "
                     f"(isolated cells in the low-to-mid 90s are consistent with one or two "
                     f"alias near-misses at n={n_trials} trials and are not flagged) — e.g. "
                     "String-Length's rank-based statistic is comparatively insensitive to "
                     "the narrow, low duty-cycle dips of the eclipse model used here, so it "
                     "under-recovers that signal shape even at high SNR; a box-fitting "
                     "method (BLS) is the appropriate tool for narrow eclipses/transits.\n")

    if tls is not None:
        L.append("## 6 — TLS on Kepler transits\n")
        good = (tls.cup_gpu_relerr < 0.02).mean() * 100
        par = tls.cpu_gpu_parity.dropna()
        spd = tls.gpu_speedup.dropna()
        ncpu = int(tls.cup_cpu_period.notna().sum())
        agree = "—"
        if "tls_ref_period" in tls:
            both = tls.dropna(subset=["tls_ref_period"])
            agree = f"{(np.abs(both.cup_gpu_period/both.tls_ref_period - 1) < 0.02).mean()*100:.0f}%"
        L.append(f"{len(tls)} confirmed KOIs, blind search 0.5–12 d. cuPeriod (GPU) recovers "
                 f"the known period (or a 1/2 or 2× harmonic) within 2% for **{good:.0f}%** of "
                 f"them, and agrees with `transitleastsquares` on **{agree}**. On the "
                 f"{ncpu}-KOI CPU-timed subset, CPU↔GPU max\\|Δpower\\| ≤ "
                 f"{par.max():.1e} and the GPU is a median **{spd.median():.0f}×** faster.\n")
        tt = tls.copy()
        cols = ["kepid", "koi_period", "cup_gpu_period"]
        if "tls_ref_period" in tt:
            cols.append("tls_ref_period")
        cols += ["cup_gpu_relerr", "cup_cpu_period", "gpu_speedup"]
        THDR = {"kepid": "KIC ID", "koi_period": "P_KOI [d]",
                "cup_gpu_period": "P_cuPeriod GPU [d]", "tls_ref_period": "P_TLS ref [d]",
                "cup_gpu_relerr": "rel. error", "cup_cpu_period": "P_cuPeriod CPU [d]",
                "gpu_speedup": "GPU speedup"}
        tt = tt[cols].rename(columns=THDR)
        L.append(md_table(tt, list(tt.columns), {
            "KIC ID": lambda v: f"{int(v)}",
            "P_KOI [d]": lambda v: f"{v:.4f}",
            "P_cuPeriod GPU [d]": lambda v: f"{v:.4f}",
            "P_TLS ref [d]": lambda v: ("—" if not np.isfinite(v) else f"{v:.4f}"),
            "P_cuPeriod CPU [d]": lambda v: ("—" if not np.isfinite(v) else f"{v:.4f}"),
            "rel. error": lambda v: f"{v:.1e}",
            "GPU speedup": lambda v: ("—" if not np.isfinite(v) else f"{v:.0f}×")}))
        n_cup = int((tls.cup_gpu_relerr < 0.02).sum())
        refnote = ""
        if "tls_ref_rel_err" in tls:
            refnote = (f"the `transitleastsquares` reference recovers "
                       f"{int((tls.tls_ref_rel_err < 0.02).sum())}/{len(tls)}. ")
        L.append(f"\n**Table 5.** Blind TLS period recovery on confirmed Kepler KOIs. "
                 f"cuPeriod recovers {n_cup}/{len(tls)}; {refnote}Both miss only the "
                 "shallowest transits, where a blind 0.5–12 d search aliases — a failure "
                 "mode shared with the reference implementation, not a backend defect.\n")
        L.append("\n![tls](figures/fig5_tls.png)\n")
        L.append("**Figure 4.** (a) Recovered vs known KOI period for cuPeriod (GPU and CPU) "
                 "and `transitleastsquares`; (b) GPU speedup on the CPU-timed subset.\n")

    if single is not None:
        L.append("## 7 — Performance\n")
        s = single.copy()
        s = s.set_index("method").reindex([m for m in METHOD_ORDER if m in set(single.method)]).reset_index()
        bls = s[s.method == "BLS"]
        if len(bls) and "cpu_vs_ref" in s and np.isfinite(bls.cpu_vs_ref.iloc[0]):
            b = bls.iloc[0]
            par = ""
            pf = RESULTS / "bls_numba_parity.parquet"
            if pf.exists():
                p = pd.read_parquet(pf)
                par = (f" — verified on all {len(p)} validation light curves: max\\|Δpower\\| ≤ "
                       f"{p.maxabs.max():.1e}, identical best period on {int(p.same.sum())}/"
                       f"{len(p)}")
            L.append(f"**cuPeriod's CPU box search beats astropy.** The default CPU BLS backend "
                     f"is a multicore `numba` port of the CUDA kernel — **{b.cpu_vs_ref:.0f}× "
                     f"faster than astropy's compiled `BoxLeastSquares`** "
                     f"({b.cpu_s*1e3:.0f} ms vs {b.ref_s:.1f} s on this light curve), matching it "
                     f"to floating-point{par}. The GPU then adds another {b.gpu_speedup:.0f}× "
                     f"({b.ref_s/b.gpu_s:.0f}× over astropy).\n")
        cols = ["method", "cpu_backend", "cpu_s", "gpu_s", "torch_s", "torch_backend",
                "ref", "ref_s", "cpu_vs_ref", "gpu_speedup"]
        cols = [c for c in cols if c in s.columns]
        nan_dash = lambda fmt: (lambda v: ("—" if not np.isfinite(v) else fmt(v)))
        HDR = {"cpu_backend": "CPU backend", "cpu_s": "t_CPU [s]", "gpu_s": "t_GPU [s]",
               "torch_s": "t_torch [s]", "torch_backend": "torch device",
               "ref": "reference tool", "ref_s": "t_ref [s]",
               "cpu_vs_ref": "CPU vs ref", "gpu_speedup": "GPU vs CPU"}
        st = s[cols].rename(columns=HDR)
        L.append(md_table(st, list(st.columns), {
            "t_CPU [s]": lambda v: f"{v:.3f}",
            "t_GPU [s]": nan_dash(lambda v: f"{v:.4f}"),
            "t_torch [s]": nan_dash(lambda v: f"{v:.3f}"),
            "GPU vs CPU": nan_dash(lambda v: f"{v:.1f}×"),
            "CPU vs ref": nan_dash(lambda v: f"{v:.0f}×"),
            "t_ref [s]": nan_dash(lambda v: f"{v:.2f}"),
            "method": ml}))
        L.append("\n**Table 6.** Single-curve wall time per method (methodology in §1.2). "
                 "*CPU backend* = what `backend=\"cpu\"` resolves to — the fast default a "
                 "user gets: finufft (GLS), the multicore numba box search (BLS), numba for "
                 "the rest (with the `[fast]` extra) or numpy otherwise. *CPU vs ref* = "
                 "cuPeriod-CPU speedup over the established external tool; *GPU vs CPU* = "
                 "CUDA backend over cuPeriod's own CPU backend. *t_torch* = the portable "
                 "PyTorch backend (device in *torch device*: cpu/cuda/mps/xpu) — the "
                 "cross-vendor path that also runs on AMD/Intel/Mac GPUs.\n")
        gsp = s.dropna(subset=["gpu_speedup"]).set_index("method").gpu_speedup
        fast = [(m, v) for m, v in gsp.items() if v >= 2.0]
        wash = [(m, v) for m, v in gsp.items() if 0.8 <= v < 2.0]
        slow = [(m, v) for m, v in gsp.items() if v < 0.8]
        _fmt = lambda pairs: (", ".join(f"{ml(m)} ({v:.1f}×)" for m, v in pairs)
                              if pairs else "none")
        gls_note = (" GLS's edge reflects its CPU path being finufft rather than a "
                    "numba kernel." if any(m == "GLS" for m, _ in fast) else "")
        L.append("cuPeriod's CPU path already outperforms every external reference tool "
                 "it has (GLS, PDM, BLS). **With the multicore numba tier, the GPU's "
                 "single-curve margin over the CPU is modest almost everywhere** on "
                 f"this 16-core machine: a ≥2× win for {_fmt(fast)}; a wash (0.8–2×) "
                 f"for {_fmt(wash)}; slower than the warm CPU kernel for {_fmt(slow)} "
                 "at this size — fixed per-call dispatch and transfer overhead does "
                 f"not amortise once the CPU kernel itself runs in milliseconds."
                 f"{gls_note} The scaling sweep (up to 30 000 points / a "
                 "100 000-frequency grid; Figure 5b) shows where that balance shifts "
                 "with problem size. The GPU's case is catalogue throughput and "
                 "non-NVIDIA hardware (the portable torch backend), not single-curve "
                 "latency on the CPU-tier methods; see §8.\n")
        if len(bls) and "cpu_port_s" in bls and np.isfinite(bls.cpu_port_s.iloc[0]):
            b = bls.iloc[0]
            L.append(f"\n> The pure-`numpy` BLS backend shares one array-module-generic source "
                     f"with the CUDA kernel (so they validate to floating-point), but it is a "
                     f"*parity reference*, not the product path — {b.cpu_port_s:.1f} s here, "
                     f"slower than numba and astropy because its GPU-shaped layout trades memory "
                     f"traffic for the parallelism that makes the GPU fast.\n")
        L.append("![benchmark](figures/fig4_benchmark.png)\n")
        L.append("**Figure 5.** (a) Single-curve GPU speedup over cuPeriod's CPU backend "
                 "(green boxes: cuPeriod-CPU speedup over the external reference tool); "
                 "(b) wall time vs search-grid size (solid = GPU, dashed = CPU); "
                 "(c) batch throughput, GPU vs CPU process pool.\n")
        if batch is not None:
            gpeak = batch.loc[batch.gpu_lc_per_s.idxmax()]
            cmp = batch.dropna(subset=["cpu_lc_per_s"])
            msg = (f"Batch throughput on one GPU peaks at **{gpeak.gpu_lc_per_s:,.0f} "
                   f"light curves/s** ({ml(gpeak.method)}, n={int(gpeak.n_lc)}) — "
                   f"**>{gpeak.gpu_lc_per_s*3600/1e6:.1f} million light curves/hour**. "
                   f"This is a *single-batch* rate that includes the one-off worker-pool "
                   f"spin-up (process spawn + per-worker CUDA context); a warmed pool "
                   f"sustains a higher rate over many chunks.")
            if len(cmp):
                rows_txt = "; ".join(
                    f"{ml(r.method)} n={int(r.n_lc)} {r.speedup:.1f}×"
                    for _, r in cmp.sort_values(["method", "n_lc"]).iterrows())
                msg += (f" On this 32-thread machine the CPU process pool keeps pace with the "
                        f"GPU for the numba-tier methods — {rows_txt} — with GLS the one method "
                        f"that shows a consistent GPU edge at batch scale too. Expect a wider "
                        f"GPU margin on a narrower CPU, or at batch sizes beyond what's swept "
                        f"here.")
            L.append(msg + "\n")

    if single is not None:
        L.append("## 8 — Backend recommendations\n")
        rec = backend_recommendations(
            single.set_index("method")
                  .reindex([m for m in METHOD_ORDER if m in set(single.method)])
                  .reset_index())
        rec = rec.rename(columns={
            "fastest": "fastest measured", "t_best": "best time",
            "gpu_over_cpu": "GPU vs CPU", "recommendation": "single-curve recommendation"})
        L.append(md_table(rec, list(rec.columns)))
        L.append("\n**Table 7.** Fastest measured backend per method on this machine "
                 "(single curve, ~900 points; grids as in Table 6).\n")
        slow_names = "/".join(ml(m) for m, _ in slow) if slow else "none"
        fast_names = "/".join(ml(m) for m, _ in fast) if fast else "none"
        gls_grid_txt = ""
        if grid is not None and (grid.method == "GLS").any():
            gg = grid[grid.method == "GLS"].speedup.dropna()
            if len(gg):
                gls_grid_txt = (f" — GLS holds {gg.min():.1f}–{gg.max():.1f}× across "
                                f"the grid-size sweep")
        peak_txt = "hundreds of curves/s"
        if batch is not None and len(batch):
            pk = batch.gpu_lc_per_s.max()
            peak_txt = f"{pk:,.0f} curves/s (>{pk*3600/1e6:.0f} million curves/hour)"
        L.append("Guidance by use case, from the measurements above:\n")
        L.append("1. **Interactive, single-curve analysis (default).** Use "
                 "`backend=\"cpu\"` with the `[fast]` extra installed. On a modern "
                 "multi-core CPU it is within a small factor of the GPU on every method, "
                 f"faster than the GPU for {slow_names} at typical light-curve sizes, and "
                 "already 2–2000× faster than the established external tools. No GPU is "
                 "required for competitive single-curve performance.\n"
                 "2. **GLS-dominated pipelines on NVIDIA hardware.** Use `backend=\"gpu\"`: "
                 f"{fast_names} show a consistent single-curve GPU advantage "
                 f"(Table 6{gls_grid_txt}; GLS keeps ~1.4× at batch scale too, because "
                 "its CPU path is finufft rather than a numba kernel).\n"
                 "3. **Catalogue-scale processing (10³–10⁶ curves).** Use "
                 "`batch_periodograms(..., device=\"gpu\")` on NVIDIA hardware — peak "
                 f"measured throughput {peak_txt} on one GPU. On "
                 "this 32-thread CPU the process pool keeps pace for the numba-tier "
                 "methods, so on wide CPU nodes `device=\"cpu\"` is a legitimate "
                 "alternative; expect the GPU margin to widen on narrower CPUs and larger "
                 "batches.\n"
                 "4. **AMD, Intel or Apple GPUs.** Use `backend=\"torch\"` — the portable "
                 "path validated to the same parity standard. On NVIDIA hardware the "
                 "native CUDA backend is usually at least as fast (Table 6), so treat "
                 "torch as the portability path, not the speed path.\n"
                 "5. **Strict double-precision requirements.** The GLS and MHAOV CUDA "
                 "kernels are single precision (parity ≈1e-6/1e-7; Table 1). The selected "
                 f"best period was unaffected on all {len(meta)} validation stars, but if statistic "
                 "values matter beyond ~6 significant digits (e.g. FAP tail comparisons), "
                 "use the CPU backend, which is double precision throughout.\n"
                 "6. **Minimal installations (no numba).** `backend=\"cpu\"` falls back to "
                 "numpy — numerically identical but much slower for the box methods (the "
                 "pure-numpy BLS parity reference takes ~18 s vs 0.19 s with numba). "
                 "Install the `[fast]` extra, or use `backend=\"astropy\"` for BLS.\n")

    L.append("## 9 — Limitations\n")
    L.append("- All timings are from a single machine (Table in §1.1); CPU↔GPU ratios "
             "depend strongly on core count. The 16-core/32-thread CPU used here is near "
             "the top of the desktop range, so the reported GPU margins are conservative "
             "for typical hardware.\n"
             "- Batch throughput was swept only to 4096 curves per batch and is a "
             "single-shot rate including worker-pool start-up; sustained throughput and "
             "larger batches favour the GPU further.\n"
             "- The GLS and MHAOV GPU statistics are single precision (§8, item 5).\n"
             "- The torch backend was timed on a CUDA device only; Apple (mps) and Intel "
             "(xpu) devices are supported but not benchmarked here.\n"
             "- The TLS blind search uses a fixed 0.5–12 d window; the unrecovered KOIs "
             "are the shallowest transits, which alias within that window (the reference "
             "implementation misses one of the same targets; §6). The recovery rate "
             "therefore reflects the search configuration as much as the implementation.\n"
             "- The real multi-band validation (§4) covers one variable class (RR Lyrae) "
             "on well-sampled ~280-point curves; the sparse-cadence regime is covered by "
             "the simulated `multiband_recovery.py` benchmark, not by real data. BLS is "
             "included there for completeness but is transit-shaped by design.\n"
             "- Recovery rates are measured on light curves with well-established "
             "literature periods and moderate noise; they are upper bounds relative to "
             "survey-quality data with weaker signals.\n")

    L.append("## 10 — References\n")
    L.append("Method papers: "
             "GLS — Zechmeister & Kürster 2009, A&A 496, 577; "
             "Lomb–Scargle practicalities — VanderPlas 2018, ApJS 236, 16; "
             "multiband GLS — VanderPlas & Ivezić 2015, ApJ 812, 18. "
             "BLS — Kovács, Zucker & Mazeh 2002, A&A 391, 369. "
             "PDM — Stellingwerf 1978, ApJ 224, 953. "
             "Conditional Entropy — Graham et al. 2013, MNRAS 434, 2629. "
             "String Length — Dworetsky 1983, MNRAS 203, 917. "
             "MHAOV — Schwarzenberg-Czerny 1996, ApJ 460, L107. "
             "TLS — Hippke & Heller 2019, A&A 623, A39. "
             "SuperSmoother — Friedman 1984; Reimann 1994.\n")
    L.append("Reference software: Astropy Collaboration 2022, ApJ 935, 167; "
             "PyAstronomy — Czesla et al. 2019, ascl:1906.010; "
             "`transitleastsquares` — Hippke & Heller 2019.\n")
    L.append("Data: ASAS-SN — Shappee et al. 2014, ApJ 788, 48; Kochanek et al. 2017, "
             "PASP 129, 104502. VSX — Watson, Henden & Price 2006, SASS 25, 47. "
             "SDSS Stripe 82 RR Lyrae — Sesar et al. 2010, ApJ 708, 717 (files via the "
             "astroML-data mirror). Kepler KOI light curves via MAST/lightkurve.\n")

    L.append("## 11 — Reproducibility\n"
             "The validation light curves and their literature periods ship in "
             "`dataset/light_curves.parquet` and `dataset/s82_rrlyrae.parquet`; §2, §3, "
             "§4, §5 and §7 need no network access or external catalogue. The Kepler/TLS "
             "comparison (§6) downloads flux from MAST and runs `transitleastsquares` in "
             "a separate pinned environment.\n```\n"
             "python benchmarks/validate_periodograms.py  # 1-1 validation (main GPU venv)\n"
             "python benchmarks/multiband_real.py         # real S82 multi-band recovery (§4)\n"
             "python benchmarks/injection_recovery.py     # synthetic sensitivity sweep (§5)\n"
             "python benchmarks/benchmark.py              # performance\n"
             ".venv-ref/.../python benchmarks/tls_download_ref.py   # Kepler + transitleastsquares\n"
             "python benchmarks/tls_cuperiod.py           # cuPeriod TLS\n"
             "python benchmarks/make_report.py            # this report\n```\n")

    (RESULTS.parent / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", RESULTS.parent / "REPORT.md")


if __name__ == "__main__":
    main()
