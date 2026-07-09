"""1-1 validation of every cuPeriod method on real ASAS-SN light curves.

For each selected star and method we record three things:

  * CPU vs GPU parity   - cuPeriod's two backends on an identical grid (max|d|,
    max relative error). These MUST agree to floating-point round-off.
  * cuPeriod vs reference - cuPeriod (CPU) against an independent implementation
    on the identical grid (max|d|, Pearson r, agreement of the recovered period).
  * Period recovery     - cuPeriod's best period vs the VSX literature period,
    both directly and allowing a small-integer harmonic (the method-appropriate
    fold ambiguity, e.g. GLS finding P/2 for contact binaries).

Frequency-grid methods (GLS, PDM, CE, StringLength, MHAOV) use an explicit
shared frequency grid. Box methods (BLS, TLS) use cuPeriod's settings-built grid;
BLS's reference is astropy BoxLeastSquares via cuPeriod's own ``astropy`` backend
(identical grid by construction).

Writes results/validation_metrics.parquet and a few full spectra to
results/spectra/ for the report's overlay figures.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

import references as R  # noqa: E402
from _common import RESULTS, exact_match, load_dataset, period_match  # noqa: E402
from cuperiod.core.backend import torch_available  # noqa: E402
from cuperiod.core.errors import BackendUnavailableError  # noqa: E402
from cuperiod.methods.base import get_method  # noqa: E402


def supports_torch(method):
    """Whether ``method`` has the portable torch backend and torch is importable."""
    return torch_available() and "torch" in get_method(method).all_backends


def safe_periodogram(*args, **kwargs):
    """``cup.periodogram``, but returns ``None`` if the backend is unavailable here."""
    try:
        return cup.periodogram(*args, **kwargs)
    except BackendUnavailableError:
        return None

SPECTRA = RESULTS / "spectra"
SPECTRA.mkdir(exist_ok=True)

FREQ_METHODS = ["GLS", "PDM", "CE", "STRINGLENGTH", "MHAOV"]
SETTINGS = {
    "GLS": lambda: cup.GLSSettings(fap_method="none"),
    "PDM": lambda: cup.PDMSettings(),
    "CE": lambda: cup.CESettings(),
    "STRINGLENGTH": lambda: cup.StringLengthSettings(),
    "MHAOV": lambda: cup.MHAOVSettings(),
}
REF = {
    "GLS": lambda t, y, e, f: R.ref_gls(t, y, e, f),
    "PDM": lambda t, y, e, f: R.ref_pdm(t, y, f),
    "CE": lambda t, y, e, f: R.ref_ce(t, y, f),
    "STRINGLENGTH": lambda t, y, e, f: R.ref_stringlength(t, y, f),
    "MHAOV": lambda t, y, e, f: R.ref_mhaov(t, y, f),
}
SENSE = {"GLS": "max", "PDM": "min", "CE": "min", "STRINGLENGTH": "min",
         "MHAOV": "max", "BLS": "max", "TLS": "max"}
# save a full spectrum for the first star of each class (per method)
SAVE_SPECTRUM_FOR: set[str] = set()


def grids(p_true, baseline, span=2.5):
    f_true = 1.0 / p_true
    f_lo = max(1.0 / baseline, f_true / span)
    f_hi = min(80.0, f_true * span)
    if f_hi <= f_lo:
        f_hi = f_lo * 2
    agree = np.linspace(f_lo, f_hi, 12000)
    df = 1.0 / (6 * baseline)
    n_fine = int(np.clip(round((f_hi - f_lo) / df), 12000, 60000))
    fine = np.linspace(f_lo, f_hi, n_fine)
    return agree, fine


def best_from(power, freqs, sense):
    idx = np.argmin(power) if sense == "min" else np.argmax(power)
    return 1.0 / freqs[idx]


def run_freq_method(method, t, y, e, p_true, baseline, save_tag=None):
    agree, fine = grids(p_true, baseline)
    st = SETTINGS[method]
    sense = SENSE[method]
    ga = cup.GridSpec(kind="frequency", values=agree, uniform=True)
    gf = cup.GridSpec(kind="frequency", values=fine, uniform=True)

    # agreement grid: cuPeriod cpu + gpu + reference
    pc = cup.periodogram((t, y, e), method, backend="cpu", grid=ga, settings=st())
    pg = cup.periodogram((t, y, e), method, backend="gpu", grid=ga, settings=st())
    ref = REF[method](t, y, e, agree)

    # parity on the fine grid (the harder test: more points)
    fc = cup.periodogram((t, y, e), method, backend="cpu", grid=gf, settings=st())
    fg = cup.periodogram((t, y, e), method, backend="gpu", grid=gf, settings=st())

    par = np.abs(fc.power - fg.power)
    scale = np.maximum(np.abs(fc.power), 1e-30)
    cpu_ref = np.abs(pc.power - ref)
    finite = np.isfinite(ref) & np.isfinite(pc.power)
    corr = float(np.corrcoef(pc.power[finite], ref[finite])[0, 1]) if finite.sum() > 2 else np.nan

    p_cpu = fc.best_period()
    p_gpu = fg.best_period()
    p_ref = best_from(ref, agree, sense)
    ok_h, ratio = period_match(p_cpu, p_true)

    # torch backend: same fine grid, parity vs cuPeriod-CPU (mirrors the gpu-parity
    # check above). Absent gracefully if torch/no device is unavailable for this method.
    torch_row = torch_parity(method, t, y, e, fc.power, p_cpu, p_true, grid=gf, settings_fn=st)

    if save_tag:
        np.savez(SPECTRA / f"{save_tag}_{method}.npz",
                 freq=agree, cup=pc.power, ref=ref, sense=sense,
                 p_true=p_true, p_cpu=p_cpu, p_ref=p_ref)
    row = {
        "parity_max_abs": float(par.max()),
        "parity_max_rel": float((par / scale).max()),
        "cup_ref_max_abs": float(cpu_ref[finite].max()),
        "cup_ref_corr": corr,
        "p_cpu": p_cpu, "p_gpu": p_gpu, "p_ref": p_ref,
        "cpu_gpu_same": bool(abs(p_cpu / p_gpu - 1) < 1e-6),
        "recover_exact": exact_match(p_cpu, p_true),
        "recover_harmonic": ok_h, "harmonic_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "ref_recover_exact": exact_match(p_ref, p_true),
        "n_grid_fine": fine.size,
    }
    row.update(torch_row)
    return row


def torch_parity(method, t, y, e, cpu_power, p_cpu, p_true, *, grid=None, settings_fn=None,
                  settings=None):
    """CPU-vs-torch parity on the same grid/settings the CPU reference used.

    Returns NaN/False columns if the method has no torch backend, or torch/a
    usable device is not present on this machine -- mirrors how the GPU
    columns degrade gracefully without a CUDA device.
    """
    cols = {"max_diff_torch": np.nan, "rel_diff_torch": np.nan,
            "best_period_torch": np.nan, "torch_cpu_same": False,
            "torch_harmonic": False, "torch_backend": "-"}
    if not supports_torch(method):
        return cols
    kwargs = dict(backend="torch", settings=(settings_fn() if settings_fn else settings))
    if grid is not None:
        kwargs["grid"] = grid
    pt = safe_periodogram((t, y, e), method, **kwargs)
    if pt is None:
        return cols
    diff = np.abs(cpu_power - pt.power)
    scale = np.maximum(np.abs(cpu_power), 1e-30)
    p_torch = pt.best_period()
    ok_h, _ = period_match(p_torch, p_true)
    cols.update({
        "max_diff_torch": float(diff.max()),
        "rel_diff_torch": float((diff / scale).max()),
        "best_period_torch": p_torch,
        "torch_cpu_same": bool(abs(p_torch / p_cpu - 1) < 1e-6),
        "torch_harmonic": ok_h,
        "torch_backend": pt.backend,
    })
    return cols


def run_bls(t, y, e, p_true, baseline, save_tag=None):
    st = lambda: cup.BLSSettings(min_period_days=max(0.05, p_true / 2.0),
                                 max_period_days=p_true * 2.0, grid_oversample=1)
    pn = cup.periodogram((t, y, e), "BLS", backend="numpy", settings=st())
    pa = cup.periodogram((t, y, e), "BLS", backend="astropy", settings=st())
    pg = cup.periodogram((t, y, e), "BLS", backend="gpu", settings=st())
    par = np.abs(pn.power - pg.power)
    scale = np.maximum(np.abs(pn.power), 1e-30)
    nr = np.abs(pn.power - pa.power)
    p_n, p_g, p_a = pn.best_period(), pg.best_period(), pa.best_period()
    ok_h, ratio = period_match(p_n, p_true)
    torch_row = torch_parity("BLS", t, y, e, pn.power, p_n, p_true, settings=st())
    if save_tag:
        np.savez(SPECTRA / f"{save_tag}_BLS.npz", period=pn.period, cup=pn.power,
                 ref=pa.power, sense="max", p_true=p_true, p_cpu=p_n, p_ref=p_a)
    row = {
        "parity_max_abs": float(par.max()), "parity_max_rel": float((par / scale).max()),
        "cup_ref_max_abs": float(nr.max()),
        "cup_ref_corr": float(np.corrcoef(pn.power, pa.power)[0, 1]),
        "p_cpu": p_n, "p_gpu": p_g, "p_ref": p_a,
        "cpu_gpu_same": bool(abs(p_n / p_g - 1) < 1e-6),
        "recover_exact": exact_match(p_n, p_true), "recover_harmonic": ok_h,
        "harmonic_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "ref_recover_exact": exact_match(p_a, p_true), "n_grid_fine": pn.power.size,
    }
    row.update(torch_row)
    return row


def run_tls(t, y, e, p_true, baseline):
    st = lambda: cup.TLSSettings(min_period_days=max(0.05, p_true / 2.0),
                                 max_period_days=p_true * 2.0)
    pn = cup.periodogram((t, y, e), "TLS", backend="numpy", settings=st())
    pg = cup.periodogram((t, y, e), "TLS", backend="gpu", settings=st())
    par = np.abs(pn.power - pg.power)
    scale = np.maximum(np.abs(pn.power), 1e-30)
    p_n, p_g = pn.best_period(), pg.best_period()
    ok_h, ratio = period_match(p_n, p_true)
    torch_row = torch_parity("TLS", t, y, e, pn.power, p_n, p_true, settings=st())
    row = {
        "parity_max_abs": float(par.max()), "parity_max_rel": float((par / scale).max()),
        "cup_ref_max_abs": np.nan, "cup_ref_corr": np.nan,
        "p_cpu": p_n, "p_gpu": p_g, "p_ref": np.nan,
        "cpu_gpu_same": bool(abs(p_n / p_g - 1) < 1e-6),
        "recover_exact": exact_match(p_n, p_true), "recover_harmonic": ok_h,
        "harmonic_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "ref_recover_exact": np.nan, "n_grid_fine": pn.power.size,
    }
    row.update(torch_row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N stars (smoke-testing)")
    args = parser.parse_args()

    meta, jd, mag, err = load_dataset()
    if args.limit is not None:
        meta = meta.iloc[:args.limit].reset_index(drop=True)

    seen_class: set[str] = set()
    rows = []
    t_start = time.perf_counter()
    for i, row in meta.iterrows():
        sid = str(row["asas_sn_id"])
        t, y, e = jd[sid], mag[sid], err[sid]
        P = float(row["vsx_period"]); bl = float(row["baseline"])
        bclass = row["broad_class"]
        save_tag = sid if bclass not in seen_class else None
        seen_class.add(bclass)
        methods = FREQ_METHODS + ["BLS"]
        if bclass == "ECLIPSING":
            methods = methods + ["TLS"]  # TLS only meaningful for eclipses here
        for method in methods:
            try:
                if method in FREQ_METHODS:
                    m = run_freq_method(method, t, y, e, P, bl, save_tag=save_tag)
                elif method == "BLS":
                    m = run_bls(t, y, e, P, bl, save_tag=save_tag)
                else:
                    m = run_tls(t, y, e, P, bl)
                m.update(asas_sn_id=sid, method=method, broad_class=bclass,
                         vsx_type=row["vsx_type"], vsx_period=P, baseline=bl,
                         n_det=int(row["n_det"]))
                rows.append(m)
            except Exception as ex:
                print(f"  FAIL {sid} {method}: {type(ex).__name__}: {ex}")
        done = i + 1
        if done % 5 == 0 or done == len(meta):
            el = time.perf_counter() - t_start
            print(f"[{done}/{len(meta)}] {bclass:12s} {sid}  ({el:.0f}s elapsed)")

    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / "validation_metrics.parquet", index=False)
    print(f"\nwrote {RESULTS/'validation_metrics.parquet'}  ({len(res)} rows)")
    # quick summary
    for method in FREQ_METHODS + ["BLS", "TLS"]:
        sub = res[res["method"] == method]
        if sub.empty:
            continue
        tsub = sub.dropna(subset=["rel_diff_torch"])
        tstr = (f" | torch max_rel|d|={tsub['rel_diff_torch'].max():.1e} "
                f"cpu==torch={tsub['torch_cpu_same'].mean()*100:.0f}% (n={len(tsub)})"
                if len(tsub) else " | torch=—")
        print(f"  {method:12s} n={len(sub):3d} | parity max|d|={sub['parity_max_abs'].max():.1e} "
              f"| cup-ref max|d|={np.nanmax(sub['cup_ref_max_abs']):.1e} "
              f"corr>={np.nanmin(sub['cup_ref_corr']):.4f} "
              f"| recover(harm)={sub['recover_harmonic'].mean()*100:.0f}% "
              f"cpu==gpu={sub['cpu_gpu_same'].mean()*100:.0f}%{tstr}")


if __name__ == "__main__":
    main()
