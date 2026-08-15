"""Performance benchmarks: cuPeriod CPU vs GPU vs reference tools.

Three views, all on a realistic ASAS-SN-like light curve (~900 points, multi-year
baseline):

  1. single-LC wall time per method (CPU backend, GPU backend, reference tool),
     at a representative grid size -> speedup table;
  2. scaling of CPU/GPU wall time with the search-grid size;
  3. batch throughput (light curves / second), GPU vs a CPU worker pool.

Grids are bounded explicitly so no setting can explode (box methods use a fixed
period window, frequency methods an explicit length). Times are the best of a few
repeats with the GPU warmed up first. Results -> results/bench_*.parquet, written
per section so progress is durable.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

from _common import RESULTS, load_dataset  # noqa: E402
from cuperiod.core.backend import torch_available  # noqa: E402
from cuperiod.core.errors import BackendUnavailableError  # noqa: E402
from cuperiod.methods.base import get_method  # noqa: E402

# bounded period window for the box methods (BLS/TLS) — independent of the star,
# so the trial-period count can never blow up on a short-period target.
BOX_PMIN, BOX_PMAX = 0.5, 4.0
FREQ_LO, FREQ_HI = 0.05, 6.0  # c/d window for the frequency methods' fixed grid


def representative_lc():
    """A real ASAS-SN light curve with a moderate (~1 d) period and ~median size."""
    meta, jd, mag, err = load_dataset()
    cand = meta[(meta.vsx_period.between(0.8, 2.5)) & (meta.n_det.between(700, 1300))]
    cand = cand if len(cand) else meta
    row = cand.iloc[(cand.n_det - 900).abs().argmin()]
    sid = str(row["asas_sn_id"])
    return jd[sid], mag[sid], err[sid], float(row["vsx_period"])


def resample(t, y, e, n):
    rng = np.random.default_rng(0)
    if n <= t.size:
        idx = np.sort(rng.choice(t.size, n, replace=False))
        return t[idx], y[idx], e[idx]
    reps = int(np.ceil(n / t.size))
    base = t.max() - t.min()
    tt = np.concatenate([t + k * base * 1.0001 for k in range(reps)])[:n]
    o = np.argsort(tt)
    return tt[o], np.tile(y, reps)[:n][o], np.tile(e, reps)[:n][o]


def best_time(fn, repeat=3):
    fn()  # warmup (kernel/plan/JIT compile)
    return min(_timed(fn) for _ in range(repeat))


def safe_best_time(fn, repeat=3):
    """:func:`best_time`, but a backend unavailable here records NaN instead of raising.

    Lets the sweep run on machines missing a backend — no CUDA GPU (the ``gpu`` column),
    or no torch / no torch GPU device (the ``torch`` column) — leaving that cell blank.
    """
    try:
        return best_time(fn, repeat)
    except BackendUnavailableError:
        return float("nan")


def supports_torch(method):
    """Whether ``method`` has the portable torch backend and torch is importable."""
    return torch_available() and "torch" in get_method(method).all_backends


def _timed(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def freq_grid(n):
    return cup.GridSpec(kind="frequency",
                        values=np.linspace(FREQ_LO, FREQ_HI, int(n)), uniform=True)


FREQ_SETTINGS = {
    "GLS": lambda: cup.GLSSettings(fap_method="none"),
    "PDM": lambda: cup.PDMSettings(),
    "CE": lambda: cup.CESettings(),
    "STRINGLENGTH": lambda: cup.StringLengthSettings(),
    "MHAOV": lambda: cup.MHAOVSettings(),
    "SUPERSMOOTHER": lambda: cup.SuperSmootherSettings(),
}

#: Methods swept in the scaling sections (2 and 3). SuperSmoother joins GLS/PDM/MHAOV
#: as the costliest fold method; no external reference tool is timed for it (gatspy's
#: pure-python smoother would dominate the sweep; accuracy parity is pinned in tests).
SCALING_METHODS = ["GLS", "PDM", "MHAOV", "SUPERSMOOTHER"]


# --------------------------------------------------------------------------
def bench_single(t, y, e):
    """Per-method CPU vs GPU vs reference tool at a fixed 30k-point grid."""
    import references as R
    grid = freq_grid(30000)
    f = grid.frequency
    rows = []
    reftime = {"GLS": lambda: R.ref_gls(t, y, e, f),
               "PDM": lambda: R.ref_pdm(t, y, f)}      # established third-party tools
    refname = {"GLS": "astropy", "PDM": "PyAstronomy"}
    for m, st in FREQ_SETTINGS.items():
        be = cup.periodogram((t, y, e), m, backend="cpu", grid=grid, settings=st()).backend
        tc = best_time(lambda: cup.periodogram((t, y, e), m, backend="cpu", grid=grid, settings=st()))
        tg = safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="gpu", grid=grid, settings=st()))
        has_torch = supports_torch(m)
        tt = (safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="torch", grid=grid, settings=st()))
              if has_torch else np.nan)
        tbe = (cup.periodogram((t, y, e), m, backend="torch", grid=grid, settings=st()).backend
               if has_torch and np.isfinite(tt) else "—")
        # Reference tools are ad-hoc installs (`uv sync` prunes them) — a missing
        # one blanks its cell rather than killing the whole section.
        tr = np.nan
        if m in reftime:
            try:
                tr = best_time(reftime[m], repeat=1)
            except ImportError as exc:
                print(f"  {m}: reference tool unavailable ({exc}); skipping ref timing",
                      flush=True)
        rows.append(dict(method=m, n_grid=grid.size, cpu_backend=be, cpu_s=tc, gpu_s=tg,
                         torch_s=tt, torch_backend=tbe,
                         ref_s=tr, ref=refname.get(m, "—"),
                         gpu_speedup=(tc / tg if np.isfinite(tg) else np.nan),
                         torch_speedup=(tc / tt if np.isfinite(tt) else np.nan),
                         gpu_vs_ref=(tr / tg if np.isfinite(tr) and np.isfinite(tg) else np.nan),
                         cpu_vs_ref=(tr / tc if np.isfinite(tr) else np.nan)))
        gstr = f"gpu={tg:.4f}s ({tc/tg:.0f}x)" if np.isfinite(tg) else "gpu=—"
        tstr = f" torch({tbe})={tt:.3f}s" if np.isfinite(tt) else ""
        print(f"  {m:12s} cpu({be})={tc:.3f}s {gstr}{tstr}", flush=True)
    # box methods on a fixed, bounded period window. cpu_s is cuPeriod's *default*
    # CPU backend: "cpu" resolves to the multicore numba box search for BLS (or
    # astropy if numba is not installed), to numpy for TLS.
    for m in ["BLS", "TLS"]:
        S = (cup.BLSSettings(min_period_days=BOX_PMIN, max_period_days=BOX_PMAX, grid_oversample=1)
             if m == "BLS" else
             cup.TLSSettings(min_period_days=BOX_PMIN, max_period_days=BOX_PMAX))
        be = cup.periodogram((t, y, e), m, backend="cpu", settings=S).backend
        tc = best_time(lambda: cup.periodogram((t, y, e), m, backend="cpu", settings=S), repeat=2)
        tg = safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="gpu", settings=S), repeat=2)
        ng = cup.periodogram((t, y, e), m, backend="cpu", settings=S).power.size  # backend-independent
        has_torch = supports_torch(m)
        tt = (safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="torch", settings=S), repeat=2)
              if has_torch else np.nan)
        tbe = (cup.periodogram((t, y, e), m, backend="torch", settings=S).backend
               if has_torch and np.isfinite(tt) else "—")
        # BLS reference = astropy's compiled BoxLeastSquares; also time the pure-numpy
        # GPU-parity reference port to document it is not the product path.
        ref_s = (best_time(lambda: cup.periodogram((t, y, e), "BLS", backend="astropy", settings=S), repeat=1)
                 if m == "BLS" else np.nan)
        ref = "astropy" if m == "BLS" else "—"
        port_s = (best_time(lambda: cup.periodogram((t, y, e), "BLS", backend="numpy", settings=S), repeat=1)
                  if m == "BLS" else np.nan)
        rows.append(dict(method=m, n_grid=int(ng), cpu_backend=be, cpu_s=tc, gpu_s=tg,
                         torch_s=tt, torch_backend=tbe,
                         ref_s=ref_s, ref=ref,
                         gpu_speedup=(tc / tg if np.isfinite(tg) else np.nan),
                         torch_speedup=(tc / tt if np.isfinite(tt) else np.nan),
                         gpu_vs_ref=(ref_s / tg if np.isfinite(ref_s) and np.isfinite(tg) else np.nan),
                         cpu_vs_ref=(ref_s / tc if np.isfinite(ref_s) else np.nan),
                         cpu_port_s=port_s))
        gstr = f"gpu={tg:.4f}s (gpu {tc/tg:.0f}x)" if np.isfinite(tg) else "gpu=—"
        tstr = f" torch({tbe})={tt:.3f}s" if np.isfinite(tt) else ""
        extra = (f"  [vs astropy {ref_s/tc:.0f}x faster; numpy-port {port_s:.1f}s]"
                 if m == "BLS" else "")
        print(f"  {m:12s} cpu({be})={tc:.3f}s {gstr}{tstr}{extra}  [{ng:,} periods]",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "bench_single.parquet", index=False)
    return df


def bench_scaling_npoints(t, y, e):
    grid = freq_grid(30000)
    rows = []
    for n in [100, 300, 1000, 3000, 10000, 30000]:
        tt, yy, ee = resample(t, y, e, n)
        for m in SCALING_METHODS:
            st = FREQ_SETTINGS[m]
            tc = best_time(lambda: cup.periodogram((tt, yy, ee), m, backend="cpu", grid=grid, settings=st()), repeat=1)
            tg = safe_best_time(lambda: cup.periodogram((tt, yy, ee), m, backend="gpu", grid=grid, settings=st()), repeat=1)
            ttor = (safe_best_time(lambda: cup.periodogram((tt, yy, ee), m, backend="torch", grid=grid, settings=st()), repeat=1)
                    if supports_torch(m) else np.nan)
            rows.append(dict(axis="npoints", method=m, n=n, cpu_s=tc, gpu_s=tg, torch_s=ttor,
                             speedup=(tc / tg if np.isfinite(tg) else np.nan),
                             torch_speedup=(tc / ttor if np.isfinite(ttor) else np.nan)))
        print(f"  N={n:>6}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "bench_npoints.parquet", index=False)
    return df


def bench_scaling_grid(t, y, e):
    rows = []
    for n in [1000, 3000, 10000, 30000, 100000]:
        grid = freq_grid(n)
        for m in SCALING_METHODS:
            st = FREQ_SETTINGS[m]
            tc = best_time(lambda: cup.periodogram((t, y, e), m, backend="cpu", grid=grid, settings=st()), repeat=1)
            tg = safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="gpu", grid=grid, settings=st()), repeat=1)
            ttor = (safe_best_time(lambda: cup.periodogram((t, y, e), m, backend="torch", grid=grid, settings=st()), repeat=1)
                    if supports_torch(m) else np.nan)
            rows.append(dict(axis="grid", method=m, n=n, cpu_s=tc, gpu_s=tg, torch_s=ttor,
                             speedup=(tc / tg if np.isfinite(tg) else np.nan),
                             torch_speedup=(tc / ttor if np.isfinite(ttor) else np.nan)))
        print(f"  grid={n:>7}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "bench_grid.parquet", index=False)
    return df


def bench_batch(t, y, e):
    grid = freq_grid(10000)
    rng = np.random.default_rng(1)
    rows = []
    cpu_cap = 1024
    # warm up the parent process (imports, JIT/kernel compilation). The pool itself is
    # rebuilt per call, so each timed run still pays its own spawn + CUDA-context cost —
    # see the single-shot note below.
    warm = [(f"w{i}", cup.LightCurve.from_arrays(t, y, e)) for i in range(64)]
    for method in ["GLS", "PDM"]:
        cup.batch_periodograms(warm, method, device="gpu", grid=grid,
                               settings=FREQ_SETTINGS[method](), n_best=5)
    for n_lc in [256, 1024, 4096]:  # GPU only on the largest; CPU pool capped to keep it quick
        lcs = [(f"s{i}", cup.LightCurve.from_arrays(t, y * (1 + 0.02 * rng.standard_normal(y.size)), e))
               for i in range(n_lc)]
        for method in ["GLS", "PDM"]:
            st = FREQ_SETTINGS[method]
            # One batch pass. This is a *single-shot* rate: it includes the one-off
            # worker-pool spin-up (spawn + per-worker CUDA context), so it is a
            # conservative floor — a warmed pool sustains a higher rate over many chunks.
            # (We don't loop for a best-of-N here: repeatedly recreating the GPU process
            # pool in one process can deadlock the spawn pool against the parent's CUDA
            # context.)
            t0 = time.perf_counter()
            cup.batch_periodograms(lcs, method, device="gpu", grid=grid, settings=st(), n_best=5)
            tg = time.perf_counter() - t0
            tc, cpu_rate = np.nan, np.nan
            if n_lc <= cpu_cap:
                t0 = time.perf_counter()
                cup.batch_periodograms(lcs, method, device="cpu", grid=grid, settings=st(), n_best=5)
                tc = time.perf_counter() - t0
                cpu_rate = n_lc / tc
            rows.append(dict(method=method, n_lc=n_lc, gpu_s=tg, cpu_s=tc,
                             gpu_lc_per_s=n_lc / tg, cpu_lc_per_s=cpu_rate,
                             speedup=(tc / tg if np.isfinite(tc) else np.nan)))
            cpu_str = f"CPU {cpu_rate:,.0f} lc/s ({tc/tg:.1f}x)" if np.isfinite(tc) else "CPU skipped"
            print(f"  batch {method} n={n_lc}: GPU {n_lc/tg:,.0f} lc/s, {cpu_str}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "bench_batch.parquet", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections", default=None,
        help="comma-separated subset of single,npoints,grid,batch. Default runs "
             "all four, skipping 'single' when results/bench_single.parquet "
             "already exists (resume behaviour); naming sections runs exactly "
             "those, re-measuring even if a parquet exists.")
    args = parser.parse_args()
    known = ("single", "npoints", "grid", "batch")
    chosen = None
    if args.sections is not None:
        chosen = [s.strip() for s in args.sections.split(",") if s.strip()]
        bad = sorted(set(chosen) - set(known))
        if bad:
            parser.error(f"unknown sections {bad}; choose from {known}")

    def want(name: str, default: bool = True) -> bool:
        return name in chosen if chosen is not None else default

    t, y, e, P = representative_lc()
    print(f"representative LC: N={t.size}, baseline={t.max()-t.min():.0f} d, P={P:.4f}", flush=True)
    if want("single", default=not (RESULTS / "bench_single.parquet").exists()):
        print("\n[1/4] single-LC per-method timing...", flush=True)
        bench_single(t, y, e)
    else:
        print("\n[1/4] single-LC: skipped", flush=True)
    if want("npoints"):
        print("\n[2/4] scaling vs N points...", flush=True)
        bench_scaling_npoints(t, y, e)
    if want("grid"):
        print("\n[3/4] scaling vs grid size...", flush=True)
        bench_scaling_grid(t, y, e)
    if want("batch"):
        print("\n[4/4] batch throughput...", flush=True)
        bench_batch(t, y, e)
    print("\nBENCHMARK_DONE — wrote results/bench_*.parquet", flush=True)


if __name__ == "__main__":
    main()
