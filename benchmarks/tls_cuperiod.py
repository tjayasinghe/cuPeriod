"""TLS validation, part B (runs in the main GPU venv).

Loads the raw Kepler light curves saved by tls_download_ref.py and recovers each
period with cuPeriod's TLS on the CPU and GPU backends, on the same broad blind
search range used by the `transitleastsquares` reference. Merges the reference
recoveries and writes results/tls_results.parquet.
"""

from __future__ import annotations

import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

from _common import DATA, RESULTS  # noqa: E402

MEN = str(DATA / "mendeley")


def cuperiod_tls(t, f, backend, pmin=0.5, pmax=12.0):
    st = cup.TLSSettings(min_period_days=pmin, max_period_days=pmax)
    # Kepler flux: dips below 1.0 -> tell cuPeriod it is flux, not magnitude.
    try:
        dom = cup.Domain.FLUX
    except AttributeError:
        dom = None
    t0 = time.perf_counter()
    pg = cup.periodogram((t, f), "TLS", backend=backend, settings=st, domain=dom)
    dt = time.perf_counter() - t0
    return pg, dt


def main():
    files = sorted(glob.glob(f"{MEN}/lc_*.npz"))
    print(f"{len(files)} Kepler light curves")
    # GPU recovers every curve; the CPU backend (heavy on ~20k-point Kepler
    # curves) is timed on a subset to document parity + speedup.
    n_cpu = 5
    rows = []
    for i, fp in enumerate(files):
        kepid = int(os.path.basename(fp)[3:-4])
        d = np.load(fp)
        t, f, P = d["t"], d["f"], float(d["koi_period"])
        pg, tg = cuperiod_tls(t, f, "gpu")
        pgpu = pg.best_period()

        def relerr(p):
            return min(abs(p / P - 1.0), abs(p / (2 * P) - 1.0), abs(p / (P / 2) - 1.0))
        row = dict(kepid=kepid, koi_period=P, n=int(t.size),
                   cup_gpu_period=pgpu, cup_gpu_relerr=relerr(pgpu), gpu_s=tg,
                   cup_cpu_period=np.nan, cup_cpu_relerr=np.nan,
                   cpu_gpu_parity=np.nan, cpu_s=np.nan, gpu_speedup=np.nan)
        if i < n_cpu:
            pc, tc = cuperiod_tls(t, f, "cpu")
            row.update(cup_cpu_period=pc.best_period(),
                       cup_cpu_relerr=relerr(pc.best_period()),
                       cpu_gpu_parity=float(np.max(np.abs(pc.power - pg.power))),
                       cpu_s=tc, gpu_speedup=tc / tg)
        rows.append(row)
        cpu_str = (f"cpu={row['cup_cpu_period']:.4f} parity={row['cpu_gpu_parity']:.1e} "
                   f"x{row['gpu_speedup']:.0f}" if i < n_cpu else "cpu=skipped")
        print(f"KIC {kepid:>9} P={P:7.4f} | gpu={pgpu:7.4f} relerr={relerr(pgpu):.2e} "
              f"| {cpu_str} | gpu={tg:.2f}s", flush=True)

    res = pd.DataFrame(rows)
    ref_path = f"{MEN}/tls_reference_results.csv"
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)[["kepid", "tls_ref_period", "tls_ref_rel_err",
                                     "tls_ref_sde", "tls_s"]]
        res = res.merge(ref, on="kepid", how="left")
    res.to_parquet(RESULTS / "tls_results.parquet", index=False)
    print(f"\nwrote {RESULTS/'tls_results.parquet'}")
    if "tls_ref_period" in res:
        print(res[["kepid", "koi_period", "cup_cpu_period", "tls_ref_period",
                   "cup_cpu_relerr", "gpu_speedup"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
