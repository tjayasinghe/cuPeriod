"""TLS validation, part A (runs in the reference venv .venv-ref).

For each confirmed Kepler KOI target: download a few quarters of PDCSAP flux with
lightkurve (the same tool the Mendeley dataset used), flatten/normalize, save the
raw (time, flux) series, and recover the period with the `transitleastsquares`
package. cuPeriod's TLS is run separately in the main GPU venv (tls_cuperiod.py)
on the identical saved series.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightkurve as lk  # noqa: E402
from transitleastsquares import transitleastsquares  # noqa: E402

from pathlib import Path  # noqa: E402

_HERE = Path(__file__).resolve().parent
MEN = str(_HERE / "data" / "mendeley")          # Kepler downloads (gitignored)
TARGETS = _HERE / "dataset" / "kepler_koi_targets.csv"   # committed KOI list
QUARTERS = [1, 2, 3, 4, 5, 6]


def fetch(kepid):
    sr = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler",
                              cadence="long", author="Kepler")
    # quarter is encoded in the 'mission' column, e.g. "Kepler Quarter 03"
    keep = [any(f"Quarter {q:02d}" in str(mis) for q in QUARTERS)
            for mis in sr.table["mission"]]
    sr = sr[keep]
    if len(sr) == 0:
        return None
    lcc = sr.download_all(flux_column="pdcsap_flux")
    lc = lcc.stitch().remove_nans().remove_outliers(sigma=5)
    flat = lc.flatten(window_length=401)
    t = np.asarray(flat.time.value, float)
    f = np.asarray(flat.flux.value, float)
    m = np.isfinite(t) & np.isfinite(f)
    return t[m], f[m]


def main():
    Path(MEN).mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(TARGETS)
    only = None
    if len(sys.argv) > 1:
        only = int(sys.argv[1])
    rows = []
    for _, tg in targets.iterrows():
        kepid = int(tg["kepid"]); P = float(tg["koi_period"])
        if only is not None and kepid != only:
            continue
        try:
            t0 = time.perf_counter()
            res = fetch(kepid)
            if res is None:
                print(f"KIC {kepid}: no data"); continue
            t, f = res
            np.savez(f"{MEN}/lc_{kepid}.npz", t=t, f=f, koi_period=P,
                     koi_duration=float(tg["koi_duration"]))
            dt_dl = time.perf_counter() - t0
            # transitleastsquares reference recovery (blind broad search)
            t0 = time.perf_counter()
            model = transitleastsquares(t, f)
            r = model.power(period_min=0.5, period_max=12.0,
                            oversampling_factor=2, duration_grid_step=1.15,
                            show_progress_bar=False, use_threads=1)
            dt_tls = time.perf_counter() - t0
            rec = float(r.period)
            err = abs(rec / P - 1.0)
            rows.append(dict(kepid=kepid, koi_period=P, n=t.size,
                             tls_ref_period=rec, tls_ref_rel_err=err,
                             tls_ref_sde=float(r.SDE), dl_s=dt_dl, tls_s=dt_tls))
            print(f"KIC {kepid:>9} P={P:7.4f} N={t.size:6d} | TLS-ref={rec:7.4f} "
                  f"relerr={err:.2e} SDE={r.SDE:5.1f} | dl={dt_dl:.0f}s tls={dt_tls:.0f}s",
                  flush=True)
        except Exception as ex:
            print(f"KIC {kepid}: FAIL {type(ex).__name__}: {ex}", flush=True)
    if rows:
        out = pd.DataFrame(rows)
        path = f"{MEN}/tls_reference_results.csv"
        if only is None:
            out.to_csv(path, index=False)
            print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
