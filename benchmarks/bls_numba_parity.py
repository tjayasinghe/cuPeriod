"""Verify the default CPU BLS backend (multicore numba) against the pure-NumPy
BLS port, on every star in the bundled validation dataset.

Context
-------
``validate_periodograms.py``'s BLS check uses ``backend="numpy"`` as the
ground truth for CPU<->GPU parity (the numpy port is slow but simple, so it
doubles as a GPU-parity reference) and never exercises ``backend="cpu"`` (the
multicore numba box search that ``backend="cpu"`` actually resolves to for
most users). This script closes that gap: it runs both CPU backends on an
identical bounded period grid and checks they agree to round-off. The output,
``results/bls_numba_parity.parquet`` (columns: ``sid``, ``maxabs``, ``same``),
is read directly by ``make_report.py`` for the "CPU box search beats astropy"
paragraph.

Run with the main GPU venv: ``.venv\\Scripts\\python.exe benchmarks\\bls_numba_parity.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cuperiod as cup

from _common import RESULTS, load_dataset


def main() -> None:
    meta, jd, mag, err = load_dataset()
    rows = []
    for r in meta.itertuples():
        sid = str(r.asas_sn_id)
        t, y, e = jd[sid], mag[sid], err[sid]
        p_true = float(r.vsx_period)
        settings = cup.BLSSettings(
            min_period_days=max(0.05, p_true / 2.0),
            max_period_days=p_true * 2.0,
            grid_oversample=1,
        )
        p_numpy = cup.periodogram((t, y, e), "BLS", backend="numpy", settings=settings)
        p_numba = cup.periodogram((t, y, e), "BLS", backend="cpu", settings=settings)
        maxabs = float(np.abs(p_numpy.power - p_numba.power).max())
        same = bool(abs(p_numpy.best_period() / p_numba.best_period() - 1) < 1e-6)
        rows.append({"sid": r.asas_sn_id, "maxabs": maxabs, "same": same})

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "bls_numba_parity.parquet", index=False)
    print(f"wrote {RESULTS / 'bls_numba_parity.parquet'}  ({len(out)} stars)")
    print(f"max|d(power)| = {out.maxabs.max():.2e} | identical best period on "
          f"{int(out.same.sum())}/{len(out)}")


if __name__ == "__main__":
    main()
