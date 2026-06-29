"""Shared helpers for the cuPeriod validation + benchmark suite.

Dependency-light (numpy/pandas only) so it imports identically in the main GPU
venv and the isolated reference venv.

The validation light curves ship with the suite as a single self-contained
Parquet (``dataset/light_curves.parquet``) — 72 real ASAS-SN g-band light
curves spanning six variability classes, each with its VSX literature period.
No external catalogue or network access is needed to reproduce the validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
DATASET = BENCH / "dataset"
DATA = BENCH / "data"          # scratch: Kepler downloads, intermediates (gitignored)
RESULTS = BENCH / "results"
FIGURES = BENCH / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

LIGHT_CURVES = DATASET / "light_curves.parquet"


def load_dataset() -> tuple[pd.DataFrame, dict, dict, dict]:
    """Load the bundled validation set.

    Returns ``(meta, jd, mag, err)`` where ``meta`` is one row per star (id,
    ``vsx_type``, ``broad_class``, ``vsx_period``, ``n_det``, ``baseline``) and
    ``jd``/``mag``/``mag_err`` are dicts keyed by the string star id.
    """
    df = pd.read_parquet(LIGHT_CURVES)
    jd = {str(r.asas_sn_id): np.asarray(r.jd, dtype=np.float64) for r in df.itertuples()}
    mag = {str(r.asas_sn_id): np.asarray(r.mag, dtype=np.float64) for r in df.itertuples()}
    err = {str(r.asas_sn_id): np.asarray(r.mag_err, dtype=np.float64) for r in df.itertuples()}
    meta = df.drop(columns=["jd", "mag", "mag_err"]).reset_index(drop=True)
    return meta, jd, mag, err


# --- frequency-grid builders -----------------------------------------------
def bracket_grid(p_true: float, baseline: float, *, span: float, n: int) -> np.ndarray:
    """Uniform frequency grid bracketing f_true=1/p_true within [f/span, f*span].

    Floored at 1/baseline, capped at 80 c/d. Returns an ascending float64 array.
    """
    f_true = 1.0 / p_true
    f_lo = max(1.0 / baseline, f_true / span)
    f_hi = min(80.0, f_true * span)
    if f_hi <= f_lo:
        f_hi = f_lo * 2.0
    return np.linspace(f_lo, f_hi, int(n), dtype=np.float64)


def fine_grid(p_true: float, baseline: float, *, span: float, samples_per_peak: int,
              cap: int) -> np.ndarray:
    """Peak-resolving uniform frequency grid (>= samples_per_peak per Rayleigh)."""
    f_true = 1.0 / p_true
    f_lo = max(1.0 / baseline, f_true / span)
    f_hi = min(80.0, f_true * span)
    if f_hi <= f_lo:
        f_hi = f_lo * 2.0
    df = 1.0 / (samples_per_peak * baseline)
    n = int(min(cap, max(1000, round((f_hi - f_lo) / df))))
    return np.linspace(f_lo, f_hi, n, dtype=np.float64)


# --- recovery scoring: accept harmonic / sub-harmonic / alias relationships -
HARMONIC_RATIOS = (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 2.0 / 3.0, 3.0 / 2.0)


def period_match(p_found: float, p_true: float, *, rel_tol: float = 0.02) -> tuple[bool, float]:
    """Does p_found match p_true up to a small-integer harmonic? Returns (ok, ratio)."""
    if not (np.isfinite(p_found) and p_found > 0 and np.isfinite(p_true) and p_true > 0):
        return False, np.nan
    ratio = p_found / p_true
    for r in HARMONIC_RATIOS:
        if abs(ratio / r - 1.0) <= rel_tol:
            return True, r
    return False, ratio


def exact_match(p_found: float, p_true: float, *, rel_tol: float = 0.02) -> bool:
    """Does p_found match p_true directly (no harmonic), within rel_tol?"""
    if not (np.isfinite(p_found) and p_found > 0):
        return False
    return abs(p_found / p_true - 1.0) <= rel_tol


__all__ = [
    "load_dataset", "bracket_grid", "fine_grid", "period_match", "exact_match",
    "DATASET", "DATA", "RESULTS", "FIGURES", "LIGHT_CURVES",
]
