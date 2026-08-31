"""Injection-recovery and timing for the period-change (dP/dt) tools.

Plants linearly drifting two-harmonic signals across a range of drift
strengths and asks three questions the docs make claims about:

* does ``pdot_search`` localize the planted (P, dP/dt) — and at what drift
  strength does the constant-period row actually lose the signal;
* is ``phase_track``'s reported uncertainty honest (pull distribution), and
  does its F-test detect real drifts without firing on null stars;
* what does a search + track chain cost per star, per backend.

Setup (stated so the numbers can be judged):

* 1500-day baseline, 800 points, irregular sampling; amplitudes 0.30 + 0.10
  (first harmonic), 0.05 mag noise — a bright ground-based survey star.
* Periods ~ U(0.35, 0.9) d. Drift strength is parameterized the way the
  search grids it: the accumulated phase drift over the half-baseline,
  ``D = f * |a| * T^2 / 8`` cycles, swept over U(0.25, 8), random sign
  (``a = (dP/dt)/P``). D ~ 0.5 is where a constant-period search starts to
  suffer; D >~ 2 is destroyed.
* Search recovery = best cell within 2 dP/dt steps and 0.1% in period of the
  truth. Track detection = F-test p < 1e-3. A separate null population
  (dP/dt = 0) measures the false-alarm side of the same threshold.

Writes results/pdot_recovery.parquet and prints the summary tables.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from _common import RESULTS  # noqa: E402

import cuperiod as cup  # noqa: E402

SEED = 42
SPAN = 1500.0
N_POINTS = 800
NOISE = 0.05
AMP, AMP2 = 0.30, 0.10
DRIFT_RANGE = (0.25, 8.0)
F_TEST_ALPHA = 1e-3


def simulate(
    rng: np.random.Generator, drift_cycles: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One drifting star; returns (t, mag, err, period, pdot_over_p)."""
    period = float(rng.uniform(0.35, 0.9))
    f0 = 1.0 / period
    a = 8.0 * drift_cycles / (f0 * SPAN * SPAN) * float(rng.choice([-1.0, 1.0]))
    t = np.sort(rng.uniform(0.0, SPAN, N_POINTS)) + 2455000.0
    dt = t - t.mean()
    tau = np.log1p(a * dt) / a if a != 0.0 else dt
    mag = (
        14.0
        + AMP * np.sin(2 * np.pi * f0 * tau + rng.uniform(0, 2 * np.pi))
        + AMP2 * np.sin(4 * np.pi * f0 * tau + rng.uniform(0, 2 * np.pi))
        + rng.normal(0.0, NOISE, N_POINTS)
    )
    return t, mag, np.full(N_POINTS, NOISE), period, a


def run_one(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    period: float,
    a_true: float,
    backend: str,
) -> dict[str, float | bool | str]:
    t0 = time.perf_counter()
    res = cup.pdot_search((t, mag, err), period=period, backend=backend)
    t_search = time.perf_counter() - t0
    step = float(res.pdot_over_p[1] - res.pdot_over_p[0])
    b = res.best
    row: dict[str, float | bool | str] = {
        "period_true": period,
        "a_true": a_true,
        "a_est": b.pdot_over_p,
        "a_step": step,
        "search_ok": bool(
            abs(b.pdot_over_p - a_true) <= 2 * step
            and abs(b.period - period) <= 1e-3 * period
        ),
        "power_best": b.power,
        "power_zero": b.zero_pdot_power,
        "t_search_s": t_search,
        "backend": res.backend,
    }
    t0 = time.perf_counter()
    try:
        track = cup.phase_track((t, mag, err), period=period, pdot=b.pdot)
        row.update(
            track_pull=(track.pdot / period - a_true) / (track.pdot_err / period),
            track_detect=bool(track.f_test_p < F_TEST_ALPHA),
            f_test_p=track.f_test_p,
            wrap_risk=track.wrap_risk,
        )
    except cup.InsufficientDataError:
        row.update(track_pull=np.nan, track_detect=False, f_test_p=np.nan)
    row["t_track_s"] = time.perf_counter() - t0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stars", type=int, default=100)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(args.n_stars):
        drift = float(rng.uniform(*DRIFT_RANGE))
        rows.append(run_one(*simulate(rng, drift), backend=args.backend))
        rows[-1]["drift_cycles"] = drift
        rows[-1]["null"] = False
    for _ in range(args.n_null):
        rows.append(run_one(*simulate(rng, 0.0), backend=args.backend))
        rows[-1]["drift_cycles"] = 0.0
        rows[-1]["null"] = True
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / "pdot_recovery.parquet", index=False)

    inj = df[~df["null"]]
    bins = pd.cut(inj["drift_cycles"], [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    print(f"\nInjection ({len(inj)} stars, backend={inj['backend'].iloc[0]}):")
    print(
        inj.groupby(bins, observed=True)
        .agg(
            n=("search_ok", "size"),
            search_ok=("search_ok", "mean"),
            track_detect=("track_detect", "mean"),
            pull_rms=("track_pull", lambda p: float(np.sqrt(np.mean(p**2)))),
        )
        .round(3)
        .to_string()
    )
    null = df[df["null"]]
    print(
        f"\nNull ({len(null)} stars): false detections at p<{F_TEST_ALPHA:g}: "
        f"{int(null['track_detect'].sum())}; "
        f"pull rms {float(np.sqrt(np.nanmean(null['track_pull']**2))):.2f}"
    )
    print(
        f"\nTiming per star: search {df['t_search_s'].median():.2f} s, "
        f"track {df['t_track_s'].median():.2f} s (median)"
    )


if __name__ == "__main__":
    main()
