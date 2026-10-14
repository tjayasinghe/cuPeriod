"""Injection-recovery for the cyclic O-C: light-time orbits shorter than the baseline.

A third body makes the O-C of a close binary periodic, with the light-travel
time across the outer orbit as its amplitude. When that outer period is
comparable to or shorter than the baseline the O-C is not monotonic, and the
number a quadratic ephemeris returns for it is a baseline-weighted average of
a wobble — its sign as much as its value depends on which part of the cycle
the data cover. This benchmark plants such orbits and asks:

* how often the parabola alone reports a "significant" dP/dt that was never
  there (the false secular rate a cyclic variation buys);
* whether :func:`cuperiod.ltte_search` recovers the planted outer period and
  amplitude, and whether the joint fit's secular rate is null as planted;
* how the answers depend on the outer period in units of the baseline, and
  what the false-alarm rate of the cycle detection is on stars with no cycle.

Setup (stated so the numbers can be judged):

* 3000-day baseline, 1500 points, irregular sampling; amplitudes 0.30 + 0.10
  mag (first harmonic), 0.03 mag noise — a bright ground-based survey star,
  tracked in 30 segments (50 points, ~100 days each).
* Inner periods ~ U(0.3, 0.9) d; no secular drift is planted.
* Outer period ``P2 = r * baseline`` with ``r ~ U(0.25, 1.5)`` (Hong et al.
  2022's OGLE triples span 5-14 yr on a 15-yr baseline, i.e. r = 0.3-0.9);
  light-time semi-amplitude ~ U(120, 600) s, the range of their sinusoids;
  random phase. A separate null population has no orbit at all.
* Detection = dBIC > 10 (positive favours the cycle). Recovery = the outer
  period within 10% and the amplitude within 3 sigma of the planted values,
  scored only where the baseline covers at least one cycle (r <= 1); beyond
  that a light-time orbit and a parabola are nearly degenerate, which is the
  regime the docs' "distant companion" caution already covers.

Writes results/pdot_ltte_recovery.parquet and prints the summary tables.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from _common import RESULTS  # noqa: E402

import cuperiod as cup  # noqa: E402

# The independent Irwin (1952) delay of the test suite, so the planted orbit
# shares no code with the implementation under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synth import ltte_delay_reference  # noqa: E402

SEED = 42
SPAN = 3000.0
N_POINTS = 1500
NOISE = 0.03
AMP, AMP2 = 0.30, 0.10
RATIO_RANGE = (0.25, 1.5)
AMPLITUDE_RANGE_S = (120.0, 600.0)
N_SEGMENTS = 30
DELTA_BIC_DETECT = 10.0
RATIO_BINS = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5))


def simulate(
    rng: np.random.Generator, ratio: float, amplitude_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """One star with a light-time orbit; returns (t, mag, err, truth)."""
    period = float(rng.uniform(0.3, 0.9))
    outer = ratio * SPAN
    amplitude = amplitude_s / 86400.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.sort(rng.uniform(0.0, SPAN, N_POINTS)) + 2455000.0
    dt = t - t.mean()
    delay = ltte_delay_reference(dt, outer, amplitude, phase) if amplitude > 0 else 0.0
    retarded = dt - delay
    f0 = 1.0 / period
    mag = (
        14.0
        + AMP * np.sin(2 * np.pi * f0 * retarded + rng.uniform(0, 2 * np.pi))
        + AMP2 * np.sin(4 * np.pi * f0 * retarded + rng.uniform(0, 2 * np.pi))
        + rng.normal(0.0, NOISE, N_POINTS)
    )
    truth = {
        "period": period,
        "outer_period": outer,
        "amplitude_s": amplitude_s,
        "ratio": ratio,
    }
    return t, mag, np.full(N_POINTS, NOISE), truth


def run_one(
    t: np.ndarray, mag: np.ndarray, err: np.ndarray, truth: dict[str, float]
) -> dict[str, float | bool]:
    settings = cup.PdotSettings(n_segments=N_SEGMENTS)
    t0 = time.perf_counter()
    track = cup.phase_track((t, mag, err), period=truth["period"], settings=settings)
    res = track.ltte_search(settings=settings)
    elapsed = time.perf_counter() - t0
    best = res.best
    quad_snr = abs(track.pdot) / track.pdot_err_scaled if track.pdot_err_scaled else 0.0
    sec_snr = abs(best.pdot) / best.pdot_err_scaled if best.pdot_err_scaled else 0.0
    return {
        **truth,
        "n_segments": track.n_segments,
        "quad_pdot_snr": quad_snr,
        "quad_spurious_3sigma": quad_snr >= 3.0,
        "ltte_outer_period": best.outer_period,
        "ltte_amplitude_s": best.amplitude_s,
        "ltte_amplitude_err_s": best.amplitude_err_s * best.error_inflation,
        "ltte_delta_bic": res.delta_bic,
        "ltte_detected": res.delta_bic > DELTA_BIC_DETECT,
        "ltte_at_grid_edge": bool(best.at_grid_edge),
        "period_recovered": abs(best.outer_period - truth["outer_period"])
        <= 0.10 * truth["outer_period"],
        "amplitude_recovered": abs(best.amplitude_s - truth["amplitude_s"])
        <= 3.0 * best.amplitude_err_s * best.error_inflation,
        "secular_pdot_snr": sec_snr,
        "secular_spurious_3sigma": sec_snr >= 3.0,
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--nulls", type=int, default=40)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(args.trials):
        ratio = float(rng.uniform(*RATIO_RANGE))
        amp_s = float(rng.uniform(*AMPLITUDE_RANGE_S))
        rows.append({**run_one(*simulate(rng, ratio, amp_s)), "planted": True})
    for _ in range(args.nulls):
        rows.append({**run_one(*simulate(rng, 1.0, 0.0)), "planted": False})
    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS / "pdot_ltte_recovery.parquet", index=False)

    planted = table[table["planted"]]
    nulls = table[~table["planted"]]
    print(
        f"{len(planted)} planted light-time orbits, {len(nulls)} nulls; "
        f"{table['seconds'].median():.2f} s per star (track + search)\n"
    )
    print(
        f"{'P2 / baseline':<16}{'n':>4}{'parabola >=3 sigma':>20}"
        f"{'cycle dBIC>10':>15}{'P2 within 10%':>15}{'A within 3 sigma':>18}"
        f"{'joint >=3 sigma':>17}"
    )
    for lo, hi in RATIO_BINS:
        sel = planted[(planted["ratio"] >= lo) & (planted["ratio"] < hi)]
        if sel.empty:
            continue
        print(
            f"[{lo:.2f}, {hi:.2f})     {len(sel):>4}"
            f"{sel['quad_spurious_3sigma'].mean():>20.0%}"
            f"{sel['ltte_detected'].mean():>15.0%}"
            f"{sel['period_recovered'].mean():>15.0%}"
            f"{sel['amplitude_recovered'].mean():>18.0%}"
            f"{sel['secular_spurious_3sigma'].mean():>17.0%}"
        )
    covered = planted[planted["ratio"] <= 1.0]
    print(
        f"\ncovered cycles (r <= 1): outer period within 10% in "
        f"{covered['period_recovered'].mean():.0%}, amplitude within 3 sigma in "
        f"{covered['amplitude_recovered'].mean():.0%}, cycle detected in "
        f"{covered['ltte_detected'].mean():.0%}"
    )
    quad_false = planted["quad_spurious_3sigma"].mean()
    joint_false = planted["secular_spurious_3sigma"].mean()
    print(
        f"parabola alone: a >= 3 sigma 'dP/dt' on {quad_false:.0%} of the "
        f"planted stars (none has a secular rate); {joint_false:.0%} once "
        "the cycle is fitted jointly"
    )
    print(
        f"nulls: cycle 'detected' (dBIC > 10) on {nulls['ltte_detected'].mean():.0%}; "
        f"parabola >= 3 sigma on {nulls['quad_spurious_3sigma'].mean():.0%}"
    )
    print(f"\n-> {RESULTS / 'pdot_ltte_recovery.parquet'}")


if __name__ == "__main__":
    main()
