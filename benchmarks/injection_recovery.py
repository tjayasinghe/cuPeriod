"""Synthetic signal injection-recovery test (VanderPlas 2018-style sensitivity curves).

Complements ``validate_periodograms.py`` (real, bright, well-established stars only)
with a controlled sweep: known signal, known period, tunable signal-to-noise, drawn
onto *real* observation cadences from the bundled validation set so the irregular
sampling / gaps of ground-based photometry are represented realistically.

Three signal models, each run through the methods it is diagnostic for:

  sinusoid (+ mild 2nd harmonic)     -> GLS, MHAOV, PDM, CE, STRINGLENGTH
  eclipsing-binary-like fold         -> PDM, CE, STRINGLENGTH, BLS
    (two unequal narrow Gaussian dips per cycle)
  box transit                        -> BLS, TLS

For each (method, signal_type, snr) cell we run ~10 trials (period + phase varied,
fixed seed) and score recovery with the same harmonic-aware 2% tolerance used by the
main validation suite (:func:`_common.period_match`).

Writes results/injection_recovery.parquet and prints a recovery-fraction summary.
Runtime is self-budgeted to stay under ~15 minutes: a quick per-method timing probe
scales the trial count down if the naive grid would run long.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

from _common import RESULTS, fine_grid, load_dataset, period_match  # noqa: E402

SEED = 42
TIME_BUDGET_S = 15 * 60
N_CADENCES = 5
N_TRIALS_DEFAULT = 10
MIN_TRIALS = 3
# amplitude / depth in units of the injected noise sigma: spans clear non-detection
# to easy, unambiguous detection.
SNR_LEVELS = (0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
MAG_SIGMA = 0.02  # typical ASAS-SN g-band photometric precision

FREQ_SETTINGS = {
    "GLS": lambda: cup.GLSSettings(fap_method="none"),
    "PDM": lambda: cup.PDMSettings(),
    "CE": lambda: cup.CESettings(),
    "STRINGLENGTH": lambda: cup.StringLengthSettings(),
    "MHAOV": lambda: cup.MHAOVSettings(),
}
SENSE = {"GLS": "max", "MHAOV": "max", "PDM": "min", "CE": "min", "STRINGLENGTH": "min"}

SIGNAL_METHODS = {
    "sinusoid": ["GLS", "MHAOV", "PDM", "CE", "STRINGLENGTH"],
    "eclipse": ["PDM", "CE", "STRINGLENGTH", "BLS"],
    "transit": ["BLS", "TLS"],
}
PERIOD_RANGE = {  # days
    "sinusoid": (0.3, 15.0),
    "eclipse": (0.4, 10.0),
    "transit": (1.0, 20.0),
}
# common day-aliases to steer clear of when drawing a trial period
ALIAS_PERIODS = (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0)
ALIAS_TOL = 0.03


# --------------------------------------------------------------------------
# cadences: real observation times from the bundled validation set
def pick_cadences(meta: pd.DataFrame, jd: dict, n: int = N_CADENCES) -> list[dict]:
    """~n cadences spanning the n_det range of the real validation light curves."""
    quantiles = np.linspace(0.1, 0.9, n)
    targets = meta["n_det"].quantile(quantiles).to_numpy()
    chosen, used = [], set()
    for target in targets:
        order = (meta["n_det"] - target).abs().sort_values().index
        for idx in order:
            sid = str(meta.loc[idx, "asas_sn_id"])
            if sid not in used:
                used.add(sid)
                chosen.append(sid)
                break
    cadences = []
    for sid in chosen:
        t = np.sort(np.asarray(jd[sid], dtype=np.float64))
        cadences.append({"id": sid, "t": t, "n_det": int(t.size),
                         "baseline": float(t.max() - t.min())})
    return cadences


# --------------------------------------------------------------------------
# signal models (all on a magnitude-like scale: positive delta = fainter)
def draw_period(rng: np.random.Generator, signal_type: str, baseline: float,
                 max_frac: float = 0.33) -> float:
    lo, hi = PERIOD_RANGE[signal_type]
    hi = min(hi, baseline * max_frac)
    if hi <= lo:
        hi = lo * 1.5
    for _ in range(50):
        p = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if all(abs(p / a - 1.0) > ALIAS_TOL for a in ALIAS_PERIODS):
            return p
    return p


def make_sinusoid(t, p, phase, amp, sigma, rng):
    ang = 2.0 * np.pi * (t / p + phase)
    y = amp * np.sin(ang) + 0.3 * amp * np.sin(2.0 * ang + 0.7)
    return y + rng.normal(0.0, sigma, size=t.size)


def _gaussian_dip(phase, center, width):
    d = np.mod(phase - center + 0.5, 1.0) - 0.5
    return np.exp(-0.5 * (d / width) ** 2)


def make_eclipse(t, p, phase0, depth, sigma, rng):
    ph = np.mod(t / p - phase0, 1.0)
    y = depth * _gaussian_dip(ph, 0.0, 0.04) + 0.4 * depth * _gaussian_dip(ph, 0.5, 0.025)
    return y + rng.normal(0.0, sigma, size=t.size)


def make_transit(t, p, phase0, depth, sigma, rng, dur_frac=0.06):
    ph = np.mod(t / p - phase0, 1.0)
    d = np.minimum(ph, 1.0 - ph)
    y = np.where(d < dur_frac / 2.0, depth, 0.0)
    return y + rng.normal(0.0, sigma, size=t.size)


SIGNAL_MAKERS = {"sinusoid": make_sinusoid, "eclipse": make_eclipse, "transit": make_transit}


# --------------------------------------------------------------------------
# method runners -> best_period only (this is a recovery test, not a parity test)
def run_freq_method(method: str, t: np.ndarray, y: np.ndarray, p_true: float,
                    baseline: float, *, cap: int) -> float:
    grid = fine_grid(p_true, baseline, span=2.5, samples_per_peak=4, cap=cap)
    gspec = cup.GridSpec(kind="frequency", values=grid, uniform=True)
    res = cup.periodogram((t, y), method, backend="cpu", grid=gspec, settings=FREQ_SETTINGS[method]())
    sense = SENSE[method]
    idx = int(np.argmin(res.power) if sense == "min" else np.argmax(res.power))
    return 1.0 / grid[idx]


def run_bls(t: np.ndarray, y: np.ndarray, p_true: float) -> float:
    st = cup.BLSSettings(min_period_days=max(0.05, p_true / 2.0),
                         max_period_days=p_true * 2.0, grid_oversample=1)
    res = cup.periodogram((t, y), "BLS", backend="cpu", settings=st)
    return res.best_period()


def run_tls(t: np.ndarray, y: np.ndarray, p_true: float) -> float:
    st = cup.TLSSettings(min_period_days=max(0.05, p_true / 2.0), max_period_days=p_true * 2.0)
    res = cup.periodogram((t, y), "TLS", backend="cpu", settings=st)
    return res.best_period()


def run_method(method: str, t, y, p_true, baseline, *, cap: int) -> float:
    if method == "BLS":
        return run_bls(t, y, p_true)
    if method == "TLS":
        return run_tls(t, y, p_true)
    return run_freq_method(method, t, y, p_true, baseline, cap=cap)


# --------------------------------------------------------------------------
def one_trial_signal(signal_type, cadence, snr, rng, dur_frac=0.06):
    """Build one synthetic light curve for ``signal_type`` at the given SNR."""
    t = cadence["t"]
    baseline = cadence["baseline"]
    p_true = draw_period(rng, signal_type, baseline)
    phase0 = float(rng.uniform(0.0, 1.0))
    amp = snr * MAG_SIGMA
    maker = SIGNAL_MAKERS[signal_type]
    if signal_type == "transit":
        y = maker(t, p_true, phase0, amp, MAG_SIGMA, rng, dur_frac=dur_frac)
    else:
        y = maker(t, p_true, phase0, amp, MAG_SIGMA, rng)
    return t, y, p_true


def calibrate_grid_cap(cadences, rng) -> tuple[int, dict]:
    """Time one trial per method to pick a frequency-grid cap that fits the budget."""
    probe_cadence = max(cadences, key=lambda c: c["n_det"])
    timings = {}
    cap = 8000
    for signal_type, methods in SIGNAL_METHODS.items():
        t, y, p_true = one_trial_signal(signal_type, probe_cadence, snr=4.0, rng=rng)
        for method in methods:
            t0 = time.perf_counter()
            run_method(method, t, y, p_true, probe_cadence["baseline"], cap=cap)
            timings[(signal_type, method)] = time.perf_counter() - t0
    return cap, timings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT,
                   help="trials per (method, signal_type, snr) cell before time-budget "
                        "scale-down (default: %(default)s). Output is deterministic given "
                        "(seed, trials).")
    return p.parse_args()


def main():
    args = parse_args()
    n_trials_target = args.trials
    script_start = time.perf_counter()
    meta, jd, _mag, _err = load_dataset()
    cadences = pick_cadences(meta, jd, N_CADENCES)
    print("cadences:")
    for c in cadences:
        print(f"  {c['id']:>12s}  n_det={c['n_det']:5d}  baseline={c['baseline']:8.1f} d")

    rng_calib = np.random.default_rng(SEED)
    cap, timings = calibrate_grid_cap(cadences, rng_calib)
    print("\nper-trial timing probe (n_det={} cadence, cap={} grid points):".format(
        max(c["n_det"] for c in cadences), cap))
    for k, v in timings.items():
        print(f"  {k[0]:9s} {k[1]:12s} {v*1000:7.1f} ms")

    pairs = [(s, m) for s, ms in SIGNAL_METHODS.items() for m in ms]
    n_cells = len(pairs) * len(SNR_LEVELS)
    per_trial_avg = float(np.mean(list(timings.values())))
    est_total = per_trial_avg * n_cells * n_trials_target
    n_trials = n_trials_target
    if est_total > TIME_BUDGET_S:
        n_trials = max(MIN_TRIALS, int(n_trials_target * TIME_BUDGET_S / est_total))
        print(f"\nprojected {est_total:.0f}s > budget {TIME_BUDGET_S:.0f}s "
              f"-> scaling trials/cell {n_trials_target} -> {n_trials}")
    else:
        print(f"\nprojected {est_total:.0f}s <= budget {TIME_BUDGET_S:.0f}s -> "
              f"keeping {n_trials} trials/cell")
    print(f"grid: {len(SIGNAL_METHODS)} signals, {len(pairs)} method-signal pairs, "
          f"{len(SNR_LEVELS)} SNR levels, {n_trials} trials/cell, {len(cadences)} cadences "
          f"-> {len(pairs) * len(SNR_LEVELS) * n_trials} runs")

    rng = np.random.default_rng(SEED)
    rows = []
    t_start = time.perf_counter()
    n_done = 0
    n_total = len(pairs) * len(SNR_LEVELS) * n_trials
    for signal_type, methods in SIGNAL_METHODS.items():
        for snr in SNR_LEVELS:
            for trial in range(n_trials):
                cadence = cadences[rng.integers(0, len(cadences))]
                t, y, p_true = one_trial_signal(signal_type, cadence, snr, rng)
                for method in methods:
                    try:
                        p_best = run_method(method, t, y, p_true, cadence["baseline"], cap=cap)
                        ok, ratio = period_match(p_best, p_true)
                    except Exception as ex:
                        print(f"  FAIL {signal_type} {method} trial={trial} snr={snr}: "
                              f"{type(ex).__name__}: {ex}")
                        p_best, ok, ratio = np.nan, False, np.nan
                    rows.append(dict(
                        method=method, signal_type=signal_type, snr=float(snr), trial=trial,
                        true_period=p_true, best_period=p_best, recovered=bool(ok),
                        harmonic_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
                        cadence_id=cadence["id"], n_det=cadence["n_det"],
                    ))
                    n_done += 1
            elapsed = time.perf_counter() - t_start
            print(f"  [{n_done:5d}/{n_total}] {signal_type:9s} snr={snr:5.1f} "
                  f"done  ({elapsed:.0f}s elapsed)", flush=True)

    res = pd.DataFrame(rows)
    out_path = RESULTS / "injection_recovery.parquet"
    res.to_parquet(out_path, index=False)
    main_loop_elapsed = time.perf_counter() - t_start
    total_elapsed = time.perf_counter() - script_start
    print(f"\nwrote {out_path}  ({len(res)} rows, {main_loop_elapsed:.0f}s main loop, "
          f"{total_elapsed:.0f}s total incl. calibration)")

    # summary: recovery fraction per method x signal x snr
    summary = (res.groupby(["signal_type", "method", "snr"])["recovered"]
               .mean().reset_index().sort_values(["signal_type", "method", "snr"]))
    print("\nrecovery fraction (method x signal x SNR):")
    for (signal_type, method), sub in summary.groupby(["signal_type", "method"]):
        line = "  ".join(f"snr={r.snr:4.1f}:{r.recovered*100:3.0f}%" for r in sub.itertuples())
        print(f"  {signal_type:9s} {method:12s} {line}")


if __name__ == "__main__":
    main()
