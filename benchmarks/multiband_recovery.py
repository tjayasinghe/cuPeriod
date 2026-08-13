"""Multi-band vs single-band period recovery under Rubin/LSST-like cadence.

The motivating result for the multi-band methods: Rubin's alert-production study
found single-band Lomb-Scargle recovers periods poorly at LSST cadence while the
joint multi-band periodogram recovers most RR Lyrae. This benchmark reproduces
that comparison end-to-end with cuPeriod's native models on simulated RRab-like
stars observed with a sparse six-band cadence.

Setup (deliberately simple, stated so the numbers can be judged):

* 3-year campaign; per-band epochs land on random nights with a random
  within-night time (no rolling cadence, no lunation weighting).
* The per-band epoch share follows the WFD flavor (r/i deepest:
  u 6%, g 9%, r 26%, i 26%, z 17%, y 16%); the *total* number of epochs across
  all six bands is swept (30 / 60 / 120) to span the first survey years.
* RRab proxy: fundamental sine plus a phase-locked 0.35-amplitude second
  harmonic; period ~ U(0.35, 0.9) d; g amplitude ~ U(0.5, 1.0) mag scaled by
  band (u 1.05, g 1.0, r 0.72, i 0.57, z 0.53, y 0.48); photometric noise
  0.20 mag per point by default (--noise) — an r ~ 23 halo RR Lyrae in
  single Rubin visits, the population the multiband methods exist for.
* Recovery = the periodogram's top period within 1% of the truth, no harmonic
  credit.

Strategies compared per star, all on the same frequency grid:

  r-band GLS            single-band on the best-sampled band
  any-band GLS          recovered if ANY band's top period matches (optimistic
                        upper bound for per-band searching)
  multiband offsets     shared-phase (1, 0) model (cuPeriod default)
  multiband perband     chi2_0-weighted per-band combination (0, 1)
  multiband flex        regularized (1, 1) model

Writes results/multiband_recovery.parquet and prints the recovery table.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

from _common import RESULTS  # noqa: E402

SEED = 42
SPAN_DAYS = 3.0 * 365.25
BANDS = ("u", "g", "r", "i", "z", "y")
BAND_SHARE = {"u": 0.06, "g": 0.09, "r": 0.26, "i": 0.26, "z": 0.17, "y": 0.16}
AMP_SCALE = {"u": 1.05, "g": 1.00, "r": 0.72, "i": 0.57, "z": 0.53, "y": 0.48}
MEAN_MAG = {"u": 23.4, "g": 22.8, "r": 22.5, "i": 22.4, "z": 22.4, "y": 22.3}
NOISE_MAG = 0.20
HARMONIC_FRACTION = 0.35
EPOCH_BUDGETS = (30, 60, 120)
REL_TOL = 0.01


def simulate_star(
    rng: np.random.Generator, total_epochs: int, noise: float
) -> tuple[cup.MultiBandLightCurve, float]:
    """One RRab-like star on a sparse six-band cadence; returns (bands, period)."""
    period = float(rng.uniform(0.35, 0.9))
    amp_g = float(rng.uniform(0.5, 1.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    bands: dict[str, cup.LightCurve] = {}
    for band in BANDS:
        n = max(3, int(round(total_epochs * BAND_SHARE[band])))
        nights = rng.integers(0, int(SPAN_DAYS), size=n)
        t = np.sort(nights + rng.uniform(0.05, 0.45, size=n))
        amp = amp_g * AMP_SCALE[band]
        signal = amp * np.sin(2 * np.pi * t / period + phase)
        signal += HARMONIC_FRACTION * amp * np.sin(
            2 * (2 * np.pi * t / period + phase)
        )
        err = np.full(n, noise)
        mag = MEAN_MAG[band] + signal + rng.normal(0.0, noise, size=n)
        bands[band] = cup.LightCurve.from_arrays(t, mag, err)
    return cup.MultiBandLightCurve.from_light_curves(bands), period


def top_period(power: np.ndarray, frequency: np.ndarray) -> float:
    return float(1.0 / frequency[int(np.argmax(power))])


def recovered(p_found: float, p_true: float) -> bool:
    return abs(p_found / p_true - 1.0) <= REL_TOL


def run(n_stars: int, backend: str, noise: float) -> pd.DataFrame:
    from cuperiod.multiband.gls_mb import gls_multiband_power

    grid = cup.GridSpec(
        kind="frequency",
        values=np.arange(0.05, 3.5, 1.0 / (5.0 * SPAN_DAYS)),
        uniform=True,
    )
    frequency = grid.frequency
    settings = {
        "offsets": cup.GLSSettings(fap_method="none"),
        "perband": cup.GLSSettings(mb_model="perband", fap_method="none"),
        "flex": cup.GLSSettings(mb_model="flex", fap_method="none"),
    }
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    for budget in EPOCH_BUDGETS:
        for star in range(n_stars):
            mblc, p_true = simulate_star(rng, budget, noise)
            row: dict[str, object] = {
                "star": star, "epochs": budget, "p_true": p_true,
            }
            band_hits = {}
            for band, lc in mblc.bands.items():
                # A band too sparse to fit alone counts as a non-recovery for
                # the single-band strategies -- that sparsity is the point.
                try:
                    pg = cup.periodogram(
                        lc, "GLS", backend=backend, grid=grid,
                        settings=settings["offsets"],
                    )
                    assert isinstance(pg, cup.Periodogram)
                    band_hits[band] = recovered(
                        top_period(pg.power, frequency), p_true
                    )
                except cup.InsufficientDataError:
                    band_hits[band] = False
            row["r_band"] = band_hits["r"]
            row["any_band"] = any(band_hits.values())
            for model, s in settings.items():
                pg_mb = gls_multiband_power(grid, mblc, s, backend)
                row[f"mb_{model}"] = recovered(
                    top_period(pg_mb.power, frequency), p_true
                )
            rows.append(row)
        done = len(rows)
        print(f"  budget {budget}: {done} rows, {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stars", type=int, default=250)
    parser.add_argument("--backend", default="finufft")
    parser.add_argument("--noise", type=float, default=NOISE_MAG,
                        help="per-point photometric noise (mag)")
    args = parser.parse_args()

    print(f"multiband recovery: {args.n_stars} stars x {EPOCH_BUDGETS} epochs, "
          f"noise {args.noise} mag, backend {args.backend}")
    df = run(args.n_stars, args.backend, args.noise)
    out = RESULTS / "multiband_recovery.parquet"
    df.to_parquet(out)
    print(f"wrote {out}")

    strategies = ["r_band", "any_band", "mb_perband", "mb_flex", "mb_offsets"]
    summary = df.groupby("epochs")[strategies].mean().T
    summary.index.name = "strategy"
    print("\nRecovery fraction (top period within 1%, no harmonic credit):\n")
    header = "| strategy | " + " | ".join(
        f"{e} epochs" for e in summary.columns
    ) + " |"
    print(header)
    print("|" + "---|" * (len(summary.columns) + 1))
    for name, vals in summary.iterrows():
        cells = " | ".join(f"{v:.1%}" for v in vals)
        print(f"| {name} | {cells} |")


if __name__ == "__main__":
    main()
