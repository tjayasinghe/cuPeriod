"""Validate every multi-band method on real SDSS Stripe 82 RR Lyrae.

The companion to ``multiband_recovery.py`` (which is simulated): the same
question -- do the joint multi-band searches find the right period? -- asked of
**real survey photometry with literature answers**. The data are the Sesar et
al. 2010 (ApJ 708, 717) Stripe 82 RR Lyrae: real SDSS ugriz cadence over ~10
years, including the 1-day alias structure of ground-based sampling, with
periods known from the discovery paper. This is the dataset VanderPlas &
Ivezic 2015 built the shared-phase multiband model on -- the model that ships
as cuPeriod's default ``offsets`` GLS -- so it doubles as an end-to-end check
against that paper's headline result. Bundle: ``dataset/s82_rrlyrae.parquet``
(first 100 stars by Sesar ID, no quality selection; see
``dataset/download_s82_rrlyrae.py``).

Setup
-----
* Blind search, identical for every star: periods 0.15-1.2 d (brackets the
  RR Lyrae instability strip; the bundle spans 0.26-0.91 d), uniform frequency
  grid at 5 samples per Rayleigh width of each star's ~3200 d baseline
  (~96k trial frequencies). BLS builds its native duration/period grid inside
  the same period window, as elsewhere in the suite.
* Every multi-band method at default settings: GLS in all three models
  (``offsets``, ``perband``, ``flex``), PDM, conditional entropy,
  string-length, MHAOV, SuperSmoother, and BLS (transit-shaped by design; it
  is included for completeness and read with that caveat).
* Single-band GLS on each of u,g,r,i,z as the baseline the joint methods must
  beat; a band with too few points counts as a non-recovery.
* Scoring follows the suite: **strict** = top period within 1% of Sesar's, no
  harmonic credit (as in ``multiband_recovery.py``); **harmonic-aware** = the
  top period matches up to a small-integer harmonic ratio within 2%
  (``_common.period_match``). Fold-family statistics (PDM/CE/SL and
  SuperSmoother) are genuinely periodic at integer multiples of the true
  period, so for them the strict column mostly measures multiple/submultiple
  picks rather than wrong periods -- report both, judge with both.

Backends: the ``cpu`` pass (numba/finufft tiers) is the scored reference; a
``gpu`` pass records its own top periods and timings so agreement and speed
are reported from the same run. Per-call timing includes any per-call GPU
plan/kernel setup (a warm-up pass absorbs one-time JIT/compile); the batch
runner amortizes more via engine reuse.

Writes results/multiband_real.parquet and prints the summary tables.

Run (main .venv, GPU optional):

    .venv/Scripts/python.exe benchmarks/multiband_real.py
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import cuperiod as cup  # noqa: E402

from _common import DATASET, RESULTS, period_match  # noqa: E402

S82_PARQUET = DATASET / "s82_rrlyrae.parquet"
P_MIN_DAYS, P_MAX_DAYS = 0.15, 1.2
SAMPLES_PER_PEAK = 5
REL_TOL_STRICT = 0.01
BANDS = ("u", "g", "r", "i", "z")

_GLS_BASE = {"fap_method": "none"}
MB_MODELS: dict[str, tuple[str, object]] = {
    "gls_offsets": ("GLS", cup.GLSSettings(**_GLS_BASE)),
    "gls_perband": ("GLS", cup.GLSSettings(mb_model="perband", **_GLS_BASE)),
    "gls_flex": ("GLS", cup.GLSSettings(mb_model="flex", **_GLS_BASE)),
    "pdm": ("PDM", cup.PDMSettings()),
    "ce": ("CE", cup.CESettings()),
    "stringlength": ("STRINGLENGTH", cup.StringLengthSettings()),
    "mhaov": ("MHAOV", cup.MHAOVSettings()),
    "supersmoother": ("SUPERSMOOTHER", cup.SuperSmootherSettings()),
    "bls": ("BLS", cup.BLSSettings(
        min_period_days=P_MIN_DAYS, max_period_days=P_MAX_DAYS,
    )),
}
# BLS owns its period/duration grid; everything else shares the star's grid.
NATIVE_GRID_MODELS = frozenset({"bls"})


def load_stars() -> list[tuple[int, str, float, cup.MultiBandLightCurve]]:
    df = pd.read_parquet(S82_PARQUET)
    stars = []
    for sesar_id, group in df.groupby("sesar_id", sort=True):
        bands = {
            str(r.band): cup.LightCurve.from_arrays(
                np.asarray(r.mjd, dtype=np.float64),
                np.asarray(r.mag, dtype=np.float64),
                np.asarray(r.mag_err, dtype=np.float64),
            )
            for r in group.itertuples()
            if len(r.mjd) > 0
        }
        first = group.iloc[0]
        stars.append((
            int(sesar_id), str(first.rrl_type), float(first.p_sesar),
            cup.MultiBandLightCurve.from_light_curves(bands),
        ))
    return stars


def star_grid(mblc: cup.MultiBandLightCurve) -> cup.GridSpec:
    time_all = mblc.finite().stacked()[0]
    baseline = float(time_all.max() - time_all.min())
    df = 1.0 / (SAMPLES_PER_PEAK * baseline)
    values = np.arange(1.0 / P_MAX_DAYS, 1.0 / P_MIN_DAYS, df)
    return cup.GridSpec(kind="frequency", values=values, uniform=True)


def top_period(pg: cup.Periodogram) -> float:
    power = np.asarray(pg.power, dtype=np.float64)
    idx = int(
        np.argmin(power) if pg.objective_sense == "min" else np.argmax(power)
    )
    return float(1.0 / pg.frequency[idx])


def run_star(
    model: str,
    mblc: cup.MultiBandLightCurve,
    grid: cup.GridSpec,
    backend: str,
) -> tuple[float, float]:
    """Return (top period, wall seconds) for one model on one star."""
    method, settings = MB_MODELS[model]
    t0 = time.perf_counter()
    pg = cup.periodogram(
        mblc, method, backend=backend,
        grid=None if model in NATIVE_GRID_MODELS else grid,
        settings=settings,
    )
    seconds = time.perf_counter() - t0
    assert isinstance(pg, cup.Periodogram)
    return top_period(pg), seconds


def run(backends: list[str], limit: int | None) -> pd.DataFrame:
    stars = load_stars()
    if limit is not None:
        stars = stars[:limit]
    grids = {sid: star_grid(mblc) for sid, _, _, mblc in stars}
    n_freq = int(np.median([g.frequency.size for g in grids.values()]))
    print(f"{len(stars)} stars, blind {P_MIN_DAYS}-{P_MAX_DAYS} d "
          f"(~{n_freq} trial frequencies), backends {backends}")

    # Absorb one-time JIT/kernel-compile cost so per-star timings are steady
    # state. The first star is then re-run inside the timed loop.
    sid0, _, _, mblc0 = stars[0]
    for backend in backends:
        for model in MB_MODELS:
            run_star(model, mblc0, grids[sid0], backend)

    rows: list[dict[str, object]] = []
    t_start = time.perf_counter()
    for backend in backends:
        for k, (sesar_id, rrl_type, p_true, mblc) in enumerate(stars, 1):
            for model in MB_MODELS:
                p_top, seconds = run_star(model, mblc, grids[sesar_id], backend)
                ok, ratio = period_match(p_top, p_true)
                rows.append({
                    "sesar_id": sesar_id, "rrl_type": rrl_type,
                    "p_true": p_true, "model": model, "backend": backend,
                    "p_top": p_top, "seconds": seconds,
                    "strict": abs(p_top / p_true - 1.0) <= REL_TOL_STRICT,
                    "harmonic_ok": ok, "ratio": ratio,
                })
            if backend == backends[0]:
                # Single-band GLS baseline (scored pass only).
                for band in BANDS:
                    lc = mblc.bands.get(band)
                    p_top = np.nan
                    if lc is not None:
                        try:
                            pg = cup.periodogram(
                                lc, "GLS", backend=backend,
                                grid=grids[sesar_id],
                                settings=MB_MODELS["gls_offsets"][1],
                            )
                            assert isinstance(pg, cup.Periodogram)
                            p_top = top_period(pg)
                        except cup.InsufficientDataError:
                            pass
                    ok, ratio = period_match(p_top, p_true)
                    rows.append({
                        "sesar_id": sesar_id, "rrl_type": rrl_type,
                        "p_true": p_true, "model": f"single_{band}",
                        "backend": backend, "p_top": p_top, "seconds": np.nan,
                        "strict": bool(
                            np.isfinite(p_top)
                            and abs(p_top / p_true - 1.0) <= REL_TOL_STRICT
                        ),
                        "harmonic_ok": ok, "ratio": ratio,
                    })
            if k % 20 == 0:
                print(f"  {backend}: {k}/{len(stars)} stars, "
                      f"{time.perf_counter() - t_start:.0f}s", flush=True)
        # Checkpoint after each backend pass so a crash or kill in a later
        # pass cannot lose the scored results.
        pd.DataFrame(rows).to_parquet(
            RESULTS / "multiband_real.parquet", index=False
        )
        print(f"  checkpoint: {backend} pass written "
              f"({len(rows)} rows)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, scored_backend: str) -> None:
    scored = df[df["backend"] == scored_backend]
    n_stars = scored["sesar_id"].nunique()

    single = scored[scored["model"].str.startswith("single_")]
    any_band = single.groupby("sesar_id")["strict"].any()
    print(f"\nSingle-band GLS baseline ({n_stars} stars, strict within 1%):")
    for band in BANDS:
        frac = single[single["model"] == f"single_{band}"]["strict"].mean()
        print(f"  {band}: {frac:.1%}")
    print(f"  any band: {any_band.mean():.1%}")

    mb = scored[~scored["model"].str.startswith("single_")]
    print(f"\nMulti-band methods ({n_stars} stars):")
    print("| model | strict (1%) | harmonic-aware (2%) | median |dP|/P "
          "| median s/star |")
    print("|---|---|---|---|---|")
    for model in MB_MODELS:
        g = mb[mb["model"] == model]
        hits = g[g["strict"]]
        frac_err = np.median(np.abs(hits["p_top"] / hits["p_true"] - 1.0))
        print(f"| {model} | {g['strict'].mean():.1%} "
              f"| {g['harmonic_ok'].mean():.1%} "
              f"| {frac_err:.1e} | {g['seconds'].median():.3f} |")

    other = [b for b in df["backend"].unique() if b != scored_backend]
    for backend in other:
        alt = df[(df["backend"] == backend)
                 & ~df["model"].str.startswith("single_")]
        merged = alt.merge(
            mb[["sesar_id", "model", "p_top"]], on=["sesar_id", "model"],
            suffixes=("", "_ref"),
        )
        agree = (np.abs(merged["p_top"] / merged["p_top_ref"] - 1.0)
                 <= 1e-4).mean()
        print(f"\n{backend} backend: top period agrees with {scored_backend} "
              f"for {agree:.1%} of runs; median s/star by model:")
        for model in MB_MODELS:
            sec = alt[alt["model"] == model]["seconds"].median()
            ref = mb[mb["model"] == model]["seconds"].median()
            print(f"  {model}: {sec:.3f}s (vs {ref:.3f}s {scored_backend})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="cpu,gpu",
                        help="comma-separated; first one is the scored pass")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N stars (smoke test)")
    args = parser.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    df = run(backends, args.limit)
    out = RESULTS / "multiband_real.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}", flush=True)
    summarize(df, backends[0])
    print("\nMULTIBAND_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
