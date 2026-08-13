"""Download the SDSS Stripe 82 RR Lyrae (Sesar et al. 2010) multi-band light curves
and bundle a deterministic subset into benchmarks/dataset/s82_rrlyrae.parquet.

Why this dataset
----------------
Sesar et al. 2010 (ApJ 708, 717) published ugriz light curves for 483 RR Lyrae
in SDSS Stripe 82, each with a precise literature period from a ~10-year
baseline. It is the canonical real multi-band validation set: VanderPlas &
Ivezic 2015 -- whose shared-phase model is cuPeriod's default ``offsets``
multi-band GLS -- developed and demonstrated their method on exactly these
stars, as does the reference ``gatspy`` package. Five real bands, real SDSS
cadence (including the 1-day alias structure of ground-based data), and known
answers make it the right target for validating every multi-band method.

Provenance
----------
The original data host (B. Sesar's MPIA page, ``www.mpia.de/~bsesar/S82_RRLyr``,
the URL hardcoded in released gatspy 0.3) is dead. The maintained mirror is the
astroML organization's data repository, which gatspy's development branch now
points at:

    https://github.com/astroML/astroML-data/tree/main/datasets/S82_RRLyr

Files used (downloaded into benchmarks/data/s82/, gitignored):

* ``table1.tar.gz`` (~1.2 MB) -- one whitespace file per star,
  ``table1/<id>.dat``: columns ``RA DEC`` then ``(MJD mag err)`` for each of
  u, g, r, i, z (17 columns). Missing observations carry the sentinel
  ``-99.99`` in all three fields of that band.
* ``table2.dat.gz`` (~64 KB) -- per-star fit parameters; columns used here are
  ``id``, ``type`` (``ab``/``c``) and ``P`` (period, days).

Cite Sesar et al. 2010, ApJ 708, 717 when using these data.

Subset
------
The committed bundle keeps the first ``--n-stars`` (default 100) stars by
ascending Sesar ID -- a deterministic, selection-bias-free cut (no filtering on
photometric quality or on how well any method performs). ``--all`` bundles all
483. Schema (one row per star and band):

    sesar_id, rrl_type, p_sesar, band, n_det, baseline, mjd[], mag[], mag_err[]

Run (main .venv; needs network on first run only)
-------------------------------------------------
    .venv/Scripts/python.exe benchmarks/dataset/download_s82_rrlyrae.py
"""

from __future__ import annotations

import argparse
import gzip
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATASET_DIR.parent / "data" / "s82"
OUTPUT_PARQUET = DATASET_DIR / "s82_rrlyrae.parquet"

BASE_URL = "https://github.com/astroML/astroML-data/raw/main/datasets/S82_RRLyr/"
FILES = ("table1.tar.gz", "table2.dat.gz")

BANDS = ("u", "g", "r", "i", "z")
SENTINEL = -99.99
N_STARS_DEFAULT = 100

# Sanity anchors from gatspy's doctests (gatspy.datasets.rrlyrae): the total
# star count, a handful of ids that must be present (gatspy lists them in tar
# order, so only membership is checked), and the first-epoch photometry of one
# star, which catches column-layout mistakes.
EXPECTED_N_STARS = 483
EXPECTED_IDS_PRESENT = [1013184, 1019544, 1027882, 1052471, 1056152]
EXPECTED_FIRST_STAR = {  # band -> (mjd[0], mag[0]) for star 1013184
    "u": (51081.347856, 18.702),
    "g": (51081.349522, 17.553),
    "r": (51081.346189, 17.236),
    "i": (51081.347022, 17.124),
}


def download(force: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = CACHE_DIR / name
        if target.exists() and not force:
            print(f"cached: {target}")
            continue
        url = BASE_URL + name
        print(f"downloading {url} ...")
        with urllib.request.urlopen(url) as resp:
            target.write_bytes(resp.read())
        print(f"  -> {target} ({target.stat().st_size:,} bytes)")


def load_periods() -> pd.DataFrame:
    """Table 2: Sesar ID, RRL subtype (ab/c) and period in days."""
    with gzip.open(CACHE_DIR / "table2.dat.gz", "rt") as fh:
        raw = np.loadtxt(
            fh,
            dtype={"names": ("id", "type", "P"), "formats": ("i8", "U2", "f8")},
            usecols=(0, 1, 2),
        )
    return pd.DataFrame({
        "sesar_id": raw["id"], "rrl_type": raw["type"], "p_sesar": raw["P"],
    })


def load_light_curves() -> dict[int, dict[str, np.ndarray]]:
    """Table 1: per-star (17-column) files -> {id: {band: (n, 3) [mjd, mag, err]}}."""
    stars: dict[int, dict[str, np.ndarray]] = {}
    with tarfile.open(CACHE_DIR / "table1.tar.gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) != 2 or not parts[1].endswith(".dat"):
                continue
            sesar_id = int(parts[1].removesuffix(".dat"))
            fh = tar.extractfile(member)
            assert fh is not None
            data = np.loadtxt(fh, dtype=np.float64)
            data = np.atleast_2d(data)
            per_band: dict[str, np.ndarray] = {}
            for k, band in enumerate(BANDS):
                block = data[:, 2 + 3 * k : 5 + 3 * k]  # mjd, mag, err
                good = (
                    np.all(block != SENTINEL, axis=1)
                    & np.all(np.isfinite(block), axis=1)
                    & (block[:, 2] > 0)
                )
                per_band[band] = block[good][np.argsort(block[good][:, 0])]
            stars[sesar_id] = per_band
    return stars


def sanity_check(stars: dict[int, dict[str, np.ndarray]]) -> None:
    ids = sorted(stars)
    print(f"parsed {len(ids)} stars; first five ids: {ids[:5]}")
    if len(ids) != EXPECTED_N_STARS:
        sys.exit(f"FATAL: expected {EXPECTED_N_STARS} stars, parsed {len(ids)}")
    if not set(EXPECTED_IDS_PRESENT) <= set(ids):
        sys.exit("FATAL: known Sesar ids are missing from the archive")
    first = stars[EXPECTED_IDS_PRESENT[0]]
    for band, (mjd0, mag0) in EXPECTED_FIRST_STAR.items():
        got = first[band][0]
        if abs(got[0] - mjd0) > 1e-6 or abs(got[1] - mag0) > 1e-3:
            sys.exit(
                f"FATAL: star {EXPECTED_IDS_PRESENT[0]} band {band} first epoch "
                f"is ({got[0]}, {got[1]}), expected ({mjd0}, {mag0}) -- "
                "column layout mismatch"
            )
    print("column-layout sanity check against gatspy doctest values: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stars", type=int, default=N_STARS_DEFAULT,
                        help="bundle the first N stars by ascending Sesar ID")
    parser.add_argument("--all", action="store_true",
                        help="bundle all 483 stars (overrides --n-stars)")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    periods = load_periods()
    stars = load_light_curves()
    sanity_check(stars)

    with_period = sorted(set(stars) & set(periods["sesar_id"]))
    missing = sorted(set(stars) - set(periods["sesar_id"]))
    if missing:
        print(f"note: {len(missing)} stars lack a table2 period and are skipped: "
              f"{missing}")
    n_keep = len(with_period) if args.all else min(args.n_stars, len(with_period))
    keep = with_period[:n_keep]

    period_of = periods.set_index("sesar_id")
    rows = []
    for sesar_id in keep:
        for band in BANDS:
            block = stars[sesar_id][band]
            rows.append({
                "sesar_id": sesar_id,
                "rrl_type": str(period_of.loc[sesar_id, "rrl_type"]),
                "p_sesar": float(period_of.loc[sesar_id, "p_sesar"]),
                "band": band,
                "n_det": int(len(block)),
                "baseline": float(block[-1, 0] - block[0, 0]) if len(block) else 0.0,
                "mjd": block[:, 0].tolist(),
                "mag": block[:, 1].tolist(),
                "mag_err": block[:, 2].tolist(),
            })
    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT_PARQUET, index=False)

    meta = out.drop_duplicates("sesar_id")
    print(f"\nwrote {OUTPUT_PARQUET}: {len(meta)} stars x {len(BANDS)} bands")
    print(f"  types: {meta['rrl_type'].value_counts().to_dict()}")
    print(f"  periods: {meta['p_sesar'].min():.4f} - {meta['p_sesar'].max():.4f} d")
    per_band = out.groupby("band")["n_det"].median().astype(int).to_dict()
    print(f"  median epochs per band: {per_band}")
    print(f"  file size: {OUTPUT_PARQUET.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
