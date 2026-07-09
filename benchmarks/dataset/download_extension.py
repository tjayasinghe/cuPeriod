"""Download additional ASAS-SN Sky Patrol variable stars to extend the real-data
validation dataset (benchmarks/dataset/light_curves.parquet).

Context
-------
The bundled 72-star dataset has no provenance script in the repo. This script
fixes that gap AND adds ~40-60 new stars, prioritizing the classes that were
underrepresented in the original bundle (ROTATIONAL: 6, LONG_PERIOD: 8), while
topping up the others a bit for balance. It writes its output to a SEPARATE
file, benchmarks/dataset/light_curves_extension.parquet, with exactly the same
schema as light_curves.parquet. It does NOT modify light_curves.parquet -- a
human/maintainer step merges the two after reviewing the extension.

Run date: 2026-07-08.

Environment
-----------
This script MUST be run with the isolated reference venv, which has the
`skypatrol` (import name `pyasassn`) package installed:

    C:\\Users\\thari\\Documents\\GitHub\\cuPeriod\\.venv-ref\\Scripts\\python.exe benchmarks\\dataset\\download_extension.py

Do NOT run it with the main .venv (cuperiod/GPU env) -- skypatrol is not
installed there and is intentionally kept isolated (see
benchmarks/dataset/ and the .venv-ref split documented for the rest of the
benchmark suite).

Query criteria (ASAS-SN Sky Patrol `aavsovsx` catalog, via ADQL)
------------------------------------------------------------------
For each broad class we query one or more "clean" VSX `variability_type`
values (no `|`, `:`, or `/` qualifiers -- ambiguous/composite classifications
are excluded at the SQL level with NOT LIKE):

    ROTATIONAL   : BY, RS, ROT, ACV      (underrepresented -> priority)
    LONG_PERIOD  : M, SR, SRA, SRB       (underrepresented -> priority)
    ECLIPSING    : EA, EB, EW            (top-up only)
    RR_LYRAE     : RRAB, RRC             (top-up only)
    CEPHEID      : DCEP, CWA             (top-up only)
    DELTA_SCUTI  : DSCT, HADS            (top-up only)

Additional filters applied in ADQL:
    - period IS NOT NULL, 0.05 < period < 1000 (days)
    - max_mag_system = 'V' (a consistent, well-populated photometric system
      in the VSX cross-match; avoids mixing V/Kp/G/R/CR mag scales)
    - 11 <= max_mag <= 14.5 (bright enough for ASAS-SN photometry, faint
      enough to avoid saturation issues)
    - variability_type NOT LIKE '%|%' AND NOT LIKE '%:%' AND NOT LIKE '%/%'
    - asas_sn_id NOT IN the ids already present in light_curves.parquet

Light curves are then downloaded in g band via
SkyPatrolClient.query_list(download=True, threads=1) in batches of <=25
IDs, and a quality cut is applied:
    - phot_filter == 'g'
    - quality == 'G' (ASAS-SN good-quality flag; drops 'B'/'U' points)
    - mag_err < 99 (drop sentinel/bad errors) and finite
    - keep only stars with >= 300 surviving points and baseline >= 1000 d

Windows gotcha
--------------
SkyPatrolClient.query_list(download=True) uses multiprocessing internally.
On Windows this REQUIRES the `if __name__ == "__main__":` guard below and
threads=1, otherwise the multiprocessing spawn mechanism re-imports and
re-executes this module in each child process ("spawn bomb").

Re-running
----------
This script is idempotent-ish: it always re-queries the catalog and
re-downloads; it does not cache. Because the pool of catalog candidates can
shift slightly between ASAS-SN deployments, results may vary a small amount
run to run, but the query criteria and quality cuts are fixed and documented
above so results are reproducible in spirit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
EXISTING_PARQUET = DATASET_DIR / "light_curves.parquet"
OUTPUT_PARQUET = DATASET_DIR / "light_curves_extension.parquet"

# broad_class -> (list of clean VSX variability_type values, target #candidates to query)
CLASS_TYPES: dict[str, list[str]] = {
    "ROTATIONAL": ["BY", "RS", "ROT", "ACV"],
    "LONG_PERIOD": ["M", "SR", "SRA", "SRB"],
    "ECLIPSING": ["EA", "EB", "EW"],
    "RR_LYRAE": ["RRAB", "RRC"],
    "CEPHEID": ["DCEP", "CWA"],
    "DELTA_SCUTI": ["DSCT", "HADS"],
}

# How many *new, passing* stars we want per class (priority classes get more
# headroom; top-up classes get a small amount to keep totals balanced).
CLASS_TARGETS: dict[str, int] = {
    "ROTATIONAL": 16,
    "LONG_PERIOD": 14,
    "ECLIPSING": 6,
    "RR_LYRAE": 6,
    "CEPHEID": 6,
    "DELTA_SCUTI": 6,
}

# Over-fetch factor: not every catalog candidate will survive the quality
# cuts (missing data, short baseline, sparse detections), so query more
# candidates per class than the target count.
CANDIDATE_OVERFETCH = 4

MIN_N_DET = 300
MIN_BASELINE = 1000.0
BATCH_SIZE = 25

MAG_LO, MAG_HI = 11.0, 14.5
PERIOD_LO, PERIOD_HI = 0.05, 1000.0


def get_existing_ids() -> set[int]:
    df = pd.read_parquet(EXISTING_PARQUET)
    return set(df["asas_sn_id"].astype(int).tolist())


def fetch_candidates(client, vsx_type: str, existing_ids: set[int], limit: int) -> pd.DataFrame:
    """Query aavsovsx for clean-type candidates of a single VSX type."""
    query = f"""
    SELECT asas_sn_id, variability_type, period, max_mag, max_mag_system
    FROM aavsovsx
    WHERE variability_type = '{vsx_type}'
    AND period IS NOT NULL
    AND period > {PERIOD_LO} AND period < {PERIOD_HI}
    AND max_mag_system = 'V'
    AND max_mag >= {MAG_LO} AND max_mag <= {MAG_HI}
    LIMIT {limit}
    """
    df = client.adql_query(query)
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    # Belt-and-suspenders clean-type filter (should already be exact-match,
    # but guard against catalog quirks).
    mask = ~df["variability_type"].astype(str).str.contains(r"[|:/]", regex=True)
    df = df[mask]
    df = df[~df["asas_sn_id"].isin(existing_ids)]
    return df


def classify_broad_class(vsx_type: str) -> str | None:
    for broad, types in CLASS_TYPES.items():
        if vsx_type in types:
            return broad
    return None


def download_batch(client, ids: list[int]) -> pd.DataFrame:
    lcs = client.query_list(
        ids, id_col="asas_sn_id", catalog="master_list", download=True, threads=1
    )
    return lcs.data


def quality_filter(lc: pd.DataFrame) -> pd.DataFrame:
    """Apply ASAS-SN good-quality + g-band cut used elsewhere in the suite."""
    lc = lc[(lc["phot_filter"] == "g") & (lc["quality"] == "G")]
    lc = lc[np.isfinite(lc["mag_err"]) & (lc["mag_err"] < 99) & np.isfinite(lc["mag"])]
    lc = lc.sort_values("jd")
    return lc


def main() -> None:
    from pyasassn.client import SkyPatrolClient

    existing_ids = get_existing_ids()
    print(f"Existing dataset has {len(existing_ids)} stars.")

    client = SkyPatrolClient()

    # ---- 1. Gather candidates per class -----------------------------------
    candidates: list[dict] = []  # each: asas_sn_id, vsx_type, broad_class, vsx_period
    for broad, types in CLASS_TYPES.items():
        target = CLASS_TARGETS[broad]
        per_type_limit = max(10, (target * CANDIDATE_OVERFETCH) // len(types) + 5)
        class_candidates = []
        for vsx_type in types:
            try:
                df = fetch_candidates(client, vsx_type, existing_ids, per_type_limit)
            except Exception as exc:  # pragma: no cover - network/service issues
                print(f"  WARNING: query failed for type {vsx_type}: {exc}")
                continue
            if df is None or len(df) == 0:
                continue
            for _, row in df.iterrows():
                class_candidates.append(
                    {
                        "asas_sn_id": int(row["asas_sn_id"]),
                        "vsx_type": vsx_type,
                        "broad_class": broad,
                        "vsx_period": float(row["period"]),
                    }
                )
        # Over-fetch, then cap; download step will further reduce via quality cuts.
        cap = target * CANDIDATE_OVERFETCH
        class_candidates = class_candidates[:cap]
        print(f"{broad}: {len(class_candidates)} raw candidates gathered "
              f"(target {target} passing stars)")
        candidates.extend(class_candidates)

    if not candidates:
        print("No candidates found at all -- Sky Patrol query path may be broken. Aborting.")
        sys.exit(1)

    cand_df = pd.DataFrame(candidates).drop_duplicates(subset="asas_sn_id")
    print(f"\nTotal unique candidates across all classes: {len(cand_df)}")

    # ---- 2. Download light curves in batches, apply quality cuts ----------
    kept_rows = []
    rejected = []
    all_ids = cand_df["asas_sn_id"].tolist()

    # Stop early per class once target is reached, to avoid hammering the
    # service with more downloads than needed.
    kept_per_class: dict[str, int] = {b: 0 for b in CLASS_TYPES}

    for start in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[start : start + BATCH_SIZE]
        # Skip classes that are already full.
        batch_meta = cand_df[cand_df["asas_sn_id"].isin(batch_ids)]
        batch_meta = batch_meta[
            batch_meta["broad_class"].map(lambda b: kept_per_class[b] < CLASS_TARGETS[b])
        ]
        batch_ids = batch_meta["asas_sn_id"].tolist()
        if not batch_ids:
            continue

        print(f"\nDownloading batch of {len(batch_ids)} ids "
              f"({start // BATCH_SIZE + 1}/{(len(all_ids) - 1) // BATCH_SIZE + 1})...")
        try:
            raw = download_batch(client, batch_ids)
        except Exception as exc:  # pragma: no cover - network/service issues
            print(f"  WARNING: batch download failed: {exc}")
            continue

        for asas_sn_id, group in raw.groupby("asas_sn_id"):
            meta_row = cand_df.loc[cand_df["asas_sn_id"] == asas_sn_id].iloc[0]
            broad = meta_row["broad_class"]
            if kept_per_class[broad] >= CLASS_TARGETS[broad]:
                continue

            good = quality_filter(group)
            n_det = len(good)
            if n_det == 0:
                rejected.append((asas_sn_id, meta_row["vsx_type"], broad, "no good g-band points"))
                continue
            baseline = float(good["jd"].max() - good["jd"].min())

            if n_det < MIN_N_DET:
                rejected.append(
                    (asas_sn_id, meta_row["vsx_type"], broad, f"n_det={n_det} < {MIN_N_DET}")
                )
                continue
            if baseline < MIN_BASELINE:
                rejected.append(
                    (asas_sn_id, meta_row["vsx_type"], broad,
                     f"baseline={baseline:.0f}d < {MIN_BASELINE}d")
                )
                continue

            kept_rows.append(
                {
                    "asas_sn_id": int(asas_sn_id),
                    "band": "g",
                    "vsx_type": meta_row["vsx_type"],
                    "broad_class": broad,
                    "vsx_period": float(meta_row["vsx_period"]),
                    "n_det": int(n_det),
                    "baseline": baseline,
                    "jd": good["jd"].to_numpy(dtype=float).tolist(),
                    "mag": good["mag"].to_numpy(dtype=float).tolist(),
                    "mag_err": good["mag_err"].to_numpy(dtype=float).tolist(),
                }
            )
            kept_per_class[broad] += 1

        if all(kept_per_class[b] >= CLASS_TARGETS[b] for b in CLASS_TYPES):
            print("\nAll class targets reached; stopping early.")
            break

    # ---- 3. Write output ----------------------------------------------------
    if not kept_rows:
        print("\nNo stars passed quality cuts. Nothing written.")
        sys.exit(1)

    out_df = pd.DataFrame(kept_rows)
    # Match column order/schema of light_curves.parquet exactly.
    out_df = out_df[
        ["asas_sn_id", "band", "vsx_type", "broad_class", "vsx_period",
         "n_det", "baseline", "jd", "mag", "mag_err"]
    ]
    out_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {len(out_df)} stars to {OUTPUT_PARQUET}")

    # ---- 4. Report ------------------------------------------------------
    print("\nPer-class counts added:")
    print(out_df["broad_class"].value_counts())

    print("\nSummary table:")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(
            out_df[
                ["broad_class", "vsx_type", "asas_sn_id", "vsx_period", "n_det", "baseline"]
            ].sort_values(["broad_class", "vsx_type"])
        )

    if rejected:
        print(f"\nRejected {len(rejected)} candidates by quality cuts:")
        for asas_sn_id, vsx_type, broad, reason in rejected:
            print(f"  {broad:12s} {vsx_type:6s} {asas_sn_id}: {reason}")


if __name__ == "__main__":
    main()
