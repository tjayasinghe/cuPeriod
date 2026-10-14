"""Fetch OGLE-III + OGLE-IV photometry of six bulge contact binaries with a third body.

The stars are drawn from Table 3 of Hong et al. (2022, AJ 163, 157), who
classified 307 OGLE bulge contact binaries as showing a *sinusoidal* eclipse
timing variation — the light-travel-time signature of a distant companion —
from 15 years of seasonal timings. Six were picked by hand to be bright and to
have an outer period short enough that the OGLE-III + OGLE-IV baseline covers
at least one cycle (see ``pdot_ogle_triples.py`` for the list and the published
values).

The photometry is the public OGLE Collection of Variable Stars release of
Soszynski et al. (2016, Acta Astron. 66, 405), mirrored at
ftp://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/blg/ecl/ — one file per star and
survey phase, three columns: HJD - 2450000, I magnitude, error. OGLE-III
(2001-2009) and OGLE-IV (2010-2015) are separate files with independent
photometric zero points; the analysis detrends them jointly.

The data land in ``benchmarks/data/ogle_triples/`` (gitignored, ~700 KB).
OGLE data are public; please cite Soszynski et al. (2016) and Hong et al.
(2022) if you use them.
"""

from __future__ import annotations

import urllib.request

from _common import DATA

BASE = "ftp://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/blg/ecl/"

#: OGLE-BLG-ECL identifiers (Hong et al. 2022, Table 3).
STARS = ("061372", "246583", "252457", "289126", "299343", "299686")

#: OGLE phases with their own photometry directories.
PHASES = ("ogle3", "ogle4")


def main() -> None:
    out = DATA / "ogle_triples"
    out.mkdir(parents=True, exist_ok=True)
    for star in STARS:
        for phase in PHASES:
            name = f"OGLE-BLG-ECL-{star}.dat"
            target = out / f"OGLE-BLG-ECL-{star}_{phase}.dat"
            if target.exists():
                print(f"  have {target.name}")
                continue
            url = f"{BASE}phot_{phase}/I/{name}"
            print(f"  fetching {target.name} ...", end=" ", flush=True)
            with urllib.request.urlopen(url, timeout=120) as fh:
                target.write_bytes(fh.read())
            print(f"{target.stat().st_size} bytes")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
