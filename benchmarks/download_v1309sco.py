"""Fetch the OGLE photometry of V1309 Sco (the 2008 stellar-merger progenitor).

The files are the supplementary material of Tylenda et al. (2011, A&A 528,
A114), "V1309 Scorpii: merger of a contact binary", mirrored by the OGLE project
at ftp://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/. Each file is three columns:
HJD, magnitude, magnitude error.

The data land in ``benchmarks/data/v1309sco/`` (gitignored, ~70 KB); the
analysis is ``pdot_v1309sco.py``. OGLE data are public; please cite Tylenda et
al. (2011) and the OGLE project if you use them.
"""

from __future__ import annotations

import urllib.request

from _common import DATA

BASE = "ftp://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/"
FILES = (
    "README",
    "V1309_SCO_OIII.dat",    # OGLE-III, 2001-2009, I band (the progenitor)
    "V1309_SCO_OIII_V.dat",  # OGLE-III, V band
    "V1309_SCO.dat",         # OGLE-IV, 2010-, I band (the remnant)
    "V1309_SCO_V.dat",       # OGLE-IV, V band
)


def main() -> None:
    out = DATA / "v1309sco"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name
        if target.exists():
            print(f"  have {name}")
            continue
        print(f"  fetching {name} ...", end=" ", flush=True)
        with urllib.request.urlopen(BASE + name, timeout=120) as fh:
            target.write_bytes(fh.read())
        print(f"{target.stat().st_size} bytes")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
