"""Fetch the OGLE photometry of V1309 Sco (the 2008 stellar-merger progenitor).

The files are the supplementary material of Tylenda et al. (2011, A&A 528,
A114), "V1309 Scorpii: merger of a contact binary", mirrored by the OGLE project
at https://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/ (the anonymous-FTP tree,
over TLS). Each file is three columns: HJD, magnitude, magnitude error.

Every file is checked against the SHA-256 pinned in ``SHA256`` (the copies behind
the committed ``results/pdot_v1309sco*.parquet``), on download and when reused
from disk; see ``_download.py``.

The data land in ``benchmarks/data/v1309sco/`` (gitignored, ~70 KB); the
analysis is ``pdot_v1309sco.py``. OGLE data are public; please cite Tylenda et
al. (2011) and the OGLE project if you use them.
"""

from __future__ import annotations

import sys

from _common import DATA
from _download import DownloadError, fetch

BASE = "https://ftp.astrouw.edu.pl/ogle/ogle3/V1309_SCO/"
FILES = (
    "README",
    "V1309_SCO_OIII.dat",    # OGLE-III, 2001-2009, I band (the progenitor)
    "V1309_SCO_OIII_V.dat",  # OGLE-III, V band
    "V1309_SCO.dat",         # OGLE-IV, 2010-, I band (the remnant)
    "V1309_SCO_V.dat",       # OGLE-IV, V band
)

#: SHA-256 of each file, recorded from the copies fetched on 2026-08-31.
SHA256 = {
    "README": "3a20d0463ed6177435934fbe228875979d17f023ecac6ea006c044ae65bfc6ff",
    "V1309_SCO_OIII.dat":
        "47a8f4cce8b9dbf7a6d32c5a3ccc714fddd0762cb05d813a54c58f1dc6853ec9",
    "V1309_SCO_OIII_V.dat":
        "07d9495da3201dacb103fc0331610810bdb3c9da6c2e6c2991fe63493a29342d",
    "V1309_SCO.dat":
        "357cf57f1680fa2d36f52f3417278f9ecd50415099b2d0981fe136c9858f6b1b",
    "V1309_SCO_V.dat":
        "b6c9e18265998b9e39701fa74080097d65e0078f7e614b4a20b7333fff1e517b",
}


def main() -> None:
    out = DATA / "v1309sco"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name
        print(f"  {name} ...", end=" ", flush=True)
        if fetch(BASE + name, target, SHA256[name]):
            print(f"fetched {target.stat().st_size} bytes, SHA-256 OK")
        else:
            print("have it, SHA-256 OK")
    print(f"\n-> {out}")


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        sys.exit(f"\nerror: {exc}")
