"""Fetch OGLE-III + OGLE-IV photometry of six bulge contact binaries with a third body.

The stars are drawn from Table 3 of Hong et al. (2022, AJ 163, 157), who
classified 307 OGLE bulge contact binaries as showing a *sinusoidal* eclipse
timing variation — the light-travel-time signature of a distant companion —
from 15 years of seasonal timings. Six were picked by hand to be bright and to
have an outer period short enough that the OGLE-III + OGLE-IV baseline covers
at least one cycle (see ``pdot_ogle_triples.py`` for the list and the published
values).

The photometry is the public OGLE Collection of Variable Stars release of
Soszynski et al. (2016, Acta Astron. 66, 405), served by the OGLE archive at
https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/blg/ecl/ (the same tree as the
anonymous-FTP mirror, over TLS) — one file per star and survey phase, three
columns: HJD - 2450000, I magnitude, error. OGLE-III (2001-2009) and OGLE-IV
(2010-2015) are separate files with independent photometric zero points; the
analysis detrends them jointly.

Every file is checked against the SHA-256 pinned in ``SHA256`` (the copies behind
``REPORT_ogle_triples.md``), on download and when reused from disk; see
``_download.py``.

The data land in ``benchmarks/data/ogle_triples/`` (gitignored, ~700 KB).
OGLE data are public; please cite Soszynski et al. (2016) and Hong et al.
(2022) if you use them.
"""

from __future__ import annotations

import sys

from _common import DATA
from _download import DownloadError, fetch

BASE = "https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/blg/ecl/"

#: OGLE-BLG-ECL identifiers (Hong et al. 2022, Table 3).
STARS = ("061372", "246583", "252457", "289126", "299343", "299686")

#: OGLE phases with their own photometry directories.
PHASES = ("ogle3", "ogle4")

#: SHA-256 of each saved file, recorded from the copies fetched on 2026-09-01.
SHA256 = {
    "OGLE-BLG-ECL-061372_ogle3.dat":
        "bf9356ea66c1525b97155db1fbdb50240401d13b72e3c1d6c4ee43a4def796be",
    "OGLE-BLG-ECL-061372_ogle4.dat":
        "fa9f9ba72ab433bc301db9d0f1c042ffd13a8e18a7d4e5399a494e25bc79fe9b",
    "OGLE-BLG-ECL-246583_ogle3.dat":
        "088a85041c83df7f174ba73f84ea43ab80e1c7143dbc7625aca1b52adc5e76a6",
    "OGLE-BLG-ECL-246583_ogle4.dat":
        "e44e821bd8bdf660aebbe190f288ff41dae9c1de7b5901ddaaf0e20d0e3c2db8",
    "OGLE-BLG-ECL-252457_ogle3.dat":
        "51f06dbdc78c93f77f85e00b384b26fbc6e29c0070c9b3b0fab1b6125daf8e45",
    "OGLE-BLG-ECL-252457_ogle4.dat":
        "b22273dd95b12433a4d0bcd0789e2c8f643e36cb3aa0fbae0b0cdec7da92700c",
    "OGLE-BLG-ECL-289126_ogle3.dat":
        "2a5c6e79cbf52da67c45e3786293e5ff634cfb1b22ec672e210b54845604edc9",
    "OGLE-BLG-ECL-289126_ogle4.dat":
        "8e0d4333b299f445586b3d7c689aab88da4ee797a195da2ff29f7e6888af2684",
    "OGLE-BLG-ECL-299343_ogle3.dat":
        "34962b41fa3f7f07e36a16544767f1b34ea0954ff76a702fcc6853dc5305ad7e",
    "OGLE-BLG-ECL-299343_ogle4.dat":
        "543bcba42cf9677359cd47e666b00dfbbbdde4c868494082deda25033452e5bb",
    "OGLE-BLG-ECL-299686_ogle3.dat":
        "4dd3520495d84c7bfdbf8889032b9c4a943305121f7b6d226874de6f1fe74a87",
    "OGLE-BLG-ECL-299686_ogle4.dat":
        "87a375a754bfcfac96465e44572bc3a9c69c2737ed94c4f1ba208b1a68f48532",
}


def main() -> None:
    out = DATA / "ogle_triples"
    out.mkdir(parents=True, exist_ok=True)
    for star in STARS:
        for phase in PHASES:
            name = f"OGLE-BLG-ECL-{star}.dat"
            target = out / f"OGLE-BLG-ECL-{star}_{phase}.dat"
            url = f"{BASE}phot_{phase}/I/{name}"
            print(f"  {target.name} ...", end=" ", flush=True)
            if fetch(url, target, SHA256[target.name]):
                print(f"fetched {target.stat().st_size} bytes, SHA-256 OK")
            else:
                print("have it, SHA-256 OK")
    print(f"\n-> {out}")


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        sys.exit(f"\nerror: {exc}")
