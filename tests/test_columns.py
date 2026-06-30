"""Column auto-detection, overrides, and domain inference."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.columns import ColumnMap, Domain, infer_domain
from cuperiod.core.errors import ColumnResolutionError


def test_autodetect_standard_names() -> None:
    cols = ["HJD", "mag", "mag_err", "extra"]
    resolved = ColumnMap().resolve(cols)
    assert resolved.time == "HJD"
    assert resolved.value == "mag"
    assert resolved.error == "mag_err"
    assert resolved.domain is Domain.MAGNITUDE


def test_explicit_override_wins() -> None:
    cols = ["t", "Vmag", "e_Vmag", "jd"]
    resolved = ColumnMap(time="jd", value="Vmag", error="e_Vmag").resolve(cols)
    assert resolved.time == "jd"
    assert resolved.value == "Vmag"
    assert resolved.error == "e_Vmag"


def test_flux_domain_inferred() -> None:
    resolved = ColumnMap().resolve(["bjd", "sap_flux", "sap_flux_err"])
    assert resolved.value == "sap_flux"
    assert resolved.domain is Domain.FLUX


def test_missing_time_raises() -> None:
    with pytest.raises(ColumnResolutionError, match="time"):
        ColumnMap().resolve(["mag", "mag_err"])


def test_unknown_explicit_column_raises() -> None:
    with pytest.raises(ColumnResolutionError, match="nope"):
        ColumnMap(time="nope").resolve(["jd", "mag"])


def test_case_insensitive_match() -> None:
    resolved = ColumnMap().resolve(["JD", "MAGNITUDE", "ERR"])
    assert resolved.time == "JD"
    assert resolved.value == "MAGNITUDE"
    assert resolved.error == "ERR"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("flux", Domain.FLUX), ("Vmag", Domain.MAGNITUDE), ("weird", None)],
)
def test_infer_domain(name: str, expected: Domain | None) -> None:
    assert infer_domain(name) is expected


def test_dict_of_arrays_resolves() -> None:
    data = {"jd": np.arange(3.0), "mag": np.ones(3), "mag_err": np.ones(3)}
    resolved = ColumnMap().resolve(list(data))
    assert (resolved.time, resolved.value, resolved.error) == ("jd", "mag", "mag_err")


# Real light-curve headers from each survey's standard data product:
# (id, columns, expected time, value, error, band).
_SURVEY_HEADERS = [
    ("asas-sn-v2",
     ["asas_sn_id", "jd", "flux", "flux_err", "mag", "mag_err", "camera",
      "quality", "phot_filter"],
     "jd", "flux", "flux_err", "phot_filter"),
    ("asas-3",
     ["HJD", "MAG_0", "MAG_1", "MAG_2", "MAG_3", "MAG_4",
      "MER_0", "MER_1", "MER_2", "MER_3", "MER_4", "GRADE"],
     "HJD", "MAG_0", "MER_0", None),
    ("atlas",  # mixed mag (m/dm) + flux (uJy/duJy) + a non-error `err` flag column
     ["MJD", "m", "dm", "uJy", "duJy", "F", "err", "chi/N"],
     "MJD", "uJy", "duJy", None),
    ("crts",
     ["MasterID", "Mag", "Magerr", "RA", "Dec", "MJD"],
     "MJD", "Mag", "Magerr", None),
    ("ztf-api",
     ["oid", "mjd", "mag", "magerr", "catflags", "filtercode", "fid"],
     "mjd", "mag", "magerr", "filtercode"),
    ("ztf-bulk",
     ["ObjectID", "HMJD", "mag", "magerr", "catflags", "FilterID"],
     "HMJD", "mag", "magerr", "FilterID"),
    ("lsst",
     ["band", "midpointMjdTai", "psfFlux", "psfFluxErr"],
     "midpointMjdTai", "psfFlux", "psfFluxErr", "band"),
    ("lsst-dp0.2",
     ["filterName", "midPointTai", "psFlux", "psFluxErr"],
     "midPointTai", "psFlux", "psFluxErr", "filterName"),
    ("panstarrs",
     ["objID", "obsTime", "psfFlux", "psfFluxErr", "filterID"],
     "obsTime", "psfFlux", "psfFluxErr", "filterID"),
    ("tess-spoc",  # PDCSAP (corrected) must pair with PDCSAP_FLUX_ERR, not SAP
     ["TIME", "SAP_FLUX", "SAP_FLUX_ERR", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"],
     "TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", None),
    ("tess-qlp",  # prefer the detrended KSPSAP flux
     ["TIME", "SAP_FLUX", "KSPSAP_FLUX", "KSPSAP_FLUX_ERR", "QUALITY"],
     "TIME", "KSPSAP_FLUX", "KSPSAP_FLUX_ERR", None),
    ("kepler",
     ["TIME", "SAP_FLUX", "SAP_FLUX_ERR", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR",
      "SAP_QUALITY"],
     "TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", None),
    ("gaia",
     ["source_id", "band", "time", "mag", "flux", "flux_error", "flux_over_error"],
     "time", "flux", "flux_error", "band"),
]


@pytest.mark.parametrize(
    ("cols", "time", "value", "error", "band"),
    [c[1:] for c in _SURVEY_HEADERS],
    ids=[c[0] for c in _SURVEY_HEADERS],
)
def test_survey_headers_autodetect(
    cols: list[str], time: str, value: str, error: str, band: str | None
) -> None:
    resolved = ColumnMap().resolve(cols)
    assert (resolved.time, resolved.value, resolved.error, resolved.band) == (
        time, value, error, band,
    )


def test_domain_aware_error_pairs_with_value() -> None:
    # A magnitude value pairs with the magnitude error, a flux value with the flux
    # error, even when both are present (the ATLAS m/dm + uJy/duJy case).
    cols = ["MJD", "m", "dm", "uJy", "duJy"]
    assert ColumnMap(value="m").resolve(cols).error == "dm"
    assert ColumnMap(value="uJy").resolve(cols).error == "duJy"
