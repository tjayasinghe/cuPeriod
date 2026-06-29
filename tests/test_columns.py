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
