"""The periodogram() dispatcher: input forms, multi-method, settings."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core.result import MultiResult, Periodogram
from synth import synthetic_eclipser, synthetic_sine


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every backend as on a CPU-only machine.

    Keeps this module deterministic across machines (and off a shared GPU): ``auto``
    resolves to the CPU path everywhere.
    """
    import cuperiod.methods.base as base
    from cuperiod.core.backend import cpu_backends

    monkeypatch.setattr(base, "available_backends", cpu_backends)
    monkeypatch.setattr(base, "cuda_available", lambda: False)
    monkeypatch.setattr(base, "torch_gpu_available", lambda: False)


def test_single_method_returns_periodogram() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS")
    assert isinstance(pg, Periodogram)
    assert pg.method == "GLS"


def test_multi_method_returns_multiresult() -> None:
    t, mag, err = synthetic_sine()
    res = cup.periodogram((t, mag, err), ["gls", "bls"], domain="flux")
    assert isinstance(res, MultiResult)
    assert set(res.keys()) == {"GLS", "BLS"}


def test_dict_of_arrays_input() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram({"jd": t, "mag": mag, "mag_err": err}, "GLS")
    assert pg.best_period() == pytest.approx(0.6234, rel=1e-3)


def test_mapping_of_lightcurves_is_multiband() -> None:
    t, mag, err = synthetic_sine(n=200)
    lcs = {
        "g": cup.LightCurve.from_arrays(t, mag, err),
        "r": cup.LightCurve.from_arrays(t, mag + 1.0, err),
    }
    pg = cup.periodogram(lcs, "GLS")
    assert "bands" in pg.meta


def test_settings_override_changes_grid() -> None:
    t, mag, err = synthetic_sine()
    coarse = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(samples_per_peak=2)
    )
    fine = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(samples_per_peak=10)
    )
    assert fine.size > coarse.size


def test_best_periods_convenience() -> None:
    t, mag, err = synthetic_sine()
    pg = cup.periodogram((t, mag, err), "GLS")
    assert len(cup.best_periods(pg, 3)) == 3


def test_unknown_method_raises() -> None:
    t, mag, err = synthetic_sine()
    with pytest.raises(cup.UnknownMethodError):
        cup.periodogram((t, mag, err), "NOPE")


class _Abort(Exception):
    """Raised by the resolver spy to stop a run before any compute."""


def _spy_resolver(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record ``(method, selector)`` handed to ``resolve_backend``, then abort."""
    from cuperiod.methods.base import PeriodogramMethod

    seen: list[tuple[str, str]] = []

    def spy(self: PeriodogramMethod, requested: str) -> str:
        seen.append((self.name, requested))
        raise _Abort

    monkeypatch.setattr(PeriodogramMethod, "resolve_backend", spy)
    return seen


# --- settings.backend is honoured (CORE-A-9, CORE-B-4, METH-A-8, METH-C-5) ------------

_PINS = {
    "GLS": ("GLSSettings", "astropy"),
    "BLS": ("BLSSettings", "astropy"),
    "PDM": ("PDMSettings", "numpy"),
    "CE": ("CESettings", "numpy"),
    "MHAOV": ("MHAOVSettings", "numpy"),
    "STRINGLENGTH": ("StringLengthSettings", "numpy"),
    "SUPERSMOOTHER": ("SuperSmootherSettings", "numpy"),
    "TLS": ("TLSSettings", "numpy"),
}


@pytest.mark.parametrize("name", sorted(_PINS))
def test_settings_backend_is_the_default_backend(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls_name, pin = _PINS[name]
    settings = getattr(cup, cls_name)(backend=pin)
    seen = _spy_resolver(monkeypatch)
    t, mag, err = synthetic_sine(n=60)
    with pytest.raises(_Abort):
        cup.periodogram((t, mag, err), name, settings=settings)
    assert seen == [(name, pin)]


def test_settings_backend_runs_end_to_end() -> None:
    t, mag, err = synthetic_sine(n=200)
    pg = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(backend="astropy")
    )
    assert pg.backend == "astropy"


def test_explicit_backend_overrides_settings() -> None:
    t, mag, err = synthetic_sine(n=200)
    pg = cup.periodogram(
        (t, mag, err),
        "GLS",
        backend="finufft",
        settings=cup.GLSSettings(backend="astropy"),
    )
    assert pg.backend == "finufft"


def test_backend_environment_variable_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_BACKEND", "astropy")
    t, mag, err = synthetic_sine(n=200)
    assert cup.periodogram((t, mag, err), "GLS").backend == "astropy"


def test_default_backend_unchanged() -> None:
    t, mag, err = synthetic_sine(n=200)
    expected = cup.get_method("GLS").resolve_backend("auto")
    assert cup.periodogram((t, mag, err), "GLS").backend == expected


def test_settings_gpu_pin_raises_without_a_gpu() -> None:
    t, mag, err = synthetic_sine(n=60)
    with pytest.raises(cup.BackendUnavailableError):
        cup.periodogram((t, mag, err), "GLS", settings=cup.GLSSettings(backend="gpu"))


def test_per_method_backend_pins_in_a_multi_method_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuperiod.methods.base import PeriodogramMethod

    seen: list[tuple[str, str]] = []
    real = PeriodogramMethod.resolve_backend

    def spy(self: PeriodogramMethod, requested: str) -> str:
        seen.append((self.name, requested))
        return real(self, requested)

    monkeypatch.setattr(PeriodogramMethod, "resolve_backend", spy)
    t, mag, err = synthetic_sine(n=120, span=30.0)
    res = cup.periodogram(
        (t, mag, err),
        ["GLS", "BLS"],
        domain="flux",
        settings={
            "GLS": cup.GLSSettings(backend="astropy"),
            "BLS": cup.BLSSettings(backend="numpy", max_period_days=5.0),
        },
    )
    assert isinstance(res, MultiResult)
    assert dict(seen) == {"GLS": "astropy", "BLS": "numpy"}
    assert res["GLS"].backend == "astropy"
    assert res["BLS"].backend == "numpy"


# --- peak / stored-spectrum settings reach the result (CORE-A-10, METH-A-18) ----------


def test_peak_settings_shape_the_default_peak_list_and_spectrum() -> None:
    t, mag, err = synthetic_sine(n=200)
    tuned = cup.GLSSettings(
        n_peaks=3, peak_separation_rayleigh=50.0, downsample_points=50
    )
    pg = cup.periodogram((t, mag, err), "GLS", settings=tuned)
    assert isinstance(pg, Periodogram)
    peaks = pg.best_periods()
    assert len(peaks) == 3
    freqs = sorted(p.frequency for p in peaks)
    assert min(np.diff(freqs)) >= 50.0 / pg.baseline
    assert pg.downsample()[0].size == 50
    assert len(pg.to_dict()["peaks"]) == 3
    assert len(cup.best_periods(pg)) == 3
    # Explicit arguments still win.
    assert len(pg.best_periods(5)) == 5
    assert pg.downsample(100)[0].size == 100


def test_default_peak_list_is_unchanged() -> None:
    t, mag, err = synthetic_sine(n=200)
    pg = cup.periodogram((t, mag, err), "GLS")
    assert isinstance(pg, Periodogram)
    assert pg.best_periods() == pg.best_periods(10, min_separation_rayleigh=3.0)
    assert pg.downsample()[0].size == 2000


def test_alias_tolerance_follows_the_settings_value() -> None:
    # The BLS/TLS alias_freq_tolerance is carried whether or not the caller spelled
    # it out: settings that compare equal must give the same peak list. (The GUI
    # builds every settings object with all fields set; an "only if explicitly set"
    # rule would make its alias-diverse lists differ from cup.periodogram's.)
    t, flux, err = synthetic_eclipser(n=300, span=40.0)
    defaults = cup.BLSSettings(max_period_days=6.0)
    spelled_out = cup.BLSSettings(**defaults.model_dump())
    assert spelled_out == defaults
    a = cup.periodogram((t, flux, err), "BLS", domain="flux", settings=defaults)
    b = cup.periodogram((t, flux, err), "BLS", domain="flux", settings=spelled_out)
    assert isinstance(a, Periodogram) and isinstance(b, Periodogram)
    assert a.peak_defaults == b.peak_defaults
    assert a.peak_defaults["alias_tolerance"] == defaults.alias_freq_tolerance
    assert a.best_periods(alias_diverse=True) == b.best_periods(alias_diverse=True)
    # ... and a different value changes what is carried.
    wide = cup.periodogram(
        (t, flux, err),
        "BLS",
        domain="flux",
        settings=cup.BLSSettings(max_period_days=6.0, alias_freq_tolerance=0.05),
    )
    assert isinstance(wide, Periodogram)
    assert wide.peak_defaults["alias_tolerance"] == 0.05


def test_peak_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_N_PEAKS", "2")
    t, mag, err = synthetic_sine(n=200)
    pg = cup.periodogram((t, mag, err), "GLS")
    assert isinstance(pg, Periodogram)
    assert len(pg.best_periods()) == 2


# --- settings selection (CORE-A-11, METH-A-9) -----------------------------------------


def test_wrong_class_settings_raise() -> None:
    t, mag, err = synthetic_sine(n=60)
    with pytest.raises(TypeError, match="PDMSettings"):
        cup.periodogram((t, mag, err), "PDM", settings=cup.CESettings())
    with pytest.raises(TypeError, match="GLSSettings"):
        cup.periodogram((t, mag, err), "GLS", settings=cup.MHAOVSettings())


def test_bare_settings_apply_to_their_method_in_a_multi_method_call() -> None:
    from cuperiod.methods.base import get_method, select_settings

    gls, bls = get_method("GLS"), get_method("BLS")
    chosen = select_settings([gls, bls], cup.GLSSettings(samples_per_peak=2))
    assert chosen["GLS"].samples_per_peak == 2  # type: ignore[attr-defined]
    assert chosen["BLS"] == cup.BLSSettings()
    with pytest.raises(TypeError, match="GLSSettings or BLSSettings"):
        select_settings([gls, bls], cup.PDMSettings())


@pytest.mark.parametrize(
    "key", ["String-Length", "string_length", "StringLength", "STRINGLENGTH"]
)
def test_mapping_keys_use_the_method_name_normalisation(key: str) -> None:
    from cuperiod.methods.base import get_method, select_settings

    pinned = cup.StringLengthSettings(maximum_frequency=1.0)
    chosen = select_settings([get_method("String-Length")], {key: pinned})
    assert chosen["STRINGLENGTH"] is pinned


def test_documented_hyphenated_key_changes_the_run() -> None:
    t, mag, err = synthetic_sine(n=120)
    res = cup.periodogram(
        (t, mag, err),
        ["GLS", "String-Length"],
        backend="cpu",
        settings={
            "gls": cup.GLSSettings(samples_per_peak=2),
            "String-Length": cup.StringLengthSettings(
                maximum_frequency=1.0, samples_per_peak=1, backend="numpy"
            ),
        },
    )
    assert isinstance(res, MultiResult)
    # Ignored (defaults) the grid would run to the pseudo-Nyquist limit, ~3 c/d here.
    assert res["String-Length"].frequency.max() < 1.01


def test_unknown_mapping_keys_raise() -> None:
    t, mag, err = synthetic_sine(n=60)
    with pytest.raises(cup.UnknownMethodError, match="GSL"):
        cup.periodogram((t, mag, err), "GLS", settings={"GSL": cup.GLSSettings()})
    # A plain kwargs dict is not a {method: settings} mapping.
    kwargs: Any = {"samples_per_peak": 1}
    with pytest.raises(cup.UnknownMethodError):
        cup.periodogram((t, mag, err), "GLS", settings=kwargs)


def test_superset_mapping_is_accepted() -> None:
    t, mag, err = synthetic_sine(n=200)
    pg = cup.periodogram(
        (t, mag, err),
        "GLS",
        settings={"GLS": cup.GLSSettings(samples_per_peak=2), "BLS": cup.BLSSettings()},
    )
    coarse = cup.periodogram(
        (t, mag, err), "GLS", settings=cup.GLSSettings(samples_per_peak=2)
    )
    assert isinstance(pg, Periodogram) and isinstance(coarse, Periodogram)
    assert pg.size == coarse.size


def test_duplicate_mapping_keys_raise() -> None:
    t, mag, err = synthetic_sine(n=60)
    with pytest.raises(ValueError, match="GLS"):
        cup.periodogram(
            (t, mag, err),
            "GLS",
            settings={"GLS": cup.GLSSettings(), "gls": cup.GLSSettings()},
        )


# --- default multi-band grids (CORE-A-12, MULTIBAND-3) --------------------------------


def _near_simultaneous_bands(period: float) -> cup.MultiBandLightCurve:
    """Five filters 72 s apart on 60 nights over ~9 years (SDSS Stripe 82-like)."""
    rng = np.random.default_rng(5)
    nights = np.sort(rng.choice(np.arange(3300), 60, replace=False)).astype(float)
    bands = {}
    for k, name in enumerate("ugriz"):
        t = 51000.0 + nights + k * 72.0 / 86400.0
        mag = 17 + 0.3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, t.size)
        bands[name] = cup.LightCurve.from_arrays(t, mag, np.full(t.size, 0.02))
    return cup.MultiBandLightCurve.from_light_curves(bands)


@pytest.mark.parametrize(
    "name", ["PDM", "CE", "MHAOV", "STRINGLENGTH", "SUPERSMOOTHER"]
)
def test_default_multiband_fold_grid_stays_bounded(name: str) -> None:
    # The stacked median gap is the 72 s inter-filter lag: the default ceiling used to
    # be ~3000 c/d and the grid ~5e7 trial frequencies (1.2 GB per star).
    from cuperiod.api import _multiband_grid

    period = 0.5731
    mblc = _near_simultaneous_bands(period)
    m = cup.get_method(name)
    grid = _multiband_grid(m, mblc, m.coerce_settings(None))
    assert grid.size < 1_000_000
    assert grid.frequency.min() <= 1.0 / period <= grid.frequency.max()


def test_one_band_multiband_grid_equals_the_single_band_grid() -> None:
    from cuperiod.api import _multiband_grid

    lc = _near_simultaneous_bands(0.5731).bands["g"]
    m = cup.get_method("PDM")
    one = cup.MultiBandLightCurve.from_light_curves({"g": lc})
    grid = _multiband_grid(m, one, m.coerce_settings(None))
    np.testing.assert_array_equal(
        grid.frequency, m.default_grid(lc, m.coerce_settings(None)).frequency
    )


# --- domain= on multi-band input (CORE-A-18) ------------------------------------------


def _two_bands() -> dict[str, cup.LightCurve]:
    t, mag, err = synthetic_sine(n=80)
    return {
        "g": cup.LightCurve.from_arrays(t, mag, err),
        "r": cup.LightCurve.from_arrays(t, mag + 0.5, err),
    }


@pytest.mark.parametrize("domain", [cup.Domain.FLUX, "flux"])
def test_domain_applies_to_every_band(domain: Any) -> None:
    bands = _two_bands()
    mblc = cup.MultiBandLightCurve.from_light_curves(bands, meta={"id": 7})
    for data in (bands, mblc):
        out = cup.to_input(data, domain=domain)
        assert isinstance(out, cup.MultiBandLightCurve)
        assert [lc.domain for lc in out.bands.values()] == [cup.Domain.FLUX] * 2
        for name, lc in out.bands.items():
            np.testing.assert_allclose(lc.value, bands[name].as_flux().value)
    out = cup.to_input(mblc, domain=domain)
    assert isinstance(out, cup.MultiBandLightCurve)
    assert out.meta == {"id": 7}


def test_domain_unifies_mixed_band_domains() -> None:
    bands = _two_bands()
    mixed = {"g": bands["g"], "r": bands["r"].as_flux()}
    out = cup.to_input(mixed, domain="flux")
    assert isinstance(out, cup.MultiBandLightCurve)
    assert {lc.domain for lc in out.bands.values()} == {cup.Domain.FLUX}
    np.testing.assert_allclose(out.bands["r"].value, bands["r"].as_flux().value)


def test_mixed_band_domains_without_domain_raise() -> None:
    # A magnitude band and a flux band vary in anti-phase; the joint 'offsets' fit
    # used to run silently and lose the signal (1/40 recovered in the audit).
    bands = _two_bands()
    mixed = {"g": bands["g"], "r": bands["r"].as_flux()}
    with pytest.raises(ValueError, match="domain="):
        cup.to_input(mixed)
    with pytest.raises(ValueError, match="domain="):
        cup.periodogram(cup.MultiBandLightCurve(bands=mixed), "GLS")


def test_non_lightcurve_band_raises_a_type_error() -> None:
    t, mag, err = synthetic_sine(n=60)
    bad: Any = {"g": (t, mag, err)}
    with pytest.raises(TypeError, match="'g'"):
        cup.periodogram(cup.MultiBandLightCurve(bands=bad), "GLS")


def test_string_domain_keeps_a_flux_curve_in_flux() -> None:
    t, mag, err = synthetic_sine(n=60)
    flux_lc = cup.LightCurve.from_arrays(t, mag, err).as_flux()
    out = cup.to_input(flux_lc, domain="flux")
    assert isinstance(out, cup.LightCurve)
    assert out.domain is cup.Domain.FLUX
    np.testing.assert_array_equal(out.value, flux_lc.value)


# --- an explicit band column is honoured (CORE-A-6) -----------------------------------


def _ztf_like() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    nights = np.arange(240, dtype=float)
    t = 58000.0 + nights + rng.uniform(-0.2, 0.2, nights.size)
    band = np.where(nights % 2 == 0, "zg", "zr")
    offset = np.where(band == "zg", 16.9, 16.3)
    mag = offset + 0.3 * np.sin(2 * np.pi * t / 0.5731) + rng.normal(0, 0.02, t.size)
    return {"mjd": t, "mag": mag, "magerr": np.full(t.size, 0.02), "filtercode": band}


def test_explicit_band_column_gives_a_multiband_curve() -> None:
    pd = pytest.importorskip("pandas")
    from astropy.table import Table

    data = _ztf_like()
    cmap = cup.ColumnMap(band="filtercode")
    for table in (pd.DataFrame(data), data, Table(data)):
        out = cup.to_input(table, columns=cmap)
        assert isinstance(out, cup.MultiBandLightCurve)
        assert set(out.band_names) == {"zg", "zr"}
        assert sum(lc.n for lc in out.bands.values()) == 240
    pg = cup.periodogram(pd.DataFrame(data), "GLS", columns=cmap)
    assert isinstance(pg, Periodogram)
    assert pg.best_period() == pytest.approx(0.5731, rel=2e-3)


def test_without_a_band_column_the_table_stays_single_band() -> None:
    pd = pytest.importorskip("pandas")
    data = _ztf_like()
    del data["filtercode"]
    assert isinstance(cup.to_input(pd.DataFrame(data)), cup.LightCurve)


# --- flux tuple with a flux-natural method (METH-A-21) --------------------------------


def test_flux_tuple_without_domain_warns_for_box_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t, flux, err = synthetic_eclipser(n=200)
    _spy_resolver(monkeypatch)
    for name in ("BLS", "TLS"):
        with pytest.warns(UserWarning, match="domain='flux'"), pytest.raises(_Abort):
            cup.periodogram((t, flux, err), name)


@pytest.mark.parametrize(
    ("kind", "method", "kwargs"),
    [
        ("flux", "BLS", {"domain": "flux"}),
        ("mag", "BLS", {}),
        ("delta", "BLS", {}),
        ("flux", "GLS", {}),
        ("lightcurve", "BLS", {}),
    ],
)
def test_no_flux_warning_otherwise(
    kind: str, method: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    t, flux, err = synthetic_eclipser(n=200)
    data: Any = {
        "flux": (t, flux, err),
        "mag": (t, 12.0 - 2.5 * np.log10(flux), err),
        "delta": (t, flux - 1.0, err),
        "lightcurve": cup.LightCurve.from_arrays(t, flux, err),
    }[kind]
    _spy_resolver(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_Abort):
            cup.periodogram(data, method, **kwargs)


# --- an unusable error column is named as such (CORE-A-14) ----------------------------

_PLACEHOLDER_ERRORS = {
    "zeros": lambda err: np.zeros_like(err),
    "nan": lambda err: np.full_like(err, np.nan),
    "minus_one": lambda err: -np.ones_like(err),
}


@pytest.mark.parametrize("placeholder", sorted(_PLACEHOLDER_ERRORS))
@pytest.mark.parametrize("name", cup.method_names())
def test_placeholder_error_column_is_reported(name: str, placeholder: str) -> None:
    # An all-0/-1/NaN error column empties the curve; GLS, BLS, TLS and SuperSmoother
    # used to blame it on "no usable time baseline" or a too-short baseline.
    t, mag, err = synthetic_sine()
    bad = _PLACEHOLDER_ERRORS[placeholder](err)
    with pytest.raises(cup.InsufficientDataError, match="usable error"):
        cup.periodogram((t, mag, bad), name)


def test_placeholder_error_column_is_reported_for_explicit_finufft() -> None:
    t, mag, err = synthetic_sine()
    with pytest.raises(cup.InsufficientDataError, match="GLS.*usable error"):
        cup.periodogram((t, mag, np.zeros_like(err)), "GLS", backend="finufft")


def test_placeholder_error_columns_are_reported_for_multiband() -> None:
    bands = {
        name: cup.LightCurve.from_arrays(lc.time, lc.value, np.zeros(lc.n))
        for name, lc in _two_bands().items()
    }
    mblc = cup.MultiBandLightCurve.from_light_curves(bands)
    with pytest.raises(cup.InsufficientDataError, match="multiband.*band 'g'"):
        cup.periodogram(mblc, "GLS")


def test_partly_bad_error_column_still_runs() -> None:
    t, mag, err = synthetic_sine()
    err = err.copy()
    err[::10] = 0.0  # 10% unusable: those points are dropped, the rest analysed
    pg = cup.periodogram((t, mag, err), "GLS")
    assert isinstance(pg, Periodogram)
    assert np.isfinite(pg.best_period())
