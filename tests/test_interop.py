"""LINCC-stack interoperability: nested-pandas row-wise and partition-wise adapters."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

pytest.importorskip("nested_pandas")

import pandas as pd  # noqa: E402
from nested_pandas import NestedFrame  # noqa: E402

import cuperiod as cup  # noqa: E402
from cuperiod.core.errors import ColumnResolutionError  # noqa: E402
from cuperiod.interop import (  # noqa: E402
    COLUMN_PRESETS,
    PRESET_KEYS,
    nested_periodogram,
    partition_periodogram,
    resolve_nested_columns,
)

PERIOD = 0.7
F_TRUE = 1.0 / PERIOD
N_OBJECTS = 6  # the last one is the 3-point object that must come back NaN
N_POINTS = 60
BASELINE = 30.0

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def make_grid(nf: int = 601, half_width: float = 0.35) -> cup.GridSpec:
    """A narrow uniform frequency grid bracketing the planted frequency."""
    freq = np.linspace(F_TRUE - half_width, F_TRUE + half_width, nf)
    return cup.GridSpec(kind="frequency", values=freq, uniform=True)


def _sine(rng: np.random.Generator, n: int, phase: float) -> Arrays:
    time = np.sort(rng.uniform(0.0, BASELINE, n))
    mag = 15.0 + 0.3 * np.sin(2 * np.pi * time / PERIOD + phase)
    return time, mag + rng.normal(0.0, 0.005, n), np.full(n, 0.01)


def make_objects(seed: int = 11) -> list[Arrays]:
    """Per-object ``(time, mag, err)``; the last object has only 3 points."""
    rng = np.random.default_rng(seed)
    return [
        _sine(rng, 3 if i == N_OBJECTS - 1 else N_POINTS, phase=0.3 * i)
        for i in range(N_OBJECTS)
    ]


def frame_from_objects(objects: list[Arrays]) -> NestedFrame:
    """Nest ``(time, mag, err)`` per object as ``lc.hmjd``/``lc.mag``/``lc.magerr``."""
    flat = pd.DataFrame(
        {
            "id": np.concatenate(
                [np.full(t.size, i) for i, (t, _, _) in enumerate(objects)]
            ),
            "hmjd": np.concatenate([t for t, _, _ in objects]),
            "mag": np.concatenate([m for _, m, _ in objects]),
            "magerr": np.concatenate([e for _, _, e in objects]),
        }
    ).set_index("id")
    base = pd.DataFrame(
        {"id": np.arange(len(objects)), "ra": np.arange(len(objects)) * 1.0}
    ).set_index("id")
    return NestedFrame(base).join_nested(flat, name="lc")


def make_frame(seed: int = 11) -> NestedFrame:
    return frame_from_objects(make_objects(seed))


def make_multiband_frame(seed: int = 5) -> NestedFrame:
    """Two bands per object inside the nest (``lc.band``), same planted period."""
    rng = np.random.default_rng(seed)
    ids, times, mags, errs, bands = [], [], [], [], []
    for i in range(4):
        for offset, name in ((15.0, "g"), (14.2, "r")):
            time, mag, err = _sine(rng, 40, phase=0.4 * i)
            ids.append(np.full(time.size, i))
            times.append(time)
            mags.append(mag - 15.0 + offset)
            errs.append(err)
            bands.append(np.full(time.size, name, dtype=object))
    flat = pd.DataFrame(
        {
            "id": np.concatenate(ids),
            "hmjd": np.concatenate(times),
            "mag": np.concatenate(mags),
            "magerr": np.concatenate(errs),
            "band": np.concatenate(bands),
        }
    ).set_index("id")
    base = pd.DataFrame({"id": np.arange(4), "ra": np.arange(4) * 1.0}).set_index("id")
    return NestedFrame(base).join_nested(flat, name="lc")


def make_rubin_frame(columns: tuple[str, ...] | None = None) -> NestedFrame:
    """A synthetic frame with the verified Rubin DP1 names and negative fluxes."""
    rng = np.random.default_rng(3)
    n = 40
    time = np.sort(rng.uniform(0.0, BASELINE, n))
    # Difference-imaging fluxes straddle zero: no magnitude exists for half of them.
    flux = 5.0 + 20.0 * np.sin(2 * np.pi * time / PERIOD)
    with np.errstate(invalid="ignore", divide="ignore"):
        psf_mag = 31.4 - 2.5 * np.log10(flux)
    nest = {
        "midpointMjdTai": time,
        "psfFlux": flux,
        "psfFluxErr": np.full(n, 1.0),
        "psfMag": psf_mag,
        "band": np.array(["g", "r"] * (n // 2), dtype=object),
    }
    keep = nest if columns is None else {k: nest[k] for k in columns}
    flat = pd.DataFrame({"id": np.zeros(n, dtype=int), **keep}).set_index("id")
    base = pd.DataFrame({"id": [0], "ra": [1.0]}).set_index("id")
    return NestedFrame(base).join_nested(flat, name="objectForcedSource")


# --- tier 1: row-wise ---------------------------------------------------------


def test_nested_periodogram_recovers_period() -> None:
    """Every well-sampled row recovers the planted period; the tiny one is NaN."""
    out = nested_periodogram(make_frame(), "lc", grid=make_grid())
    assert isinstance(out, NestedFrame)
    periods = out["best_period"].to_numpy()
    assert periods.size == N_OBJECTS
    assert np.allclose(periods[:-1], PERIOD, rtol=2e-2)
    assert np.isnan(periods[-1])
    assert np.isnan(out["best_power"].to_numpy()[-1])
    assert np.all(np.isfinite(out["best_power"].to_numpy()[:-1]))


def test_nested_periodogram_appends_and_keeps_base_columns() -> None:
    out = nested_periodogram(make_frame(), "lc", grid=make_grid(), append_columns=True)
    assert "ra" in out.columns
    assert "lc" in out.columns
    assert {"best_period", "best_power", "fap"} <= set(out.columns)


def test_nested_periodogram_without_append_returns_results_only() -> None:
    out = nested_periodogram(make_frame(), "lc", grid=make_grid(), append_columns=False)
    assert list(out.columns) == ["best_period", "best_power", "fap"]


def test_prefix_and_n_best_column_names() -> None:
    out = nested_periodogram(
        make_frame(), "lc", grid=make_grid(), n_best=3, prefix="gls_",
        append_columns=False,
    )
    assert list(out.columns) == [
        "gls_best_period", "gls_best_power", "gls_fap", "gls_period_2", "gls_period_3",
    ]
    assert out["gls_best_period"].to_numpy()[0] == pytest.approx(PERIOD, rel=2e-2)
    assert np.all(np.isnan(out.iloc[-1].to_numpy()))


def test_fap_column_is_populated_for_gls() -> None:
    """GLS reports a false-alarm probability at the best peak."""
    out = nested_periodogram(make_frame(), "lc", grid=make_grid(), append_columns=False)
    fap = out["fap"].to_numpy()
    assert np.all(np.isfinite(fap[:-1]))
    assert np.isnan(fap[-1])


def test_fap_is_nan_when_the_method_reports_none() -> None:
    out = nested_periodogram(
        make_frame(),
        "lc",
        grid=make_grid(),
        settings=cup.GLSSettings(fap_method="none"),
        append_columns=False,
    )
    assert np.all(np.isnan(out["fap"].to_numpy()))
    assert out["best_period"].to_numpy()[0] == pytest.approx(PERIOD, rel=2e-2)


def test_dotted_and_bare_column_names_agree() -> None:
    frame, grid = make_frame(), make_grid()
    bare = nested_periodogram(
        frame, "lc", time="hmjd", value="mag", error="magerr",
        grid=grid, append_columns=False,
    )
    dotted = nested_periodogram(
        frame, "lc", time="lc.hmjd", value="lc.mag", error="lc.magerr",
        grid=grid, append_columns=False,
    )
    assert np.allclose(
        bare["best_period"].to_numpy(), dotted["best_period"].to_numpy(),
        rtol=1e-12, equal_nan=True,
    )


# --- multi-band ---------------------------------------------------------------


def test_multiband_in_nest_band_recovers_period() -> None:
    out = nested_periodogram(
        make_multiband_frame(), "lc", band="lc.band", grid=make_grid(),
        append_columns=False,
    )
    periods = out["best_period"].to_numpy()
    assert periods.size == 4
    assert np.allclose(periods, PERIOD, rtol=2e-2)


def test_band_is_not_auto_detected() -> None:
    """A nest with a band sub-column stays single-band unless band= is given."""
    frame = make_multiband_frame()
    assert resolve_nested_columns(frame, "lc").band is None
    assert resolve_nested_columns(frame, "lc", band="band").band == "lc.band"


def test_multiband_with_single_band_method_raises() -> None:
    """TLS has no multi-band model: raise up front, never a frame of NaNs."""
    frame, grid = make_multiband_frame(), make_grid()
    for run in (nested_periodogram, partition_periodogram):
        with pytest.raises(ValueError, match="does not support multi-band"):
            run(frame, "lc", band="lc.band", method="TLS", grid=grid)


def test_unknown_band_column_raises() -> None:
    with pytest.raises(ColumnResolutionError, match="band column"):
        nested_periodogram(make_frame(), "lc", band="filterid", grid=make_grid())


# --- tier 2: partition-wise ---------------------------------------------------


def test_partition_matches_row_wise() -> None:
    frame, grid = make_frame(), make_grid()
    row_wise = nested_periodogram(frame, "lc", grid=grid, append_columns=False)
    batched = partition_periodogram(frame, "lc", grid=grid)
    assert isinstance(batched, pd.DataFrame)
    assert list(batched.columns) == ["best_period", "best_power", "fap"]
    assert np.allclose(
        batched["best_period"].to_numpy(), row_wise["best_period"].to_numpy(),
        rtol=1e-9, equal_nan=True,
    )
    # The batched tier reuses one bucketed NUFFT plan for every object, so its powers
    # agree with the per-object transforms to the NUFFT tolerance, not bit for bit.
    assert np.allclose(
        batched["best_power"].to_numpy(), row_wise["best_power"].to_numpy(),
        rtol=1e-6, equal_nan=True,
    )
    assert np.isnan(batched["best_period"].to_numpy()[-1])


def test_partition_multiband_matches_row_wise() -> None:
    frame, grid = make_multiband_frame(), make_grid()
    row_wise = nested_periodogram(
        frame, "lc", band="lc.band", grid=grid, append_columns=False
    )
    batched = partition_periodogram(frame, "lc", band="lc.band", grid=grid)
    assert np.allclose(
        batched["best_period"].to_numpy(), row_wise["best_period"].to_numpy(), rtol=1e-9
    )


def test_partition_preserves_index_and_n_best() -> None:
    out = partition_periodogram(
        make_frame(), "lc", grid=make_grid(), n_best=2, prefix="p_"
    )
    assert list(out.columns) == ["p_best_period", "p_best_power", "p_fap", "p_period_2"]
    assert out.index.equals(make_frame().index)


def test_partition_on_empty_frame_returns_typed_empty() -> None:
    out = partition_periodogram(make_frame().iloc[:0], "lc", grid=make_grid())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert list(out.columns) == ["best_period", "best_power", "fap"]
    assert all(dtype == np.float64 for dtype in out.dtypes)


def test_partition_slices_each_object_correctly() -> None:
    """Ragged nests: every object's slice must carry exactly its own epochs."""
    objects, grid = make_objects(), make_grid()
    batched = partition_periodogram(frame_from_objects(objects), "lc", grid=grid)
    for i, (time, mag, err) in enumerate(objects[:-1]):
        pg = cup.periodogram((time, mag, err), "GLS", grid=grid)
        assert isinstance(pg, cup.Periodogram)
        assert batched["best_period"].to_numpy()[i] == pytest.approx(
            pg.best_period(), rel=1e-12
        )


# --- the lsdb (lazy catalog) branch ------------------------------------------


class _FakeCatalog:
    """Duck-typed stand-in for a lazy lsdb ``Catalog`` (lsdb is not a test dep).

    Mirrors the two lsdb entry points the adapter uses: ``map_rows`` with a mandatory
    ``meta`` keyword and ``map_partitions`` with an optional one.
    """

    def __init__(self, frame: NestedFrame) -> None:
        self._frame = frame
        self.meta: object = None

    @property
    def dtypes(self) -> pd.Series:
        return self._frame.dtypes

    def map_rows(self, func: object, columns: object = None, *, meta: object,
                 **kwargs: object) -> NestedFrame:
        self.meta = meta
        return self._frame.map_rows(func, columns, **kwargs)  # type: ignore[arg-type]

    def map_partitions(self, func: object, *args: object, meta: object = None,
                       **kwargs: object) -> pd.DataFrame:
        self.meta = meta
        return func(self._frame)  # type: ignore[operator]


_FakeCatalog.__module__ = "lsdb.catalog.catalog"


def test_catalog_path_passes_auto_built_meta() -> None:
    catalog = _FakeCatalog(make_frame())
    out = nested_periodogram(
        catalog, "lc", grid=make_grid(), n_best=2, append_columns=True
    )
    assert catalog.meta == {
        "best_period": float, "best_power": float, "fap": float, "period_2": float,
    }
    assert out["best_period"].to_numpy()[0] == pytest.approx(PERIOD, rel=2e-2)


def test_catalog_partition_path_passes_empty_typed_meta() -> None:
    catalog = _FakeCatalog(make_frame())
    out = partition_periodogram(catalog, "lc", grid=make_grid())
    meta = catalog.meta
    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == 0
    assert list(meta.columns) == ["best_period", "best_power", "fap"]
    assert all(dtype == np.float64 for dtype in meta.dtypes)
    assert out["best_period"].to_numpy()[0] == pytest.approx(PERIOD, rel=2e-2)


def test_explicit_meta_is_forwarded_unchanged() -> None:
    catalog = _FakeCatalog(make_frame())
    custom = {"best_period": "float64", "best_power": "float64", "fap": "float64"}
    nested_periodogram(catalog, "lc", grid=make_grid(), meta=custom)
    assert catalog.meta is custom


def test_kernels_are_picklable() -> None:
    """dask ships the kernel to workers; it must never carry a compute engine."""
    from cuperiod.interop.lincc import _PartitionKernel, _RowKernel

    columns = resolve_nested_columns(make_frame(), "lc")
    for cls in (_RowKernel, _PartitionKernel):
        kernel = cls(
            columns=columns, method="GLS", settings=cup.GLSSettings(),
            backend="cpu", grid=make_grid(), n_best=2, prefix="p_",
        )
        kernel.prepare()
        clone = pickle.loads(pickle.dumps(kernel))
        assert clone.out_columns == kernel.out_columns
        assert clone.columns == columns


# --- presets and column resolution -------------------------------------------


def test_presets_have_the_documented_keys() -> None:
    assert set(COLUMN_PRESETS) == {
        "ztf_dr22", "ztf_alerts", "rubin_dp1_object", "rubin_dp1_dia",
    }
    for name, entry in COLUMN_PRESETS.items():
        assert set(entry) == set(PRESET_KEYS), name
        assert isinstance(entry["nested"], str)
        assert isinstance(entry["time"], str)
        assert isinstance(entry["value"], str)
        assert entry["domain"] in (cup.Domain.MAGNITUDE, cup.Domain.FLUX)


def test_ztf_dr22_preset_has_no_in_nest_band() -> None:
    """DR22 keeps the filter in the base column ``filterid`` (one row per filter)."""
    assert COLUMN_PRESETS["ztf_dr22"]["band"] is None
    assert COLUMN_PRESETS["ztf_alerts"]["band"] == "lc_fid"


def test_rubin_presets_search_in_flux() -> None:
    for name in ("rubin_dp1_object", "rubin_dp1_dia"):
        assert COLUMN_PRESETS[name]["domain"] is cup.Domain.FLUX


def test_rubin_preset_resolves_columns_and_flux_domain() -> None:
    columns = resolve_nested_columns(make_rubin_frame(), preset="rubin_dp1_object")
    assert columns.nested == "objectForcedSource"
    assert columns.time == "objectForcedSource.midpointMjdTai"
    assert columns.value == "objectForcedSource.psfFlux"
    assert columns.error == "objectForcedSource.psfFluxErr"
    assert columns.band == "objectForcedSource.band"
    assert columns.domain is cup.Domain.FLUX
    assert columns.subcolumns == ["midpointMjdTai", "psfFlux", "psfFluxErr", "band"]


def test_rubin_preset_runs_on_negative_fluxes() -> None:
    """Difference-imaging flux straddles zero; the flux-domain search must not care."""
    out = nested_periodogram(
        make_rubin_frame(), preset="rubin_dp1_object", grid=make_grid(),
        append_columns=False,
    )
    assert out["best_period"].to_numpy()[0] == pytest.approx(PERIOD, rel=2e-2)


def test_preset_error_and_band_are_dropped_when_absent() -> None:
    """A preset must not demand columns the nest does not carry."""
    frame = make_rubin_frame(columns=("midpointMjdTai", "psfFlux"))
    columns = resolve_nested_columns(frame, preset="rubin_dp1_object")
    assert columns.error is None
    assert columns.band is None
    assert columns.domain is cup.Domain.FLUX
    assert columns.read_columns == [
        "objectForcedSource.midpointMjdTai", "objectForcedSource.psfFlux",
    ]


def test_missing_nested_column_raises() -> None:
    with pytest.raises(ColumnResolutionError, match="no nested column"):
        resolve_nested_columns(make_frame(), "sources")


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_nested_columns(make_frame(), preset="nope")


def test_auto_detection_without_preset() -> None:
    columns = resolve_nested_columns(make_frame(), "lc")
    assert (columns.time, columns.value, columns.error) == (
        "lc.hmjd", "lc.mag", "lc.magerr",
    )
    assert columns.domain is cup.Domain.MAGNITUDE
    assert columns.read_columns == ["lc.hmjd", "lc.mag", "lc.magerr"]


def test_default_nest_is_lc_without_a_preset() -> None:
    assert resolve_nested_columns(make_frame()).nested == "lc"


# --- a preset with an overridden value column (BATCH-B-3) ---------------------

ECLIPSE_PERIOD = 2.37


def make_rubin_eclipse_frame(
    n: int = 300, with_mag_err: bool = True, name: str = "objectForcedSource"
) -> NestedFrame:
    """A 20%-deep eclipse in nJy fluxes, with the magnitudes DP1 carries beside them."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(60000.0, 60030.0, n))
    phase = ((t - 60000.3) / ECLIPSE_PERIOD) % 1.0
    in_eclipse = (phase < 0.04) | (phase > 0.96)
    flux_err = rng.uniform(300.0, 900.0, n)
    flux = 50_000.0 * np.where(in_eclipse, 0.80, 1.0) + rng.normal(0.0, flux_err)
    nest = {
        "midpointMjdTai": t,
        "psfFlux": flux,
        "psfFluxErr": flux_err,
        "psfMag": 31.4 - 2.5 * np.log10(flux),
    }
    if with_mag_err:
        nest["psfMagErr"] = 1.0857 * flux_err / flux
    flat = pd.DataFrame({"id": np.zeros(n, dtype=int), **nest}).set_index("id")
    base = pd.DataFrame({"id": [0], "ra": [1.0]}).set_index("id")
    return NestedFrame(base).join_nested(flat, name=name)


def test_overriding_the_preset_value_drops_its_error_and_domain() -> None:
    """psfMag must not be weighted by nJy errors nor searched as a flux."""
    columns = resolve_nested_columns(
        make_rubin_eclipse_frame(), preset="rubin_dp1_object", value="psfMag"
    )
    assert columns.value == "objectForcedSource.psfMag"
    assert columns.error == "objectForcedSource.psfMagErr"
    assert columns.domain is cup.Domain.MAGNITUDE
    # The preset's time column still applies.
    assert columns.time == "objectForcedSource.midpointMjdTai"


def test_overridden_value_without_its_own_error_runs_unweighted() -> None:
    frame = make_rubin_eclipse_frame(with_mag_err=False)
    columns = resolve_nested_columns(frame, preset="rubin_dp1_object", value="psfMag")
    assert columns.error is None
    assert columns.domain is cup.Domain.MAGNITUDE


def test_value_matching_the_preset_keeps_its_error_and_domain() -> None:
    """Only a *different* value column cuts the preset's pairing (case-insensitive)."""
    frame = make_rubin_eclipse_frame()
    for value in ("PSFFLUX", "objectForcedSource.psfFlux"):
        columns = resolve_nested_columns(frame, preset="rubin_dp1_object", value=value)
        assert columns.error == "objectForcedSource.psfFluxErr"
        assert columns.domain is cup.Domain.FLUX


def test_explicit_error_and_domain_still_win_over_an_overridden_value() -> None:
    columns = resolve_nested_columns(
        make_rubin_eclipse_frame(),
        preset="rubin_dp1_object",
        value="psfMag",
        error="psfFluxErr",
        domain="flux",
    )
    assert columns.error == "objectForcedSource.psfFluxErr"
    assert columns.domain is cup.Domain.FLUX


def test_bls_on_an_overridden_magnitude_column_finds_the_eclipse() -> None:
    """The flux-natural BLS searched an inverted signal before (best 0.92 d)."""
    out = nested_periodogram(
        make_rubin_eclipse_frame(),
        preset="rubin_dp1_object",
        value="psfMag",
        method="BLS",
        backend="cpu",
        append_columns=False,
    )
    assert out["best_period"].to_numpy()[0] == pytest.approx(ECLIPSE_PERIOD, rel=0.01)


# --- an explicit nest name beside a preset (BATCH-B-4) ------------------------


def make_two_nest_frame() -> NestedFrame:
    """The survey nest (P = 3.1 d) and the user's own cleaned ``lc`` (P = 0.9 d)."""
    rng = np.random.default_rng(21)
    base = pd.DataFrame({"id": [0], "ra": [1.0]}).set_index("id")
    frame = NestedFrame(base)
    for name, period in (("objectForcedSource", 3.1), ("lc", 0.9)):
        t = np.sort(rng.uniform(0.0, 60.0, 200))
        flux = 100.0 + 20.0 * np.sin(2 * np.pi * t / period)
        flat = pd.DataFrame(
            {
                "id": np.zeros(t.size, dtype=int),
                "midpointMjdTai": t,
                "psfFlux": flux + rng.normal(0.0, 0.5, t.size),
                "psfFluxErr": np.full(t.size, 0.5),
            }
        ).set_index("id")
        frame = frame.join_nested(flat, name=name)
    return frame


def test_explicit_lc_nest_beats_the_preset_nest() -> None:
    frame = make_two_nest_frame()
    columns = resolve_nested_columns(frame, "lc", preset="rubin_dp1_object")
    assert columns.nested == "lc"
    assert columns.time == "lc.midpointMjdTai"
    assert columns.domain is cup.Domain.FLUX
    keyword = resolve_nested_columns(frame, nested="lc", preset="rubin_dp1_object")
    assert keyword == columns
    # Both tiers must search the nest that was asked for.
    row_wise = nested_periodogram(
        frame, "lc", preset="rubin_dp1_object", backend="cpu", append_columns=False
    )
    batched = partition_periodogram(
        frame, "lc", preset="rubin_dp1_object", backend="cpu"
    )
    for out in (row_wise, batched):
        assert out["best_period"].to_numpy()[0] == pytest.approx(0.9, rel=0.01)


def test_preset_alone_still_picks_its_own_nest() -> None:
    columns = resolve_nested_columns(make_two_nest_frame(), preset="rubin_dp1_object")
    assert columns.nested == "objectForcedSource"


def test_preset_on_a_frame_whose_only_nest_is_lc() -> None:
    frame = make_rubin_eclipse_frame(name="lc")
    columns = resolve_nested_columns(frame, "lc", preset="rubin_dp1_object")
    assert columns.time == "lc.midpointMjdTai"
    assert columns.value == "lc.psfFlux"
    assert columns.domain is cup.Domain.FLUX


# --- empty frames and colliding result columns (BATCH-B-8, BATCH-B-13) --------


@pytest.mark.parametrize("append", [False, True])
def test_row_wise_on_empty_frame_returns_typed_empty(append: bool) -> None:
    frame = make_frame()
    reference = nested_periodogram(
        frame, "lc", grid=make_grid(), n_best=2, prefix="gls_", backend="cpu",
        append_columns=append,
    )
    out = nested_periodogram(
        frame.iloc[:0], "lc", grid=make_grid(), n_best=2, prefix="gls_",
        append_columns=append,
    )
    assert isinstance(out, NestedFrame)
    assert len(out) == 0
    assert list(out.columns) == list(reference.columns)
    results = ["gls_best_period", "gls_best_power", "gls_fap", "gls_period_2"]
    assert all(out[name].dtype == np.float64 for name in results)
    assert out.index.name == reference.index.name


def test_empty_frame_still_validates_the_configuration() -> None:
    empty = make_multiband_frame().iloc[:0]
    with pytest.raises(ValueError, match="does not support multi-band"):
        nested_periodogram(empty, "lc", band="lc.band", method="TLS", grid=make_grid())


def test_appending_onto_existing_result_columns_raises() -> None:
    frame, grid = make_frame(), make_grid()
    first = nested_periodogram(frame, "lc", grid=grid, backend="cpu")
    with pytest.raises(ValueError, match="prefix") as info:
        nested_periodogram(first, "lc", method="PDM", grid=grid)
    assert "best_period" in str(info.value)
    # The escape hatches named in the message both work.
    second = nested_periodogram(
        first, "lc", method="PDM", grid=grid, prefix="pdm_", backend="cpu"
    )
    assert second.columns.is_unique
    assert {"best_period", "pdm_best_period"} <= set(second.columns)
    only = nested_periodogram(
        first, "lc", method="PDM", grid=grid, append_columns=False, backend="cpu"
    )
    assert list(only.columns) == ["best_period", "best_power", "fap"]


def test_collision_lists_only_the_clashing_names() -> None:
    frame = make_frame()
    frame["fap"] = 0.0
    with pytest.raises(ValueError, match="prefix") as info:
        nested_periodogram(frame, "lc", grid=make_grid(), n_best=2)
    message = str(info.value)
    assert "'fap'" in message
    assert "period_2" not in message and "best_power" not in message


def test_catalog_collision_is_caught_from_the_schema() -> None:
    first = nested_periodogram(make_frame(), "lc", grid=make_grid(), backend="cpu")
    catalog = _FakeCatalog(first)
    with pytest.raises(ValueError, match="prefix"):
        nested_periodogram(catalog, "lc", grid=make_grid())
    assert catalog.meta is None  # refused before map_rows was called


# --- backend and peak defaults come from the settings -------------------------


def test_settings_backend_is_honoured() -> None:
    """``GLSSettings(backend=...)`` (and CUPERIOD_GLS_BACKEND) pick the backend."""
    from cuperiod.interop.lincc import _RowKernel

    columns = resolve_nested_columns(make_frame(), "lc")
    kernel = _RowKernel(
        columns=columns, method="GLS", settings=cup.GLSSettings(backend="astropy"),
        backend="auto", grid=make_grid(), n_best=1, prefix="",
    )
    assert kernel.prepare()[2] == "astropy"
    explicit = _RowKernel(
        columns=columns, method="GLS", settings=cup.GLSSettings(backend="astropy"),
        backend="cpu", grid=make_grid(), n_best=1, prefix="",
    )
    assert explicit.prepare()[2] != "astropy"


def test_settings_backend_env_override_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuperiod.interop.lincc import _RowKernel

    monkeypatch.setenv("CUPERIOD_GLS_BACKEND", "astropy")
    kernel = _RowKernel(
        columns=resolve_nested_columns(make_frame(), "lc"), method="GLS",
        settings=None, backend="auto", grid=make_grid(), n_best=1, prefix="",
    )
    assert kernel.prepare()[2] == "astropy"


def test_results_carry_the_settings_peak_defaults() -> None:
    """period_2 is picked with the settings' separation, as in cup.periodogram."""
    from cuperiod.interop.lincc import _RowKernel

    objects = make_objects()
    time, mag, err = objects[0]
    settings = cup.GLSSettings(peak_separation_rayleigh=40.0)
    kernel = _RowKernel(
        columns=resolve_nested_columns(make_frame(), "lc"), method="GLS",
        settings=settings, backend="cpu", grid=make_grid(), n_best=2, prefix="",
    )
    method, coerced, backend = kernel.prepare()
    pg = kernel.compute(
        method, coerced, backend, kernel.light_curve(time, mag, err, None), None
    )
    assert pg.peak_defaults["min_separation_rayleigh"] == 40.0
    reference = cup.periodogram((time, mag, err), "GLS", grid=make_grid(),
                                settings=settings, backend="cpu")
    assert isinstance(reference, cup.Periodogram)
    expected = [p.period for p in reference.best_periods(2)]
    assert [p.period for p in pg.best_periods(2)] == pytest.approx(expected)


# --- engine teardown ------------------------------------------------------------


def _partition_kernel(backend: str) -> Any:
    from cuperiod.interop.lincc import _PartitionKernel

    return _PartitionKernel(
        columns=resolve_nested_columns(make_frame(), "lc"), method="GLS",
        settings=cup.GLSSettings(), backend=backend, grid=make_grid(), n_best=1,
        prefix="",
    )


def test_cpu_engine_partition_does_not_touch_the_gpu_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLS on finufft builds a CPU engine; tearing it down must not flush cupy."""
    pytest.importorskip("finufft")
    import cuperiod.core.device as device

    calls: list[int] = []
    monkeypatch.setattr(device, "free_gpu_memory", lambda: calls.append(1))
    kernel = _partition_kernel("finufft")
    out = kernel(make_frame())
    assert kernel.prepare()[2] == "finufft"
    assert np.isfinite(out["best_period"].to_numpy()[:-1]).all()
    assert calls == []


def test_gpu_engine_partition_frees_the_gpu_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU engine still returns its cached blocks after the partition."""
    import cuperiod.core.device as device

    calls: list[int] = []
    monkeypatch.setattr(device, "free_gpu_memory", lambda: calls.append(1))
    kernel = _partition_kernel("cpu")
    method, settings, _ = kernel.prepare()
    # Pretend the backend resolved to the CUDA engine without running on a GPU.
    monkeypatch.setattr(method, "make_engine", lambda backend, settings: object())
    kernel._prepared = (method, settings, "cufinufft")
    monkeypatch.setattr(kernel, "evaluate", lambda *a, **k: kernel.nan_row())
    kernel(make_frame())
    assert calls == [1]
