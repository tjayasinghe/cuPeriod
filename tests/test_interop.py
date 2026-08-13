"""LINCC-stack interoperability: nested-pandas row-wise and partition-wise adapters."""

from __future__ import annotations

import pickle

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
