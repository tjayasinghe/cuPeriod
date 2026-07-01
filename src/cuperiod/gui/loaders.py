"""Loading light curves for the GUI: arbitrary files plus the bundled demos.

:func:`load_path` reads a single light-curve file through the same auto-detecting loader
the CLI uses. :func:`demo_sources` returns ready-to-explore :class:`SourceItem`s for the
example data in ``examples/data`` (a Kepler transit and six ASAS-SN stars), plus an
always-available synthetic multiband curve, so the app has something to show on first
launch even outside a source checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cuperiod.batch.io import resolve_inputs
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import ColumnResolutionError
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.models import LoadedCurve, SourceItem

#: Environment override pointing at a directory of example data.
_ENV_DATA_DIR = "CUPERIOD_EXAMPLE_DATA"


def _read_dataframe(path: Path) -> Any:
    """Read a tabular file into a pandas DataFrame, or None if unreadable."""
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".tsv", ".tab"}:
            return pd.read_csv(path, sep="\t")
    except Exception:  # noqa: BLE001 - fall back to single-band loading
        return None
    return None


def load_path(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> LoadedCurve:
    """Load a light curve from a file, returning multiband if a band column is present.

    Tabular files (CSV/TSV/Parquet) are probed for a band/filter column; when one is
    found the result is a :class:`MultiBandLightCurve`, otherwise a single
    :class:`LightCurve` (also the path for FITS/ECSV).
    """
    frame = _read_dataframe(Path(path))
    if frame is not None:
        try:
            return MultiBandLightCurve.from_dataframe(
                frame, columns=columns, domain=domain
            )
        except ColumnResolutionError:
            pass  # no band column -> single-band below
    return LightCurve.from_file(path, columns=columns, domain=domain)


@dataclass(frozen=True)
class FilePreview:
    """A cheap preview of a tabular file: first rows + the detected column mapping."""

    columns: list[str]
    rows: list[list[str]]
    n_rows: int
    mapping: dict[str, str]  # role -> column (time/value/error/band)
    domain: str
    multiband: bool


def _fmt_cell(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def preview_file(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    max_rows: int = 12,
) -> FilePreview | None:
    """First rows of a tabular file plus its auto-detected column mapping.

    Returns None for formats not cheaply previewable here (e.g. FITS/ECSV); the caller
    then simply loads without a preview.
    """
    frame = _read_dataframe(Path(path))
    if frame is None:
        return None
    names = [str(c) for c in frame.columns]
    mapping: dict[str, str] = {}
    resolved_domain = "?"
    multiband = False
    try:
        resolved = (columns or ColumnMap()).resolve(names, domain=domain)
    except ColumnResolutionError:
        resolved = None
    if resolved is not None:
        mapping["time"] = resolved.time
        mapping["value"] = resolved.value
        if resolved.error:
            mapping["error"] = resolved.error
        if resolved.band:
            mapping["band"] = resolved.band
            multiband = True
        resolved_domain = resolved.domain.value
    head = frame.head(max_rows)
    rows = [[_fmt_cell(head.iloc[i][c]) for c in names] for i in range(len(head))]
    return FilePreview(
        names, rows, int(len(frame)), mapping, resolved_domain, multiband
    )


def example_data_dir() -> Path | None:
    """Locate the ``examples/data`` directory, or None if it is not on disk.

    Checks ``$CUPERIOD_EXAMPLE_DATA`` first, then walks up from this file to find an
    ``examples/data`` folder (present in a source checkout / editable install).
    """
    override = os.environ.get(_ENV_DATA_DIR)
    if override and Path(override).is_dir():
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples" / "data"
        if candidate.is_dir():
            return candidate
    return None


# -- ASAS-SN wide-format adapter ---------------------------------------------------


@lru_cache(maxsize=1)
def _asassn_frame(parquet_path: str) -> Any:
    """Read the ASAS-SN demo parquet into a pandas DataFrame (cached)."""
    import pandas as pd

    return pd.read_parquet(parquet_path)


def _lightcurve_from_asassn_row(row: Any) -> LightCurve:
    """Build a :class:`LightCurve` from one wide-format ASAS-SN row (list columns)."""
    return LightCurve.from_arrays(
        np.asarray(row.jd, dtype=np.float64),
        np.asarray(row.mag, dtype=np.float64),
        np.asarray(row.mag_err, dtype=np.float64),
        domain=Domain.MAGNITUDE,
        meta={
            "id": str(row.asas_sn_id),
            "band": str(row.band),
            "vsx_type": str(row.vsx_type),
            "broad_class": str(row.broad_class),
            "vsx_period": float(row.vsx_period),
        },
    )


def asassn_sources(parquet_path: Path) -> list[SourceItem]:
    """One :class:`SourceItem` per row of the ASAS-SN demo parquet (lazy loaders)."""
    frame = _asassn_frame(str(parquet_path))
    items: list[SourceItem] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        key = f"asassn_{row.asas_sn_id}"
        label = f"{row.asas_sn_id} — {row.vsx_type} (P={float(row.vsx_period):.3f} d)"

        def _loader(r: Any = row) -> LoadedCurve:
            return _lightcurve_from_asassn_row(r)

        items.append(
            SourceItem(
                key=key,
                label=label,
                loader=_loader,
                meta={
                    "vsx_period": float(row.vsx_period),
                    "vsx_type": str(row.vsx_type),
                },
            )
        )
    return items


# -- synthetic multiband (always available) ---------------------------------------


def make_demo_multiband() -> MultiBandLightCurve:
    """A deterministic 3-band sinusoid (P=3.214 d) for the multiband overlay demo."""
    rng = np.random.default_rng(7)
    period = 3.214
    n, t0, baseline = 280, 2458000.0, 120.0
    specs = {
        "g": (0.90, 15.2, 0.020),
        "r": (0.62, 14.6, 0.015),
        "i": (0.41, 14.3, 0.012),
    }
    bands: dict[str, LightCurve] = {}
    for i, (name, (amp, base, noise)) in enumerate(specs.items()):
        time = np.sort(t0 + rng.uniform(0.0, baseline, n))
        mag = base + amp * np.sin(2.0 * np.pi * time / period + i * 0.35)
        mag = mag + rng.normal(0.0, noise, n)
        err = np.full(n, noise)
        bands[name] = LightCurve.from_arrays(
            time, mag, err, domain=Domain.MAGNITUDE, meta={"band": name}
        )
    return MultiBandLightCurve.from_light_curves(
        bands, meta={"id": "demo_multiband", "period": period}
    )


# -- demo registry -----------------------------------------------------------------


def enumerate_sources(
    spec: Any,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    band_column: str | None = None,
) -> list[SourceItem]:
    """Lazily enumerate a batch input into :class:`SourceItem`s.

    ``spec`` is what :func:`cuperiod.batch.io.resolve_inputs` accepts: a glob string,
    a directory, a list of paths, or a ``(DataFrame, group_column)`` pair. Each item's
    ``loader`` reads its source only on selection, so a big folder isn't read up front.
    """
    pairs = resolve_inputs(
        spec, columns=columns, domain=domain, band_column=band_column
    )
    items: list[SourceItem] = []
    for key, source in pairs:

        def _loader(src: Any = source) -> LoadedCurve:
            if isinstance(src, (LightCurve, MultiBandLightCurve)):
                return src
            return load_path(src, columns=columns, domain=domain)

        items.append(SourceItem(key=str(key), label=str(key), loader=_loader))
    return items


def demo_sources() -> list[SourceItem]:
    """All demo sources: bundled example files (if present) + the synthetic one."""
    items: list[SourceItem] = []
    data_dir = example_data_dir()
    if data_dir is not None:
        kepler = data_dir / "kepler_KIC7532973.csv"
        if kepler.is_file():

            def _load_kepler(path: Path = kepler) -> LoadedCurve:
                return load_path(path)

            items.append(
                SourceItem(
                    key="kepler_KIC7532973",
                    label="Kepler KIC 7532973 — hot-Jupiter transit (flux)",
                    loader=_load_kepler,
                    meta={"hint_method": "BLS"},
                )
            )
        asassn = data_dir / "asassn_examples.parquet"
        if asassn.is_file():
            items.extend(asassn_sources(asassn))
    items.append(
        SourceItem(
            key="demo_multiband",
            label="Synthetic multiband g,r,i — GLS demo (P=3.214 d)",
            loader=make_demo_multiband,
            meta={"hint_method": "GLS"},
        )
    )
    return items


__all__ = [
    "FilePreview",
    "asassn_sources",
    "demo_sources",
    "enumerate_sources",
    "example_data_dir",
    "load_path",
    "make_demo_multiband",
    "preview_file",
]
