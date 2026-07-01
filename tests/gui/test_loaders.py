"""Tests for source enumeration and the demo/multiband loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.loaders import (
    asassn_sources,
    enumerate_sources,
    example_data_dir,
    load_path,
    make_demo_multiband,
    preview_file,
)


def _write_single(path: Path, period: float, n: int = 200) -> None:
    time = np.linspace(0.0, 50.0, n)
    frame = pd.DataFrame(
        {
            "time": time,
            "mag": np.sin(2 * np.pi * time / period),
            "mag_err": np.full(n, 0.01),
        }
    )
    frame.to_csv(path, index=False)


def test_make_demo_multiband_bands() -> None:
    mblc = make_demo_multiband()
    assert isinstance(mblc, MultiBandLightCurve)
    assert set(mblc.band_names) == {"g", "r", "i"}


def test_enumerate_sources_directory(tmp_path: Path) -> None:
    for k, period in enumerate([1.5, 2.0, 3.0]):
        _write_single(tmp_path / f"lc_{k}.csv", period)
    items = enumerate_sources(str(tmp_path))
    assert len(items) == 3
    lc = items[0].loader()
    assert isinstance(lc, LightCurve)
    assert lc.n == 200


def test_enumerate_sources_dataframe_multiband() -> None:
    time = np.linspace(0.0, 50.0, 300)
    frame = pd.DataFrame(
        {
            "id": np.repeat(["A", "B"], 150),
            "band": np.tile(np.repeat(["g", "r"], 75), 2),
            "time": time,
            "mag": np.sin(time),
            "mag_err": np.full(300, 0.01),
        }
    )
    items = enumerate_sources((frame, "id"), band_column="band")
    assert len(items) == 2
    assert isinstance(items[0].loader(), MultiBandLightCurve)


def test_load_path_single_vs_multiband(tmp_path: Path) -> None:
    _write_single(tmp_path / "single.csv", 2.0)
    assert isinstance(load_path(tmp_path / "single.csv"), LightCurve)

    time = np.linspace(0.0, 50.0, 200)
    multi = pd.DataFrame(
        {
            "band": np.repeat(["g", "r"], 100),
            "time": time,
            "mag": np.sin(time),
            "mag_err": np.full(200, 0.01),
        }
    )
    multi.to_csv(tmp_path / "multi.csv", index=False)
    assert isinstance(load_path(tmp_path / "multi.csv"), MultiBandLightCurve)


def test_preview_file_detects_mapping(tmp_path: Path) -> None:
    _write_single(tmp_path / "p.csv", 2.0, n=50)
    preview = preview_file(tmp_path / "p.csv")
    assert preview is not None
    assert preview.mapping["time"] == "time"
    assert preview.mapping["value"] == "mag"
    assert preview.mapping["error"] == "mag_err"
    assert preview.n_rows == 50
    assert 0 < len(preview.rows) <= 12


def test_asassn_adapter_when_available() -> None:
    data_dir = example_data_dir()
    parquet = None if data_dir is None else data_dir / "asassn_examples.parquet"
    if parquet is None or not parquet.is_file():
        pytest.skip("ASAS-SN demo data not present")
    items = asassn_sources(parquet)
    assert len(items) == 6
    lc = items[0].loader()
    assert isinstance(lc, LightCurve)
    assert lc.n > 0
