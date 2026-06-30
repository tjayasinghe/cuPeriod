"""Typer command-line interface.

Every command is a thin wrapper over the Python API, so the CLI and library share one
code path. Commands:

* ``run`` — one light curve, one or more methods; prints the N best periods.
* ``batch`` — many light curves with CPU or GPU workers, written to Parquet/CSV.
* ``methods`` — list registered methods and their backends.
* ``gpu-info`` — show the CUDA GPU and suggested worker counts.
* ``doctor`` — diagnose available backends, torch devices, and the precision each uses.
* ``grid-info`` — show a method's trial grid for a light curve without computing it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import typer

from cuperiod.api import periodogram
from cuperiod.batch.runner import batch_periodograms
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.device import gpu_info as _gpu_info
from cuperiod.core.device import suggest_gpu_workers
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import MultiResult, Periodogram
from cuperiod.methods.base import get_method, list_methods

app = typer.Typer(
    add_completion=False,
    help="Optimized, GPU-accelerated periodograms for astronomy.",
    no_args_is_help=True,
)


def _column_map(
    time: str | None, value: str | None, error: str | None, band: str | None
) -> ColumnMap | None:
    if time is None and value is None and error is None and band is None:
        return None
    return ColumnMap(time=time, value=value, error=error, band=band)


def _parse_methods(method: str) -> list[str]:
    return [m.strip() for m in method.split(",") if m.strip()]


def _domain(value: str | None) -> Domain | None:
    if value is None:
        return None
    try:
        return Domain(value.lower())
    except ValueError as exc:
        raise typer.BadParameter("domain must be 'magnitude' or 'flux'") from exc


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for standard JSON output.

    ``json.dumps`` defaults to emitting bare ``Infinity``/``NaN`` tokens that most
    non-Python JSON parsers reject. Peaks are finite by construction, but this sanitizes
    the serialization boundary so a written ``--out`` file is always valid JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.command()
def run(
    path: Path = typer.Argument(..., help="Light-curve file (CSV/ECSV/FITS/Parquet)."),
    method: str = typer.Option("GLS", "--method", "-m", help="Methods (comma-sep)."),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    band: str | None = typer.Option(None, help="Band column (multi-band input)."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    n_best: int = typer.Option(10, "--n-best", help="Number of peaks to report."),
    out: Path | None = typer.Option(None, help="Write results as JSON here."),
    save_periodogram: Path | None = typer.Option(
        None, "--save-periodogram", help="Write raw spectra to this .npz."
    ),
) -> None:
    """Compute periodogram(s) for a single light curve and print the best periods."""
    columns = _column_map(time, value, error, band)
    lc = LightCurve.from_file(path, columns=columns, domain=_domain(domain))
    result = periodogram(lc, _parse_methods(method), backend=backend)
    results = (
        result.results if isinstance(result, MultiResult) else {result.method: result}
    )
    for name, pg in results.items():
        _print_peaks(name, pg, n_best)
    if out is not None:
        payload = (
            result.to_dict(n_best=n_best)
            if isinstance(result, (MultiResult, Periodogram))
            else {}
        )
        out.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        typer.echo(f"\nWrote {out}")
    if save_periodogram is not None:
        arrays: dict[str, np.ndarray] = {}
        for name, pg in results.items():
            arrays[f"{name}_frequency"] = pg.frequency
            arrays[f"{name}_power"] = pg.power
        # numpy's stub collides **kwargs with the bool allow_pickle keyword.
        np.savez_compressed(save_periodogram, **arrays)  # type: ignore[arg-type]
        typer.echo(f"Wrote {save_periodogram}")


def _print_peaks(name: str, pg: Periodogram, n_best: int) -> None:
    typer.echo(f"\n{name}  (backend={pg.backend}, n={pg.n_samples})")
    typer.echo(f"  {'rank':>4}  {'period':>14}  {'power':>12}")
    for peak in pg.best_periods(n_best, alias_diverse=(name == "BLS")):
        typer.echo(
            f"  {peak.rank:>4}  {peak.period:>14.8f}  {peak.power:>12.6g}"
        )


@app.command()
def batch(
    inputs: str = typer.Argument(..., help="Glob, directory, or file of light curves."),
    method: str = typer.Option("GLS", "--method", "-m", help="Methods (comma-sep)."),
    device: str = typer.Option("cpu", help="cpu | gpu."),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    workers: int | None = typer.Option(None, help="Worker count (None = auto)."),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    out: Path = typer.Option(..., "--out", help="Output .parquet/.csv file or dir."),
    n_best: int = typer.Option(10, "--n-best", help="Peaks stored per light curve."),
    store_raw: bool = typer.Option(False, "--store-raw", help="Store raw spectra."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip done chunks."),
) -> None:
    """Compute periodograms for many light curves with CPU or GPU workers."""
    columns = _column_map(time, value, error, None)
    summary = batch_periodograms(
        inputs,
        _parse_methods(method),
        backend=backend,
        device=device,
        workers=workers,
        columns=columns,
        domain=_domain(domain),
        sink=out,
        n_best=n_best,
        store_raw=store_raw,
        resume=resume,
    )
    typer.echo(
        f"Processed {summary.n_done} results from {summary.n_inputs} inputs "
        f"({summary.n_skipped} skipped, {summary.n_failed} failed) -> {summary.sink}"
    )
    for key, msg in summary.errors[:10]:
        typer.echo(f"  ! {key}: {msg}")


@app.command()
def methods() -> None:
    """List registered periodogram methods and their backends."""
    typer.echo(f"  {'method':>6}  {'objective':>9}  {'multiband':>9}  backends")
    for info in list_methods():
        mb = "yes" if info.supports_multiband else "no"
        typer.echo(
            f"  {info.name:>6}  {info.objective_sense:>9}  {mb:>9}  "
            f"{', '.join(info.all_backends)}"
        )


@app.command(name="gpu-info")
def gpu_info_cmd() -> None:
    """Show the CUDA device and suggested worker counts per method."""
    info = _gpu_info()
    if info is None:
        typer.echo("No CUDA GPU available (install the [gpu] extra and check drivers).")
        raise typer.Exit(code=0)
    typer.echo(str(info))
    typer.echo("  suggested workers:")
    for m in list_methods():
        if any(b in {"cupy", "cufinufft"} for b in m.all_backends):
            typer.echo(f"    {m.name}: {suggest_gpu_workers(m.name)}")


@app.command()
def doctor() -> None:
    """Diagnose available backends, devices, and the precision each will use.

    A one-stop "will the accelerated paths run here, and on what?" check: the installed
    backends, the NVIDIA CUDA fast paths, the portable torch backend and its devices
    (CUDA/ROCm/MPS/XPU/CPU), and what ``backend="auto"`` resolves to per method.
    """
    import os
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    # This probe only enumerates devices (no numerics), so on Windows it allows the
    # torch + numpy/MKL OpenMP duplicate so importing torch can't abort it. The library
    # never sets this for compute paths (see the install docs) — a torch workload on a
    # conflicting Windows env should set KMP_DUPLICATE_LIB_OK itself.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from cuperiod.core._arrayapi import resolve_precision
    from cuperiod.core.backend import (
        available_backends,
        best_torch_device,
        torch_available,
        torch_devices,
    )

    try:
        ver = _pkg_version("cuperiod")
    except PackageNotFoundError:  # pragma: no cover - editable/source runs
        ver = "?"
    typer.echo(
        f"cuPeriod {ver}  |  Python {sys.version.split()[0]}  |  "
        f"{platform.system()} {platform.machine()}"
    )

    avail = available_backends()
    typer.echo("\nbackends installed:")
    for name in ("numpy", "finufft", "numba", "astropy", "cupy", "cufinufft", "torch"):
        typer.echo(f"  {'OK' if name in avail else '--':>2}  {name}")

    typer.echo("\nNVIDIA CUDA fast paths (cufinufft, cupy kernels):")
    info = _gpu_info()
    typer.echo(
        f"  {info}" if info is not None
        else "  no CUDA device (needs the [gpu] extra and an NVIDIA GPU)"
    )

    typer.echo("\nportable torch backend (AMD/Intel/Mac/CPU):")
    if not torch_available():
        typer.echo("  torch not installed — `pip install 'cuperiod[torch]'`")
    else:
        devices = torch_devices()
        best = best_torch_device()
        for d in ("cuda", "xpu", "mps", "cpu"):
            if d in devices:
                tag = "   <- best (used by backend='auto')" if d == best else ""
                prec = resolve_precision("auto", d)
                typer.echo(f"  OK  torch:{d:4s} precision auto -> {prec}{tag}")
            else:
                typer.echo(f"  --  torch:{d}")

    typer.echo("\nbackend='auto' resolves to:")
    for m in list_methods():
        try:
            resolved = get_method(m.name).resolve_backend("auto")
        except Exception as exc:  # pragma: no cover - defensive
            resolved = f"error: {exc}"
        typer.echo(f"  {m.name:14s} {resolved}")


@app.command(name="grid-info")
def grid_info_cmd(
    path: Path = typer.Argument(..., help="Light-curve file."),
    method: str = typer.Option("GLS", "--method", "-m", help="Single method."),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
) -> None:
    """Show a method's trial grid for a light curve without computing the spectrum."""
    columns = _column_map(time, value, error, None)
    lc = LightCurve.from_file(path, columns=columns)
    m = get_method(method)
    single = lc.in_domain(m.natural_domain) if m.natural_domain else lc
    grid = m.default_grid(single, m.coerce_settings(None))
    period = grid.period
    typer.echo(f"{m.name} grid for {path.name}:")
    typer.echo(f"  samples:     {grid.size}")
    typer.echo(f"  period (d):  {period.min():.6g} .. {period.max():.6g}")
    typer.echo(
        f"  freq (1/d):  {grid.frequency.min():.6g} .. {grid.frequency.max():.6g}"
    )


def main() -> None:
    """Console-script entry point."""
    app()


__all__ = ["app", "main"]
