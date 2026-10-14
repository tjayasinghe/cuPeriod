"""Typer command-line interface.

Every command is a thin wrapper over the Python API, so the CLI and library share one
code path. Commands:

* ``run`` — one light curve, one or more methods; prints the N best periods.
* ``batch`` — many light curves with CPU or GPU workers, written to Parquet/CSV.
* ``prewhiten`` — automated iterative frequency extraction for a pulsator; it can also
  save the run as a project file and export a script that reproduces it.
* ``batch-prewhiten`` — the same over many light curves, written to Parquet/CSV.
* ``pdot`` — joint (period, dP/dt) search, then a phase-tracking dP/dt measurement.
* ``pdot-batch`` — the whole period-change chain over many light curves, written to
  Parquet/CSV as up to three tables (summary / O-C / P(t) profile).
* ``methods`` — list registered methods and their backends.
* ``gpu-info`` — show the CUDA GPU and suggested worker counts.
* ``doctor`` — diagnose available backends, torch devices, and the precision each uses.
* ``grid-info`` — show a method's trial grid for a light curve without computing it.
* ``session verify | script | replay`` — check a saved pre-whitening project against
  its own action log, export it as a script, or re-create it from the log.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import typer
from pydantic import ValidationError

from cuperiod.api import periodogram
from cuperiod.batch.io import alias_diverse_for, write_rows
from cuperiod.batch.runner import BatchSummary, batch_periodograms
from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.config import PdotSettings, PreWhitenSettings
from cuperiod.core.device import gpu_info as _gpu_info
from cuperiod.core.device import suggest_gpu_workers
from cuperiod.core.errors import (
    BackendUnavailableError,
    ColumnResolutionError,
    CuPeriodError,
    InsufficientDataError,
    UnknownMethodError,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import MultiResult, Peak, Periodogram
from cuperiod.methods.base import get_method, list_methods
from cuperiod.prewhiten.result import PreWhitenResult

if TYPE_CHECKING:  # annotations only — the pdot modules load inside the command.
    from cuperiod import (
        LTTESearchResult,
        PdotSearchResult,
        PeriodLawFit,
        PeriodProfile,
        PhaseTrackResult,
    )
    from cuperiod.pdot.detrend import TrendMethod

#: Seconds per Julian year, for reporting a dP/dt error in the eclipse-timing unit
#: (the canonical definition lives in :mod:`cuperiod.pdot.result`).
_SECONDS_PER_YEAR: float = 86400.0 * 365.25

#: Error lines a batch command prints before summarising the rest.
_MAX_ERROR_LINES = 10

app = typer.Typer(
    add_completion=False,
    help="Optimized, GPU-accelerated periodograms for astronomy.",
    no_args_is_help=True,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _clean_errors(func: _F) -> _F:
    """Report the library's user-facing errors as CLI errors, not tracebacks.

    A bad method or backend name, a missing input file or column is a usage error
    (exit status 2), like a bad option value; any other cuPeriod error (too few
    points, say) prints ``Error: ...`` and exits with status 1. Anything else is a bug
    and keeps its traceback.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (
            UnknownMethodError,
            BackendUnavailableError,
            ColumnResolutionError,
            FileNotFoundError,
        ) as exc:
            raise typer.BadParameter(str(exc)) from exc
        except CuPeriodError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    return cast(_F, wrapper)


def _column_map(
    time: str | None, value: str | None, error: str | None, band: str | None
) -> ColumnMap | None:
    if time is None and value is None and error is None and band is None:
        return None
    return ColumnMap(time=time, value=value, error=error, band=band)


def _parse_methods(method: str) -> list[str]:
    names = [m.strip() for m in method.split(",") if m.strip()]
    if not names:
        raise typer.BadParameter(f"no method named in --method {method!r}")
    return names


def _domain(value: str | None) -> Domain | None:
    try:
        return coerce_domain(value)
    except ValueError as exc:
        raise typer.BadParameter("domain must be 'magnitude' or 'flux'") from exc


def _device(value: str) -> str:
    """Validate ``--device`` up front, so a typo is a usage error."""
    key = value.strip().lower()
    if key not in {"cpu", "gpu"}:
        raise typer.BadParameter(f"device must be 'cpu' or 'gpu', got {value!r}")
    return key


def _sink_path(value: Path) -> Path:
    """Validate a batch ``--out``/``--sink``: a .parquet/.csv file, or a directory."""
    suffix = value.suffix.lower()
    if suffix in {".parquet", ".pq", ".csv"}:
        if value.is_dir():
            raise typer.BadParameter(
                f"{value} is an existing directory; a {suffix} sink must be a file "
                "(pass a path without an extension for a directory sink)"
            )
        return value
    if suffix and not value.is_dir():
        raise typer.BadParameter(
            f"unsupported sink {str(value)!r}: use a .parquet or .csv file, or a "
            "directory (no extension) for a resumable multi-part sink"
        )
    return value


def _positive_period(value: float | None) -> float | None:
    """Validate ``--period``: a seed period must be positive and finite."""
    if value is not None and not (math.isfinite(value) and value > 0.0):
        raise typer.BadParameter(f"period must be positive and finite, got {value}")
    return value


def _npz_path(path: Path) -> Path:
    """Where ``numpy.savez_compressed`` really writes: it appends ``.npz`` if absent."""
    return path if path.suffix.lower() == ".npz" else path.with_name(path.name + ".npz")


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    target = _npz_path(path)
    # numpy's stub collides **kwargs with the bool allow_pickle keyword.
    np.savez_compressed(target, **arrays)  # type: ignore[arg-type]
    typer.echo(f"Wrote {target}")


def _errors_file(sink: Path) -> Path:
    """Where a batch command lists every failure: beside a file sink, in a directory."""
    if sink.suffix.lower() in {".parquet", ".pq", ".csv"}:
        return sink.with_name(f"{sink.stem}.errors.csv")
    return sink / "_errors.csv"


def _report_batch(
    summary: BatchSummary, *, errors_file: Path | None, where: str | None = None
) -> None:
    """Print a batch's failures; exit 1 when every light curve that ran failed.

    The first few errors are printed, then how many more there are and where the
    full list is: ``errors_file`` (written here, or removed when this run had no
    failures so a stale list cannot mislead), or ``where`` for a driver that stores
    them itself.
    """
    errors = summary.errors
    for key, msg in errors[:_MAX_ERROR_LINES]:
        typer.echo(f"  ! {key}: {msg}")
    # With no inputs the driver left the sink untouched; its errors list stays too.
    if errors_file is not None and summary.n_inputs > 0:
        if errors:
            write_rows([{"key": k, "error": m} for k, m in errors], errors_file)
        else:
            errors_file.unlink(missing_ok=True)
    hidden = len(errors) - _MAX_ERROR_LINES
    if errors_file is not None and errors:
        more = f"  ... and {hidden} more; " if hidden > 0 else "  "
        typer.echo(f"{more}all {len(errors)} errors are listed in {errors_file}")
    elif hidden > 0:
        typer.echo(f"  ... and {hidden} more" + (f" ({where})" if where else ""))
    if summary.n_inputs == 0:
        typer.echo("warning: no inputs to process", err=True)
    if summary.n_done == 0 and summary.n_failed > 0:
        typer.echo(f"Error: all {summary.n_failed} light curves failed", err=True)
        raise typer.Exit(code=1)


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
@_clean_errors
def run(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Light-curve file (CSV/ECSV/FITS/Parquet).",
    ),
    method: str = typer.Option("GLS", "--method", "-m", help="Methods (comma-sep)."),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    band: str | None = typer.Option(None, help="Band column (multi-band input)."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    n_best: int = typer.Option(
        10, "--n-best", min=1, help="Number of peaks to report."
    ),
    out: Path | None = typer.Option(None, help="Write results as JSON here."),
    save_periodogram: Path | None = typer.Option(
        None, "--save-periodogram", help="Write raw spectra to this .npz."
    ),
) -> None:
    """Compute periodogram(s) for a single light curve and print the best periods."""
    names = _parse_methods(method)
    columns = _column_map(time, value, error, band)
    lc: LightCurve | MultiBandLightCurve
    if band is not None:
        lc = MultiBandLightCurve.from_file(
            path, band_column=band, columns=columns, domain=_domain(domain)
        )
    else:
        lc = LightCurve.from_file(path, columns=columns, domain=_domain(domain))
    result = periodogram(lc, names, backend=backend)
    results = (
        result.results if isinstance(result, MultiResult) else {result.method: result}
    )
    for name, pg in results.items():
        _print_peaks(name, pg, n_best)
    if out is not None:
        # The same peaks as printed (and as `cuperiod batch` stores): BLS/TLS
        # alias-diverse, not the harmonics a plain top-N list would hold.
        payload = {
            name: _result_payload(name, pg, n_best) for name, pg in results.items()
        }
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
        _save_npz(save_periodogram, arrays)


def _ranked_peaks(name: str, pg: Periodogram, n_best: int) -> list[Peak]:
    """The peaks the CLI reports for a method: one selection policy everywhere."""
    return pg.best_periods(n_best, alias_diverse=alias_diverse_for(name))


def _result_payload(name: str, pg: Periodogram, n_best: int) -> dict[str, Any]:
    """``pg.to_dict`` with the reported (policy-selected) peaks."""
    payload = pg.to_dict(n_best=0)
    payload["peaks"] = [peak.to_dict() for peak in _ranked_peaks(name, pg, n_best)]
    return payload


def _print_peaks(name: str, pg: Periodogram, n_best: int) -> None:
    typer.echo(f"\n{name}  (backend={pg.backend}, n={pg.n_samples})")
    typer.echo(f"  {'rank':>4}  {'period':>14}  {'power':>12}")
    for peak in _ranked_peaks(name, pg, n_best):
        typer.echo(
            f"  {peak.rank:>4}  {peak.period:>14.8f}  {peak.power:>12.6g}"
        )


@app.command()
@_clean_errors
def batch(
    inputs: str = typer.Argument(..., help="Glob, directory, or file of light curves."),
    method: str = typer.Option("GLS", "--method", "-m", help="Methods (comma-sep)."),
    device: str = typer.Option("cpu", help="cpu | gpu."),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    workers: int | None = typer.Option(None, help="Worker count (None = auto)."),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Light curves per chunk/task (None = about four tasks per worker).",
    ),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    out: Path = typer.Option(..., "--out", help="Output .parquet/.csv file or dir."),
    n_best: int = typer.Option(
        10, "--n-best", min=0, help="Peaks stored per light curve."
    ),
    store_raw: bool = typer.Option(False, "--store-raw", help="Store raw spectra."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip light curves already in --out."
    ),
) -> None:
    """Compute periodograms for many light curves with CPU or GPU workers.

    Exits with status 1 when every light curve that ran failed; all failures are
    listed in a CSV beside the sink (inside it, for a directory sink).
    """
    columns = _column_map(time, value, error, None)
    sink = _sink_path(out)
    try:
        summary = batch_periodograms(
            inputs,
            _parse_methods(method),
            backend=backend,
            device=_device(device),
            workers=workers,
            columns=columns,
            domain=_domain(domain),
            sink=sink,
            n_best=n_best,
            store_raw=store_raw,
            resume=resume,
            chunk_size=chunk_size,
        )
    except ValueError as exc:  # up-front validation; per-curve errors are collected
        raise typer.BadParameter(str(exc)) from exc
    n_workers = summary.n_workers
    workers_note = (
        "" if n_workers is None
        else f", {n_workers} worker{'' if n_workers == 1 else 's'}"
    )
    typer.echo(
        f"Processed {summary.n_done} results from {summary.n_inputs} inputs "
        f"({summary.n_skipped} skipped, {summary.n_failed} failed{workers_note}) "
        f"-> {summary.sink}"
    )
    _report_batch(summary, errors_file=_errors_file(sink))


def _prewhiten_settings(
    *,
    max_frequencies: int,
    snr: float,
    stop: str,
    minimum_frequency: float | None,
    maximum_frequency: float | None,
    uncertainty: str,
    combinations: bool,
    backend: str | None,
    store_spectra: bool = True,
) -> PreWhitenSettings:
    """Build a :class:`PreWhitenSettings` from CLI options (validated by pydantic)."""
    criteria = tuple(c.strip().lower() for c in stop.split(",") if c.strip())
    # Pass --backend only when given: an explicit backend="auto" would pin the field
    # and hide CUPERIOD_PREWHITEN_BACKEND, which the library API honours.
    pinned: dict[str, Any] = {} if backend is None else {"backend": backend}
    try:
        return PreWhitenSettings(
            max_frequencies=max_frequencies,
            snr_threshold=snr,
            stop_criteria=criteria,  # type: ignore[arg-type]
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            uncertainty=uncertainty,  # type: ignore[arg-type]
            combinations=combinations,
            store_spectra=store_spectra,
            **pinned,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
@_clean_errors
def prewhiten(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Light-curve file (CSV/ECSV/FITS/Parquet).",
    ),
    backend: str | None = typer.Option(
        None,
        help="auto | cpu | gpu | concrete backend (default: the settings' backend,"
        " CUPERIOD_PREWHITEN_BACKEND, else auto).",
    ),
    max_frequencies: int = typer.Option(
        30, "--max-frequencies", "-n", help="Cap on extracted components."
    ),
    snr: float = typer.Option(4.0, "--snr", help="Breger signal-to-noise threshold."),
    stop: str = typer.Option(
        "snr", "--stop", help="Stopping criteria (comma-sep): snr,fap,bic,amplitude."
    ),
    minimum_frequency: float | None = typer.Option(
        None, "--fmin", help="Lowest trial frequency (cycles/day)."
    ),
    maximum_frequency: float | None = typer.Option(
        None, "--fmax", help="Highest trial frequency (cycles/day)."
    ),
    uncertainty: str = typer.Option(
        "covariance", help="covariance | analytic | bootstrap."
    ),
    combinations: bool = typer.Option(
        True,
        "--combinations/--no-combinations",
        help="Identify combination frequencies.",
    ),
    spacing: bool = typer.Option(
        False, "--spacing", help="Also search for a g-mode period-spacing pattern."
    ),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    out: Path | None = typer.Option(None, help="Write the full solution as JSON here."),
    csv_out: Path | None = typer.Option(
        None, "--csv", help="Write the component table as CSV here."
    ),
    save_spectrum: Path | None = typer.Option(
        None, "--save-spectrum", help="Write the amplitude spectra to this .npz."
    ),
    save_project: Path | None = typer.Option(
        None,
        "--save-project",
        help="Also save the run as a pre-whitening project (*.cupw.json) to open, "
        "edit or verify later.",
    ),
    export_script: Path | None = typer.Option(
        None,
        "--export-script",
        help="Also write a Python script that reproduces the run with the library.",
    ),
) -> None:
    """Extract a pulsator's frequency solution by automated iterative pre-whitening."""
    from cuperiod.prewhiten import find_period_spacing
    from cuperiod.prewhiten import prewhiten as _prewhiten

    columns = _column_map(time, value, error, None)
    lc = LightCurve.from_file(path, columns=columns, domain=_domain(domain))
    settings = _prewhiten_settings(
        max_frequencies=max_frequencies,
        snr=snr,
        stop=stop,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        uncertainty=uncertainty,
        combinations=combinations,
        backend=backend,
    )
    session = None
    if save_project is None and export_script is None:
        result = _prewhiten(lc, settings=settings)
    else:
        # The same run as a recorded session (identical numbers), so it can be saved.
        from cuperiod.prewhiten.session import PreWhitenSession

        session = PreWhitenSession(lc, settings=settings, columns=columns)
        session.run()
        result = session.finalize()
    typer.echo(result.summary())

    if spacing:
        independent = result.independent()
        series = (
            find_period_spacing(
                np.asarray([c.period for c in independent], dtype=np.float64),
                np.asarray([c.amplitude for c in independent], dtype=np.float64),
            )
            if len(independent) >= 4
            else None
        )
        typer.echo("")
        typer.echo(
            series.summary()
            if series is not None
            else "Period spacing: no regular series found."
        )

    if out is not None:
        out.write_text(
            json.dumps(_json_safe(result.to_dict()), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        typer.echo(f"\nWrote {out}")
    if csv_out is not None:
        _write_component_csv(result, csv_out)
        typer.echo(f"Wrote {csv_out}")
    if save_spectrum is not None:
        arrays: dict[str, np.ndarray] = {}
        if result.spectrum is not None:
            arrays["frequency"] = result.spectrum.frequency
            arrays["amplitude"] = result.spectrum.amplitude
        if result.residual_spectrum is not None:
            arrays["residual_amplitude"] = result.residual_spectrum.amplitude
        if result.window is not None:
            arrays["window_amplitude"] = result.window.amplitude
        _save_npz(save_spectrum, arrays)
    if session is not None and save_project is not None:
        session.save(save_project, title=path.stem)
        typer.echo(f"Wrote {save_project}")
    if session is not None and export_script is not None:
        written = session.export_script(export_script, title=path.stem)
        typer.echo(f"Wrote {written}")


def _write_component_csv(result: PreWhitenResult, path: Path) -> None:
    """Write one row per extracted component (the frequency-solution table)."""
    import csv

    rows = result.to_table()
    fields = list(rows[0]) if rows else ["rank", "label", "frequency", "amplitude"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@app.command(name="batch-prewhiten")
@_clean_errors
def batch_prewhiten_cmd(
    inputs: str = typer.Argument(..., help="Glob, directory, or file of light curves."),
    out: Path = typer.Option(..., "--out", help="Output .parquet/.csv file or dir."),
    device: str = typer.Option("cpu", help="cpu | gpu."),
    backend: str | None = typer.Option(
        None,
        help="auto | cpu | gpu | concrete backend (default: the settings' backend,"
        " CUPERIOD_PREWHITEN_BACKEND, else auto).",
    ),
    workers: int | None = typer.Option(None, help="Worker count (None = auto)."),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Light curves per chunk/task (None = the batch_prewhiten default).",
    ),
    max_frequencies: int = typer.Option(
        30, "--max-frequencies", "-n", help="Cap on extracted components."
    ),
    snr: float = typer.Option(4.0, "--snr", help="Breger signal-to-noise threshold."),
    stop: str = typer.Option("snr", "--stop", help="Stopping criteria (comma-sep)."),
    minimum_frequency: float | None = typer.Option(
        None, "--fmin", help="Lowest trial frequency (cycles/day)."
    ),
    maximum_frequency: float | None = typer.Option(
        None, "--fmax", help="Highest trial frequency (cycles/day)."
    ),
    uncertainty: str = typer.Option(
        "covariance", help="covariance | analytic | bootstrap."
    ),
    combinations: bool = typer.Option(
        True, "--combinations/--no-combinations", help="Identify combinations."
    ),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip light curves already in the sink and retry failures "
        "(resume is by key).",
    ),
) -> None:
    """Pre-whiten many light curves; writes one row per extracted component.

    Exits with status 1 when every light curve that ran failed; all failures are
    listed in a CSV beside the sink (inside it, for a directory sink).
    """
    from cuperiod.prewhiten import batch_prewhiten as _batch_prewhiten

    settings = _prewhiten_settings(
        max_frequencies=max_frequencies,
        snr=snr,
        stop=stop,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        uncertainty=uncertainty,
        combinations=combinations,
        backend=backend,
        store_spectra=False,
    )
    sink = _sink_path(out)
    extra: dict[str, Any] = {} if chunk_size is None else {"chunk_size": chunk_size}
    try:
        summary = _batch_prewhiten(
            inputs,
            settings=settings,
            backend=backend,
            device=_device(device),
            workers=workers,
            columns=_column_map(time, value, error, None),
            domain=_domain(domain),
            sink=sink,
            resume=resume,
            **extra,
        )
    except ValueError as exc:  # up-front validation; per-curve errors are collected
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Wrote {summary.n_done} component rows from {summary.n_inputs} inputs "
        f"({summary.n_skipped} skipped, {summary.n_failed} failed) -> {summary.sink}"
    )
    _report_batch(summary, errors_file=_errors_file(sink))


def _pdot_range(value: str | None) -> tuple[float, float] | None:
    """Parse ``--pdot-range "lo,hi"`` into explicit (dP/dt)/P bounds (1/day)."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise typer.BadParameter(
            f"pdot-range must be two comma-separated numbers 'lo,hi', got {value!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise typer.BadParameter(
            f"pdot-range: {value!r} is not a pair of numbers"
        ) from exc


def _pdot_settings(
    *,
    max_cycle_drift: float,
    samples_per_drift: int,
    samples_per_peak: int,
    window_rayleigh: float,
    pdot_range: str | None,
    harmonics: int | None,
    segments: int | None,
    harmonic_scale: float | None,
) -> PdotSettings:
    """Build a :class:`PdotSettings` from CLI options (validated by pydantic)."""
    try:
        return PdotSettings(
            max_cycle_drift=max_cycle_drift,
            samples_per_drift=samples_per_drift,
            samples_per_peak=samples_per_peak,
            window_rayleigh=window_rayleigh,
            pdot_over_p_range=_pdot_range(pdot_range),
            harmonics=harmonics,
            n_segments=segments,
            harmonic_scale=harmonic_scale,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _detrend_width(value: float | None) -> float | None:
    """Validate ``--detrend-width`` up front, so a bad one is a usage error."""
    if value is None:
        return None
    if not (math.isfinite(value) and value > 0.0):
        raise typer.BadParameter(
            f"detrend-width must be positive and finite, got {value}"
        )
    return float(value)


def _trend_method(value: str) -> TrendMethod:
    """Validate ``--detrend-method`` against the two registered filter kernels."""
    key = value.strip().lower()
    if key == "median":
        return "median"
    if key == "binned":
        return "binned"
    raise typer.BadParameter(
        f"detrend-method must be 'median' or 'binned', got {value!r}"
    )


def _period_model(value: str | None) -> str | None:
    """Resolve ``--model`` to a registered period-evolution model name."""
    if value is None:
        return None
    from cuperiod.pdot.models import get_model

    try:
        return get_model(value).name
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _period_laws(value: str) -> tuple[str, ...] | None:
    """Parse ``--laws``: ``auto``, ``none``, or a comma-separated selection.

    Unknown names are rejected here rather than in the workers, where
    :func:`cuperiod.compare_period_laws` would simply skip them and leave a
    whole catalogue silently without a fitted law.
    """
    from cuperiod.pdot.laws import AUTO_LAWS, available_laws

    key = value.strip().lower()
    if key in {"", "auto"}:
        return tuple(AUTO_LAWS)
    if key in {"none", "off"}:
        return None
    known = available_laws()
    names = tuple(n.strip().lower() for n in value.split(",") if n.strip())
    unknown = [n for n in names if n not in known]
    if unknown:
        raise typer.BadParameter(
            f"unknown period law(s) {', '.join(unknown)}; choose from "
            f"{', '.join(known)}, or 'auto' / 'none'"
        )
    return names


def _comma_list(value: str | None) -> tuple[str, ...] | None:
    """Split a comma-separated option into keys; ``None`` stays ``None``."""
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _require_matplotlib(option: str) -> None:
    """Fail before a long run when a figure option cannot possibly work.

    :func:`cuperiod.batch_pdot` swallows per-star plotting failures by design —
    one unplottable star must not fail a survey — so a missing matplotlib would
    otherwise surface only as thousands of empty ``figure`` cells at the end.
    """
    try:
        import matplotlib  # noqa: F401  (importability probe only)
    except ImportError:
        typer.echo(
            f"{option} needs matplotlib, which is not installed.\n"
            '  install it with:  pip install "cuperiod[plot]"',
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
@_clean_errors
def pdot(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Light-curve file (CSV/ECSV/FITS/Parquet).",
    ),
    period: float | None = typer.Option(
        None, "--period", help="Window center in days (None = seed from a first run)."
    ),
    method: str = typer.Option("GLS", "--method", "-m", help="Wrapped method (one)."),
    model: str | None = typer.Option(
        None, "--model", help="Period law: constant-pdot (default) | exponential."
    ),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    max_cycle_drift: float = typer.Option(
        10.0, "--max-cycle-drift", help="dP/dt half-range, in accumulated cycles."
    ),
    samples_per_drift: int = typer.Option(
        5, "--samples-per-drift", help="Oversampling of the (dP/dt)/P axis."
    ),
    samples_per_peak: int = typer.Option(
        10, "--samples-per-peak", help="Frequency oversampling inside the window."
    ),
    window_rayleigh: float = typer.Option(
        30.0, "--window-rayleigh", help="Frequency half-window in Rayleigh widths."
    ),
    pdot_range: str | None = typer.Option(
        None, "--pdot-range", help="Explicit (dP/dt)/P bounds 'lo,hi' in 1/day."
    ),
    harmonics: int | None = typer.Option(
        None, "--harmonics", help="Tracking template order (None = BIC-selected)."
    ),
    segments: int | None = typer.Option(
        None,
        "--segments",
        min=4,
        help="Target tracking segment count (at least 4; None = auto).",
    ),
    harmonic_scale: float | None = typer.Option(
        None, "--harmonic-scale", help="Extra (dP/dt)/P refinement (None = per method)."
    ),
    detrend_width: float | None = typer.Option(
        None, "--detrend-width", help="Running-median window (days) removed first."
    ),
    track: bool = typer.Option(
        True, "--track/--no-track", help="Measure dP/dt by segmented phase tracking."
    ),
    ltte: bool = typer.Option(
        True,
        "--ltte/--no-ltte",
        help="Search the tracked O-C for a cyclic (light-travel-time) variation.",
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Also measure the model-free P(t) track and rank period laws on it.",
    ),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    out: Path | None = typer.Option(None, help="Write the results as JSON here."),
    save_map: Path | None = typer.Option(
        None, "--save-map", help="Write the 2-D search map to this .npz."
    ),
) -> None:
    """Search a light curve jointly over (period, dP/dt) and measure the drift.

    The search localizes the drifting signal on a (dP/dt)/P x frequency grid; the
    phase-tracking pass then measures dP/dt at that period with an uncertainty, and
    --profile adds the model-free P(t) track that neither of them assumes.
    """
    from cuperiod.pdot.scan import pdot_search
    from cuperiod.pdot.track import phase_track

    columns = _column_map(time, value, error, None)
    lc = LightCurve.from_file(path, columns=columns, domain=_domain(domain))
    width = _detrend_width(detrend_width)
    if width is not None:
        from cuperiod.pdot.detrend import detrend as _detrend

        # replace() rather than a fresh LightCurve: the domain and metadata must
        # survive, and detrending changes only the values.
        lc = replace(lc, value=_detrend(lc.time, lc.value, width))
        typer.echo(f"detrended with a {width:g} d running median")
    model_name = _period_model(model)
    settings = _pdot_settings(
        max_cycle_drift=max_cycle_drift,
        samples_per_drift=samples_per_drift,
        samples_per_peak=samples_per_peak,
        window_rayleigh=window_rayleigh,
        pdot_range=pdot_range,
        harmonics=harmonics,
        segments=segments,
        harmonic_scale=harmonic_scale,
    )
    result = pdot_search(
        lc,
        period=_positive_period(period),
        method=method,
        model=model_name,
        settings=settings,
        backend=backend,
    )
    _print_pdot_search(result)

    tracked: PhaseTrackResult | None = None
    if track:
        try:
            tracked = phase_track(
                lc,
                period=result.best.period,
                pdot=result.best.pdot,
                model=model_name,
                settings=settings,
            )
        except InsufficientDataError as exc:
            typer.echo(f"\nphase tracking skipped: {exc}")
        else:
            _print_phase_track(tracked)

    cyclic: LTTESearchResult | None = None
    if tracked is not None and ltte:
        try:
            cyclic = tracked.ltte_search(settings=settings)
        except (InsufficientDataError, ValueError) as exc:
            typer.echo(f"\nlight-time orbit search skipped: {exc}")
        else:
            _print_ltte(cyclic)

    prof: PeriodProfile | None = None
    laws: list[PeriodLawFit] = []
    if profile:
        from cuperiod.pdot.laws import compare_period_laws
        from cuperiod.pdot.profile import period_profile

        try:
            prof = period_profile(
                lc, period=result.best.period, settings=settings, backend=backend
            )
        except (InsufficientDataError, ValueError) as exc:
            typer.echo(f"\nperiod profile skipped: {exc}")
        else:
            laws = compare_period_laws(prof.t_mid, prof.period, prof.period_err)
            _print_period_profile(prof, laws)

    if out is not None:
        payload = result.to_dict()
        if tracked is not None:
            payload["track"] = tracked.to_dict()
        if cyclic is not None:
            payload["ltte"] = cyclic.to_dict()
        if prof is not None:
            payload["profile"] = prof.to_dict()
            payload["laws"] = [fit.to_dict() for fit in laws]
        out.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        typer.echo(f"\nWrote {out}")
    if save_map is not None:
        arrays: dict[str, np.ndarray] = {
            "frequency": result.frequency,
            "pdot_over_p": result.pdot_over_p,
            "power": result.power,
        }
        _save_npz(save_map, arrays)


def _print_pdot_search(res: PdotSearchResult) -> None:
    """The joint-search block: the best cell and what the extra axis bought."""
    best = res.best
    n_pdot, n_freq = res.shape
    typer.echo(
        f"\n{res.method} dP/dt search  (backend={res.backend}, n={res.n_samples}, "
        f"baseline={res.baseline:.6g} d)"
    )
    typer.echo(f"  {'grid':<15} {n_pdot} x {n_freq}  ((dP/dt)/P x frequency)")
    typer.echo(f"  {'period (d)':<15} {best.period:.8f}   at t_ref={best.t_ref:.6f}")
    typer.echo(f"  {'dP/dt (d/d)':<15} {best.pdot:.6g}")
    typer.echo(f"  {'dP/dt (s/yr)':<15} {best.pdot_s_per_yr:.6g}")
    typer.echo(f"  {'dP/dt (ms/yr)':<15} {best.pdot_ms_per_yr:.6g}")
    typer.echo(
        f"  {'power':<15} {best.power:.6g}   (dP/dt=0: {best.zero_pdot_power:.6g}, "
        f"sense={res.objective_sense})"
    )


def _print_phase_track(res: PhaseTrackResult) -> None:
    """The segmented phase-tracking (O-C) measurement block."""
    typer.echo(f"\nphase tracking at P = {res.period:.8f} d")
    typer.echo(f"  {'dP/dt (d/d)':<15} {res.pdot:.6g} +- {res.pdot_err:.3g}")
    typer.echo(
        f"  {'dP/dt (s/yr)':<15} {res.pdot_s_per_yr:.6g} +- "
        f"{res.pdot_err * _SECONDS_PER_YEAR:.3g}"
    )
    typer.echo(
        f"  {'F-test p':<15} {res.f_test_p:.4g}   "
        "(the drift against the constant-period model)"
    )
    typer.echo(
        f"  {'segments':<15} {res.n_segments}   "
        f"(template harmonics: {res.template.n_harmonics})"
    )
    if res.wrap_risk:
        typer.echo(
            "  ! wrap risk: consecutive segment delays jump by more than 0.3 period; "
            "the cycle count may have slipped (retry with fewer --segments)"
        )


def _print_ltte(res: LTTESearchResult) -> None:
    """The cyclic O-C block: the light-time orbit found in the tracked delays."""
    best = res.best
    typer.echo(f"\nlight-time orbit search over the O-C  ({res.n_used} points)")
    typer.echo(
        f"  {'outer period':<15} {best.outer_period_yr:.3f} +- "
        f"{best.outer_period_err_yr:.3f} yr   ({res.n_cycles:.2f} cycles covered)"
    )
    typer.echo(
        f"  {'amplitude':<15} {best.amplitude_s:.1f} +- {best.amplitude_err_s:.1f} s"
        f"   (a sin i = {best.a_sini_au:.3f} au, f(m) = {best.mass_function:.4f} Msun)"
    )
    typer.echo(
        f"  {'dBIC':<15} {res.delta_bic:+.1f}   (positive favours the cycle over the "
        f"polynomial alone; F-test p = {res.f_test_p:.2g}, searched)"
    )
    typer.echo(
        f"  {'secular dP/dt':<15} {best.pdot_s_per_yr:+.4g} +- "
        f"{best.pdot_err_scaled * _SECONDS_PER_YEAR:.2g} s/yr   "
        f"(parabola alone: {res.null.pdot_s_per_yr:+.4g})"
    )
    if best.at_grid_edge:
        typer.echo(
            "  ! the best outer period sits at the edge of the scan; the cycle "
            "may be longer or shorter than the range searched"
        )
    if res.n_cycles < 1.0:
        typer.echo(
            "  ! less than one outer cycle covered: a light-time orbit and a "
            "parabola are nearly degenerate here"
        )


def _print_period_profile(prof: PeriodProfile, laws: Sequence[PeriodLawFit]) -> None:
    """The model-free P(t) track, then the period laws ranked over it by BIC."""
    first, last = prof.points[0], prof.points[-1]
    typer.echo(
        f"\nperiod profile  ({prof.n_intervals} intervals, "
        f"{prof.n_harmonics} harmonics, window=+-{prof.period_window:.3g})"
    )
    typer.echo(f"  {'P first (d)':<15} {first.period:.8f} +- {first.period_err:.3g}")
    typer.echo(f"  {'P last (d)':<15} {last.period:.8f} +- {last.period_err:.3g}")
    typer.echo(
        f"  {'dP/P total':<15} {prof.total_fractional_change():+.6g}   "
        "(end intervals, no model assumed)"
    )
    if prof.any_at_window_edge:
        typer.echo(
            "  ! an interval's period sits at the edge of its search window; the "
            "track may be clipped (widen the profile window)"
        )

    if not laws:
        typer.echo("\nno period law could be fitted to this track")
        return
    typer.echo(
        f"\nperiod laws over P(t)  (ranked by error-scaled BIC, {len(laws)} fitted)"
    )
    typer.echo(f"  {'law':<12} {'BIC':>10} {'dBIC':>9} {'dP/P':>12}  parameters")

    # The column must be the criterion the list is sorted by. compare_period_laws
    # ranks on bic_scaled (chi-squared rescaled by the excess scatter); the plain
    # bic would make dBIC negative for a lower-ranked law when errors are
    # understated.
    def rank_bic(fit: PeriodLawFit) -> float:
        scaled = float(fit.bic_scaled)
        return scaled if math.isfinite(scaled) else float(fit.bic)

    best_bic = rank_bic(laws[0])
    for fit in laws:
        params = ", ".join(f"{k}={v:.6g}" for k, v in fit.params.items())
        bound = "  (at a bound)" if fit.at_bound else ""
        typer.echo(
            f"  {fit.law:<12} {rank_bic(fit):>10.4g} {rank_bic(fit) - best_bic:>9.3g} "
            f"{fit.fractional_change(first.t_start, last.t_end):>+12.4g}  "
            f"{params}{bound}"
        )


@app.command(name="pdot-batch")
@_clean_errors
def pdot_batch_cmd(
    inputs: str = typer.Argument(..., help="Glob, directory, or file of light curves."),
    sink: Path = typer.Option(
        ...,
        "--sink",
        "--out",
        help="Summary table: .parquet/.csv file or dir (one row per star).",
    ),
    period: float | None = typer.Option(
        None, "--period", help="Seed period in days for every star (None = search)."
    ),
    method: str = typer.Option("GLS", "--method", "-m", help="Wrapped method (one)."),
    model: str | None = typer.Option(
        None, "--model", help="Period law: constant-pdot (default) | exponential."
    ),
    device: str = typer.Option("cpu", help="cpu | gpu."),
    backend: str = typer.Option("auto", help="auto | cpu | gpu | concrete backend."),
    workers: int | None = typer.Option(None, help="Worker count (None = auto)."),
    chunk_size: int = typer.Option(
        32, "--chunk-size", help="Light curves per chunk/task."
    ),
    max_cycle_drift: float = typer.Option(
        10.0, "--max-cycle-drift", help="dP/dt half-range, in accumulated cycles."
    ),
    samples_per_drift: int = typer.Option(
        5, "--samples-per-drift", help="Oversampling of the (dP/dt)/P axis."
    ),
    samples_per_peak: int = typer.Option(
        10, "--samples-per-peak", help="Frequency oversampling inside the window."
    ),
    window_rayleigh: float = typer.Option(
        30.0, "--window-rayleigh", help="Frequency half-window in Rayleigh widths."
    ),
    pdot_range: str | None = typer.Option(
        None, "--pdot-range", help="Explicit (dP/dt)/P bounds 'lo,hi' in 1/day."
    ),
    harmonics: int | None = typer.Option(
        None, "--harmonics", help="Tracking template order (None = BIC-selected)."
    ),
    segments: int | None = typer.Option(
        None,
        "--segments",
        min=4,
        help="Target tracking segment count (at least 4; None = auto).",
    ),
    harmonic_scale: float | None = typer.Option(
        None, "--harmonic-scale", help="Extra (dP/dt)/P refinement (None = per method)."
    ),
    detrend_width: float | None = typer.Option(
        None, "--detrend-width", help="Running-median window (days) removed first."
    ),
    detrend_method: str = typer.Option(
        "median", "--detrend-method", help="Detrend kernel: median | binned."
    ),
    profile: bool = typer.Option(
        True, "--profile/--no-profile", help="Measure the model-free P(t) track."
    ),
    ltte: bool = typer.Option(
        True, "--ltte/--no-ltte", help="Search each O-C for a light-time orbit."
    ),
    laws: str = typer.Option(
        "auto",
        "--laws",
        help="Period laws ranked on each track (comma-sep), 'auto', or 'none'.",
    ),
    points_per_interval: int | None = typer.Option(
        None, "--points-per-interval", help="Target points per profile interval."
    ),
    period_window: float | None = typer.Option(
        None, "--period-window", help="Fractional half-width of an interval's search."
    ),
    oc_sink: Path | None = typer.Option(
        None, "--oc-sink", help="Per-segment O-C table (its own file or directory)."
    ),
    profile_sink: Path | None = typer.Option(
        None, "--profile-sink", help="Per-interval P(t) table (its own file or dir)."
    ),
    plots: Path | None = typer.Option(
        None, "--plots", help="Directory for per-star figures (needs the [plot] extra)."
    ),
    plot_keys: str | None = typer.Option(
        None, "--plot-keys", help="Restrict figures to these keys (comma-sep)."
    ),
    time: str | None = typer.Option(None, help="Time column name override."),
    value: str | None = typer.Option(None, help="Value column name override."),
    error: str | None = typer.Option(None, help="Error column name override."),
    domain: str | None = typer.Option(None, help="magnitude | flux."),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip light curves already in the sink and retry failures "
        "(resume is by key).",
    ),
) -> None:
    """Measure period changes for many light curves; writes up to three tables.

    --sink takes the per-star summary; --oc-sink and --profile-sink add the
    per-segment O-C points and the per-interval P(t) track behind it. A light curve
    that fails still gets a summary row, with the reason in its status column.
    """
    from cuperiod.pdot.batch import batch_pdot as _batch_pdot

    # batch_pdot swallows plotting failures per star, so an unusable --plots must
    # be caught before a survey-scale run, not discovered from empty cells after.
    if plots is not None:
        _require_matplotlib("--plots")
    seed_period = _positive_period(period)
    device = _device(device)
    sink = _sink_path(sink)
    settings = _pdot_settings(
        max_cycle_drift=max_cycle_drift,
        samples_per_drift=samples_per_drift,
        samples_per_peak=samples_per_peak,
        window_rayleigh=window_rayleigh,
        pdot_range=pdot_range,
        harmonics=harmonics,
        segments=segments,
        harmonic_scale=harmonic_scale,
    )
    try:
        summary = _batch_pdot(
            inputs,
            period=seed_period,
            method=method,
            model=_period_model(model),
            settings=settings,
            detrend_width=_detrend_width(detrend_width),
            detrend_method=_trend_method(detrend_method),
            profile=profile,
            laws=_period_laws(laws),
            ltte=ltte,
            points_per_interval=points_per_interval,
            period_window=period_window,
            backend=backend,
            device=device,
            workers=workers,
            columns=_column_map(time, value, error, None),
            domain=_domain(domain),
            sink=sink,
            oc_sink=oc_sink,
            profile_sink=profile_sink,
            plots=plots,
            plot_keys=_comma_list(plot_keys),
            resume=resume,
            chunk_size=chunk_size,
        )
    except ValueError as exc:  # up-front validation; per-curve errors are collected
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Measured {summary.n_done} light curves from {summary.n_inputs} inputs "
        f"({summary.n_skipped} skipped, {summary.n_failed} failed) -> {summary.sink}"
    )
    if oc_sink is not None:
        typer.echo(f"  O-C segments -> {oc_sink}")
    if profile_sink is not None:
        typer.echo(f"  P(t) profile -> {profile_sink}")
    if plots is not None:
        typer.echo(f"  figures      -> {plots}")
    # Every failure already has a summary row whose status column holds the error.
    _report_batch(
        summary, errors_file=None, where=f"the status column of {summary.sink}"
    )


session_app = typer.Typer(
    add_completion=False,
    help="Verify, script or replay a saved pre-whitening project (*.cupw.json).",
    no_args_is_help=True,
)
app.add_typer(session_app, name="session")

_PROJECT_ARGUMENT = typer.Argument(
    ..., exists=True, dir_okay=False, help="A pre-whitening project (*.cupw.json)."
)
_DATA_OPTION = typer.Option(
    None,
    "--data",
    exists=True,
    dir_okay=False,
    help="Light-curve file to use instead of the project's own data (it must hold "
    "the same points: the content hash is checked).",
)
_SESSION_BACKEND_OPTION = typer.Option(
    None, help="Spectrum backend (default: the recorded one when available, else CPU)."
)


def _project_data(data: Path | None) -> LightCurve | None:
    return None if data is None else LightCurve.from_file(data)


@contextlib.contextmanager
def _session_warnings() -> Iterator[None]:
    """Print a session's warnings as ``warning: ...`` lines, not Python tracebacks."""
    from cuperiod.prewhiten.session import SessionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SessionWarning)
        try:
            yield
        finally:
            for item in caught:
                if issubclass(item.category, SessionWarning):
                    typer.echo(f"warning: {item.message}", err=True)
                else:
                    warnings.warn_explicit(
                        item.message, item.category, item.filename, item.lineno
                    )


@session_app.command("verify")
@_clean_errors
def session_verify_cmd(
    project: Path = _PROJECT_ARGUMENT,
    data: Path | None = _DATA_OPTION,
    backend: str | None = _SESSION_BACKEND_OPTION,
) -> None:
    """Replay a project's action log and check it reproduces the stored solution."""
    from cuperiod.prewhiten.session import PreWhitenSession

    with _session_warnings():
        session = PreWhitenSession.load(
            project, data=_project_data(data), verify=True, backend=backend
        )
    typer.echo(
        f"Verified {project}: its {len(session.history)} logged action(s) reproduce "
        f"the stored solution ({len(session.state.active)} active component(s), "
        f"backend {session.backend})."
    )


@session_app.command("script")
@_clean_errors
def session_script_cmd(
    project: Path = _PROJECT_ARGUMENT,
    out: Path = typer.Option(
        ..., "--out", "-o", dir_okay=False, help="The Python script to write."
    ),
    data: Path | None = _DATA_OPTION,
) -> None:
    """Export a project as a Python script that reproduces it with the library."""
    from cuperiod.prewhiten.session import PreWhitenSession

    with _session_warnings():
        session = PreWhitenSession.load(
            project, data=_project_data(data), backend="cpu"
        )
    written = session.export_script(out, title=project.name.split(".")[0])
    typer.echo(f"Wrote {written}")


@session_app.command("replay")
@_clean_errors
def session_replay_cmd(
    project: Path = _PROJECT_ARGUMENT,
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        dir_okay=False,
        help="Save the replayed session (with its full undo history) as a project.",
    ),
    data: Path | None = _DATA_OPTION,
    backend: str | None = _SESSION_BACKEND_OPTION,
) -> None:
    """Re-create a project from its action log and print the resulting solution."""
    from cuperiod.prewhiten.session import PreWhitenSession
    from cuperiod.prewhiten.session.project import divergence, rerun

    with _session_warnings():
        stored = PreWhitenSession.load(
            project, data=_project_data(data), backend=backend
        )
        replayed = rerun(stored, backend=backend)
    typer.echo(replayed.result().summary())
    problem = divergence(stored.state, replayed.state)
    if problem is not None:
        typer.echo(
            f"Error: the replay does not reproduce the stored solution: {problem}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(
        f"\nReplayed {len(replayed.history)} logged action(s); the stored solution "
        "is reproduced."
    )
    if out is not None:
        replayed.save(out, title=project.name.split(".")[0])
        typer.echo(f"Wrote {out}")


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
@_clean_errors
def grid_info_cmd(
    path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Light-curve file."
    ),
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
