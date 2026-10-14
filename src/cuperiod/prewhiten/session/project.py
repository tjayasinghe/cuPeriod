"""Project files (``*.cupw.json``) and deterministic replay of a session's log.

A project is a session's :class:`~cuperiod.prewhiten.session.state.SessionSnapshot`
written as versioned JSON: the creation arguments, the head state and the effective
(seeded) action log, plus the data — embedded, or referenced by a relative path and
pinned by a content hash. The undo stack is not stored: after :func:`load` only the
head is in memory, and :func:`replay` (``load(verify=True)``) re-creates every state
by feeding the logged, resolved arguments to the reducers again.

Opening a file never executes anything from it. It is read as JSON only (no pickle,
no ``eval``), its size is checked before reading, ``NaN``/``Infinity`` tokens are
refused, every object is validated by a strict pydantic model with
``extra="forbid"``, every cross reference by :func:`validate_state`, and a data
reference must stay inside the project's directory. Whatever goes wrong while reading
surfaces as :class:`~cuperiod.ProjectError` and nothing else.
"""

from __future__ import annotations

import contextlib
import csv
import io
import itertools
import json
import math
import os
import platform
import re
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NoReturn, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PreWhitenSettings, SessionSettings
from cuperiod.core.errors import BackendUnavailableError, CuPeriodError
from cuperiod.core.lightcurve import LightCurve
from cuperiod.prewhiten.constraints import (
    MAX_TIE_COEFFICIENT,
    MAX_TIE_ORDER,
    format_tie,
    parse_tie,
)
from cuperiod.prewhiten.session import actions
from cuperiod.prewhiten.session.state import (
    COMMITTED_KINDS,
    Component,
    CreationArgs,
    FitStats,
    LogEntry,
    PointMask,
    ProjectError,
    RefusalHint,
    SessionData,
    SessionError,
    SessionSnapshot,
    SessionState,
    SessionWarning,
    StepReport,
    Tie,
    content_sha256,
    settings_dump,
)
from cuperiod.prewhiten.subsets import MAX_LABEL_LENGTH, Partition, parse_origin

if TYPE_CHECKING:
    from cuperiod.prewhiten.session.session import PreWhitenSession

#: The ``format`` member of every project file.
FORMAT: Final = "cuperiod-prewhiten-project"
#: The schema version this module writes (and the newest it reads).
VERSION: Final = 1
#: Conventional file-name suffix.
SUFFIX: Final = ".cupw.json"

#: ``_MIGRATIONS[v]`` turns a version-``v`` document into a version-``v + 1`` one. A
#: migration that meets a missing settings field writes that version's default, never
#: a value from the environment.
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

#: Largest frequency grid a project may name (8e8 bytes of float64 frequencies).
MAX_GRID_POINTS: Final = 100_000_000
#: Most points a project may hold.
MAX_POINTS: Final = 50_000_000
#: Relative tolerance between a stored component value and the rebuilt fit.
RESTORE_RTOL: Final = 1e-9
#: Tolerances of ``verify``: frequencies (relative), amplitudes (relative) and
#: phases (radians).
VERIFY_FREQUENCY_RTOL: Final = 1e-9
VERIFY_AMPLITUDE_RTOL: Final = 1e-6
VERIFY_PHASE_ATOL: Final = 1e-6

#: Extensions a referenced data file may have (a compression suffix may follow).
DATA_SUFFIXES: Final = frozenset(
    {
        ".csv",
        ".ecsv",
        ".tsv",
        ".tab",
        ".dat",
        ".txt",
        ".fits",
        ".fit",
        ".fts",
        ".fz",
        ".parquet",
        ".pq",
    }
)
_COMPRESSION: Final = frozenset({".gz", ".bz2", ".xz"})

#: What a hostile or broken file can raise while being read and validated; ``load``
#: re-raises each as ProjectError.
_UNTRUSTED: tuple[type[BaseException], ...] = (
    ValidationError,
    ValueError,  # includes json.JSONDecodeError and UnicodeDecodeError
    TypeError,
    KeyError,
    IndexError,
    OverflowError,
    RecursionError,
    MemoryError,
    OSError,
    CuPeriodError,  # SessionError, InsufficientDataError, BackendUnavailableError
)

_TWO_PI: Final = 2.0 * math.pi
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_S = TypeVar("_S", PreWhitenSettings, SessionSettings)


# --- the schema (enforced; section 2.10 of the design documents it) ---------------


class _Strict(BaseModel):
    """No unknown key, no coercion (``"1"`` is not a number), finite floats only."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


Text64 = Annotated[str, Field(max_length=MAX_LABEL_LENGTH)]
Pair = Annotated[list[int], Field(min_length=2, max_length=2)]
Triple = Annotated[list[int], Field(min_length=3, max_length=3)]


class GridModel(_Strict):
    f0: float
    df: float = Field(gt=0.0)
    nf: int = Field(ge=1, le=MAX_GRID_POINTS)


class TieModel(_Strict):
    parents: list[int] = Field(min_length=1, max_length=MAX_TIE_ORDER)
    coefficients: list[int] = Field(min_length=1, max_length=MAX_TIE_ORDER)


class ComponentModel(_Strict):
    id: int = Field(ge=1)
    frequency: float = Field(gt=0.0)
    amplitude: float
    phase: float
    active: bool
    fix: list[bool] = Field(min_length=3, max_length=3)
    tie: TieModel | None
    variation: Literal["none", "amplitude", "phase", "both"]
    subset_amplitude: list[float | None] | None = None
    subset_phase: list[float | None] | None = None
    origin: Literal["auto", "manual"]
    fap: float | None = None
    delta_bic: float | None = None
    note: str = Field(default="", max_length=200)


class PartitionModel(_Strict):
    name: str = Field(min_length=1, max_length=64)
    labels: list[Text64] = Field(min_length=1)
    runs: list[Triple]
    weights: list[float]
    origin: dict[str, Any]


class StatsModel(_Strict):
    rss: float | None
    rms: float | None
    bic: float | None
    n_samples: int = Field(ge=0)
    n_parameters: int = Field(ge=0)
    weighted: bool


class MaskModel(_Strict):
    n: int = Field(ge=0, le=MAX_POINTS)
    deleted: list[Pair]


class StateModel(_Strict):
    components: list[ComponentModel] = Field(max_length=10_000)
    offset: float
    subset_offsets: list[float | None] | None
    stats: StatsModel
    mask: MaskModel
    partitions: list[PartitionModel] = Field(max_length=64)
    excluded: list[list[Any]]
    offsets_by: str | None
    variation_by: str | None
    weights_by: str | None
    settings: dict[str, Any]
    grid: GridModel
    blocked: list[float]
    n_iterations: int = Field(ge=0)
    budget_origin: int = Field(ge=0)
    n_pruned: int = Field(ge=0)
    status: Literal["searching", "stopped"]
    stop_reason: str = Field(max_length=500)
    next_id: int = Field(ge=1)


class CreationModel(_Strict):
    backend: str = Field(max_length=32)
    t_ref: float
    grid: GridModel
    settings: dict[str, Any]
    session_settings: dict[str, Any]
    partitions: list[PartitionModel] = Field(max_length=64)
    label_names: list[Text64] = Field(max_length=16)


class ColumnsModel(_Strict):
    time: str | None = Field(default=None, max_length=256)
    value: str | None = Field(default=None, max_length=256)
    error: str | None = Field(default=None, max_length=256)
    band: str | None = Field(default=None, max_length=256)


class DataModel(_Strict):
    mode: Literal["embedded", "reference"]
    n: int = Field(ge=3, le=MAX_POINTS)
    domain: Literal["magnitude", "flux"]
    has_error: bool
    content_sha256: str = Field(pattern=_SHA256.pattern)
    path: str | None = Field(default=None, max_length=1024)
    columns: ColumnsModel | None = None
    time: list[float] | None = None
    value: list[float] | None = None
    error: list[float] | None = None
    labels: dict[Text64, list[str]] = Field(default_factory=dict, max_length=16)
    meta: dict[str, Any] = Field(default_factory=dict)


class ReportModel(_Strict):
    kind: Literal["accepted", "blocked", "rejected", "stopped", "edited"]
    message: str = Field(max_length=500)
    component_id: int | None = None
    frequency: float | None = None
    statistics: dict[Text64, float | None] = Field(default_factory=dict)
    warning: str = Field(default="", max_length=500)
    # Only refusals carry a hint and they are never logged, so files hold none; the
    # field keeps StepReport.to_dict()/from_dict() round-tripping through a project.
    hint: RefusalHint | None = None


class LogEntryModel(_Strict):
    seq: int = Field(ge=1)
    group: int = Field(ge=1)
    op: str = Field(max_length=32)
    args: dict[str, Any]
    report: ReportModel
    timestamp: str = Field(max_length=64)


class ProjectModel(_Strict):
    """A whole project file, version :data:`VERSION`."""

    format: Literal["cuperiod-prewhiten-project"]
    version: int
    created_with: dict[str, Any]
    saved: str = Field(max_length=64)
    title: str = Field(default="", max_length=200)
    data: DataModel
    creation: CreationModel
    state: StateModel
    log: list[LogEntryModel]
    ui: dict[str, Any] = Field(default_factory=dict)


# --- small helpers --------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _nans(values: Sequence[float | None] | None) -> tuple[float, ...] | None:
    return None if values is None else tuple(_nan(v) for v in values)


def _opt(value: float) -> float | None:
    """A float for JSON: ``None`` when it is not finite."""
    x = float(value)
    return x if math.isfinite(x) else None


def _opts(values: Iterable[float] | None) -> list[float | None] | None:
    return None if values is None else [_opt(v) for v in values]


def _finite(value: float, where: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ProjectError(f"{where} is not finite ({x!r}); it cannot be saved")
    return x


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a JSON number")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text[:32]} overflows a float")
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object, refusing a repeated key (which would silently shadow one)."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key[:64]!r}")
        out[key] = value
    return out


def _plain_meta(meta: Mapping[Any, Any]) -> dict[str, Any]:
    """The JSON-safe scalar entries of a light curve's metadata (others are dropped)."""
    out: dict[str, Any] = {}
    for key, raw in meta.items():
        if len(out) >= 64:
            break
        if not isinstance(key, str) or len(key) > 64:
            continue
        value = raw.item() if isinstance(raw, np.generic) else raw
        keep = (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, int) and abs(value) < 2**53)
            or (isinstance(value, float) and math.isfinite(value))
            or (isinstance(value, str) and len(value) <= 4096)
        )
        if keep:
            out[key] = value
    return out


def _settings(model: type[_S], raw: Mapping[str, Any], where: str) -> _S:
    """Validate a complete settings dump without ever consulting the environment.

    A field the file lacks takes the model's declared default — ``model_validate``
    on a partial mapping would read ``CUPERIOD_*`` variables for it instead.
    """
    fields = model.model_fields
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ProjectError(f"{where}: unknown settings {unknown}")
    full = dict(raw)
    for name, info in fields.items():
        if name not in full:
            full[name] = info.get_default(call_default_factory=True)
    try:
        out: _S = model.model_validate(full)
    except ValidationError as exc:
        raise ProjectError(f"{where}: {_brief(exc)}") from exc
    return out


def _brief(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'value'}: {err['msg']}"
        for err in exc.errors()[:5]
    )


def created_with() -> dict[str, str]:
    """Versions of what wrote a project (informative only)."""
    from cuperiod import __version__

    return {
        "cuperiod": str(__version__),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
    }


# --- serialization ------------------------------------------------------------------


def _grid_dict(grid: tuple[float, float, int]) -> dict[str, Any]:
    return {
        "f0": _finite(grid[0], "grid.f0"),
        "df": _finite(grid[1], "grid.df"),
        "nf": int(grid[2]),
    }


def _partition_dict(partition: Partition) -> dict[str, Any]:
    return {
        "name": partition.name,
        "labels": list(partition.labels),
        "runs": [list(r) for r in partition.runs()],
        "weights": [float(w) for w in partition.weights],
        "origin": partition.origin.model_dump(mode="json"),
    }


def _component_dict(c: Component) -> dict[str, Any]:
    where = f"state.components (F{c.id})"
    return {
        "id": int(c.id),
        "frequency": _finite(c.frequency, f"{where}.frequency"),
        "amplitude": _finite(c.amplitude, f"{where}.amplitude"),
        "phase": _finite(c.phase, f"{where}.phase"),
        "active": bool(c.active),
        "fix": [bool(c.fix_frequency), bool(c.fix_amplitude), bool(c.fix_phase)],
        "tie": None
        if c.tie is None
        else {
            "parents": [int(p) for p in c.tie.parents],
            "coefficients": [int(k) for k in c.tie.coefficients],
        },
        "variation": c.variation,
        "subset_amplitude": _opts(c.subset_amplitude),
        "subset_phase": _opts(c.subset_phase),
        "origin": c.origin,
        "fap": _opt(c.fap),
        "delta_bic": _opt(c.delta_bic),
        "note": c.note,
    }


def state_dict(state: SessionState) -> dict[str, Any]:
    """A state as the project file stores it (the ``serial`` is not stored)."""
    stats = state.stats
    return {
        "components": [_component_dict(c) for c in state.components],
        "offset": _finite(state.offset, "state.offset"),
        "subset_offsets": _opts(state.subset_offsets),
        "stats": {
            "rss": _opt(stats.rss),
            "rms": _opt(stats.rms),
            "bic": _opt(stats.bic),
            "n_samples": int(stats.n_samples),
            "n_parameters": int(stats.n_parameters),
            "weighted": bool(stats.weighted),
        },
        "mask": {"n": state.mask.n, "deleted": [list(r) for r in state.mask.runs()]},
        "partitions": [_partition_dict(p) for p in state.partitions],
        "excluded": [
            [name, [int(s) for s in subsets]] for name, subsets in state.excluded
        ],
        "offsets_by": state.offsets_by,
        "variation_by": state.variation_by,
        "weights_by": state.weights_by,
        "settings": settings_dump(state.settings),
        "grid": _grid_dict(state.grid),
        "blocked": [_finite(b, "state.blocked") for b in state.blocked],
        "n_iterations": int(state.n_iterations),
        "budget_origin": int(state.budget_origin),
        "n_pruned": int(state.n_pruned),
        "status": state.status,
        "stop_reason": state.stop_reason[:500],
        "next_id": int(state.next_id),
    }


def _creation_dict(creation: CreationArgs) -> dict[str, Any]:
    return {
        "backend": creation.backend[:32],
        "t_ref": _finite(creation.t_ref, "creation.t_ref"),
        "grid": _grid_dict(creation.grid),
        "settings": settings_dump(creation.settings),
        "session_settings": creation.session_settings.model_dump(mode="json"),
        "partitions": [_partition_dict(p) for p in creation.partitions],
        "label_names": list(creation.label_names),
    }


def _columns_dict(columns: ColumnMap | None) -> dict[str, str | None] | None:
    if columns is None:
        return None
    return {
        "time": columns.time,
        "value": columns.value,
        "error": columns.error,
        "band": columns.band,
    }


def _data_dict(
    data: SessionData,
    *,
    reference: str | None,
    columns: ColumnMap | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": "embedded" if reference is None else "reference",
        "n": data.n,
        "domain": str(Domain(data.domain).value),
        "has_error": data.error is not None,
        "content_sha256": data.content_sha256,
        "meta": _plain_meta(data.meta),
    }
    if reference is not None:
        out["path"] = reference
        out["columns"] = _columns_dict(columns)
        return out
    out["time"] = np.asarray(data.time, dtype=np.float64).tolist()
    out["value"] = np.asarray(data.value, dtype=np.float64).tolist()
    out["error"] = (
        None
        if data.error is None
        else np.asarray(data.error, dtype=np.float64).tolist()
    )
    out["labels"] = {
        name: [str(v) for v in np.asarray(column).tolist()]
        for name, column in data.labels.items()
    }
    return out


# --- data references ----------------------------------------------------------------


def _data_suffix_ok(name: str) -> bool:
    suffixes = [s.lower() for s in PurePosixPath(name).suffixes]
    if suffixes and suffixes[-1] in _COMPRESSION:
        suffixes = suffixes[:-1]
    return bool(suffixes) and suffixes[-1] in DATA_SUFFIXES


def _session_columns(session: PreWhitenSession) -> ColumnMap | None:
    columns = getattr(session, "_columns", None)
    return columns if isinstance(columns, ColumnMap) else None


def source_file(session: PreWhitenSession) -> tuple[Path, str] | tuple[None, str]:
    """The file the session's data can be re-read from, or why there is none.

    The file is ``meta["source"]``; it qualifies when the session has no label columns
    and :meth:`LightCurve.from_file` of it (same columns and domain) reproduces the
    session's ``content_sha256`` exactly.
    """
    data = session.kernel.data
    source = data.meta.get("source")
    if not isinstance(source, str) or not source:
        return None, "the data have no source file"
    if data.labels:
        return None, "the session has label columns, which only an embedded copy keeps"
    columns = _session_columns(session)
    try:
        path = Path(source).resolve()
        if not path.is_file():
            return None, f"the source file {source!r} does not exist"
        if not _data_suffix_ok(path.name):
            return (
                None,
                f"the source file {path.name!r} is not a known light-curve type",
            )
        stat = path.stat()
        key = (
            str(path), stat.st_mtime_ns, stat.st_size,
            repr(_columns_dict(columns)),
            str(data.domain),
        )
        digest = _SOURCE_DIGESTS.get(key)
        if digest is None:
            lc = LightCurve.from_file(path, columns=columns, domain=data.domain)
            lc = lc.finite()
            digest = content_sha256(lc.time, lc.value, lc.error)
            # Saving (or autosaving) re-checks the file each time; re-reading a large
            # FITS file for an unchanged (path, mtime, size) would be wasted.
            if len(_SOURCE_DIGESTS) >= 16:
                _SOURCE_DIGESTS.pop(next(iter(_SOURCE_DIGESTS)))
            _SOURCE_DIGESTS[key] = digest
    except (OSError, ValueError, CuPeriodError) as exc:
        return None, f"the source file cannot be read: {exc}"
    if digest != data.content_sha256:
        return None, "the source file no longer holds the session's data"
    return path, ""


#: Content hash of recently checked source files, by (path, mtime, size, columns,
#: domain).
_SOURCE_DIGESTS: dict[tuple[Any, ...], str] = {}


def _reference(session: PreWhitenSession, directory: Path) -> tuple[str | None, str]:
    """A relative path inside ``directory`` to the session's data file, or why not."""
    path, why = source_file(session)
    if path is None:
        return None, why
    try:
        relative = path.relative_to(directory.resolve())
    except ValueError:
        return None, "the source file is not inside the project's directory"
    return relative.as_posix(), ""


def _resolve_reference(text: str, directory: Path) -> Path:
    """A stored data reference as a path inside ``directory`` (ProjectError else)."""
    if "\x00" in text or not text.strip():
        raise ProjectError("data.path: an empty or malformed path")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", text) or text.startswith(("\\", "//")):
        # A URL, a drive ("C:..."), or a UNC path ("\\host\share", "//host/share").
        raise ProjectError(f"data.path: {text!r} is not a relative path")
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ProjectError(f"data.path: {text!r} is not a relative path")
    if ".." in posix.parts or ".." in windows.parts:
        raise ProjectError(f"data.path: {text!r} leaves the project's directory")
    if not _data_suffix_ok(windows.name):
        raise ProjectError(f"data.path: {text!r} is not a known light-curve file type")
    base = directory.resolve()
    target = (base / Path(*windows.parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ProjectError(
            f"data.path: {text!r} resolves outside the project's directory"
        ) from None
    if not target.is_file():
        raise ProjectError(
            f"data file {text!r} not found next to the project; pass data= to "
            "supply the light curve"
        )
    return target


# --- writing --------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file beside ``path``, then move it into place."""
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def document(
    session: PreWhitenSession,
    *,
    reference: str | None = None,
    title: str | None = None,
    ui: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The project document of ``session`` (data embedded unless ``reference``)."""
    snapshot = session.snapshot()
    data = session.kernel.data
    if len(data.labels) > 16:
        raise ProjectError("a project keeps at most 16 label columns")
    return {
        "format": FORMAT,
        "version": VERSION,
        "created_with": created_with(),
        "saved": _now(),
        "title": (title or "")[:200],
        "data": _data_dict(
            data, reference=reference, columns=_session_columns(session)
        ),
        "creation": _creation_dict(snapshot.creation),
        "state": state_dict(snapshot.head),
        "log": [entry.to_dict() for entry in snapshot.log],
        "ui": dict(ui or {}),
    }


def save(
    session: PreWhitenSession,
    path: str | Path,
    *,
    embed_data: Literal["auto", "always", "never"] = "auto",
    title: str | None = None,
    ui: Mapping[str, Any] | None = None,
) -> None:
    """Write ``session`` as a project file (atomically).

    Parameters
    ----------
    session : PreWhitenSession
    path : str or Path
        The file to write (conventionally ``*.cupw.json``).
    embed_data : {"auto", "always", "never"}
        ``"auto"`` references the data file when it lies in the project's directory,
        the session has no label columns and the file still reproduces the data's
        content hash; otherwise it embeds the data, refusing more than
        ``embed_data_max_points``. ``"always"`` embeds, ``"never"`` references (or
        fails).
    title : str, optional
        Free text (at most 200 characters).
    ui : Mapping, optional
        Free-form GUI state, stored as given (JSON data only).

    Raises
    ------
    ProjectError
        When the data can be neither referenced nor embedded as asked, or the file
        would exceed ``max_project_bytes``.
    """
    target = Path(path)
    if embed_data not in ("auto", "always", "never"):
        raise ProjectError("embed_data must be 'auto', 'always' or 'never'")
    data = session.kernel.data
    limits = session.session_settings
    reference: str | None = None
    if embed_data != "always":
        reference, why = _reference(session, target.parent)
        if reference is None and embed_data == "never":
            raise ProjectError(f"cannot reference the data: {why}")
    if (
        reference is None
        and embed_data == "auto"
        and (data.n > limits.embed_data_max_points)
    ):
        raise ProjectError(
            f"{data.n} points exceed embed_data_max_points "
            f"({limits.embed_data_max_points}); save with embed_data='always', or "
            "keep the data file next to the project so it can be referenced"
        )
    doc = document(session, reference=reference, title=title, ui=ui)
    try:
        ProjectModel.model_validate(doc)
        text = json.dumps(doc, allow_nan=False, separators=(",", ":"))
    except (ValidationError, ValueError, TypeError) as exc:
        detail = _brief(exc) if isinstance(exc, ValidationError) else str(exc)
        raise ProjectError(f"the session cannot be saved: {detail}") from exc
    size = len(text.encode("utf-8"))
    if size > limits.max_project_bytes:
        raise ProjectError(
            f"the project would be {size} bytes, more than max_project_bytes "
            f"({limits.max_project_bytes})"
        )
    _atomic_write(target, text)


# --- reading -------------------------------------------------------------------------


def read_document(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    max_actions: int | None = None,
) -> dict[str, Any]:
    """Read, size-check, parse and migrate a project file to the current version.

    Returns the plain document; :class:`ProjectModel` validates it.

    Raises
    ------
    ProjectError
        For anything but a readable project of version :data:`VERSION` or older.
    """
    file = Path(path)
    # Reader limits come from the environment (the file's own settings are unread).
    limits = SessionSettings()
    cap = limits.max_project_bytes if max_bytes is None else int(max_bytes)
    most = limits.max_actions if max_actions is None else int(max_actions)
    try:
        size = file.stat().st_size
        if size > cap:
            raise ProjectError(
                f"{file.name} is {size} bytes, more than max_project_bytes ({cap}); "
                "it was not read"
            )
        with file.open("rb") as fh:
            raw = fh.read(cap + 1)
        if len(raw) > cap:
            raise ProjectError(f"{file.name} grew past max_project_bytes ({cap})")
        doc = json.loads(
            raw.decode("utf-8-sig"),
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_keys,
        )
    except ProjectError:
        raise
    except _UNTRUSTED as exc:
        raise ProjectError(
            f"{file.name} is not a readable project file: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise ProjectError(f"{file.name} is not a project file (no JSON object)")
    if doc.get("format") != FORMAT:
        raise ProjectError(f"{file.name} is not a cuPeriod pre-whitening project")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ProjectError(f"{file.name}: invalid project version {version!r}")
    if version > VERSION:
        raise ProjectError(
            f"{file.name} was written by a newer cuPeriod (project version {version}; "
            f"this one reads versions up to {VERSION})"
        )
    while version < VERSION:
        migrate = _MIGRATIONS.get(version)
        if migrate is None:
            raise ProjectError(
                f"{file.name}: no migration from project version {version}"
            )
        try:
            doc = migrate(doc)
        except _UNTRUSTED as exc:
            raise ProjectError(
                f"{file.name}: migrating from version {version} failed: {exc}"
            ) from exc
        version += 1
        doc["version"] = version
    log = doc.get("log")
    if isinstance(log, list) and len(log) > most:
        raise ProjectError(
            f"{file.name}: the log holds {len(log)} actions, more than max_actions "
            f"({most})"
        )
    out: dict[str, Any] = doc
    return out


def _partition(model: PartitionModel, n: int, where: str) -> Partition:
    try:
        return Partition.from_runs(
            n,
            model.runs,
            name=model.name,
            labels=model.labels,
            weights=model.weights,
            origin=parse_origin(model.origin),
        )
    except ValidationError as exc:
        raise ProjectError(f"{where} ({model.name!r}).origin: {_brief(exc)}") from exc
    except ValueError as exc:
        raise ProjectError(f"{where} ({model.name!r}): {exc}") from exc


def _component(model: ComponentModel, where: str) -> Component:
    tie = None
    if model.tie is not None:
        if len(model.tie.parents) != len(model.tie.coefficients):
            raise ProjectError(f"{where}.tie: one coefficient per parent is needed")
        tie = Tie(tuple(model.tie.parents), tuple(model.tie.coefficients))
    return Component(
        id=model.id,
        frequency=model.frequency,
        amplitude=model.amplitude,
        phase=model.phase,
        active=model.active,
        fix_frequency=model.fix[0],
        fix_amplitude=model.fix[1],
        fix_phase=model.fix[2],
        tie=tie,
        variation=model.variation,
        subset_amplitude=_nans(model.subset_amplitude),
        subset_phase=_nans(model.subset_phase),
        origin=model.origin,
        fap=_nan(model.fap),
        delta_bic=_nan(model.delta_bic),
        note=model.note,
    )


def _excluded(raw: list[list[Any]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    out: list[tuple[str, tuple[int, ...]]] = []
    for i, item in enumerate(raw):
        where = f"state.excluded[{i}]"
        if (
            len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], list)
        ):
            raise ProjectError(f"{where}: expected [partition name, [subset, ...]]")
        subsets = item[1]
        if not all(isinstance(s, int) and not isinstance(s, bool) for s in subsets):
            raise ProjectError(f"{where}: subset indices must be integers")
        out.append((item[0], tuple(int(s) for s in subsets)))
    return tuple(out)


def _mask(model: MaskModel) -> PointMask:
    previous = 0
    for start, stop in model.deleted:
        if not (previous <= start < stop <= model.n):
            raise ProjectError(
                f"state.mask: deleted runs must be sorted, disjoint and inside "
                f"[0, {model.n}); got ({start}, {stop})"
            )
        previous = stop
    return PointMask.from_runs(model.n, model.deleted)


def build_state(
    model: StateModel, *, serial: int = 0, n: int | None = None
) -> SessionState:
    """A :class:`SessionState` from its validated file form (validated next)."""
    size = model.mask.n if n is None else n
    components = tuple(
        _component(c, f"state.components[{i}]") for i, c in enumerate(model.components)
    )
    partitions = tuple(
        _partition(p, size, f"state.partitions[{i}]")
        for i, p in enumerate(model.partitions)
    )
    stats = model.stats
    return SessionState(
        serial=serial,
        components=components,
        offset=model.offset,
        subset_offsets=_nans(model.subset_offsets),
        stats=FitStats(
            rss=_nan(stats.rss),
            rms=_nan(stats.rms),
            bic=_nan(stats.bic),
            n_samples=stats.n_samples,
            n_parameters=stats.n_parameters,
            weighted=stats.weighted,
        ),
        mask=_mask(model.mask),
        partitions=partitions,
        excluded=_excluded(model.excluded),
        offsets_by=model.offsets_by,
        variation_by=model.variation_by,
        weights_by=model.weights_by,
        settings=_settings(PreWhitenSettings, model.settings, "state.settings"),
        grid=(model.grid.f0, model.grid.df, model.grid.nf),
        blocked=tuple(model.blocked),
        n_iterations=model.n_iterations,
        budget_origin=model.budget_origin,
        n_pruned=model.n_pruned,
        status=model.status,
        stop_reason=model.stop_reason,
        next_id=model.next_id,
    )


def log_entry(raw: LogEntry | Mapping[str, Any]) -> LogEntry:
    """A validated :class:`LogEntry` from its JSON form (or a LogEntry, checked).

    Raises
    ------
    ProjectError
        For an unknown op, invalid arguments or a malformed report.
    """
    data = raw.to_dict() if isinstance(raw, LogEntry) else raw
    try:
        model = LogEntryModel.model_validate(dict(data))
    except ValidationError as exc:
        raise ProjectError(f"log entry: {_brief(exc)}") from exc
    return _entry(model)


def _entry(model: LogEntryModel) -> LogEntry:
    where = f"log entry {model.seq} ({model.op})"
    try:
        parsed = actions.parse(model.op, model.args)
    except SessionError as exc:
        raise ProjectError(f"{where}: {exc}") from exc
    report = model.report
    if report.kind not in COMMITTED_KINDS:  # pragma: no cover - the model forbids it
        raise ProjectError(f"{where}: report kind {report.kind!r} is never logged")
    return LogEntry(
        seq=model.seq,
        group=model.group,
        op=model.op,
        args=actions.canonical(parsed),
        report=StepReport.from_dict(report.model_dump()),
        timestamp=model.timestamp,
    )


def _log(models: Sequence[LogEntryModel]) -> tuple[LogEntry, ...]:
    entries = tuple(_entry(model) for model in models)
    _check_sequence(entries)
    return entries


def _check_sequence(entries: Sequence[LogEntry]) -> None:
    for position, entry in enumerate(entries, start=1):
        if entry.seq != position:
            raise ProjectError(
                f"log entry {position}: seq {entry.seq} breaks the sequence 1, 2, ..."
            )


# --- referential integrity -----------------------------------------------------------


def _fail(field: str, message: str) -> NoReturn:
    raise ProjectError(f"state.{field}: {message}")


def _check_ties(state: SessionState) -> None:
    by_id = {c.id: c for c in state.components}
    for c in state.components:
        if c.tie is None:
            continue
        where = f"components (F{c.id}).tie"
        parents, coefficients = c.tie.parents, c.tie.coefficients
        if len(parents) > MAX_TIE_ORDER or len(set(parents)) != len(parents):
            _fail(where, f"parents must be distinct (at most {MAX_TIE_ORDER})")
        for parent, k in zip(parents, coefficients, strict=True):
            if parent == c.id:
                _fail(where, f"F{c.id} is tied to itself")
            if parent not in by_id:
                _fail(where, f"parent F{parent} does not exist")
            if not isinstance(k, int) or k == 0 or abs(k) > MAX_TIE_COEFFICIENT:
                _fail(
                    where,
                    f"coefficient {k!r} is not a non-zero integer within "
                    f"+-{MAX_TIE_COEFFICIENT}",
                )
        # Ties go through the grammar's own parser, as a typed tie would.
        try:
            ids, coefs = parse_tie(format_tie(parents, coefficients), child=c.id)
        except ValueError as exc:
            _fail(where, str(exc))
        if sorted(zip(ids, coefs, strict=True)) != sorted(
            zip(parents, coefficients, strict=True)
        ):
            _fail(where, "the tie does not survive its own grammar")
    # Acyclic: depth-first search with colouring over child -> parents.
    colour: dict[int, int] = {}

    def visit(node: int, path: list[int]) -> None:
        colour[node] = 1
        tie = by_id[node].tie
        for parent in tie.parents if tie is not None else ():
            if colour.get(parent) == 1:
                cycle = " -> ".join(f"F{i}" for i in [*path, node, parent])
                _fail("components.tie", f"cyclic tie: {cycle}")
            if colour.get(parent, 0) == 0:
                visit(parent, [*path, node])
        colour[node] = 2

    for ident in by_id:
        if colour.get(ident, 0) == 0:
            visit(ident, [])


def validate_state(state: SessionState, data: SessionData) -> None:
    """Check every cross reference of ``state`` against itself and ``data``.

    Raises
    ------
    ProjectError
        Naming the first field that fails: component ids unique and below
        ``next_id``; ties to existing, distinct components without a cycle; partitions
        uniquely named and as long as the data; roles and exclusions naming existing
        partitions and subsets; per-subset arrays as long as their partition; the
        mask as long as the data; ``budget_origin <= n_iterations``; finite blocked
        seeds; a valid grid; settings that validate.
    """
    n = data.n
    seen: set[int] = set()
    for i, c in enumerate(state.components):
        where = f"components[{i}] (F{c.id})"
        if not 1 <= c.id < state.next_id:
            _fail(f"{where}.id", f"id {c.id} is not in [1, next_id={state.next_id})")
        if c.id in seen:
            _fail(f"{where}.id", f"duplicate id {c.id}")
        seen.add(c.id)
        if not (math.isfinite(c.frequency) and c.frequency > 0.0):
            _fail(f"{where}.frequency", f"{c.frequency!r} is not a positive number")
        if not (math.isfinite(c.amplitude) and math.isfinite(c.phase)):
            _fail(where, "amplitude and phase must be finite")
        if c.variation not in ("none", "amplitude", "phase", "both"):
            _fail(f"{where}.variation", f"unknown mode {c.variation!r}")
        if c.origin not in ("auto", "manual"):
            _fail(f"{where}.origin", f"unknown origin {c.origin!r}")
    _check_ties(state)

    names: dict[str, Partition] = {}
    for i, p in enumerate(state.partitions):
        if p.name in names:
            _fail(f"partitions[{i}].name", f"duplicate partition {p.name!r}")
        if p.n_points != n:
            _fail(
                f"partitions[{i}] ({p.name!r})",
                f"{p.n_points} codes for {n} points",
            )
        if len(p.labels) != len(p.weights) or (
            p.codes.size and int(p.codes.max()) >= len(p.labels)
        ):
            _fail(f"partitions[{i}] ({p.name!r})", "codes, labels and weights disagree")
        names[p.name] = p
    for role in ("offsets_by", "variation_by", "weights_by"):
        name = getattr(state, role)
        if name is not None and name not in names:
            _fail(role, f"no partition named {name!r}")
    excluded_names: set[str] = set()
    for i, (name, subsets) in enumerate(state.excluded):
        if name not in names:
            _fail(f"excluded[{i}]", f"no partition named {name!r}")
        if name in excluded_names:
            _fail(f"excluded[{i}]", f"partition {name!r} listed twice")
        excluded_names.add(name)
        count = names[name].n_subsets
        if len(set(subsets)) != len(subsets) or any(
            not 0 <= s < count for s in subsets
        ):
            _fail(
                f"excluded[{i}] ({name!r})",
                f"subset indices must be distinct and in [0, {count})",
            )
    if state.subset_offsets is not None:
        if state.offsets_by is None:
            _fail("subset_offsets", "zero points without an offsets partition")
        else:
            expected = names[state.offsets_by].n_subsets
            if len(state.subset_offsets) != expected:
                _fail(
                    "subset_offsets",
                    f"{len(state.subset_offsets)} values for {expected} subsets of "
                    f"{state.offsets_by!r}",
                )
    for i, c in enumerate(state.components):
        for attribute in ("subset_amplitude", "subset_phase"):
            values = getattr(c, attribute)
            if values is None:
                continue
            where = f"components[{i}] (F{c.id}).{attribute}"
            if state.variation_by is None:
                _fail(where, "per-subset values without a variation partition")
            else:
                expected = names[state.variation_by].n_subsets
                if len(values) != expected:
                    _fail(
                        where,
                        f"{len(values)} values for {expected} subsets of "
                        f"{state.variation_by!r}",
                    )
    if state.mask.n != n:
        _fail("mask.n", f"{state.mask.n} for {n} points")
    if state.n_iterations < 0 or state.n_pruned < 0:
        _fail("n_iterations", "counters must not be negative")
    if not 0 <= state.budget_origin <= state.n_iterations:
        _fail(
            "budget_origin",
            f"{state.budget_origin} is not in [0, n_iterations={state.n_iterations}]",
        )
    if state.next_id < 1:
        _fail("next_id", "must be at least 1")
    if not all(math.isfinite(b) for b in state.blocked):
        _fail("blocked", "blocked seeds must be finite")
    f0, df, nf = state.grid
    if not (math.isfinite(f0) and math.isfinite(df) and df > 0.0 and int(nf) >= 1):
        _fail("grid", f"({f0!r}, {df!r}, {nf!r}) is not a uniform grid")
    if state.status not in ("searching", "stopped"):
        _fail("status", f"unknown status {state.status!r}")
    if not math.isfinite(state.offset):
        _fail("offset", "must be finite")
    try:
        PreWhitenSettings.model_validate(state.settings.model_dump())
    except ValidationError as exc:
        _fail("settings", _brief(exc))


# --- opening a project ---------------------------------------------------------------


def _backend_available(name: str) -> bool:
    from cuperiod.prewhiten.spectrum import resolve_spectrum_backend

    try:
        resolve_spectrum_backend(name)
    except BackendUnavailableError:
        return False
    return True


def _requested(concrete: str) -> str:
    """The backend name to ask for to run like ``concrete`` again."""
    return "torch" if concrete.startswith("torch") else concrete


def choose_backend(recorded: str, backend: str | None) -> str:
    """``backend`` if given; else the recorded backend when this machine has it.

    Otherwise the best CPU backend, with a :class:`SessionWarning` naming both (a file
    never opens a GPU context it did not ask for). Seeded replay makes the log
    independent of the choice.
    """
    if backend is not None:
        return backend
    if recorded and _backend_available(recorded):
        return _requested(recorded)
    warnings.warn(
        f"the project ran on {recorded!r}, which is not available here; using the "
        "best CPU backend instead",
        SessionWarning,
        stacklevel=5,
    )
    return "cpu"


def _embedded(model: DataModel) -> LightCurve:
    if model.time is None or model.value is None:
        raise ProjectError("data: an embedded project needs time and value arrays")
    arrays = {"time": model.time, "value": model.value}
    if model.has_error:
        if model.error is None:
            raise ProjectError("data.error: has_error is set but no errors are stored")
        arrays["error"] = model.error
    elif model.error is not None:
        raise ProjectError("data.error: errors are stored but has_error is false")
    for name, values in arrays.items():
        if len(values) != model.n:
            raise ProjectError(f"data.{name}: {len(values)} values for n={model.n}")
    for name, column in model.labels.items():
        if len(column) != model.n:
            raise ProjectError(
                f"data.labels[{name!r}]: {len(column)} values for n={model.n}"
            )
    return LightCurve.from_arrays(
        np.asarray(model.time, dtype=np.float64),
        np.asarray(model.value, dtype=np.float64),
        None if model.error is None else np.asarray(model.error, dtype=np.float64),
        domain=model.domain,
        meta=_plain_meta(model.meta),
    )


def _columns(model: ColumnsModel | None) -> ColumnMap | None:
    if model is None:
        return None
    return ColumnMap(
        time=model.time, value=model.value, error=model.error, band=model.band
    )


def _session_data(lc: LightCurve, labels: Mapping[str, np.ndarray]) -> SessionData:
    return SessionData(
        time=lc.time,
        value=lc.value,
        error=lc.error,
        domain=lc.domain,
        source_index=np.arange(lc.n, dtype=np.int64),
        labels=dict(labels),
        meta=dict(lc.meta),
        content_sha256=content_sha256(lc.time, lc.value, lc.error),
    )


def _coerce(data: Any) -> LightCurve:
    from cuperiod.prewhiten.engine import _as_light_curve

    return _as_light_curve(data, None, None).finite()


def _open(
    project: ProjectModel,
    file: Path,
    *,
    data: Any | None,
    verify: bool,
    allow_modified_data: bool,
    backend: str | None,
) -> PreWhitenSession:
    from cuperiod.prewhiten.session.session import PreWhitenSession

    stored = project.data
    columns = _columns(stored.columns) if stored.mode == "reference" else None
    if data is not None:
        lc = _coerce(data)
    elif stored.mode == "embedded":
        lc = _embedded(stored)
    else:
        if stored.path is None:
            raise ProjectError("data.path: a referenced project needs a path")
        target = _resolve_reference(stored.path, file.parent)
        lc = LightCurve.from_file(
            target, columns=columns, domain=stored.domain
        ).finite()
    labels: dict[str, np.ndarray] = {}
    if stored.mode == "embedded" and (data is None or lc.n == stored.n):
        labels = {k: np.asarray(v, dtype=object) for k, v in stored.labels.items()}
    names = tuple(project.creation.label_names)
    if data is None and tuple(labels) != names:
        raise ProjectError(
            f"data.labels: columns {list(labels)} differ from creation.label_names "
            f"{list(names)}"
        )
    actual = content_sha256(lc.time, lc.value, lc.error)
    expected = stored.content_sha256
    if actual != expected:
        if not allow_modified_data:
            raise ProjectError(
                "the data differ from the project's (content hash mismatch: "
                f"{actual[:12]}... instead of {expected[:12]}...); pass "
                "allow_modified_data=True to re-fit the stored solution to them"
            )
        warnings.warn(
            "the data differ from the project's; the stored solution is re-fitted "
            "to them",
            SessionWarning,
            stacklevel=4,
        )
    n = lc.n
    creation_model = project.creation
    creation = CreationArgs(
        backend=creation_model.backend,
        t_ref=creation_model.t_ref,
        grid=(creation_model.grid.f0, creation_model.grid.df, creation_model.grid.nf),
        settings=_settings(
            PreWhitenSettings, creation_model.settings, "creation.settings"
        ),
        session_settings=_settings(
            SessionSettings,
            creation_model.session_settings,
            "creation.session_settings",
        ),
        partitions=tuple(
            _partition(p, n, f"creation.partitions[{i}]")
            for i, p in enumerate(creation_model.partitions)
        ),
        label_names=tuple(labels),
        content_sha256=actual,
    )
    head = build_state(project.state, n=n)
    validate_state(head, _session_data(lc, labels))
    log = _log(project.log)
    if len(log) > creation.session_settings.max_actions:
        raise ProjectError(
            f"the log holds {len(log)} actions, more than this project's max_actions "
            f"({creation.session_settings.max_actions})"
        )
    chosen = choose_backend(creation.backend, backend)
    snapshot = SessionSnapshot(creation=creation, head=head, log=log)
    session = PreWhitenSession.restore(
        snapshot, lc, backend=chosen, labels=labels or None, columns=columns
    )
    _recommit(session)
    session._saved_serial = session.state.serial  # as saved: not dirty
    if verify:
        _verify(session, head)
    return session


def _component_drift(stored: SessionState, rebuilt: SessionState) -> float:
    """Largest relative difference of the active components' values (phase / 2 pi)."""
    by_id = {c.id: c for c in rebuilt.components}
    worst = 0.0
    for c in stored.components:
        other = by_id.get(c.id)
        if other is None or not c.active:
            continue
        for a, b in ((c.frequency, other.frequency), (c.amplitude, other.amplitude)):
            scale = max(abs(a), abs(b))
            if scale > 0.0:
                worst = max(worst, abs(a - b) / scale)
        dphi = (other.phase - c.phase + math.pi) % _TWO_PI - math.pi
        worst = max(worst, abs(dphi) / _TWO_PI)
    return worst


def _recommit(session: PreWhitenSession) -> None:
    """Re-derive the restored head's fit statistics and zero points from its data.

    ``stats``, ``offset`` and ``subset_offsets`` are never trusted from a file; the
    component values are kept unless the rebuilt fit moves them by more than
    :data:`RESTORE_RTOL`, in which case the rebuilt values replace them (with a
    :class:`SessionWarning`).
    """
    kernel = session.kernel
    head = session.state
    derived = kernel.derive(head)
    mm = kernel.model_map(head)
    rebuilt, rebuilt_derived = kernel.commit_fit(
        head, derived.fit, mm=mm, spectrum=derived.scan
    )
    drift = _component_drift(head, rebuilt)
    if drift > RESTORE_RTOL:
        warnings.warn(
            f"the stored component values differ from a re-fit of the data by up to "
            f"{drift:.3g} (relative); the re-fitted values replace them",
            SessionWarning,
            stacklevel=5,
        )
        rebuilt_derived = replace(rebuilt_derived, drift=drift)
    else:
        rebuilt = replace(rebuilt, components=head.components)
    session._adopt_head(rebuilt, rebuilt_derived)


def load(
    path: str | Path,
    *,
    data: Any | None = None,
    verify: bool = False,
    allow_modified_data: bool = False,
    backend: str | None = None,
) -> PreWhitenSession:
    """Open a project file.

    Parameters
    ----------
    path : str or Path
    data : optional
        The light curve, replacing the embedded or referenced data (still checked by
        content hash).
    verify : bool, default False
        Replay the logged actions on a fresh session and compare its head with the
        stored one (frequencies to 1e-9 relative, amplitudes to 1e-6, phases to 1e-6
        rad).
    allow_modified_data : bool, default False
        Accept data whose content hash differs; the stored solution is re-fitted.
    backend : str, optional
        Overrides the recorded backend (see :func:`choose_backend`).

    Returns
    -------
    PreWhitenSession
        At the stored head; only the head is in memory (undo starts here).

    Raises
    ------
    ProjectError
        For anything wrong with the file, its data or (``verify``) its log.
    """
    file = Path(path)
    try:
        doc = read_document(file)
        project = ProjectModel.model_validate(doc)
        return _open(
            project,
            file,
            data=data,
            verify=verify,
            allow_modified_data=allow_modified_data,
            backend=backend,
        )
    except ProjectError:
        raise
    except ValidationError as exc:
        raise ProjectError(f"{file.name}: {_brief(exc)}") from exc
    except _UNTRUSTED as exc:
        raise ProjectError(f"{file.name}: {exc}") from exc


# --- replay and verify -------------------------------------------------------------


def _check_kind(entry: LogEntry, report: StepReport) -> None:
    where = f"log entry {entry.seq} ({actions.describe(entry)})"
    if report.kind == "refused":
        raise ProjectError(f"{where}: refused on replay: {report.message}")
    if report.kind == "noop":
        raise ProjectError(f"{where}: did nothing on replay ({report.message})")
    if report.kind != entry.report.kind:
        raise ProjectError(
            f"{where}: the replay was {report.kind!r} where the log says "
            f"{entry.report.kind!r} ({report.message})"
        )
    if report.warning:
        warnings.warn(
            f"log entry {entry.seq}: {report.warning}", SessionWarning, stacklevel=5
        )


def _close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def _check_close(entry: LogEntry, report: StepReport) -> None:
    """``verify``: the replayed report must name the logged component and frequency."""
    logged = entry.report
    where = f"log entry {entry.seq} ({actions.describe(entry)})"
    if logged.component_id != report.component_id:
        raise ProjectError(
            f"verify failed at {where}: component {report.component_id!r} where the "
            f"log says {logged.component_id!r}"
        )
    a, b = logged.frequency, report.frequency
    if (a is None) != (b is None) or (
        a is not None and b is not None and not _close(a, b, VERIFY_FREQUENCY_RTOL)
    ):
        raise ProjectError(
            f"verify failed at {where}: frequency {b!r} where the log says {a!r}"
        )


def _replay(
    data: Any,
    entries: Sequence[LogEntry],
    create: Mapping[str, Any],
    *,
    on_entry: Callable[[LogEntry, StepReport], None] | None = None,
) -> PreWhitenSession:
    from cuperiod.prewhiten.session.session import PreWhitenSession

    kwargs = dict(create)
    kwargs["record"] = True
    session = PreWhitenSession(data, **kwargs)
    for _, members in itertools.groupby(entries, key=lambda e: e.group):
        with session.group():
            for entry in members:
                report = session.do(entry.op, **dict(entry.args))
                _check_kind(entry, report)
                if on_entry is not None:
                    on_entry(entry, report)
                session._restamp_last(entry.timestamp)
    return session


def replay(
    data: Any, log: Sequence[LogEntry | Mapping[str, Any]], **create: Any
) -> PreWhitenSession:
    """A new session on ``data`` with every logged action applied again, in order.

    ``create`` are the constructor's keyword arguments (``settings``, ``grid``,
    ``t_ref``, ``partitions``, ``session_settings``, ...). Steps and added peaks use
    their logged seeds, so no peak search decides anything: the replay is bitwise
    the original on any backend. Entries of one logged group stay one undo group.

    Raises
    ------
    ProjectError
        For an invalid entry, or one whose outcome differs from the log (refused,
        no-op, or another report kind), naming the entry.
    """
    entries = [log_entry(e) for e in log]
    _check_sequence(entries)
    return _replay(data, entries, create)


def divergence(stored: SessionState, replayed: SessionState) -> str | None:
    """How ``replayed`` differs from ``stored`` beyond ``verify``'s tolerances."""
    if [c.id for c in stored.components] != [c.id for c in replayed.components]:
        return (
            f"components {[c.label for c in replayed.components]} instead of "
            f"{[c.label for c in stored.components]}"
        )
    for a, b in zip(stored.components, replayed.components, strict=True):
        flags_a = (
            a.active,
            a.fix_frequency,
            a.fix_amplitude,
            a.fix_phase,
            a.tie,
            a.variation,
            a.origin,
        )
        flags_b = (
            b.active,
            b.fix_frequency,
            b.fix_amplitude,
            b.fix_phase,
            b.tie,
            b.variation,
            b.origin,
        )
        if flags_a != flags_b:
            return f"{a.label}: flags {flags_b} instead of {flags_a}"
        if not _close(a.frequency, b.frequency, VERIFY_FREQUENCY_RTOL):
            return f"{a.label}: frequency {b.frequency!r} instead of {a.frequency!r}"
        if not _close(a.amplitude, b.amplitude, VERIFY_AMPLITUDE_RTOL):
            return f"{a.label}: amplitude {b.amplitude!r} instead of {a.amplitude!r}"
        dphi = (b.phase - a.phase + math.pi) % _TWO_PI - math.pi
        if abs(dphi) > VERIFY_PHASE_ATOL:
            return f"{a.label}: phase {b.phase!r} instead of {a.phase!r}"
    if stored.mask.digest != replayed.mask.digest:
        return "the deleted points differ"
    pairs = (
        (
            "partitions",
            [p.name for p in stored.partitions],
            [p.name for p in replayed.partitions],
        ),
        ("excluded", stored.excluded, replayed.excluded),
        ("offsets_by", stored.offsets_by, replayed.offsets_by),
        ("variation_by", stored.variation_by, replayed.variation_by),
        ("weights_by", stored.weights_by, replayed.weights_by),
        ("status", stored.status, replayed.status),
        ("n_iterations", stored.n_iterations, replayed.n_iterations),
        ("next_id", stored.next_id, replayed.next_id),
        ("settings", settings_dump(stored.settings), settings_dump(replayed.settings)),
    )
    for name, a_value, b_value in pairs:
        if a_value != b_value:
            return f"{name} {b_value!r} instead of {a_value!r}"
    for a_part, b_part in zip(stored.partitions, replayed.partitions, strict=True):
        if (
            a_part.labels != b_part.labels
            or a_part.weights != b_part.weights
            or not np.array_equal(a_part.codes, b_part.codes)
        ):
            return f"partition {a_part.name!r} differs"
    return None


def rerun(
    session: PreWhitenSession, *, backend: str | None = None, strict: bool = False
) -> PreWhitenSession:
    """``session``'s effective log replayed on a fresh session over the same data.

    The data, label columns and creation arguments (settings, grid, ``t_ref``, root
    partitions, session settings) are the session's own; ``backend`` defaults to the
    one it runs on. With ``strict`` every replayed report must also name the logged
    component and frequency (what ``verify`` checks).

    Raises
    ------
    ProjectError
        Naming the first entry whose replay differs from the log.
    """
    data = session.kernel.data
    creation = session.kernel.creation
    lc = LightCurve.from_arrays(
        data.time, data.value, data.error, domain=data.domain, meta=dict(data.meta)
    )
    create = {
        "settings": creation.settings,
        "grid": creation.grid,
        "backend": backend or _requested(session.backend),
        "t_ref": creation.t_ref,
        "labels": dict(data.labels) or None,
        "partitions": creation.partitions,
        "session_settings": creation.session_settings,
    }
    return _replay(
        lc, session.history, create, on_entry=_check_close if strict else None
    )


def _verify(session: PreWhitenSession, stored: SessionState) -> None:
    replayed = rerun(session, strict=True)
    problem = divergence(stored, replayed.state)
    if problem is not None:
        raise ProjectError(
            f"verify failed: replaying all {len(session.history)} log entries gives "
            f"{problem}; the stored head does not follow from its log"
        )


def verify(
    path: str | Path, *, data: Any | None = None, backend: str | None = None
) -> PreWhitenSession:
    """``load(path, verify=True)``: open a project after replaying its log."""
    return load(path, data=data, verify=True, backend=backend)


# --- the data file of an exported script --------------------------------------------

_RESERVED_COLUMNS = frozenset({"time", "value", "error"})


def write_session_csv(data: SessionData, path: str | Path) -> Path:
    """Write the session's data as ``time,value[,error]`` plus its label columns.

    Floats are written with ``repr`` (they read back bit for bit); the first line is a
    ``# domain=...`` comment that :func:`read_session_csv` honours.
    """
    target = Path(path)
    names = list(data.labels)
    clash = sorted(set(names) & _RESERVED_COLUMNS)
    if clash:
        raise SessionError(f"label columns {clash} clash with the data columns")
    header = ["time", "value"] + (["error"] if data.error is not None else []) + names
    buffer = io.StringIO()
    buffer.write(f"# cuperiod session data; domain={Domain(data.domain).value}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    columns: list[list[str]] = [
        [repr(float(x)) for x in np.asarray(data.time).tolist()],
        [repr(float(x)) for x in np.asarray(data.value).tolist()],
    ]
    if data.error is not None:
        columns.append([repr(float(x)) for x in np.asarray(data.error).tolist()])
    for name in names:
        columns.append([str(v) for v in np.asarray(data.labels[name]).tolist()])
    writer.writerows(zip(*columns, strict=True))
    _atomic_write(target, buffer.getvalue())
    return target


def read_session_csv(path: str | Path) -> tuple[LightCurve, dict[str, np.ndarray]]:
    """Read a data file written by :func:`write_session_csv`.

    Returns
    -------
    lc : LightCurve
        ``time``, ``value`` and (when present) ``error``, in the recorded domain.
    labels : dict
        Every further column, as an array of strings.

    Raises
    ------
    ValueError
        For a file that is not in that layout.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    domain = Domain.MAGNITUDE
    lines = text.splitlines(keepends=True)
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        match = re.search(r"domain\s*=\s*(\w+)", lines[start])
        if match:
            domain = Domain(match.group(1))
        start += 1
    reader = csv.reader(io.StringIO("".join(lines[start:])))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError(f"{source.name}: no header line") from None
    if header[:2] != ["time", "value"]:
        raise ValueError(
            f"{source.name}: expected the columns time,value[,error],..., got {header}"
        )
    has_error = len(header) > 2 and header[2] == "error"
    names = header[3:] if has_error else header[2:]
    if len(set(header)) != len(header):
        raise ValueError(f"{source.name}: duplicate column names")
    rows = [row for row in reader if row]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"{source.name}: row {number} has {len(row)} fields, not {len(header)}"
            )
    columns = list(zip(*rows, strict=True)) if rows else [()] * len(header)
    time = np.asarray([float(v) for v in columns[0]], dtype=np.float64)
    value = np.asarray([float(v) for v in columns[1]], dtype=np.float64)
    error = (
        np.asarray([float(v) for v in columns[2]], dtype=np.float64)
        if has_error
        else None
    )
    offset = 3 if has_error else 2
    labels = {
        name: np.asarray(columns[offset + i], dtype=object)
        for i, name in enumerate(names)
    }
    lc = LightCurve.from_arrays(
        time, value, error, domain=domain, meta={"source": str(source)}
    )
    return lc, labels


__all__ = [
    "DATA_SUFFIXES",
    "FORMAT",
    "SUFFIX",
    "VERSION",
    "ProjectModel",
    "build_state",
    "choose_backend",
    "created_with",
    "divergence",
    "document",
    "load",
    "log_entry",
    "read_document",
    "read_session_csv",
    "replay",
    "rerun",
    "save",
    "source_file",
    "state_dict",
    "validate_state",
    "verify",
    "write_session_csv",
]
