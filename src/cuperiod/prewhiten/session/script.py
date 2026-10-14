"""Export a session as a runnable Python script that uses only the public API.

The script re-creates the session from its data and creation arguments — every
:class:`~cuperiod.PreWhitenSettings` and :class:`~cuperiod.SessionSettings` field
written out, the frequency grid, ``t_ref`` and root partitions pinned — checks the
data's content hash, then repeats the effective log one action at a time. Steps
carry their logged peak seeds (consecutive steps fold into one
``session.run(seeds=[...])``), so the extraction does not depend on the spectrum
backend the script happens to run on.

The script is generated from a project that may come from anyone, so nothing from
the session is ever pasted into code as is: every value is the ``repr()`` of a
validated value (numbers, ids, names, labels, paths, tie text), and free text — a
title, notes, report messages, stop reasons, metadata — only lands in comments and
the docstring, through :func:`sanitize_comment`.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from cuperiod.core.columns import Domain
from cuperiod.prewhiten.session.state import LogEntry, SessionError
from cuperiod.prewhiten.subsets import Partition

if TYPE_CHECKING:
    from cuperiod.prewhiten.session.session import PreWhitenSession

#: Longest free text a comment carries.
COMMENT_LENGTH: Final = 120

# Control characters (C0, DEL, C1) and the Unicode line/paragraph separators: none
# may reach a comment, where a line break would turn the rest of the text into code.
_UNSAFE = re.compile("[\x00-\x1f\x7f-\x9f  ]")

#: The relative tolerance the script's self-check compares frequencies with.
CHECK_RTOL: Final = 1e-9


def sanitize_comment(text: str) -> str:
    """Free text made safe for a ``#`` comment or the script's docstring.

    Control characters, U+2028 and U+2029 are removed (so the text cannot end its
    line), the length is capped at :data:`COMMENT_LENGTH`, and ``\\`` and ``"`` are
    escaped (so it cannot end a docstring either).
    """
    clean = _UNSAFE.sub("", str(text))[:COMMENT_LENGTH]
    return clean.replace("\\", "\\\\").replace('"', '\\"')


def literal(value: Any) -> str:
    """A Python literal for plain data (``repr`` of each validated scalar).

    Raises
    ------
    SessionError
        For a non-finite float or a type that has no safe literal.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SessionError(f"cannot write the non-finite number {value!r}")
        return repr(float(value))
    if isinstance(value, str):
        return repr(str(value))
    if isinstance(value, tuple):
        items = [literal(v) for v in value]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
    if isinstance(value, list):
        return f"[{', '.join(literal(v) for v in value)}]"
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise SessionError("mapping keys must be strings")
        return (
            "{" + ", ".join(f"{repr(k)}: {literal(v)}" for k, v in value.items()) + "}"
        )
    raise SessionError(f"cannot write a {type(value).__name__} into a script")


# --- one renderer per op --------------------------------------------------------------


def _seeded(name: str) -> Callable[[Mapping[str, Any]], str]:
    def render(a: Mapping[str, Any]) -> str:
        if a.get("seed") is None:
            return f"session.{name}()"
        return f"session.{name}(seed={literal(a['seed'])}, index={literal(a['index'])})"

    return render


def _plain(name: str) -> Callable[[Mapping[str, Any]], str]:
    return lambda a: f"session.{name}()"


def _use(name: str) -> Callable[[Mapping[str, Any]], str]:
    return lambda a: f"session.{name}({literal(a.get('name'))})"


def _runs(runs: Sequence[Sequence[int]] | None) -> Any:
    return None if runs is None else [tuple(int(v) for v in r) for r in runs]


def _partition_call(
    n: str,
    runs: Sequence[Sequence[int]],
    name: str,
    labels: Sequence[str],
    weights: Sequence[float],
    origin: Mapping[str, Any],
) -> str:
    return (
        f"cup.Partition.from_runs({n}, {literal(_runs(runs))}, name={literal(name)}, "
        f"labels={literal(list(labels))}, "
        f"weights={literal([float(w) for w in weights])}, "
        f"origin=parse_origin({literal(dict(origin))}))"
    )


#: The script statement of every op (arguments as the log stores them).
RENDERERS: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "step": _seeded("step"),
    "polish": _plain("polish"),
    "prune": _plain("prune"),
    "renumber": _plain("renumber"),
    "add_peak": _seeded("add_peak"),
    "add": lambda a: (
        f"session.add({literal(a['frequency'])}, snap=False, "
        f"refine={literal(a.get('refine'))}, "
        f"allow_close={literal(bool(a.get('allow_close', False)))})"
    ),
    "remove": lambda a: (
        f"session.remove(*{literal(list(a['ids']))}, "
        f"cascade={literal(bool(a.get('cascade', False)))})"
    ),
    "set_active": lambda a: (
        f"session.set_active({literal(list(a['ids']))}, {literal(bool(a['active']))})"
    ),
    "fix": lambda a: (
        f"session.fix({literal(list(a['ids']))}, "
        f"frequency={literal(a.get('frequency'))}, "
        f"amplitude={literal(a.get('amplitude'))}, phase={literal(a.get('phase'))})"
    ),
    "set_values": lambda a: (
        f"session.set_values({literal(a['id'])}, "
        f"frequency={literal(a.get('frequency'))}, "
        f"amplitude={literal(a.get('amplitude'))}, phase={literal(a.get('phase'))})"
    ),
    "tie": lambda a: (
        f"session.tie({literal(a['id'])}, {literal(a.get('expression'))}, "
        f"allow_close={literal(bool(a.get('allow_close', False)))})"
    ),
    "set_variation": lambda a: (
        f"session.set_variation({literal(list(a['ids']))}, {literal(a['mode'])}, "
        f"hold_frequency={literal(bool(a.get('hold_frequency', True)))})"
    ),
    "calculate": _plain("calculate"),
    "improve": lambda a: (
        f"session.improve({literal(None if a.get('ids') is None else list(a['ids']))})"
    ),
    "set_note": lambda a: f"session.set_note({literal(a['id'])}, {literal(a['note'])})",
    "delete_points": lambda a: f"session.delete_points({literal(_runs(a['runs']))})",
    "restore_points": lambda a: (
        f"session.restore_points({literal(_runs(a.get('runs')))})"
    ),
    "exclude_subsets": lambda a: (
        f"session.exclude_subsets({literal(a['name'])}, "
        f"{literal([int(s) for s in a['subsets']])}, "
        f"excluded={literal(bool(a.get('excluded', True)))})"
    ),
    "define_partition": lambda a: (
        "session.define_partition("
        + _partition_call(
            "session.data.n",
            a["runs"],
            a["name"],
            a["labels"],
            a["weights"],
            a.get("origin") or {"kind": "manual"},
        )
        + ")"
    ),
    "drop_partition": lambda a: f"session.drop_partition({literal(a['name'])})",
    "set_subset_weights": lambda a: (
        f"session.set_subset_weights({literal(a['name'])}, "
        f"{literal([float(w) for w in a['weights']])})"
    ),
    "use_offsets": _use("use_offsets"),
    "use_variation": _use("use_variation"),
    "use_weights": _use("use_weights"),
    "configure": lambda a: (
        f"session.configure(cup.PreWhitenSettings(**{literal(dict(a['settings']))}))"
    ),
}


def _note(entry: LogEntry) -> str:
    report = entry.report
    return f"{report.kind}: {report.message}" if report.message else report.kind


def render_entry(entry: LogEntry) -> str:
    """One logged action as one script statement, its outcome as a comment.

    Raises
    ------
    SessionError
        For an op without a renderer.
    """
    renderer = RENDERERS.get(entry.op)
    if renderer is None:
        raise SessionError(f"no script renderer for op {entry.op!r}")
    return f"{renderer(entry.args)}  # {sanitize_comment(_note(entry))}"


def _run_block(entries: Sequence[LogEntry]) -> list[str]:
    """Consecutive steps as one seeded ``session.run``."""
    lines = ["session.run(seeds=["]
    for entry in entries:
        seed = entry.args.get("seed")
        item = (
            "None"
            if seed is None
            else f"({literal(seed)}, {literal(entry.args.get('index'))})"
        )
        lines.append(f"    {item},  # {sanitize_comment(_note(entry))}")
    lines.append("])")
    return lines


def _body(log: Sequence[LogEntry]) -> list[str]:
    lines: list[str] = []
    steps: list[LogEntry] = []
    for entry in log:
        if entry.op == "step":
            steps.append(entry)
            continue
        if steps:
            lines.extend(_run_block(steps))
            steps = []
        lines.append(render_entry(entry))
    if steps:
        lines.extend(_run_block(steps))
    return lines


# --- the whole script ---------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    """Where the script reads its data from."""

    kind: Literal["file", "csv", "expr"]
    text: str
    relative: bool = False


def _settings_block(name: str, cls: str, dump: Mapping[str, Any]) -> list[str]:
    lines = [f"{name} = cup.{cls}("]
    lines.extend(f"    {field}={literal(value)}," for field, value in dump.items())
    lines.append(")")
    return lines


def _labels_partition(partition: Partition, column: Any) -> bool:
    """Whether ``partitions=[name]`` over the string label column rebuilds it."""
    if partition.origin.kind != "labels":
        return False
    try:
        rebuilt = Partition.from_labels(
            np.asarray([str(v) for v in np.asarray(column).tolist()], dtype=object),
            name=partition.name,
        )
    except ValueError:
        return False
    return (
        rebuilt.labels == partition.labels
        and rebuilt.weights == partition.weights
        and np.array_equal(rebuilt.codes, partition.codes)
        and rebuilt.origin.model_dump() == partition.origin.model_dump()
    )


def _partition_expr(partition: Partition, n: int) -> str:
    return _partition_call(
        literal(n),
        partition.runs(),
        partition.name,
        partition.labels,
        partition.weights,
        partition.origin.model_dump(mode="json"),
    )


def _roots(session: PreWhitenSession, source: _Source) -> tuple[list[str], bool]:
    """The ``partitions=`` items and whether ``parse_origin`` is needed."""
    data = session.kernel.data
    items: list[str] = []
    explicit = False
    for partition in session.kernel.creation.partitions:
        column = data.labels.get(partition.name)
        if (
            source.kind == "csv"
            and column is not None
            and _labels_partition(partition, column)
        ):
            items.append(literal(partition.name))
        else:
            items.append(_partition_expr(partition, data.n))
            explicit = True
    return items, explicit


def _render(session: PreWhitenSession, source: _Source, title: str | None) -> str:
    from cuperiod import __version__

    kernel = session.kernel
    data = kernel.data
    creation = kernel.creation
    log = session.history
    head = session.state
    roots, explicit = _roots(session, source)
    needs_origin = explicit or any(e.op == "define_partition" for e in log)
    domain = str(Domain(data.domain).value)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    name = f" '{sanitize_comment(title)}'" if title else ""
    origin = data.meta.get("source")
    origin_text = (
        f" from {sanitize_comment(str(origin))}" if isinstance(origin, str) else ""
    )

    lines = [
        f'"""cuPeriod pre-whitening session{name} - exported {stamp} by cuperiod '
        f"{sanitize_comment(str(__version__))}.",
        "",
        f"Repeats the session's {len(log)} logged action(s) with the public API. Steps",
        "use their logged peak seeds, so the extraction is the same on any spectrum",
        f"backend. Data: {data.n} points ({domain}){origin_text}.",
        '"""',
        "",
        "import numpy as np",
        "",
        "import cuperiod as cup",
    ]
    if needs_origin:
        lines.append("from cuperiod.prewhiten.subsets import parse_origin")
    lines.append("")
    if source.kind == "expr":
        lines.append(f"DATA = {source.text}  # supplied when the script was written")
    elif source.relative:
        lines.append("# Data paths are relative to this script's folder.")
        lines.append(r'HERE = __file__.replace("\\", "/").rpartition("/")[0] or "."')
        lines.append(f"DATA = HERE + {literal('/' + source.text)}  # {data.n} points")
    else:
        lines.append(f"DATA = {literal(source.text)}  # {data.n} points")
    for key, value in list(data.meta.items())[:8]:
        if key != "source":
            lines.append(
                f"# meta {sanitize_comment(str(key))}: {sanitize_comment(str(value))}"
            )
    lines.append(
        f"DATA_SHA256 = {literal(data.content_sha256)}  "
        "# of the arrays the session used"
    )
    lines.append(
        f"BACKEND = None  # the session ran on {sanitize_comment(kernel.backend)}; "
        "None = what SETTINGS.backend asks for"
    )
    lines.extend(
        _settings_block(
            "SETTINGS", "PreWhitenSettings", creation.settings.model_dump(mode="json")
        )
    )
    lines.extend(
        _settings_block(
            "SESSION_SETTINGS",
            "SessionSettings",
            creation.session_settings.model_dump(mode="json"),
        )
    )
    lines.append("")
    f0, df, nf = creation.grid
    kwargs = [
        "settings=SETTINGS",
        "session_settings=SESSION_SETTINGS",
        "backend=BACKEND",
        f"grid=({literal(float(f0))}, {literal(float(df))}, {literal(int(nf))})",
        f"t_ref={literal(float(creation.t_ref))}",
    ]
    if roots:
        kwargs.append("partitions=[" + ", ".join(roots) + "]")
    if source.kind == "csv":
        lines.append("session = cup.PreWhitenSession.from_csv(")
        lines.append("    DATA,")
    elif source.kind == "file":
        columns = getattr(session, "_columns", None)
        columns_text = (
            "None"
            if columns is None
            else (
                f"cup.ColumnMap(time={literal(columns.time)}, "
                f"value={literal(columns.value)}, error={literal(columns.error)}, "
                f"band={literal(columns.band)})"
            )
        )
        lines.append(
            f"lc = cup.LightCurve.from_file(DATA, columns={columns_text}, "
            f"domain={literal(domain)})"
        )
        lines.append("session = cup.PreWhitenSession(")
        lines.append("    lc,")
    else:
        lines.append("session = cup.PreWhitenSession(")
        lines.append("    DATA,")
    lines.extend(f"    {kw}," for kw in kwargs)
    lines.append(")")
    lines.append("if session.data.content_sha256 != DATA_SHA256:")
    lines.append(
        "    raise SystemExit(\"the data differ from the exported session's; "
        'nothing was run")'
    )
    lines.append("")
    lines.extend(_body(log))
    if head.stop_reason:
        lines.append(f"# stopped: {sanitize_comment(head.stop_reason)}")
    expected = [float(c.frequency) for c in head.active]
    lines.extend(
        [
            "",
            "solution = session.result()",
            "print(solution.summary())",
            "",
            f"EXPECTED_FREQUENCY = np.array({literal(expected)}, dtype=float)",
            "FOUND = np.array([c.frequency for c in session.components if c.active], "
            "dtype=float)",
            "if FOUND.shape == EXPECTED_FREQUENCY.shape and np.allclose(",
            f"    FOUND, EXPECTED_FREQUENCY, rtol={literal(CHECK_RTOL)}, atol=0.0",
            "):",
            '    print("Reproduced the exported session.")',
            "else:",
            '    print("WARNING: this run does not reproduce the exported session.")',
            "",
        ]
    )
    text = "\n".join(lines)
    try:
        ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - a renderer bug, never input
        raise SessionError(f"the generated script does not parse: {exc}") from exc
    return text


def _check_expression(text: str) -> str:
    if _UNSAFE.search(text):
        raise SessionError("data_expr must be a single-line Python expression")
    try:
        ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise SessionError(f"data_expr is not a Python expression: {exc}") from exc
    return text


def render(
    session: PreWhitenSession,
    *,
    data_expr: str | None = None,
    title: str | None = None,
) -> str:
    """The session as a Python script.

    Parameters
    ----------
    session : PreWhitenSession
    data_expr : str, optional
        A Python expression for the data, inserted as code (it is the caller's code,
        never the session's). Without it the script re-reads the data's source file,
        which must still hold exactly these data (and the session must have no label
        columns).
    title : str, optional
        Named in the docstring (sanitised).

    Raises
    ------
    SessionError
        When no data source is usable and no ``data_expr`` is given.
    """
    if data_expr is not None:
        source = _Source("expr", _check_expression(data_expr))
    else:
        from cuperiod.prewhiten.session.project import source_file

        path, why = source_file(session)
        if path is None:
            raise SessionError(
                f"to_script() needs data_expr= here ({why}); export_script() writes "
                "the data next to the script instead"
            )
        source = _Source("file", path.as_posix())
    return _render(session, source, title)


def export(
    session: PreWhitenSession, path: str | Path, *, title: str | None = None
) -> Path:
    """Write the session's script to ``path`` (and its data file when needed).

    The data's source file is used when it still holds exactly these data and the
    session has no label columns; otherwise ``<stem>.data.csv`` is written beside the
    script (``time,value[,error]`` plus every label column, floats exact) and the
    script reads it with :meth:`PreWhitenSession.from_csv`. Data centred by
    :meth:`PreWhitenSession.from_bands` are written centred and read as they are.

    Returns
    -------
    Path
        The script.
    """
    from cuperiod.prewhiten.session.project import (
        _atomic_write,
        source_file,
        write_session_csv,
    )

    target = Path(path)
    directory = target.parent.resolve()
    file, _ = source_file(session)
    if file is not None:
        try:
            source = _Source("file", file.relative_to(directory).as_posix(), True)
        except ValueError:
            source = _Source("file", file.as_posix())
    else:
        stem = target.name[:-3] if target.name.lower().endswith(".py") else target.name
        data_file = target.with_name(f"{stem}.data.csv")
        write_session_csv(session.kernel.data, data_file)
        source = _Source("csv", data_file.name, True)
    _atomic_write(target, _render(session, source, title))
    return target


__all__ = [
    "CHECK_RTOL",
    "COMMENT_LENGTH",
    "RENDERERS",
    "export",
    "literal",
    "render",
    "render_entry",
    "sanitize_comment",
]
