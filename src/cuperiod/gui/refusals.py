"""Refused session edits: the library's remedy, in the window's terms.

A session refuses an edit with a message that names the remedy in the library's own
keyword arguments ("pass allow_close=True to add it anyway"), and with the same
remedy as a code, ``StepReport.hint``: ``"allow_close"``, ``"cascade"``, ``"exact"``
or ``"fit_mean"``. A GUI user cannot pass a keyword argument, so the window words its
own advice from the code — or, for a report without one, rewrites the known messages
(:func:`gui_wording`) — and, for the remedies it can carry out itself, offers to:
dissolve the ties and remove a tie parent, add or tie a frequency anyway, add one
exactly where it is.

Nothing here touches Qt or a session: :class:`Refusal` is what the session bridge
reports, and the functions below turn it into text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

#: The remedy codes of ``StepReport.hint`` this module knows.
KNOWN_HINTS: Final = frozenset({"allow_close", "cascade", "exact", "fit_mean"})

#: The arguments a remedy sets when the user accepts it, by hint. ``fit_mean`` has
#: none: it needs the Options form's *floating mean*, which the user ticks.
HINT_ARGUMENTS: Final[Mapping[str, Mapping[str, Any]]] = {
    "allow_close": {"allow_close": True},
    "cascade": {"cascade": True},
    "exact": {"refine": "none"},
}

#: What to do instead, by hint (after the refusal's cause and a colon).
_HINT_ADVICE: Final[Mapping[str, str]] = {
    "allow_close": "too close to tell apart at this baseline",
    "cascade": "untie it first (select it, T, clear the tie) or delete both",
    "exact": (
        "add it exactly instead: Ctrl+Shift+click, or Add exactly here in the "
        "spectrum's right-click menu"
    ),
    "fit_mean": (
        "zero points per subset need a floating mean: tick “floating mean” in "
        "the Options"
    ),
}

#: The library's keyword advice: from the clause that names one of the remedies'
#: arguments to the end of the message.
_KEYWORD_ADVICE = re.compile(
    r"(?:^|[;:,]\s*)[^;:]*\b(?:allow_close|cascade|refine|fit_mean)\s*=.*$",
    re.DOTALL,
)

#: Library refusals name keyword arguments a GUI user cannot pass; say what to do
#: here instead (the scripting API still has them). For reports without a hint.
_GUI_WORDING: tuple[tuple[str, str], ...] = (
    (
        r"add it with refine='none' or allow_close=True",
        "add it exactly instead: Ctrl+Shift+click, or Add exactly here in the "
        "spectrum's right-click menu",
    ),
    (
        r"add it with refine='none'",
        "add it exactly: Ctrl+Shift+click, or Add exactly here in the spectrum's "
        "right-click menu",
    ),
    (
        r"pass allow_close=True to (add|tie) it anyway",
        r"too close to tell apart at this baseline (a script can still \1 it "
        "with allow_close=True)",
    ),
    (
        r"remove with cascade=True to dissolve the tie",
        "untie it first (select it, T, clear the tie) or delete both",
    ),
    (
        r"zero points per subset need fit_mean=True \(configure it first\)",
        "zero points per subset need a floating mean: tick “floating mean” in "
        "the Options",
    ),
)


def gui_wording(text: str) -> str:
    """``text`` with library advice ("pass allow_close=True") put in GUI terms."""
    for pattern, replacement in _GUI_WORDING:
        text = re.sub(pattern, replacement, text)
    return text


def report_hint(report: object) -> str | None:
    """The remedy code of a refused ``report``, or ``None``.

    Read with ``getattr``: a library from before ``StepReport.hint`` has no such
    field, and a code this version does not know is treated as none.
    """
    hint = getattr(report, "hint", None)
    return hint if isinstance(hint, str) and hint in KNOWN_HINTS else None


@dataclass(frozen=True, eq=False)
class Refusal:
    """An edit a session refused, with what can be done about it.

    Attributes
    ----------
    key : object
        The :class:`~cuperiod.gui.viewmodels.SessionKey` of the session.
    text : str
        The refusal's message, as the library wrote it.
    hint : str or None
        Its remedy code (see :func:`report_hint`).
    op, args : str, mapping
        The refused operation and its arguments as they were submitted (``""`` and
        empty for an action that was not an operation, a Step say).
    retry : callable or None
        Queues the refused operation(s) again with some arguments added (the
        remedy's, see :data:`HINT_ARGUMENTS`); ``None`` when it cannot be repeated
        or the session is not on show any more.
    """

    key: object
    text: str
    hint: str | None = None
    op: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    retry: Callable[[Mapping[str, Any]], None] | None = None


def refusal_text(refusal: Refusal) -> str:
    """The refusal for the status bar, its remedy in the window's terms.

    With a hint, the library's keyword advice is cut and the window's advice for
    that hint follows the cause; without one, the known messages are rewritten
    (:func:`gui_wording`).
    """
    advice = _HINT_ADVICE.get(refusal.hint or "")
    if advice is None:
        return gui_wording(refusal.text)
    cause = _cause(refusal.text)
    return f"{cause}: {advice}" if cause else advice


def _cause(text: str) -> str:
    """``text`` without the library's keyword advice (and its trailing punctuation)."""
    return _KEYWORD_ADVICE.sub("", text.strip()).strip().rstrip(";:,. ")


@dataclass(frozen=True)
class Offer:
    """A remedy the window can carry out, as a question for the user.

    ``text`` asks it ("F1 has tied children (F5, F7). Dissolve their ties and
    remove F1?"); ``accept`` labels the button that does it.
    """

    title: str
    text: str
    accept: str


def refusal_offer(
    refusal: Refusal, rows: Sequence[Any] = ()
) -> Offer | None:
    """What to offer for ``refusal``, or ``None`` when there is nothing to offer.

    ``rows`` are the session's components (``ComponentView``: ``id``, ``label``,
    ``tie``), to name a tie parent's children.
    """
    if refusal.retry is None or refusal.hint not in HINT_ARGUMENTS:
        return None
    if refusal.hint == "cascade":
        return _cascade_offer(refusal, rows)
    if refusal.hint == "allow_close":
        near = re.search(r"\bof (F\d+)\b", refusal.text)
        other = near.group(1) if near else "another component"
        if refusal.op == "tie":
            what = refusal.args.get("id")
            name = f"F{what}" if isinstance(what, int) else "This component"
            return Offer(
                "Tie anyway?",
                f"This tie puts {name} within the collision guard of {other}: the "
                "two may be too close to tell apart at this baseline. Tie it "
                "anyway?",
                "Tie anyway",
            )
        return Offer(
            "Add anyway?",
            f"This frequency is within the collision guard of {other}: the two may "
            "be too close to tell apart at this baseline. Add it anyway?",
            "Add anyway",
        )
    # "exact": the frequency the refusal names (after any snap), else "here".
    found = re.search(r"(\d[\d.]*(?:e[-+]?\d+)?) c/d", refusal.text)
    where = f"at {found.group(1)} c/d" if found else "here"
    cause = _cause(refusal.text)
    lead = f"{cause[:1].upper()}{cause[1:]}. " if cause else ""
    return Offer(
        "Add exactly?",
        f"{lead}Add it exactly {where} (no refinement)?",
        "Add exactly",
    )


def _cascade_offer(refusal: Refusal, rows: Sequence[Any]) -> Offer:
    ids = _ids(refusal.args.get("ids"))
    doomed = set(ids)
    labels = [f"F{i}" for i in ids] or ["it"]
    children = [
        str(getattr(row, "label", f"F{getattr(row, 'id', '?')}"))
        for row in rows
        if getattr(row, "id", None) not in doomed
        and doomed & _tie_parents(getattr(row, "tie", None))
    ]
    parents = _join(labels)
    if not children:
        return Offer(
            "Remove and untie?",
            f"Other components are tied to {parents}. Dissolve their ties and "
            f"remove {parents}?",
            "Dissolve ties and remove",
        )
    has = "has" if len(labels) == 1 else "have"
    if len(children) == 1:
        text = (
            f"{parents} {has} a tied child ({children[0]}). Dissolve its tie and "
            f"remove {parents}?"
        )
    else:
        text = (
            f"{parents} {has} tied children ({', '.join(children)}). Dissolve "
            f"their ties and remove {parents}?"
        )
    return Offer("Remove and untie?", text, "Dissolve ties and remove")


def _ids(value: object) -> list[int]:
    """Component ids from an op's ``ids`` argument (ints, ``"F3"`` strings)."""
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return []
    out: list[int] = []
    for item in value:
        text = str(item).strip().lstrip("Ff")
        if text.isdigit():
            out.append(int(text))
    return out


def _tie_parents(tie: object) -> set[int]:
    """The component ids a tie expression ("2F1 - F3") names."""
    if not isinstance(tie, str):
        return set()
    return {int(m) for m in re.findall(r"[Ff](\d+)", tie)}


def _join(labels: Sequence[str]) -> str:
    """``F1``, ``F1 and F3``, ``F1, F3 and F4``."""
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


__all__ = [
    "HINT_ARGUMENTS",
    "KNOWN_HINTS",
    "Offer",
    "Refusal",
    "gui_wording",
    "refusal_offer",
    "refusal_text",
    "report_hint",
]
