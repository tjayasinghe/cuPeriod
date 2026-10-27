"""Session warnings recorded on the worker without touching global warning state
(GUI-5), and kept when opening the session fails (GUI-6).

``warnings.catch_warnings`` swaps the process-wide filters and output hook. On the
worker it raced with the GUI thread's own block (the file loader records the reader's
warnings that way): when the two exits interleaved, every later warning of the
process went to a list nobody read. The interleavings are forced with events.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import cuperiod as cup
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.session_bridge import (
    SessionBridge,
    SessionKey,
    _install_warning_capture,
    _making,
)
from cuperiod.prewhiten.session.state import SessionWarning

TEXT = "the project ran on 'cuda'; using 'cpu' here"


def _state() -> tuple[object, object, list[object]]:
    return (
        warnings.showwarning,
        warnings._showwarnmsg_impl,  # type: ignore[attr-defined]
        list(warnings.filters),
    )


def _run_interleaved(gui_exits_first: bool) -> tuple[tuple[str, ...], list[object]]:
    """Run :func:`_making` on a thread while this one is in ``catch_warnings``.

    ``gui_exits_first``: this thread enters, then the worker; this thread leaves
    first (the order of the review's reproducer). Otherwise the worker enters
    first and leaves first.
    """
    worker_in = threading.Event()
    gui_in = threading.Event()
    gui_out = threading.Event()
    result: dict[str, object] = {}

    def make() -> str:
        worker_in.set()
        if gui_exits_first:
            gui_out.wait(5.0)
        else:
            gui_in.wait(5.0)
        warnings.warn(TEXT, SessionWarning, stacklevel=2)
        return "session"

    def worker() -> None:
        result["out"] = _making(make)  # type: ignore[arg-type]

    thread = threading.Thread(target=worker)
    if not gui_exits_first:
        thread.start()
        worker_in.wait(5.0)
    with warnings.catch_warnings(record=True) as gui_log:
        warnings.simplefilter("always")
        gui_in.set()
        if gui_exits_first:
            thread.start()
            worker_in.wait(5.0)
        else:
            thread.join(5.0)  # the worker leaves while this block is still open
    gui_out.set()
    thread.join(5.0)
    assert not thread.is_alive()
    session, texts = result["out"]  # type: ignore[misc]
    assert session == "session"
    return texts, list(gui_log)


@pytest.fixture
def capture() -> Iterator[None]:
    """The hook and filter a bridge installs, with this test's state restored."""
    before = _state()
    with warnings.catch_warnings():
        _install_warning_capture()
        yield
    assert _state()[2] == before[2]


@pytest.mark.parametrize("gui_exits_first", [True, False])
@pytest.mark.usefixtures("capture")
def test_a_worker_records_session_warnings_without_breaking_the_gui_thread(
    gui_exits_first: bool,
) -> None:
    before = _state()
    texts, gui_log = _run_interleaved(gui_exits_first)
    assert texts == (TEXT,)
    # The session warning went to the worker's list, not to the GUI block's log
    # (where the file loader would have taken it for a note on the light curve).
    assert not [w for w in gui_log if issubclass(w.category, SessionWarning)]  # type: ignore[attr-defined]
    # The global state is the GUI thread's again: no dead list left installed as
    # the output of every later warning, the filters as they were.
    assert _state() == before


@pytest.mark.usefixtures("capture")
def test_other_warnings_from_the_worker_pass_through_as_raised() -> None:
    def make() -> str:
        warnings.warn(TEXT, SessionWarning, stacklevel=2)
        warnings.warn("about the code", FutureWarning, stacklevel=2)
        return "session"

    out: dict[str, object] = {}
    with pytest.warns(FutureWarning, match="about the code") as record:
        thread = threading.Thread(target=lambda: out.update(r=_making(make)))  # type: ignore[arg-type]
        thread.start()
        thread.join(5.0)
    assert out["r"] == ("session", (TEXT,))
    assert not [w for w in record if issubclass(w.category, SessionWarning)]


@pytest.mark.usefixtures("capture")
def test_the_same_session_warning_is_recorded_every_time() -> None:
    """Opening the same project twice warns twice (not once per code location)."""

    def make() -> str:
        warnings.warn(TEXT, SessionWarning, stacklevel=1)
        return "session"

    assert _making(make)[1] == (TEXT,)  # type: ignore[arg-type]
    assert _making(make)[1] == (TEXT,)  # type: ignore[arg-type]


@pytest.mark.usefixtures("capture")
def test_session_warnings_are_kept_when_making_the_session_fails() -> None:
    """GUI-6: a failed open dropped what it had warned about."""

    def make() -> str:
        warnings.warn(TEXT, SessionWarning, stacklevel=2)
        raise RuntimeError("the project file is damaged")

    with pytest.raises(RuntimeError, match="damaged") as info:
        _making(make)  # type: ignore[arg-type]
    assert getattr(info.value, "__notes__", None) == [f"Warning: {TEXT}"]


# -- through the bridge ----------------------------------------------------------------
def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 40.0, 300))
    y = 0.3 * np.sin(2 * np.pi * 1.3 * t) + rng.normal(0.0, 0.01, 300)
    return t, y, np.full(300, 0.01)


@pytest.fixture
def rig(qtbot: QtBot) -> Iterator[tuple[ComputeManager, SessionBridge]]:
    compute = ComputeManager()
    bridge = SessionBridge(compute, progress_interval=0.0)
    yield compute, bridge
    compute.wait_for_done(20_000)


def _idle(qtbot: QtBot, bridge: SessionBridge, compute: ComputeManager) -> None:
    qtbot.waitUntil(
        lambda: bridge.is_idle() and compute.session_jobs == 0, timeout=30_000
    )


def test_a_failed_open_says_what_it_warned_about(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    compute, bridge = rig
    key = SessionKey("star", "")
    failures: list[str] = []
    bridge.failed.connect(lambda _key, text: failures.append(text))

    def failing() -> cup.PreWhitenSession:
        warnings.warn(TEXT, SessionWarning, stacklevel=2)
        raise RuntimeError("the project file is damaged")

    bridge.display(key)
    bridge.open(key, failing, backend="numpy")
    _idle(qtbot, bridge, compute)
    assert len(failures) == 1
    assert failures[0].splitlines() == [
        "RuntimeError: the project file is damaged",
        f"Warning: {TEXT}",
    ]


def test_a_gui_thread_block_during_an_open_leaves_warnings_working(
    rig: tuple[ComputeManager, SessionBridge], qtbot: QtBot
) -> None:
    """The GUI thread loads a file (``catch_warnings`` in the loader) while the
    worker opens a session: afterwards warnings still reach the output."""
    compute, bridge = rig
    key = SessionKey("star", "")
    messages: list[tuple[str, str]] = []
    bridge.message.connect(lambda severity, text: messages.append((severity, text)))
    started = threading.Event()
    release = threading.Event()

    def slow() -> cup.PreWhitenSession:
        started.set()
        release.wait(10.0)
        warnings.warn(TEXT, SessionWarning, stacklevel=2)
        return cup.PreWhitenSession(_data(), backend="numpy")

    before = _state()
    bridge.display(key)
    with warnings.catch_warnings(record=True) as gui_log:
        warnings.simplefilter("always")
        bridge.open(key, slow, backend="numpy")
        assert started.wait(10.0)
    release.set()
    _idle(qtbot, bridge, compute)
    assert ("warning", "The project ran on 'cuda'; using 'cpu' here.") in messages
    assert not gui_log
    assert _state()[:2] == before[:2]
