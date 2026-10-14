"""Integrity of the benchmark data downloads (benchmarks/_download.py and its users).

The benchmark suite is not packaged; these tests skip when ``benchmarks/`` is absent,
as in an sdist. Every transfer goes to a loopback HTTP server, never the network.
"""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import sys
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
SCRIPTS = {
    "ogle": BENCH / "download_ogle_triples.py",
    "v1309": BENCH / "download_v1309sco.py",
    "s82": BENCH / "dataset" / "download_s82_rrlyrae.py",
}

pytestmark = pytest.mark.skipif(
    not SCRIPTS["ogle"].is_file(),
    reason="benchmarks/ is not in this tree (an sdist does not ship it)",
)

GOOD = b"2455000.1 15.01 0.01\n" * 40
EVIL = GOOD.replace(b"15.01", b"16.01")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _Server:
    url: str
    bodies: dict[str, bytes] = field(default_factory=dict)
    default: bytes | None = None
    mode: str = "normal"  # "normal" | "truncate" | "stall"
    requested: list[str] = field(default_factory=list)
    release: threading.Event = field(default_factory=threading.Event)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    # A system proxy must not intercept the loopback requests.
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    state = _Server(url="")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            state.requested.append(self.path)
            if state.mode == "stall":
                state.release.wait(10)
                return
            body = state.bodies.get(self.path, state.default)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.mode == "truncate":
                body = body[: len(body) // 2]
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield state
    finally:
        state.release.set()
        httpd.shutdown()
        httpd.server_close()


def _load(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> types.ModuleType:
    """Import a benchmark file as a module, registered only for this test."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dl(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    return _load(monkeypatch, "_download", BENCH / "_download.py")


def _script(monkeypatch: pytest.MonkeyPatch, key: str, data: Path) -> types.ModuleType:
    """Load a download script with ``_common.DATA`` pointed at a scratch folder.

    The real ``_common`` creates folders inside the checkout when imported.
    """
    common = types.ModuleType("_common")
    common.DATA = data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_common", common)
    monkeypatch.setattr(sys, "path", list(sys.path))  # undo any insert
    _load(monkeypatch, "_download", BENCH / "_download.py")
    return _load(monkeypatch, f"_bench_{key}", SCRIPTS[key])


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.rglob("*") if p.is_file())


# --- the helper --------------------------------------------------------------------


def test_fetch_accepts_the_pinned_file(
    dl: types.ModuleType, server: _Server, tmp_path: Path
) -> None:
    server.bodies["/a.dat"] = GOOD
    target = tmp_path / "a.dat"
    assert dl.fetch(f"{server.url}/a.dat", target, _sha256(GOOD)) is True
    assert target.read_bytes() == GOOD
    assert _leftovers(tmp_path) == ["a.dat"]
    # A verified cached copy is kept without touching the network again.
    assert dl.fetch(f"{server.url}/a.dat", target, _sha256(GOOD).upper()) is False
    assert server.requested == ["/a.dat"]


@pytest.mark.parametrize("mode", ["altered", "truncate", "oversize"])
def test_fetch_rejects_a_bad_transfer(
    dl: types.ModuleType, server: _Server, tmp_path: Path, mode: str
) -> None:
    # An on-path attacker's substitute, or a dropped connection, used to be written
    # straight to the cache and then trusted forever.
    server.bodies["/a.dat"] = EVIL if mode == "altered" else GOOD
    server.mode = "truncate" if mode == "truncate" else "normal"
    kwargs = {"max_bytes": len(GOOD) - 1} if mode == "oversize" else {}
    with pytest.raises(dl.DownloadError):
        dl.fetch(f"{server.url}/a.dat", tmp_path / "a.dat", _sha256(GOOD), **kwargs)
    assert _leftovers(tmp_path) == []  # neither the target nor a .part file


def test_fetch_rejects_a_stale_cached_copy(
    dl: types.ModuleType, server: _Server, tmp_path: Path
) -> None:
    target = tmp_path / "a.dat"
    target.write_bytes(GOOD[: len(GOOD) // 2])  # e.g. an earlier truncated transfer
    with pytest.raises(dl.DownloadError, match="delete it"):
        dl.fetch(f"{server.url}/a.dat", target, _sha256(GOOD))
    assert server.requested == []
    # force= re-downloads it in place.
    server.bodies["/a.dat"] = GOOD
    assert dl.fetch(f"{server.url}/a.dat", target, _sha256(GOOD), force=True)
    assert target.read_bytes() == GOOD


def test_fetch_times_out_on_a_silent_server(
    dl: types.ModuleType, server: _Server, tmp_path: Path
) -> None:
    server.mode = "stall"
    with pytest.raises(OSError):  # TimeoutError, or URLError wrapping it
        dl.fetch(f"{server.url}/a.dat", tmp_path / "a.dat", _sha256(GOOD), timeout=0.3)
    assert _leftovers(tmp_path) == []


def test_fetch_validates_the_pinned_digest(
    dl: types.ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        dl.fetch("https://example.invalid/a.dat", tmp_path / "a.dat", "abc123")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://ftp.astrouw.edu.pl/ogle/a.dat",
        "http://ftp.astrouw.edu.pl/ogle/a.dat",
        "file:///etc/passwd",
        "ftp.astrouw.edu.pl/ogle/a.dat",
    ],
)
def test_fetch_refuses_anything_but_https(
    dl: types.ModuleType, tmp_path: Path, url: str
) -> None:
    # The digest would catch a substituted file, but a script that regressed to a
    # plaintext mirror must fail loudly, before any connection is made.
    with pytest.raises(ValueError, match="non-HTTPS"):
        dl.fetch(url, tmp_path / "a.dat", _sha256(GOOD))
    assert _leftovers(tmp_path) == []


# --- the scripts -------------------------------------------------------------------


def _literal(path: Path, name: str) -> Any:
    """The literal value a script assigns to a module-level ``name``."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                assert node.value is not None
                return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} assigns no {name}")


@pytest.mark.parametrize("path", [*SCRIPTS.values(), BENCH / "_download.py"])
def test_no_plaintext_or_unbounded_transfers(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "ftp://" not in node.value and "http://" not in node.value, (
                f"{path.name}: plaintext URL in {node.value[:60]!r}"
            )
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else ""
            if called == "urlopen":
                # Only the helper opens URLs, and never without a timeout.
                assert path.name == "_download.py", f"{path.name} calls urlopen"
                assert any(k.arg == "timeout" for k in node.keywords)


def test_every_fetched_file_has_a_pinned_digest() -> None:
    expected = {
        "ogle": {
            f"OGLE-BLG-ECL-{star}_{phase}.dat"
            for star in _literal(SCRIPTS["ogle"], "STARS")
            for phase in _literal(SCRIPTS["ogle"], "PHASES")
        },
        "v1309": set(_literal(SCRIPTS["v1309"], "FILES")),
        "s82": set(_literal(SCRIPTS["s82"], "FILES")),
    }
    for key, names in expected.items():
        pinned = _literal(SCRIPTS[key], "SHA256")
        assert set(pinned) == names, key
        for digest in pinned.values():
            assert len(digest) == 64 and int(digest, 16) >= 0
    # The S82 mirror is read at a fixed commit, not a moving branch.
    assert "/raw/main/" not in _literal(SCRIPTS["s82"], "BASE_URL")


@pytest.mark.parametrize("key", ["ogle", "v1309"])
def test_ogle_scripts_refuse_a_substituted_file(
    monkeypatch: pytest.MonkeyPatch, server: _Server, tmp_path: Path, key: str
) -> None:
    script = _script(monkeypatch, key, tmp_path)
    monkeypatch.setattr(script, "BASE", f"{server.url}/x/")
    server.default = EVIL
    with pytest.raises(Exception, match="SHA-256"):
        script.main()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("key", ["ogle", "v1309"])
def test_ogle_scripts_refuse_a_stale_cached_file(
    monkeypatch: pytest.MonkeyPatch, server: _Server, tmp_path: Path, key: str
) -> None:
    script = _script(monkeypatch, key, tmp_path)
    monkeypatch.setattr(script, "BASE", f"{server.url}/x/")
    first = next(iter(_literal(SCRIPTS[key], "SHA256")))
    folder = tmp_path / ("ogle_triples" if key == "ogle" else "v1309sco")
    folder.mkdir()
    (folder / first).write_bytes(EVIL)
    with pytest.raises(Exception, match="SHA-256"):
        script.main()
    assert server.requested == []


def test_ogle_script_fetches_the_archive_layout(
    monkeypatch: pytest.MonkeyPatch, server: _Server, tmp_path: Path
) -> None:
    # With the digests pointed at the served bytes, the happy path writes every
    # file under its own name, from the same archive paths as before.
    script = _script(monkeypatch, "ogle", tmp_path)
    monkeypatch.setattr(script, "BASE", f"{server.url}/x/")
    monkeypatch.setattr(script, "SHA256", dict.fromkeys(script.SHA256, _sha256(GOOD)))
    server.default = GOOD
    script.main()
    names = _leftovers(tmp_path)
    assert names == sorted(script.SHA256)
    assert "/x/phot_ogle4/I/OGLE-BLG-ECL-061372.dat" in server.requested
    assert len(server.requested) == len(names)
    script.main()  # second run: everything cached and verified, no transfer
    assert len(server.requested) == len(names)


def test_s82_download_refuses_a_substituted_archive(
    monkeypatch: pytest.MonkeyPatch, server: _Server, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    _load(monkeypatch, "_download", BENCH / "_download.py")
    script = _load(monkeypatch, "_bench_s82", SCRIPTS["s82"])
    monkeypatch.setattr(script, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(script, "BASE_URL", f"{server.url}/s82/")
    server.default = EVIL
    with pytest.raises(Exception, match="SHA-256"):
        script.download()
    assert _leftovers(tmp_path) == []
