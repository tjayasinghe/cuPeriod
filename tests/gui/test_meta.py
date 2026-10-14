"""Tests for method/backend introspection and app-info helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import cuperiod.core.backend as core_backend
from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.gui import appinfo
from cuperiod.gui.appinfo import (
    app_version,
    checkout_root,
    git_short_hash,
    read_git_revision,
    version_label,
)
from cuperiod.gui.meta import (
    backend_options,
    method_display_names,
    multiband_method_names,
    resolved_backend,
    resolved_prewhiten_backend,
)


def test_multiband_method_names() -> None:
    multiband = set(multiband_method_names())
    # The joint-fit methods plus the pooled fold statistics; TLS is single-band only.
    assert multiband == {
        "GLS", "BLS", "MHAOV", "PDM", "CE", "STRINGLENGTH", "SUPERSMOOTHER"
    }
    assert multiband.issubset(method_display_names())


def test_backend_options_always_include_auto_and_cpu() -> None:
    for name in method_display_names():
        options = backend_options(name)
        assert "auto" in options
        assert "cpu" in options
        assert len(options) == len(set(options))  # no duplicates


def test_version_label_format() -> None:
    label = version_label()
    assert label.startswith("v")
    assert app_version() in label


# -- GUI-A-18: the "runs on" hint for bare torch follows torch's real device --------


@pytest.mark.parametrize(
    ("devices", "on_gpu"), [({"cpu"}, False), ({"cpu", "cuda"}, True)]
)
def test_bare_torch_hint_follows_the_device_torch_will_use(
    monkeypatch: pytest.MonkeyPatch, devices: set[str], on_gpu: bool
) -> None:
    monkeypatch.setattr(core_backend, "torch_available", lambda: True)
    monkeypatch.setattr(core_backend, "torch_devices", lambda: set(devices))
    import importlib

    spectrum = importlib.import_module("cuperiod.prewhiten.spectrum")
    monkeypatch.setattr(spectrum, "torch_available", lambda: True)
    for name in method_display_names():
        hint = resolved_backend(name, "torch")
        assert hint is not None and hint[1] is on_gpu, name
    hint = resolved_prewhiten_backend("torch")
    assert hint is not None and hint[1] is on_gpu


def test_torch_hint_honours_the_device_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core_backend, "torch_devices", lambda: {"cpu", "cuda"})
    import importlib

    spectrum = importlib.import_module("cuperiod.prewhiten.spectrum")
    monkeypatch.setattr(spectrum, "torch_available", lambda: True)
    cpu = PreWhitenSettings(device="cpu")
    hint = resolved_prewhiten_backend("torch", cpu)
    assert hint is not None and hint[1] is False
    hint = resolved_prewhiten_backend("torch:cpu")
    assert hint is not None and hint[1] is False
    hint = resolved_backend("GLS", "torch", GLSSettings(device="cpu"))
    assert hint is not None and hint[1] is False


def test_auto_hint_defers_to_the_settings_backend() -> None:
    # cup.periodogram(backend="auto") runs on settings.backend when that pins one, so
    # the hint must name that backend rather than what "auto" alone would pick.
    assert resolved_backend("GLS", "auto", GLSSettings(backend="astropy")) == (
        "astropy",
        False,
    )
    explicit = resolved_backend("GLS", "cpu", GLSSettings(backend="astropy"))
    assert explicit is not None and explicit[0] != "astropy"  # explicit wins


# -- GUI-A-16 / SEC-01: the revision comes from cuPeriod's own checkout, no git -----


_SHA = "0d01c7e" + "a" * 33
_OTHER = "68c8b6c" + "b" * 33


def _checkout(root: Path) -> Path:
    """A fake source checkout; returns the git directory."""
    (root / "src" / "cuperiod" / "gui").mkdir(parents=True)
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    return git


def test_revision_from_a_loose_ref(tmp_path: Path) -> None:
    git = _checkout(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(_SHA + "\n")
    assert read_git_revision(tmp_path) == "0d01c7e"


def test_revision_from_packed_refs(tmp_path: Path) -> None:
    git = _checkout(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/release/v1.3.0\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{_OTHER} refs/heads/main\n"
        f"{_SHA} refs/heads/release/v1.3.0\n"
        f"^{_OTHER}\n"
    )
    assert read_git_revision(tmp_path) == "0d01c7e"


def test_revision_from_a_detached_head(tmp_path: Path) -> None:
    git = _checkout(tmp_path)
    (git / "HEAD").write_text(_SHA + "\n")
    assert read_git_revision(tmp_path) == "0d01c7e"


def test_revision_from_a_worktree_gitdir_file(tmp_path: Path) -> None:
    main = tmp_path / "main"
    git = _checkout(main)
    (git / "refs" / "heads" / "feature").write_text(_SHA + "\n")
    wt_git = git / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    (wt / "src" / "cuperiod").mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert read_git_revision(wt) == "0d01c7e"


@pytest.mark.parametrize(
    "head",
    ["ref: ../../../outside\n", "ref: refs/heads/../../x\n", "not-a-sha\n", ""],
)
def test_hostile_or_garbage_heads_give_no_revision(tmp_path: Path, head: str) -> None:
    git = _checkout(tmp_path)
    (git / "HEAD").write_text(head)
    assert read_git_revision(tmp_path) is None


def test_no_git_directory_gives_no_revision(tmp_path: Path) -> None:
    (tmp_path / "src" / "cuperiod").mkdir(parents=True)
    assert read_git_revision(tmp_path) is None


def test_an_enclosing_project_repository_is_not_cuperiods_revision(
    tmp_path: Path,
) -> None:
    # a wheel install inside someone else's repository: <proj>/.venv/Lib/site-packages
    project = tmp_path / "proj"
    git = project / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(_OTHER + "\n")
    module = project / ".venv" / "Lib" / "site-packages" / "cuperiod" / "gui"
    module.mkdir(parents=True)
    assert checkout_root(module / "appinfo.py") is None
    # the source layout is recognised
    root = tmp_path / "checkout"
    _checkout(root)
    found = checkout_root(root / "src" / "cuperiod" / "gui" / "appinfo.py")
    assert found is not None and found == root.resolve()


def test_git_short_hash_never_spawns_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*_: object, **__: object) -> None:
        raise AssertionError("git must not be executed")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    git_short_hash.cache_clear()
    try:
        revision = git_short_hash()
        assert revision is None or len(revision) == 7
        assert version_label().startswith(f"v{app_version()}")
    finally:
        git_short_hash.cache_clear()
    assert not hasattr(appinfo, "subprocess")
