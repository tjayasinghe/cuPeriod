"""Supply-chain guards: the GitHub workflows and the declared dependency floors.

These read repository files rather than the installed package. The workflow checks
skip when ``.github/`` is absent, as in an sdist, which does not ship it.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
GITHUB = ROOT / ".github"
WORKFLOWS = ("ci.yml", "release.yml")

# A `uses:` line: the reference, then an optional trailing comment.
_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>[^\s#]+)\s*(?:#\s*(?P<note>.*))?$")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _workflow_path(name: str) -> Path:
    path = GITHUB / "workflows" / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this tree (an sdist does not ship .github/)")
    return path


def _workflow(name: str) -> dict[str, Any]:
    loaded = yaml.safe_load(_workflow_path(name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _steps(workflow: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            yield job_name, step


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_action_is_pinned_to_a_commit(name: str) -> None:
    # A tag or a branch (`@v5`, `@release/v1`) can be moved to other code after
    # review, as in the 2025 tj-actions incident; a full commit SHA cannot. The
    # trailing comment names the release so Dependabot and readers can follow it.
    refs = []
    for line in _workflow_path(name).read_text(encoding="utf-8").splitlines():
        match = _USES.match(line)
        if match is None:
            continue
        ref = match["ref"]
        refs.append(ref)
        if ref.startswith(("./", "docker://")):
            continue
        _, _, pin = ref.partition("@")
        assert _COMMIT_SHA.fullmatch(pin), f"{name}: {ref} is not pinned to a commit"
        assert match["note"] and match["note"].startswith("v"), (
            f"{name}: {ref} has no '# vX.Y.Z' comment naming the pinned release"
        )
    assert refs, f"{name}: no `uses:` lines found"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_token_is_read_only_by_default(name: str) -> None:
    workflow = _workflow(name)
    assert workflow.get("permissions") == {"contents": "read"}, (
        f"{name}: needs a top-level `permissions: contents: read`"
    )
    for job_name, job in workflow["jobs"].items():
        granted = job.get("permissions", {})
        writes = {scope for scope, level in granted.items() if level == "write"}
        # Trusted Publishing needs an OIDC token and nothing else.
        assert writes <= {"id-token"}, f"{name}:{job_name} asks for {writes}"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_checkout_does_not_keep_the_token(name: str) -> None:
    checkouts = [
        (job, step)
        for job, step in _steps(_workflow(name))
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts
    for job, step in checkouts:
        assert step.get("with", {}).get("persist-credentials") is False, (
            f"{name}:{job} leaves the token in .git/config"
        )


def test_release_build_pins_its_tools() -> None:
    build = _workflow("release.yml")["jobs"]["build"]
    runs = [step["run"] for step in build["steps"] if "run" in step]
    setup_uv = [s for s in build["steps"] if "setup-uv@" in str(s.get("uses", ""))]
    assert setup_uv
    uv = setup_uv[0]["with"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", uv["version"])
    # setup-uv only verifies the uv tarball against `checksum:` or its built-in
    # table, and that table has no entry for a uv newer than the action itself, so
    # without this a swapped uv (the frontend that enforces the hashes below) would
    # install unverified. setup-uv compares lowercase hex exactly.
    assert re.fullmatch(r"[0-9a-f]{64}", str(uv.get("checksum"))), uv
    # The two are bumped together: the checksum line names its uv version.
    lines = _workflow_path("release.yml").read_text(encoding="utf-8").splitlines()
    checksum_line = next(line for line in lines if uv["checksum"] in line)
    assert re.search(rf"#.*\b{re.escape(uv['version'])}\b", checksum_line), (
        f"the checksum comment must name uv {uv['version']}: {checksum_line.strip()}"
    )
    # A release must not restore a cache another workflow run could have written.
    assert uv["enable-cache"] is False

    build_cmd = next(r for r in runs if r.strip().startswith("uv build"))
    assert "--require-hashes" in build_cmd
    constraints = re.search(r"--build-constraints\s+(\S+)", build_cmd)
    assert constraints is not None
    pinned = _hashed_requirements(ROOT / constraints[1])
    assert "hatchling" in pinned

    twine = next(r for r in runs if "twine" in r)
    assert re.search(r"twine==\d+\.\d+", twine), twine
    assert "--exclude-newer" in twine


def _hashed_requirements(path: Path) -> set[str]:
    """Names of a hash-pinned requirements file; asserts each is ``==`` + hash."""
    text = path.read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9])", re.sub(r"(?m)^\s*#.*\n?", "", text))
    names = set()
    for entry in filter(str.strip, entries):
        requirement = Requirement(entry.split("\\")[0].split("--hash")[0].strip())
        assert str(requirement.specifier).startswith("=="), entry
        assert "--hash=sha256:" in entry, entry
        names.add(requirement.name)
    return names


def _posix_bash() -> str | None:
    """A POSIX bash to run a workflow `run:` block, or None.

    On Windows ``bash`` on PATH may be the WSL launcher (System32 or WindowsApps),
    which runs in another filesystem, so only a bash outside those is accepted,
    falling back to the one installed next to Git for Windows' ``git``.
    """
    found = shutil.which("bash")
    if sys.platform != "win32":
        return found
    system = os.environ.get("SYSTEMROOT", r"C:\Windows").lower()
    candidates = [Path(found)] if found else []
    git = shutil.which("git")
    if git is not None:
        candidates += [p / "bin" / "bash.exe" for p in Path(git).resolve().parents]
    for candidate in candidates:
        text = str(candidate).lower()
        if (
            candidate.is_file()
            and not text.startswith(system)
            and "windowsapps" not in text
        ):
            return str(candidate)
    return None


def _version_guard() -> dict[str, Any]:
    steps: list[dict[str, Any]] = _workflow("release.yml")["jobs"]["build"]["steps"]
    guard = [step for step in steps if step.get("id") == "version-guard"]
    assert len(guard) == 1, "release.yml build job needs a `version-guard` step"
    return guard[0]


def test_release_publishes_to_pypi_only_from_a_tag_push() -> None:
    jobs = _workflow("release.yml")["jobs"]
    condition = jobs["pypi"]["if"]
    # A manual run started on a tag is a TestPyPI dry run, never a PyPI release.
    assert "github.event_name == 'push'" in condition
    assert "refs/tags/v" in condition
    assert jobs["pypi"]["needs"] == "build"
    guard = _version_guard()
    assert guard["if"] == "github.event_name == 'push'"
    # The tag reaches the script through the environment, never as `${{ }}` text
    # pasted into it (a crafted tag name would otherwise run as shell code).
    assert "${{" not in guard["run"]
    assert guard["env"]["TAG"] == "${{ github.ref_name }}"


@pytest.mark.parametrize(
    ("tag", "built", "ok"),
    [
        ("v1.3.0", "1.3.0", True),
        ("v1.3.0rc1", "1.3.0rc1", True),
        ("v1.3.0", "1.3.0.dev0", False),  # pyproject not bumped before tagging
        ("v1.3.1", "1.3.0", False),
        ("v1.3", "1.3.0", False),
    ],
)
def test_release_version_guard(tmp_path: Path, tag: str, built: str, ok: bool) -> None:
    guard = _version_guard()
    bash = _posix_bash()
    if bash is None:
        pytest.skip("no POSIX bash to run the workflow step")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / f"cuperiod-{built}.tar.gz").write_bytes(b"")
    (dist / f"cuperiod-{built}-py3-none-any.whl").write_bytes(b"")
    script = tmp_path / "guard.sh"
    script.write_bytes(guard["run"].encode())
    proc = subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", script.name],
        cwd=tmp_path,
        env={**os.environ, "TAG": tag},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (proc.returncode == 0) is ok, proc.stdout + proc.stderr


def _all_requirements() -> list[Requirement]:
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        lines = list(project["dependencies"])
        for extra in project.get("optional-dependencies", {}).values():
            lines.extend(extra)
    else:  # an installed copy without its source tree
        lines = importlib.metadata.requires("cuperiod") or []
    return [Requirement(line) for line in lines]


def test_pyarrow_floor_excludes_cve_2023_47248() -> None:
    # PyArrow 0.14.0-14.0.0 unpickles PyExtensionType metadata while reading
    # Parquet/IPC, so opening a crafted .parquet light curve runs code; 14.0.1
    # fixed it. cuPeriod reads user Parquet in the CLI, the batch runner and the GUI.
    pyarrow = [req for req in _all_requirements() if req.name == "pyarrow"]
    assert any(req.marker is None for req in pyarrow), "pyarrow is a core dependency"
    for req in pyarrow:
        for vulnerable in ("0.14.0", "13.0.0", "14.0.0"):
            assert not req.specifier.contains(vulnerable), f"{req} admits {vulnerable}"
