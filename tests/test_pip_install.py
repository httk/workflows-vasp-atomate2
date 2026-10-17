"""The in-tree build backend (``tools/workflow_repository_build.py``) and a real pip build.

Stdlib and pytest only, and repository-agnostic: the same file is used in every
httk workflow repository. Each test builds in a throwaway git repository holding
this working tree's ``pyproject.toml`` and backend (so neither the developer's
uncommitted state nor a shallow CI checkout matters), with two commits and the
tags ``wf-v1.0.0`` (annotated, first commit), ``wf-decl-v0.1.0`` (first commit)
and ``wf-v9.9.9`` (lightweight, HEAD). Hooks run in a subprocess with the
repository as cwd, as PEP 517 prescribes.
"""

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
NAME = PYPROJECT["project"]["name"]
NORM = re.sub(r"[-_.]+", "_", NAME).lower()
RECORD_FILE = "httk_workflow_repository.json"
BUNDLE_FILE = "httk_workflow_repository.bundle"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


def _git(cwd: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null", GIT_TERMINAL_PROMPT="0")
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args]
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / NAME
    (repo / "tools").mkdir(parents=True)
    _git(repo, "init", "-q")
    shutil.copy(REPO_ROOT / "pyproject.toml", repo / "pyproject.toml")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-q", "-m", "first")
    _git(repo, "tag", "-a", "-m", "release", "wf-v1.0.0")
    _git(repo, "tag", "wf-decl-v0.1.0")
    shutil.copy(REPO_ROOT / "tools" / "workflow_repository_build.py", repo / "tools")
    _git(repo, "add", "tools")
    _git(repo, "commit", "-q", "-m", "second")
    _git(repo, "tag", "wf-v9.9.9")
    return repo


def _hook(source: Path, hook: str, *args: str) -> subprocess.CompletedProcess[str]:
    code = "import sys; sys.path.insert(0, 'tools'); import workflow_repository_build as b; "
    code += "print(getattr(b, sys.argv[1])(*sys.argv[2:]))"
    return subprocess.run(
        [sys.executable, "-c", code, hook, *args], cwd=source, capture_output=True, text=True, check=False
    )


def _build(source: Path, hook: str, out: Path) -> zipfile.ZipFile:
    proc = _hook(source, hook, str(out))
    assert proc.returncode == 0, proc.stderr
    return zipfile.ZipFile(out / proc.stdout.strip())


def _check_record_hashes(wheel: zipfile.ZipFile, dist_info: str) -> None:
    rows = list(csv.reader(io.StringIO(wheel.read(f"{dist_info}/RECORD").decode())))
    assert sorted(row[0] for row in rows) == sorted(wheel.namelist())
    for path, digest, size in rows:
        if path == f"{dist_info}/RECORD":
            assert (digest, size) == ("", "")
            continue
        data = wheel.read(path)
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        assert (digest, size) == (f"sha256={expected}", str(len(data)))


def test_build_wheel(clone: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    wheel = _build(clone, "build_wheel", out)
    dist_info = f"{NORM}-9.9.9.dist-info"
    assert wheel.filename == str(out / f"{NORM}-9.9.9-py3-none-any.whl")
    assert all(name.startswith(f"{dist_info}/") for name in wheel.namelist())
    _check_record_hashes(wheel, dist_info)

    metadata = wheel.read(f"{dist_info}/METADATA").decode().splitlines()
    assert f"Name: {NAME}" in metadata and "Version: 9.9.9" in metadata
    requires = [line.removeprefix("Requires-Dist: ") for line in metadata if line.startswith("Requires-Dist: ")]
    base = [req for req in requires if "extra ==" not in req]
    assert base == PYPROJECT["project"]["dependencies"]

    record = json.loads(wheel.read(f"{dist_info}/{RECORD_FILE}"))
    head = _git(clone, "rev-parse", "HEAD")
    first = _git(clone, "rev-parse", "HEAD~1")
    assert record == {
        "format": "httk-workflow-repository",
        "format_version": 1,
        "repository": PYPROJECT["tool"]["httk"]["workflow-repository"]["repository"],
        "source": "bundle",
        "bundle": BUNDLE_FILE,
        "head": head,
        "tags": {"wf-v1.0.0": first, "wf-v9.9.9": head},
    }

    bundle = out / BUNDLE_FILE
    bundle.write_bytes(wheel.read(f"{dist_info}/{BUNDLE_FILE}"))
    _git(clone, "bundle", "verify", "-q", str(bundle))
    restored = tmp_path / "restored"
    _git(tmp_path, "clone", "-q", str(bundle), str(restored))
    assert _git(restored, "rev-parse", "HEAD") == head
    assert _git(restored, "rev-parse", "wf-v1.0.0^{commit}") == first
    assert "wf-decl-v0.1.0" not in _git(restored, "tag", "--list")


def test_build_editable(clone: Path, tmp_path: Path) -> None:
    wheel = _build(clone, "build_editable", tmp_path / "out")
    dist_info = f"{NORM}-9.9.9.dist-info"
    assert sorted(wheel.namelist()) == [f"{dist_info}/{name}" for name in ("METADATA", "RECORD", "WHEEL", RECORD_FILE)]
    _check_record_hashes(wheel, dist_info)
    record = json.loads(wheel.read(f"{dist_info}/{RECORD_FILE}"))
    assert record["source"] == "clone" and record["clone"] == str(clone.resolve())
    assert "bundle" not in record and "head" not in record


def test_build_sdist_is_refused(clone: Path, tmp_path: Path) -> None:
    proc = _hook(clone, "build_sdist", str(tmp_path))
    assert proc.returncode != 0 and "sdists are not supported" in proc.stderr


def test_shallow_clone_is_refused(clone: Path, tmp_path: Path) -> None:
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{clone}", str(shallow))
    proc = _hook(shallow, "build_wheel", str(tmp_path / "out"))
    assert proc.returncode != 0 and "unshallow" in proc.stderr


def test_version_after_the_tag(clone: Path, tmp_path: Path) -> None:
    _git(clone, "commit", "-q", "--allow-empty", "-m", "x")
    wheel = _build(clone, "build_wheel", tmp_path / "out")
    version = Path(str(wheel.filename)).name.split("-")[1]
    assert re.fullmatch(r"9\.9\.9\.post1\+g[0-9a-f]{12}", version)
    _git(clone, "tag", "-d", "wf-v1.0.0", "wf-v9.9.9")
    wheel = _build(clone, "build_wheel", tmp_path / "untagged")
    assert Path(str(wheel.filename)).name.split("-")[1] == "0.0.0+g" + _git(clone, "rev-parse", "HEAD")[:12]


def test_extras_metadata() -> None:
    spec = importlib.util.spec_from_file_location("backend", REPO_ROOT / "tools" / "workflow_repository_build.py")
    assert spec and spec.loader
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    project = {"name": "x", "optional-dependencies": {"Dis_patcher": ["a>=1", 'b; python_version < "4"']}}
    assert backend._metadata(project, "1.0").splitlines() == [
        "Metadata-Version: 2.1",
        "Name: x",
        "Version: 1.0",
        "Provides-Extra: dis-patcher",
        'Requires-Dist: a>=1; extra == "dis-patcher"',
        'Requires-Dist: b; (python_version < "4") and extra == "dis-patcher"',
    ]


@pytest.mark.skipif(importlib.util.find_spec("pip") is None, reason="pip is not importable")
def test_real_pip_build_and_install(clone: Path, tmp_path: Path) -> None:
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-input", "--no-cache-dir"]
    offline = ["--no-deps", "--no-index", "--no-build-isolation"]
    wheels = tmp_path / "wheels"
    subprocess.run([*pip, "wheel", *offline, "-w", str(wheels), str(clone)], check=True, capture_output=True)
    assert [path.name for path in wheels.iterdir()] == [f"{NORM}-9.9.9-py3-none-any.whl"]

    target = tmp_path / "target"
    subprocess.run([*pip, "install", *offline, "--target", str(target), str(clone)], check=True, capture_output=True)
    assert (target / f"{NORM}-9.9.9.dist-info" / RECORD_FILE).is_file()
    (dist,) = importlib.metadata.distributions(path=[str(target)])
    assert dist.metadata["Name"] == NAME
    record = json.loads(dist.read_text(RECORD_FILE) or "")
    (bundle,) = (Path(str(file.locate())) for file in dist.files or [] if file.name == record["bundle"])
    _git(clone, "bundle", "verify", "-q", str(bundle))
