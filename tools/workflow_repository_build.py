"""In-tree PEP 517 build backend for httk workflow repositories.

A workflow repository is a git repository of ``httk_workflow.toml`` packages.
``pip install CLONE`` and ``pip install -e CLONE`` build, through this backend,
a ``py3-none-any`` wheel holding no importable files, only its ``.dist-info``:
the distribution metadata (so pip installs the workflows' Python dependencies)
plus ``httk_workflow_repository.json``, the record *httk-workflow* discovers
the repository by. A regular wheel also carries ``httk_workflow_repository.bundle``,
a git bundle of HEAD and every ``wf-v*`` tag, so the offered versions are
materialized from git objects even after the clone is gone; an editable wheel
records the clone's path instead and follows it.

Everything repository-specific comes from ``pyproject.toml`` (``[project]`` and
``[tool.httk.workflow-repository] repository``). This file is stdlib-only, uses
no network, and is copied byte-identically into each httk workflow repository;
the reference copy lives in workflows-examples.
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

__all__ = [
    "WorkflowRepositoryBuildError",
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_wheel",
]

TAG_PATTERN = "wf-v*"
RECORD_FILE = "httk_workflow_repository.json"
BUNDLE_FILE = "httk_workflow_repository.bundle"
_PROJECT_KEYS = {"name", "dynamic", "description", "requires-python", "dependencies", "optional-dependencies"}
_NAME = re.compile(r"[A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9]", re.IGNORECASE)
_RELEASE_TAG = re.compile(r"wf-v(\d+(?:\.\d+)*)")


class WorkflowRepositoryBuildError(RuntimeError):
    """The repository cannot be built into a workflow-repository wheel."""


def get_requires_for_build_wheel(config_settings: dict[str, object] | None = None) -> list[str]:
    """Return the extra build requirements of a wheel build: none.

    :param config_settings: Frontend settings (unused).
    :return: An empty list.
    """
    return []


def get_requires_for_build_editable(config_settings: dict[str, object] | None = None) -> list[str]:
    """Return the extra build requirements of an editable build: none.

    :param config_settings: Frontend settings (unused).
    :return: An empty list.
    """
    return []


def build_wheel(
    wheel_directory: str, config_settings: dict[str, object] | None = None, metadata_directory: str | None = None
) -> str:
    """Build the wheel: metadata, repository record and a git bundle of HEAD and the ``wf-v*`` tags.

    :param wheel_directory: Directory to write the wheel into.
    :param config_settings: Frontend settings (unused).
    :param metadata_directory: Prepared metadata (unused; no prepare hook exists).
    :return: The wheel's basename.
    :raises WorkflowRepositoryBuildError: If the source is not a full git clone or ``pyproject.toml`` is invalid.
    """
    return _build(Path(wheel_directory), editable=False)


def build_editable(
    wheel_directory: str, config_settings: dict[str, object] | None = None, metadata_directory: str | None = None
) -> str:
    """Build the PEP 660 editable wheel: metadata and a record naming this clone.

    :param wheel_directory: Directory to write the wheel into.
    :param config_settings: Frontend settings (unused).
    :param metadata_directory: Prepared metadata (unused; no prepare hook exists).
    :return: The wheel's basename.
    :raises WorkflowRepositoryBuildError: If the source is not a full git clone or ``pyproject.toml`` is invalid.
    """
    return _build(Path(wheel_directory), editable=True)


def build_sdist(sdist_directory: str, config_settings: dict[str, object] | None = None) -> str:
    """Refuse to build an sdist.

    :param sdist_directory: Directory an sdist would be written into.
    :param config_settings: Frontend settings (unused).
    :return: Never returns.
    :raises WorkflowRepositoryBuildError: Always.
    """
    raise WorkflowRepositoryBuildError(
        "httk workflow repositories install only from a git clone (pip install CLONE or pip install -e CLONE); "
        "sdists are not supported"
    )


def _build(wheel_directory: Path, *, editable: bool) -> str:
    project, repository = _read_pyproject(Path("pyproject.toml"))
    root = _clone_root()
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").lower()
    tags = _wf_tags(root)
    name = project["name"]
    version = _version(root, head, tags)
    norm = re.sub(r"[-_.]+", "_", name).lower()
    dist_info = f"{norm}-{version}.dist-info"

    record: dict[str, object] = {"format": "httk-workflow-repository", "format_version": 1, "repository": repository}
    files: dict[str, bytes] = {
        f"{dist_info}/METADATA": _metadata(project, version).encode(),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: workflow_repository_build\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    if editable:
        record.update(source="clone", clone=str(root))
    else:
        record.update(source="bundle", bundle=BUNDLE_FILE, head=head, tags=tags)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / BUNDLE_FILE
            _git(root, "bundle", "create", "-q", str(bundle), "HEAD", *(f"refs/tags/{tag}" for tag in tags))
            # ponytail: the bundle is read into memory; stream it if a repository's history ever outgrows RAM
            files[f"{dist_info}/{BUNDLE_FILE}"] = bundle.read_bytes()
    files[f"{dist_info}/{RECORD_FILE}"] = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()

    lines = []
    for path, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        lines.append(f"{path},sha256={digest},{len(data)}\n")
    lines.append(f"{dist_info}/RECORD,,\n")
    entries = [*sorted(files.items()), (f"{dist_info}/RECORD", "".join(lines).encode())]

    wheel_name = f"{norm}-{version}-py3-none-any.whl"
    wheel_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_directory / wheel_name, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path, data in entries:
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            wheel.writestr(info, data)
    return wheel_name


def _read_pyproject(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        raise WorkflowRepositoryBuildError("pyproject.toml has no [project] table")
    if "version" in project:
        raise WorkflowRepositoryBuildError(
            'pyproject.toml [project] must not set version: it comes from the wf-v* tags (dynamic = ["version"])'
        )
    if unknown := sorted(set(project) - _PROJECT_KEYS):
        raise WorkflowRepositoryBuildError(
            f"pyproject.toml [project] keys {unknown} are not supported by workflow_repository_build "
            f"(supported: {sorted(_PROJECT_KEYS)})"
        )
    name = project.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise WorkflowRepositoryBuildError(f"pyproject.toml [project] name must be a valid distribution name: {name!r}")
    if project.get("dynamic") != ["version"]:
        raise WorkflowRepositoryBuildError('pyproject.toml [project] must declare dynamic = ["version"]')
    for key in ("description", "requires-python"):
        if not isinstance(project.get(key, ""), str):
            raise WorkflowRepositoryBuildError(f"pyproject.toml [project] {key} must be a string")
    if not _is_string_list(project.get("dependencies", [])):
        raise WorkflowRepositoryBuildError("pyproject.toml [project] dependencies must be a list of strings")
    extras = project.get("optional-dependencies", {})
    if not isinstance(extras, dict) or not all(_is_string_list(reqs) for reqs in extras.values()):
        raise WorkflowRepositoryBuildError(
            "pyproject.toml [project] optional-dependencies must map extra names to lists of strings"
        )

    table = data.get("tool", {}).get("httk", {}).get("workflow-repository", {})
    repository = table.get("repository") if isinstance(table, dict) else None
    if not isinstance(repository, str):
        raise WorkflowRepositoryBuildError(
            "pyproject.toml must set [tool.httk.workflow-repository] repository, "
            'e.g. repository = "git+https://github.com/httk/<repo>"'
        )
    _check_repository(repository)
    return project, repository


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _check_repository(repository: str) -> None:
    problem = None
    scheme, sep, rest = repository.partition("://")
    host = rest.split("/", 1)[0]
    if not sep or scheme not in ("git+https", "git+http", "git+file"):
        problem = "must start with git+https://, git+http:// or git+file://"
    elif re.search(r"[\s@#?]", repository):
        problem = "must not contain whitespace, '@', '#' or '?' (the commit and directory are added per workflow)"
    elif repository.endswith("/"):
        problem = "must not end with '/'"
    elif host != host.lower() or (scheme != "git+file" and not host):
        problem = "must have a lowercase host"
    if problem:
        raise WorkflowRepositoryBuildError(f"[tool.httk.workflow-repository] repository {repository!r} {problem}")


def _metadata(project: dict[str, Any], version: str) -> str:
    lines = ["Metadata-Version: 2.1", f"Name: {project['name']}", f"Version: {version}"]
    if description := project.get("description"):
        lines.append(f"Summary: {description}")
    if requires_python := project.get("requires-python"):
        lines.append(f"Requires-Python: {requires_python}")
    lines += [f"Requires-Dist: {req}" for req in project.get("dependencies", [])]
    for declared, reqs in project.get("optional-dependencies", {}).items():
        extra = re.sub(r"[-_.]+", "-", declared).lower()
        lines.append(f"Provides-Extra: {extra}")
        for req in reqs:
            requirement, _, marker = (part.strip() for part in req.partition(";"))
            condition = f"({marker}) and " if marker else ""
            lines.append(f'Requires-Dist: {requirement}; {condition}extra == "{extra}"')
    return "\n".join(lines) + "\n"


def _clone_root() -> Path:
    cwd = Path.cwd().resolve()
    try:
        top = _git(cwd, "rev-parse", "--show-toplevel")
    except WorkflowRepositoryBuildError as error:
        raise WorkflowRepositoryBuildError(
            f"{cwd} is not a git work tree: install from a git clone of the repository ({error})"
        ) from None
    if Path(top).resolve() != cwd:
        raise WorkflowRepositoryBuildError(f"{cwd} is not the root of a git clone (its work tree is {top})")
    if _git(cwd, "rev-parse", "--is-shallow-repository") == "true":
        raise WorkflowRepositoryBuildError(
            f"{cwd} is a shallow clone; run `git fetch --unshallow --tags` in it, then install again"
        )
    return cwd


def _wf_tags(root: Path) -> dict[str, str]:
    """Return ``{tag: commit}`` for the ``wf-v*`` tags that peel to a commit."""
    out = _git(
        root,
        "for-each-ref",
        "--format=%(refname:strip=2)%09%(objecttype)%09%(objectname)%09%(*objecttype)%09%(*objectname)",
        f"refs/tags/{TAG_PATTERN}",
    )
    tags: dict[str, str] = {}
    for line in out.splitlines():
        tag, kind, obj, peeled_kind, peeled = (line.split("\t") + [""] * 5)[:5]
        if kind == "commit":
            tags[tag] = obj.lower()
        elif peeled_kind == "commit":
            tags[tag] = peeled.lower()
    return tags


def _version(root: Path, head: str, tags: dict[str, str]) -> str:
    at_head = [m.group(1) for tag, commit in tags.items() if commit == head and (m := _RELEASE_TAG.fullmatch(tag))]
    if at_head:
        return ".".join(map(str, max(tuple(map(int, v.split("."))) for v in at_head)))
    try:
        described = _git(root, "describe", "--tags", "--match", "wf-v[0-9]*", "--long", "--abbrev=12", "HEAD")
    except WorkflowRepositoryBuildError:  # no matching tag in HEAD's history
        described = ""
    match = re.fullmatch(r"(.+)-(\d+)-g([0-9a-f]+)", described)
    if match and (release := _RELEASE_TAG.fullmatch(match.group(1))):
        release_version = ".".join(str(int(part)) for part in release.group(1).split("."))
        return f"{release_version}.post{match.group(2)}+g{match.group(3)}"
    return f"0.0.0+g{head[:12]}"


def _git(root: Path, *args: str) -> str:
    """Run git in ``root`` isolated from user/system config, hooks and inherited ``GIT_*`` variables."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null", GIT_TERMINAL_PROMPT="0")
    command = ["git", "-c", "core.hooksPath=/dev/null", *args]
    try:
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env, check=False)
    except FileNotFoundError:
        raise WorkflowRepositoryBuildError("git is required to build an httk workflow repository") from None
    if proc.returncode:
        raise WorkflowRepositoryBuildError(f"`git {' '.join(args)}` failed in {root}: {proc.stderr.strip()}")
    return proc.stdout.strip()
