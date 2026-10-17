"""Each package runs end to end through httk's real TaskManager, with the fake VASP.

Only workspace settings configure atomate2 here: ``vasp.command`` names the fake and
``pymatgen.psp_dir`` the fake POTCAR library, and httk-workflow's jobflow runner maps
them to ``ATOMATE2_VASP_CMD``/``PMG_VASP_PSP_DIR`` in its own processes. This test
process never sets those variables (``conftest.py`` removes them), so the fake's
invocation log proves the mapping.
"""

import importlib.util
import json
import os
import shutil
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")

from httk.atomistic import UnitcellStructureView
from httk.core import DataRecord, FileRecord
from httk.workflow import TaskManager, Workspace, collect
from httk.workflow.introspection import read_state, resolve_job
from httk.workflow.scaffold import new_job
from pymatgen.core import Lattice, Structure

from conftest import FAKE_PSP, FAKE_VASP, TestProfile, register_ws

REPO_ROOT = Path(__file__).resolve().parent.parent
# Package directory -> VASP runs its Maker makes.
_RUNS = {"atomate2-static": 1, "atomate2-relax": 1, "atomate2-tight-relax": 1, "atomate2-double-relax": 2}
_NORMAL = ("atomate2-static", "atomate2-double-relax")
# The hooks only collect a band gap where httk-analyse registers its typed record.
_BAND_GAP = {"band_gap"} if importlib.util.find_spec("httk.analyse") is not None else set()


@pytest.fixture(autouse=True)
def _installed_plugin(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Install this working tree as the plugin in the test's isolated data home.

    That is what ``httk plugin install`` of this repository does. A jobflow package's
    collect hook is only used while its package is registered, so collecting needs it;
    the runner processes a manager starts inherit ``HTTK_DATA_HOME``. This process
    caches plugin discovery, so the cache is cleared around the test.
    """

    from httk.core.plugins import install_plugin
    from httk.workflow.packages import _reset_plugin_workflow_cache

    source = tmp_path_factory.mktemp("plugin-source") / "workflows-vasp-atomate2"
    source.mkdir()
    manifest = tomllib.loads((REPO_ROOT / "httk_plugin.toml").read_text(encoding="utf-8"))
    for directory in manifest["plugin"]["workflows"]:
        shutil.copytree(REPO_ROOT / directory, source / directory, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO_ROOT / "httk_plugin.toml", source)
    install_plugin(source)
    _reset_plugin_workflow_cache()
    yield
    _reset_plugin_workflow_cache()


def _failure(workspace: Workspace, job_id: str) -> object:
    """The job's recorded failure, or ``None``: the assertion message of a failed run."""

    state, _damaged = read_state(resolve_job(workspace, job_id))
    return None if state is None else state.failure


def _run(
    root: Path, directory: str, command: str = FAKE_VASP, psp_dir: str = FAKE_PSP, timeout: float = 600.0
) -> tuple[Workspace, str]:
    """Run one job of the package on conventional diamond Si until the workspace is idle."""

    poscar = root / "POSCAR"
    root.mkdir(parents=True, exist_ok=True)
    Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]).to(filename=str(poscar))
    # An empty configuration file for each library keeps a developer's ~/.atomate2.yaml and
    # ~/.config/.pmgrc.yaml out of the runner processes (through ATOMATE2_CONFIG_FILE/PMG_CONFIG_FILE).
    empty = root / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    workspace = Workspace.initialize(root / "workspace")
    workspace.configure_settings(
        {
            "vasp.command": command,
            "pymatgen.psp_dir": psp_dir,
            "atomate2.config_file": str(empty),
            "pymatgen.config_file": str(empty),
        }
    )
    name = tomllib.loads((REPO_ROOT / directory / "httk_workflow.toml").read_text(encoding="utf-8"))["workflow"]["name"]
    job = new_job(workspace, name, inputs={"structure": str(poscar)}, install=True)
    with TaskManager(workspace, heartbeat_interval=0.01, maximum_workers=4) as manager:
        manager.run_until_idle(timeout=timeout)
    return workspace, job.job_id


@pytest.mark.parametrize("directory", tuple(_RUNS))
def test_package_runs_and_collects(tmp_path: Path, directory: str, test_profile: TestProfile) -> None:
    if directory not in _NORMAL and not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    workspace, job_id = _run(tmp_path, directory)
    assert resolve_job(workspace, job_id).state == "succeeded", _failure(workspace, job_id)

    invocations = [
        json.loads(line)
        for log in workspace.root.rglob("fake-vasp-invocations.jsonl")
        for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == _RUNS[directory]
    assert {item["env"]["ATOMATE2_VASP_CMD"] for item in invocations} == {FAKE_VASP}
    assert {item["env"]["PMG_VASP_PSP_DIR"] for item in invocations} == {FAKE_PSP}
    # The POTCARs atomate2 wrote came from the fake library, not from a real one.
    titels = [titel for item in invocations for titel in item["potcar_titels"]]
    assert titels and all("FAKE-test-fixture-not-a-real-POTCAR" in titel for titel in titels)

    (item,) = [item for item in collect(workspace, fail_fast=True, allow_job_collector=True) if item.outputs]
    outputs = item.outputs
    assert isinstance(outputs["total_energy"], DataRecord)
    assert outputs["total_energy"].value == pytest.approx(-40.0, abs=0.1)
    assert isinstance(outputs["task_document"], FileRecord)
    assert outputs["task_document"].name == "jobflow-outputs.json"
    expected = {"total_energy", "task_document", *_BAND_GAP}
    if directory != "atomate2-static":
        expected.add("relaxed_structure")
        assert isinstance(outputs["relaxed_structure"], UnitcellStructureView)
    assert set(outputs) == expected
    if _BAND_GAP:
        assert outputs["band_gap"].value == pytest.approx(1.0)


def test_double_relax_collects_into_a_store(tmp_path: Path) -> None:
    pytest.importorskip("httk.store")
    from httk.atomistic.entries.structures import StructureEntry
    from httk.core import Run, TotalEnergyRecord
    from httk.core.cli import CLIContext
    from httk.core.storage import content_id
    from httk.store import Backend, SqlStore
    from httk.workflow.workflow_cli import command

    workspace, job_id = _run(tmp_path / "run", "atomate2-double-relax")
    assert resolve_job(workspace, job_id).state == "succeeded", _failure(workspace, job_id)
    (item,) = [item for item in collect(workspace, fail_fast=True, allow_job_collector=True) if item.outputs]
    store = tmp_path / "runs.sqlite"
    arguments = ["collect", "--workspace", register_ws(workspace.root), "--into", str(store)]
    arguments += ["--id-base", "atomate2.test", "--no-id-ledger"]
    assert command(arguments, CLIContext("httk", tmp_path)) == 0

    declaration = json.loads((REPO_ROOT / "atomate2-double-relax" / "declaration.json").read_text(encoding="utf-8"))
    with Backend.sqlite(store) as database:
        sql = SqlStore(database)
        searcher = sql.searcher()
        runs = [row.run for row in searcher.results(run=searcher.variable(Run))]
        (run,) = [run for run in runs if run.workflow_declaration_uri == declaration["$id"]]
        assert {edge.label for edge in run.outputs} == {
            "relaxed_structure",
            "total_energy",
            "task_document",
            *_BAND_GAP,
        }
        searcher = sql.searcher()
        (energy,) = [row.energy for row in searcher.results(energy=searcher.variable(TotalEnergyRecord))]
        assert energy.total_energy == pytest.approx(item.outputs["total_energy"].value)
        structure_id = content_id(item.outputs["relaxed_structure"])
        assert sql.fetch_entry(StructureEntry, structure_id, eager=True) is not None


def test_static_runs_on_real_vasp(tmp_path: Path) -> None:
    """The opt-in real-VASP arm: atomate2.static with a real VASP command and POTCAR library."""

    command = os.environ.get("HTTK_TEST_VASP_COMMAND")
    psp_dir = os.environ.get("HTTK_TEST_PMG_VASP_PSP_DIR")
    if not command or not psp_dir:
        pytest.skip("set HTTK_TEST_VASP_COMMAND and HTTK_TEST_PMG_VASP_PSP_DIR to run atomate2.static on real VASP")
    workspace, job_id = _run(tmp_path, "atomate2-static", command, psp_dir, timeout=7200.0)
    assert resolve_job(workspace, job_id).state == "succeeded", _failure(workspace, job_id)
    (item,) = [item for item in collect(workspace, fail_fast=True, allow_job_collector=True) if item.outputs]
    assert {"total_energy", "task_document"} <= set(item.outputs)
