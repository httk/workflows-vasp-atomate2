"""Each package's ``collect.py`` turns the jobflow runner's task document into its declared roles.

The document is a real one: the fake VASP runs on atomate2's static input set and
emmet builds the task document from its outputs, as atomate2's StaticMaker does.
"""

import importlib.util
import json
import logging
import os
import runpy
import subprocess
import time
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")

from atomate2.vasp.sets.core import StaticSetGenerator
from emmet.core.tasks import TaskDoc
from httk.atomistic import UnitcellStructureView
from httk.core import DataRecord, FileRecord
from httk.workflow.collecting import JobRecord
from monty.json import jsanitize
from pymatgen.core import SETTINGS, Lattice, Structure

from conftest import FAKE_PSP, FAKE_VASP

REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = ("atomate2-static", "atomate2-relax", "atomate2-tight-relax", "atomate2-double-relax")
_HAS_ANALYSE = importlib.util.find_spec("httk.analyse") is not None
_PAYLOAD = "jobs/job--12345678-1234-4234-8234-123456789abc"


@pytest.fixture(scope="module")
def task_document(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A task document of the fake VASP's static run on 8-atom Si, as plain JSON data."""

    directory = tmp_path_factory.mktemp("static")
    silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])
    with pytest.MonkeyPatch.context() as monkeypatch:
        # pymatgen shares one lazily built SETTINGS dict; patching it in place works whatever imported it first.
        monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", FAKE_PSP)
        StaticSetGenerator().get_input_set(silicon).write_input(directory)
    subprocess.run([FAKE_VASP], cwd=directory, check=True, capture_output=True)
    return jsanitize(TaskDoc.from_directory(directory), strict=True, enum_values=True)


def _record(root: Path, package: str, *, transactional: bool, parameters: dict | None = None) -> JobRecord:
    declaration = json.loads((REPO_ROOT / package / "declaration.json").read_text(encoding="utf-8"))
    return JobRecord(
        workspace_root=root,
        workspace_id="ws",
        job_id="12345678-1234-4234-8234-123456789abc",
        job_key="job--12345678-1234-4234-8234-123456789abc",
        job={"parameters": parameters or {}},
        runner_provenance=None,
        state="succeeded",
        failure=None,
        placement=PurePosixPath("jobs"),
        payload_path=PurePosixPath("jobs/job--12345678-1234-4234-8234-123456789abc"),
        workdir_path=PurePosixPath("run"),
        data_path=PurePosixPath("data") if transactional else None,
        provenance={},
        runner_steps=None,
        children={},
        declarations={"workflow": {"declared": declaration, "observed": None}},
    )


def _write(path: Path, task: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"output": task}), encoding="utf-8")
    return path


def _collect(package: str, record: JobRecord) -> dict:
    return runpy.run_path(str(REPO_ROOT / package / "collect.py"))["collect"](record)


@pytest.mark.parametrize("transactional", (False, True))
@pytest.mark.parametrize("package", _PACKAGES)
def test_collect_hook_reads_the_task_document(
    tmp_path: Path, task_document: dict, package: str, transactional: bool
) -> None:
    # A job that committed data is read from data/<jobflow_data_prefix>/, any other from its workdir.
    path = tmp_path / ("data/custom" if transactional else "run") / "jobflow-outputs.json"
    _write(path, task_document)
    parameters = {"jobflow_data_prefix": "custom"} if transactional else {}
    outputs = _collect(package, _record(tmp_path, package, transactional=transactional, parameters=parameters))

    relax = package != "atomate2-static"
    expected = {"total_energy", "task_document"} | ({"band_gap"} if _HAS_ANALYSE else set())
    assert set(outputs) == expected | ({"relaxed_structure"} if relax else set())
    energy = outputs["total_energy"]
    assert isinstance(energy, DataRecord)
    assert energy.name == "_httk_total_energy"
    assert energy.value == pytest.approx(task_document["output"]["energy"]) == pytest.approx(-40.0)
    if _HAS_ANALYSE:
        assert outputs["band_gap"].name == "_httk_band_gap"
        assert outputs["band_gap"].value == pytest.approx(1.0)
    document = outputs["task_document"]
    assert isinstance(document, FileRecord)
    assert document.url == path.relative_to(tmp_path).as_posix()
    assert (document.name, document.media_type, document.size) == (
        "jobflow-outputs.json",
        "application/json",
        path.stat().st_size,
    )
    if relax:
        structure = outputs["relaxed_structure"]
        assert isinstance(structure, UnitcellStructureView)
        assert len(Structure.from_dict(task_document["output"]["structure"])) == 8


def test_collect_hook_omits_a_missing_band_gap(tmp_path: Path, task_document: dict) -> None:
    task = json.loads(json.dumps(task_document))
    task["output"]["bandgap"] = None
    _write(tmp_path / "run" / "jobflow-outputs.json", task)
    outputs = _collect("atomate2-static", _record(tmp_path, "atomate2-static", transactional=False))
    assert set(outputs) == {"total_energy", "task_document"}


def test_collect_hook_omits_the_band_gap_without_httk_analyse(
    tmp_path: Path, task_document: dict, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *args: None if name == "httk.analyse" else find_spec(name, *args)
    )
    _write(tmp_path / "run" / "jobflow-outputs.json", task_document)
    with caplog.at_level(logging.WARNING):
        outputs = _collect("atomate2-static", _record(tmp_path, "atomate2-static", transactional=False))
    assert set(outputs) == {"total_energy", "task_document"}
    assert "httk-analyse" in caplog.text


@pytest.mark.parametrize("missing", ("energy", "structure"))
def test_collect_hook_rejects_a_document_without_energy_or_structure(
    tmp_path: Path, task_document: dict, missing: str
) -> None:
    task = json.loads(json.dumps(task_document))
    del task["output"][missing]
    _write(tmp_path / "run" / "jobflow-outputs.json", task)
    with pytest.raises(ValueError, match="output.structure or output.energy"):
        _collect("atomate2-relax", _record(tmp_path, "atomate2-relax", transactional=False))


@pytest.mark.parametrize("document", ([], {"output": []}, {"output": {"output": "not a mapping"}}))
def test_collect_hook_rejects_a_malformed_document(tmp_path: Path, document: object) -> None:
    path = tmp_path / "run" / "jobflow-outputs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="output.structure or output.energy"):
        _collect("atomate2-static", _record(tmp_path, "atomate2-static", transactional=False))


def test_collect_hook_rejects_a_non_finite_energy(tmp_path: Path, task_document: dict) -> None:
    task = json.loads(json.dumps(task_document))
    task["output"]["energy"] = float("inf")
    _write(tmp_path / "run" / "jobflow-outputs.json", task)
    with pytest.raises(ValueError, match="not finite"):
        _collect("atomate2-static", _record(tmp_path, "atomate2-static", transactional=False))


@pytest.mark.parametrize("field", ("energy", "bandgap"))
def test_collect_hook_rejects_a_non_numeric_value(tmp_path: Path, task_document: dict, field: str) -> None:
    task = json.loads(json.dumps(task_document))
    task["output"][field] = {"not": "a number"}
    _write(tmp_path / "run" / "jobflow-outputs.json", task)
    with pytest.raises(ValueError, match="is not a number"):
        _collect("atomate2-static", _record(tmp_path, "atomate2-static", transactional=False))


def test_collect_hook_yields_nothing_for_a_child_job(tmp_path: Path) -> None:
    # The child jobs running the flow's atomate2 jobs share the workflow; their parent holds the result.
    record = _record(tmp_path, "atomate2-double-relax", transactional=False, parameters={"jobflow_child_job": "x.json"})
    assert _collect("atomate2-double-relax", record) == {}


def test_collect_hook_yields_the_task_document_of_a_vasp_child_job(
    tmp_path: Path, task_document: dict, caplog: pytest.LogCaptureFixture
) -> None:
    record = _record(tmp_path, "atomate2-relax", transactional=False, parameters={"jobflow_child_job": "x.json"})
    # An absent or empty child-outputs.json yields nothing.
    assert _collect("atomate2-relax", record) == {}
    path = tmp_path / _PAYLOAD / "run" / "child-outputs.json"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    assert _collect("atomate2-relax", record) == {}
    task = json.loads(json.dumps(task_document))
    task["calcs_reversed"][0]["vasp_version"] = "6.6.0"
    task["run_stats"]["overall"].update(cores=32, elapsed_time=2.99)
    path.write_text(json.dumps({"docs": [{"output": task}], "response": {}}), encoding="utf-8")
    document = _collect("atomate2-relax", record)["task_document"]
    assert (document.url, document.name, document.size) == (
        f"{_PAYLOAD}/run/child-outputs.json",
        path.name,
        path.stat().st_size,
    )
    assert document.description == "VASP 6.6.0, PBEsol Static, 32 cores, 2.99 s"
    # A missing part is left out.
    del task["calc_type"]
    path.write_text(json.dumps({"docs": [{"output": task}], "response": {}}), encoding="utf-8")
    assert _collect("atomate2-relax", record)["task_document"].description == "VASP 6.6.0, 32 cores, 2.99 s"
    # A child job that ran no VASP (a generator or a fit) yields nothing.
    path.write_text(json.dumps({"docs": [{"output": [1, 2]}], "response": {}}), encoding="utf-8")
    assert _collect("atomate2-relax", record) == {}
    # A symlinked child-outputs.json is not followed: nothing, with a warning.
    target = path.with_name("elsewhere.json")
    path.rename(target)
    path.symlink_to(target)
    with caplog.at_level(logging.WARNING):
        assert _collect("atomate2-relax", record) == {}
    assert "not a regular file" in caplog.text


def test_collect_hook_hands_over_the_port_structure_only_from_the_payload(tmp_path: Path, task_document: dict) -> None:
    from httk.workflow.hookapi import Collected

    _write(tmp_path / "run" / "jobflow-outputs.json", task_document)
    files = tmp_path / _PAYLOAD / "files"
    files.mkdir(parents=True)
    structure = Structure.from_dict(task_document["output"]["structure"])
    (files / "structure.json").write_text(json.dumps(structure.as_dict()), encoding="utf-8")
    structure.to(filename=str(files / "POSCAR"))
    structure.to(filename=str(files / "structure.cif"))
    (files / "structure.yaml").write_text("{}", encoding="utf-8")
    (files / "link.json").symlink_to(files / "structure.json")
    os.mkfifo(files / "fifo.json")

    def collect(value: str) -> object:
        port = {"jobflow_inputs": {"structure": {"kind": "path", "value": value}}}
        return _collect("atomate2-relax", _record(tmp_path, "atomate2-relax", transactional=False, parameters=port))

    # A JSON file is a pymatgen structure dictionary, any other a structure file read as the jobflow runner reads it.
    for name in ("structure.json", "POSCAR", "structure.cif"):
        result = collect(f"files/{name}")
        assert isinstance(result, Collected), name
        assert isinstance(result.inputs["initial_structure"], UnitcellStructureView), name
    for value in ("../../run/jobflow-outputs.json", str(files / "structure.json")):
        with pytest.raises(ValueError, match="not relative to the job's payload"):
            collect(value)
    # pymatgen's name-based reader would hand other names (YAML, monty JSON) to monty's decoder.
    with pytest.raises(ValueError, match="is not a .json, POSCAR/CONTCAR, .vasp or .cif file"):
        collect("files/structure.yaml")
    # A symlink and a FIFO are refused at once, never followed or opened for reading.
    for name in ("link.json", "fifo.json"):
        started = time.monotonic()
        with pytest.raises(ValueError, match=f"cannot read files/{name}"):
            collect(f"files/{name}")
        assert time.monotonic() - started < 10, name
