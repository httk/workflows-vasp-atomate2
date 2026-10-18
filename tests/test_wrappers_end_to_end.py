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
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")

from httk.atomistic import UnitcellStructureView
from httk.core import DataRecord, FileRecord
from httk.workflow import TaskManager, Workspace, collect
from httk.workflow.collecting import JobRecord
from httk.workflow.introspection import read_state, resolve_job
from httk.workflow.packages import load_workflow_package
from httk.workflow.postprocessing import run_postprocess_script
from httk.workflow.scaffold import new_job
from pymatgen.core import Structure

import fake_vasp_atomate2 as fake
from conftest import FAKE_PSP, FAKE_VASP, TestProfile, aluminium, register_ws, silicon

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
    root: Path,
    directory: str,
    command: str = FAKE_VASP,
    psp_dir: str = FAKE_PSP,
    timeout: float = 600.0,
    *,
    structure: object = None,
    settings: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
) -> tuple[Workspace, str]:
    """Run one job of the package (default: on conventional diamond Si) until the workspace is idle."""

    poscar = root / "POSCAR"
    root.mkdir(parents=True, exist_ok=True)
    cast(Any, structure or silicon()).to(filename=str(poscar))
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
            **(settings or {}),
        }
    )
    name = tomllib.loads((REPO_ROOT / directory / "httk_workflow.toml").read_text(encoding="utf-8"))["workflow"]["name"]
    job = new_job(workspace, name, inputs={"structure": str(poscar)}, parameters=parameters, install=True)
    with TaskManager(workspace, heartbeat_interval=0.01, maximum_workers=4) as manager:
        manager.run_until_idle(timeout=timeout)
    return workspace, job.job_id


def _postprocess(directory: str, record: JobRecord, script: str) -> Path:
    """Run the package's postprocess *script* for *record*, check it succeeded, and return its output directory."""

    result = run_postprocess_script(load_workflow_package(REPO_ROOT / directory, register=False), script, record)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.output_dir


def _report(directory: str, record: JobRecord) -> dict[str, Any]:
    """Run the ``report`` script and return its ``report.json``."""

    output = _postprocess(directory, record, "report")
    assert (output / "report.txt").is_file()
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def _plots(directory: str, record: JobRecord) -> set[str]:
    """Run the ``plots`` script and return the names of the files it wrote."""

    return {path.name for path in _postprocess(directory, record, "plots").iterdir()}


def _plot_data(record: JobRecord, name: str) -> dict[str, Any]:
    """A JSON file the ``plots`` script wrote next to its figures."""

    output = Path(record.workspace_root, "postprocess", *record.placement.parts, record.job_key, "plots")
    return json.loads((output / name).read_text(encoding="utf-8"))


def _check_child_postprocess(directory: str, records: list[JobRecord]) -> None:
    """A jobflow child job's scripts succeed silently and write nothing: its parent's flow output holds the results."""

    child = next(record for record in records if record.parameter("jobflow_child_job", None) is not None)
    package = load_workflow_package(REPO_ROOT / directory, register=False)
    for script in ("report", "plots"):
        result = run_postprocess_script(package, script, child)
        assert (result.returncode, result.stdout, result.stderr) == (0, "", ""), script
        assert not list(result.output_dir.iterdir()), script


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

    collected = list(collect(workspace, fail_fast=True, allow_job_collector=True))
    (item,) = [item for item in collected if item.outputs]
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
    # The same run feeds the postprocess checks.
    report = _report(directory, item.record)
    assert report["energy_eV"] == outputs["total_energy"].value
    if _BAND_GAP:
        assert report["band_gap_eV"] == outputs["band_gap"].value
    if directory == "atomate2-static":
        assert report["fermi_energy_eV"] == pytest.approx(5.0) and report["total_magnetization_muB"] == 0
        # The DOS comes from the static run's vasprun.xml.gz, found through the document's dir_name token.
        assert _plots(directory, item.record) == {"dos.png", "dos.json"}
        # A gapped system's DOS is plotted relative to its VBM, the axis label says so.
        reference = _plot_data(item.record, "dos.json")
        assert (reference["zero_eV"], reference["label"]) == (report["vbm_eV"], "E − E_VBM (eV)")
    else:
        assert report["relaxed"]["space_group_number"] == report["initial"]["space_group_number"] == 227
        assert report["initial_source"] == "job input"
        output = _postprocess(directory, item.record, "report")
        relaxed = Structure.from_file(output / "relaxed.POSCAR")
        assert relaxed.volume == pytest.approx(report["relaxed"]["volume_angstrom3"])
        assert relaxed.matches(Structure.from_file(output / "relaxed.cif"))
        assert _plots(directory, item.record) == {"relaxation.png", "relaxation.json"}
    _check_child_postprocess(directory, [item.record for item in collected])
    if directory == "atomate2-double-relax" and importlib.util.find_spec("httk.store") is not None:
        # The same run feeds the store check, rather than a second run.
        _check_double_relax_store(tmp_path, workspace, item)


def _check_double_relax_store(tmp_path: Path, workspace: Workspace, item: Any) -> None:
    """``httk collect --into`` stores the double-relax run, its relaxed structure and its energy."""

    from httk.atomistic.entries.structures import StructureEntry
    from httk.core import Run, TotalEnergyRecord
    from httk.core.cli import CLIContext
    from httk.core.storage import content_id
    from httk.store import Backend, SqlStore
    from httk.workflow.workflow_cli import command

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


# Physics package directory -> (normal profile?, structure, workspace settings, job parameters, VASP runs).
_PHYSICS = {
    "atomate2-band-structure": (True, silicon, {}, {}, 3),
    "atomate2-dielectric": (True, silicon, {}, {}, 1),
    # fake_vasp.reference_a reaches the attempts as HTTK_FAKE_VASP_REFERENCE_A: the fake's elastic reference.
    "atomate2-elastic": (False, aluminium, {"fake_vasp.reference_a": 4.05}, {}, 8),
    # Two relaxations, a static and one displacement; born_maker = null drops the dielectric run.
    "atomate2-phonons": (False, aluminium, {}, {"min_length": 8.0, "born_maker": None}, 4),
}


@pytest.mark.parametrize("directory", tuple(_PHYSICS))
def test_physics_package_runs_and_collects(tmp_path: Path, directory: str, test_profile: TestProfile) -> None:
    normal, structure, settings, parameters, runs = _PHYSICS[directory]
    if not normal and not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    pytest.importorskip("httk.analyse")
    if directory == "atomate2-phonons":
        pytest.importorskip("phonopy")
    # Elastic and phonon flows run 8 and 4 VASP jobs; custodian polls each for 10 s.
    workspace, job_id = _run(
        tmp_path, directory, timeout=1800.0, structure=structure(), settings=settings, parameters=parameters
    )
    assert resolve_job(workspace, job_id).state == "succeeded", _failure(workspace, job_id)
    invocations = [line for log in workspace.root.rglob("fake-vasp-invocations.jsonl") for line in log.open()]
    assert len(invocations) == runs

    collected = list(collect(workspace, fail_fast=True, allow_job_collector=True))
    (item,) = [item for item in collected if item.outputs]
    declaration = json.loads((REPO_ROOT / directory / "declaration.json").read_text(encoding="utf-8"))
    assert set(item.outputs) == {output["name"] for output in declaration["outputs"]}
    assert not item.unfulfilled
    outputs = item.outputs
    _check_physics_postprocess(directory, item)
    _check_child_postprocess(directory, [item.record for item in collected])
    if directory == "atomate2-band-structure":
        assert outputs["band_gap"].value == pytest.approx(1.0)
        assert "X" in outputs["electronic_band_structure"].value["kpoint_labels"]
    elif directory == "atomate2-dielectric":
        assert outputs["piezoelectric_tensor"].value[0] == pytest.approx([1.22, 1.24, 1.26, 1.30, 1.32, 1.28])
    elif directory == "atomate2-elastic":
        stiffness = np.reshape(outputs["elastic_tensor"].elastic_tensor, (6, 6))  # the typed record is flat
        assert stiffness[0, 0] == pytest.approx(108.0, abs=1e-3) and stiffness[3, 3] == pytest.approx(28.0, abs=1e-3)
    else:
        assert outputs["zero_point_energy"].value > 0
    if directory == "atomate2-band-structure" and importlib.util.find_spec("httk.store") is not None:
        _check_band_structure_store(tmp_path, workspace, item)


#: The figures (and plotted data) each physics package's ``plots`` script writes.
_FIGURES = {
    "atomate2-band-structure": {
        "band_structure.png",
        "band_structure.json",
        "dos.png",
        "dos.json",
        "band_structure_dos.png",
    },
    "atomate2-dielectric": {"dielectric.png", "born_charges.png"},
    "atomate2-elastic": {
        "elastic_tensor.png",
        "youngs_modulus.png",
        "youngs_modulus.json",
        "stress_strain.png",
        "stress_strain.json",
    },
    "atomate2-phonons": {
        "phonon_band_structure.png",
        "phonon_dos.png",
        "phonon_band_structure_dos.png",
        "phonon_thermodynamics.png",
    },
}


def _check_physics_postprocess(directory: str, item: Any) -> None:
    """The package's report repeats the collected values, and its plots script writes every figure."""

    report, outputs = _report(directory, item.record), item.outputs
    if directory == "atomate2-band-structure":
        assert report["band_gap_eV"] == outputs["band_gap"].value
        assert report["fermi_energy_eV"] == outputs["fermi_energy"].value
        # Headline values from the uniform mesh; the line path's are named line_*.
        assert report["direct_gap"] is True and report["vbm_eV"] == pytest.approx(fake.EFERMI - 0.5)
        assert report["line_direct_gap"] is True
        assert report["line_vbm"]["kpoint_label"] == report["line_cbm"]["kpoint_label"]
    elif directory == "atomate2-dielectric":
        averages = report["averages"]
        high_frequency = outputs["high_frequency_relative_permittivity"].high_frequency_relative_permittivity
        assert averages["electronic"] == pytest.approx(high_frequency)
        assert averages["total"] == pytest.approx(outputs["static_relative_permittivity"].static_relative_permittivity)
        assert report["refractive_index"] == pytest.approx(averages["electronic"] ** 0.5)
        assert report["piezoelectric_tensor_C_per_m2"] == outputs["piezoelectric_tensor"].value
        charges = [site["tensor"] for site in report["born_effective_charges"]]
        assert charges == outputs["born_effective_charges"].value
    elif directory == "atomate2-elastic":
        assert report["bulk_modulus_GPa"]["hill"] == pytest.approx(outputs["bulk_modulus_hill"].bulk_modulus_hill)
        assert report["shear_modulus_GPa"]["reuss"] == pytest.approx(outputs["shear_modulus_reuss"].shear_modulus_reuss)
        assert report["born_stable_zero_stress"] is True and report["max_abs_equilibrium_stress_GPa"] >= 0
    else:
        assert report["zero_point_energy_eV_per_cell"] == pytest.approx(outputs["zero_point_energy"].value)
        assert report["non_analytical_correction"] is False
    assert _plots(directory, item.record) == _FIGURES[directory]
    if directory == "atomate2-band-structure":
        # Every electronic figure shares one zero, the uniform mesh's VBM, where the line path puts its edges too.
        bands, dos = _plot_data(item.record, "band_structure.json"), _plot_data(item.record, "dos.json")
        assert bands["reference"] == dos["reference"] == "valence band maximum"
        assert bands["zero_eV"] == dos["zero_eV"] == report["vbm_eV"] and bands["label"] == "E − E_VBM (eV)"
        assert bands["plotted_line_vbm_eV"] == pytest.approx(0) and bands["plotted_line_cbm_eV"] == pytest.approx(1.0)


def _check_band_structure_store(tmp_path: Path, workspace: Workspace, item: Any) -> None:
    """``httk collect --into`` stores the band structure and DOS as typed records that read back equal."""

    from httk.analyse.property_records import RECORD_KINDS
    from httk.core.cli import CLIContext
    from httk.store import Backend, SqlStore
    from httk.workflow.workflow_cli import command

    store = tmp_path / "runs.sqlite"
    arguments = ["collect", "--workspace", register_ws(workspace.root), "--into", str(store)]
    arguments += ["--id-base", "atomate2.test", "--no-id-ledger"]
    assert command(arguments, CLIContext("httk", tmp_path)) == 0

    properties = "https://schemas.httk.org/defs/v0.1/properties/electronic/"
    members = {
        "electronic_band_structure": ("kpoint_labels", "kpoint_label_indices", "kpoints", "energies", "fermi_energy"),
        "electronic_density_of_states": ("energies", "density", "integrated_density"),
    }
    with Backend.sqlite(store) as database:
        sql = SqlStore(database)
        for name, fields in members.items():
            kind = RECORD_KINDS[properties + name]
            searcher = sql.searcher()
            (stored,) = [row.entry for row in searcher.results(entry=searcher.variable(kind))]
            expected = kind.from_data_record(item.outputs[name])
            for field in fields:
                assert np.asarray(getattr(stored, field)) == pytest.approx(np.asarray(getattr(expected, field))), field
