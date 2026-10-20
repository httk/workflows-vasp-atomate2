"""The physics packages' collect hooks turn real fake-VASP flow outputs into their declared roles.

Each document is the ``jobflow-outputs.json`` httk's jobflow runner writes, produced by
running the package's Maker in-process on the fake VASP (``conftest.flow_document``).
The values are checked against the fake's synthetic physics, including the component
orders and unit conversions the hooks apply.
"""

import ast
import importlib.util
import json
import re
import runpy
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")
pytest.importorskip("httk.analyse")

import numpy as np
from httk.analyse import property_records
from httk.atomistic import UnitcellStructureView
from httk.core import DataRecord, FileRecord, content_id
from httk.workflow import Workspace
from httk.workflow.collecting import JobRecord
from httk.workflow.packages import load_workflow_package
from httk.workflow.postprocessing import run_postprocess_script

import fake_vasp_atomate2 as fake
from conftest import TestProfile, aluminium, configure_fake_vasp, flow_document, silicon

REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = ("atomate2-band-structure", "atomate2-dielectric", "atomate2-elastic", "atomate2-phonons")
#: h in eV/THz (CODATA 2018, exact): the energy of one THz quantum.
_PLANCK_EV_PER_THZ = 6.62607015e-34 / 1.602176634e-19 * 1e12


def _document(tmp_path_factory: pytest.TempPathFactory, make: object) -> dict:
    with pytest.MonkeyPatch.context() as monkeypatch:
        configure_fake_vasp(monkeypatch, tmp_path_factory.mktemp("flow"))
        return flow_document(make())  # type: ignore[operator]


@pytest.fixture(scope="session")
def band_structure_document(tmp_path_factory: pytest.TempPathFactory) -> dict:
    from atomate2.vasp.flows.core import BandStructureMaker

    return _document(tmp_path_factory, lambda: BandStructureMaker().make(silicon()))


@pytest.fixture(scope="session")
def dielectric_document(tmp_path_factory: pytest.TempPathFactory) -> dict:
    from atomate2.vasp.jobs.core import DielectricMaker

    return _document(tmp_path_factory, lambda: DielectricMaker().make(silicon()))


@pytest.fixture(scope="session")
def phonon_document(tmp_path_factory: pytest.TempPathFactory) -> dict:
    pytest.importorskip("phonopy")
    from atomate2.vasp.flows.phonons import PhononMaker

    return _document(tmp_path_factory, lambda: PhononMaker(born_maker=None, min_length=8.0).make(aluminium()))


def _record(tmp_path: Path, package: str, document: dict) -> JobRecord:
    """A finished job of *package* in a workspace at *tmp_path*, with *document* as its workdir ``jobflow-outputs.json``."""

    Workspace.initialize(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "jobflow-outputs.json").write_text(json.dumps(document), encoding="utf-8")
    declaration = json.loads((REPO_ROOT / package / "declaration.json").read_text(encoding="utf-8"))
    workflow = tomllib.loads((REPO_ROOT / package / "httk_workflow.toml").read_text(encoding="utf-8"))["workflow"]
    payload = tmp_path / "jobs" / "job--12345678-1234-4234-8234-123456789abc"
    payload.mkdir(parents=True)
    job = {"id": "12345678-1234-4234-8234-123456789abc", "workflow": {"name": workflow["name"]}, "parameters": {}}
    (payload / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return JobRecord(
        workspace_root=tmp_path,
        workspace_id="ws",
        job_id="12345678-1234-4234-8234-123456789abc",
        job_key="job--12345678-1234-4234-8234-123456789abc",
        job={"parameters": {}},
        runner_provenance=None,
        state="succeeded",
        failure=None,
        placement=PurePosixPath("jobs"),
        payload_path=PurePosixPath("jobs/job--12345678-1234-4234-8234-123456789abc"),
        workdir_path=PurePosixPath("run"),
        data_path=None,
        provenance={},
        runner_steps=None,
        children={},
        declarations={"workflow": {"declared": declaration, "observed": None}},
    )


def _hook(tmp_path: Path, package: str, document: dict) -> dict:
    """Run the package's hook on *document*, as the workdir ``jobflow-outputs.json`` of a finished job."""

    return runpy.run_path(str(REPO_ROOT / package / "collect.py"))["collect"](_record(tmp_path, package, document))


def _postprocess(tmp_path: Path, package: str, document: dict, script: str) -> tuple[Path, str]:
    """Run the package's postprocess *script* on *document*; return its output directory and stdout."""

    record = _record(tmp_path, package, document)
    result = run_postprocess_script(load_workflow_package(REPO_ROOT / package, register=False), script, record)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.output_dir, result.stdout


def _report_dir(tmp_path: Path) -> Path:
    """The ``report`` output directory of the job ``_record`` makes in *tmp_path*."""

    return tmp_path / "report" / "postprocess" / "jobs" / "job--12345678-1234-4234-8234-123456789abc" / "report"


def _report(tmp_path: Path, package: str, document: dict) -> dict:
    output, _ = _postprocess(tmp_path, package, document, "report")
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def _collect(tmp_path: Path, package: str, document: dict) -> dict:
    """Run the package's hook on *document* and check it fulfils every declared role with valid records."""

    outputs = _hook(tmp_path, package, document)
    declaration = json.loads((REPO_ROOT / package / "declaration.json").read_text(encoding="utf-8"))
    assert set(outputs) == {output["name"] for output in declaration["outputs"]}
    assert isinstance(outputs["task_document"], FileRecord)
    for value in outputs.values():
        # Every generic record must convert to the typed record the store keeps it as.
        if isinstance(value, DataRecord) and value.definition_id in property_records.RECORD_KINDS:
            property_records.RECORD_KINDS[value.definition_id].from_data_record(value)
    return outputs


def _value(record: object) -> object:
    assert isinstance(record, DataRecord)
    return record.value


def test_band_structure_hook(tmp_path: Path, band_structure_document: dict) -> None:
    outputs = _collect(tmp_path, "atomate2-band-structure", band_structure_document)

    assert isinstance(outputs["band_structure_cell"], UnitcellStructureView)
    bands = _value(outputs["electronic_band_structure"])
    assert isinstance(bands, dict)
    labels, indices = bands["kpoint_labels"], bands["kpoint_label_indices"]
    assert len(labels) == len(indices) and indices == sorted(set(indices)) and indices[0] == 0
    assert indices[-1] == len(bands["kpoints"]) - 1
    # Segment junctions are the same point twice with the same label; K|U is a discontinuity.
    pairs = [(labels[i], labels[i + 1]) for i in range(len(indices) - 1) if indices[i + 1] == indices[i] + 1]
    assert ("X", "X") in pairs and ("K", "U") in pairs
    energies = np.array(bands["energies"])
    assert energies.shape[1] == len(bands["kpoints"]) and (np.diff(energies, axis=0) >= 0).all()
    gamma = np.linalg.norm(bands["kpoints"], axis=1) < 1e-8
    occupied = energies[energies.max(axis=1) < fake.EFERMI]
    assert energies[len(occupied), gamma].min() - occupied[-1, gamma].max() == pytest.approx(1.0)
    assert bands["fermi_energy"] == pytest.approx(fake.EFERMI) and "energies_spin_down" not in bands

    dos = _value(outputs["electronic_density_of_states"])
    assert isinstance(dos, dict)
    # The fake's DOS holds two electrons per band and k-point weight: the integral counts all band states.
    assert dos["integrated_density"][-1] == pytest.approx(2 * energies.shape[0], rel=0.05)
    assert _value(outputs["band_gap"]) == pytest.approx(1.0)
    assert _value(outputs["fermi_energy"]) == pytest.approx(fake.EFERMI)


def test_dielectric_hook_reorders_the_piezoelectric_tensor(tmp_path: Path, dielectric_document: dict) -> None:
    outputs = _collect(tmp_path, "atomate2-dielectric", dielectric_document)

    electronic, ionic = np.array(fake.epsilon_electronic()), np.array(fake.epsilon_ionic())
    assert outputs["high_frequency_relative_permittivity"].high_frequency_relative_permittivity == pytest.approx(
        np.trace(electronic) / 3
    )
    assert outputs["static_relative_permittivity"].static_relative_permittivity == pytest.approx(
        np.trace(electronic + ionic) / 3
    )
    assert np.allclose(_value(outputs["born_effective_charges"]), fake.born_charges(8))
    # VASP prints XX YY ZZ XY YZ ZX; the definition orders xx yy zz yz xz xy. Row x of the total
    # (electronic + ionic) is 1.22 ... 1.32 in VASP's order, every component distinct.
    piezo = np.array(_value(outputs["piezoelectric_tensor"]))
    total = np.array(fake.piezo(ionic=False)) + np.array(fake.piezo(ionic=True))
    assert np.allclose(piezo, total[:, [0, 1, 2, 4, 5, 3]])
    assert piezo[0] == pytest.approx([1.22, 1.24, 1.26, 1.30, 1.32, 1.28])
    assert _value(outputs["total_energy"]) == pytest.approx(-40.0)


def test_phonon_hook_converts_to_the_phonon_cell(tmp_path: Path, phonon_document: dict) -> None:
    outputs = _collect(tmp_path, "atomate2-phonons", phonon_document)

    output = phonon_document["output"]
    assert len(outputs["phonon_cell"].sites) == 1
    bands = _value(outputs["phonon_band_structure"])
    assert isinstance(bands, dict)
    frequencies = np.array(bands["frequencies"])
    assert frequencies.shape == (3, len(bands["qpoints"])) and (np.diff(frequencies, axis=0) >= 0).all()
    assert "\\Gamma" in bands["qpoint_labels"] or "GAMMA" in bands["qpoint_labels"]
    thermodynamics = _value(outputs["vibrational_thermodynamics"])
    assert isinstance(thermodynamics, dict)
    # J/mol per formula unit -> eV per cell; the Al phonon cell is one formula unit.
    to_ev = 1 / (6.02214076e23 * 1.602176634e-19)
    assert thermodynamics["helmholtz_free_energies"] == pytest.approx(np.array(output["free_energies"]) * to_ev)
    assert thermodynamics["entropies"] == pytest.approx(np.array(output["entropies"]) * to_ev)
    zero_point = _value(outputs["zero_point_energy"])
    assert zero_point == thermodynamics["helmholtz_free_energies"][0]

    # Unit consistency: the zero-point energy is half of h nu summed over the modes, here integrated over
    # the stored DOS. phonopy computes both on the same q-mesh (observed agreement 3e-10); 1e-3 leaves
    # room for the trapezoid rule on the DOS grid.
    dos = _value(outputs["phonon_density_of_states"])
    assert isinstance(dos, dict)
    nu, density = np.array(dos["frequencies"]), np.array(dos["density"])
    positive = nu > 0
    integral = 0.5 * _PLANCK_EV_PER_THZ * np.trapezoid(density[positive] * nu[positive], nu[positive])
    assert integral == pytest.approx(zero_point, rel=1e-3)
    assert np.trapezoid(density, nu) == pytest.approx(3.0, rel=0.01)  # three modes in the one-atom cell

    report = _report(tmp_path / "report", "atomate2-phonons", phonon_document)
    assert report["zero_point_energy_eV_per_cell"] == pytest.approx(zero_point)
    assert report["formula_units_per_phonon_cell"] == 1 and report["non_analytical_correction"] is False
    at_300 = report["at_300K_per_formula_unit"]
    assert at_300["entropy_J_per_K_mol"] == output["entropies"][output["temperatures"].index(300)]
    # A temperature range below 300 K: no 300 K values are made up from the end point.
    cold = json.loads(json.dumps(phonon_document))
    for name in ("temperatures", "free_energies", "internal_energies", "entropies", "heat_capacities"):
        cold["output"][name] = cold["output"][name][: output["temperatures"].index(200) + 1]
    report = _report(tmp_path / "cold", "atomate2-phonons", cold)
    assert "at_300K_per_formula_unit" not in report and "0-200 K" in report["at_300K_note"]
    # Dynamic stability tolerates acoustic-sum-rule noise near Γ, not a real imaginary mode.
    for frequency, stable in ((-0.01, True), (-0.5, False)):
        noisy = json.loads(json.dumps(phonon_document))
        noisy["output"]["phonon_bandstructure"]["bands"][0][0] = frequency  # branch 0, first q-point
        report = _report(tmp_path / f"noise{frequency}", "atomate2-phonons", noisy)
        assert report["lowest_frequency_THz"] == frequency and report["imaginary_tolerance_THz"] == 0.05
        assert report["dynamically_stable"] is stable, frequency
    plots, _ = _postprocess(tmp_path / "plots", "atomate2-phonons", phonon_document, "plots")
    assert {path.name for path in plots.iterdir()} == {
        "phonon_band_structure.png",
        "phonon_dos.png",
        "phonon_band_structure_dos.png",
        "phonon_thermodynamics.png",
    }


def test_elastic_hook_reads_the_raw_tensor(tmp_path: Path, test_profile: TestProfile) -> None:
    if not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    from atomate2.vasp.flows.elastic import ElasticMaker

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("HTTK_FAKE_VASP_REFERENCE_A", "4.05")
        (tmp_path / "flow").mkdir()
        configure_fake_vasp(monkeypatch, tmp_path / "flow")
        document = flow_document(ElasticMaker().make(aluminium()))
    (tmp_path / "collect").mkdir()
    outputs = _collect(tmp_path / "collect", "atomate2-elastic", document)

    stiffness = np.array(outputs["elastic_tensor"].elastic_tensor).reshape(6, 6)  # the typed record is flat
    assert stiffness[0, 0] == pytest.approx(fake.C11, abs=1e-3)
    assert stiffness[0, 1] == pytest.approx(fake.C12, abs=1e-3)
    assert stiffness[3, 3] == pytest.approx(fake.C44, abs=1e-3)
    assert outputs["bulk_modulus_voigt"].bulk_modulus_voigt == pytest.approx((fake.C11 + 2 * fake.C12) / 3, abs=1e-3)


def _elastic_document(raw: np.ndarray) -> dict:
    return {"output": {"structure": aluminium().as_dict(), "elastic_tensor": {"raw": raw.tolist()}}}


def test_elastic_hook_keeps_only_the_tensors_of_an_unstable_fit(tmp_path: Path) -> None:
    unstable = np.diag([100.0, 100.0, 100.0, 30.0, 30.0, -5.0])  # a negative shear stiffness
    outputs = _hook(tmp_path, "atomate2-elastic", _elastic_document(unstable))
    assert set(outputs) == {"relaxed_structure", "elastic_tensor", "compliance_tensor", "task_document"}
    assert np.allclose(_value(outputs["elastic_tensor"]), unstable)

    report = _report(tmp_path / "report", "atomate2-elastic", _elastic_document(unstable))
    assert report["born_stable_zero_stress"] is False and min(report["stiffness_eigenvalues_GPa"]) == pytest.approx(
        -5.0
    )
    assert report["max_abs_equilibrium_stress_GPa"] is None  # the document carries no equilibrium stress
    assert "residual stress" in (_report_dir(tmp_path) / "report.txt").read_text(encoding="utf-8")
    plots, stdout = _postprocess(tmp_path / "plots", "atomate2-elastic", _elastic_document(unstable), "plots")
    assert {path.name for path in plots.iterdir()} == {
        "elastic_tensor.png",
        "youngs_modulus.png",
        "youngs_modulus.json",
    }
    assert "skipped stress_strain.png" in stdout


def test_elastic_hook_warns_about_an_asymmetric_fit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    stiffness = np.diag([100.0, 100.0, 100.0, 30.0, 30.0, 30.0])
    stiffness[0, 1], stiffness[1, 0] = 50.0, 48.0  # 2 GPa apart: above the 0.5 GPa floor
    with caplog.at_level("WARNING"):
        outputs = _hook(tmp_path, "atomate2-elastic", _elastic_document(stiffness))
    assert "asymmetric by 2 GPa" in caplog.text
    assert np.reshape(outputs["elastic_tensor"].elastic_tensor, (6, 6))[0, 1] == pytest.approx(49.0)


def test_dielectric_hook_rejects_born_charges_of_another_cell(tmp_path: Path, dielectric_document: dict) -> None:
    document = json.loads(json.dumps(dielectric_document))
    born = document["output"]["calcs_reversed"][0]["output"]["dielectric_properties"]["born_charges"]
    del born[-1]
    with pytest.raises(ValueError, match="7 Born charge tensors for 8 sites"):
        _hook(tmp_path, "atomate2-dielectric", document)


def test_phonon_hook_converts_per_formula_unit_for_two_formula_units(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, test_profile: TestProfile
) -> None:
    if not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    pytest.importorskip("phonopy")
    from atomate2.vasp.flows.phonons import PhononMaker
    from pymatgen.core import Structure
    from pymatgen.phonon.dos import PhononDos

    primitive = silicon().get_primitive_structure()
    document = _document(tmp_path_factory, lambda: PhononMaker(born_maker=None, min_length=8.0).make(primitive))
    outputs = _collect(tmp_path, "atomate2-phonons", document)

    output = document["output"]
    cell = Structure.from_dict(output["phonon_bandstructure"]["structure"])
    assert cell.composition.get_reduced_composition_and_factor()[1] == 2
    thermodynamics = _value(outputs["vibrational_thermodynamics"])
    assert isinstance(thermodynamics, dict)
    # Independently of the hook: pymatgen's structure-less DOS integrals are per phonopy cell, in J/mol.
    dos = PhononDos.from_dict(output["phonon_dos"])
    to_ev = 1 / (6.02214076e23 * 1.602176634e-19)
    for index, temperature in enumerate(thermodynamics["temperatures"]):
        if temperature < 100:
            continue
        free_energy = thermodynamics["helmholtz_free_energies"][index]
        assert free_energy == pytest.approx(dos.helmholtz_free_energy(temperature) * to_ev, rel=1e-3, abs=1e-6)
        assert thermodynamics["entropies"][index] == pytest.approx(dos.entropy(temperature) * to_ev, rel=1e-3)
    density = _value(outputs["phonon_density_of_states"])
    assert isinstance(density, dict)
    assert np.trapezoid(density["density"], density["frequencies"]) == pytest.approx(3 * len(cell), rel=0.01)

    report = _report(tmp_path / "report", "atomate2-phonons", document)
    assert report["formula_units_per_phonon_cell"] == 2
    assert report["zero_point_energy_eV_per_cell"] == pytest.approx(_value(outputs["zero_point_energy"]))


def test_physics_hooks_share_their_helpers() -> None:
    blocks = set()
    for package in _PACKAGES:
        text = (REPO_ROOT / package / "collect.py").read_text(encoding="utf-8")
        match = re.search(r"# --- shared helpers.*?# --- end of shared helpers ---", text, re.DOTALL)
        assert match, package
        blocks.add(match.group(0))
    assert len(blocks) == 1
    # The structure, port and sub-job helpers are also the structure/energy hook's, the structure ones the parent's.
    every = {"_sanitized", "_normalized", "_structure_view"}
    packages = every | {"_declared_roles", "_port_document", "_port_structure", "_with_inputs", "_child_task_document"}
    sources: dict[str, set[str | None]] = {}
    for path in REPO_ROOT.glob("atomate2-*/collect.py"):
        text = path.read_text(encoding="utf-8")
        functions = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        expected = every if path.parent.name == "atomate2-full-characterization" else packages
        assert expected <= set(functions), path.parent.name
        for name in expected:
            sources.setdefault(name, set()).add(ast.get_source_segment(text, functions[name]))
    assert {name: len(found) for name, found in sources.items()} == dict.fromkeys(packages, 1)


def test_structure_view_drops_only_physics_free_site_properties() -> None:
    hook = runpy.run_path(str(REPO_ROOT / "atomate2-full-characterization" / "collect.py"))
    plain: Any = silicon()

    def with_properties(**properties: list) -> dict:
        structure = plain.copy()
        for name, values in properties.items():
            structure.add_site_property(name, values)
        return structure.as_dict()

    reference = content_id(hook["_structure_view"](plain.as_dict()))
    zero = with_properties(magmom=[-0.0, 0.0] * 4, phonopy_masses=[28.0855] * 8)
    assert hook["_normalized"](zero).site_properties == {}
    assert content_id(hook["_structure_view"](zero)) == reference
    magnetic = with_properties(magmom=[1.0, -1.0] * 4, phonopy_masses=[28.0855] * 8)
    assert hook["_normalized"](magnetic).site_properties == {"magmom": [1.0, -1.0] * 4}
    assert content_id(hook["_structure_view"](magnetic)) != reference


def test_structure_view_never_decodes_job_written_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pymatgen.core import Structure

    hook = runpy.run_path(str(REPO_ROOT / "atomate2-full-characterization" / "collect.py"))
    plain: Any = silicon()
    # The sanitized document is the same structure: the content id is the one the unsanitized document gives.
    reference = content_id(UnitcellStructureView(Structure.from_dict(plain.as_dict())))
    assert content_id(hook["_structure_view"](plain.as_dict())) == reference
    # An importable module whose from_dict records that it ran: monty's decoder would import and call it.
    (tmp_path / "gadget_probe.py").write_text(
        "class Probe:\n    @classmethod\n    def from_dict(cls, d):\n        raise SystemExit('decoded')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "gadget_probe", raising=False)
    gadget = {"@module": "gadget_probe", "@class": "Probe"}
    document = plain.as_dict()
    document["properties"] = {"x": gadget}
    for site in document["sites"]:
        site["properties"] = {"magmom": -0.0, "x": gadget}
        site["species"][0]["x"] = gadget
    assert content_id(hook["_structure_view"](document)) == reference
    assert "gadget_probe" not in sys.modules
    with pytest.raises(SystemExit, match="decoded"):  # the control: unsanitized, the document is decoded
        Structure.from_dict(document)
    # A species' spin is kept: cells that differ only in it stay distinct.
    spins = [plain.as_dict(), plain.as_dict()]
    for spin, spinned in zip((2, 4), spins, strict=True):
        spinned["sites"][0]["species"][0].update(oxidation_state=2, spin=spin)
    assert hook["_sanitized"](spins[0])["sites"][0]["species"] == [
        {"element": "Si", "occu": 1, "oxidation_state": 2, "spin": 2}
    ]
    assert content_id(hook["_structure_view"](spins[0])) != content_id(hook["_structure_view"](spins[1]))
    document["sites"][0]["abc"] = ["0", 0, 0]
    with pytest.raises(ValueError, match="expected a number"):
        hook["_sanitized"](document)


def test_phonon_cell_is_the_input_only_when_it_is_the_same_cell() -> None:
    same_cell = runpy.run_path(str(REPO_ROOT / "atomate2-phonons" / "collect.py"))["_same_cell"]
    plain: Any = silicon()
    noisy = plain.copy()
    noisy.lattice = noisy.lattice.__class__(plain.lattice.matrix + 4e-16)
    noisy.add_site_property("phonopy_masses", [28.0855] * 8)
    assert same_cell(noisy.as_dict(), plain.as_dict())
    # Oxidation states and the cell's charge are part of the cell.
    ionic = plain.copy()
    ionic.add_oxidation_state_by_site([2, -2] * 4)
    assert not same_cell(ionic.as_dict(), plain.as_dict())
    charged = plain.as_dict()
    charged["charge"] = 1.0
    assert not same_cell(charged, plain.as_dict())


def _postprocess_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The shared postprocess module, imported in-process without writing bytecode into the package."""

    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    path = REPO_ROOT / "atomate2-static" / "scripts" / "atomate2_postprocess.py"
    spec = importlib.util.spec_from_file_location("atomate2_postprocess", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


def _symmetric(voigt: np.ndarray, shear: float) -> list[list[float]]:
    """A Voigt 6-vector as a symmetric 3x3 tensor; *shear* scales the off-diagonal (1/2 for engineering strain)."""

    xx, yy, zz, yz, xz, xy = voigt
    yz, xz, xy = shear * yz, shear * xz, shear * xy
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


def test_elastic_plots_compute_directional_moduli_and_the_fit_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent checks of youngs_modulus.json and stress_strain.json for an anisotropic, shear-coupled tensor."""

    stiffness = np.array(
        [
            [160.0, 70.0, 60.0, 8.0, -5.0, 0.0],
            [70.0, 150.0, 65.0, 4.0, 3.0, 0.0],
            [60.0, 65.0, 140.0, 0.0, 6.0, 2.0],
            [8.0, 4.0, 0.0, 50.0, 0.0, 3.0],
            [-5.0, 3.0, 6.0, 0.0, 45.0, 1.0],
            [0.0, 0.0, 2.0, 3.0, 1.0, 40.0],
        ]
    )
    assert np.linalg.eigvalsh(stiffness).min() > 0
    equilibrium = np.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.4])  # GPa, Voigt
    strains, stresses = [], []
    for component in range(6):
        for magnitude in (-0.01, -0.005, 0.005, 0.01):
            strain = np.zeros(6)
            strain[component] = magnitude  # engineering shear, as Voigt strains are
            strains.append(_symmetric(strain, 0.5))
            stresses.append(_symmetric(equilibrium + stiffness @ strain, 1.0))
    document = {
        "structure": aluminium().as_dict(),
        "elastic_tensor": {"raw": stiffness.tolist()},
        "fitting_data": {"strains": strains, "pk_stresses": stresses},
        "eq_stress": _symmetric(equilibrium, 1.0),
    }
    _postprocess_module(monkeypatch)._plots_elastic(document, None, tmp_path)

    # 1/E(n) = v^T S v with v = (nx², ny², nz², ny nz, nx nz, nx ny) and S the Voigt compliance.
    youngs = json.loads((tmp_path / "youngs_modulus.json").read_text(encoding="utf-8"))
    compliance, angles = np.linalg.inv(stiffness), np.radians(youngs["angles_deg"])
    for plane, (first, second) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        n = np.zeros((len(angles), 3))
        n[:, first], n[:, second] = np.cos(angles), np.sin(angles)
        v = np.stack(
            [n[:, 0] ** 2, n[:, 1] ** 2, n[:, 2] ** 2, n[:, 1] * n[:, 2], n[:, 0] * n[:, 2], n[:, 0] * n[:, 1]], 1
        )
        expected = 1 / np.einsum("ni,ij,nj->n", v, compliance, v)
        assert youngs["youngs_modulus_GPa"][plane] == pytest.approx(expected, rel=1e-9), plane

    # Each plotted point lies on its fit line, sigma_eq + C[:, j] * strain_j.
    data = json.loads((tmp_path / "stress_strain.json").read_text(encoding="utf-8"))
    assert data["line_intercept_GPa"] == pytest.approx(equilibrium)
    for column, (label, group) in enumerate(data["points"].items()):
        assert label == ["xx", "yy", "zz", "yz", "xz", "xy"][column]
        assert group["line_slope_GPa"] == pytest.approx(stiffness[:, column])
        assert sorted(group["strain"]) == pytest.approx([-0.01, -0.005, 0.005, 0.01])
        for strain, stress in zip(group["strain"], group["stress_voigt_GPa"], strict=True):
            assert stress == pytest.approx(equilibrium + stiffness[:, column] * strain), label


def test_relaxation_initial_structure_decodes_literal_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A literal structure input (the jobflow runner's ``kind = "value"``) is the initial structure; else, labelled."""

    module = _postprocess_module(monkeypatch)
    initial, final = silicon(), aluminium()
    task = {"input": {"structure": final.as_dict()}}
    literal = {"jobflow_inputs": {"structure": {"kind": "value", "value": initial.as_dict()}}}
    job = module.Job(Path("/nonexistent"), {"parameters": literal}, None, None)
    structure, source = module._initial_structure(job, task)
    assert structure == initial and source == "job input"
    structure, source = module._initial_structure(module.Job(Path("/nonexistent"), {}, None, None), task)
    assert structure == final and source == "input of the last calculation in the task document"


def test_full_characterization_fails_for_an_unloadable_succeeded_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A succeeded child that cannot be loaded is a failed analysis; a failed child is only listed."""

    module = _postprocess_module(monkeypatch)
    Workspace.initialize(tmp_path / "ws")
    monkeypatch.setenv("HTTK_WORKFLOW_WORKSPACE_DIR", str(tmp_path / "ws"))
    parent = tmp_path / "parent"
    (parent / "data").mkdir(parents=True)
    (parent / "job.json").write_text(json.dumps({"workflow": {"name": "atomate2.full-characterization"}}))
    parts = {
        "elastic": {"kind": "succeeded", "job_id": "00000000-0000-4000-8000-000000000000"},
        "dielectric": {"kind": "failed", "job_id": "x", "failure": {"code": "c", "message": "m"}},
    }
    (parent / "data" / "characterization.json").write_text(json.dumps({"parts": parts, "decisions": {}}))
    job = module.Job.at(parent)
    for script in ("report", "plots"):
        (tmp_path / script).mkdir()
        with pytest.raises(module.AnalysisError, match="elastic load"):
            module._run(script, job, tmp_path / script)
    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["failed_analyses"] == ["elastic load"]
    assert report["children"]["dielectric"]["state"] == "failed (c: m)"


def test_latex_report_escapes_job_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """``atomate2_latex`` imports beside the shared module, as its entry point does, and escapes every TeX special."""

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "atomate2-full-characterization" / "scripts"))
    monkeypatch.delitem(sys.modules, "atomate2_latex", raising=False)
    latex = importlib.import_module("atomate2_latex")
    assert latex._tex("\\ & % $ # _ { } ~ ^") == (
        r"\textbackslash{} \& \% \$ \# \_ \{ \} \textasciitilde{} \textasciicircum{}"
    )
    assert latex._tex("a<b|c>ε") == r"a\textless{}b\textbar{}c\textgreater{}?"
    # pdflatex refuses control characters; a line break inside a table cell would end its paragraph.
    assert latex._tex("a\x00b\x08c\x0bd\x1fe\x7ff\tg\nh\r") == "a?b?c?d?e?f\tg h "
    assert latex._label("\\Gamma-X") == latex._label("GAMMA-X") == r"$\Gamma$-X"
    assert latex._label(None) == "--"
    assert (latex._num(None), latex._num(True), latex._num(-0.0001, 2), latex._num(-1.25, 1)) == (
        "--",
        "yes",
        "0.00",
        r"\textminus{}1.2",
    )


def test_full_characterization_refuses_a_child_label_that_leaves_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child labels name output directories: a job-written ``../x`` is a failed analysis, never a path."""

    module = _postprocess_module(monkeypatch)
    Workspace.initialize(tmp_path / "ws")
    monkeypatch.setenv("HTTK_WORKFLOW_WORKSPACE_DIR", str(tmp_path / "ws"))
    parent = tmp_path / "parent"
    (parent / "data").mkdir(parents=True)
    (parent / "job.json").write_text(json.dumps({"workflow": {"name": "atomate2.full-characterization"}}))
    parts = {"../x": {"kind": "succeeded", "job_id": "00000000-0000-4000-8000-000000000000"}}
    (parent / "data" / "characterization.json").write_text(json.dumps({"parts": parts, "decisions": {}}))
    out = tmp_path / "out" / "report"
    out.mkdir(parents=True)
    with pytest.raises(module.AnalysisError, match="invalid label"):
        module._run("report", module.Job.at(parent), out)
    assert not (tmp_path / "out" / "x").exists()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["failed_analyses"] == ["../x (invalid label)"] and report["children"] == {}
