"""The physics packages' collect hooks turn real fake-VASP flow outputs into their declared roles.

Each document is the ``jobflow-outputs.json`` httk's jobflow runner writes, produced by
running the package's Maker in-process on the fake VASP (``conftest.flow_document``).
The values are checked against the fake's synthetic physics, including the component
orders and unit conversions the hooks apply.
"""

import json
import re
import runpy
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")
pytest.importorskip("httk.analyse")

import numpy as np
from httk.analyse import property_records
from httk.atomistic import UnitcellStructureView
from httk.core import DataRecord, FileRecord
from httk.workflow.collecting import JobRecord

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


def _hook(tmp_path: Path, package: str, document: dict) -> dict:
    """Run the package's hook on *document*, as the workdir ``jobflow-outputs.json`` of a finished job."""

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "jobflow-outputs.json").write_text(json.dumps(document), encoding="utf-8")
    declaration = json.loads((REPO_ROOT / package / "declaration.json").read_text(encoding="utf-8"))
    record = JobRecord(
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
    return runpy.run_path(str(REPO_ROOT / package / "collect.py"))["collect"](record)


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


def test_physics_hooks_share_their_helpers() -> None:
    blocks = set()
    for package in _PACKAGES:
        text = (REPO_ROOT / package / "collect.py").read_text(encoding="utf-8")
        match = re.search(r"# --- shared helpers.*?# --- end of shared helpers ---", text, re.DOTALL)
        assert match, package
        blocks.add(match.group(0))
    assert len(blocks) == 1
