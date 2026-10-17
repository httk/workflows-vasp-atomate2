"""The fake VASP and fake POTCAR library satisfy pymatgen, custodian, emmet and atomate2.

The physics flows (band structure, dielectric, elastic, phonons) also check that the
synthetic data comes back as atomate2 and emmet report it, including the component order.
"""

import json
import subprocess
from pathlib import Path

import pytest

from conftest import FAKE_PSP, FAKE_VASP, TestProfile, aluminium, silicon

# conftest.py exported the fake before atomate2 was first imported, so ``run_vasp``'s
# defaults name it; its ``fake_vasp`` fixture points pymatgen at the fake POTCAR library.
pytest.importorskip("atomate2")

import numpy as np
from atomate2.vasp.flows.core import BandStructureMaker, DoubleRelaxMaker
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from emmet.core.tasks import TaskDoc
from emmet.core.vasp.calculation import VaspObject
from jobflow import run_locally
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Outcar, Vasprun

import fake_vasp_atomate2 as fake


def _outputs(responses: dict) -> list:
    """Every job output of a ``run_locally`` result."""

    return [response.output for indexed in responses.values() for response in indexed.values()]


def _invocations(root: Path) -> list[dict]:
    return [json.loads(line) for log in sorted(root.rglob("fake-vasp-invocations.jsonl")) for line in log.open()]


def test_outputs_parse(fake_vasp: Path) -> None:
    StaticSetGenerator().get_input_set(silicon()).write_input(fake_vasp)  # NSW = 0
    subprocess.run([FAKE_VASP], cwd=fake_vasp, check=True, capture_output=True)

    vasprun = Vasprun(str(fake_vasp / "vasprun.xml"))
    assert vasprun.converged
    assert vasprun.final_energy < 0
    assert vasprun.efermi is not None
    assert vasprun.eigenvalue_band_properties[0] == pytest.approx(1.0)
    assert vasprun.potcar_spec
    Outcar(str(fake_vasp / "OUTCAR"))
    task = TaskDoc.from_directory(fake_vasp)
    assert task.output.structure is not None
    assert task.output.energy is not None
    assert task.output.bandgap == pytest.approx(1.0)
    [invocation] = _invocations(fake_vasp)
    assert invocation["env"] == {"ATOMATE2_VASP_CMD": FAKE_VASP, "PMG_VASP_PSP_DIR": FAKE_PSP}
    assert invocation["incar"]["NSW"] == 0


def test_static_maker_runs_locally(fake_vasp: Path) -> None:
    flow = StaticMaker().make(silicon())
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    task = responses[flow.uuid][1].output
    assert isinstance(task, TaskDoc)
    assert task.output.energy is not None
    [invocation] = _invocations(fake_vasp)
    assert invocation["argv"] == [FAKE_VASP]
    assert invocation["env"]["ATOMATE2_VASP_CMD"] == FAKE_VASP


def test_double_relax_maker_runs_locally(fake_vasp: Path) -> None:
    structure = silicon()
    flow = DoubleRelaxMaker().make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    assert len(_invocations(fake_vasp)) == 2
    task = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(task, TaskDoc)
    assert task.output.structure != structure


def test_band_structure_maker_runs_locally(fake_vasp: Path) -> None:
    responses = run_locally(BandStructureMaker().make(silicon()), create_folders=True, ensure_success=True)

    tasks = {task.task_label: task for task in _outputs(responses) if isinstance(task, TaskDoc)}
    assert set(tasks) == {"static", "non-scf uniform", "non-scf line"}
    line = tasks["non-scf line"].vasp_objects[VaspObject.BANDSTRUCTURE]
    assert {"\\Gamma", "X", "L"} <= set(line.labels_dict)
    assert line.get_band_gap()["energy"] == pytest.approx(1.0)
    # The bands disperse: the top valence band is 0.4 eV lower at the zone boundary than at Gamma.
    assert np.ptp(line.bands[Spin.up][line.nb_bands // 2 - 1]) > 0.1
    dos = tasks["non-scf uniform"].vasp_objects[VaspObject.DOS]
    assert dos.efermi == pytest.approx(fake.EFERMI)
    assert dos.get_gap() == pytest.approx(1.0, abs=0.1)
    assert len(_invocations(fake_vasp)) == 3


def test_dielectric_maker_reports_the_synthetic_tensors_in_vasp_order(fake_vasp: Path) -> None:
    responses = run_locally(DielectricMaker().make(silicon()), create_folders=True, ensure_success=True)

    (task,) = [task for task in _outputs(responses) if isinstance(task, TaskDoc)]
    output = task.calcs_reversed[0].output
    assert np.allclose(output.epsilon_static, fake.epsilon_electronic())
    assert np.allclose(output.epsilon_ionic, fake.epsilon_ionic())
    dielectric = output.dielectric_properties
    assert np.allclose(dielectric.dielectric_tensor, fake.epsilon_electronic())
    assert np.allclose(dielectric.dielectric_ionic_tensor, fake.epsilon_ionic())
    # Born charges: one 3x3 matrix per site, rows and columns exactly as VASP prints them.
    assert np.allclose(dielectric.born_charges, fake.born_charges(8))
    assert dielectric.born_charges[0][0] == pytest.approx([1.11, 1.12, 1.13])
    # Piezoelectric tensors: rows x y z, columns in VASP's printed order XX YY ZZ XY YZ ZX, not reordered.
    assert np.allclose(dielectric.piezo_tensor, fake.piezo(ionic=False))
    assert dielectric.piezo_tensor[0] == pytest.approx([0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    assert np.allclose(dielectric.piezo_ionic_tensor, fake.piezo(ionic=True))


def test_elastic_maker_fits_the_synthetic_constants(
    fake_vasp: Path, monkeypatch: pytest.MonkeyPatch, test_profile: TestProfile
) -> None:
    if not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    monkeypatch.setenv("HTTK_FAKE_VASP_REFERENCE_A", "4.05")
    responses = run_locally(ElasticMaker().make(aluminium()), create_folders=True, ensure_success=True)

    (document,) = [output for output in _outputs(responses) if hasattr(output, "elastic_tensor")]
    expected = np.zeros((6, 6))
    expected[:3, :3] = fake.C12
    expected[np.diag_indices(3)] = fake.C11
    expected[3:, 3:] = np.diag([fake.C44] * 3)
    # The fake writes the stress at full double precision (no kbar rounding) and the stress it
    # writes is exactly linear in the Green-Lagrange strain, so the fit reproduces C up to
    # floating-point error (observed: 2e-6 GPa); 1e-3 GPa leaves a wide margin.
    assert np.allclose(document.elastic_tensor.raw, expected, atol=1e-3)


def test_phonon_maker_gives_spring_model_phonons(fake_vasp: Path, test_profile: TestProfile) -> None:
    if not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    pytest.importorskip("phonopy")
    from atomate2.vasp.flows.phonons import PhononMaker

    maker = PhononMaker(born_maker=None, min_length=8.0)
    responses = run_locally(maker.make(aluminium()), create_folders=True, ensure_success=True)

    (document,) = [output for output in _outputs(responses) if hasattr(output, "phonon_bandstructure")]
    bands = np.array(document.phonon_bandstructure.as_dict()["bands"])  # [mode][qpoint], THz
    qpoints = np.array([qpoint.frac_coords for qpoint in document.phonon_bandstructure.qpoints])
    gamma = np.linalg.norm(qpoints, axis=1) < 1e-8
    assert gamma.any() and np.abs(bands[:, gamma]).max() < 1e-3  # acoustic modes vanish at Gamma
    assert (bands[:, ~gamma] > 0).all() and bands.max() > 1.0
    assert len(document.temperatures) == len(document.free_energies) == len(document.heat_capacities) > 1
