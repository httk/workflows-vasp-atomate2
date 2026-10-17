"""The fake VASP and fake POTCAR library satisfy pymatgen, custodian, emmet and atomate2."""

import inspect
import json
import subprocess
from pathlib import Path

import pytest

from conftest import FAKE_PSP, FAKE_VASP

FAKE_ENV = {"ATOMATE2_VASP_CMD": FAKE_VASP, "ATOMATE2_VASP_GAMMA_CMD": FAKE_VASP, "PMG_VASP_PSP_DIR": FAKE_PSP}

# conftest.py exported the fake before atomate2 was first imported, so ``run_vasp``'s
# defaults name it. pymatgen instead builds one SETTINGS dict lazily and shares it, so the
# ``fake_vasp`` fixture patches that dict in place, which works whatever imported pymatgen
# first.
pytest.importorskip("atomate2")

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.run import run_vasp
from atomate2.vasp.sets.core import StaticSetGenerator
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import SETTINGS, Lattice, Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun


def _silicon() -> Structure:
    """Conventional diamond Si, 8 atoms."""

    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def fake_vasp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point atomate2 and pymatgen at the fakes and run in *tmp_path*."""

    defaults = inspect.signature(run_vasp).parameters
    if defaults["vasp_cmd"].default != FAKE_VASP or defaults["vasp_gamma_cmd"].default != FAKE_VASP:
        pytest.fail("atomate2 was imported before conftest.py set ATOMATE2_VASP_CMD")
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    for key in list(SETTINGS):
        monkeypatch.delitem(SETTINGS, key)
    monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", FAKE_PSP)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _invocations(root: Path) -> list[dict]:
    return [json.loads(line) for log in sorted(root.rglob("fake-vasp-invocations.jsonl")) for line in log.open()]


def test_outputs_parse(fake_vasp: Path) -> None:
    StaticSetGenerator().get_input_set(_silicon()).write_input(fake_vasp)  # NSW = 0
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
    flow = StaticMaker().make(_silicon())
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    task = responses[flow.uuid][1].output
    assert isinstance(task, TaskDoc)
    assert task.output.energy is not None
    [invocation] = _invocations(fake_vasp)
    assert invocation["argv"] == [FAKE_VASP]
    assert invocation["env"]["ATOMATE2_VASP_CMD"] == FAKE_VASP


def test_double_relax_maker_runs_locally(fake_vasp: Path) -> None:
    silicon = _silicon()
    flow = DoubleRelaxMaker().make(silicon)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    assert len(_invocations(fake_vasp)) == 2
    task = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(task, TaskDoc)
    assert task.output.structure != silicon
