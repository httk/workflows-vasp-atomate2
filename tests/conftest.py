"""Shared fixtures for the workflows-vasp-atomate2 test suite.

Ported from workflows-vasp's ``tests/conftest.py``: config isolation and the
normal/extended test-depth knob, the workspace-registry helper and the
``httk.workflow`` logger restoration.
"""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from httk.workflow.registry import register_workspace

for _thread_limit in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_limit, "1")

FAKE_VASP = str(Path(__file__).resolve().parent / "fake_vasp_atomate2.py")
FAKE_PSP = str(Path(__file__).resolve().parent / "fake_psp")

# atomate2 reads ATOMATE2_* once, when first imported, and binds SETTINGS.VASP_CMD as the
# default argument of ``run_vasp``. The in-process Maker tests need that to be the fake, so
# it is exported here, before any test module imports atomate2, with every developer
# setting removed. The autouse fixture below removes the variables again for each test:
# the processes a test starts never inherit them.
for _key in [key for key in os.environ if key.startswith(("ATOMATE2_", "PMG_"))]:
    del os.environ[_key]
os.environ.update({"ATOMATE2_VASP_CMD": FAKE_VASP, "ATOMATE2_VASP_GAMMA_CMD": FAKE_VASP})


@pytest.fixture(autouse=True)
def _isolated_httk_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own httk config and data home, and no ambient VASP configuration.

    A developer's own ``HTTK_VASP_COMMAND``, ``ATOMATE2_*`` and ``PMG_*``
    variables never reach a test: a test that needs them sets them itself.
    """

    monkeypatch.setenv("HTTK_CONFIG_HOME", str(tmp_path_factory.mktemp("httk-config")))
    monkeypatch.setenv("HTTK_DATA_HOME", str(tmp_path_factory.mktemp("httk-store")))
    # A machine exporting a real VASP command must never run it from a test.
    monkeypatch.delenv("HTTK_VASP_COMMAND", raising=False)
    # The fake VASP's elastic reference lattice is set by the tests that need it.
    monkeypatch.delenv("HTTK_FAKE_VASP_REFERENCE_A", raising=False)
    for knob in ("HTTK_FAKE_VASP_FAIL_ON", "HTTK_FAKE_VASP_IONIC_BORN", "HTTK_FAKE_VASP_MAGNETIZATION"):
        monkeypatch.delenv(knob, raising=False)
    for key in list(os.environ):
        if key.startswith(("ATOMATE2_", "PMG_")):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _isolated_workflow_logging() -> Iterator[None]:
    """Restore the ``httk.workflow`` logger after every test."""

    logger = logging.getLogger("httk.workflow")
    propagate = logger.propagate
    handlers = list(logger.handlers)
    level = logger.level
    yield
    logger.propagate = propagate
    logger.handlers[:] = handlers
    logger.setLevel(level)


def register_ws(path: object, name: str = "ws") -> str:
    """Register *path* under *name* and return the name."""

    register_workspace(name, str(path))
    return name


def silicon() -> object:
    """Conventional diamond Si, 8 atoms, a = 5.43 Angstrom (a pymatgen ``Structure``)."""

    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])


def aluminium() -> object:
    """Primitive fcc Al, 1 atom, a = 4.05 Angstrom (a pymatgen ``Structure``)."""

    from pymatgen.core import Lattice, Structure

    fcc = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    return Structure(Lattice.cubic(4.05), ["Al"] * 4, fcc).get_primitive_structure()


def configure_fake_vasp(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Point in-process atomate2 and pymatgen at the fakes and run in *directory*.

    pymatgen builds one SETTINGS dict lazily and shares it, so it is patched in place,
    which works whatever imported pymatgen first. custodian polls its job every 10 s
    by default; the fake finishes in well under a second, so in-process runs poll faster.
    """

    import functools
    import inspect

    import atomate2.vasp.run
    from custodian import Custodian
    from pymatgen.core import SETTINGS

    defaults = inspect.signature(atomate2.vasp.run.run_vasp).parameters
    if defaults["vasp_cmd"].default != FAKE_VASP or defaults["vasp_gamma_cmd"].default != FAKE_VASP:
        pytest.fail("atomate2 was imported before conftest.py set ATOMATE2_VASP_CMD")
    for key, value in {"ATOMATE2_VASP_CMD": FAKE_VASP, "ATOMATE2_VASP_GAMMA_CMD": FAKE_VASP}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PMG_VASP_PSP_DIR", FAKE_PSP)
    for key in list(SETTINGS):
        monkeypatch.delitem(SETTINGS, key)
    monkeypatch.setitem(SETTINGS, "PMG_VASP_PSP_DIR", FAKE_PSP)
    monkeypatch.setattr(
        atomate2.vasp.run, "Custodian", functools.partial(Custodian, polling_time_step=0.1, monitor_freq=1)
    )
    monkeypatch.chdir(directory)


@pytest.fixture
def fake_vasp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point atomate2 and pymatgen at the fakes and run in *tmp_path*."""

    configure_fake_vasp(monkeypatch, tmp_path)
    return tmp_path


def flow_document(flow: object) -> dict:
    """Run *flow* in-process and return the ``jobflow-outputs.json`` document httk's runner writes for it.

    As httk's jobflow runner, the flow's output references are resolved with the data
    store's blobs loaded, and the result is sanitized to plain JSON values.
    """

    from jobflow import JobStore, run_locally
    from jobflow.core.reference import find_and_resolve_references
    from maggma.stores import MemoryStore
    from monty.json import jsanitize

    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    run_locally(flow, store=store, create_folders=True, ensure_success=True)
    output = find_and_resolve_references(cast(Any, flow).output, store)
    return {"output": jsanitize(output, strict=True, enum_values=True)}


@dataclass(frozen=True)
class TestProfile:
    """Select normal or full-depth values without duplicating a test body."""

    __test__ = False  # imported by test modules; not a test class
    name: str

    @property
    def extended(self) -> bool:
        return self.name == "extended"


@pytest.fixture(scope="session", autouse=True)
def test_profile() -> TestProfile:
    """The test-depth knob shared by every profiled test.

    Normal is deliberately the default for direct pytest invocations. Extended
    runs set ``HTTK_TEST_PROFILE=extended``.
    """

    name = os.environ.get("HTTK_TEST_PROFILE", "normal")
    if name not in {"normal", "extended"}:
        raise pytest.UsageError("HTTK_TEST_PROFILE must be 'normal' or 'extended'")
    return TestProfile(name)
