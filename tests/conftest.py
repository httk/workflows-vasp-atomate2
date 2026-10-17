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
