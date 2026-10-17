"""Each package loads as a jobflow package of its Maker and matches its declaration.

The four structure/energy packages share one collect hook; the physics packages share
helper code, pinned in ``test_physics_hooks.py``.
"""

import importlib
import json
from pathlib import Path

import pytest
from httk.core.plugins.manifest import parse_plugin_manifest
from httk.workflow.packages import load_workflow_package

REPO_ROOT = Path(__file__).resolve().parent.parent
_MAKERS = {
    "atomate2-static": ("atomate2.static", "atomate2.vasp.jobs.core:StaticMaker"),
    "atomate2-relax": ("atomate2.relax", "atomate2.vasp.jobs.core:RelaxMaker"),
    "atomate2-tight-relax": ("atomate2.tight-relax", "atomate2.vasp.jobs.core:TightRelaxMaker"),
    "atomate2-double-relax": ("atomate2.double-relax", "atomate2.vasp.flows.core:DoubleRelaxMaker"),
    "atomate2-band-structure": ("atomate2.band-structure", "atomate2.vasp.flows.core:BandStructureMaker"),
    "atomate2-dielectric": ("atomate2.dielectric", "atomate2.vasp.jobs.core:DielectricMaker"),
    "atomate2-elastic": ("atomate2.elastic", "atomate2.vasp.flows.elastic:ElasticMaker"),
    "atomate2-phonons": ("atomate2.phonons", "atomate2.vasp.flows.phonons:PhononMaker"),
}
_DIRECTORIES = tuple(_MAKERS)
_STRUCTURE_ENERGY = ("atomate2-static", "atomate2-relax", "atomate2-tight-relax", "atomate2-double-relax")
#: Declared Maker parameters: only these are passed to the Maker, and only when given.
_PARAMETERS = {"atomate2-phonons": {"min_length", "born_maker", "use_symmetrized_structure"}}


@pytest.mark.parametrize("directory", _DIRECTORIES)
def test_package_is_a_jobflow_package_of_its_maker(directory: str) -> None:
    provider = load_workflow_package(REPO_ROOT / directory, register=False)
    assert provider.workflow_id == _MAKERS[directory][0]
    assert provider.runner_options == {"maker": _MAKERS[directory][1]}
    # The declared [workflow.collect] hook replaces the jobflow default collector.
    assert provider.collect_file == "collect.py"
    assert callable(provider.collector)
    assert provider.inputs == {"structure": None}
    assert set(provider.parameters) == _PARAMETERS.get(directory, set())
    # No resource table ships: sites differ (the README shows one).
    assert "resources" not in provider.runner_options


@pytest.mark.parametrize("directory", _DIRECTORIES)
def test_package_maker_imports(directory: str) -> None:
    pytest.importorskip("atomate2")
    module, _, name = _MAKERS[directory][1].partition(":")
    assert isinstance(getattr(importlib.import_module(module), name), type)


@pytest.mark.parametrize("directory", _DIRECTORIES)
def test_package_declaration_matches_the_committed_file_and_manifest(directory: str) -> None:
    provider = load_workflow_package(REPO_ROOT / directory, register=False)
    declared = json.loads((REPO_ROOT / directory / "declaration.json").read_text(encoding="utf-8"))
    assert provider.declarations["workflow"] == declared
    outputs = {output["name"]: output for output in declared["outputs"]}
    assert set(outputs) == {metadata["role"] for metadata in provider.outputs.values()}
    for metadata in provider.outputs.values():
        output = outputs[metadata["role"]]
        assert output["entry_type"] == metadata["entry_type"]
        assert output.get("ref") == metadata.get("ref")
        assert output["description"] == metadata["description"]
        assert metadata["product_of"] in {"initial_structure", *outputs}
    (structure,) = declared["inputs"]
    assert structure["name"] == "initial_structure" and structure["entry_type"] == "structures"


def test_plugin_manifest_lists_exactly_the_package_directories() -> None:
    manifest = parse_plugin_manifest(REPO_ROOT)
    assert set(manifest.workflows) == set(_DIRECTORIES)
    packages = {path.parent.name for path in REPO_ROOT.glob("*/httk_workflow.toml")}
    assert packages == set(_DIRECTORIES)


def test_structure_energy_collect_hooks_are_identical() -> None:
    hooks = {(REPO_ROOT / directory / "collect.py").read_bytes() for directory in _STRUCTURE_ENERGY}
    assert len(hooks) == 1
