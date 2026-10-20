"""atomate2.full-characterization runs the other packages as child jobs, end to end with the fake VASP.

The parent's children are found from their ``job.json`` ``parent`` member, and each fake
VASP run is attributed to the child whose flow ran it through the same chain, so the
test sees which calculations each child ran.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("atomate2")
pytest.importorskip("httk.atomistic")
pytest.importorskip("httk.analyse")
pytest.importorskip("phonopy")

from httk.atomistic import UnitcellStructureView
from httk.workflow import Workspace, collect, job_records
from httk.workflow.collecting import JobRecord
from httk.workflow.introspection import resolve_job
from httk.workflow.packages import load_workflow_package
from httk.workflow.postprocessing import run_postprocess_script
from pymatgen.core import Lattice, Structure

from conftest import FAKE_VASP, TestProfile, aluminium, silicon
from test_wrappers_end_to_end import (  # noqa: F401  (autouse fixture)
    _FIGURES,
    _failure,
    _installed_plugin,
    _plots,
    _report,
    _run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def silicon_carbide() -> Structure:
    """Conventional zincblende SiC, 8 atoms, a = 4.36 Angstrom: a polar insulator to the fake."""

    return Structure.from_spacegroup("F-43m", Lattice.cubic(4.36), ["Si", "C"], [[0, 0, 0], [0.25, 0.25, 0.25]])


# Case -> (structure, the fake's elastic reference lattice parameter, other workspace settings, normal profile?).
# The reference is a little off the structure's own, so the relaxed cell carries a stress.
_CASES = {
    # Non-default atomate2 settings, which the parent's own decisions must use too.
    "metal": (aluminium, 4.00, {"atomate2.bandgap_tol": 0.5, "atomate2.symprec": 0.01}, True),
    # Isotropic, species-dependent Born charges that survive atomate2's symmetrization.
    "polar_insulator": (silicon_carbide, 4.33, {"fake_vasp.ionic_born": 1.5}, False),
    # Only the dielectric run sets LEPSILON.
    "dielectric_failed": (silicon, 5.40, {"fake_vasp.fail_on": "LEPSILON"}, False),
    "magnetic_metal": (aluminium, 4.00, {"fake_vasp.magnetization": 1.0}, False),
}


#: A harmless relax override the metal case passes through ``child_parameters``.
_NELM = {
    "kind": "function",
    "function": "atomate2.vasp.powerups:update_user_incar_settings",
    "kwargs": {"incar_updates": {"NELM": 150}},
}
#: A phonons override of a Maker field ``make()`` reads: the displacement runs' ENCUT. Its ISPIN loses to the
#: parent's non-magnetic ``function`` override, which the runner applies after every ``maker`` override.
_DISPLACEMENT_ENCUT = {
    "kind": "maker",
    "update": {
        "phonon_displacement_maker.input_set_generator.user_incar_settings.ENCUT": 555,
        "phonon_displacement_maker.input_set_generator.user_incar_settings.ISPIN": 2,
    },
}
#: The override the parent gives every child after the relaxation of a non-magnetic structure.
_NO_SPIN = {
    "kind": "function",
    "function": "atomate2.vasp.powerups:update_user_incar_settings",
    "kwargs": {"incar_updates": {"ISPIN": 1}},
}
_NO_SPIN_NOTE = "non-magnetic: the children after the relaxation run without spin polarization (ISPIN = 1)"


def _jobs(workspace: Workspace) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every job of the workspace: job id -> (job directory, job.json)."""

    jobs = {}
    for path in workspace.root.glob("jobs/*/*/job.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        jobs[document["id"]] = (path.parent, document)
    return jobs


def _characterize(
    tmp_path: Path, structure: object, settings: dict[str, object], parameters: dict[str, object] | None = None
) -> tuple[Workspace, str, dict[str, tuple[Path, dict[str, Any]]]]:
    """Run the parent until idle; return the workspace, its job id and its children by label."""

    # The workspace names no usable VASP: the parent's job parameter must reach every child.
    workspace, job_id = _run(
        tmp_path,
        "atomate2-full-characterization",
        "/nonexistent/vasp_std",
        timeout=3600.0,
        structure=structure,
        settings=settings,
        parameters={"vasp.command": FAKE_VASP, **(parameters or {})},
    )
    children = {
        document["tag"]: (path, document)
        for path, document in _jobs(workspace).values()
        if (document.get("parent") or {}).get("job_id") == job_id
    }
    return workspace, job_id, children


def _record(workspace: Workspace, job_id: str) -> JobRecord:
    """The record of a finished job, succeeded or failed."""

    return next(record for record in job_records(workspace, states=("succeeded", "failed")) if record.job_id == job_id)


#: Files the parent's ``plots`` script writes for each child label, besides the child's report.
_CHILD_FIGURES = {
    "relax": {"relaxation.png", "relaxation.json", "relaxed.cif", "relaxed.POSCAR"},
    **{label: _FIGURES[f"atomate2-{label.replace('_', '-')}"] for label in ("band_structure", "dielectric", "elastic")},
    "phonons": _FIGURES["atomate2-phonons"],
}


#: The section of each child's results in the ``latex`` report, and the figures it copies into ``figures/``.
_LATEX_SECTIONS = {
    "relax": "Structure",
    "band_structure": "Electronic structure",
    "elastic": "Elastic properties",
    "dielectric": "Dielectric response",
    "phonons": "Lattice dynamics",
}
_LATEX_FIGURES = {
    "band_structure": {"band_structure_band_structure_dos.png"},
    "elastic": {"elastic_stress_strain.png", "elastic_youngs_modulus.png"},
    "phonons": {"phonons_phonon_band_structure_dos.png", "phonons_phonon_thermodynamics.png"},
}


def _check_latex(result: Any, succeeded: set[str], report: dict[str, Any], compiled: bool) -> str:
    """The ``latex`` script's report.tex has a section for each succeeded child only, and exactly their figures."""

    tex = (result.output_dir / "report.tex").read_text(encoding="utf-8")
    assert f"Characterization of {report['formula']}" in tex
    for label, title in _LATEX_SECTIONS.items():
        assert (f"\\section{{{title}}}" in tex) is (label in succeeded), label
    assert {path.name for path in (result.output_dir / "figures").iterdir()} == {
        name for label in succeeded for name in _LATEX_FIGURES.get(label, ())
    }
    if compiled:
        assert (result.output_dir / "report.pdf").read_bytes().startswith(b"%PDF"), result.stderr
    assert "wrote report.tex" in result.stdout
    return tex


def _check_postprocess(workspace: Workspace, job_id: str, children: dict, succeeded: set[str], case: str) -> None:
    """The parent's report summarizes its children's documents; its plots script runs every succeeded child's."""

    directory = "atomate2-full-characterization"
    report = _report(directory, _record(workspace, job_id))
    assert {label for label, child in report["children"].items() if child["state"] == "succeeded"} == succeeded
    assert {child["job_id"] for child in report["children"].values()} == {doc["id"] for _, doc in children.values()}
    if "band_structure" in succeeded:
        uniform = _output(children["band_structure"][0])["uniform"]
        # The headline pair is the uniform mesh's.
        assert report["band_gap_eV"] == uniform["output"]["bandgap"]
        assert report["direct_gap"] == uniform["calcs_reversed"][0]["output"]["is_gap_direct"]
    if "elastic" in succeeded:
        elastic = _output(children["elastic"][0])
        derived = elastic["derived_properties"]
        assert report["max_abs_equilibrium_stress_GPa"] == pytest.approx(np.abs(elastic["eq_stress"]).max())
        assert report["max_abs_equilibrium_stress_GPa"] > 0
        # atomate2 derives its moduli from the IEEE-rotated tensor, the report from the raw one: invariants agree.
        assert report["bulk_modulus_hill_GPa"] == pytest.approx(derived["k_vrh"])
        assert report["shear_modulus_hill_GPa"] == pytest.approx(derived["g_vrh"])
    if "phonons" in succeeded:
        phonons = _output(children["phonons"][0])
        units = Structure.from_dict(phonons["phonon_bandstructure"]["structure"]).composition
        per_cell = units.get_reduced_composition_and_factor()[1] / (6.02214076e23 * 1.602176634e-19)
        assert report["zero_point_energy_eV_per_phonon_cell"] == pytest.approx(phonons["free_energies"][0] * per_cell)
        # Only the polar insulator's phonons get nonzero Born charges, so only there does atomate2 apply NAC.
        (path,) = (workspace.root / "postprocess").glob("**/report/phonons/report.json")
        nac = json.loads(path.read_text(encoding="utf-8"))["non_analytical_correction"]
        assert nac is (case == "polar_insulator")
    if "dielectric" in succeeded:
        calculation = _output(children["dielectric"][0])["calcs_reversed"][0]["output"]
        assert report["epsilon_electronic_average"] == pytest.approx(np.trace(calculation["epsilon_static"]) / 3)
    files = _plots(directory, _record(workspace, job_id))
    assert files == succeeded
    output = workspace.root / "postprocess"
    for label in succeeded:
        (path,) = output.glob(f"**/plots/{label}")
        assert {child.name for child in path.iterdir()} == {"report.json", "report.txt", *_CHILD_FIGURES[label]}, label
    package = load_workflow_package(REPO_ROOT / directory, register=False)
    latexmk = shutil.which("latexmk") is not None
    result = run_postprocess_script(package, "latex", _record(workspace, job_id))
    assert result.returncode == 0, result.stdout + result.stderr
    tex = _check_latex(result, succeeded, report, latexmk)
    if case == "dielectric_failed":
        state = report["children"]["dielectric"]["state"]
        assert state.startswith("failed") and "\\textbf{Dielectric response:} not reported" in tex
        assert f"\\texttt{{dielectric}}: {state.split(' ')[0]}" in tex
    if case == "metal":
        # Without latexmk on the PATH, only report.tex is written, and the script says why.
        bare = workspace.root.parent / "bare-path"
        bare.mkdir()
        (bare / "python3").symlink_to(sys.executable)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("PATH", str(bare))
            result = run_postprocess_script(
                package, "latex", _record(workspace, job_id), output_root=workspace.root.parent / "no-latexmk"
            )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "report.pdf not compiled: latexmk is not on the PATH" in result.stdout
        assert not (result.output_dir / "report.pdf").exists()
        _check_latex(result, succeeded, report, False)
        # A succeeded child whose analysis fails: the others still run, then the scripts fail naming it.
        for name in ("data/jobflow/jobflow-outputs.json", "run/jobflow-outputs.json"):
            source = children["elastic"][0] / name
            source.rename(source.with_suffix(".hidden"))
        for script in ("report", "plots"):
            result = run_postprocess_script(package, script, _record(workspace, job_id))
            assert result.returncode == 1 and "elastic report" in result.stderr, script
            assert (result.output_dir / "relax" / "report.json").is_file(), script
        broken = json.loads((result.output_dir.parent / "report" / "report.json").read_text(encoding="utf-8"))
        assert broken["failed_analyses"] == ["elastic report"]
        # The LaTeX report is still written, with a note in place of the elastic section; the run reused above's
        # output directory, so no stale elastic analysis or figure may survive.
        result = run_postprocess_script(package, "latex", _record(workspace, job_id))
        assert result.returncode == 1, result.stdout + result.stderr
        tex = _check_latex(result, succeeded - {"elastic"}, report, latexmk)
        assert "\\textbf{Elastic properties:} not reported" in tex
        # Both analyses report the failed child report; it is listed once.
        assert tex.count("\\item elastic report") == 1 and "\\item elastic plots" in tex


def _relaxes_cell(run: dict[str, Any]) -> bool:
    """Whether one fake VASP run was a relaxation of the cell."""

    return run["incar"].get("ISIF") == 3 and run["incar"].get("NSW", 0) > 1


def _output(directory: Path) -> dict[str, Any]:
    """The flow output a jobflow job published."""

    return json.loads((directory / "data" / "jobflow" / "jobflow-outputs.json").read_text(encoding="utf-8"))["output"]


def test_relax_failure_stops_the_characterization(tmp_path: Path) -> None:
    workspace, job_id, children = _characterize(tmp_path, aluminium(), {"fake_vasp.fail_on": "NSW"})
    failure = _failure(workspace, job_id)
    assert resolve_job(workspace, job_id).state == "failed" and failure["code"] == "relax_failed"
    assert set(children) == {"relax"}
    # No characterization.json: the report script says so and fails.
    package = load_workflow_package(REPO_ROOT / "atomate2-full-characterization", register=False)
    for script in ("report", "latex"):
        result = run_postprocess_script(package, script, _record(workspace, job_id))
        assert result.returncode == 1 and "characterization.json" in result.stderr, script
    assert not (result.output_dir / "report.tex").exists()


def test_cif_input_is_staged_as_a_poscar(tmp_path: Path) -> None:
    """``--input-from structure Si.cif`` passes a loaded structure, which is staged as the POSCAR the run reads."""

    import httk.core
    from httk.workflow.scaffold import new_job

    cif = tmp_path / "Si.cif"
    silicon().to(filename=str(cif))
    workspace = Workspace.initialize(tmp_path / "workspace")
    new_job(workspace, "atomate2.full-characterization", inputs={"structure": httk.core.load(cif)}, install=True)
    (poscar,) = workspace.root.glob("jobs/*/*/files/POSCAR")
    assert Structure.from_file(poscar).matches(Structure.from_file(cif))


@pytest.mark.parametrize(
    "child_parameters",
    [
        {"static": {}},
        {"phonons": {"bulk_relax_maker": None}},
        {"phonons": {"jobflow_overrides": [{"kind": "maker", "update": {"bulk_relax_maker.name": "again"}}]}},
    ],
    ids=["unknown_label", "owned_key", "owned_maker_field"],
)
def test_invalid_child_parameters_fail_before_any_child(tmp_path: Path, child_parameters: dict) -> None:
    workspace, job_id, children = _characterize(tmp_path, aluminium(), {}, {"child_parameters": child_parameters})
    failure = _failure(workspace, job_id)
    assert failure["code"] == "child_parameters_invalid", failure
    assert children == {}


@pytest.mark.parametrize("label", ["band_structure", "elastic"])
def test_child_refusing_its_parameters_fails_the_characterization(tmp_path: Path, label: str) -> None:
    """A malformed override only the child's creation detects, for a child called after the relaxation."""

    child_parameters = {label: {"jobflow_overrides": [{"kind": "nope"}]}}
    workspace, job_id, children = _characterize(tmp_path, aluminium(), {}, {"child_parameters": child_parameters})
    failure = _failure(workspace, job_id)
    assert failure["code"] == "child_parameters_invalid" and failure["message"].startswith(f"{label}: ")
    # The relaxation ran; no child of the failed step started, band_structure before elastic included.
    assert set(children) == {"relax"}


@pytest.mark.parametrize("case", tuple(_CASES))
def test_full_characterization(tmp_path: Path, case: str, test_profile: TestProfile) -> None:
    structure, reference, settings, normal = _CASES[case]
    if not normal and not test_profile.extended:
        pytest.skip("extended profile only (HTTK_TEST_PROFILE=extended)")
    # The metal case also passes the phonons child a min_length and a Maker-field override, the relax child an override.
    child_parameters = {
        "phonons": {"min_length": 8.0, "jobflow_overrides": [_DISPLACEMENT_ENCUT]},
        "relax": {"jobflow_overrides": [_NELM]},
    }
    child_parameters = child_parameters if case == "metal" else {}
    workspace, job_id, children = _characterize(
        tmp_path,
        structure(),
        {"fake_vasp.reference_a": reference, **settings},
        {"child_parameters": child_parameters} if child_parameters else {},
    )
    insulator = case in ("polar_insulator", "dielectric_failed")
    dielectric_failed = case == "dielectric_failed"
    magnetic = case == "magnetic_metal"
    if dielectric_failed:
        failure = _failure(workspace, job_id)
        assert failure["code"] == "characterization_incomplete", failure
        assert "dielectric" in failure["message"]
    else:
        assert resolve_job(workspace, job_id).state == "succeeded", _failure(workspace, job_id)

    expected = {"relax", "band_structure", "elastic", "phonons", *({"dielectric"} if insulator else ())}
    succeeded = expected - ({"dielectric"} if dielectric_failed else set())
    assert set(children) == expected
    states = {label: resolve_job(workspace, document["id"]).state for label, (_path, document) in children.items()}
    assert {label for label, state in states.items() if state == "succeeded"} == succeeded

    # Each fake VASP run, attributed to the child whose flow's job ran it.
    label_of = {document["id"]: label for label, (_path, document) in children.items()}
    runs: dict[str, list[dict[str, Any]]] = {label: [] for label in children}
    for log in workspace.root.glob("jobs/*/*/run/**/fake-vasp-invocations.jsonl"):
        directory = log.relative_to(workspace.root).parts[:3]
        document = json.loads(workspace.root.joinpath(*directory, "job.json").read_text(encoding="utf-8"))
        runs[label_of[document["parent"]["job_id"]]] += [json.loads(line) for line in log.read_text().splitlines()]
    # One double relaxation, with TightRelaxMaker's INCAR; no other child relaxes the cell.
    assert len(runs["relax"]) == 2
    for run in runs["relax"]:
        assert (run["incar"]["ENCUT"], run["incar"]["EDIFF"], run["incar"]["EDIFFG"]) == pytest.approx(
            (700, 1e-7, -0.001)
        )
        assert (run["incar"]["ISIF"], run["incar"]["NSW"]) == (3, 99)
        if child_parameters:
            assert run["incar"]["NELM"] == 150
    # The parent's TightRelaxMaker override comes first, then the given ones.
    tight, *given_overrides = children["relax"][1]["parameters"]["jobflow_overrides"]
    assert tight["function"] == "atomate2.vasp.powerups:update_user_incar_settings"
    assert tight["kwargs"]["incar_updates"]["EDIFF"] == pytest.approx(1e-7)
    assert given_overrides == child_parameters.get("relax", {}).get("jobflow_overrides", [])
    # The relaxation keeps atomate2's spin polarization, which detects magnetism; after a non-magnetic one every
    # other child runs without it, the parent's ISPIN override first, then the given ones.
    assert {run["incar"]["ISPIN"] for run in runs["relax"]} == {2}
    for label in set(children) - {"relax"}:
        overrides = children[label][1]["parameters"].get("jobflow_overrides", [])
        if magnetic:
            assert not [o for o in overrides if "ISPIN" in o.get("kwargs", {}).get("incar_updates", {})], label
            assert not [run for run in runs[label] if run["incar"].get("ISPIN") == 1], label
        else:
            assert overrides == [_NO_SPIN, *child_parameters.get(label, {}).get("jobflow_overrides", [])], label
            assert {run["incar"]["ISPIN"] for run in runs[label]} == {1}, label
    # Only relaxations move the cell (ISIF 3 with ionic steps): every other run keeps it or is a static.
    assert not [label for label in children if label != "relax" for run in runs[label] if _relaxes_cell(run)]
    assert len(runs["band_structure"]) == 3
    assert len(runs.get("dielectric", ())) == int(insulator)

    relax = _output(children["relax"][0])["output"]
    phonons = _output(children["phonons"][0])
    # Phonons ran only its displacements: no relaxation, no static energy run, no Born charge run.
    assert len(runs["phonons"]) == len(phonons["jobdirs"]["displacements_job_dirs"])
    if child_parameters:
        assert {run["incar"]["ENCUT"] for run in runs["phonons"]} == {555}
    assert phonons["jobdirs"]["optimization_run_job_dir"] is None
    assert phonons["jobdirs"]["static_run_job_dir"] is None
    assert phonons["jobdirs"]["born_run_job_dir"] is None
    composition = Structure.from_dict(relax["structure"]).composition
    formula_units = composition.num_atoms / composition.reduced_composition.num_atoms
    assert phonons["total_dft_energy"] == pytest.approx(relax["energy"] / formula_units)
    # The elastic fit used the relaxation's stress (atomate2 stores it in GPa, tension positive).
    elastic = _output(children["elastic"][0])
    assert np.any(np.asarray(relax["stress"]))
    assert np.asarray(elastic["eq_stress"]) == pytest.approx(-0.1 * np.asarray(relax["stress"]))

    plan_activation = children["band_structure"][1]["parent"]["activation_id"]
    assert children["elastic"][1]["parent"]["activation_id"] == plan_activation
    # Phonons waits for the dielectric child only when it takes its Born charges.
    phonons_later = insulator and not magnetic
    assert (children["phonons"][1]["parent"]["activation_id"] != plan_activation) is phonons_later
    if insulator:
        assert children["dielectric"][1]["parent"]["activation_id"] == plan_activation
    given = children["phonons"][1]["parameters"]
    assert given.get("use_symmetrized_structure") == ("primitive" if magnetic else None)
    assert given.get("min_length") == child_parameters.get("phonons", {}).get("min_length")
    if case == "polar_insulator":
        # The dielectric's tensors reach phonons, which applies the non-analytical correction with them:
        # isotropic charges survive atomate2's symmetrization, site by site in the phonon cell's order.
        calculation = _output(children["dielectric"][0])["calcs_reversed"][0]["output"]
        assert given["jobflow_inputs"]["born"]["value"] == calculation["outcar"]["born"]
        assert given["jobflow_inputs"]["epsilon_static"]["value"] == calculation["epsilon_static"]
        dielectric_cell = Structure.from_dict(calculation["structure"])
        phonon_cell = Structure.from_dict(phonons["structure"])
        assert [site.species_string for site in phonon_cell] == [site.species_string for site in dielectric_cell]
        assert np.asarray(phonons["born"]) == pytest.approx(np.asarray(calculation["outcar"]["born"]), abs=1e-6)
        assert np.any(np.asarray(phonons["born"]))
    else:
        # No Born charges at all: none given, and none computed.
        assert given["born_maker"] is None and "born" not in given["jobflow_inputs"]
        assert phonons["born"] is None

    parent_directory = _jobs(workspace)[job_id][0]
    characterization = json.loads((parent_directory / "data" / "characterization.json").read_text(encoding="utf-8"))
    decisions = characterization["decisions"]
    assert {label for label, part in characterization["parts"].items() if part["kind"] == "succeeded"} == succeeded
    assert (decisions["insulator"], decisions["magnetic"]) == (insulator, magnetic)
    assert decisions["child_parameters"] == child_parameters
    assert (_NO_SPIN_NOTE in decisions["notes"]) is not magnetic
    if case == "metal":
        # The parent read the workspace's atomate2 settings, as the children do.
        assert (decisions["band_gap_tolerance"], decisions["symprec"]) == (0.5, 0.01)
    if magnetic:
        assert any(Structure.from_dict(relax["structure"]).site_properties["magmom"])

    # The parent yields its standardized structure and links to its children's runs; each succeeded
    # child is collected under its own package.
    collected = {item.record.job_id: item for item in collect(workspace, fail_fast=True, allow_job_collector=True)}
    if dielectric_failed:
        # A failed job is not collected, the parent included; its succeeded children are.
        assert job_id not in collected and children["dielectric"][1]["id"] not in collected
    else:
        assert isinstance(collected[job_id].outputs["standardized_structure"], UnitcellStructureView)
        assert {label for label, _run_id in collected[job_id].child_runs} == expected
    for label in succeeded:
        package = REPO_ROOT / ("atomate2-double-relax" if label == "relax" else f"atomate2-{label.replace('_', '-')}")
        declaration = json.loads((package / "declaration.json").read_text(encoding="utf-8"))
        outputs = collected[children[label][1]["id"]].outputs
        assert set(outputs) == {output["name"] for output in declaration["outputs"]}, label
    _check_postprocess(workspace, job_id, children, succeeded, case)
