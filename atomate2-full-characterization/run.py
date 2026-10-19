#!/usr/bin/env python3
"""atomate2.full-characterization: characterize one material with the other atomate2 packages as child jobs.

The schedule::

    start ── standardize, call relax ──► plan ── call band_structure, elastic, and
             (atomate2.double-relax,             dielectric (insulator) or phonons (metal) ──► progress
              TightRelaxMaker's INCAR)
    progress ── whenever a child ends: once dielectric has ended, call phonons; when no child is live, finish

No VASP calculation runs twice. The relaxation is shared: elastic and phonons skip
their own bulk relaxation (``bulk_relax_maker = null``) and take its stress and
energy, which also replaces phonons' static energy run; phonons takes the dielectric
child's Born charges instead of running its own dielectric calculation (metals get
none); and the band-structure flow's static run is the only static. Every child
gets the same relaxed structure, in VASP's site order, so the Born charges refer to
the cell phonons runs on (phonons keeps ``use_symmetrized_structure`` unset).

A magnetic relaxed structure (nonzero ``magmom`` site moments) is the exception:
atomate2's phonon analysis refuses it unless phonons runs on its own symmetrized
primitive cell, whose site order differs. Phonons is then called at once with
``use_symmetrized_structure = "primitive"`` and computes its own Born charges
(insulators; a deliberate duplicate of the dielectric child) or none (metals).

The steps that import atomate2 and pymatgen first export the ``atomate2.*`` and
``pymatgen.*`` settings as the variables those libraries read, as the jobflow
runner does for the children, so the parent's decisions use the same tolerances.

The job parameter ``child_parameters`` maps child labels to further job parameters for
that child: its own declared parameters and ``jobflow_overrides``, which follow the
parent's. Unknown labels and the parameters the parent sets itself (``_OWNED``, also as
the first field of a ``maker`` override's path) are refused before any child is called.
Each child's own checks (declared parameter types, ``jobflow_overrides`` syntax) run when
that child is created, so for the children after the relaxation they fail the job only
then, and no child of that step is started. Either way the job fails ``child_parameters_invalid``.

The job fails ``relax_failed`` when the relaxation fails. Otherwise it waits for
every child, then succeeds when all succeeded and fails ``characterization_incomplete``
naming the failed ones. ``characterization.json`` (in the workdir and the job data)
records every child and the decisions taken.
"""

import json
import os

from httk.workflow import Attempt, ChildResult, Runner

run = Runner("atomate2.full-characterization")

#: Job parameters forwarded to every child: those httk-workflow's jobflow runner forwards to its own children.
_VASP_COMMANDS = ("vasp.command", "vasp.gamma_command", "vasp.ncl_command")
_LIBRARY_PREFIXES = ("atomate2.", "pymatgen.")
_TERMINAL = ("succeeded", "failed", "cancelled")
#: Every child's label, and the job parameters the parent sets itself for it, which ``child_parameters`` may not.
_OWNED = {
    "relax": (),
    "band_structure": (),
    "elastic": ("bulk_relax_maker",),
    "dielectric": (),
    "phonons": ("bulk_relax_maker", "born_maker", "use_symmetrized_structure"),
}


def _forwarded(a: Attempt) -> dict[str, object]:
    """Return the VASP command and library settings among the job's parameters."""

    return {k: v for k, v in a.parameters.items() if k in _VASP_COMMANDS or k.startswith(_LIBRARY_PREFIXES)}


def _export_library_settings(a: Attempt) -> None:
    """Export the settings as ``ATOMATE2_<KEY>``/``PMG_<KEY>`` before atomate2 or pymatgen is imported.

    The jobflow runner's rule: workspace settings overridden by job parameters, string and
    number values only, ``pymatgen.psp_dir`` as ``PMG_VASP_PSP_DIR``.
    """

    for key, value in {**a.context.settings, **a.parameters}.items():
        if key == "pymatgen.psp_dir" or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        for prefix, variable in (("atomate2.", "ATOMATE2_"), ("pymatgen.", "PMG_")):
            if key.startswith(prefix):
                os.environ[variable + key.removeprefix(prefix).upper().replace(".", "_")] = str(value)
    psp_dir = a.setting("pymatgen.psp_dir", None)
    if isinstance(psp_dir, str) and psp_dir:
        os.environ["PMG_VASP_PSP_DIR"] = psp_dir


def _invalid_child_parameters(given: object) -> str | None:
    """Return why the ``child_parameters`` job parameter is invalid, or None."""

    if not isinstance(given, dict):
        return "child_parameters must map child labels to job parameters"
    for label, parameters in given.items():
        if label not in _OWNED:
            return f"child_parameters names an unknown child {label!r}; the children are {', '.join(_OWNED)}"
        if not isinstance(parameters, dict):
            return f"child_parameters[{label!r}] must map job parameter names to values"
        overrides = parameters.get("jobflow_overrides", [])
        if not isinstance(overrides, list):
            return f"child_parameters[{label!r}]['jobflow_overrides'] must be a list"
        names = set(parameters)
        # A maker override sets Maker fields too; function and maker_kwargs overrides are not inspected.
        for override in overrides:
            if (
                isinstance(override, dict)
                and override.get("kind") == "maker"
                and isinstance(override.get("update"), dict)
            ):
                names |= {str(path).split(".", 1)[0] for path in override["update"]}
        owned = sorted(names & set(_OWNED[label]))
        if owned:
            return f"child_parameters[{label!r}] sets {', '.join(owned)}, which the parent sets itself"
    return None


def _call(
    a: Attempt,
    alias: str,
    inputs: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
    structure: str | None = None,
) -> dict[str, object] | None:
    """Call one package on the relaxed structure (or ``structure``) and return its bookkeeping entry.

    The child's parameters are the forwarded ones, then its ``child_parameters``, then the
    parent's own ``parameters``; its ``jobflow_overrides`` are the parent's followed by the given ones.
    A child that refuses them fails the job ``child_parameters_invalid``; this and every later
    call then return None, and the step returns without its own outcome.
    """

    if a.published:
        return None

    owned = parameters or {}
    child = a.parameters.get("child_parameters", {}).get(alias, {})
    overrides = [*owned.get("jobflow_overrides", ()), *child.get("jobflow_overrides", ())]
    try:
        reference = a.call(
            alias,
            label=alias,
            # Absolute: a relative path would resolve against the called package's directory.
            inputs={"structure": structure or str(a.workdir / "relaxed.json"), **(inputs or {})},
            parameters={**_forwarded(a), **child, **owned} | ({"jobflow_overrides": overrides} if overrides else {}),
        )
    except ValueError as exc:
        a.fail("child_parameters_invalid", f"{alias}: {exc}")
        return None
    return {"job_id": reference.job_id, "job_key": reference.job_key, "kind": "called", "failure": None}


def _part(child: ChildResult) -> dict[str, object]:
    """Return the bookkeeping entry of one observed child."""

    failure = None if child.failure is None else {"code": child.failure.code, "message": child.failure.message}
    return {"job_id": child.job_id, "job_key": child.job_key, "kind": child.kind, "failure": failure}


def _flow_output(child: ChildResult) -> dict:
    """Return the flow output a succeeded atomate2 child published."""

    assert child.data is not None
    return json.loads((child.data / "jobflow" / "jobflow-outputs.json").read_text(encoding="utf-8"))["output"]


@run.step
def start(a: Attempt) -> None:
    """Check ``child_parameters``, standardize the structure and call the shared tight double relaxation."""

    problem = _invalid_child_parameters(a.parameters.get("child_parameters", {}))
    if problem is not None:
        a.fail("child_parameters_invalid", problem)
        return
    _export_library_settings(a)
    from atomate2 import SETTINGS
    from atomate2.vasp.sets.core import TightRelaxSetGenerator
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = Structure.from_file(a.payload / "files" / "POSCAR")
    if a.parameter("standardize"):
        # atomate2's structure_to_primitive, keeping site properties such as magnetic moments.
        analyzer = SpacegroupAnalyzer(structure, symprec=SETTINGS.SYMPREC)
        structure = analyzer.get_primitive_standard_structure(keep_site_properties=True)
    path = a.workdir / "standardized.json"
    path.write_text(json.dumps(structure.as_dict()), encoding="utf-8")
    a.put(path, "standardized.json")
    # atomate2's own bulk relaxation for phonons and elastic is a double relaxation of TightRelaxMakers:
    # atomate2.double-relax with TightRelaxMaker's INCAR settings is that relaxation.
    tight = [
        {
            "kind": "function",
            "function": "atomate2.vasp.powerups:update_user_incar_settings",
            "kwargs": {"incar_updates": TightRelaxSetGenerator().incar_updates},
        }
    ]
    if _call(a, "relax", parameters={"jobflow_overrides": tight}, structure=str(path)) is None:
        return
    a.gather("plan", when="all_succeeded", on_impossible="relax_failed")


@run.step
def relax_failed(a: Attempt) -> None:
    """Fail: nothing can be characterized without the relaxation."""

    relax = a.children["relax"]
    a.fail("relax_failed", f"the relaxation did not succeed: {relax.failure.message if relax.failure else relax.kind}")


@run.step
def plan(a: Attempt) -> None:
    """Call every child that only needs the relaxation, in parallel."""

    _export_library_settings(a)
    import numpy as np
    from atomate2 import SETTINGS
    from pymatgen.core import Structure

    relax = a.children["relax"]
    output = _flow_output(relax)["output"]
    (a.workdir / "relaxed.json").write_text(json.dumps(output["structure"]), encoding="utf-8")
    structure = Structure.from_dict(output["structure"])
    composition = structure.composition
    # atomate2's get_total_energy_per_cell divides by the same number of formula units, of whatever cell
    # phonons runs on, so the value holds for its symmetrized cell too.
    energy = output["energy"] / (composition.num_atoms / composition.reduced_composition.num_atoms)
    # The tolerance atomate2's input sets use to tell metals from insulators (VaspInputGenerator.bandgap_tol).
    insulator = output["bandgap"] > SETTINGS.BANDGAP_TOL
    # atomate2's phonon analysis check: nonzero moments need its symmetrized primitive cell.
    magmoms = structure.site_properties.get("magmom")
    magnetic = bool(magmoms is not None and np.any(np.asarray(magmoms) != 0.0))
    parts = {"relax": _part(relax)}
    parts["band_structure"] = _call(a, "band_structure")
    parts["elastic"] = _call(a, "elastic", {"equilibrium_stress": output["stress"]}, {"bulk_relax_maker": None})
    if insulator:
        parts["dielectric"] = _call(a, "dielectric")
    phonons = {"bulk_relax_maker": None} | ({} if insulator else {"born_maker": None})
    if magnetic:
        phonons["use_symmetrized_structure"] = "primitive"
        parts["phonons"] = _call(a, "phonons", {"total_dft_energy_per_formula_unit": energy}, phonons)
        born = "with its own Born charges" if insulator else "without Born charges"
        notes = [f"magnetic: phonons on atomate2's symmetrized primitive cell {born}"]
    elif insulator:
        notes = ["insulator: phonons waits for the dielectric child's Born charges"]
    else:
        parts["phonons"] = _call(a, "phonons", {"total_dft_energy_per_formula_unit": energy}, phonons)
        notes = ["metal: no dielectric child, phonons without non-analytical correction"]
    decisions = {
        "band_gap": output["bandgap"],
        "band_gap_tolerance": SETTINGS.BANDGAP_TOL,
        "symprec": SETTINGS.SYMPREC,
        "insulator": insulator,
        "magnetic": magnetic,
        "notes": notes,
        "child_parameters": a.parameters.get("child_parameters", {}),
    }
    if a.published:
        return
    a.state.merge({"parts": parts, "decisions": decisions, "energy_per_formula_unit": energy})
    a.gather("progress", when="any_terminal")


@run.step
def progress(a: Attempt) -> None:
    """Record the children that ended, call phonons once dielectric has ended, and finish when none is live."""

    # Every live child is in this join (each wait rejoins them), so only terminal entries are kept from state.
    parts = {label: part for label, part in a.state["parts"].items() if part["kind"] in _TERMINAL}
    parts.update({child.label: _part(child) for child in a.children})
    decisions = a.state["decisions"]
    dielectric = parts.get("dielectric")
    if dielectric is not None and dielectric["kind"] in _TERMINAL and "phonons" not in parts:
        inputs = {"total_dft_energy_per_formula_unit": a.state["energy_per_formula_unit"]}
        parameters = {"bulk_relax_maker": None}
        if dielectric["kind"] == "succeeded":
            # Where atomate2's PhononMaker takes them from its own born_maker's task document.
            calculation = _flow_output(a.children["dielectric"])["calcs_reversed"][0]["output"]
            inputs |= {"born": calculation["outcar"]["born"], "epsilon_static": calculation["epsilon_static"]}
        else:
            parameters["born_maker"] = None
            decisions["notes"].append("phonons without non-analytical correction: dielectric failed")
        parts["phonons"] = _call(a, "phonons", inputs, parameters)
        if a.published:
            return
    a.state.merge({"parts": parts, "decisions": decisions})
    live = [label for label, part in parts.items() if part["kind"] not in _TERMINAL]
    if live:
        a.gather("progress", when="any_terminal", rejoin=[label for label in live if label in a.children.labels])
        return
    path = a.workdir / "characterization.json"
    path.write_text(json.dumps({"parts": parts, "decisions": decisions}, indent=2) + "\n", encoding="utf-8")
    a.put(path, "characterization.json")
    failed = [label for label, part in parts.items() if part["kind"] != "succeeded"]
    if failed:
        a.fail("characterization_incomplete", f"these parts did not succeed: {', '.join(failed)}")
        return
    a.succeed()


if __name__ == "__main__":
    raise SystemExit(run.main())
