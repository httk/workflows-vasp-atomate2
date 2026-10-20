"""Postprocess reports and plots of the workflows-vasp-atomate2 packages (every package carries this same file).

``scripts/report`` and ``scripts/plots`` call :func:`main`, which reads the job's
``job.json``, loads the flow output (``{"output": ...}`` in ``jobflow-outputs.json``)
and dispatches on the job's workflow name. ``report`` writes ``report.json`` (plain
JSON numbers, units in the key names) and ``report.txt``; ``plots`` writes PNG
figures. Everything is written to the current directory, the postprocess output
directory; the job payload is only read. The jobflow child jobs that run the
flow's individual atomate2 jobs exit 0 without output: their parent's flow output
holds the results. ``atomate2.full-characterization`` runs its succeeded children's
analyses in-process into sub-directories named by child label.

Values come from the atomate2 documents where the hooks in ``collect.py`` read
them; VASP files are read only where a document lacks the data (a DOS), from the
atomate2 job directory its ``dir_name`` token names.
"""

import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: One J/mol in eV per particle: 1 / (N_A e), exact in the 2019 SI.
_EV_PER_JOULE_PER_MOL = 1 / (6.02214076e23 * 1.602176634e-19)
#: Voigt columns xx yy zz yz xz xy, taken from VASP's printed order XX YY ZZ XY YZ ZX.
_VASP_TO_VOIGT = [0, 1, 2, 4, 5, 3]
_EXEC_TOKEN = "httk-jobflow-exec://"
_DPI = 150
#: Imaginary frequencies (THz) within this of zero are acoustic-sum-rule noise at and near Γ (0.05 THz = 1.7 cm⁻¹).
_IMAGINARY_TOLERANCE_THZ = 0.05


class AnalysisError(Exception):
    """The job cannot be analysed: its flow output (or characterization record) is unusable, or a child's failed."""


@dataclass(frozen=True)
class Job:
    """One collected job: its payload directory, ``job.json``, and result directories."""

    dir: Path
    spec: dict
    data: Path | None
    workdir: Path | None

    @classmethod
    def at(cls, directory: Path, data: Path | None = None, workdir: Path | None = None) -> "Job":
        """Read the job at *directory*; result directories default to its ``data/`` and ``run/``."""

        spec = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        return cls(directory, spec, data or directory / "data", workdir or directory / "run")

    @property
    def workflow(self) -> str:
        return (self.spec.get("workflow") or {}).get("name", "")

    @property
    def parameters(self) -> dict:
        return self.spec.get("parameters") or {}

    def result(self, *names: str) -> Path:
        """The first existing file among *names* (relative paths) under the data dir, then the workdir."""

        for root in (self.data, self.workdir):
            for name in names:
                if root is not None and (root / name).is_file():
                    return root / name
        raise AnalysisError(f"job {self.spec.get('id')}: none of {', '.join(names)} in its data dir or workdir")

    def flow_output(self) -> dict:
        """The flow's final output, from ``jobflow-outputs.json``."""

        prefix = self.parameters.get("jobflow_data_prefix", "jobflow") or ""
        path = self.result(f"{prefix}/jobflow-outputs.json".lstrip("/"), "jobflow-outputs.json")
        try:
            output = json.loads(path.read_text(encoding="utf-8"))["output"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AnalysisError(f"{path}: unreadable flow output: {exc}") from exc
        if not isinstance(output, dict):
            raise AnalysisError(f"{path}: the flow output is not a JSON object")
        return output


# --- shared helpers ---


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    # pymatgen's plotters ask for Times New Roman, which most machines lack.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    return plt


def _save(figure, out: Path, name: str) -> None:
    """Save a figure (or the figure of an axes) as a PNG and close every figure."""

    plt = _pyplot()
    getattr(figure, "figure", figure).savefig(out / name, dpi=_DPI, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {name}")


def _skip(name: str, reason: str) -> None:
    print(f"skipped {name}: {reason}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _number(value: object) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _text(value: dict, indent: str = "") -> list[str]:
    """Render a report dictionary as indented ``key: value`` lines, matrices row by row."""

    lines = []
    for key, item in value.items():
        if isinstance(item, dict):
            lines += [f"{indent}{key}:", *_text(item, indent + "  ")]
        elif isinstance(item, list) and item and all(isinstance(row, dict) for row in item):
            for number, row in enumerate(item):
                lines += [f"{indent}{key}[{number}]:", *_text(row, indent + "  ")]
        elif isinstance(item, list) and item and all(isinstance(row, list) for row in item):
            lines += [f"{indent}{key}:", *(f"{indent}  " + "  ".join(f"{x:>11.5g}" for x in row) for row in item)]
        elif isinstance(item, list):
            lines.append(f"{indent}{key}: " + ", ".join(_number(x) for x in item))
        else:
            lines.append(f"{indent}{key}: {_number(item)}")
    return lines


def _structure(document: dict):
    from pymatgen.core import Structure

    return Structure.from_dict(document)


def _symprec(job: Job) -> float:
    """atomate2's SYMPREC for the job: its ``atomate2.symprec`` parameter or workspace setting, else atomate2's."""

    from atomate2 import SETTINGS
    from httk.workflow import Workspace

    value = job.parameters.get("atomate2.symprec")
    if value is None:
        value = Workspace(os.environ["HTTK_WORKFLOW_WORKSPACE_DIR"]).read_settings().get("atomate2.symprec")
    return SETTINGS.SYMPREC if value is None else float(value)


def _cell(structure, job: Job) -> dict:
    """Formula, space group (SpacegroupAnalyzer at the job's SYMPREC) and lattice of a pymatgen structure."""

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    symprec = _symprec(job)
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    lattice = structure.lattice
    return {
        "formula": structure.composition.reduced_formula,
        "sites": len(structure),
        "space_group": analyzer.get_space_group_symbol(),
        "space_group_number": analyzer.get_space_group_number(),
        "symprec": symprec,
        "a_angstrom": lattice.a,
        "b_angstrom": lattice.b,
        "c_angstrom": lattice.c,
        "alpha_deg": lattice.alpha,
        "beta_deg": lattice.beta,
        "gamma_deg": lattice.gamma,
        "volume_angstrom3": lattice.volume,
    }


def _exec_dir(dir_name: str | None) -> Path | None:
    """The atomate2 job directory an ``httk-jobflow-exec://<job id>`` token (or a plain path) names."""

    if not dir_name:
        return None
    if not dir_name.startswith(_EXEC_TOKEN):
        path = Path(dir_name.partition(":")[2] if ":" in dir_name and not dir_name.startswith("/") else dir_name)
        return path if path.is_dir() else None
    from httk.workflow import Workspace
    from httk.workflow.introspection import resolve_job

    job_id, _, suffix = dir_name.removeprefix(_EXEC_TOKEN).partition("/")
    try:
        path = resolve_job(Workspace(os.environ["HTTK_WORKFLOW_WORKSPACE_DIR"]), job_id).path / "run" / "exec"
    except (KeyError, ValueError, OSError):
        return None
    return path / suffix if suffix else path


def _complete_dos(task: dict):
    """The task's CompleteDos: the document's ``vasp_objects.dos``, else its run's ``vasprun.xml(.gz)``."""

    from pymatgen.electronic_structure.dos import CompleteDos

    dos = (task.get("vasp_objects") or {}).get("dos")
    if isinstance(dos, dict):
        return CompleteDos.from_dict(dos)
    directory = _exec_dir(task.get("dir_name"))
    for name in ("vasprun.xml.gz", "vasprun.xml"):
        if directory is not None and (directory / name).is_file():
            from pymatgen.io.vasp import Vasprun

            return Vasprun(str(directory / name), parse_potcar_file=False).complete_dos
    return None


def _reference(gap: float | None, vbm: float | None, efermi: float | None) -> dict:
    """The energy zero of the electronic figures: the VBM of a gapped system, else the Fermi level."""

    if gap and vbm is not None:
        return {"reference": "valence band maximum", "zero_eV": vbm, "label": "E − E_VBM (eV)"}
    return {"reference": "Fermi level", "zero_eV": efermi, "label": "E − E_F (eV)"}


def _dos_axis(axis, dos, zero: float, vertical: bool = False) -> None:
    """Spin-summed total and element-projected DOS (projections only where VASP wrote them) against E - zero."""

    curves = {"total": dos.get_densities()}
    curves |= {
        str(element): part.get_densities() for element, part in (dos.get_element_dos() if dos.pdos else {}).items()
    }
    energies = dos.energies - zero
    for label, density in curves.items():
        axis.plot(*((density, energies) if vertical else (energies, density)), label=label, linewidth=1)
    axis.legend(fontsize=8)


def _plot_dos(dos, out: Path, name: str, title: str, reference: dict) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    _dos_axis(axis, dos, reference["zero_eV"])
    window = (dos.energies - reference["zero_eV"] > -10) & (dos.energies - reference["zero_eV"] < 10)
    axis.set(xlim=(-10, 10), ylim=(0, 1.05 * float(dos.get_densities()[window].max(initial=0)) or 1))
    axis.axvline(0, color="grey", linewidth=0.5)
    axis.set(
        xlabel=reference["label"],
        ylabel="DOS (states/eV per cell)",
        title=title + ("" if dos.pdos else " (no projections)"),
    )
    _write_json(out / f"{name.removesuffix('.png')}.json", reference)
    _save(figure, out, name)


def _final_calculation(task: dict) -> dict:
    return ((task.get("calcs_reversed") or [{}])[0].get("output")) or {}


# --- structure and energy packages ---


def _initial_structure(job: Job, task: dict) -> tuple[object, str]:
    """The job's input structure (a staged file or a literal value, as the jobflow runner decodes them) and its source.

    Without one, the input of the document's last calculation: for a double relaxation, its second stage's.
    """

    from pymatgen.core import Structure

    given = (job.parameters.get("jobflow_inputs") or {}).get("structure") or {}
    if given.get("kind") == "path" and (job.dir / given["value"]).is_file():
        return Structure.from_file(job.dir / given["value"]), "job input"
    if given.get("kind") == "value":
        from monty.json import MontyDecoder

        value = MontyDecoder().process_decoded(given.get("value"))
        if isinstance(value, Structure):
            return value, "job input"
    return _structure(task["input"]["structure"]), "input of the last calculation in the task document"


def _energies(task: dict) -> dict:
    output = task.get("output") or {}
    calculation = _final_calculation(task)
    return {
        "formula": task.get("formula_pretty"),
        "energy_eV": output.get("energy"),
        "energy_per_atom_eV": output.get("energy_per_atom"),
        "band_gap_eV": output.get("bandgap"),
    } | ({"vbm_eV": calculation.get("vbm"), "cbm_eV": calculation.get("cbm")} if calculation else {})


def _report_relax(output: dict, job: Job, out: Path) -> dict:
    import numpy as np

    (initial, source), relaxed = _initial_structure(job, output), _structure(output["output"]["structure"])
    before, after = _cell(initial, job), _cell(relaxed, job)
    forces = np.asarray(output["output"].get("forces") or [[0.0, 0.0, 0.0]])
    report = _energies(output) | {
        "initial": before,
        "initial_source": source,
        "relaxed": after,
        "relative_volume_change": after["volume_angstrom3"] / before["volume_angstrom3"] - 1,
        "max_force_eV_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "stress_kbar": output["output"].get("stress"),
    }
    relaxed.to(filename=str(out / "relaxed.cif"))
    relaxed.to(filename=str(out / "relaxed.POSCAR"), fmt="poscar")
    return report


def _plots_relax(output: dict, job: Job, out: Path) -> None:
    import numpy as np

    calculations = list(reversed(output.get("calcs_reversed") or []))
    steps = [
        step for calculation in calculations for step in (calculation.get("output") or {}).get("ionic_steps") or []
    ]
    if not steps:
        return _skip("relaxation.png", "the task document has no ionic steps")
    sites = len(steps[0]["structure"]["sites"])
    energies = [step["e_0_energy"] / sites for step in steps]
    forces = [float(np.linalg.norm(step["forces"], axis=1).max()) for step in steps]
    _write_json(out / "relaxation.json", {"energy_per_atom_eV": energies, "max_force_eV_per_angstrom": forces})
    plt = _pyplot()
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), constrained_layout=True)
    steps_axis = range(1, len(steps) + 1)
    top.plot(steps_axis, energies, "o-")
    top.set_ylabel("energy per atom (eV)")
    bottom.semilogy(steps_axis, forces, "o-")
    bottom.set(xlabel="ionic step", ylabel="max force (eV/Å)")
    bottom.xaxis.get_major_locator().set_params(integer=True)
    # atomate2's document keeps only the final calculation of a double relaxation.
    top.set_title(f"{output.get('formula_pretty')}: ionic steps of {len(calculations)} calculation(s) in the document")
    _save(figure, out, "relaxation.png")
    return None


def _report_static(output: dict, job: Job, out: Path) -> dict:
    calculation = _final_calculation(output)
    return _energies(output) | {
        "fermi_energy_eV": calculation.get("efermi"),
        "total_magnetization_muB": (calculation.get("outcar") or {}).get("total_magnetization"),
        "structure": _cell(_structure(output["output"]["structure"]), job),
    }


def _plots_static(output: dict, job: Job, out: Path) -> None:
    dos = _complete_dos(output)
    if dos is None:
        return _skip("dos.png", "no DOS in the document and no vasprun.xml(.gz) in its run directory")
    calculation = _final_calculation(output)
    reference = _reference(
        (output.get("output") or {}).get("bandgap"), calculation.get("vbm"), calculation.get("efermi")
    )
    return _plot_dos(dos, out, "dos.png", f"{output.get('formula_pretty')}: static DOS", reference)


# --- band structure ---


def _band_edge(edge: dict) -> dict | None:
    if edge.get("energy") is None or edge.get("kpoint") is None:
        return None
    return {
        "energy_eV": edge["energy"],
        "kpoint_label": edge["kpoint"].label,
        "kpoint_frac": edge["kpoint"].frac_coords.tolist(),
    }


def _line_bands(output: dict):
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine

    if not isinstance(output.get("line_bs"), dict):
        raise AnalysisError('the flow output has no line_bs; this package needs bandstructure_type="both"')
    return BandStructureSymmLine.from_dict(output["line_bs"])


def _uniform_bands(output: dict):
    from pymatgen.electronic_structure.bandstructure import BandStructure

    if not isinstance(output.get("uniform_bs"), dict):
        raise AnalysisError('the flow output has no uniform_bs; this package needs bandstructure_type="both"')
    return BandStructure.from_dict(output["uniform_bs"])


def _band_reference(output: dict) -> dict:
    """The energy zero of every band-structure figure, from the uniform calculation (the headline values)."""

    uniform = output.get("uniform") or {}
    bands = _uniform_bands(output)
    vbm = None if bands.is_metal() else bands.get_vbm()["energy"]
    return _reference((uniform.get("output") or {}).get("bandgap"), vbm, _final_calculation(uniform).get("efermi"))


def _report_band_structure(output: dict, job: Job, out: Path) -> dict:
    bands, uniform_bands = _line_bands(output), _uniform_bands(output)
    uniform = output.get("uniform") or {}
    # The headline values are the uniform mesh's; the line path's are named line_*.
    gap, line_gap = uniform_bands.get_band_gap(), bands.get_band_gap()
    metal = uniform_bands.is_metal()
    return {
        "formula": uniform.get("formula_pretty"),
        "band_gap_eV": (uniform.get("output") or {}).get("bandgap"),
        "direct_gap": gap["direct"],
        "transition": gap["transition"],
        "vbm_eV": None if metal else uniform_bands.get_vbm()["energy"],
        "cbm_eV": None if metal else uniform_bands.get_cbm()["energy"],
        "line_band_gap_eV": line_gap["energy"],
        "line_direct_gap": line_gap["direct"],
        "line_transition": line_gap["transition"],
        "line_vbm": _band_edge(bands.get_vbm()),
        "line_cbm": _band_edge(bands.get_cbm()),
        "fermi_energy_eV": _final_calculation(uniform).get("efermi"),
        "line_fermi_energy_eV": bands.efermi,
        "number_of_bands": bands.nb_bands,
        "spin_polarized": bands.is_spin_polarized,
        "cell": _cell(_structure(output["line"]["output"]["structure"]), job),
    }


def _bands_axis(axis, bands, reference: dict) -> None:
    """The line-mode bands against E - zero, spin down dashed, with merged labels at segment junctions."""

    from pymatgen.electronic_structure.plotter import BSPlotter

    data = BSPlotter(bands).bs_plot_data(zero_to_efermi=False)
    for spin, style in (("1", "-"), ("-1", "--")):
        for distances, energies in zip(data["distances"], data["energy"].get(spin, ()), strict=False):
            for band in energies:
                axis.plot(distances, band - reference["zero_eV"], style, color="C0", linewidth=1)
    ticks: list[list] = []
    for distance, label in zip(data["ticks"]["distance"], data["ticks"]["label"], strict=True):
        if ticks and abs(ticks[-1][0] - distance) < 1e-9:
            if label != ticks[-1][1]:
                ticks[-1][1] += "|" + label  # a discontinuity of the path
            continue
        ticks.append([distance, label])
    axis.set_xticks([distance for distance, _ in ticks], [label for _, label in ticks])
    for distance, _ in ticks:
        axis.axvline(distance, color="grey", linewidth=0.5)
    axis.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    axis.set(xlim=(ticks[0][0], ticks[-1][0]), ylabel=reference["label"])


def _plots_band_structure(output: dict, job: Job, out: Path) -> None:
    bands, reference = _line_bands(output), _band_reference(output)
    plt = _pyplot()
    gap = float((output["uniform"].get("output") or {}).get("bandgap") or 0)
    window = (-4, 4 + gap)
    # Where the line path puts its band edges on the plotted scale (0 for a VBM on the path).
    edges = [bands.get_vbm()["energy"], bands.get_cbm()["energy"]] if not bands.is_metal() else [None, None]
    plotted = {
        f"plotted_line_{name}_eV": None if e is None else e - reference["zero_eV"]
        for name, e in zip(("vbm", "cbm"), edges, strict=True)
    }
    _write_json(out / "band_structure.json", reference | plotted)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    _bands_axis(axis, bands, reference)
    axis.set(
        ylim=window, title=f"{output['uniform'].get('formula_pretty')}: band structure (zero: {reference['reference']})"
    )
    _save(figure, out, "band_structure.png")
    dos = _complete_dos(output.get("uniform") or {})
    if dos is None:
        _skip("dos.png", "no DOS in the uniform document or its run directory")
        return _skip("band_structure_dos.png", "no DOS")
    _plot_dos(dos, out, "dos.png", f"{output['uniform'].get('formula_pretty')}: DOS (uniform mesh)", reference)
    figure, (left, right) = plt.subplots(
        1, 2, sharey=True, figsize=(9, 5), width_ratios=[3, 1], constrained_layout=True
    )
    _bands_axis(left, bands, reference)
    _dos_axis(right, dos, reference["zero_eV"], vertical=True)
    right.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    right.set(xlabel="DOS (states/eV)", ylim=window)
    right.set_xlim(left=0)
    return _save(figure, out, "band_structure_dos.png")


# --- dielectric ---


def _report_dielectric(output: dict, job: Job, out: Path) -> dict:
    import numpy as np

    calculation = _final_calculation(output)
    structure = _structure(output["output"]["structure"])
    tensors = {}
    if calculation.get("epsilon_static") is not None:
        tensors["electronic"] = np.asarray(calculation["epsilon_static"], dtype=float)
    if calculation.get("epsilon_ionic") is not None:
        tensors["ionic"] = np.asarray(calculation["epsilon_ionic"], dtype=float)
    if len(tensors) == 2:
        tensors["total"] = tensors["electronic"] + tensors["ionic"]
    report: dict = {"formula": output.get("formula_pretty")}
    report["dielectric_tensors"] = {kind: tensor.tolist() for kind, tensor in tensors.items()}
    # Eigenvalues of the symmetric part: the asymmetry is print rounding.
    report["eigenvalues"] = {kind: np.linalg.eigvalsh((t + t.T) / 2).tolist() for kind, t in tensors.items()}
    report["averages"] = {kind: float(np.trace(tensor) / 3) for kind, tensor in tensors.items()}
    if "electronic" in tensors:
        report["refractive_index"] = math.sqrt(report["averages"]["electronic"])
    properties = calculation.get("dielectric_properties") or {}
    born = properties.get("born_charges")
    if born:
        report["born_effective_charges"] = [
            {"site": index, "species": site.species_string, "tensor": charges, "trace": float(np.trace(charges))}
            for index, (site, charges) in enumerate(zip(structure, born, strict=True))
        ]
        # The acoustic sum rule makes this zero up to the calculation's precision.
        report["born_charge_sum"] = np.sum(born, axis=0).tolist()
    if properties.get("piezo_tensor") is not None and properties.get("piezo_ionic_tensor") is not None:
        total = np.add(properties["piezo_tensor"], properties["piezo_ionic_tensor"])
        report["piezoelectric_tensor_C_per_m2"] = total[:, _VASP_TO_VOIGT].tolist()
        report["piezoelectric_voigt_order"] = "xx yy zz yz xz xy (relaxed-ion: electronic + ionic)"
    return report


def _plots_dielectric(output: dict, job: Job, out: Path) -> None:
    import numpy as np

    report = _report_dielectric(output, job, out)
    plt = _pyplot()
    tensors = {kind: np.asarray(tensor) for kind, tensor in report["dielectric_tensors"].items()}
    if "electronic" not in tensors:
        _skip("dielectric.png", "the document has no epsilon_static")
    else:
        # Principal axes of the static (or electronic) tensor; each part's component along them.
        reference = tensors.get("total", tensors["electronic"])
        _, axes_vectors = np.linalg.eigh((reference + reference.T) / 2)
        parts = {
            kind: [float(v @ tensors[kind] @ v) for v in axes_vectors.T]
            for kind in ("electronic", "ionic")
            if kind in tensors
        }
        figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        bottom = np.zeros(3)
        for kind, values in parts.items():
            axis.bar(["1", "2", "3"], values, bottom=bottom, label=f"{kind} (ε∞)" if kind == "electronic" else kind)
            bottom += values
        axis.set(
            xlabel="principal direction",
            ylabel="relative permittivity",
            title=f"{report['formula']}: dielectric tensor",
        )
        axis.legend()
        _save(figure, out, "dielectric.png")
    born = report.get("born_effective_charges")
    if not born:
        return _skip("born_charges.png", "the document has no Born effective charges")
    figure, axis = plt.subplots(figsize=(max(4, 0.6 * len(born) + 2), 4), constrained_layout=True)
    sites = np.arange(len(born))
    for component, (marker, label) in enumerate(zip("^sv", ("xx", "yy", "zz"), strict=True)):
        axis.plot(sites, [site["tensor"][component][component] for site in born], marker, label=f"Z*_{label}")
    axis.plot(sites, [site["trace"] / 3 for site in born], "ko", label="trace/3")
    axis.axhline(0, color="grey", linewidth=0.5)
    axis.set_xticks(sites, [f"{site['species']}{site['site']}" for site in born])
    axis.set(ylabel="Born effective charge (e)", title=f"{report['formula']}: Born effective charges per site")
    axis.legend()
    return _save(figure, out, "born_charges.png")


# --- elastic ---


def _elastic(output: dict):
    import numpy as np
    from pymatgen.analysis.elasticity import ElasticTensor

    raw = (output.get("elastic_tensor") or {}).get("raw")
    if raw is None:
        raise AnalysisError("the elastic document has no elastic_tensor.raw")
    stiffness = np.asarray(raw, dtype=float)
    # The fit is symmetric only to fitting precision (collect.py stores the symmetric part too).
    return ElasticTensor.from_voigt((stiffness + stiffness.T) / 2), float(np.abs(stiffness - stiffness.T).max())


def _report_elastic(output: dict, job: Job, out: Path) -> dict:
    import numpy as np

    tensor, asymmetry = _elastic(output)
    stiffness = tensor.voigt
    eigenvalues = np.linalg.eigvalsh(stiffness)
    k_hill, g_hill = tensor.k_vrh, tensor.g_vrh
    derived = output.get("derived_properties") or {}
    equilibrium = output.get("eq_stress")  # GPa, as atomate2 stores it
    report = {
        "formula": _structure(output["structure"]).composition.reduced_formula,
        "frame": "Cartesian frame of the relaxed structure (elastic_tensor.raw, not the IEEE frame)",
        "elastic_tensor_GPa": stiffness.tolist(),
        "fit_asymmetry_GPa": asymmetry,
        # ponytail: a singular fit (no compliance) ends the script with numpy's LinAlgError; catch it if one ever occurs.
        "compliance_tensor_per_GPa": np.linalg.inv(stiffness).tolist(),
        "stiffness_eigenvalues_GPa": eigenvalues.tolist(),
        # The zero-stress Born criterion; finite-stress criteria are not evaluated.
        "born_stable_zero_stress": bool(eigenvalues.min() > 0),
        "stability_criterion": "C positive definite (Born, zero stress); assumes negligible residual stress",
        "max_abs_equilibrium_stress_GPa": None if equilibrium is None else float(np.abs(equilibrium).max()),
        "bulk_modulus_GPa": {"voigt": tensor.k_voigt, "reuss": tensor.k_reuss, "hill": k_hill},
        "shear_modulus_GPa": {"voigt": tensor.g_voigt, "reuss": tensor.g_reuss, "hill": g_hill},
        "youngs_modulus_hill_GPa": 9 * k_hill * g_hill / (3 * k_hill + g_hill),
        "poisson_ratio_hill": (3 * k_hill - 2 * g_hill) / (2 * (3 * k_hill + g_hill)),
        "universal_anisotropy": tensor.universal_anisotropy,
        "relaxed_structure": _cell(_structure(output["structure"]), job),
    }
    # atomate2's derived properties, in its SI units, where it computed them.
    for key, name in (
        ("debye_temperature", "debye_temperature_K"),
        ("trans_v", "transverse_sound_velocity_m_per_s"),
        ("long_v", "longitudinal_sound_velocity_m_per_s"),
    ):
        if derived.get(key) is not None:
            report[name] = derived[key]
    return report


def _plots_elastic(output: dict, job: Job, out: Path) -> None:
    import numpy as np

    tensor, _ = _elastic(output)
    plt = _pyplot()
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    limit = float(np.abs(tensor.voigt).max()) or 1.0
    image = axis.imshow(tensor.voigt, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for (i, j), value in np.ndenumerate(tensor.voigt):
        axis.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(6), labels)
    axis.set_yticks(range(6), labels)
    axis.set_title("C_ij (GPa), relaxed-structure Cartesian frame")
    figure.colorbar(image, ax=axis, label="GPa")
    _save(figure, out, "elastic_tensor.png")

    # E(n) = 1 / (n_i n_j n_k n_l S_ijkl), from the full compliance tensor.
    compliance = np.asarray(tensor.compliance_tensor)
    angles = np.linspace(0, 2 * np.pi, 361)
    figure, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(12, 4), constrained_layout=True)
    planes = {}
    for axis, (name, (first, second)) in zip(axes, {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items(), strict=True):
        directions = np.zeros((len(angles), 3))
        directions[:, first], directions[:, second] = np.cos(angles), np.sin(angles)
        moduli = 1 / np.einsum("ijkl,ni,nj,nk,nl->n", compliance, directions, directions, directions, directions)
        planes[name] = moduli.tolist()
        axis.plot(angles, moduli)
        axis.set_title(f"{name} plane")
    figure.suptitle("Directional Young's modulus (GPa)")
    _write_json(out / "youngs_modulus.json", {"angles_deg": np.degrees(angles).tolist(), "youngs_modulus_GPa": planes})
    _save(figure, out, "youngs_modulus.png")

    fitting = output.get("fitting_data") or {}
    if not fitting.get("strains") or not fitting.get("pk_stresses"):
        return _skip("stress_strain.png", "the document has no fitting_data strains and stresses")
    from pymatgen.analysis.elasticity import Strain, Stress

    strains = np.array([Strain(strain).voigt for strain in fitting["strains"]])
    stresses = np.array([Stress(stress).voigt for stress in fitting["pk_stresses"]])
    equilibrium = Stress(output["eq_stress"]).voigt if output.get("eq_stress") else np.zeros(6)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    plotted = {}
    for column, axis in enumerate(axes.flat):
        # The deformations straining (mainly) this Voigt component.
        # ponytail: assumes atomate2's single-component strain states; a mixed strain set needs a per-deformation fit view.
        rows = np.abs(strains).argmax(axis=1) == column
        line = np.linspace(
            min(strains[rows, column].min(initial=0), -0.01), max(strains[rows, column].max(initial=0), 0.01), 2
        )
        for row in range(6):
            colour = f"C{row}"
            axis.plot(strains[rows, column], stresses[rows, row], "o", color=colour, label=f"σ_{labels[row]}")
            axis.plot(line, equilibrium[row] + tensor.voigt[row, column] * line, "-", color=colour, linewidth=0.8)
        axis.set(xlabel=f"strain ε_{labels[column]} (Voigt)", ylabel="PK2 stress (GPa)")
        plotted[labels[column]] = {
            "strain": strains[rows, column].tolist(),
            "stress_voigt_GPa": stresses[rows].tolist(),
            "line_slope_GPa": tensor.voigt[:, column].tolist(),
        }
    axes.flat[0].legend(fontsize=7)
    _write_json(out / "stress_strain.json", {"line_intercept_GPa": equilibrium.tolist(), "points": plotted})
    figure.suptitle("Stress-strain data of the fit (points) and the fitted C_ij (lines)")
    return _save(figure, out, "stress_strain.png")


# --- phonons ---


def _phonon_bands(output: dict):
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

    if not isinstance(output.get("phonon_bandstructure"), dict):
        raise AnalysisError("the phonon document has no phonon_bandstructure")
    return PhononBandStructureSymmLine.from_dict(output["phonon_bandstructure"])


def _report_phonons(output: dict, job: Job, out: Path) -> dict:
    import numpy as np

    bands = _phonon_bands(output)
    cell = bands.structure
    units = cell.composition.get_reduced_composition_and_factor()[1]
    branch, point = np.unravel_index(np.argmin(bands.bands), bands.bands.shape)
    lowest, frequency = bands.qpoints[point], float(bands.bands[branch, point])
    report = {
        "formula": cell.composition.reduced_formula,
        "cell": "phonopy's primitive cell (phonon_bandstructure.structure)",
        "phonon_cell": _cell(cell, job),
        "formula_units_per_phonon_cell": units,
        "supercell_matrix": output.get("supercell_matrix"),
        "has_imaginary_modes": output.get("has_imaginary_modes"),  # atomate2's flag, without a tolerance
        "lowest_frequency_THz": frequency,
        "imaginary_tolerance_THz": _IMAGINARY_TOLERANCE_THZ,
        "dynamically_stable": frequency >= -_IMAGINARY_TOLERANCE_THZ,
        "lowest_frequency_qpoint": {"label": lowest.label, "frac": lowest.frac_coords.tolist()},
        # atomate2 applies it for VASP only when the (symmetrized) Born charges are not all zero.
        "non_analytical_correction": output.get("born") is not None
        and output.get("epsilon_static") is not None
        and output.get("code") == "vasp"
        and not bool(np.all(np.isclose(output["born"], 0.0))),
    }
    temperatures = output.get("temperatures")
    if temperatures:
        series = {name: output[name] for name in ("free_energies", "internal_energies", "entropies", "heat_capacities")}
        if float(temperatures[0]) == 0.0:
            # As collect.py: J/mol per formula unit -> eV per phonon cell.
            report["zero_point_energy_eV_per_cell"] = series["free_energies"][0] * units * _EV_PER_JOULE_PER_MOL
        if not min(temperatures) <= 300 <= max(temperatures):
            report["at_300K_note"] = f"300 K is outside the sampled {min(temperatures)}-{max(temperatures)} K"
            return report
        report["at_300K_per_formula_unit"] = {
            "free_energy_J_per_mol": float(np.interp(300, temperatures, series["free_energies"])),
            "internal_energy_J_per_mol": float(np.interp(300, temperatures, series["internal_energies"])),
            "entropy_J_per_K_mol": float(np.interp(300, temperatures, series["entropies"])),
            "heat_capacity_J_per_K_mol": float(np.interp(300, temperatures, series["heat_capacities"])),
        }
    return report


def _plots_phonons(output: dict, job: Job, out: Path) -> None:
    from pymatgen.phonon.dos import PhononDos
    from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter

    bands = _phonon_bands(output)
    plt = _pyplot()
    _save(PhononBSPlotter(bands).get_plot(), out, "phonon_band_structure.png")
    if not isinstance(output.get("phonon_dos"), dict):
        _skip("phonon_dos.png", "the document has no phonon_dos")
        _skip("phonon_band_structure_dos.png", "no phonon DOS")
    else:
        dos = PhononDos.from_dict(output["phonon_dos"])
        plotter = PhononDosPlotter()
        plotter.add_dos("total", dos)
        _save(plotter.get_plot(), out, "phonon_dos.png")
        # Bands and DOS side by side on one frequency axis.
        data = PhononBSPlotter(bands).bs_plot_data()
        figure, (left, right) = plt.subplots(
            1, 2, sharey=True, figsize=(9, 5), width_ratios=[3, 1], constrained_layout=True
        )
        for distances, frequencies in zip(data["distances"], data["frequency"], strict=True):
            for branch in frequencies:
                left.plot(distances, branch, color="C0", linewidth=1)
        left.set_xticks(
            data["ticks"]["distance"],
            [label.replace("GAMMA", "Γ").replace("$\\Gamma$", "Γ") for label in data["ticks"]["label"]],
        )
        left.set_xlim(data["ticks"]["distance"][0], data["ticks"]["distance"][-1])
        left.axhline(0, color="grey", linewidth=0.5)
        left.set_ylabel("frequency (THz)")
        right.plot(dos.densities, dos.frequencies, color="C0")
        right.set_xlabel("DOS (states/THz per cell)")
        _save(figure, out, "phonon_band_structure_dos.png")
    temperatures = output.get("temperatures")
    if not temperatures:
        return _skip("phonon_thermodynamics.png", "the document has no thermodynamics")
    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 6), constrained_layout=True)
    for axis, (name, label) in zip(
        axes.flat,
        (
            ("free_energies", "F (J/mol)"),
            ("internal_energies", "U (J/mol)"),
            ("entropies", "S (J/(K mol))"),
            ("heat_capacities", "C_v (J/(K mol))"),
        ),
        strict=True,
    ):
        axis.plot(temperatures, output[name])
        axis.set_ylabel(label)
    for axis in axes[1]:
        axis.set_xlabel("temperature (K)")
    figure.suptitle("Harmonic vibrational thermodynamics per formula unit (atomate2's units)")
    return _save(figure, out, "phonon_thermodynamics.png")


# --- full characterization ---


def _children(job: Job, record: dict) -> dict[str, Job | str | AnalysisError]:
    """Each child label's job, the reason a child that did not succeed is not analysed, or a succeeded one's load error."""

    from httk.workflow import Workspace
    from httk.workflow.introspection import resolve_job

    workspace = Workspace(os.environ["HTTK_WORKFLOW_WORKSPACE_DIR"])
    children: dict[str, Job | str | AnalysisError] = {}
    for label, part in record["parts"].items():
        if part["kind"] != "succeeded":
            failure = part.get("failure") or {}
            children[label] = f"{part['kind']}" + (
                f" ({failure.get('code')}: {failure.get('message')})" if failure else ""
            )
            continue
        try:
            children[label] = Job.at(resolve_job(workspace, part["job_id"]).path)
        except (ValueError, OSError) as exc:
            children[label] = AnalysisError(f"cannot load the succeeded child: {exc}")
    return children


def _run_children(job: Job, out: Path, scripts: tuple[str, ...]) -> tuple[dict, dict]:
    """Run *scripts* for every succeeded child into ``<label>/``; return their reports, every child's state and
    the analyses that failed."""

    record = json.loads(job.result("characterization.json").read_text(encoding="utf-8"))
    reports, states, failed = {}, {}, []
    for label, child in _children(job, record).items():
        # Labels name output directories: a job-written one like ../x must not leave the output directory.
        if not re.fullmatch(r"[a-z_]+", label):
            failed.append(f"{label} (invalid label)")
            print(f"{label!r}: not analysed: a child label must match [a-z_]+", file=sys.stderr)
            continue
        if isinstance(child, str):
            states[label] = child
            print(f"{label}: not analysed: {child}")
            continue
        if isinstance(child, AnalysisError):
            states[label] = f"succeeded; {child}"
            failed.append(f"{label} load")
            print(f"{label}: {child}", file=sys.stderr)
            continue
        states[label] = "succeeded"
        directory = out / label
        directory.mkdir(exist_ok=True)
        for script in scripts:
            try:
                result = _run(script, child, directory)
            except Exception as exc:  # noqa: BLE001 (one child's analysis must not stop the others)
                states[label] = f"succeeded; {script} failed: {exc}"
                failed.append(f"{label} {script}")
                print(f"{label}: {script} failed: {exc}", file=sys.stderr)
                continue
            if script == "report":
                reports[label] = result
    return reports, {"record": record, "states": states, "failed": failed}


def _report_full(job: Job, out: Path) -> dict:
    reports, extra = _run_children(job, out, ("report",))
    record, relax = extra["record"], reports.get("relax") or {}
    dielectric, elastic, phonons = (
        reports.get("dielectric") or {},
        reports.get("elastic") or {},
        reports.get("phonons") or {},
    )
    return {
        "formula": relax.get("formula"),
        "relaxed_structure": relax.get("relaxed"),
        "energy_per_atom_eV": relax.get("energy_per_atom_eV"),
        "band_gap_eV": (reports.get("band_structure") or {}).get("band_gap_eV"),
        "direct_gap": (reports.get("band_structure") or {}).get("direct_gap"),
        "epsilon_electronic_average": (dielectric.get("averages") or {}).get("electronic"),
        "epsilon_static_average": (dielectric.get("averages") or {}).get("total"),
        "bulk_modulus_hill_GPa": (elastic.get("bulk_modulus_GPa") or {}).get("hill"),
        "shear_modulus_hill_GPa": (elastic.get("shear_modulus_GPa") or {}).get("hill"),
        "youngs_modulus_hill_GPa": elastic.get("youngs_modulus_hill_GPa"),
        "born_stable_zero_stress": elastic.get("born_stable_zero_stress"),
        "max_abs_equilibrium_stress_GPa": elastic.get("max_abs_equilibrium_stress_GPa"),
        "phonon_imaginary_modes": phonons.get("has_imaginary_modes"),
        "phonons_dynamically_stable": phonons.get("dynamically_stable"),
        "zero_point_energy_eV_per_phonon_cell": phonons.get("zero_point_energy_eV_per_cell"),
        "decisions": record.get("decisions"),
        "children": {
            label: {"state": state, "job_id": record["parts"][label]["job_id"]}
            for label, state in extra["states"].items()
        },
        "failed_analyses": extra["failed"],
    }


# --- dispatch ---

_RELAX = (_report_relax, _plots_relax)
_ANALYSES = {
    "atomate2.static": (_report_static, _plots_static),
    "atomate2.relax": _RELAX,
    "atomate2.tight-relax": _RELAX,
    "atomate2.double-relax": _RELAX,
    "atomate2.band-structure": (_report_band_structure, _plots_band_structure),
    "atomate2.dielectric": (_report_dielectric, _plots_dielectric),
    "atomate2.elastic": (_report_elastic, _plots_elastic),
    "atomate2.phonons": (_report_phonons, _plots_phonons),
}
_FULL = "atomate2.full-characterization"


def _run(script: str, job: Job, out: Path) -> dict | None:
    """Run *script* for *job* into *out*; return the report for ``report``."""

    if job.workflow == _FULL:
        if script == "plots":
            failed = _run_children(job, out, ("report", "plots"))[1]["failed"]
            if failed:
                raise AnalysisError(f"child analyses failed: {', '.join(failed)}")
            return None
        report = {"workflow": job.workflow, "job_id": job.spec.get("id")} | _report_full(job, out)
    elif job.workflow in _ANALYSES:
        output = job.flow_output()
        if script == "plots":
            return _ANALYSES[job.workflow][1](output, job, out)
        report = {"workflow": job.workflow, "job_id": job.spec.get("id")}
        report |= _ANALYSES[job.workflow][0](output, job, out)
    else:
        raise AnalysisError(f"no analysis for workflow {job.workflow!r}")
    _write_json(out / "report.json", report)
    (out / "report.txt").write_text("\n".join(_text(report)) + "\n", encoding="utf-8")
    print("wrote report.json and report.txt")
    if report.get("failed_analyses"):
        raise AnalysisError(f"child analyses failed: {', '.join(report['failed_analyses'])}")
    return report


def main(script: str) -> int:
    """Run the ``report`` or ``plots`` script for the job the postprocess environment names.

    :param script: ``"report"`` or ``"plots"``.
    :return: The exit status: 0 on success or for a jobflow child job (silently), 1 when the job cannot be analysed.
    """
    job = Job.at(
        Path(os.environ["HTTK_WORKFLOW_JOB_DIR"]),
        Path(os.environ["HTTK_WORKFLOW_DATA_DIR"]) if os.environ.get("HTTK_WORKFLOW_DATA_DIR") else None,
        Path(os.environ["HTTK_WORKFLOW_WORKDIR"]) if os.environ.get("HTTK_WORKFLOW_WORKDIR") else None,
    )
    if job.parameters.get("jobflow_child_job") is not None:
        return 0  # one atomate2 job of a flow: its parent's flow output holds the results
    try:
        _run(script, job, Path.cwd())
    except AnalysisError as exc:
        print(f"{script}: {exc}", file=sys.stderr)
        return 1
    return 0
