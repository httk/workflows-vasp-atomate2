"""A light LaTeX report of an ``atomate2.full-characterization`` job (only this package carries this file).

``scripts/latex`` calls :func:`main`, which runs the shared ``report`` and ``plots``
analyses (``atomate2_postprocess.py``) into ``analysis/``, copies the figures it
uses into ``figures/``, reads the VASP runs of the job tree, and writes
``report.tex``: the data as tables and figures, with only the sentences that follow
mechanically from the results (no experimental comparison, no interpretation). It
compiles ``report.pdf`` with ``latexmk`` when that is on the PATH. Everything is
written to the current directory, the postprocess output directory; the job tree
is only read, and VASP files only below each run's ``run/exec/``.
"""

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from atomate2_postprocess import _FULL, AnalysisError, Job, _phonon_bands, _run

#: Bound on what is read of one job-written file (the OUTCAR is streamed line by line instead).
_MAX_BYTES = 4 << 20
_MAX_LINE = 1 << 16
_THZ_TO_PER_CM = 33.35641
#: The figures of each child the report shows (the relaxation and dielectric figures only repeat the tables).
_FIGURES = {
    "band_structure": {"band_structure_dos": "Band structure along the line path and DOS of the uniform mesh."},
    "elastic": {
        "stress_strain": "Stress-strain data of the fit (points) and the fitted $C_{ij}$ (lines).",
        "youngs_modulus": "Directional Young's modulus in the xy, xz and yz planes (GPa).",
    },
    "phonons": {
        "phonon_band_structure_dos": "Phonon band structure and DOS.",
        "phonon_thermodynamics": "Harmonic vibrational thermodynamics per formula unit.",
    },
}
_SECTIONS = {
    "relax": "Structure",
    "band_structure": "Electronic structure",
    "elastic": "Elastic properties",
    "dielectric": "Dielectric response",
    "phonons": "Lattice dynamics",
}
_ESCAPES = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
        "|": r"\textbar{}",
        "Γ": r"$\Gamma$",
        # pdflatex refuses control characters ("invalid character"); a line break would end a table cell's paragraph.
        **dict.fromkeys(map(chr, [*range(0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x7F]), "?"),
        "\n": " ",
        "\r": " ",
    }
)


# --- formatting ---


def _tex(text: object) -> str:
    """*text* with TeX's special characters escaped; control and non-ASCII characters become ``?``."""

    # ponytail: pdflatex refuses unmapped Unicode; map more characters here if job data ever carries them.
    return str(text).translate(_ESCAPES).encode("ascii", "replace").decode("ascii")


def _label(text: object) -> str:
    """An escaped k-point or q-point label (or a transition between them), with Γ as ``$\\Gamma$``."""

    if text is None:
        return "--"
    return _tex(text).replace(r"\textbackslash{}Gamma", r"$\Gamma$").replace("GAMMA", r"$\Gamma$")


def _formula(text: object) -> str:
    return re.sub(r"(\d+)", r"$_{\1}$", _tex(text))


def _num(value: object, digits: int = 3) -> str:
    """*value* with *digits* decimals (``--`` for ``None``, yes/no for a boolean), with a true minus sign."""

    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", r"\textminus{}")


def _table(columns: str, rows: list[list[str]], header: list[str] | None = None, caption: str = "") -> list[str]:
    lines = [r"\begin{table}[H]", r"\centering\small", rf"\begin{{tabular}}{{{columns}}}", r"\toprule"]
    if header:
        lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines + ([rf"\caption{{{caption}}}"] if caption else []) + [r"\end{table}", ""]


def _figure(figures: set[str], label: str, name: str) -> list[str]:
    file = f"{label}_{name}.png"
    if file not in figures:
        return []
    width = "0.85" if name == "phonon_thermodynamics" else ""
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width={width}\textwidth]{{figures/{file}}}",
        rf"\caption{{{_FIGURES[label][name]}}}",
        r"\end{figure}",
        "",
    ]


def _sorted_unique(values: list[str]) -> str:
    """Distinct *values*, numbers in numeric order, as one comma-separated escaped string."""

    def key(value: str) -> tuple[float, str]:
        try:
            return float(value), value
        except ValueError:
            return float("inf"), value

    # A true minus, which a line never breaks after (-5/0.05 is one value).
    return _tex(", ".join(sorted(set(values), key=key))).replace("-", r"\textminus{}")


# --- the VASP runs of the job tree ---


def _file(directory: Path, name: str) -> Path | None:
    """*name* or *name*.gz in *directory* (resolved), refusing a link that leaves it."""

    for path in (directory / name, directory / f"{name}.gz"):
        if path.is_file() and path.resolve().is_relative_to(directory):
            return path
    return None


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _read(directory: Path, name: str) -> str | None:
    """At most ``_MAX_BYTES`` characters of *name* (or *name*.gz) in *directory*; ``None`` when absent or unreadable."""

    path = _file(directory, name)
    if path is None:
        return None
    try:
        with _open(path) as handle:
            return handle.read(_MAX_BYTES)
    except (OSError, EOFError):
        return None


def _incar(text: str | None) -> dict[str, str]:
    tags = {}
    for line in (text or "").splitlines():
        for part in re.split(r"[!#]", line, maxsplit=1)[0].split(";"):
            key, equals, value = part.partition("=")
            if equals and key.strip():
                tags[key.strip().upper()] = value.strip()
    return tags


def _kpoints(text: str | None) -> str | None:
    """The mesh of a KPOINTS file (``Gamma 11x11x11``), its line path's point count, or its explicit list's."""

    lines = (text or "").splitlines()
    try:
        count, style = int(lines[1].split()[0]), lines[2].strip()[:1].lower()
        if style == "l":
            segments = len([line for line in lines[4:] if line.strip()]) // 2
            return f"line path, {count * segments} points"
        if count == 0 and style in ("g", "m"):
            return ("Gamma " if style == "g" else "MP ") + "x".join(lines[3].split()[:3])
        if count == 0:
            return f"automatic, {lines[3].strip()}"
        return f"{count} explicit points"
    except (IndexError, ValueError):
        return None


def _outcar(directory: Path) -> dict:
    """The VASP version, POTCAR titles, NKPTS and elapsed time of the run's OUTCAR, streamed line by line."""

    found: dict = {"vasp": None, "potcar": [], "nkpts": None, "elapsed_s": None}
    path = _file(directory, "OUTCAR")
    if path is None:
        return found
    try:
        with _open(path) as handle:
            for line in iter(lambda: handle.readline(_MAX_LINE), ""):
                text = line.strip()
                if found["vasp"] is None and text.startswith("vasp."):
                    found["vasp"] = " ".join(text.split()[:2])
                elif text.startswith("POTCAR:") and text[7:].strip() not in found["potcar"]:
                    found["potcar"].append(text[7:].strip())
                elif found["nkpts"] is None and (match := re.search(r"NKPTS\s*=\s*(\d+)", text)):
                    found["nkpts"] = int(match[1])
                elif text.startswith("Elapsed time (sec):"):
                    found["elapsed_s"] = float(text.partition(":")[2])
    except (OSError, EOFError, ValueError):
        pass  # what was read stands; the rest stays empty
    return found


def _corrections(text: str | None) -> int | None:
    try:
        jobs = json.loads(text or "")
        return sum(len(job.get("corrections") or []) for job in jobs)
    except (ValueError, TypeError, AttributeError):
        return None


def _plan(text: str | None) -> str:
    """The launch's parallelization plan: the last ``parallel:`` line of ``httk-vasp-launch.log``."""

    lines = [line for line in (text or "").splitlines() if "parallel:" in line]
    if not lines:
        return ""
    line = lines[-1].partition("parallel:")[2]
    if match := re.search(r"KPAR=(\d+) NCORE=(\d+)", line):
        return f"KPAR={match[1]} NCORE={match[2]}"
    for marker in ("left to VASP:", "INCAR sets"):
        if marker in line:
            return _tex(marker + " " + line.partition(marker)[2].partition(", ranks from")[0].strip())
    return _tex(line.partition(";")[0].strip())


def _vasp_runs(record: dict) -> list[dict]:
    """Each VASP run of the tree: the jobflow jobs of every succeeded child whose ``run/exec/`` holds an INCAR."""

    from httk.workflow import Workspace
    from httk.workflow.introspection import read_state, resolve_job

    workspace = Workspace(os.environ["HTTK_WORKFLOW_WORKSPACE_DIR"])
    runs = []
    for label, part in record["parts"].items():
        if part.get("kind") != "succeeded":
            continue
        try:
            state, damage = read_state(resolve_job(workspace, part["job_id"]))
        except (KeyError, ValueError, OSError) as exc:
            print(f"{label}: cannot read its jobs: {exc}")
            continue
        if damage:
            print(f"{label}: its state.json is damaged: {damage}")
        for entry in state.children if state is not None else ():
            try:
                path = resolve_job(workspace, entry["job_id"]).path.resolve()
            except (KeyError, ValueError, OSError) as exc:
                print(f"{label}: cannot resolve {entry.get('job_key')}: {exc}")
                continue
            directory = (path / "run" / "exec").resolve()
            if not directory.is_relative_to(path) or _file(directory, "INCAR") is None:
                continue
            runs.append(
                {
                    "label": label,
                    "key": entry.get("job_key") or path.name,
                    "incar": _incar(_read(directory, "INCAR")),
                    "kpoints": _kpoints(_read(directory, "KPOINTS")),
                    "corrections": _corrections(_read(directory, "custodian.json")),
                    "plan": _plan(_read(directory, "httk-vasp-launch.log")),
                }
                | _outcar(directory)
            )
    return runs


# --- sections ---


def _title(report: dict) -> list[str]:
    return [
        rf"\title{{Characterization of {_formula(report.get('formula') or '--')}\\[0.3em]",
        rf"\large \texttt{{{_tex(report['workflow'])}}}, job \texttt{{{_tex(report['job_id'])}}}}}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]


def _overview(report: dict, failures: list[str]) -> list[str]:
    decisions = report.get("decisions") or {}
    lines = [r"\section{Overview}", ""]
    lines += _table(
        "lp{10cm}",
        [
            ["Band gap used (eV)", _tex(json.dumps(decisions.get("band_gap")))],
            ["Band-gap tolerance (eV)", _tex(json.dumps(decisions.get("band_gap_tolerance")))],
            ["Symmetry precision", _tex(json.dumps(decisions.get("symprec")))],
            ["Insulator", _num(decisions.get("insulator"))],
            ["Magnetic", _num(decisions.get("magnetic"))],
            [
                "Child parameters",
                r"\texttt{" + _tex(json.dumps(decisions.get("child_parameters"), sort_keys=True)) + "}",
            ],
        ],
        caption="Decisions of the parent workflow.",
    )
    if decisions.get("notes"):
        lines += [r"\begin{itemize}", *(rf"\item {_tex(note)}" for note in decisions["notes"]), r"\end{itemize}", ""]
    lines += _table(
        "lp{6cm}l",
        [
            [_tex(label), _tex(child["state"]), r"\texttt{" + _tex(child["job_id"]) + "}"]
            for label, child in (report.get("children") or {}).items()
        ],
        ["Child", "State", "Job id"],
        "Child jobs.",
    )
    if failures:
        lines += [r"Failed analyses:", r"\begin{itemize}", *(rf"\item {_tex(item)}" for item in failures)]
        lines += [r"\end{itemize}", ""]
    return lines


def _settings(runs: list[dict]) -> list[str]:
    if not runs:
        return [r"\section{Calculation settings}", "", "No VASP run was found in the succeeded children.", ""]
    rows = []
    for label in dict.fromkeys(run["label"] for run in runs):
        mine = [run for run in runs if run["label"] == label]
        tags = [run["incar"] for run in mine]
        rows.append(
            [
                _tex(label),
                str(len(mine)),
                _sorted_unique(
                    [f"METAGGA={t['METAGGA']}" if "METAGGA" in t else t.get("GGA", "POTCAR default") for t in tags]
                ),
                _sorted_unique([_value(t["ENCUT"]) for t in tags if "ENCUT" in t]),
                _sorted_unique(
                    [f"{_value(t.get('ISMEAR', 'default'))}/{_value(t.get('SIGMA', 'default'))}" for t in tags]
                ),
                _sorted_unique([_value(t["ISPIN"]) for t in tags if "ISPIN" in t]),
                _sorted_unique([_value(t["EDIFF"]) for t in tags if "EDIFF" in t]),
                _sorted_unique([run["kpoints"] for run in mine if run["kpoints"]]),
            ]
        )
    lines = [r"\section{Calculation settings}", "", r"{\setlength{\tabcolsep}{4pt}"]
    lines += _table(
        "lrllp{2.1cm}llp{3cm}",
        rows,
        ["Child", "Runs", "XC", "ENCUT (eV)", "ISMEAR/ SIGMA", "ISPIN", "EDIFF (eV)", "k-points"],
        "INCAR and KPOINTS settings of the VASP runs, distinct values per child.",
    )
    versions = list(dict.fromkeys(run["vasp"] for run in runs if run["vasp"]))
    potcars = list(dict.fromkeys(title for run in runs for title in run["potcar"]))
    lines += ["}", rf"VASP: {_tex(', '.join(versions) or '--')}. POTCAR: {_tex(', '.join(potcars) or '--')}.", ""]
    return lines


def _value(text: str) -> str:
    """An INCAR value, a number in its shortest form (``700.0`` as ``700``)."""

    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _structure(report: dict) -> list[str]:
    initial, relaxed = report["initial"], report["relaxed"]

    def group(cell: dict) -> str:
        return f"{_tex(cell['space_group'])} (No.~{cell['space_group_number']})"

    rows = [["Space group", group(initial), group(relaxed)]]
    for key, name, digits in (
        ("a_angstrom", r"$a$ (\AA)", 5),
        ("b_angstrom", r"$b$ (\AA)", 5),
        ("c_angstrom", r"$c$ (\AA)", 5),
        ("alpha_deg", r"$\alpha$ (deg)", 3),
        ("beta_deg", r"$\beta$ (deg)", 3),
        ("gamma_deg", r"$\gamma$ (deg)", 3),
        ("volume_angstrom3", r"Volume (\AA$^3$)", 4),
    ):
        rows.append([name, _num(initial[key], digits), _num(relaxed[key], digits)])
    stress = report.get("stress_kbar")
    max_stress = None if not stress else max(abs(x) for row in stress for x in row) / 10
    lines = [r"\section{Structure}", ""]
    lines += _table(
        "lrr", rows, ["", "Initial", "Relaxed"], f"Initial ({_tex(report['initial_source'])}) and relaxed cell."
    )
    lines += _table(
        "lr",
        [
            ["Energy per atom (eV)", _num(report.get("energy_per_atom_eV"), 5)],
            [r"Relative volume change (\%)", _num(100 * report["relative_volume_change"], 3)],
            [r"Maximum force (eV/\AA)", _num(report.get("max_force_eV_per_angstrom"), 4)],
            ["Maximum stress component (GPa)", _num(max_stress, 4)],
        ],
        caption="The relaxed structure.",
    )
    if initial["space_group"] == relaxed["space_group"]:
        lines += [f"The space group is preserved ({_tex(relaxed['space_group'])}).", ""]
    else:
        lines += [f"The space group changes from {group(initial)} to {group(relaxed)}.", ""]
    return lines


def _electronic(report: dict, figures: set[str]) -> list[str]:
    def kind(direct: object) -> str:
        return "--" if direct is None else ("direct" if direct else "indirect")

    gap = report.get("band_gap_eV")
    lines = [r"\section{Electronic structure}", ""]
    lines += _table(
        "lll",
        [
            ["Band gap (eV)", _num(gap, 4), _num(report.get("line_band_gap_eV"), 4)],
            ["Gap type", kind(report.get("direct_gap")), kind(report.get("line_direct_gap"))],
            ["Transition", _label(report.get("transition")), _label(report.get("line_transition"))],
            ["VBM (eV)", _num(report.get("vbm_eV"), 4), _num((report.get("line_vbm") or {}).get("energy_eV"), 4)],
            ["CBM (eV)", _num(report.get("cbm_eV"), 4), _num((report.get("line_cbm") or {}).get("energy_eV"), 4)],
            ["Fermi energy (eV)", _num(report.get("fermi_energy_eV"), 4), _num(report.get("line_fermi_energy_eV"), 4)],
        ],
        ["", "Uniform mesh", "Line path"],
        "Band edges of the uniform mesh and of the line path.",
    )
    if gap is not None:
        lines += [
            "The material is metallic (no gap)."
            if gap == 0
            else f"The band gap is {kind(report.get('direct_gap'))}, {_num(gap, 4)} eV.",
            "",
        ]
    return lines + _figure(figures, "band_structure", "band_structure_dos")


def _elastic(report: dict, figures: set[str]) -> list[str]:
    stiffness = report["elastic_tensor_GPa"]
    floor = 1e-6 * max(abs(x) for row in stiffness for x in row)
    rows = [[str(i + 1), *(_num(0.0 if abs(x) < floor else x, 1) for x in row)] for i, row in enumerate(stiffness)]
    lines = [r"\section{Elastic properties}", ""]
    lines += _table(
        "c" + "r" * 6,
        rows,
        ["$C_{ij}$", *map(str, range(1, 7))],
        "Elastic tensor (GPa), Voigt order xx yy zz yz xz xy.",
    )

    def vrh(moduli: dict) -> str:
        return " / ".join(_num(moduli.get(kind), 1) for kind in ("voigt", "reuss", "hill"))

    lines += _table(
        "lr",
        [
            ["Bulk modulus $B$, Voigt / Reuss / Hill (GPa)", vrh(report["bulk_modulus_GPa"])],
            ["Shear modulus $G$, Voigt / Reuss / Hill (GPa)", vrh(report["shear_modulus_GPa"])],
            ["Young's modulus, Hill (GPa)", _num(report.get("youngs_modulus_hill_GPa"), 1)],
            ["Poisson ratio, Hill", _num(report.get("poisson_ratio_hill"), 3)],
            ["Universal anisotropy", _num(report.get("universal_anisotropy"), 3)],
            ["Debye temperature (K)", _num(report.get("debye_temperature_K"), 0)],
            ["Transverse sound velocity (m/s)", _num(report.get("transverse_sound_velocity_m_per_s"), 0)],
            ["Longitudinal sound velocity (m/s)", _num(report.get("longitudinal_sound_velocity_m_per_s"), 0)],
            ["Stiffness eigenvalues (GPa)", ", ".join(_num(x, 1) for x in report["stiffness_eigenvalues_GPa"])],
        ],
        caption="Moduli derived from the elastic tensor.",
    )
    if report.get("born_stable_zero_stress"):
        sentence = (
            "All stiffness eigenvalues are positive: the zero-stress Born criterion for mechanical stability holds."
        )
    else:
        sentence = (
            "Not all stiffness eigenvalues are positive: the zero-stress Born criterion for mechanical stability fails."
        )
    stress = report.get("max_abs_equilibrium_stress_GPa")
    sentence += (
        " The document records no residual equilibrium stress."
        if stress is None
        else f" The largest residual equilibrium stress component is {_num(stress, 4)} GPa."
    )
    lines += [sentence, ""]
    return lines + _figure(figures, "elastic", "stress_strain") + _figure(figures, "elastic", "youngs_modulus")


def _dielectric(report: dict) -> list[str]:
    eigenvalues, averages = report.get("eigenvalues") or {}, report.get("averages") or {}
    names = {"electronic": r"Electronic ($\varepsilon^\infty$)", "ionic": "Ionic", "total": r"Total ($\varepsilon^0$)"}
    rows = [
        [names[kind], *(_num(x, 3) for x in values), _num(averages.get(kind), 3)]
        for kind, values in eigenvalues.items()
    ]
    lines = [r"\section{Dielectric response}", ""]
    lines += _table(
        "lrrrr",
        rows,
        ["", r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", "Average"],
        "Eigenvalues and averages of the relative permittivity tensors.",
    )
    lines += [f"Refractive index $\\sqrt{{\\bar\\varepsilon^\\infty}}$: {_num(report.get('refractive_index'), 3)}.", ""]
    born = report.get("born_effective_charges") or []
    if born:
        lines += _table(
            "rlr",
            [[str(site["site"]), _tex(site["species"]), _num(site["trace"], 3)] for site in born],
            ["Site", "Species", "Trace (e)"],
            "Traces of the Born effective charge tensors.",
        )
    piezo = report.get("piezoelectric_tensor_C_per_m2")
    sentences = []
    if "ionic" in eigenvalues and all(abs(x) < 1e-3 for x in eigenvalues["ionic"]):
        sentences.append(r"The ionic contribution vanishes ($\varepsilon^0 = \varepsilon^\infty$).")
    if born and all(abs(x) < 1e-3 for site in born for row in site["tensor"] for x in row):
        sentences.append("The Born effective charges vanish.")
    if piezo is not None and all(abs(x) < 1e-6 for row in piezo for x in row):
        sentences.append("The piezoelectric tensor vanishes.")
    elif piezo is not None:
        lines += _table(
            "c" + "r" * 6,
            [[axis, *(_num(x, 4) for x in row)] for axis, row in zip("xyz", piezo, strict=True)],
            ["", "xx", "yy", "zz", "yz", "xz", "xy"],
            "Piezoelectric tensor (C/m$^2$, relaxed-ion).",
        )
    return lines + ([" ".join(sentences), ""] if sentences else [])


def _qpoint_frequencies(job_id: str) -> list[list[str]] | str:
    """Rows (point, mode, THz, cm⁻¹) at each labelled q-point's first occurrence, or why there are none."""

    from httk.workflow import Workspace
    from httk.workflow.introspection import resolve_job

    try:
        path = resolve_job(Workspace(os.environ["HTTK_WORKFLOW_WORKSPACE_DIR"]), job_id).path
        bands = _phonon_bands(Job.at(path).flow_output())
    except (AnalysisError, KeyError, ValueError, OSError) as exc:
        return f"No q-point frequencies: {exc}"
    rows, seen = [], set()
    for index, qpoint in enumerate(bands.qpoints):
        if not qpoint.label or qpoint.label in seen:
            continue
        seen.add(qpoint.label)
        for mode, frequency in enumerate(sorted(float(f) for f in bands.bands[:, index])):
            rows.append(
                [
                    _label(qpoint.label) if mode == 0 else "",
                    str(mode + 1),
                    _num(frequency, 2),
                    _num(frequency * _THZ_TO_PER_CM, 1),
                ]
            )
    return rows


def _phonons(report: dict, job_id: str, figures: set[str]) -> list[str]:
    lowest, tolerance = report["lowest_frequency_THz"], report.get("imaginary_tolerance_THz")
    point = report["lowest_frequency_qpoint"]
    where = (
        _label(point["label"]) if point.get("label") else "q = (" + ", ".join(_num(x, 3) for x in point["frac"]) + ")"
    )
    matrix = report.get("supercell_matrix")
    lines = [r"\section{Lattice dynamics}", ""]
    lines += _table(
        "ll",
        [
            ["Supercell matrix", "; ".join(" ".join(_num(x, 0) for x in row) for row in matrix) if matrix else "--"],
            ["Formula units per phonon cell", str(report.get("formula_units_per_phonon_cell"))],
            ["Non-analytical correction applied", _num(report.get("non_analytical_correction"))],
            ["Lowest frequency (THz)", f"{_num(lowest, 4)} at {where}"],
        ],
        caption=_tex(f"Phonon calculation: {report.get('cell')}."),
    )
    if report.get("dynamically_stable"):
        lines.append(
            f"No imaginary frequency exceeds {_num(tolerance, 2)} THz in magnitude; the lowest, {_num(lowest, 4)} THz"
            f" at {where}, is within the tolerance: dynamically stable."
        )
    elif report.get("dynamically_stable") is False:
        lines.append(
            f"The lowest frequency, {_num(lowest, 4)} THz at {where}, is imaginary beyond {_num(tolerance, 2)} THz:"
            " dynamically unstable."
        )
    lines.append("")
    rows = _qpoint_frequencies(job_id)
    if isinstance(rows, str):
        lines += [_tex(rows), ""]
    else:
        lines += _table(
            "clrr",
            rows,
            ["Point", "Mode", "THz", r"cm$^{-1}$"],
            r"Frequencies at the labelled q-points of the phonon band structure (1 THz = 33.35641 cm$^{-1}$).",
        )
    at_300 = report.get("at_300K_per_formula_unit")
    thermodynamics = [["Zero-point energy (eV per phonon cell)", _num(report.get("zero_point_energy_eV_per_cell"), 4)]]
    if at_300:
        thermodynamics += [
            ["$F$ at 300 K (kJ/mol)", _num(at_300["free_energy_J_per_mol"] / 1000, 3)],
            ["$U$ at 300 K (kJ/mol)", _num(at_300["internal_energy_J_per_mol"] / 1000, 3)],
            ["$S$ at 300 K (J/(K mol))", _num(at_300["entropy_J_per_K_mol"], 3)],
            ["$C_v$ at 300 K (J/(K mol))", _num(at_300["heat_capacity_J_per_K_mol"], 3)],
        ]
    lines += _table("lr", thermodynamics, caption="Harmonic thermodynamics; 300 K values per formula unit.")
    if report.get("at_300K_note"):
        lines += [_tex(report["at_300K_note"]) + ".", ""]
    return (
        lines
        + _figure(figures, "phonons", "phonon_band_structure_dos")
        + _figure(figures, "phonons", "phonon_thermodynamics")
    )


def _execution(runs: list[dict]) -> list[str]:
    rows = [
        [
            _tex(run["label"]),
            r"\texttt{" + _tex(run["key"][:16]) + "}",
            "" if run["nkpts"] is None else str(run["nkpts"]),
            _num(run["elapsed_s"], 1) if run["elapsed_s"] is not None else "",
            "" if run["corrections"] is None else str(run["corrections"]),
            run["plan"],
        ]
        for run in runs
    ]
    elapsed = [run["elapsed_s"] for run in runs if run["elapsed_s"] is not None]
    corrections = [run["corrections"] for run in runs if run["corrections"] is not None]
    lines = [r"\appendix", r"\section{Execution}", ""]
    if rows:
        lines += _table(
            "llrrrp{3.6cm}",
            rows,
            ["Child", "Job", "NKPTS", "Elapsed (s)", "Corrections", "Parallelization"],
            "The VASP runs: k-points, elapsed time (OUTCAR), custodian corrections, launch plan.",
        )
    missing = len(runs) - len(elapsed)
    lines += [
        f"VASP runs: {len(runs)}. Elapsed time: {_num(sum(elapsed), 1)} s"
        + (f" ({missing} runs without one)" if missing else "")
        + f". Custodian corrections: {sum(corrections)}.",
        "",
    ]
    return lines


def _write(report: dict, children: dict[str, dict], runs: list[dict], figures: set[str], failures: list[str]) -> str:
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2.4cm]{geometry}",
        *(rf"\usepackage{{{name}}}" for name in ("amsmath", "booktabs", "graphicx", "float", "textcomp")),
        r"\usepackage[hidelinks]{hyperref}",
        "",
        *_title(report),
        *_overview(report, failures),
        *_settings(runs),
    ]
    states = report.get("children") or {}
    for label, title in _SECTIONS.items():
        if label in children:
            if label == "relax":
                lines += _structure(children[label])
            elif label == "band_structure":
                lines += _electronic(children[label], figures)
            elif label == "elastic":
                lines += _elastic(children[label], figures)
            elif label == "dielectric":
                lines += _dielectric(children[label])
            else:
                lines += _phonons(children[label], states[label]["job_id"], figures)
        elif label in states:
            lines += [
                rf"\noindent\textbf{{{title}:}} not reported (child \texttt{{{_tex(label)}}}: {_tex(states[label]['state'])}).",
                "",
            ]
    lines += [*_execution(runs), r"\end{document}"]
    return "\n".join(lines) + "\n"


# --- entry point ---


def _compile(out: Path) -> bool:
    """Compile report.tex with latexmk; ``False`` (after saying why) when that fails or latexmk is missing."""

    if shutil.which("latexmk") is None:
        print("report.pdf not compiled: latexmk is not on the PATH; compile report.tex with 'latexmk -pdf report.tex'")
        return True
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
    try:
        status = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=600, check=False).returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    if status == 0:
        print("wrote report.tex and report.pdf")
        return True
    print(f"report.pdf not compiled: latexmk failed (exit {status})", file=sys.stderr)
    log = out / "report.log"
    if log.is_file():
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]), file=sys.stderr)
    return False


#: How the shared module's AnalysisError lists the failed child analyses.
_FAILED = "child analyses failed: "


def _clean(out: Path, *directories: Path) -> str | None:
    """Remove a previous run's *directories* and report.pdf and recreate the directories; why not, when refused."""

    # A rerun starts clean: stale analyses, figures or a stale PDF would contradict the new report.
    for directory in directories:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            return f"{directory} is not a directory this script made; remove it"
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.mkdir()
        except OSError as exc:
            return f"cannot recreate {directory} (a previous run's files could not all be removed): {exc}"
    try:
        (out / "report.pdf").unlink(missing_ok=True)
    except OSError as exc:
        return f"cannot remove the previous report.pdf: {exc}"
    return None


def main() -> int:
    """Write report.tex (and report.pdf) of the ``atomate2.full-characterization`` job the postprocess environment names.

    :return: The exit status: 0 on success or for a jobflow child job (silently); 1 for another workflow, when the
        parent cannot be analysed (no report.tex), or after writing everything when a child analysis or the
        compilation failed.
    """
    job = Job.at(
        Path(os.environ["HTTK_WORKFLOW_JOB_DIR"]),
        Path(os.environ["HTTK_WORKFLOW_DATA_DIR"]) if os.environ.get("HTTK_WORKFLOW_DATA_DIR") else None,
        Path(os.environ["HTTK_WORKFLOW_WORKDIR"]) if os.environ.get("HTTK_WORKFLOW_WORKDIR") else None,
    )
    if job.parameters.get("jobflow_child_job") is not None:
        return 0
    if job.workflow != _FULL:
        print(f"latex: only {_FULL} jobs have a LaTeX report", file=sys.stderr)
        return 1
    out = Path(os.environ.get("HTTK_WORKFLOW_POSTPROCESS_DIR") or "")
    if not out.parts or out.resolve() != Path.cwd().resolve():
        print("latex: HTTK_WORKFLOW_POSTPROCESS_DIR must name the current directory", file=sys.stderr)
        return 1
    analysis, figures = out / "analysis", out / "figures"
    problem = _clean(out, analysis, figures)
    if problem:
        print(f"latex: {problem}", file=sys.stderr)
        return 1
    failures: list[str] = []
    for script in ("report", "plots"):
        try:
            _run(script, job, analysis)
        except AnalysisError as exc:
            if not (analysis / "report.json").is_file():
                print(f"latex: {exc}", file=sys.stderr)
                return 1
            # plots re-runs every child's report: list each failed analysis once.
            message = str(exc)
            items = message.removeprefix(_FAILED).split(", ") if message.startswith(_FAILED) else [message]
            failures += [item for item in items if item not in failures]
    report = json.loads((analysis / "report.json").read_text(encoding="utf-8"))
    children = {
        label: json.loads((analysis / label / "report.json").read_text(encoding="utf-8"))
        for label in report.get("children") or {}
        if (analysis / label / "report.json").is_file()
    }
    for label, names in _FIGURES.items():
        for name in names:
            if (analysis / label / f"{name}.png").is_file():
                shutil.copyfile(analysis / label / f"{name}.png", figures / f"{label}_{name}.png")
    copied = {path.name for path in figures.iterdir()}
    runs = _vasp_runs(json.loads(job.result("characterization.json").read_text(encoding="utf-8")))
    (out / "report.tex").write_text(_write(report, children, runs, copied, failures), encoding="utf-8")
    compiled = _compile(out)
    print("wrote report.tex")
    return 0 if compiled and not failures else 1
