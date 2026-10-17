#!/usr/bin/env python3
"""A fake VASP for testing atomate2 Makers on a machine that has no VASP.

**This is not VASP and computes nothing.** It reads POSCAR, INCAR, KPOINTS and
POTCAR in its working directory and writes the files a finished VASP run leaves
behind (OUTCAR, OSZICAR, CONTCAR, ``vasprun.xml``), with just enough content for
pymatgen's ``Vasprun``/``Outcar``/``Oszicar`` parsers, custodian's validators and
default atomate2 handlers, and emmet's ``TaskDoc.from_directory``. The numbers
are synthetic and meaningless as physics: a converged, nonmagnetic insulator
with energy -5 eV per atom (slightly lower each ionic step), near-zero forces
and stress, and a 1 eV gap. Each ionic step nudges the last atom; the last
step's structure is the CONTCAR. Each invocation appends one JSON line to
``fake-vasp-invocations.jsonl`` so tests can assert how it was launched and
which POTCARs (by TITEL) it was given.

Use it as the command atomate2 runs::

    ATOMATE2_VASP_CMD=/path/to/fake_vasp_atomate2.py
    ATOMATE2_VASP_GAMMA_CMD=/path/to/fake_vasp_atomate2.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar


def _i(name: str, value: object) -> str:
    """One typed ``<i>`` parameter, as vasprun.xml writes it."""

    if isinstance(value, bool):
        return f'<i type="logical" name="{name}">{"T" if value else "F"}</i>'
    if isinstance(value, int):
        return f'<i type="int" name="{name}">{value}</i>'
    if isinstance(value, float):
        return f'<i name="{name}">{value}</i>'
    if isinstance(value, list):
        return f'<v name="{name}">{" ".join(str(float(v)) for v in value)}</v>'
    return f'<i type="string" name="{name}">{value}</i>'


def _varray(name: str, rows: object) -> str:
    return f'<varray name="{name}">' + "".join(f"<v>{' '.join(map(str, r))}</v>" for r in rows) + "</varray>"


def _structure(structure: Structure, name: str = "") -> str:
    attr = f' name="{name}"' if name else ""
    return (
        f"<structure{attr}><crystal>{_varray('basis', structure.lattice.matrix)}</crystal>"
        f"{_varray('positions', structure.frac_coords)}</structure>"
    )


def _energy(e: float) -> str:
    return f'<energy><i name="e_fr_energy">{e}</i><i name="e_wo_entrp">{e}</i><i name="e_0_energy">{e}</i></energy>'


def main() -> int:
    """Write one fake, converged VASP result in the working directory."""

    incar = Incar.from_file("INCAR")
    with warnings.catch_warnings():  # the fake POTCARs are, by design, unknown to pymatgen
        warnings.simplefilter("ignore")
        potcar = Potcar.from_file("POTCAR")
    poscar = Poscar.from_file("POSCAR", check_for_potcar=False)
    structure = poscar.structure
    with Path("fake-vasp-invocations.jsonl").open("a", encoding="utf-8") as log:
        env = {k: os.environ.get(k) for k in ("ATOMATE2_VASP_CMD", "PMG_VASP_PSP_DIR")}
        record = {"argv": sys.argv, "env": env, "incar": dict(incar), "potcar_titels": [p.TITEL for p in potcar]}
        log.write(json.dumps(record, default=str) + "\n")

    counts = [structure.composition[el] for el in poscar.site_symbols]
    nelect = sum(p.zval * int(n) for p, n in zip(potcar, counts, strict=True))
    nocc = int(nelect + 1) // 2
    nbands = max(int(incar.get("NBANDS", 0)), nocc + 4)
    nelm = int(incar.get("NELM", 60))
    nsw = int(incar.get("NSW", 0))
    ibrion = int(incar.get("IBRION", -1 if nsw in (-1, 0) else 0))  # VASP's default
    nions = 1 if nsw <= 1 else min(3, nsw - 1)  # fewer steps than NSW: an ionically converged relaxation
    nscf = max(1, min(3, nelm - 1))  # fewer steps than NELM: electronically converged
    efermi = 5.0

    kpoints = Kpoints.from_file("KPOINTS") if Path("KPOINTS").is_file() else None
    if kpoints is not None and kpoints.style == Kpoints.supported_modes.Reciprocal:
        kpts, weights, generation = kpoints.kpts, kpoints.kpts_weights, ""
    else:
        # ponytail: automatic grids collapse to Gamma; expand the mesh if a consumer ever counts k-points
        kpts, weights = [(0.0, 0.0, 0.0)], [1.0]
        generation = (
            f'<generation param="{kpoints.style.name}"><v type="int" name="divisions">'
            f"{' '.join(map(str, kpoints.kpts[0]))}</v></generation>"
            if kpoints is not None and kpoints.style.name in ("Gamma", "Monkhorst")
            else ""
        )
    total = sum(weights)
    weights = [w / total for w in weights]

    # Valence bands end 0.5 eV below the Fermi level, conduction bands start 0.5 eV above it.
    bands = [(efermi - 0.5 - 0.5 * (nocc - 1 - b), 1.0) for b in range(nocc)]
    bands += [(efermi + 0.5 + 0.5 * b, 0.0) for b in range(nbands - nocc)]
    eigen_k = "".join(f"<r>{e} {o}</r>" for e, o in bands)
    eigenvalues = "".join(f'<set comment="kpoint {k + 1}">{eigen_k}</set>' for k in range(len(kpts)))
    grid = [efermi - 5 + 0.5 * i for i in range(21)]
    dos = [0.0 if abs(e - efermi) < 0.5 else 1.0 for e in grid]
    dos_rows = "".join(f"<r>{e} {d} {sum(dos[: i + 1]) * 0.5}</r>" for i, (e, d) in enumerate(zip(grid, dos)))

    params = dict(incar) | {
        "ISPIN": 1,
        "NELECT": float(nelect),
        "NBANDS": nbands,
        "NELM": nelm,
        "NSW": nsw,
        "IBRION": ibrion,
    }
    xml = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n<modeling>',
        (
            '<generator><i name="program" type="string">vasp</i><i name="version" type="string">6.4.3</i>'
            '<i name="subversion" type="string">fake</i><i name="platform" type="string">fake</i>'
            '<i name="date" type="string">2026 01 01</i><i name="time" type="string">00:00:00</i></generator>'
        ),
        "<incar>" + "".join(_i(k, v) for k, v in incar.items()) + "</incar>",
        f"<kpoints>{generation}{_varray('kpointlist', kpts)}{_varray('weights', [[w] for w in weights])}</kpoints>",
        "<parameters>" + "".join(_i(k, v) for k, v in params.items()) + "</parameters>",
        "<atominfo>",
        '<array name="atoms"><set>'
        + "".join(f"<rc><c>{site.specie}</c><c>1</c></rc>" for site in structure)
        + "</set></array>",
        '<array name="atomtypes"><set>'
        + "".join(
            f"<rc><c>{int(n)}</c><c>{el}</c><c>{p.pomass}</c><c>{p.zval}</c><c>{p.header}</c></rc>"
            for el, n, p in zip(poscar.site_symbols, counts, potcar, strict=True)
        )
        + "</set></array></atominfo>",
        _structure(structure, "initialpos"),
    ]
    oszicar = []
    for step in range(nions):
        structure = structure.copy()
        structure.translate_sites([len(structure) - 1], [0.001, 0.001, 0.001])
        e = -5.0 * len(structure) - 0.01 * step
        scsteps = [e + 0.1**s for s in range(1, nscf)] + [e]
        oszicar.append("       N       E                     dE             d eps       ncg     rms")
        oszicar += [
            f"DAV:{s + 1:4d}    {x:.12E}   {x - e:.5E}   -0.10000E-03   128   0.100E-02" for s, x in enumerate(scsteps)
        ]
        oszicar.append(f"{step + 1:4d} F= {e:.8E} E0= {e:.8E}  d E ={-0.01 if step else e:.6E}")
        xml.append(
            "<calculation>"
            + "".join(f"<scstep>{_energy(x)}</scstep>" for x in scsteps)
            + _structure(structure)
            + _varray("forces", [[0.001, 0.0, 0.0]] * len(structure))
            + _varray("stress", [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
            + _energy(e)
        )
        if step == nions - 1:
            xml.append(
                f'<eigenvalues><array><set><set comment="spin 1">{eigenvalues}</set></set></array></eigenvalues>'
                f'<dos><i name="efermi">{efermi}</i><total><array><set><set comment="spin 1">{dos_rows}'
                "</set></set></array></total></dos>"
            )
        xml.append("</calculation>")
    xml.append(_structure(structure, "finalpos") + "</modeling>\n")
    Path("vasprun.xml").write_text("\n".join(xml), encoding="iso-8859-1")
    Path("OSZICAR").write_text("\n".join(oszicar) + "\n", encoding="utf-8")
    Poscar(structure).write_file("CONTCAR")

    outcar = [
        " vasp.6.4.3 fake (not VASP)",
        (
            f"   k-points           NKPTS = {len(kpts):6d}   k-points in BZ     NKDIM = {len(kpts):6d}"
            f"   number of bands    NBANDS= {nbands:6d}"
        ),
        f"   NELM   = {nelm:6d}",
        f"   NSW    = {nsw:6d}",
        f"   IBRION = {ibrion:6d}",
        "   ISPIN  =      1",
        "   total plane-waves  NPLWV =   1000",
    ]
    for step in range(nions):
        e = -5.0 * len(structure) - 0.01 * step
        outcar += [
            " total drift:                                0.000000      0.000000      0.000000",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            f"  free  energy   TOTEN  = {e:18.8f} eV",
            f"  energy  without entropy= {e:18.8f}  energy(sigma->0) = {e:18.8f}",
        ]
    outcar += [
        f" E-fermi : {efermi:10.4f}     XC(G=0):  -1.0000     alpha+bet : -1.0000",
        f" number of electron {nelect:15.7f} magnetization {0.0:15.7f}",
        " General timing and accounting informations for this job:",
        "                  Total CPU time used (sec):        1.000",
        "                            User time (sec):        1.000",
        "                          System time (sec):        0.000",
        "                         Elapsed time (sec):        1.000",
        "                   Maximum memory used (kb):     1000.",
        "                   Average memory used (kb):        N/A",
    ]
    Path("OUTCAR").write_text("\n".join(outcar) + "\n", encoding="utf-8")
    print("fake_vasp_atomate2: finished (a test fixture, not VASP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
