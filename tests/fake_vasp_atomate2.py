#!/usr/bin/env python3
"""A fake VASP for testing atomate2 Makers on a machine that has no VASP.

**This is not VASP and computes nothing.** It reads POSCAR, INCAR, KPOINTS and
POTCAR in its working directory and writes the files a finished VASP run leaves
behind (OUTCAR, OSZICAR, CONTCAR, CHGCAR, ``vasprun.xml``), with just enough
content for pymatgen's ``Vasprun``/``Outcar``/``Oszicar`` parsers, custodian's
validators and default atomate2 handlers, and emmet's ``TaskDoc.from_directory``.
The numbers are synthetic, deterministic and meaningless as physics, but
structured so that atomate2's physics flows give recognizable results:

- k-points: automatic meshes expand to the full mesh, line-mode KPOINTS to the
  interpolated path, explicit lists pass through.
- Bands: cosine dispersions, with a direct 1 eV gap at Gamma, or one band
  crossing the Fermi level when the structure contains a metal.
- Energy: -5 eV per atom, slightly lower each ionic step.
- Forces: central springs to the first (``SPRING``) and second (``SPRING / 2``)
  neighbour shells.
- Stress: near zero. When ``HTTK_FAKE_VASP_REFERENCE_A`` names a cubic lattice
  parameter (Angstrom), it is the stress of the cubic elastic constants ``C11``,
  ``C12`` and ``C44`` under the cell's strain from that reference lattice.
- With ``LEPSILON``: fixed dielectric tensors, Born effective charges and
  piezoelectric tensors (ionic parts too for IBRION 7/8), every component
  distinct.

A relaxation (NSW > 1) nudges the last atom each ionic step; the last step's
structure is the CONTCAR. Each invocation appends one JSON line to
``fake-vasp-invocations.jsonl`` so tests can assert how it was launched and
which POTCARs (by TITEL) it was given. When ``HTTK_FAKE_VASP_FAIL_ON`` names an
INCAR tag, a run whose INCAR sets that tag to a true or nonzero value exits with
status 1 after logging, writing nothing else: a crashed VASP, for failure tests.
When ``HTTK_FAKE_VASP_IONIC_BORN`` names a charge q, the Born charges are instead
isotropic, q on the sites of the first species and the balancing
-q n_first / n_rest on the others (the acoustic sum rule), as the site symmetry
of rocksalt or zincblende requires, and relaxations leave the atoms in place: the
nudge would leave the polar crystal off its spring equilibrium, with imaginary
modes that phonopy's non-analytical correction cannot take. When
``HTTK_FAKE_VASP_MAGNETIZATION`` names a moment, the OUTCAR reports it on every
site, which emmet attaches to the structure.

Use it as the command atomate2 runs::

    ATOMATE2_VASP_CMD=/path/to/fake_vasp_atomate2.py
    ATOMATE2_VASP_GAMMA_CMD=/path/to/fake_vasp_atomate2.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

EFERMI = 5.0  # eV
SPRING = 1.5  # eV/Angstrom^2
C11, C12, C44 = 108.0, 62.0, 28.0  # GPa


def epsilon_electronic() -> list[list[float]]:
    """The ion-clamped dielectric tensor: diagonal 11.1, 12.2, 13.3; off-diagonal 0.1(i+1) + 0.01(j+1)."""

    return [[10.0 + 1.1 * (i + 1) if i == j else 0.1 * (i + 1) + 0.01 * (j + 1) for j in range(3)] for i in range(3)]


def epsilon_ionic() -> list[list[float]]:
    """The ionic dielectric contribution: the electronic tensor divided by ten."""

    return [[value / 10 for value in row] for row in epsilon_electronic()]


def born_charges(nions: int) -> list[list[list[float]]]:
    """Born charges as printed: ion n (1-based), row a, column b is n + 0.1(a+1) + 0.01(b+1)."""

    return [[[n + 0.1 * (a + 1) + 0.01 * (b + 1) for b in range(3)] for a in range(3)] for n in range(1, nions + 1)]


def ionic_born_charges(structure: Structure, charge: float) -> list[list[list[float]]]:
    """Isotropic Born charges: *charge* on the first species' sites, the balancing charge on the others."""

    first = [site.specie == structure[0].specie for site in structure]
    other = -charge * sum(first) / (len(structure) - sum(first))
    return [(np.eye(3) * (charge if is_first else other)).tolist() for is_first in first]


def piezo(ionic: bool) -> list[list[float]]:
    """A piezoelectric tensor as printed (rows x y z; columns XX YY ZZ XY YZ ZX): 0.1(i+1) + 0.01(j+1), +1 if ionic."""

    return [[(1.0 if ionic else 0.0) + 0.1 * (i + 1) + 0.01 * (j + 1) for j in range(6)] for i in range(3)]


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


def _energy(e: float, scstep: bool = False) -> str:
    # An electronic step carries more terms than the three totals; pymatgen tells LEPSILON's
    # linear-response steps (only the totals) apart from the self-consistent ones by that.
    extra = '<i name="eentropy">0.0</i>' if scstep else ""
    return (
        f'<energy><i name="e_fr_energy">{e}</i><i name="e_wo_entrp">{e}</i><i name="e_0_energy">{e}</i>{extra}</energy>'
    )


def _kpoints(kpoints: Kpoints | None) -> tuple[list[list[float]], list[float], list[str]]:
    """The k-point list VASP would run on, its weights, and the vasprun ``<generation>`` element."""

    style = None if kpoints is None else kpoints.style.name
    if style in ("Gamma", "Monkhorst"):
        assert kpoints is not None
        divisions = [int(n) for n in kpoints.kpts[0]]
        shift = list(kpoints.kpts_shift or (0.0, 0.0, 0.0))
        offset = [s + (0.5 * (1 - n % 2) if style == "Monkhorst" else 0.0) for n, s in zip(divisions, shift)]
        axes = [[(i + o) / n for i in range(n)] for n, o in zip(divisions, offset)]
        mesh = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        mesh = mesh - np.round(mesh)  # into [-0.5, 0.5]
        generation = (
            f'<generation param="{style}"><v type="int" name="divisions">{" ".join(map(str, divisions))}</v>'
            "</generation>"
        )
        return mesh.tolist(), [1.0 / len(mesh)] * len(mesh), [generation]
    if style == "Line_mode":
        assert kpoints is not None
        ends = np.array(kpoints.kpts, dtype=float)
        path = [a + (b - a) * t for a, b in zip(ends[::2], ends[1::2]) for t in np.linspace(0, 1, kpoints.num_kpts)]
        return [list(k) for k in path], [1.0 / len(path)] * len(path), []
    if style == "Reciprocal":
        assert kpoints is not None
        weights = list(kpoints.kpts_weights or [1.0] * len(kpoints.kpts))
        return [list(k) for k in kpoints.kpts], [w / sum(weights) for w in weights], []
    # ponytail: fully automatic (length) and Cartesian styles collapse to Gamma; expand them if a Maker uses them
    return [[0.0, 0.0, 0.0]], [1.0], []


def _bands(kpts: list[list[float]], nocc: int, nbands: int, metallic: bool) -> np.ndarray:
    """Eigenvalues ``[kpoint][band]``: cosine bands, a 1 eV direct gap at Gamma, or a metal."""

    k = np.array(kpts, dtype=float)
    shape = (3 - np.cos(2 * np.pi * k).sum(axis=1)) / 6  # 0 at Gamma, 1 at the zone corner
    bands = []
    for b in range(nbands):
        if metallic and b == nocc - 1:
            bands.append(EFERMI - 0.3 + 0.6 * shape)
        elif b < nocc:
            bands.append(EFERMI - 0.5 - 0.5 * (nocc - 1 - b) - 0.4 * shape)
        else:
            bands.append(EFERMI + 0.5 + 0.5 * (b - nocc) + 0.4 * shape)
    return np.array(bands).T


def _forces(structure: Structure) -> np.ndarray:
    """Central springs to the nearest neighbours and to an outer group of neighbours.

    Each group has one rest length, its median distance. The outer group (``SPRING / 2``)
    holds every neighbour within 1.2 times the shortest distance beyond the first shell:
    the second shell for fcc, the second and third shells for diamond. It makes diamond
    mechanically stable too; first neighbours alone leave it with zero-frequency shear modes.
    """

    reach = 2.0 * (structure.volume / len(structure)) ** (1 / 3)
    centers, points, images, distances = structure.get_neighbor_list(reach)
    first = distances.min()
    shells = [(distances < 1.2 * first, SPRING)]
    beyond = distances[distances >= 1.2 * first]
    if beyond.size:
        shells.append(((distances >= 1.2 * first) & (distances < 1.2 * beyond.min()), SPRING / 2))
    cart = structure.cart_coords
    vectors = structure.lattice.get_cartesian_coords(structure.frac_coords[points] + images) - cart[centers]
    forces = np.zeros((len(structure), 3))
    for shell, spring in shells:
        rest = float(np.median(distances[shell]))
        pulls = (spring * (distances[shell] - rest) / distances[shell])[:, None] * vectors[shell]
        np.add.at(forces, centers[shell], pulls)
    return forces


def _stress(structure: Structure) -> list[list[float]]:
    """The stress in kbar, in VASP's sign (positive under compression)."""

    reference = os.environ.get("HTTK_FAKE_VASP_REFERENCE_A")
    if not reference:
        return [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    half = float(reference) / 2
    lattice = structure.lattice.matrix
    # The undeformed cell: every cell of a cubic lattice (conventional, primitive, supercell) has
    # vectors on the a/2 grid, and the strains atomate2 applies are far smaller than a/2.
    undeformed = half * np.round(lattice / half)
    deformation = np.linalg.solve(undeformed, lattice).T  # lattice rows v' = F v
    strain = 0.5 * (deformation.T @ deformation - np.eye(3))  # Green-Lagrange
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = C12
    stiffness[np.diag_indices(3)] = C11
    stiffness[3:, 3:] = np.diag([C44] * 3)
    voigt = [strain[0, 0], strain[1, 1], strain[2, 2], 2 * strain[1, 2], 2 * strain[0, 2], 2 * strain[0, 1]]
    s = stiffness @ voigt  # second Piola-Kirchhoff stress, GPa, Voigt xx yy zz yz xz xy
    pk2 = np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
    cauchy = deformation @ pk2 @ deformation.T / np.linalg.det(deformation)
    return (-10.0 * cauchy).tolist()  # GPa -> kbar, VASP's sign


def _lepsilon_outcar(charges: list[list[list[float]]], ionic: bool) -> list[str]:
    """The LEPSILON blocks of an OUTCAR, with the headers pymatgen's ``Outcar`` reads."""

    dashes = " " + "-" * 60
    lines = [" MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)", dashes]
    lines += [" " + "  ".join(f"{v:10.6f}" for v in row) for row in epsilon_electronic()]
    lines += [dashes, "", " PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)"]
    lines += ["               XX          YY          ZZ          XY          YZ          ZX", dashes]
    lines += [f"  {axis}" + "".join(f"{v:12.5f}" for v in row) for axis, row in zip("xyz", piezo(False))]
    lines += [dashes, "", " BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)", dashes]
    for n, charge in enumerate(charges, start=1):
        lines.append(f" ion {n:4d}")
        lines += [f"    {a + 1}" + "".join(f"{v:12.5f}" for v in row) for a, row in enumerate(charge)]
    lines += [dashes, ""]
    if ionic:
        lines += [" MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION", dashes]
        lines += [" " + "  ".join(f"{v:10.6f}" for v in row) for row in epsilon_ionic()]
        lines += [dashes, "", " PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)"]
        lines += ["               XX          YY          ZZ          XY          YZ          ZX", dashes]
        lines += [f"  {axis}" + "".join(f"{v:12.5f}" for v in row) for axis, row in zip("xyz", piezo(True))]
        lines += [dashes, ""]
    return lines


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
    if incar.get(os.environ.get("HTTK_FAKE_VASP_FAIL_ON", "")):
        print("fake_vasp_atomate2: failing on request (HTTK_FAKE_VASP_FAIL_ON)", file=sys.stderr)
        return 1

    counts = [structure.composition[el] for el in poscar.site_symbols]
    nelect = sum(p.zval * int(n) for p, n in zip(potcar, counts, strict=True))
    nocc = int(nelect + 1) // 2
    nbands = max(int(incar.get("NBANDS", 0)), nocc + 4)
    nelm = int(incar.get("NELM", 60))
    nsw = int(incar.get("NSW", 0))
    ibrion = int(incar.get("IBRION", -1 if nsw in (-1, 0) else 0))  # VASP's default
    lepsilon = bool(incar.get("LEPSILON", False))
    nions = 1 if nsw <= 1 else min(3, nsw - 1)  # fewer steps than NSW: an ionically converged relaxation
    nscf = max(2, min(3, nelm - 1))  # fewer steps than NELM: electronically converged
    metallic = any(element.is_metal for element in structure.composition.elements)
    ionic_born = os.environ.get("HTTK_FAKE_VASP_IONIC_BORN")

    kpoints = Kpoints.from_file("KPOINTS") if Path("KPOINTS").is_file() else None
    kpts, weights, generation = _kpoints(kpoints)
    eigenvalues = _bands(kpts, nocc, nbands, metallic)
    occupations = (eigenvalues < EFERMI).astype(float)
    eigen_xml = "".join(
        f'<set comment="kpoint {k + 1}">' + "".join(f"<r>{e:.6f} {o:.1f}</r>" for e, o in zip(row, occ)) + "</set>"
        for k, (row, occ) in enumerate(zip(eigenvalues, occupations))
    )
    # The total DOS: a histogram of the eigenvalues, two electrons per band and k-point weight.
    nedos = int(incar.get("NEDOS", 301))
    edges = np.linspace(eigenvalues.min() - 1, eigenvalues.max() + 1, nedos + 1)
    states, _ = np.histogram(eigenvalues, edges, weights=2 * np.repeat(weights, nbands).reshape(eigenvalues.shape))
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    dos_rows = "".join(f"<r>{e:.4f} {d:.6f} {i:.6f}</r>" for e, d, i in zip(centers, states / width, np.cumsum(states)))

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
        "<kpoints>"
        + "".join(generation)
        + _varray("kpointlist", kpts)
        + _varray("weights", [[w] for w in weights])
        + "</kpoints>",
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
        if nsw > 1 and not ionic_born:
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
            + "".join(f"<scstep>{_energy(x, scstep=True)}</scstep>" for x in scsteps)
            + _structure(structure)
            + _varray("forces", _forces(structure))
            + _varray("stress", _stress(structure))
            + _energy(e)
        )
        if step == nions - 1:
            if lepsilon:
                xml.append(_varray("epsilon", epsilon_electronic()))
                xml.append(_varray("epsilon_rpa", [[0.9 * v for v in row] for row in epsilon_electronic()]))
                if ibrion in (7, 8):
                    xml.append(_varray("epsilon_ion", epsilon_ionic()))
            xml.append(
                f'<eigenvalues><array><set><set comment="spin 1">{eigen_xml}</set></set></array></eigenvalues>'
                f'<dos><i name="efermi">{EFERMI}</i><total><array><set><set comment="spin 1">{dos_rows}'
                "</set></set></array></total></dos>"
            )
        xml.append("</calculation>")
    xml.append(_structure(structure, "finalpos") + "</modeling>\n")
    Path("vasprun.xml").write_text("\n".join(xml), encoding="iso-8859-1")
    Path("OSZICAR").write_text("\n".join(oszicar) + "\n", encoding="utf-8")
    Poscar(structure).write_file("CONTCAR")
    if incar.get("LCHARG", True):
        # A one-point charge density: non-self-consistent runs copy CHGCAR from the static run.
        Path("CHGCAR").write_text(Poscar(structure).get_str() + "\n    1    1    1\n 0.10000000000E+01\n")

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
    if lepsilon:
        outcar.append("   LEPSILON =      T")
    for step in range(nions):
        e = -5.0 * len(structure) - 0.01 * step
        outcar += [
            " total drift:                                0.000000      0.000000      0.000000",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            f"  free  energy   TOTEN  = {e:18.8f} eV",
            f"  energy  without entropy= {e:18.8f}  energy(sigma->0) = {e:18.8f}",
        ]
    if lepsilon:
        charges = ionic_born_charges(structure, float(ionic_born)) if ionic_born else born_charges(len(structure))
        outcar += _lepsilon_outcar(charges, ibrion in (7, 8))
    moment = float(os.environ.get("HTTK_FAKE_VASP_MAGNETIZATION") or 0.0)
    if moment:
        dashes = "-" * 50
        outcar += [" magnetization (x)", "", "# of ion       s       p       d       tot", dashes]
        outcar += [f"    {n:d}        0.000   0.000 {moment:7.3f} {moment:7.3f}" for n in range(1, len(structure) + 1)]
        outcar += [
            dashes,
            f"tot          0.000   0.000 {moment * len(structure):7.3f} {moment * len(structure):7.3f}",
            "",
        ]
    outcar += [
        f" E-fermi : {EFERMI:10.4f}     XC(G=0):  -1.0000     alpha+bet : -1.0000",
        f" number of electron {nelect:15.7f} magnetization {moment * len(structure):15.7f}",
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
