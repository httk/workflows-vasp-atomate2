"""Collect hook of the ``atomate2.band-structure`` package.

The flow's final output holds the task documents of atomate2's non-self-consistent
``uniform`` and ``line`` calculations and their band structures (``uniform_bs``,
``line_bs``, pymatgen dictionaries). The line calculation's structure is the cell
the k-points refer to; the band structure comes from ``line_bs``, the density of
states, band gap and Fermi energy from the uniform calculation. The flow output
holds no static task document, so no total energy is collected (the
non-self-consistent energies are not total energies).
"""

# --- shared helpers: identical in every physics package's collect.py (a test pins this block) ---
import importlib.util
import json
import logging
import math

from httk.core import DataRecord, FileRecord
from httk.core.digests import sha256_file

_LOGGER = logging.getLogger(__name__)
_PROPERTIES = "https://schemas.httk.org/defs/v0.1/properties/"


def _flow_output(record):
    """Return the path of ``jobflow-outputs.json`` and the flow's final output in it."""

    path = record.result_file(
        "jobflow-outputs.json", data_prefix=record.parameter("jobflow_data_prefix", "jobflow") or ""
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    output = document.get("output") if isinstance(document, dict) else None
    if not isinstance(output, dict):
        raise ValueError(f"{path}: the flow's output is not a JSON object")  # noqa: TRY004 (a malformed document is a collect ValueError)
    return path, output


def _file(record, path):
    """Describe the output document as a file, by its path relative to the workspace root."""

    return FileRecord(
        path.resolve().relative_to(record.workspace_root.resolve()).as_posix(),
        path.name,
        size=path.stat().st_size,
        media_type="application/json",
        sha256=sha256_file(path),
    )


def _structure(path, document):
    """Adopt a pymatgen structure dictionary as an httk-atomistic structure."""

    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ValueError("collecting atomate2 documents needs pymatgen; install it where you collect") from exc
    try:
        from httk.atomistic import UnitcellStructureView
    except ImportError as exc:
        raise ValueError("collecting atomate2 documents needs httk-atomistic; install it where you collect") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a structure document, found {type(document).__name__}")  # noqa: TRY004 (a malformed document is a collect ValueError)
    return UnitcellStructureView(Structure.from_dict(document))


def _analyse(roles):
    """Whether httk-analyse, which registers the typed records of *roles*, is installed; warn if not."""

    if importlib.util.find_spec("httk.analyse") is not None:
        return True
    # Without httk-analyse the values would be stored untyped and unserved.
    _LOGGER.warning("%s not collected: the typed records need httk-analyse", roles)
    return False


def _property(name, value):
    """A property record of one httk definition, ``<group>/<name>``."""

    return DataRecord.from_value(_PROPERTIES + name, "_httk_" + name.rsplit("/", 1)[1], value)


def _number(path, value, what):
    """Return a finite float, or raise ``ValueError`` naming the document and the value."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {what} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: {what} is not finite: {number}")
    return number


# --- end of shared helpers ---


def _path_labels(points):
    """The indices and labels of the labelled points of a pymatgen path."""

    labelled = [(index, point.label) for index, point in enumerate(points) if point.label]
    return [index for index, _ in labelled], [label for _, label in labelled]


def collect(record):
    """Return the band-structure cell, band structure, density of states, band gap and Fermi energy.

    :param record: The collected job record.
    :return: Extracted output roles.
    :raises ValueError: If pymatgen or httk-atomistic is missing, or the flow output lacks a document.
    """
    if record.parameter("jobflow_child_job", None) is not None:
        # One atomate2 job of the flow: its parent's output document holds the results.
        return {}
    path, output = _flow_output(record)
    line, line_bs, uniform = output.get("line"), output.get("line_bs"), output.get("uniform")
    if not (isinstance(line, dict) and isinstance(line_bs, dict) and isinstance(uniform, dict)):
        raise ValueError(  # noqa: TRY004 (a malformed document is a collect ValueError)
            f"{path}: the flow output lacks the line, line_bs or uniform document; "
            'this package needs BandStructureMaker(bandstructure_type="both")'
        )
    outputs = {
        "band_structure_cell": _structure(path, (line.get("output") or {}).get("structure")),
        "task_document": _file(record, path),
    }
    if not _analyse("electronic_band_structure, electronic_density_of_states, band_gap and fermi_energy"):
        return outputs

    import numpy as np
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.electronic_structure.core import Spin

    bands = BandStructureSymmLine.from_dict(line_bs)
    indices, labels = _path_labels(bands.kpoints)
    structure = {
        "kpoints": [point.frac_coords.tolist() for point in bands.kpoints],
        "kpoint_label_indices": indices,
        "kpoint_labels": labels,
        # The definition orders bands by energy at each k-point, not by band connectivity.
        "energies": np.sort(bands.bands[Spin.up], axis=0).tolist(),
        "fermi_energy": _number(path, bands.efermi, "line_bs.efermi"),
    }
    if Spin.down in bands.bands:
        structure["energies_spin_down"] = np.sort(bands.bands[Spin.down], axis=0).tolist()
    outputs["electronic_band_structure"] = _property("electronic/electronic_band_structure", structure)

    dos = (uniform.get("vasp_objects") or {}).get("dos")
    if isinstance(dos, dict):
        energies = np.asarray(dos["energies"], dtype=float)
        density = np.sum([np.asarray(values, dtype=float) for values in dos["densities"].values()], axis=0)
        steps = np.diff(energies) * (density[1:] + density[:-1]) / 2
        outputs["electronic_density_of_states"] = _property(
            "electronic/electronic_density_of_states",
            {
                "energies": energies.tolist(),
                "density": density.tolist(),
                "integrated_density": np.concatenate([[0.0], np.cumsum(steps)]).tolist(),
            },
        )
    gap = (uniform.get("output") or {}).get("bandgap")
    if gap is not None:
        outputs["band_gap"] = _property("electronic/band_gap", _number(path, gap, "uniform.output.bandgap"))
    calculations = uniform.get("calcs_reversed") or [{}]
    fermi = (calculations[0].get("output") or {}).get("efermi")
    if fermi is not None:
        outputs["fermi_energy"] = _property(
            "electronic/fermi_energy", _number(path, fermi, "uniform.calcs_reversed[0].output.efermi")
        )
    return outputs
