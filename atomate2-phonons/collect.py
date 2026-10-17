"""Collect hook of the ``atomate2.phonons`` package.

The flow's final output is atomate2's PhononBSDOSDoc. The phonon band structure,
density of states and thermodynamics refer to the cell of
``phonon_bandstructure.structure``, the cell phonopy diagonalized (the phonon
cell); the document's ``structure`` is the relaxed structure. atomate2 reports
the thermodynamics in J/mol and J/(K mol) per formula unit of the phonon cell;
they are converted to eV and eV/K per phonon cell.
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

#: One J/mol in eV per particle: 1 / (N_A e), exact in the 2019 SI.
_EV_PER_JOULE_PER_MOL = 1 / (6.02214076e23 * 1.602176634e-19)


def collect(record):
    """Return the relaxed structure, the phonon cell and its phonon band structure, DOS and thermodynamics.

    :param record: The collected job record.
    :return: Extracted output roles.
    :raises ValueError: If pymatgen or httk-atomistic is missing, or the document has no phonon band structure.
    """
    if record.parameter("jobflow_child_job", None) is not None:
        return {}
    path, document = _flow_output(record)
    band_structure = document.get("phonon_bandstructure")
    if not isinstance(band_structure, dict) or not isinstance(band_structure.get("structure"), dict):
        raise ValueError(f"{path}: the phonon document has no phonon_bandstructure with a structure")  # noqa: TRY004 (a malformed document is a collect ValueError)
    cell = _structure(path, band_structure["structure"])
    outputs = {"phonon_cell": cell, "task_document": _file(record, path)}
    if isinstance(document.get("structure"), dict):
        outputs["relaxed_structure"] = _structure(path, document["structure"])
    roles = "phonon_band_structure, phonon_density_of_states, vibrational_thermodynamics and zero_point_energy"
    if not _analyse(roles):
        return outputs

    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

    bands = PhononBandStructureSymmLine.from_dict(band_structure)
    if len(bands.bands) != 3 * len(band_structure["structure"]["sites"]):
        raise ValueError(
            f"{path}: {len(bands.bands)} phonon modes for a {len(band_structure['structure']['sites'])}-site cell"
        )
    labelled = [(index, point.label) for index, point in enumerate(bands.qpoints) if point.label]
    outputs["phonon_band_structure"] = _property(
        "thermodynamics/phonon_band_structure",
        {
            "qpoints": [point.frac_coords.tolist() for point in bands.qpoints],
            "qpoint_label_indices": [index for index, _ in labelled],
            "qpoint_labels": [label for _, label in labelled],
            # Sorted at each q-point: phonopy's band connection can reorder the branches.
            "frequencies": np.sort(bands.bands, axis=0).tolist(),
        },
    )
    dos = document.get("phonon_dos")
    if isinstance(dos, dict):
        outputs["phonon_density_of_states"] = _property(
            "thermodynamics/phonon_density_of_states",
            {"frequencies": list(map(float, dos["frequencies"])), "density": list(map(float, dos["densities"]))},
        )
    temperatures = document.get("temperatures")
    if temperatures:
        units = Structure.from_dict(band_structure["structure"]).composition.get_reduced_composition_and_factor()[1]
        scale = units * _EV_PER_JOULE_PER_MOL
        series = {
            name: [_number(path, value, source) * scale for value in document[source]]
            for name, source in (
                ("helmholtz_free_energies", "free_energies"),
                ("internal_energies", "internal_energies"),
                ("entropies", "entropies"),
                ("heat_capacities", "heat_capacities"),
            )
        }
        outputs["vibrational_thermodynamics"] = _property(
            "thermodynamics/vibrational_thermodynamics", {"temperatures": list(map(float, temperatures)), **series}
        )
        if float(temperatures[0]) == 0.0:
            outputs["zero_point_energy"] = _property(
                "thermodynamics/zero_point_energy", series["helmholtz_free_energies"][0]
            )
    return outputs
