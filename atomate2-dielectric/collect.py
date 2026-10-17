"""Collect hook of the ``atomate2.dielectric`` package.

The flow's final output is the task document of atomate2's DielectricMaker
(VASP with LEPSILON and IBRION = 8). Its ``output.structure`` is the dielectric
cell, the cell (and site order) the response refers to. The dielectric, Born effective charge and
piezoelectric data are read from ``calcs_reversed[0].output`` as pymatgen and
emmet return them: dielectric tensors from vasprun.xml, and Born charges and
piezoelectric tensors from OUTCAR, with rows and columns as VASP prints them.
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

#: Voigt columns xx yy zz yz xz xy, taken from VASP's printed order XX YY ZZ XY YZ ZX.
_VASP_TO_VOIGT = [0, 1, 2, 4, 5, 3]


def collect(record):
    """Return the permittivities, Born effective charges, piezoelectric tensor and total energy.

    :param record: The collected job record.
    :return: Extracted output roles.
    :raises ValueError: If the task document has no energy or holds malformed values.
    """
    if record.parameter("jobflow_child_job", None) is not None:
        return {}
    path, task = _flow_output(record)
    energy = (task.get("output") or {}).get("energy")
    if energy is None:
        raise ValueError(f"{path}: the task document has no output.energy")
    structure = (task.get("output") or {}).get("structure")
    outputs = {
        "dielectric_cell": _structure(path, structure),
        "total_energy": DataRecord.from_value(
            _PROPERTIES + "core/total_energy", "_httk_total_energy", _number(path, energy, "output.energy")
        ),
        "task_document": _file(record, path),
    }
    roles = "static_relative_permittivity, high_frequency_relative_permittivity, born_effective_charges and"
    if not _analyse(roles + " piezoelectric_tensor"):
        return outputs

    import numpy as np
    from httk.analyse.matsci.electronic import summarize_dielectric
    from httk.analyse.records import bound_values, records

    calculation = ((task.get("calcs_reversed") or [{}])[0].get("output")) or {}
    electronic, ionic = calculation.get("epsilon_static"), calculation.get("epsilon_ionic")
    tensors = {"high_frequency": electronic}
    if electronic is not None and ionic is not None:
        tensors["static"] = np.add(electronic, ionic)
    for kind, tensor in tensors.items():
        if tensor is None:
            continue
        # The mean (the stored value) is the trace over three; symmetrizing only absorbs print rounding.
        summary = summarize_dielectric((np.asarray(tensor) + np.asarray(tensor).T) / 2)
        for bound, value in zip(bound_values(summary, kind=kind), records(summary, kind=kind), strict=True):
            outputs[bound.binding.definition.rsplit("/", 1)[1]] = value
    dielectric = calculation.get("dielectric_properties") or {}
    if dielectric.get("born_charges"):
        if len(dielectric["born_charges"]) != len(structure["sites"]):
            raise ValueError(
                f"{path}: {len(dielectric['born_charges'])} Born charge tensors for {len(structure['sites'])} sites"
            )
        outputs["born_effective_charges"] = _property("electronic/born_effective_charges", dielectric["born_charges"])
    if dielectric.get("piezo_tensor") is not None and dielectric.get("piezo_ionic_tensor") is not None:
        total = np.add(dielectric["piezo_tensor"], dielectric["piezo_ionic_tensor"])
        outputs["piezoelectric_tensor"] = _property("mechanics/piezoelectric_tensor", total[:, _VASP_TO_VOIGT].tolist())
    return outputs
