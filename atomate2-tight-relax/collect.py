"""Collect hook of the workflows-vasp-atomate2 packages (every package carries this same file).

The jobflow runner writes the flow's final output, the atomate2 task document (an
emmet ``TaskDoc`` as a plain dict), to ``jobflow-outputs.json`` as
``{"output": <task document>}``. This hook turns that document into the roles
the job's workflow declares: the relaxed structure (relaxation packages only),
the total energy, the band gap, and the document itself as a file. The child
jobs that run the flow's individual atomate2 jobs share the workflow and yield
nothing; their parent collects the flow's result.
"""

import importlib.util
import json
import logging
import math

from httk.core import DataRecord, FileRecord
from httk.core.digests import sha256_file

_LOGGER = logging.getLogger(__name__)
_TOTAL_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/core/total_energy"
_BAND_GAP = "https://schemas.httk.org/defs/v0.1/properties/electronic/band_gap"


def _declared_outputs(record) -> set[str]:
    """Return the output roles of the job's own workflow declaration."""

    entry = record.declarations.get("workflow") or {}
    document = entry.get("observed") or entry.get("declared") or {}
    return {output["name"] for output in document.get("outputs", ())}


def collect(record):
    """Return the declared roles from the atomate2 task document.

    :param record: The collected job record.
    :return: Extracted output roles.
    :raises ValueError: If pymatgen or httk-atomistic is missing, or the document has no structure or finite energy.
    """
    if record.parameter("jobflow_child_job", None) is not None:
        # One atomate2 job of the flow: its parent's task document holds the results.
        return {}
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ValueError("collecting atomate2 task documents needs pymatgen; install it where you collect") from exc
    try:
        from httk.atomistic import UnitcellStructureView
    except ImportError as exc:
        raise ValueError(
            "collecting atomate2 task documents needs httk-atomistic; install it where you collect"
        ) from exc

    path = record.result_file(
        "jobflow-outputs.json", data_prefix=record.parameter("jobflow_data_prefix", "jobflow") or ""
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    task = document.get("output") if isinstance(document, dict) else None
    output = task.get("output") if isinstance(task, dict) else None
    if not isinstance(output, dict) or output.get("structure") is None or output.get("energy") is None:
        raise ValueError(f"{path}: the task document has no output.structure or output.energy")
    try:
        energy = float(output["energy"])
        bandgap = None if output.get("bandgap") is None else float(output["bandgap"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: the task document's output.energy or output.bandgap is not a number: {exc}") from exc
    if not math.isfinite(energy):
        # pymatgen reports a run without energies as an infinite final energy.
        raise ValueError(f"{path}: the task document's output.energy is not finite: {energy}")

    outputs = {
        "total_energy": DataRecord.from_value(_TOTAL_ENERGY, "_httk_total_energy", energy),
        "task_document": FileRecord(
            path.resolve().relative_to(record.workspace_root.resolve()).as_posix(),
            path.name,
            size=path.stat().st_size,
            media_type="application/json",
            sha256=sha256_file(path),
        ),
    }
    if "relaxed_structure" in _declared_outputs(record):
        outputs["relaxed_structure"] = UnitcellStructureView(Structure.from_dict(output["structure"]))
    if bandgap is not None:
        if importlib.util.find_spec("httk.analyse") is None:
            # Without httk-analyse the band gap would be stored untyped and unserved.
            _LOGGER.warning("band_gap not collected: the typed band-gap record needs httk-analyse")
        else:
            outputs["band_gap"] = DataRecord.from_value(_BAND_GAP, "_httk_band_gap", bandgap)
    return outputs
