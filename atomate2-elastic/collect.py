"""Collect hook of the ``atomate2.elastic`` package.

The flow's final output is atomate2's ElasticDocument. The stiffness is taken from
``elastic_tensor.raw``, in the Cartesian frame of the document's (relaxed)
structure, never from ``ieee_format``, which is rotated to the IEEE standard frame.
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


def collect(record):
    """Return the relaxed structure, the elastic tensor and, for a stable tensor, the derived moduli.

    :param record: The collected job record.
    :return: Extracted output roles.
    :raises ValueError: If pymatgen or httk-atomistic is missing, or the document has no tensor or structure.
    """
    if record.parameter("jobflow_child_job", None) is not None:
        return {}
    path, document = _flow_output(record)
    raw = (document.get("elastic_tensor") or {}).get("raw")
    if raw is None:
        raise ValueError(f"{path}: the elastic document has no elastic_tensor.raw")
    outputs = {"relaxed_structure": _structure(path, document.get("structure")), "task_document": _file(record, path)}
    if not _analyse("elastic_tensor, compliance_tensor and the elastic moduli"):
        return outputs

    import numpy as np
    from httk.analyse.matsci.elasticity import ElasticTensor
    from httk.analyse.records import bound_values, records

    # atomate2's fit is symmetric only to fitting precision; ElasticTensor requires exact symmetry.
    stiffness = np.asarray(raw, dtype=float)
    asymmetry = float(np.abs(stiffness - stiffness.T).max())
    if asymmetry > max(0.5, 1e-3 * float(np.abs(stiffness).max())):
        _LOGGER.warning(
            "%s: the fitted elastic tensor is asymmetric by %.3g GPa; storing its symmetric part", path, asymmetry
        )
    tensor = ElasticTensor((stiffness + stiffness.T) / 2)
    if tensor.is_stable():
        for bound, value in zip(bound_values(tensor), records(tensor), strict=True):
            outputs[bound.binding.definition.rsplit("/", 1)[1]] = value
        return outputs
    # An unstable stiffness has no Reuss moduli: only the tensors are collected, as for native workflows.
    outputs["elastic_tensor"] = _property("mechanics/elastic_tensor", [list(row) for row in tensor.stiffness])
    try:
        compliance = tensor.compliance
    except ValueError:
        _LOGGER.warning("%s: compliance_tensor not collected: the stiffness is singular", path)
    else:
        outputs["compliance_tensor"] = _property("mechanics/compliance_tensor", [list(row) for row in compliance])
    return outputs
