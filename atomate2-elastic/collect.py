"""Collect hook of the ``atomate2.elastic`` package.

The flow's final output is atomate2's ElasticDocument. The stiffness is taken from
``elastic_tensor.raw``, in the Cartesian frame of the document's (relaxed)
structure, never from ``ieee_format``, which is rotated to the IEEE standard frame.
"""

# --- shared helpers: identical in every physics package's collect.py (a test pins this block) ---
import hashlib
import importlib.util
import json
import logging
import math
import os
import stat
from pathlib import PurePosixPath

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
    """Adopt a pymatgen structure dictionary of the document at *path* as an httk-atomistic structure."""

    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a structure document, found {type(document).__name__}")  # noqa: TRY004 (a malformed document is a collect ValueError)
    return _structure_view(document)


def _declared_roles(record, kind):
    """Return the ``inputs`` or ``outputs`` roles of the job's own workflow declaration."""

    entry = record.declarations.get("workflow") or {}
    document = entry.get("observed") or entry.get("declared") or {}
    return {role["name"] for role in document.get(kind, ())}


def _sanitized(document):
    """Return the plain-data part of a pymatgen structure dictionary, the only part ``Structure.from_dict`` may see.

    ``Structure.from_dict`` passes site properties through monty's decoder, which imports the modules and calls
    the ``from_dict`` methods a document names. A job writes these documents, so only the lattice matrix, each
    site's species (element, oxidation state, spin, occupancy) and fractional coordinates, a numeric ``magmom`` and
    the structure's charge are kept; everything else (``phonopy_masses`` included) is dropped.

    :raises ValueError: If one of the kept members is not plain JSON data of its shape.
    """

    def is_number(value):
        return isinstance(value, int | float) and not isinstance(value, bool)

    def number(value):
        if not is_number(value):
            raise ValueError(f"structure document: expected a number, found {value!r}")
        return value

    def numbers(value, length):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"structure document: expected a list of {length} numbers, found {value!r}")
        return [number(item) for item in value]

    lattice = document.get("lattice") if isinstance(document, dict) else None
    matrix = lattice.get("matrix") if isinstance(lattice, dict) else None
    sites = document.get("sites") if isinstance(document, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 3 or not isinstance(sites, list):
        raise ValueError("structure document: expected a lattice matrix and a list of sites")
    clean = {"lattice": {"matrix": [numbers(row, 3) for row in matrix]}, "sites": []}
    if document.get("charge") is not None:
        clean["charge"] = number(document["charge"])
    for site in sites:
        species = site.get("species") if isinstance(site, dict) else None
        if not isinstance(species, list) or not species or not all(isinstance(item, dict) for item in species):
            raise ValueError(f"structure document: expected a site with a list of species, found {site!r}")
        kept = []
        for item in species:
            if not isinstance(item.get("element"), str):
                raise ValueError(f"structure document: expected an element symbol, found {item.get('element')!r}")  # noqa: TRY004 (a malformed document is a collect ValueError)
            entry = {"element": item["element"], "occu": number(item.get("occu"))}
            for key in ("oxidation_state", "spin"):
                if item.get(key) is not None:
                    entry[key] = number(item[key])
            kept.append(entry)
        clean_site = {"species": kept, "abc": numbers(site.get("abc"), 3)}
        properties = site.get("properties")
        magmom = properties.get("magmom") if isinstance(properties, dict) else None
        if is_number(magmom) or (isinstance(magmom, list) and magmom and all(map(is_number, magmom))):
            clean_site["properties"] = {"magmom": magmom}
        clean["sites"].append(clean_site)
    return clean


def _normalized(document):
    """Return a pymatgen structure dictionary as a structure of its physics alone (see ``_sanitized``).

    VASP reports the moments of a non-magnetic cell as zeros (``-0.0`` included); they would store physically
    identical cells as distinct entries, so all-zero moments are dropped. Nonzero moments are kept.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ValueError("collecting atomate2 documents needs pymatgen; install it where you collect") from exc
    import numpy as np

    structure = Structure.from_dict(_sanitized(document))
    properties = structure.site_properties
    if "magmom" in properties and not np.any(np.asarray(properties["magmom"], dtype=float)):
        structure.remove_site_property("magmom")
    return structure


def _structure_view(document):
    """Adopt a pymatgen structure dictionary, normalized by ``_normalized``, as an httk-atomistic structure."""

    try:
        from httk.atomistic import UnitcellStructureView
    except ImportError as exc:
        raise ValueError("collecting atomate2 documents needs httk-atomistic; install it where you collect") from exc
    return UnitcellStructureView(_normalized(document))


def _port_document(record):
    """Return the pymatgen structure dictionary of the file staged for the job's jobflow ``structure`` port, or ``None``.

    Only a ``path`` port member of a job whose declaration has an ``initial_structure`` input counts. The file is
    read with ``JobRecord.read_file`` (below the job's payload, bounded, never through a symlink or a special
    file) and parsed by its name: a ``.json`` file as a pymatgen structure dictionary, a ``POSCAR``/``CONTCAR``
    or ``.vasp`` file as a POSCAR and a ``.cif`` file as a CIF. Any other name is refused, since pymatgen's
    name-based reader would hand it to monty's decoder.
    """
    inputs = record.parameter("jobflow_inputs", None)
    member = inputs.get("structure") if isinstance(inputs, dict) else None
    if not isinstance(member, dict) or member.get("kind") != "path":
        return None
    if "initial_structure" not in _declared_roles(record, "inputs"):
        return None
    value = member.get("value")
    relative = PurePosixPath(value) if isinstance(value, str) else None
    if relative is None or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{record.payload}: the structure port's path {value!r} is not relative to the job's payload")
    suffix = relative.suffix.lower()
    if suffix == ".json":
        fmt = None
    elif suffix == ".vasp" or relative.name.upper().startswith(("POSCAR", "CONTCAR")):
        fmt = "poscar"
    elif suffix == ".cif":
        fmt = "cif"
    else:
        raise ValueError(
            f"{record.payload}: the structure port's file {value!r} is not a .json, POSCAR/CONTCAR, .vasp or .cif file"
        )
    data = record.read_file(relative.as_posix(), limit=16 * 2**20)
    if fmt is not None:
        try:
            from pymatgen.core import Structure
        except ImportError as exc:
            raise ValueError("collecting atomate2 documents needs pymatgen; install it where you collect") from exc
        return Structure.from_str(data.decode("utf-8"), fmt=fmt).as_dict()
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError(f"{record.payload / relative}: expected a structure document, found {type(document).__name__}")  # noqa: TRY004 (a malformed document is a collect ValueError)
    return document


def _port_structure(record, document=None):
    """Return the structure the job's jobflow ``structure`` port consumed, or ``None`` (see ``_port_document``).

    *document* is the port document when the caller has already read it.
    """
    document = _port_document(record) if document is None else document
    return None if document is None else _structure_view(document)


def _with_inputs(record, outputs, document=None):
    """Return *outputs*, with the port structure as the ``initial_structure`` input when the job has one.

    *document* is the port document when the caller has already read it.
    """
    structure = _port_structure(record, document)
    if structure is None:
        return outputs
    from httk.workflow.hookapi import Collected

    return Collected(outputs, {"initial_structure": structure})


def _child_task_document(record):
    """Return a jobflow child job's VASP task document as its ``task_document`` role, else nothing.

    A child's own result is ``child-outputs.json`` in its workdir, ``run/`` below its payload, read with
    ``JobRecord.read_file``. For a job that ran VASP its ``docs[0]["output"]`` is the emmet task document,
    summarized in the file's description. Other atomate2 jobs (generators, fits), an absent or empty file and,
    with a warning, anything but a regular file (a symlink included) yield nothing.
    """
    relative = "run/child-outputs.json"
    try:
        mode = os.lstat(record.payload / relative).st_mode
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(mode):
        _LOGGER.warning("%s: %s is not a regular file; its task document is not collected", record.job_key, relative)
        return {}
    # ponytail: 256 MiB covers task documents of large relaxations (a 2-atom cell's is 2 MB); raise it if one is refused.
    data = record.read_file(relative, limit=256 * 2**20)
    if not data.strip():
        return {}
    document = json.loads(data)
    docs = document.get("docs") if isinstance(document, dict) else None
    task = docs[0].get("output") if isinstance(docs, list) and docs and isinstance(docs[0], dict) else None
    calculations = task.get("calcs_reversed") if isinstance(task, dict) else None
    if not isinstance(calculations, list) or not calculations:
        return {}
    version = calculations[0].get("vasp_version") if isinstance(calculations[0], dict) else None
    overall = (task.get("run_stats") or {}).get("overall") or {}
    cores, elapsed = overall.get("cores"), overall.get("elapsed_time")
    parts = (
        f"VASP {version}" if version else None,
        task.get("calc_type"),
        None if cores is None else f"{cores} cores",
        None if elapsed is None else f"{elapsed:g} s",
    )
    return {
        "task_document": FileRecord(
            (record.payload_path / relative).as_posix(),
            "child-outputs.json",
            size=len(data),
            media_type="application/json",
            sha256=hashlib.sha256(data).hexdigest(),
            description=", ".join(str(part) for part in parts if part) or None,
        )
    }


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
        return _child_task_document(record)
    path, document = _flow_output(record)
    raw = (document.get("elastic_tensor") or {}).get("raw")
    if raw is None:
        raise ValueError(f"{path}: the elastic document has no elastic_tensor.raw")
    outputs = {"relaxed_structure": _structure(path, document.get("structure")), "task_document": _file(record, path)}
    if not _analyse("elastic_tensor, compliance_tensor and the elastic moduli"):
        return _with_inputs(record, outputs)

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
        return _with_inputs(record, outputs)
    # An unstable stiffness has no Reuss moduli: only the tensors are collected, as for native workflows.
    outputs["elastic_tensor"] = _property("mechanics/elastic_tensor", [list(row) for row in tensor.stiffness])
    try:
        compliance = tensor.compliance
    except ValueError:
        _LOGGER.warning("%s: compliance_tensor not collected: the stiffness is singular", path)
    else:
        outputs["compliance_tensor"] = _property("mechanics/compliance_tensor", [list(row) for row in compliance])
    return _with_inputs(record, outputs)
