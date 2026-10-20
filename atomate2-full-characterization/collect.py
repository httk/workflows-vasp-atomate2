"""Collect hook of the atomate2-full-characterization package.

The parent job yields only the structure its relaxation started from; every child job
is collected under its own package and declaration.
"""

import json


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


def collect(record):
    """Return the standardized structure the run published.

    :param record: The collected job record.
    :return: The ``standardized_structure`` role.
    """
    path = record.result_file("standardized.json")
    return {"standardized_structure": _structure_view(json.loads(path.read_text(encoding="utf-8")))}
