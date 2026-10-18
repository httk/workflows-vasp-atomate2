"""Collect hook of the atomate2-full-characterization package.

The parent job yields only the structure its relaxation started from; every child job
is collected under its own package and declaration.
"""

import json


def collect(record):
    """Return the standardized structure the run published.

    :param record: The collected job record.
    :return: The ``standardized_structure`` role.
    """
    # Imported here, as in the other hooks: the package loads where they are not installed.
    from httk.atomistic import UnitcellStructureView
    from pymatgen.core import Structure

    path = record.result_file("standardized.json")
    structure = Structure.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return {"standardized_structure": UnitcellStructureView(structure)}
