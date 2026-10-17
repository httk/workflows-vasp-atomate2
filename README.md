# workflows-vasp-atomate2

This repository runs the [atomate2](https://github.com/materialsproject/atomate2)
VASP Makers as *httk₂* workflows, through *httk-workflow*'s `jobflow`
realization. Each package directory is a manifest that names one Maker; it
carries no runner code. The calculation protocol, the VASP input sets, the
custodian error handling and the task documents are atomate2's, unchanged. For
*httk₂*'s own VASP protocols, use
[workflows-vasp](https://github.com/httk/workflows-vasp).

| Directory | Short name | Maker | Outputs |
| --- | --- | --- | --- |
| `atomate2-static` | `atomate2.static` | `atomate2.vasp.jobs.core:StaticMaker` | `total_energy`, `band_gap`, `task_document` |
| `atomate2-relax` | `atomate2.relax` | `atomate2.vasp.jobs.core:RelaxMaker` | `relaxed_structure`, `total_energy`, `band_gap`, `task_document` |
| `atomate2-tight-relax` | `atomate2.tight-relax` | `atomate2.vasp.jobs.core:TightRelaxMaker` | `relaxed_structure`, `total_energy`, `band_gap`, `task_document` |
| `atomate2-double-relax` | `atomate2.double-relax` | `atomate2.vasp.flows.core:DoubleRelaxMaker` | `relaxed_structure`, `total_energy`, `band_gap`, `task_document` |
| `atomate2-band-structure` | `atomate2.band-structure` | `atomate2.vasp.flows.core:BandStructureMaker` | `band_structure_cell`, `electronic_band_structure`, `electronic_density_of_states`, `band_gap`, `fermi_energy`, `task_document` |
| `atomate2-dielectric` | `atomate2.dielectric` | `atomate2.vasp.jobs.core:DielectricMaker` | `dielectric_cell`, `static_relative_permittivity`, `high_frequency_relative_permittivity`, `born_effective_charges`, `piezoelectric_tensor`, `total_energy`, `task_document` |
| `atomate2-elastic` | `atomate2.elastic` | `atomate2.vasp.flows.elastic:ElasticMaker` | `relaxed_structure`, `elastic_tensor`, `compliance_tensor`, `bulk_modulus_{voigt,reuss,hill}`, `shear_modulus_{voigt,reuss,hill}`, `universal_anisotropy_index`, `task_document` |
| `atomate2-phonons` | `atomate2.phonons` | `atomate2.vasp.flows.phonons:PhononMaker` | `relaxed_structure`, `phonon_cell`, `phonon_band_structure`, `phonon_density_of_states`, `vibrational_thermodynamics`, `zero_point_energy`, `task_document` |

Every package takes one input, `structure` (role `initial_structure`), a
structure file that pymatgen reads. The Makers run with their default
parameters, except that `atomate2.phonons` declares three Maker parameters:
`min_length` (the minimum supercell length in Å), `born_maker` (give `null`
to skip the Born effective charge calculation, which atomate2 runs by default
and which only makes sense for insulators) and `use_symmetrized_structure`
(see the note on paths below). Only parameters a job gives are
passed to the Maker.

## Install and launch

```console
httk plugin install 'git+https://github.com/httk/workflows-vasp-atomate2'
httk job new --install --workflow atomate2.double-relax --input structure=Si.cif
httk workflow run
httk collect
```

## Install with pip

From a clone, `pip install -e .` or `pip install .` installs the workflows'
dependencies and makes every workflow of this repository known to *httk₂* in
that Python environment, at HEAD and at every `wf-v*` tag, each under its
GitHub commit URI. Only committed content is installed; use a package path
for a working copy. An editable install follows `git pull` (restart running
httk processes); a non-editable install at an unchanged version picks up new
tags only with `pip install --force-reinstall --no-deps .`. Shallow clones are
refused (`git fetch --unshallow --tags`). `pip install -e '.[dispatcher]'`
also installs *httk-workflow-vasp*, for the VASP variant dispatcher.

## Configuration

The jobflow runner of *httk-workflow* turns workspace settings into the
environment variables atomate2 and pymatgen read. It does this in its own
processes, so configuration files on the compute node are not needed:

- `vasp.command` becomes `ATOMATE2_VASP_CMD` (and `ATOMATE2_VASP_GAMMA_CMD`
  and `ATOMATE2_VASP_NCL_CMD`, unless `vasp.gamma_command`/`vasp.ncl_command`
  are set). Name only the program, for example `vasp_std`: the launch prefix of
  the attempt (`srun`, `mpirun`) is prepended for you. atomate2 0.1.5 does not
  read `ATOMATE2_VASP_NCL_CMD`; when `vasp.ncl_command` or `vasp.gamma_command`
  is set and *httk-workflow-vasp* is installed, all three variables instead name
  its dispatcher, which picks the program from each job's INCAR and KPOINTS.
- `pymatgen.psp_dir` becomes `PMG_VASP_PSP_DIR`. The directory uses pymatgen's
  layout. atomate2's default input sets use the `PBE_54` functional, so the
  POTCARs are found at `<psp_dir>/POT_GGA_PAW_PBE_54/POTCAR.<El>`. Under the
  workspace daemon's confinement, the directory must be inside
  `confine.readonly_paths`.
- Any other `atomate2.<key>` or `pymatgen.<key>` setting becomes
  `ATOMATE2_<KEY>` or `PMG_<KEY>`. For example, `atomate2.config_file` and
  `pymatgen.config_file` choose the libraries' configuration files.

The same keys given as job parameters (for example `vasp.command`) override
the workspace settings for that job, and reach every child job of its flow.

```console
httk workspace settings set --key vasp.command --value vasp_std default
httk workspace settings set --key pymatgen.psp_dir --value /path/to/potcars default
```

This mapping needs the *httk-workflow* release that contains it (after 2.1.1).

## Resources

The packages ship no resource table, since sites differ. A copied package can
give each atomate2 job its own resources with a table under the runner
options, matched against job names in order; a trailing `"*"` row makes every
job's resources explicit:

```toml
[workflow.runner.resources."*relax*"]
procs = 32
maxtime = "4:00:00"

[workflow.runner.resources."phonon static*"]
procs = 16
maxtime = "1:00:00"

[workflow.runner.resources."*"]
procs = 8
maxtime = "1:00:00"
```

Without editing a package, a job can change resources and Maker settings
through its `jobflow_overrides` parameter (for example `update_config` with
`manager_config["httk"]["resources"]`, or an atomate2 powerup such as
`atomate2.vasp.powerups:update_user_incar_settings`). See *httk-workflow*'s
jobflow documentation for the operations.

## What is collected

Each package's `collect.py` reads the flow's final output from
`jobflow-outputs.json`. Every package collects `task_document`, the
`jobflow-outputs.json` file itself.

### Structure and energy packages

`atomate2.static`, `atomate2.relax`, `atomate2.tight-relax` and
`atomate2.double-relax` read the final task document (an emmet `TaskDoc`) and
return:

- `relaxed_structure` (relaxation packages): `output.structure`, as an
  *httk-atomistic* structure.
- `total_energy`: `output.energy`, which is pymatgen's `Vasprun.final_energy`:
  the energy(sigma->0) of the final ionic step, in eV, for the whole cell.
- `band_gap`: `output.bandgap`, in eV. It is collected only where
  *httk-analyse* is installed, since *httk-analyse* registers the typed
  band-gap record. Otherwise, and when the document has no band gap, the role
  is reported as unfulfilled.

All four `collect.py` files are the same file.

### Physics packages

The property records below are collected only where *httk-analyse* is
installed, since it registers their typed records; otherwise the roles are
reported as unfulfilled, with a warning.

- `atomate2.band-structure`: the flow output holds the non-self-consistent
  `uniform` and `line` task documents and their band structures.
  `band_structure_cell` is the line calculation's structure, the cell the
  k-points refer to. `electronic_band_structure` is the line band structure:
  k-points in that cell's reciprocal basis, labels as pymatgen reads them from
  KPOINTS (a segment junction is the same point twice with the same label),
  band energies sorted at each k-point, and pymatgen's Fermi energy.
  `electronic_density_of_states` is the uniform calculation's DOS summed over
  spin, with its cumulative trapezoid integral; `band_gap` and `fermi_energy`
  come from the uniform task document. The flow output holds no static task
  document, so no total energy is collected (non-self-consistent energies are
  not total energies). The package needs atomate2's default
  `bandstructure_type="both"`; an override to `"line"` or `"uniform"` makes
  collecting fail with a message saying so. Two Fermi energies are stored: the
  band structure's `fermi_energy` member is the line run's value as pymatgen
  computes it ("smart": mid-gap when VASP's value falls in a band), while the
  `fermi_energy` role is the uniform run's, and is the one to use.
- `atomate2.dielectric` (insulators only): `dielectric_cell` is the task
  document's `output.structure`, the cell the response refers to, in that
  document's site order; the response records are products of it. Values come
  from `calcs_reversed[0].output` of the task document.
  `high_frequency_relative_permittivity` is the isotropic mean of
  `epsilon_static` (ion-clamped, with local-field effects) and
  `static_relative_permittivity` that of `epsilon_static + epsilon_ionic`.
  `born_effective_charges` are VASP's printed matrices per site of the
  dielectric cell (collecting fails if their count differs from its site
  count), rows the field (polarization) direction and columns the displacement
  direction.
  `piezoelectric_tensor` is the relaxed-ion total, `piezo_tensor +
  piezo_ionic_tensor` in C/m², reordered from VASP's printed column order
  `XX YY ZZ XY YZ ZX` to Voigt order `xx yy zz yz xz xy`. Signs and index
  conventions are VASP's as pymatgen returns them; a comparison against a real
  VASP run remains a user acceptance step.
- `atomate2.elastic`: `relaxed_structure` is the ElasticDocument's structure.
  The stiffness is `elastic_tensor.raw`, **in the Cartesian frame of that
  structure** (never `ieee_format`, which atomate2 rotates to the IEEE standard
  frame), symmetrized to absorb fitting noise; an asymmetry above
  max(0.5 GPa, 0.1 % of the largest component) is logged as a warning. A stable tensor yields the
  elastic and compliance tensors, the Voigt, Reuss and Hill bulk and shear
  moduli and the universal anisotropy index; an unstable one only the two
  tensors.
- `atomate2.phonons`: `phonon_cell` is the cell phonopy diagonalized
  (`phonon_bandstructure.structure`); `phonon_band_structure` (frequencies in
  THz, sorted at each q-point, imaginary modes negative),
  `phonon_density_of_states`, `vibrational_thermodynamics` and
  `zero_point_energy` refer to it. atomate2 reports the thermodynamics in J/mol
  and J/(K mol) per formula unit; they are converted to eV and eV/K per phonon
  cell (× formula units / (N_A e)). The zero-point energy is the free energy at
  0 K. `relaxed_structure` is the document's relaxed structure, and
  `phonon_cell` is a product of it. Collecting fails unless the band structure
  has three modes per site of the phonon cell. No Born charges
  or dielectric constants are collected from the phonon document; use
  `atomate2.dielectric` for those.

Band-structure k-point paths and phonon q-point paths and their labels are
those pymatgen and seekpath choose for the standardized primitive cell. They are
meaningful as high-symmetry paths only when the cell the calculation runs on is
that cell. For phonons, the package's Maker parameter
`use_symmetrized_structure = "primitive"` makes it so (`httk job new …
--parameter use_symmetrized_structure=primitive`). A `jobflow_overrides`
`maker_kwargs` operation cannot do this: it changes the Makers of the flow
`make()` returned, and PhononMaker reads this field inside `make()`.

In every package `task_document` is a product of `initial_structure`.

The physics hooks share their helper code (a test pins it).

### Child jobs and provenance

The child jobs that run the flow's individual atomate2 jobs yield nothing;
their parent collects the result. Each child is still collected as a run, with
every output role reported as unfulfilled.

A run's input edges come only from a provenance document the caller supplies,
as for any *httk₂* job. Without one, the `product_of` links to
`initial_structure` are reported as unlinked.

## Requirements

- The next *httk-workflow* release after 2.1.1. The packages do not run at all
  on *httk-workflow* 2.1.1 or earlier, which lack the jobflow task-document
  serialization, the references to moved job directories and the settings
  mapping. The `requires` floor still names 2.1.1, because floors name released
  versions; it will be raised at that release.
- `atomate2` (with pymatgen, custodian and emmet) where the jobs run.
- *httk-atomistic* and `pymatgen` where the jobs are collected.
- *httk-analyse* where the jobs are collected, for the band-gap records and
  every property record of the physics packages.
- `phonopy` and `seekpath` where `atomate2.phonons` runs (its `requires` names
  them).

## Testing

```console
make test
HTTK_TEST_PROFILE=extended make test
```

The tests use `tests/fake_vasp_atomate2.py`, which is not VASP and computes
nothing, and a fake POTCAR library in `tests/fake_psp/`. They run atomate2's
Makers, custodian and emmet on these fakes, and run each package end to end
through *httk-workflow*'s task manager. The normal profile runs
`atomate2.static`, `atomate2.double-relax`, `atomate2.band-structure` and
`atomate2.dielectric` end to end; the extended profile runs every package.
The fake's synthetic physics (cosine bands with a 1 eV gap, first- and
second-neighbour spring forces, fixed dielectric tensors, and the stress of fixed cubic elastic
constants when the workspace setting `fake_vasp.reference_a` names the
reference lattice parameter) lets the tests check the collected values.

To also run `atomate2.static` on real VASP, set both
`HTTK_TEST_VASP_COMMAND` (the VASP program) and `HTTK_TEST_PMG_VASP_PSP_DIR`
(a POTCAR library in pymatgen's layout). Otherwise that test is skipped.

## Declaration identity

Each package's `declaration.json` is identified by its `$id`, the GitHub raw
URL of that file at a `wf-decl-vX.Y.Z` tag, and `httk_workflow.toml` names the
same URL as `declaration_uri`. The version is the declaration's own
`x-httk-definition.version`. Declaration versions across the repository form
one increasing sequence shared with the tags. A `wf-decl-*` tag is created only
when a declaration's content changes and is never moved. To change a
declaration, edit the document and give it a version higher than every existing
`wf-decl-*` tag (normally the next minor), set `$id` and `declaration_uri` to
that tag's URL, commit, tag `wf-decl-v<version>`, and push the tag with
`git push origin wf-decl-v<version>` (a plain `git push` does not push tags).
Declarations that did not change keep their version and `$id`, so versions may
skip numbers. The URL names the declaration document, not the workflow code.
