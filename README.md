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
| `atomate2-full-characterization` | `atomate2.full-characterization` | the packages above, as child jobs | `standardized_structure` (each child collects its own) |

Every package takes one input, `structure` (role `initial_structure`), a
structure file that pymatgen reads (for `atomate2.full-characterization`, see
below). The Makers run with their default
parameters, except that `atomate2.phonons` declares four Maker parameters:
`min_length` (the minimum supercell length in Å), `born_maker` (give `null`
to skip the Born effective charge calculation, which atomate2 runs by default
and which only makes sense for insulators), `use_symmetrized_structure`
(see the note on paths below) and `bulk_relax_maker`, and `atomate2.elastic`
declares `bulk_relax_maker`. Give `bulk_relax_maker` as `null` to skip the
flow's own bulk relaxation (by default atomate2's tight double relaxation),
for a structure that is already relaxed with tight settings. Only parameters a
job gives are passed to the Maker.

`atomate2.phonons` and `atomate2.elastic` also take optional inputs that pass
results of earlier calculations to the Maker's `make()`, as JSON values (for
example `--input total_dft_energy_per_formula_unit=-5.01`); they are not part
of the declarations. `atomate2.phonons`: `born` (one 3×3 Born effective charge
matrix per site, in the structure's site order) and `epsilon_static` (the
ion-clamped dielectric tensor), which together replace the Born charge
calculation, and `total_dft_energy_per_formula_unit` (eV), which replaces the
static energy calculation. `atomate2.elastic`: `equilibrium_stress` (3×3, in
kBar as VASP reports it), which the fit uses in place of the bulk relaxation's
stress.

## Full characterization

`atomate2.full-characterization` characterizes one material as completely as
these packages allow, running them as child jobs of its own job, in parallel
where their dependencies allow, without running any VASP calculation twice:

1. The structure is converted to its standard primitive cell (atomate2's
   `structure_to_primitive`, keeping site properties), on which the k-point
   and q-point paths are meaningful; the parameter `standardize = false` keeps
   the cell as given.
2. `relax`: `atomate2.double-relax` with the INCAR settings of atomate2's
   `TightRelaxMaker` (the `update_user_incar_settings` powerup). This is the
   bulk relaxation atomate2's elastic and phonon flows would each run.
3. On the relaxed structure, at once: `band_structure`
   (`atomate2.band-structure`), `elastic` (`atomate2.elastic` without its bulk
   relaxation, with the relaxation's stress as `equilibrium_stress`), and either
   `dielectric` (`atomate2.dielectric`, for an insulator: band gap above
   atomate2's metal tolerance of 1e-4 eV) or `phonons` (for a metal: without
   bulk relaxation and Born charge calculation).
4. For an insulator, `phonons` starts as soon as `dielectric` has ended, with
   its Born charges and dielectric tensor (atomate2's own extraction from the
   task document), or without the non-analytical correction if it failed.

The parameter `child_parameters` passes job parameters to the children: a
JSON object from child label (`relax`, `band_structure`, `elastic`,
`dielectric`, `phonons`) to that child's job parameters. Each child accepts its
package's own declared parameters and its `jobflow_overrides`, which apply
after the parent's own (so `relax` applies `TightRelaxMaker`'s INCAR settings
first, and anything given on top). With *httk-workflow*'s `maker` override kind,
`jobflow_overrides` reaches every Maker field, including those `make()` reads.
The parameters the parent sets itself (`elastic`: `bulk_relax_maker`;
`phonons`: `bulk_relax_maker`, `born_maker`, `use_symmetrized_structure`) are
refused, also as the first field of a `maker` override's path, as are unknown
labels: the job fails `child_parameters_invalid` before any child starts.
`function`- and `maker_kwargs`-kind overrides are not inspected; keeping them
off those fields is the user's responsibility. Each child's own checks
(declared parameter types, `jobflow_overrides` syntax) run when that child is
created, so for the children after the relaxation the job fails
`child_parameters_invalid` only then, and no child of that step is started. A name the child does not declare is
passed on with a warning in the log, not refused, so a misspelled name has no
effect. For example, atomate2 refuses phonon supercells
above 1200 atoms, which its default 20 Å `min_length` exceeds for a small cell:
diamond needs about 10 Å, a 216-atom supercell.

```console
httk job new --install --workflow atomate2.full-characterization --input-from structure diamond.cif \
    --parameter 'child_parameters={"phonons": {"min_length": 10}}'
```

`phonons` takes the relaxation's energy per formula unit as
`total_dft_energy_per_formula_unit`, so it runs neither a relaxation, a static
energy calculation nor a Born charge calculation. Its `total_dft_energy` is
therefore the tight relaxation's final ionic step energy (ENCUT 700,
EDIFF 1e-7), not that of the static run atomate2 would add by default
(a `StaticMaker` at its default ENCUT 680). atomate2 symmetrizes the Born charges
and dielectric tensor it is given, so the phonon document's values are the
symmetrized ones (for diamond-structure crystals the charges vanish, and no
non-analytical correction applies). The band-structure flow's
static run is the only static calculation; no standalone `atomate2.static`
runs. Every child gets the same relaxed structure in VASP's site order, so the
Born charges refer to the cell phonons runs on.

A magnetic material is the exception. When the relaxed structure carries
nonzero magnetic moments (emmet attaches VASP's site moments), atomate2's phonon
analysis accepts it only on its own symmetrized primitive cell, whose sites may
come in another order. `phonons` then starts with the other children, with
`use_symmetrized_structure = "primitive"`, the relaxation's energy per formula
unit, and no Born charges passed in: for an insulator it computes its own (a
deliberate duplicate of the dielectric run, which still runs), for a metal none.

The job fails `relax_failed` if the relaxation fails. Otherwise it waits for
every child, then succeeds if all succeeded and fails
`characterization_incomplete` naming the failed ones; successful children are
collected either way; a failed parent is not collected, so its
`standardized_structure` is then missing. `characterization.json` (in the workdir
and the job's data) records each child's job id, key and outcome and the
decisions taken (band gap, the metal tolerance and symmetry precision used,
insulator, magnetic, the `child_parameters` given). The VASP command and the
`atomate2.*`/`pymatgen.*` settings given as job parameters reach every child, as
for jobflow child jobs (a child's `child_parameters` may replace them for that
child; the parent's own decisions keep the parent's values), and the parent reads the `atomate2.*`/`pymatgen.*` workspace settings and job
parameters as the children do (for example `atomate2.bandgap_tol` and
`atomate2.symprec` set its metal tolerance and its standardization precision). The package ships no
`maxtime`, which would cap every child's.

```console
httk job new --install --workflow atomate2.full-characterization --input-from structure Si.cif
```

The structure is staged as the job's POSCAR. `--input-from` reads any structure
file *httk₂* reads and writes it as a POSCAR; `--input structure=FILE` copies the
file's bytes unchanged, so it takes only a POSCAR file.

The parent yields `standardized_structure`; each child job is collected under
its own package and declaration, and the parent's run links to the children's
runs by label.

## Install and launch

```console
httk plugin install 'git+https://github.com/httk/workflows-vasp-atomate2'
httk job new --install --workflow atomate2.double-relax --input structure=Si.cif
httk workflow run
httk collect
```

This makes the workflows known machine-wide. To install them into a workspace
instead:

```console
httk workflow install --workspace main 'git+https://github.com/httk/workflows-vasp-atomate2@wf-v0.2.0'
```

This installs every package of the tag into the workspace, with calls between
them bound to the same commit, so a daemon-run workspace can run any of them
(`httk job new --workspace main --workflow atomate2.full-characterization
--input-from structure Si.cif`). A running daemon needs no reload.

## Install with pip

From a clone, `pip install -e .` or `pip install .` installs the workflows'
dependencies and makes every workflow of this repository known to *httk₂* in
that Python environment, at HEAD and at every `wf-v*` tag, each under its
GitHub commit URI. Only committed content is installed; use a package path
for a working copy. An editable install follows `git pull` (restart running
httk processes); a non-editable install at an unchanged version picks up new
tags only with `pip install --force-reinstall --no-deps .`. Shallow clones are
refused (`git fetch --unshallow --tags`). `pip install -e '.[dispatcher]'`
also installs *httk-workflow-vasp*, for the VASP launch wrapper.

## Configuration

The jobflow runner of *httk-workflow* turns workspace settings into the
environment variables atomate2 and pymatgen read. It does this in its own
processes, so configuration files on the compute node are not needed:

- `vasp.command` becomes `ATOMATE2_VASP_CMD` (and `ATOMATE2_VASP_GAMMA_CMD`
  and `ATOMATE2_VASP_NCL_CMD`). Name only the program, for example `vasp_std`.
  With *httk-workflow-vasp* installed, all three variables name its launch
  wrapper (`python -m httk.codes.vasp.launch --standard vasp_std …`), which runs
  once per VASP launch in the job: it picks the standard, non-collinear
  (`vasp.ncl_command`) or Gamma-only (`vasp.gamma_command`) program from the
  INCAR and KPOINTS, writes planned `KPAR`/`NCORE` into the INCAR unless it
  already sets them, and then applies the launch prefix of the attempt (`srun`,
  `mpirun`) itself. Set `vasp.parallelization` to `off` (a workspace setting or
  job parameter) to leave the INCAR's parallelization to VASP. Without
  *httk-workflow-vasp* the commands are mapped directly with the launch prefix
  prepended, and nothing is planned; atomate2 0.1.5 does not read
  `ATOMATE2_VASP_NCL_CMD`.
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

## Postprocessing

Every package declares the same two postprocess scripts, so one command covers
a workspace of atomate2 jobs:

```console
httk workflow postprocess --script report
httk workflow postprocess --script plots JOB
```

`report` writes `report.json` (plain JSON numbers, units in the key names) and
`report.txt`; `plots` writes PNG figures (150 dpi), plus the plotted data as
JSON where it is computed rather than read. Both write into the job's
postprocess directory, never into the job. A figure whose data the documents
lack is skipped with a line saying so. A script exits 1, with a message, when
the flow output is missing or unreadable. The jobflow child jobs, which a
command without job selectors also visits, exit 0 silently without output.

| Package | `report` | `plots` |
| --- | --- | --- |
| `atomate2.static` | energy (total and per atom), band gap, VBM/CBM, Fermi level, total magnetization, space group and lattice | `dos.png` (+ `dos.json`, its energy zero): total and element-projected DOS from the run's `vasprun.xml` |
| `atomate2.relax`, `.tight-relax`, `.double-relax` | formula; space group and lattice (a, b, c, angles, volume) of the input and relaxed structures (and where the input came from) and the relative volume change; energy, maximum force, stress (kbar), band gap; also `relaxed.cif` and `relaxed.POSCAR` | `relaxation.png` (+ `relaxation.json`): energy per atom and maximum force per ionic step |
| `atomate2.band-structure` | band gap, directness, transition and VBM/CBM energies of the uniform mesh; the same of the line path as `line_*`, with the edges' k-point labels and coordinates; Fermi energies, number of bands | `band_structure.png`, `dos.png` (uniform mesh), `band_structure_dos.png`; `band_structure.json` and `dos.json` record their energy zero |
| `atomate2.dielectric` | electronic (ε∞), ionic and static tensors with eigenvalues and averages, refractive index √(mean ε∞), Born effective charges per site with species, trace and their sum, piezoelectric tensor (C/m², Voigt `xx yy zz yz xz xy`, relaxed-ion) | `dielectric.png` (electronic and ionic parts along the principal axes of the static tensor), `born_charges.png` |
| `atomate2.elastic` | C_ij and S_ij, stiffness eigenvalues and the zero-stress Born criterion (C positive definite) next to the largest equilibrium stress, Voigt/Reuss/Hill bulk and shear moduli, Hill Young's modulus and Poisson ratio, universal anisotropy, atomate2's Debye temperature and sound velocities | `elastic_tensor.png`, `youngs_modulus.png` (+ `.json`: directional E in the xy, xz and yz planes), `stress_strain.png` (+ `.json`: the fit's data and the fitted C_ij) |
| `atomate2.phonons` | imaginary modes, the lowest frequency and its q-point, whether atomate2 applied the non-analytical correction (Born charges and ε∞ given, not all charges zero), zero-point energy, F, U, S and C_v at 300 K (only when 300 K is sampled), supercell matrix | `phonon_band_structure.png`, `phonon_dos.png`, `phonon_band_structure_dos.png`, `phonon_thermodynamics.png` |
| `atomate2.full-characterization` | a combined summary (relaxed structure, uniform band gap and directness, ε∞ and ε0, Hill moduli, phonon stability and zero-point energy), the decisions of `characterization.json` and each child's state; each succeeded child's report in `<label>/` | each succeeded child's report and figures in `<label>/` (`relax/`, `band_structure/`, `elastic/`, `dielectric/`, `phonons/`) |

Conventions: the values are the documents' where `collect.py` reads them. The
elastic tensor is `elastic_tensor.raw`, in the Cartesian frame of the relaxed
structure, symmetrized as when collected. The phonon values refer to phonopy's
primitive cell (`phonon_bandstructure.structure`); the zero-point energy is in
eV per that cell, as collected, and the 300 K values and the thermodynamics
figure are in atomate2's units, J/mol and J/(K mol) per formula unit. Space
groups use atomate2's `SYMPREC` (the job's or workspace's `atomate2.symprec`
when set). A double relaxation's task document holds only its second
calculation, so its ionic steps are that calculation's. The static DOS is read
from the `vasprun.xml(.gz)` of the atomate2 job the document's `dir_name` names.
The electronic figures share one energy zero, which their axis labels name: the
valence band maximum of a gapped system (of the uniform mesh, for
`atomate2.band-structure`), else the Fermi level. The elastic stability field
applies the Born criterion at zero stress; it is meaningful only when the
reported residual stress is negligible. For `atomate2.full-characterization`,
a succeeded child whose analysis fails does not stop the others, but the script
then exits 1 naming it (`failed_analyses` in the report); failed children are
listed, not fatal.

The scripts are one shared module, `scripts/atomate2_postprocess.py`, and two
entry points, identical in every package (a test pins them). Edit them in
`atomate2-static/scripts/` and run `make sync-scripts`.

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
`atomate2.static`, `atomate2.double-relax`, `atomate2.band-structure`,
`atomate2.dielectric` and `atomate2.full-characterization` of a metal
(aluminium) and with a failing relaxation end to end; the extended profile runs
every package, and `atomate2.full-characterization` of a polar insulator
(zincblende SiC), of silicon with a failing dielectric run and of a magnetic
metal too. Test-only workspace settings switch the fake's behaviour:
`fake_vasp.fail_on` names an INCAR tag, and the fake fails every run that sets
it; `fake_vasp.ionic_born` gives isotropic, species-dependent Born charges that
obey the acoustic sum rule (and keeps relaxed atoms in place); `fake_vasp.magnetization` reports a moment on every
site.
The fake's synthetic physics (cosine bands with a 1 eV gap, first- and
second-neighbour spring forces, fixed dielectric tensors, and the stress of fixed cubic elastic
constants when the workspace setting `fake_vasp.reference_a` names the
reference lattice parameter) lets the tests check the collected values. The
end-to-end tests also run both postprocess scripts on the jobs they finished.

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
