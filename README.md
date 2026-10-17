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

Every package takes one input, `structure` (role `initial_structure`), a
structure file that pymatgen reads. The Makers run with their default
parameters.

## Install and launch

```console
httk plugin install 'git+https://github.com/httk/workflows-vasp-atomate2'
httk job new --install --workflow atomate2.double-relax --input structure=Si.cif
httk workflow run
httk collect
```

## Configuration

The jobflow runner of *httk-workflow* turns workspace settings into the
environment variables atomate2 and pymatgen read. It does this in its own
processes, so configuration files on the compute node are not needed:

- `vasp.command` becomes `ATOMATE2_VASP_CMD` (and `ATOMATE2_VASP_GAMMA_CMD`
  and `ATOMATE2_VASP_NCL_CMD`, unless `vasp.gamma_command`/`vasp.ncl_command`
  are set). Name only the program, for example `vasp_std`: the launch prefix of
  the attempt (`srun`, `mpirun`) is prepended for you. `vasp.ncl_command` is
  mapped, but atomate2 0.1.5 does not read `ATOMATE2_VASP_NCL_CMD`.
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

## What is collected

Each package's `collect.py` reads the flow's final task document (an emmet
`TaskDoc`) from `jobflow-outputs.json` and returns:

- `relaxed_structure` (relaxation packages): `output.structure`, as an
  *httk-atomistic* structure.
- `total_energy`: `output.energy`, which is pymatgen's `Vasprun.final_energy`:
  the energy(sigma->0) of the final ionic step, in eV, for the whole cell.
- `band_gap`: `output.bandgap`, in eV. It is collected only where
  *httk-analyse* is installed, since *httk-analyse* registers the typed
  band-gap record. Otherwise, and when the document has no band gap, the role
  is reported as unfulfilled.
- `task_document`: `jobflow-outputs.json` itself, as a file entry.

All four `collect.py` files are the same file. The child jobs that run the
flow's individual atomate2 jobs yield nothing; their parent collects the
result. Each child is still collected as a run, with every output role
reported as unfulfilled.

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
- *httk-analyse* where the jobs are collected, for the band-gap records.

## Testing

```console
make test
HTTK_TEST_PROFILE=extended make test
```

The tests use `tests/fake_vasp_atomate2.py`, which is not VASP and computes
nothing, and a fake POTCAR library in `tests/fake_psp/`. They run atomate2's
Makers, custodian and emmet on these fakes, and run each package end to end
through *httk-workflow*'s task manager. The normal profile runs
`atomate2.static` and `atomate2.double-relax` end to end. The extended
profile runs all four.

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
