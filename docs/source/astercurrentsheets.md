# Design: `AsterCurrentSheets`, a runtime SOM classifier thorn for AsterX

Status: design document, 2026-10-26. No thorn code lives in this repository.
The thorn belongs in the AsterX repository; this document fixes the contract
between aweSOM's model artifact and the thorn so that the two can be built
and validated independently.

Naming and scope follow the user's own note in the AsterX audit
(`asterx/audit/ideas.md`, section "aweSOM measurement framework", stage 2):
training, ensemble construction and model selection stay offline in Python;
the thorn does only feature evaluation and best-matching-unit (BMU)
inference; it writes label and confidence grid functions and must not change
resistivity, evolution variables, refinement or particle injection.

## 1. What aweSOM supplies

`aweSOM.model.SOMModel.save_text(path)` writes a plain-text table that a C++
code can parse with no library. The same model saved with `save(path)` is an
HDF5 file with identical content for Python users. Contents:

| Field | Shape | Meaning |
|---|---|---|
| `xdim`, `ydim` | scalars | lattice extent; node index `i = x + y * xdim` |
| `feature_names` | F strings | the feature vector the thorn must evaluate, in order |
| `scaler_kind`, `scaler_offset`, `scaler_scale` | kind, F, F | scaled feature `x' = (x - offset) / scale`; "manual" is mean and 2 sigma of the training sample |
| weights | nodes x F | node weight vectors in the scaled space |
| `node_cluster` | nodes | semantic cluster id per node, -1 for unassigned |
| `node_dist_q99` | nodes | 99th percentile of the BMU distance of training points mapped to that node |
| `reject_distance` | scalar | 99.9th percentile of the BMU distance over all training points |
| metadata | key/value | `format_version`, `awesom_version`, `git_commit`, `created`, `dataset`, `merge_cost`, `seed`, `train`, `alpha_0` |

The reference classifier is `cpp/awesom_classify.hpp` in this repository:
header-only C++17, loads the text table, scales a raw feature vector, scans
the nodes in ascending index order with a strict `<` comparison and
sequential float64 accumulation over features, and returns the node, its
cluster and the squared distance. `tests/cpp_test.py` proves it reproduces
the Python labels bitwise on a fixture. The thorn should include this header
unchanged (it is `CCTK_HOST CCTK_DEVICE`-ready: no allocation, no exceptions
in the hot path, plain pointers).

## 2. Thorn layout (AsterAnalysis conventions)

Verified conventions, from the AsterX checkout (paths relative to the asterx
repository):

- `configuration.ccl`: `REQUIRES Loop AsterUtils AsterX` and `PROVIDES AsterCurrentSheets` (pattern: `AsterAnalysis/configuration.ccl:3-6`).
- `interface.ccl`: `IMPLEMENTS: AsterCurrentSheets`, `INHERITS: HydroBaseX ADMBaseX AsterX`, `USES INCLUDE HEADER: loop_device.hxx aster_fd.hxx aster_interp.hxx aster_utils.hxx` (pattern: `AsterAnalysis/interface.ccl:3-7`). Grid functions, cell centred, not checkpointed:

```
CCTK_REAL sheet_fields TYPE=gf CENTERING={ccc} TAGS='checkpoint="no"'
{
  sheet_label      # cluster id as a real number, -1 = unassigned / masked
  sheet_node       # BMU node index
  sheet_dist       # Euclidean distance to the BMU in scaled feature space
  sheet_confident  # 1 if sheet_dist <= node_dist_q99[node] and <= reject_distance, else 0
} "SOM current-sheet classification"
```

- `param.ccl`:

```
CCTK_STRING model_file "aweSOM text model table" STEERABLE=RECOVER { ".+" :: "" } "."
CCTK_INT cadence "evaluate every N iterations" STEERABLE=always { 1:* :: "" } 1
CCTK_INT ghost_width "ghost cells needed by the feature stencil" { 0:* :: "" } 1
BOOLEAN mask_atmosphere "set label -1 where rho <= rho_atmo" STEERABLE=always { } "yes"
BOOLEAN verbose "" STEERABLE=always { } "no"
```

- `schedule.ccl`: load the table once, globally, before the grid exists
  (pattern: `EOSX/schedule.ccl:9-13`), and classify in the analysis bin after
  the fields are synchronized (pattern: `AsterAnalysis/schedule.ccl`):

```
SCHEDULE AsterCurrentSheets_LoadModel AT CCTK_WRAGH
{ LANG: C  OPTIONS: GLOBAL } "read the aweSOM table on rank 0, broadcast, upload to managed memory"

SCHEDULE AsterCurrentSheets_Classify AT analysis AFTER AsterAnalysis_MHD
{
  LANG: C
  READS: HydroBaseX::rho(everywhere) HydroBaseX::press(everywhere)
         HydroBaseX::Bvec(everywhere) HydroBaseX::vel(everywhere)
         ADMBaseX::metric(everywhere)
  WRITES: sheet_fields(interior)
  SYNC: sheet_fields
} "classify every interior cell with the SOM"
```

  `cadence` is enforced inside the function with `cctk_iteration % cadence`.
  Output is selected in the parameter file with
  `CarpetX::out_silo_vars = "AsterCurrentSheets::sheet_fields"` (or
  `out_openpmd_vars`), the same mechanism AsterAnalysis uses.

## 3. Table upload for device kernels

Follow EOSX exactly (`EOSX/src/setup_eos.cxx:188-196`,
`EOSX/src/eos_3p_tabulated3d/eos_readtable_compose.hxx:131-138, 414-432`):

1. Rank 0 parses the text table on the host (the header's `load_text`).
2. `MPI_Bcast` the sizes, then the flat arrays.
3. Allocate `weights`, `offset`, `scale`, `node_dist_q99`, `node_cluster` with
   `amrex::The_Managed_Arena()->alloc(...)` and copy with
   `amrex::Gpu::copy(amrex::Gpu::hostToDevice, ...)`.
4. Store the pointers and the scalars in a plain POD struct placed in managed
   memory (`new (ptr) SOMTableView{...}`), kept in a global pointer.
5. Kernels capture the pointer by value (`auto table = g_som_table;`), as
   `AsterX/src/fluxes.cxx:1290` does for the EOS.

The header's `best_match` operates on raw pointers, so the device version is
the same function marked `CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE`.
Keep `-ffp-contract=off` for the translation unit that contains the
classifier so FMA contraction cannot change the arithmetic that the Python
fixture pins.

## 4. Feature evaluation per cell

Loop idiom: `grid.loop_int_device<1,1,1>(grid.nghostzones, [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE { ... })`
(`AsterAnalysis/src/analysis_mhd.cxx:33-35`). Neighbouring cells are
`p.I + p.DI[d]` with spacing `p.DX[d]`.

AsterX does not provide a current density grid function; the only curl in
the code is `B = curl A` onto staggered faces (`AsterX/src/computeBfromA.cxx`).
The thorn therefore computes a curl-B proxy from the cell-centred `Bvec`
with second-order central differences, which costs one ghost layer
(`ghost_width = 1`):

```
Jx = (Bz(j+1) - Bz(j-1)) / (2 dy) - (By(k+1) - By(k-1)) / (2 dz)   (and cyclic)
```

This is a coordinate-basis, flat-metric proxy; the audit note requires the
diagnostic contract to state the frame and the distinction between this
proxy and a physical Ohm-law current. The feature names in the table must
match what the thorn evaluates. A first feature set that needs only
`HydroBaseX` variables and the proxy:

| name | definition |
|---|---|
| `jmag` | sqrt(Jx^2 + Jy^2 + Jz^2) from the proxy |
| `jpar` | (J . B) / |B| |
| `jperp` | sqrt(jmag^2 - jpar^2) |
| `bmag` | |B| |
| `rho` | `HydroBaseX::rho` |
| `beta_inv` | b^2 / (2 p) using `press` |

Smoothing kernels used in the published PIC pipeline (`examples/plasma-turbulence/obtain_features.py`) are not reproduced in the first runtime version; the offline Stage 1 adapter decides which features survive. A model trained on different features will be rejected by the thorn: it must compare `feature_names` from the table with its compiled list and abort at `CCTK_WRAGH` on mismatch.

Masking: where `rho <= rho_atmo` (parameter from AsterX), or where a C2P
failure flag is set, write `sheet_label = -1` and `sheet_confident = 0`.

## 5. Validation before use

1. Offline/runtime equivalence: dump `sheet_label` and the feature vector for
   one snapshot; feed the dumped features to `SOMModel.predict(...,
   backend="numba")`; labels must be identical (this is the same check
   `tests/cpp_test.py` performs). The default KD-tree backend can resolve exact
   ties differently and rejects non-finite features, so it is not the reference.
2. Overhead: time the analysis step with and without the thorn at the chosen
   cadence (the per-cell cost is nodes x F multiply-adds, about 2400 x 6 for
   the fiducial lattice; on a GPU this is negligible next to the RHS).
3. AMR: labels on covered coarse cells are meaningless; de-duplication and
   tracking across time are later tasks per the audit note.

## 6. Out of scope for this repository

Writing the thorn, the CarpetX build, the Stage 1 GR-aware feature adapter,
connected components and temporal tracking.
