# Reading AsterX / CarpetX output

`aweSOM.io` turns simulation output into *level blocks*: an x slab of one box
or patch of one refinement level, with a halo for the derivative stencils:

```python
from aweSOM.io import iter_level_blocks
from aweSOM.features_grmhd import INPUTS

for level, origin, spacing, arrays in iter_level_blocks("run/data/silo", INPUTS, 76800):
    ...
```

The source can be:

- **Carpet** (`CarpetHDF5Reader`): an etkplot index (`.json`) or a directory of
  `<var>.xyz.file_N.h5` files. The blocks match etkplot's `som-features` grid
  files row for row.
- **CarpetX** (`CarpetXReader`): a directory of Silo output, or of
  openPMD-HDF5 output.

`read_tsv` reads CarpetX TSV line output for 1D checks.

`examples/grmhd-bbh/analyze_chunks.py` labels whole snapshots block by block
with a frozen model. For the details of each layout, see the docstrings of
`aweSOM.io.carpet` and `aweSOM.io.carpetx`.

## What has been checked

- **Carpet (BBH run, it 180224, levels 5–7):**
  - the blocks, features and labels equal those from etkplot's grid file
    exactly, over 68,028,213 cells;
  - the native reader and the etkplot backend give identical blocks.
- **CarpetX Silo:** the layout was read from a real 2D run
  (`FM_4lvl_Noble`, xy plane, 4 levels). The levels' boxes match
  `CarpetX/FabArrayBase`. The plane values equal the TSV lines of the same
  iteration (a test checks this when the run is on disk). 3D Silo is the same
  format with more cells along z, and it is tested only on synthetic files.
- **CarpetX openPMD-HDF5:** written from CarpetX's `io_openpmd.cxx` and the
  openPMD standard, and tested only on synthetic files. No real file has
  been read yet.

## Checklist for the low-resolution AsterX test run

What the run directory must contain for the driver to label it, and what the
first look at it must confirm.

### 1. 3D output in a format h5py can read

- [ ] **Preferred:** Silo,
  `CarpetX::out_silo_vars = "..."` with `CarpetX::out_silo_every = N`. This
  layout has been verified on real CarpetX output.
- [ ] **Or** openPMD with **`CarpetX::openpmd_format = "HDF5"`**. The default
  is ADIOS2 BP, which h5py cannot read. Use file-based iteration encoding:
  one `<sim>.it%08d.h5` per output iteration.
- [ ] Do not write 2D slices only. Features need 3D derivatives, and slices
  serve only for checks (`CarpetXReader(..., plane="xy")`).

### 2. Variables (all cell-centred, the AsterX defaults)

| feature input | CarpetX group / variable | Silo name / openPMD component |
|---|---|---|
| `rho_b` | `HydroBaseX::rho` | `hydrobasex_rho` |
| `P` | `HydroBaseX::press` | `hydrobasex_press` |
| `vel[0..2]` | `HydroBaseX::vel` (Valencia `v^i`) | `hydrobasex_velx`, `_vely`, `_velz` |
| `Bx, By, Bz` | `HydroBaseX::Bvec` | `hydrobasex_bvecx`, `_bvecy`, `_bvecz` |
| `smallb2` | `AsterX::b2small` (in the diagnostics group) | `asterx_b2small` |

- [ ] All five groups are in the 3D output at the same iterations.
- [ ] If `b2small` is in a group whose name is not `asterx_diagnostics`, pass
  `names=` to `CarpetXReader`, or edit `ASTERX_NAMES`.
- [ ] Helpful: also write `ADMBaseX::lapse`, `ADMBaseX::shift` and
  `ADMBaseX::metric` at one iteration. With them we can recompute `b^2` from
  `Bvec` and `vel` and check the unit convention (section 5).
- [ ] TSV lines (`CarpetX::out_tsv_vars`) for the same variables at one
  iteration. They give the 1D cross-check the reader tests already run on
  the FM torus.

### 3. Iterations

- [ ] At least two 3D output iterations, so that both a single-iteration run
  and `--watch` can be tested. One early and one after the field has
  developed is enough.
- [ ] The iteration numbers, the time between outputs, and whether a restart
  splits them over `output-NNNN` directories. The reader takes one directory;
  run the driver once per directory.

### 4. Levels and boxes

- [ ] The parfile: `CarpetX::max_num_levels`, the regridding settings, and
  the domain (`CarpetX::xmin..zmax`, `ncells_x..z`). Every level is written by
  default; say if some are not.
- [ ] How many processes wrote the Silo files (`*.silo.dir/*.pNNNNNN.silo`).
  Several data files per iteration are handled, but so far only one has been
  seen in real output.
- [ ] Whether there are black holes or excision. If so, provide their
  positions and radii per iteration: an AHFinder / BHaHAHA output file, or a
  table. Without them the horizon interior is labelled, and the Carpet
  model was trained with horizons masked.

### 5. To confirm on the first file (open questions)

- **Silo ghosts.** The reader uses a box's own ghost zones only where no
  box of the same level has valid cells, which is at the level's outer edge.
  Confirm that the ghosts written are synchronised and prolongated, not
  stale. If they are stale, drop the own-ghost step in
  `CarpetXReader._region` (one line). The halo there then comes from the
  coarser level by injection (`--fill coarse`, the default).
- **openPMD `chunkInfo`.** The reader assumes this layout of
  `chunkInfo_patch00_levLL` (int64):
  - per box, lo then hi, in z, y, x order, with hi exclusive;
  - level cell indices counted from the domain's first cell;
  - every box listed.

  It also assumes:
  - `gridGlobalOffset` is the domain's lower edge;
  - `position` is 0.5.

  See the `TODO(AsterX run)` in `_scan_openpmd`.
- **Units of `b2small`.**
  - The Carpet model's thresholds (`b2_min`, the winsor limits, `s_v`) come
    from IllinoisGRMHD's `smallb2`, where `P_mag = smallb2 / 2`, in `M = 1`
    units.
  - Check that AsterX's `b2small` follows the same convention: its `Bvec` is
    Heaviside-Lorentz, so `b^2 / 2` should again be the magnetic pressure.
  - Check that lengths and times are in the same `M = 1` units, because
    `chi_B`, `|grad ln b^2|` and `div v` carry units of 1/length and 1/time.

  If the conventions differ, the frozen thresholds and the model do not
  transfer, and that needs a decision before labelling.
- **Velocity.** `HydroBaseX::vel` is the Valencia 3-velocity, the same
  quantity as Carpet's `HydroBase::vel` used in training. Confirm this
  against the run's thorn list.
