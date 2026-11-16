# Changelog

All notable changes to aweSOM are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `aweSOM.npy_parts` adds sharded per-rank `.npy` outputs stored as
  `<stem>.parts/part-RRRR.npy` with an atomic `manifest.json` completion marker
  at format `awesom-npy-parts` version 1. `aweSOM` and `aweSOM.stream` re-export
  `open_npy`, `ShardedNpyArray`, and `stitch_npy`. `open_npy` opens plain `.npy`
  files and `.parts` directories: `mmap_mode="r"` returns a lazy `np.memmap` or
  `ShardedNpyArray`, and `mmap_mode=None` loads an `ndarray` in memory.
  `verify_npy_parts` checks each part's CRC-32 against the manifest, and
  `stitch_npy` and `python -m aweSOM.npy_parts stitch` assemble a single `.npy`
  file on demand with CRC checks.
- `aweSOM.sce`, `aweSOM.make_sce_clusters`, `aweSOM.sheet_stats.GridFileSource`,
  `aweSOM.pool_statistics`, `examples/grmhd-bbh/production_pool.py` in
  `--statistics-only` and `--collect` modes, and the GRMHD example scripts now
  accept sharded `.parts` outputs alongside plain `.npy` files, including mixed
  single-rank and multi-rank collections.

### Changed

- `aweSOM.stream.predict_stream(..., out=None)` now allocates the smallest of
  `uint8`, `uint16` and `int32` that holds every entry of `node_cluster`
  (`aweSOM.inference.narrow_label_dtype`), also in single-process calls; it
  returned `int32` before. Pass an `int32` `out` array, or cast the result,
  where signed arithmetic on the labels is needed.
- Multi-rank `production_pool.py::label_rows` writes per-rank
  `<tag>-{labels,nodes,distances}.parts/part-RRRR.npy` files and publishes
  `manifest.json` on rank 0 without copying other ranks' payloads through rank
  0. Single-rank runs still write plain `<tag>-{labels,nodes,distances}.npy`.
  The fit record's `labelling` dictionary replaces `copy_seconds` and
  `copy_bytes` with `layout`, `ranks`, `manifest_seconds`, and
  `directory_fsync`, and `fit_disk_budget` budgets `per_row * rows` at every MPI
  rank count.

## [1.9.0] - 2026-11-14

This release adds exact `float64` patch-pruning inference to `aweSOM.inference`
and `SOMModel.predict`, documents and tests the tie-breaking and non-finite BMU
contract, and fixes `float32` runner-up overflow handling in the screen kernel.

**Training, default `backend="screen"` outputs on all tested plasma and binary
black hole lattices, pickled lattices, and `format_version` 1 `SOMModel` files
are bitwise unchanged from `v1.8.0`.**

### Added

- `best_match_nodes_patch`, `lattice_patches`, and `default_patch_side` in
  `aweSOM.inference` for exact `float64` patch-pruning best-matching-unit
  search over `side x side` lattice patch bounding balls on `(xdim, ydim)`
  lattices. `default_patch_side` selects `side=4` up to `8,000` nodes and
  `side=5` above `8,000` nodes, and `best_match_nodes_patch` returns winners
  and squared distances bitwise equal to `best_match_nodes`.
- `backend="patch"` and `lattice_shape=(xdim, ydim)` on `aweSOM.predict` and
  `aweSOM.inference.predict`, plus `backend="patch"` on `SOMModel.predict`,
  which forwards `(self.xdim, self.ydim)` automatically. `Lattice.predict` and
  `Lattice.best_match` continue to take `"screen"`, `"numba"`, and `"legacy"`.
- Documented and tested the shared tie-breaking and non-finite contract across
  `best_match_nodes`, `best_match_nodes_screen`, and `best_match_nodes_patch`:
  exact `float64` ties return the lowest node index, including ties inside the
  `float32` rounding interval for `screen`, `NaN` candidate distances never
  win, and rows with no finite node distance return node `0` and squared
  distance `inf`.
- Tests in `tests/inference_patch_test.py` and expanded cases in
  `tests/inference_screen_test.py`, `tests/inference_test.py`, and
  `tests/model_test.py`.

### Fixed

- `_screen_kernel` in `aweSOM.inference` checks `s2[p] < np.inf` before
  accepting the `float32` margin, so finite `float64` inputs whose runner-up
  squared distance overflows `float32` fall back to the exact `float64` scan
  instead of accepting a `float32` winner prematurely.

### Documentation

- Added `docs/source/backends.rst` and linked it from `docs/source/index.rst`,
  added `"Inference backends"` to `README.md`, and updated
  `docs/source/model.rst`, `docs/source/testing.rst`, and `BENCHMARKS.md`.

## [1.8.0] - 2026-11-14

This release adds target-count cluster merging to `Lattice` and updates
`SOMModel` metadata and test skipping without JAX.

**Training, `predict`, and threshold-mode cluster maps and labels are bitwise
unchanged from `v1.7.0`; pickled lattices and `format_version` 1 models from
earlier releases load and predict as before.**

### Added

- `target_clusters` and `merge_cost_floor`, with a `0.05` default applied in
  target mode only, on `Lattice.assign_cluster_to_lattice`, plus
  `target_clusters` on `Lattice.merge_similar_centroids`. Target mode steps
  along the greedy merge path with `np.nextafter` until at most
  `target_clusters` clusters remain, then stores `merge_cost_used`,
  `n_clusters`, and `target_clusters` on the `Lattice`.
- `SOMModel.from_lattice` records `merge_cost_used`, `target_clusters`, and
  `n_clusters` in model metadata when present on the `Lattice`.
- Tests in `tests/merge_target_test.py`.

### Changed

- `SOMModel.from_lattice` now records `merge_cost_used` and `n_clusters` in
  threshold mode too, because `Lattice.merge_similar_centroids` sets
  `merge_cost_used` in both modes and `Lattice.assign_cluster_to_lattice` sets
  `n_clusters` in both modes.
- Target mode in `Lattice.merge_similar_centroids` raises `ValueError` unless
  `umat` has dtype `float64`.
- `setup.py` and `docs/source/conf.py` read `__version__` from
  `src/aweSOM/__init__.py`.

### Fixed

- `test_jax_gpu_available` and `test_jax_cpu_available` in
  `tests/sce_test.py` call `pytest.importorskip("jax")`, so environments
  without JAX skip them instead of failing.
- Corrected the `1.7.0` and `1.2.0` headings in `CHANGELOG.md` to `2026-11-14`
  and `2026-11-05` to match the local dates of their GitHub releases.

### Documentation

- Added `"Choosing a target cluster count"` to `docs/source/faqs.rst` and
  `README.md`, and updated `docs/source/testing.rst`.

## [1.7.0] - 2026-11-14

This release adds `aweSOM.model`, an opt-in frozen model artifact for
pickle-free inference.

**`SOMModel` and `Scaler` are new opt-in classes that save a trained map and
its feature scaler to HDF5 or plain text; training, `Lattice`, `predict`, and
every existing output are unchanged.** Pickled lattices from v1.1.0 through
v1.6.0 load and work as before.

### Added

- `SOMModel` in `aweSOM` and `aweSOM.model` freezes a trained `Lattice` via
  `SOMModel.from_lattice` and classifies raw features in chunks with
  `SOMModel.predict`. It supports `backend="screen"` by default and
  `backend="numba"`, and can also return Euclidean BMU distances in the scaled
  space.
- `Scaler` in `aweSOM` and `aweSOM.model` fits and applies per-feature float64
  affine scaling with kinds `"none"` via `Scaler.identity`, `"manual"`,
  `"minmax"`, and `"median-iqr"`. The `"manual"` kind subtracts the mean and
  divides by two standard deviations over `bulk_range`, matching
  `run_som.manual_scaling` on float64 data.
- `SOMModel.save` writes an HDF5 file at `format_version` 1, and
  `SOMModel.save_text` writes a self-describing plain-text file with `%.17g`
  floats for readers in other languages. `SOMModel.load` reads both formats
  back bit for bit, and files with `format_version` 1 load in every later 1.x
  release. `SOMModel.load` refuses a file with a newer `format_version`.
- `SOMModel` stores per-node 99th-percentile BMU distances in `node_dist_q99`,
  the global 99.9th-percentile BMU distance in `reject_distance` via
  `distance_quantiles`, and single-line string `metadata` for
  `format_version`, `awesom_version`, `created`, `git_commit`, and training
  parameters. `distance_quantiles_fast` also ships in `aweSOM.model`.
- Tests and fixtures in `tests/model_test.py`, `tests/inference_test.py`, and
  `tests/fixtures/`.

### Documentation

- `README.md` and the new docs page `docs/source/model.rst` show how to save a
  trained map and label new data.

### Known limitations

- `Scaler` always works in float64, whereas `run_som.manual_scaling` scales
  float32 arrays in float32. When a lattice was trained on float32 features
  scaled in float32, `SOMModel.predict` on the raw float32 features can assign
  a different cluster near boundaries, changing 59 of 1,000,000 labels on the
  51x36 binary black hole map and 11,544 of 10,000,000 labels, or 0.12
  percent, on the 91x64 map. Scaling the input in float32 before `predict`
  with `Scaler.identity`, or training on float64-scaled features, reproduces
  the training labels bit for bit.

## [1.6.0] - 2026-11-05

This release speeds up training and adds a `history` keyword to
`train_lattice`.

**Training results are unchanged: for the same seed, data and settings,
v1.6.0 trains bit for bit the same lattice, U-matrix, snapshots and clusters
as v1.5.0.** Training is 6 to 9 times faster with the default `history=True`
and 12 to 13 times faster with `history=False` (binary black hole features,
29x20 and 51x36 lattices, 100,000 and 1,000,000 steps). `train_lattice` has a
new `history` keyword.

### Added

- `history` keyword for `Lattice.train_lattice`. The default `history=True`
  keeps the snapshots in `lattice_history` and `umat_history` every
  `save_frequency` steps, as before. `history=False` skips them; the final
  lattice and U-matrix are still computed.
- Compiled numba kernels for the online training steps (`_som_schedule`,
  `_som_train_steps`, `_som_train_steps_aos`, `_pairwise_sum`,
  `_neighborhood_weight`, `_first_argmin`). They repeat the NumPy loop's
  floating-point operations in the same order, so the results are bitwise
  equal.
- Tests: `tests/fast_som_test.py` (the kernels against a frozen copy of the
  NumPy loop, both history settings, restarts) and
  `test_final_umat_uses_final_lattice_without_history` in `tests/som_test.py`.
- `BENCHMARKS.md`: training and labelling times for every release since
  v1.1.0, with speed-ups against the previous release and against v1.1.0;
  updated with every release.

### Changed

- `Lattice.fast_som` computes the learning rate and neighborhood schedule up
  front and runs the training steps in numba between progress messages and
  snapshots. The progress, shrink and decay messages are unchanged.
- A `restart_lattice` is copied in C order before training, so a
  Fortran-ordered restart array now trains exactly like a C-ordered one.
  v1.5.0 summed features in a different order for Fortran-ordered arrays
  with eight or more features, so such restarts can differ from v1.5.0.
- The learning-rate schedule is computed in float64. A NumPy float32
  `alpha_0` was decayed in float32 by v1.5.0; with a Python float, the
  usual case, nothing changes.

## [1.5.0] - 2026-11-05

This release makes training reproducible for a given seed, computes the final
U-matrix from the final lattice, and deprecates the old label route.

**Three changes alter trained maps.** First, one generator seeded from
`Lattice.seed` (42 by default) now draws both the initial lattice and the
sample order, so the same seed and data give the same lattice, and a map
trained with a given seed differs from the v1.4.0 map. Second, `Lattice.umat`
at the end of training is computed from the final lattice. v1.4.0 computed it
from the last saved snapshot, so U-matrices and clusters of a fresh training
can differ from v1.4.0 unless the number of steps minus one is a multiple of
`save_frequency`. Third, a float32 `restart_lattice` is copied to float64 and
trained in float64; v1.4.0 trained it in float32. In addition, the sampled
initial lattice (`sampling_type="sampling"`) can now draw the last data row,
which v1.4.0 never picked. Pickled lattices from earlier versions load and
work unchanged; their stored `umat` is not recomputed.

### Added

- Tests: `tests/deprecation_test.py`, a same-seed test in
  `tests/regression_test.py` and `test_final_umat_uses_final_lattice` in
  `tests/som_test.py`. `tests/small_input_and_restart_test.py` now checks the
  float64 restart contract and same-seed restarts.

### Changed

- `Lattice.train_lattice` copies `restart_lattice` to float64 before
  training. The array passed in is not modified, as in v1.1.2 and later.
- `run_som.py`, the README walkthrough, the example scripts and the notebooks
  now label data with `Lattice.predict`. The label files are unchanged. The
  `som_object` pickle written by `run_som.py` no longer stores
  `projection_1d` and `projection_2d`.
- `Lattice.plot_heat` calls `best_match` directly instead of
  `map_data_to_lattice`; it still sets `projection_1d` and `projection_2d`.

### Deprecated

- `Lattice.map_data_to_lattice` and `Lattice.assign_cluster_to_data` are
  deprecated since 1.5.0 and emit a `DeprecationWarning`. Use
  `Lattice.predict`, which returns the same labels. Both methods will be
  removed in a future release. `Lattice.assign_cluster_to_lattice` and
  `Lattice.best_match` are not deprecated.

### Fixed

- Training was not reproducible: the sample order came from an unseeded
  `np.random.default_rng()` and the initial lattice from the global NumPy
  stream. Both now come from `np.random.default_rng(Lattice.seed)`. A
  restart that continues training (`epoch > 1`) moves the seed on by one, as
  before.
- The final `Lattice.umat` came from the last snapshot instead of the final
  lattice.
- The sampled initial lattice drew row indices from `0` to `N - 2`; it now
  draws from `0` to `N - 1`.

## [1.4.0] - 2026-11-05

This release adds CPU inference (`aweSOM.inference`, `Lattice.predict`) and
changes how `Lattice.best_match` searches by default.

**`Lattice.best_match` now defaults to the float32 screen backend with float64
distance accumulation (`backend="screen"`).** `backend="legacy"` and
`full=True` keep the v1.3.0 behaviour. For finite float64 data the default
returns the same nodes as v1.3.0. With a float32 lattice and float32 data, a
point whose two nearest nodes are tied within float32 rounding can get the
other node, and the default never selects a node with non-finite weights,
where v1.3.0 returned the first NaN distance. `map_data_to_lattice` uses the
default.

**Label files are now uint8 or uint16.** `run_som.save_cluster_labels`
writes uint8 when the largest cluster id is below 256 and uint16 when it is
below 65536; the ids are unchanged.

### Added

- `aweSOM.inference` with `best_match_nodes` (float64 reference scan),
  `best_match_nodes_screen` (float32 screen checked in float64, same labels
  and distances bitwise) and `predict(weights, node_cluster, data)`, which
  returns int32 cluster labels and optionally the distances. `predict` and
  `best_match_nodes` are exported from `aweSOM`.
- `Lattice.predict(data)`: cluster labels for new data in one call, equal to
  `map_data_to_lattice` followed by `assign_cluster_to_data`.
- `Lattice.best_match(..., backend=)` with `"screen"` (default), `"numba"`
  and `"legacy"`; the original kernel stays as `Lattice._best_match_legacy`.
- Tests: `inference_test.py`, `inference_screen_test.py`,
  `lattice_wrappers_test.py`, `best_match_regression_test.py` (float32
  lattice and data, exact ties, non-finite rows and nodes, fewer than ten
  points) and a label dtype test in `run_som_test.py`.

### Changed

- `Lattice.best_match` default search, as above.
- `run_som.save_cluster_labels` file dtype, as above.

## [1.3.0] - 2026-11-05

This release adds clump helpers to the plasma turbulence example. The
aweSOM package itself is unchanged apart from its version number.

### Added

- `examples/plasma-turbulence/combine_clumps.py`. `get_clumps` and
  `label_clumps` split a 2D or 3D label array into connected clumps of equal
  value; `label_clumps` returns one id per cell, in the order of the
  `get_clumps` list, and can also join clumps across periodic boundaries.
  `merge_double_sheets` merges each double sheet that touches both a
  positive and a negative clump with the clumps it touches. It runs the
  array version `merge_double_sheets_fast` by default (`fast=True`);
  `fast=False` runs the original loop. Both return the same result. On the
  128^3 plasma sheet arrays `label_clumps` took 0.06 to 0.10 s where
  `get_clumps` took 5.5 to 21 s, with the same clumps.
- `examples/plasma-turbulence/requirements.txt` with `tqdm`, which the script
  imports. `tqdm` is not a dependency of the package.
- `tests/combine_clumps_test.py`. It is skipped when `tqdm` is not
  installed.

### Changed

- The test workflow installs `examples/plasma-turbulence/requirements.txt`,
  so CI runs `tests/combine_clumps_test.py`.

## [1.2.0] - 2026-11-05

This release rewrites the SCE ensemble loop in `sce.loop_over_all_clusters`
to work on contingency tables and adds an opt-in path that does not write the
per-cluster mask files. Mask files are now float32 by default, so they are
not byte-identical to v1.1.3; `mask_dtype=np.float64` restores them exactly.
`multimap_mappings.txt` and the combined SCE cluster maps are unchanged.

### Changed

- `sce.loop_over_all_clusters` builds one contingency table per pair of runs
  instead of comparing full-size cluster masks. The quality index of two
  clusters depends only on their sizes and their overlap, so
  `multimap_mappings.txt` is byte-identical to v1.1.3, and so is every
  `mask-*.npy` file with `mask_dtype=np.float64`. On 28 plasma label maps of
  128^3 cells the loop took 2.6 s instead of 338 s.
- Mask files are written in float32 by default, half the size of the
  float64 files of v1.1.3. float32 rounds each Gsum at about 6e-8 relative.
  `mask_dtype=np.float64`, or `--mask_dtype float64` for `sce.py`, writes the
  v1.1.3 files exactly. `combine_separated_clusters` already summed the masks
  in float32, so the combined maps it returns do not change.
- The ensemble loop always runs in NumPy, also when JAX finds a GPU. With
  JAX on a GPU, v1.1.3 ran the loop on the device in float32; v1.2.0 computes
  the same float64 Gsum values there as on the CPU. `create_mask` and
  `compute_SQ` still use JAX on a GPU.
- `sce.find_number_of_clusters` counts distinct ids with `np.bincount` when
  the labels are non-negative integers below 2^24, and with `np.unique`
  otherwise. The counts are the same.
- `make_sce_clusters.get_gsum_values` recognises a Gsum line by its form,
  `<cid> <Gsum>`, and skips blank lines. v1.1.3 took any line containing a
  `-` for a run name, including a Gsum written as `2.5e-05`.
- `make_sce_clusters.get_sce_cluster_separation` returns ranges that cover
  the whole list when fewer than two minima remain after it drops the first.
  v1.1.3 returned no ranges in that case and raised `IndexError` when it
  found no minimum.

### Added

- `write_masks` keyword of `sce.loop_over_all_clusters` (default `True`) and
  the `--no_masks` flag of `sce.py`. With `write_masks=False` the loop writes
  only `multimap_mappings.txt`, the same file as with masks.
- `mask_dtype` keyword of `sce.loop_over_all_clusters` (default
  `np.float32`) and the `--mask_dtype {float32,float64}` flag of `sce.py`.
- `make_sce_clusters.combine_separated_clusters_from_labels` and the
  `--labels_path` flag of `make_sce_clusters.py`. They build the combined SCE
  map from the label files and `multimap_mappings.txt`, without mask files.
  The map is byte-identical to the one from `combine_separated_clusters`;
  its `mask_dtype` keyword names the dtype the mask files were written in.
- `sce.contingency_table` and `sce.pair_quality_from_counts`.
- `make_sce_clusters.gsum_derivative` and `stack_sce_labels`, which run the
  `make_sce_clusters.py` steps from the Gsum file to one label per cell, and
  `anchored_sce_labels` with `adjusted_rand_from_table`, which group clusters
  by their overlap with a medoid run instead of by Gsum rank.
- `tests/sce_loop_test.py`. It compares the loop with a frozen copy of the
  v1.1.3 mask-by-mask loop, byte for byte with `mask_dtype=np.float64`, and
  the label path with the mask path. A test checks the default float32 masks
  against the float64 ones. Five tests in `tests/make_sce_clusters_test.py`.

### Fixed

- `tests/som_test.py` trained its map without a seed, so
  `test_assign_cluster_to_lattice` could fail on some runs. The fixture now
  sets the NumPy global seed and the seed of the shuffle generator.

## [1.1.3] - 2026-11-04

This release rewrites the cluster merge in `Lattice.merge_similar_centroids`
as a loop. Merged cluster maps are identical to v1.1.2.

### Changed

- `Lattice.merge_similar_centroids` no longer calls itself once per merge. It
  merges the cheapest pair in a loop and stops at the same point as before.
  It returns the same centroids and sets the same `nodes_count` as v1.1.2.
- A map that needs more merges than Python's recursion limit (1000 by
  default) now merges. v1.1.2 raised `RecursionError` on such maps.
- The merge computes each pairwise cost once and reuses it in later merge
  rounds, so large lattices merge faster.

### Added

- `tests/merge_test.py`. It compares the merge with a frozen copy of the
  v1.1.2 recursive merge on 20 trained maps at three thresholds. It also
  merges a 60 by 40 map under a lowered recursion limit.

## [1.1.2] - 2026-11-04

This release fixes crashes in training, cluster merging, SCE and the
`run_som.py` command line. For every input that ran in v1.1.1, trained
lattices, cluster maps and SCE values are the same.

### Fixed

- `Lattice.train_lattice` raised `ZeroDivisionError` when `train` was below
  25 with the default decaying learning rate, or below 10 with any learning
  rate. Short runs now train.
- `Lattice.train_lattice` raised `ZeroDivisionError` when the larger lattice
  side was 7, and when a training batch had fewer than 4 steps. Both cases
  now train.
- `Lattice.train_lattice(restart_lattice=w)` trained `w` in place, so the
  caller's array changed. Training now works on a copy of `w`.
- `Lattice.assign_cluster_to_lattice` raised `IndexError` when the U-matrix
  had one centroid. It now returns a single cluster.
- `sce.compute_SQ` divided zero by zero when a mask was empty. NumPy issued a
  `RuntimeWarning`, or raised `FloatingPointError` under
  `np.errstate(all="raise")`. The function now returns 0 before it divides.
  The result is unchanged.
- SCE removed the `.npy` suffix with `str.strip(".npy")`. That call also
  removed every `.`, `n`, `p` or `y` at either end of a run name, so the run
  `labels.run-1p.npy` appeared as `labels.run-1` in `multimap_mappings.txt`
  and in its mask file names. SCE now removes only the suffix.
- The `run_som.py` command line failed before it loaded any data. Run as a
  script, it raised `ImportError` on a relative import. Run with
  `python -m aweSOM.run_som`, it raised `AttributeError` because it read
  `args.neurons_path`, which the parser does not define. It now uses an
  absolute import and reads `args.lattice_path`.
- `Lattice.best_match` printed progress every `N // 10` points, which is a
  modulus of 0 for fewer than 10 points. Numba's parallel loop returned 0
  there instead of raising, so results were correct. The divisor is now at
  least 1.
- The README and the testing page said that `sce_test.py` fails partially
  without a CUDA GPU. They now state which tests pass with and without JAX.

### Changed

- No numerical change. Training from a restart array gives the same
  `Lattice.lattice` and `Lattice.umat` as v1.1.1, in the dtype of the array
  you pass. The array
  itself no longer changes during training, so code that read the trained
  weights from it must read `Lattice.lattice` instead.
- `sce_test.py` reloads `aweSOM.sce` under the patched JAX backend and
  compares values instead of types, so it no longer depends on the real
  backend of the machine. Tests write their files into temporary directories
  and use seeded input data.
- `src/aweSOM/som.py` uses LF line endings.

### Added

- `tests/regression_test.py` and `tests/small_input_and_restart_test.py`,
  with tests for the training, merging, `best_match` and SCE fixes above. The
  restart tests compare the trained weights, the U-matrix and the cluster map
  with values from v1.1.1, stored in
  `tests/fixtures/restart_training_v1.1.1.npz`.

## [1.1.1] - 2026-11-04

The only package code change is one NumPy-compatibility line (below);
otherwise `src/aweSOM` is identical to v1.1.0 apart from the version string.
Trained lattices, cluster maps and SCE outputs are unchanged.

### Fixed

- `make_sce_clusters.combine_separated_clusters` passes the shape to
  `np.reshape` positionally. The `newshape=` keyword it used was removed in
  NumPy 2.4, where the function raised `TypeError`. The result is unchanged.
- README Iris walkthrough runs as written:
  - `plot_centroids` is created before use;
  - the comparison legend uses `scatter_som` instead of the undefined `scatter`;
  - the U-matrix plot defines `xdim` and `ydim`;
  - the SCE loop passes `true_labels` instead of the undefined `labels`;
  - the SCE loop computes `projection_2d = map.map_data_to_lattice()` instead
    of using an undefined `projection_2d`;
  - the comparison legend labels the plotted clusters (`np.unique(som_labels)`), not every lattice cluster;
  - the SCE shell steps keep the working directory: labels are moved with `mkdir -p` and SCE runs in a subshell from the scratch directory where the labels were written.
- Read the Docs installs aweSOM from the checkout instead of the `aweSOM`
  package on PyPI, so the API reference documents this repository.
- The API reference no longer documents every module twice (the duplicate
  autodoc page `aweSOM.rst` is removed), and the docs build without warnings.
- The source distribution includes `requirements.txt`, which `setup.py`
  reads; building a wheel from the sdist failed without it (`MANIFEST.in`).

### Changed

- `CITATION.cff` now records the software `version` and `date-released`;
  the JOSS preferred citation and the Zenodo DOI are unchanged.

### Added

- Test workflow (`.github/workflows/tests.yml`) that runs the test suite on
  Ubuntu and macOS with Python 3.9 and 3.12 on every push and pull request to
  `public-release`.
- README smoke test (`tests/readme_smoke_test.py`) that runs the README Iris
  SOM and SCE walkthrough with a reduced number of training steps.
- This changelog.

## [1.1.0] - 2025-03-27

JOSS publication version.

[1.9.0]: https://github.com/tvh0021/aweSOM/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/tvh0021/aweSOM/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/tvh0021/aweSOM/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/tvh0021/aweSOM/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/tvh0021/aweSOM/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/tvh0021/aweSOM/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/tvh0021/aweSOM/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/tvh0021/aweSOM/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/tvh0021/aweSOM/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/tvh0021/aweSOM/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/tvh0021/aweSOM/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/tvh0021/aweSOM/releases/tag/v1.1.0
