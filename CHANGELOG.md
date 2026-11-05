# Changelog

All notable changes to aweSOM are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [1.2.0] - 2026-11-04

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

[1.5.0]: https://github.com/tvh0021/aweSOM/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/tvh0021/aweSOM/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/tvh0021/aweSOM/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/tvh0021/aweSOM/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/tvh0021/aweSOM/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/tvh0021/aweSOM/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/tvh0021/aweSOM/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/tvh0021/aweSOM/releases/tag/v1.1.0
