# Changelog

All notable changes to aweSOM are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.2.0]: https://github.com/tvh0021/aweSOM/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/tvh0021/aweSOM/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/tvh0021/aweSOM/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/tvh0021/aweSOM/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/tvh0021/aweSOM/releases/tag/v1.1.0
