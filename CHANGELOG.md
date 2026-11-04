# Changelog

All notable changes to aweSOM are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.1.1]: https://github.com/tvh0021/aweSOM/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/tvh0021/aweSOM/releases/tag/v1.1.0
