# Benchmarks

## Purpose

Every release records its training and labelling times here, measured on the same machine and data as the earlier releases. Speed-ups are given against the previous release and against v1.1.0, the first public release.

## Method

Machine: Apple M1 Max laptop (10 cores, 64 GB memory) on mains power, macOS 27.0.1, Python 3.12.7, NumPy 2.5.4, numba 0.68.1. Each release ran from its own virtual environment with that release installed, in fresh processes, one measurement at a time. Numba used all cores. Other desktop applications stayed open, as in normal use.

Data:

- Binary black hole case: five plasma features per cell from a binary black hole simulation snapshot (iteration 180224), float32, scaled with `run_som.manual_scaling`. The 29x20 case trains on the first 100,000 points for 100,000 steps; the 51x36 case trains on the first 1,000,000 points for 1,000,000 steps. The lattice sizes come from `run_som.initialize_lattice` with ratio 0.7. Settings: `alpha_0=0.5`, `alpha_type="decay"`, `sampling_type="uniform"`, `Lattice.seed = 42`.
- Plasma case: the fiducial plasma turbulence model of the plasma-turbulence example (a 64x37 lattice with four features, trained on 2,097,152 points), loaded from its pickle. The first 1,000,000 of its stored points are labelled with its own lattice, so every release labels the same lattice.

Training time is the wall time of the `train_lattice` call only, with each release's defaults (v1.6.0 also with `history=False`). Each measurement process first trains an 8x6 lattice for 500 steps, so numba compilation is not in the training time; v1.6.0 compiles in about 3.1 s, earlier releases in 1.6 to 1.9 s. Labelling time covers the labels of all points after `assign_cluster_to_lattice(smoothing=None, merge_cost=0.2)`, which is timed separately, and follows a warm-up on 2,000 points. Releases up to v1.3.0 label with `map_data_to_lattice` followed by `assign_cluster_to_data`; from v1.4.0 on the default is `predict`, and every other route is listed too. Every time is the mean of two runs. Peak memory is the `ru_maxrss` of the 51x36 training process (1 GB = 10^9 bytes).

v1.1.0 to v1.4.0 do not seed training, so their maps differ from run to run and only cluster counts compare (4 to 6 clusters at merge cost 0.2 for 51x36). v1.5.0 and later are seeded, and v1.6.0 reproduces v1.5.0 bit for bit: lattice, U-matrix, clusters and labels, for both training cases and both history settings. The plasma case gives 9 clusters and the same labels in every release.

## Training

| Version | Date | 29x20 (s) | 51x36 (s) | Speed-up vs previous | Speed-up vs v1.1.0 | Peak memory (GB) |
|---|---|---|---|---|---|---|
| v1.1.0 | 2025-03-27 | 3.99 | 103 | 1.0x | 1.0x | 0.37 |
| v1.1.1 | 2026-11-04 | 3.94 | 103 | 1.0x | 1.0x | 0.38 |
| v1.1.2 | 2026-11-04 | 4.02 | 102 | 1.0x | 1.0x | 0.38 |
| v1.1.3 | 2026-11-04 | 4.05 | 104 | 0.98x | 0.99x | 0.38 |
| v1.2.0 | 2026-11-04 | 4.10 | 104 | 1.0x | 0.99x | 0.38 |
| v1.3.0 | 2026-11-05 | 4.05 | 105 | 0.99x | 0.98x | 0.38 |
| v1.4.0 | 2026-11-05 | 4.10 | 105 | 1.0x | 0.98x | 0.38 |
| v1.5.0 | 2026-11-05 | 4.09 | 106 | 0.98x | 0.96x | 0.38 |
| v1.6.0, history on | 2026-11-05 | 0.691 | 11.8 | 9.0x | 8.7x | 0.43 |
| v1.6.0, history off | 2026-11-05 | 0.310 | 9.17 | 12x | 11x | 0.43 |

Speed-ups use the 51x36 times. The previous release of both v1.6.0 rows is v1.5.0.

## Labelling

| Version | Route or backend | 51x36, 1,000,000 points (s) | Plasma, 1,000,000 points (s) | Speed-up vs previous (plasma) | Speed-up vs v1.1.0 (plasma) |
|---|---|---|---|---|---|
| v1.1.0 | `map_data_to_lattice` + `assign_cluster_to_data` | 4.06 | 5.29 | 1.0x | 1.0x |
| v1.1.1 | `map_data_to_lattice` + `assign_cluster_to_data` | 4.03 | 5.23 | 1.0x | 1.0x |
| v1.1.2 | `map_data_to_lattice` + `assign_cluster_to_data` | 4.02 | 5.22 | 1.0x | 1.0x |
| v1.1.3 | `map_data_to_lattice` + `assign_cluster_to_data` | 3.96 | 5.43 | 0.96x | 0.97x |
| v1.2.0 | `map_data_to_lattice` + `assign_cluster_to_data` | 4.21 | 5.49 | 0.99x | 0.96x |
| v1.3.0 | `map_data_to_lattice` + `assign_cluster_to_data` | 4.97 | 5.94 | 0.92x | 0.89x |
| v1.4.0 | `predict`, screen (default) | 0.457 | 0.540 | 11x | 9.8x |
| v1.4.0 | `predict`, numba | 0.464 | 0.707 | 8.4x | 7.5x |
| v1.4.0 | `map_data_to_lattice` + `assign_cluster_to_data` | 0.448 | 0.547 | 11x | 9.7x |
| v1.5.0 | `predict`, screen (default) | 0.550 | 0.554 | 0.97x | 9.5x |
| v1.5.0 | `predict`, numba | 0.551 | 0.707 | 0.76x | 7.5x |
| v1.5.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.548 | 0.552 | 0.98x | 9.6x |
| v1.6.0 | `predict`, screen (default) | 0.531 | 0.538 | 1.0x | 9.8x |
| v1.6.0 | `predict`, numba | 0.504 | 0.738 | 0.75x | 7.2x |
| v1.6.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.569 | 0.671 | 0.83x | 7.9x |

Speed-ups compare the plasma time with the previous release's default route and with v1.1.0, because the plasma lattice is the same in every release while each release trains its own 51x36 lattice. All routes of a release give the same labels. Times under one second varied by up to 20 percent between the two runs, so differences of that size between releases that did not change the labelling code are noise.

Cluster assignment (`assign_cluster_to_lattice`, merge cost 0.2) took 1.4 s for 51x36 and 2.4 s for the plasma lattice in v1.1.0 to v1.1.2, and 0.11 to 0.12 s and 0.18 to 0.19 s from v1.1.3 on.

## What changed where

- v1.1.3: cluster assignment became 12 to 13 times faster (2.35 s to 0.18 s on the plasma lattice). `Lattice.merge_similar_centroids` merges in a loop and computes each pairwise cost once instead of calling itself once per merge.
- v1.3.0: labelling read 18 percent slower for 51x36 and 8 percent slower for plasma. The package code is unchanged apart from the version number, and one of the two runs was slow (5.39 s and 4.56 s), so this is run-to-run noise.
- v1.4.0: labelling became about 10 times faster (5.94 s to 0.54 s on the plasma lattice). `Lattice.predict` and `Lattice.best_match` use the new float32 screen backend with float64 checks, and `map_data_to_lattice` uses it too.
- v1.5.0: 51x36 labelling read 20 percent slower, plasma labelling did not change. The labelling code is the same as in v1.4.0; the 51x36 lattice differs because training is now seeded, and the difference is within the run-to-run spread.
- v1.6.0: training became 9.0 times faster with the default `history=True` and 12 times faster with `history=False` (51x36, 106 s to 11.8 s and 9.17 s). `Lattice.fast_som` runs the online training steps in compiled numba kernels, and `history=False` skips the snapshots. Compilation takes about 1.2 s longer than in v1.5.0, and peak memory rose from 0.38 GB to 0.43 GB.

## Reproducing

The benchmark harness and every release's raw results live in the release results directory on the maintainer's machine (`releases/benchmarks`, with the scripts in `harness/` and one JSON file per release in `results/`). To add a release, create a Python 3.12 virtual environment, install the release tag into it (`pip install git+https://github.com/tvh0021/aweSOM.git@vX.Y.Z`), run `bench_release.py --version X.Y.Z --venv-python <venv>/bin/python --out results/X.Y.Z.json` with that environment's Python, then run `make_table.py`, which rewrites `results/summary.md` from all the JSON files; copy the new rows into this file. Run the measurements on the same machine on mains power, one at a time, and rerun the older releases too if the machine, operating system or NumPy and numba versions change.
