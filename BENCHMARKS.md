# Benchmarks

## Purpose

Every release records its training and labelling times here, measured on the same machine and data as the earlier releases. Speed-ups are given against the previous release and against v1.1.0, the first public release.

## Method

Machine: Apple M1 Max laptop (10 cores, 64 GB memory) on mains power, macOS 27.0.1, Python 3.12.7, NumPy 2.5.4 (2.5.5 for v1.10.0), numba 0.68.1. Each release ran from its own virtual environment with that release installed, in fresh processes, one measurement at a time. Numba used all cores. Other desktop applications stayed open, as in normal use.

Data:

- Binary black hole case: five plasma features per cell from a binary black hole simulation snapshot (iteration 180224), float32, scaled with `run_som.manual_scaling`. The 29x20 case trains on the first 100,000 points for 100,000 steps; the 51x36 case trains on the first 1,000,000 points for 1,000,000 steps. The lattice sizes come from `run_som.initialize_lattice` with ratio 0.7. Settings: `alpha_0=0.5`, `alpha_type="decay"`, `sampling_type="uniform"`, `Lattice.seed = 42`.
- Plasma case: the fiducial plasma turbulence model of the plasma-turbulence example (a 64x37 lattice with four features, trained on 2,097,152 points), loaded from its pickle. The first 1,000,000 of its stored points are labelled with its own lattice, so every release labels the same lattice.

Training time is the wall time of the `train_lattice` call only, with each release's defaults (from v1.6.0 on also with `history=False`). Each measurement process first trains an 8x6 lattice for 500 steps, so numba compilation is not in the training time; v1.6.0 to v1.10.0 compile in 2.7 to 3.1 s, earlier releases in 1.6 to 1.9 s. Labelling time covers the labels of all points after `assign_cluster_to_lattice(smoothing=None, merge_cost=0.2)`, which is timed separately, and follows a warm-up on 2,000 points. Releases up to v1.3.0 label with `map_data_to_lattice` followed by `assign_cluster_to_data`; from v1.4.0 on the default is `predict`, and every other route is listed too. For v1.1.0 through v1.8.0, every time is the mean of two runs, recorded in cold-cache worker processes with an empty `NUMBA_CACHE_DIR` per subprocess at per-worker 1-minute system load averages between `4.3` and `21.0`; that cold compilation and background load slowed `screen` by `2.6x` to `3.1x` and `numba` by `1.4x` to `1.8x` relative to warm `.nbc`-cache 5-run medians measured at 1-minute load `< 2.0`. Peak memory is the `ru_maxrss` of the 51x36 training process (1 GB = 10^9 bytes).

From `v1.9.0` onward, `bench_1.9.0.py` (and `bench_1.10.0.py` for `v1.10.0`) waits until 1-minute `os.getloadavg()[0] < 2.0` before every cross-release timed subprocess, with recorded `load_before` values from `1.74` to `1.99` across all 13 `v1.9.0` workers and from `1.81` to `1.96` across all 6 `v1.10.0` cross-release workers. It pre-warms an isolated per-venv `NUMBA_CACHE_DIR` before labelling workers while keeping an empty cache in training workers so `compile_s` (`2.72 s` for `v1.9.0`, `2.78 s` to `2.82 s` for `v1.10.0`) still measures cold compilation, runs `5` timed repetitions after warm-up, and reports the 5-run median. To compare `v1.9.0` on equal terms without altering historical rows, `bench_1.9.0.py` also measured `v1.1.0 (idle)` and `v1.8.0 (idle)` labelling under the same `< 2.0` load gate, warm `.nbc` cache, and 5-run median protocol in `1.1.0_idle.json` and `1.8.0_idle.json`. For the `v1.10.0` `100x100` CPU streaming scaling table, `bench_1.10.0.py` relaxed the 1-minute idle limit from `2.0` to `4.0` after a 30 s wait (`load_before` `2.25` to `3.89`), so those rows are descriptive scaling measurements rather than controlled comparisons under the `< 2.0` idle protocol.

v1.1.0 to v1.4.0 do not seed training, so their maps differ from run to run and only cluster counts compare (4 to 6 clusters at merge cost 0.2 for 51x36). v1.5.0 and later are seeded, and v1.6.0 to v1.10.0 reproduce v1.5.0 bit for bit: lattice, U-matrix, clusters and labels, for both training cases and both history settings. The plasma case gives 9 clusters and the same labels in every release.

## Training

| Version | Date | 29x20 (s) | 51x36 (s) | Speed-up vs previous | Speed-up vs v1.1.0 | Peak memory (GB) |
|---|---|---|---|---|---|---|
| v1.1.0 | 2025-03-27 | 3.99 | 103 | 1.0x | 1.0x | 0.37 |
| v1.1.1 | 2026-11-04 | 3.94 | 103 | 1.0x | 1.0x | 0.38 |
| v1.1.2 | 2026-11-04 | 4.02 | 102 | 1.0x | 1.0x | 0.38 |
| v1.1.3 | 2026-11-04 | 4.05 | 104 | 0.98x | 0.99x | 0.38 |
| v1.2.0 | 2026-11-05 | 4.10 | 104 | 1.0x | 0.99x | 0.38 |
| v1.3.0 | 2026-11-05 | 4.05 | 105 | 0.99x | 0.98x | 0.38 |
| v1.4.0 | 2026-11-05 | 4.10 | 105 | 1.0x | 0.98x | 0.38 |
| v1.5.0 | 2026-11-05 | 4.09 | 106 | 0.98x | 0.96x | 0.38 |
| v1.6.0, history on | 2026-11-05 | 0.691 | 11.8 | 9.0x | 8.7x | 0.43 |
| v1.6.0, history off | 2026-11-05 | 0.310 | 9.17 | 12x | 11x | 0.43 |
| v1.7.0, history on | 2026-11-14 | 0.667 | 11.3 | 1.0x | 9.1x | 0.43 |
| v1.7.0, history off | 2026-11-14 | 0.306 | 8.89 | 1.0x | 12x | 0.43 |
| v1.8.0, history on | 2026-11-14 | 0.656 | 11.2 | 1.0x | 9.2x | 0.43 |
| v1.8.0, history off | 2026-11-14 | 0.308 | 8.85 | 1.0x | 12x | 0.43 |
| v1.9.0, history on | 2026-11-14 | 0.656 | 11.1 | 1.0x | 9.3x | 0.43 |
| v1.9.0, history off | 2026-11-14 | 0.304 | 8.73 | 1.0x | 12x | 0.43 |
| v1.10.0, history on | 2026-11-18 | 0.653 | 11.2 | 0.99x | 9.2x | 0.46 |
| v1.10.0, history off | 2026-11-18 | 0.303 | 8.73 | 1.0x | 12x | 0.43 |

Speed-ups use the 51x36 times against the previous release's matching history setting when available, or its default row, and against v1.1.0.

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
| v1.7.0 | `predict`, screen (default) | 0.435 | 0.508 | 1.1x | 10x |
| v1.7.0 | `predict`, numba | 0.506 | 0.675 | 0.80x | 7.8x |
| v1.7.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.472 | 0.542 | 0.99x | 9.8x |
| v1.7.0 | `SOMModel.predict`, screen (scaler included) | 0.444 | 0.562 | 0.96x | 9.4x |
| v1.8.0 | `predict`, screen (default) | 0.434 | 0.513 | 0.99x | 10x |
| v1.8.0 | `predict`, numba | 0.453 | 0.683 | 0.74x | 7.7x |
| v1.8.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.418 | 0.542 | 0.94x | 9.8x |
| v1.8.0 | `SOMModel.predict`, screen (scaler included) | 0.440 | 0.542 | 0.94x | 9.8x |
| v1.1.0 (idle) | `map_data_to_lattice` + `assign_cluster_to_data` | 3.96 | 5.17 | 1.0x | 1.0x |
| v1.8.0 (idle) | `predict`, screen (default) | 0.149 | 0.168 | 1.0x | 31x |
| v1.8.0 (idle) | `predict`, numba | 0.317 | 0.388 | 0.43x | 13x |
| v1.8.0 (idle) | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.155 | 0.176 | 0.96x | 29x |
| v1.8.0 (idle) | `SOMModel.predict`, screen (scaler included) | 0.161 | 0.180 | 0.94x | 29x |
| v1.9.0 | `predict`, screen (default) | 0.163 | 0.177 | 0.95x | 29x |
| v1.9.0 | `predict`, numba | 0.317 | 0.395 | 0.43x | 13x |
| v1.9.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.150 | 0.173 | 0.97x | 30x |
| v1.9.0 | `aweSOM.predict`, patch | 0.119 | 0.118 | 1.4x | 44x |
| v1.9.0 | `SOMModel.predict`, screen (scaler included) | 0.170 | 0.179 | 0.94x | 29x |
| v1.9.0 | `SOMModel.predict`, patch (scaler included) | 0.138 | 0.133 | 1.3x | 39x |
| v1.10.0 | `predict`, screen (default) | 0.154 | 0.175 | 1.0x | 29x |
| v1.10.0 | `predict`, numba | 0.319 | 0.387 | 0.46x | 13x |
| v1.10.0 | `map_data_to_lattice` + `assign_cluster_to_data` (deprecated) | 0.150 | 0.186 | 0.95x | 28x |
| v1.10.0 | `aweSOM.predict`, patch | 0.129 | 0.121 | 1.5x | 43x |
| v1.10.0 | `predict_stream`, screen | 0.150 | 0.168 | 1.0x | 31x |
| v1.10.0 | `predict_stream`, patch | 0.118 | 0.123 | 1.4x | 42x |
| v1.10.0 | `SOMModel.predict`, screen (scaler included) | 0.169 | 0.174 | 1.0x | 30x |
| v1.10.0 | `SOMModel.predict`, patch (scaler included) | 0.134 | 0.134 | 1.3x | 39x |

Speed-ups compare the plasma time with the previous release's default route and with v1.1.0, because the plasma lattice is the same in every release while each release trains its own 51x36 lattice. For the `v1.8.0 (idle)` and `v1.9.0` rows, speed-ups compare against the `v1.8.0 (idle)` default route at `0.168 s` and against `v1.1.0 (idle)` at `5.17 s`; for the `v1.10.0` rows, speed-ups compare against the `v1.9.0` default route at `0.177 s` and against `v1.1.0 (idle)` at `5.17 s`. All `Lattice` and `predict_stream` routes of a release give the same labels (`predict_stream(..., out=None)` returns `uint8` on these 6-cluster and 9-cluster maps); the `SOMModel.predict` row applies `Scaler.fit(raw, kind="manual")` to raw float32 features on 51x36 (46 of 1,000,000 boundary labels differ from float32-scaled training on this benchmark's own 51x36 lattice; 59 on the paper's lattice) and `Scaler.identity` on the stored float64 plasma array (identical labels). For the historical two-run rows (`v1.1.0` through `v1.8.0`), times under one second varied by up to 20 percent between the two runs, so differences of that size between releases that did not change the labelling code are noise.

Cluster assignment (`assign_cluster_to_lattice`, merge cost 0.2) took 1.4 s for 51x36 and 2.4 s for the plasma lattice in v1.1.0 to v1.1.2, and 0.11 to 0.12 s for 51x36 and 0.18 to 0.19 s for the plasma lattice from v1.1.3 to v1.8.0. The v1.9.0 idle median was 0.175 s, and the v1.10.0 idle median was 0.173 s (0.110 s for 51x36).

On a `100x100` binary black hole lattice of `10,000` nodes trained for `100,000` steps with seed `42` and `47` initial centroids (`releases/1.8.0/benchmarks/target_100x100.json`), `assign_cluster_to_lattice(target_clusters=10)` took `2.86 s` as the mean of three runs after one warm-up call and returned `10` clusters at `merge_cost_used` `0.112`. Threshold mode at `merge_cost=0.2` on the same lattice took `2.85 s` and returned `7` clusters, because both modes walk the same greedy pairwise cost loop.

Across lattice sizes and weight smoothness (`releases/1.9.0/benchmarks/patch_100x100.json`), the table below compares `patch`, `screen`, and `numba` in `v1.9.0` against `screen` and `numba` in `v1.8.0` on identical saved maps: `100x100` with `10,000` nodes and `400` patches of side `5`, `51x36` with `1,836` nodes and `117` patches of side `4`, `plasma` `64x37` with `2,368` nodes and `160` patches of side `4`, `29x20` with `580` nodes and `40` patches of side `4`, and an untrained `51x36` lattice with random `N(0, 1)` weights. Each time is the 5-run median after a `2,000`-point warm-up with a pre-warmed `.nbc` cache and numba compilation excluded, started at 1-minute `load_before` `1.80` for `v1.9.0` and `1.93` for `v1.8.0`, and `SOMModel.predict` uses `Scaler.identity(F)`. On trained maps, `patch` evaluates `218` of `10,000` nodes per point on `100x100` for `46x` pruning, `85.1` of `1,836` on `51x36` for `22x` pruning, `78.0` of `2,368` on `plasma` for `30x` pruning, and `75.5` of `580` on `29x20` for `7.7x` pruning. On untrained random `N(0, 1)` weights on `51x36`, overlapping patch bounding balls prevent pruning and `patch` evaluates `1,835.7` of `1,836` nodes per point for `1.0x` pruning. Winner indices, squared distances, and cluster labels agree bit for bit across all backends and both versions.

| Lattice | Points | Version | Backend | `aweSOM.predict` (s) | `aweSOM.predict` vs `v1.9.0` screen / numba | `SOMModel.predict` (s) | `SOMModel.predict` vs `v1.9.0` screen / numba | Candidate nodes / point | Pruning |
|---|---|---|---|---|---|---|---|---|---|
| 100x100 | 100,000 | v1.9.0 | `patch` | 0.0355 | 2.8x / 5.1x | 0.0423 | 2.0x / 4.0x | 218 + 400 centroids | 46x |
| 100x100 | 100,000 | v1.9.0 | `screen` | 0.0997 | 1.0x / 1.8x | 0.0829 | 1.0x / 2.0x | 10,000 float32 + 10.4 float64 | 1.0x |
| 100x100 | 100,000 | v1.9.0 | `numba` | 0.182 | 0.55x / 1.0x | 0.169 | 0.49x / 1.0x | 10,000 | 1.0x |
| 100x100 | 100,000 | v1.8.0 | `screen` | 0.0839 | 1.2x / 2.2x | 0.0824 | 1.0x / 2.1x | 10,000 float32 + 10.4 float64 | 1.0x |
| 100x100 | 100,000 | v1.8.0 | `numba` | 0.191 | 0.52x / 0.95x | 0.179 | 0.46x / 0.94x | 10,000 | 1.0x |
| 100x100 | 1,000,000 | v1.9.0 | `patch` | 0.298 | 2.8x / 5.7x | 0.319 | 2.7x / 5.1x | 218 + 400 centroids | 46x |
| 100x100 | 1,000,000 | v1.9.0 | `screen` | 0.827 | 1.0x / 2.0x | 0.845 | 1.0x / 1.9x | 10,000 float32 + 11.5 float64 | 1.0x |
| 100x100 | 1,000,000 | v1.9.0 | `numba` | 1.69 | 0.49x / 1.0x | 1.64 | 0.52x / 1.0x | 10,000 | 1.0x |
| 100x100 | 1,000,000 | v1.8.0 | `screen` | 0.878 | 0.94x / 1.9x | 0.942 | 0.90x / 1.7x | 10,000 float32 + 11.5 float64 | 1.0x |
| 100x100 | 1,000,000 | v1.8.0 | `numba` | 1.66 | 0.50x / 1.0x | 1.65 | 0.51x / 1.0x | 10,000 | 1.0x |
| 51x36 | 1,000,000 | v1.9.0 | `patch` | 0.121 | 1.3x / 2.6x | 0.136 | 1.3x / 2.4x | 85.1 + 117 centroids | 22x |
| 51x36 | 1,000,000 | v1.9.0 | `screen` | 0.160 | 1.0x / 2.0x | 0.170 | 1.0x / 1.9x | 1,836 float32 + 1.28 float64 | 1.0x |
| 51x36 | 1,000,000 | v1.9.0 | `numba` | 0.321 | 0.50x / 1.0x | 0.321 | 0.53x / 1.0x | 1,836 | 1.0x |
| 51x36 | 1,000,000 | v1.8.0 | `screen` | 0.160 | 1.0x / 2.0x | 0.171 | 1.0x / 1.9x | 1,836 float32 + 1.28 float64 | 1.0x |
| 51x36 | 1,000,000 | v1.8.0 | `numba` | 0.310 | 0.52x / 1.0x | 0.358 | 0.48x / 0.90x | 1,836 | 1.0x |
| plasma 64x37 | 1,000,000 | v1.9.0 | `patch` | 0.119 | 1.5x / 3.5x | 0.142 | 1.4x / 2.9x | 78.0 + 160 centroids | 30x |
| plasma 64x37 | 1,000,000 | v1.9.0 | `screen` | 0.177 | 1.0x / 2.4x | 0.204 | 1.0x / 2.0x | 2,368 float32 + 1.44 float64 | 1.0x |
| plasma 64x37 | 1,000,000 | v1.9.0 | `numba` | 0.418 | 0.42x / 1.0x | 0.405 | 0.50x / 1.0x | 2,368 | 1.0x |
| plasma 64x37 | 1,000,000 | v1.8.0 | `screen` | 0.182 | 0.97x / 2.3x | 0.194 | 1.0x / 2.1x | 2,368 float32 + 1.44 float64 | 1.0x |
| plasma 64x37 | 1,000,000 | v1.8.0 | `numba` | 0.388 | 0.46x / 1.1x | 0.400 | 0.51x / 1.0x | 2,368 | 1.0x |
| 29x20 | 100,000 | v1.9.0 | `patch` | 0.0145 | 0.48x / 1.1x | 0.0127 | 0.72x / 1.4x | 75.5 + 40 centroids | 7.7x |
| 29x20 | 100,000 | v1.9.0 | `screen` | 0.00691 | 1.0x / 2.3x | 0.00916 | 1.0x / 1.9x | 580 float32 + 1.06 float64 | 1.0x |
| 29x20 | 100,000 | v1.9.0 | `numba` | 0.0158 | 0.44x / 1.0x | 0.0173 | 0.53x / 1.0x | 580 | 1.0x |
| 29x20 | 100,000 | v1.8.0 | `screen` | 0.00703 | 0.98x / 2.3x | 0.00916 | 1.0x / 1.9x | 580 float32 + 1.06 float64 | 1.0x |
| 29x20 | 100,000 | v1.8.0 | `numba` | 0.0136 | 0.51x / 1.2x | 0.0152 | 0.60x / 1.1x | 580 | 1.0x |
| 51x36 untrained | 100,000 | v1.9.0 | `patch` | 0.146 | 0.13x / 0.22x | 0.137 | 0.15x / 0.25x | 1,835.7 + 117 centroids | 1.0x |
| 51x36 untrained | 100,000 | v1.9.0 | `screen` | 0.0194 | 1.0x / 1.7x | 0.0207 | 1.0x / 1.7x | 1,836 float32 + 1.00 float64 | 1.0x |
| 51x36 untrained | 100,000 | v1.9.0 | `numba` | 0.0326 | 0.59x / 1.0x | 0.0347 | 0.60x / 1.0x | 1,836 | 1.0x |
| 51x36 untrained | 100,000 | v1.8.0 | `screen` | 0.0188 | 1.0x / 1.7x | 0.0211 | 0.98x / 1.6x | 1,836 float32 + 1.00 float64 | 1.0x |
| 51x36 untrained | 100,000 | v1.8.0 | `numba` | 0.0377 | 0.51x / 0.86x | 0.0470 | 0.44x / 0.74x | 1,836 | 1.0x |

## CPU streaming on 100x100

To measure out-of-core scaling in `v1.10.0`, `bench_1.10.0.py` runs `train_batch_stream` and `predict_stream` on a constant `100x100` lattice (`10,000` nodes) across `100,000`, `1,000,000`, and `10,000,000` scaled float32 5-feature binary black hole rows read from disk via `NpyReader` (`stream_100x100.json`). Because ambient desktop load stayed above `2.0`, these streaming workers ran with the 1-minute idle limit relaxed from `2.0` to `4.0` after a 30 s wait (`load_before` `2.25` to `3.89`); treat these rows as descriptive scaling measurements rather than controlled timing comparisons under the `< 2.0` idle protocol. Training uses the library defaults (`epochs=60`, `phases=(20, 10, 30)`, `r_min=8`, `bmu="auto"`, `final_pass="full"`, `chunk=262,144`, `seed=42`); `bmu="auto"` chose `patch` on 59 or 60 of 60 epochs across all three sizes (`60/60` at `10,000,000` rows), and the `10,000,000`-row training entry (`219 s`) is a single run rather than a multi-run median. Inference labels from the 10-cluster `100x100` map trained on `1,000,000` rows and writes both `uint8` cluster labels and `uint16` `out_node` indices to disk via `NpyWriter` with running CRC-32; `patch` and `screen` produced matching label and node CRC-32 checksums at every row count. Each row ran in its own fresh subprocess with a pre-warmed `.nbc` cache; peak RSS is `peak_rss_bytes()` (`1 GB = 10^9 bytes`, `1 MiB = 2^20 bytes`).

| Task | Rows (`N`) | Repeats | Wall time (s) | Throughput (Mrows/s) | Peak RSS (GB) | Peak RSS (MiB) | 1-min `load_before` |
|---|---|---|---|---|---|---|---|
| `train_batch_stream` (60 ep, `auto`) | 100,000 | 3 | 5.67 | n/a | 0.267 | 254 | 2.25 |
| `train_batch_stream` (60 ep, `auto`) | 1,000,000 | 3 | 25.3 | n/a | 0.307 | 292 | 3.89 |
| `train_batch_stream` (60 ep, `auto`) | 10,000,000 | 1 | 219 (single run) | n/a | 0.308 | 294 | 3.80 |
| `predict_stream` (`patch`, disk `out` + `out_node`) | 100,000 | 5 | 0.0379 | 2.64 | 0.221 | 210 | 3.85 |
| `predict_stream` (`screen`, disk `out` + `out_node`) | 100,000 | 5 | 0.0771 | 1.30 | 0.220 | 209 | 2.87 |
| `predict_stream` (`patch`, disk `out` + `out_node`) | 1,000,000 | 5 | 0.291 | 3.44 | 0.257 | 245 | 3.68 |
| `predict_stream` (`screen`, disk `out` + `out_node`) | 1,000,000 | 5 | 0.797 | 1.26 | 0.256 | 244 | 3.84 |
| `predict_stream` (`patch`, disk `out` + `out_node`) | 10,000,000 | 5 | 3.24 | 3.09 | 0.293 | 279 | 3.83 |
| `predict_stream` (`screen`, disk `out` + `out_node`) | 10,000,000 | 5 | 9.92 | 1.01 | 0.293 | 279 | 3.68 |

Multi-run rows report the median wall time (`3` repeats for `100,000` and `1,000,000`-row training; `5` repeats for `predict_stream`), while `10,000,000`-row training (`219 s`) is a single run. Peak memory for `train_batch_stream` rises from `0.267 GB` (`254 MiB`) at `100,000` rows (smaller than one 262,144-row chunk) to `0.307 GB` (`292 MiB`) at `1,000,000` rows, and stays near-flat from `1,000,000` to `10,000,000` rows (`0.307 GB` to `0.308 GB`, `292` to `294 MiB`). `predict_stream` peak RSS rises from `0.220 GB` (`209 MiB`) at `100,000` rows to `0.293 GB` (`279 MiB`) at `10,000,000` rows for both backends. On `100x100`, `predict_stream` with `patch` reaches `3.09` to `3.44 Mrows/s` at `1,000,000` and `10,000,000` rows versus `1.26` and `1.01 Mrows/s` for `screen`, with matching label and node CRC-32 checksums at every size.

## What changed where

- v1.1.3: cluster assignment became 12 to 13 times faster (2.35 s to 0.18 s on the plasma lattice). `Lattice.merge_similar_centroids` merges in a loop and computes each pairwise cost once instead of calling itself once per merge.
- v1.3.0: labelling read 18 percent slower for 51x36 and 8 percent slower for plasma. The package code is unchanged apart from the version number, and one of the two runs was slow (5.39 s and 4.56 s), so this is run-to-run noise.
- v1.4.0: labelling became about 10 times faster (5.94 s to 0.54 s on the plasma lattice). `Lattice.predict` and `Lattice.best_match` use the new float32 screen backend with float64 checks, and `map_data_to_lattice` uses it too.
- v1.5.0: 51x36 labelling read 20 percent slower, plasma labelling did not change. The labelling code is the same as in v1.4.0; the 51x36 lattice differs because training is now seeded, and the difference is within the run-to-run spread.
- v1.6.0: training became 9.0 times faster with the default `history=True` and 12 times faster with `history=False` (51x36, 106 s to 11.8 s and 9.17 s). `Lattice.fast_som` runs the online training steps in compiled numba kernels, and `history=False` skips the snapshots. Compilation takes about 1.2 s longer than in v1.5.0, and peak memory rose from 0.38 GB to 0.43 GB.
- v1.7.0: 51x36 `predict` and the deprecated route read 17 to 19 percent faster than in v1.6.0 (returning to their v1.4.0 and v1.5.0 levels), while training and plasma `predict` changed by under 6 percent; no training or `Lattice` labelling code changed, so the sub-second shift is run-to-run noise.
- v1.8.0: training and labelling times match `v1.7.0` within run-to-run noise at `1.0x`, outputs in threshold mode are bit for bit identical to `v1.7.0`, and target-count merging adds no overhead relative to threshold merging on a `100x100` lattice at `2.86 s` versus `2.85 s`.
- v1.9.0: `best_match_nodes_patch`, `aweSOM.predict(..., backend="patch", lattice_shape=...)`, and `SOMModel.predict(..., backend="patch")` prune `side x side` lattice patches in `float64` and return labels and distances bit for bit equal to `numba`. In 5-run idle medians on the `100x100` lattice with `46x` node pruning, `aweSOM.predict` with `backend="patch"` labels `100,000` rows in `0.0355 s` versus `0.0997 s` for `screen` and `0.182 s` for `numba`, `2.8x` and `5.1x` faster, and labels `1,000,000` rows in `0.298 s` versus `0.827 s` for `screen` and `1.69 s` for `numba`, `2.8x` and `5.7x` faster. At `1,000,000` rows on medium trained maps, `patch` takes `0.121 s` versus `0.160 s` for `screen` on `51x36`, `1.3x` faster, and `0.119 s` versus `0.177 s` on `plasma` `64x37`, `1.5x` faster; in the cross-release `plasma` run, `aweSOM.predict` and `SOMModel.predict` with `patch` take `0.118 s` and `0.133 s`, `1.4x` and `1.3x` faster than the `v1.8.0 (idle)` default at `0.168 s` and `44x` and `39x` faster than `v1.1.0 (idle)` at `5.17 s`. On the small `29x20` map at `100,000` rows, `screen` takes `0.00691 s` versus `0.0145 s` for `patch`, `2.1x` faster, and on untrained random `N(0, 1)` weights on `51x36` at `100,000` rows, overlapping patch balls prevent pruning so `patch` takes `0.146 s` versus `0.0194 s` for `screen`, `7.5x` slower while remaining exact. Training and `screen` timings match `v1.8.0` within noise.
- v1.10.0: online training (`11.2 s` default and `8.73 s` with `history=False` on `51x36`), cluster assignment (`0.173 s` plasma, `0.110 s` `51x36`), and all existing `Lattice`, `aweSOM.predict`, and `SOMModel.predict` routes match `v1.9.0` outputs bit for bit and stay within noise (`0.99x` to `1.0x`) under the `< 2.0` idle protocol. New `predict_stream` matches in-memory labelling speed on `1,000,000` points (`0.150 s` / `0.168 s` on `screen` and `0.118 s` / `0.123 s` on `patch` for `51x36` / `plasma`, `31x` and `42x` faster than `v1.1.0 (idle)`). In descriptive `100x100` out-of-core scaling runs with the idle limit relaxed from `2.0` to `4.0` (`load_before` `2.25` to `3.89`), `train_batch_stream` (60 epochs, `bmu="auto"`, `NpyReader`) took `5.67 s` (3-run median, `0.267 GB` RSS) at `100,000` rows, `25.3 s` (3-run median, `0.307 GB`) at `1,000,000` rows, and `219 s` (single run, `0.308 GB`) at `10,000,000` rows, staying near-flat in peak memory from `1,000,000` to `10,000,000` rows; `predict_stream` writing `uint8` labels and `uint16` `out_node` via `NpyWriter` labelled `10,000,000` rows in `3.24 s` (`patch`, `3.09 Mrows/s`) versus `9.92 s` (`screen`, `1.01 Mrows/s`) at `0.293 GB` peak RSS with matching CRC-32 checksums.

## Reproducing

The benchmark harness and every release's raw results live in the release results directory on the maintainer's machine (`releases/benchmarks`, with the scripts in `harness/` and one JSON file per release in `results/`). To add a release, create a Python 3.12 virtual environment, install the release tag into it (`pip install git+https://github.com/tvh0021/aweSOM.git@vX.Y.Z`), run `bench_release.py --version X.Y.Z --venv-python <venv>/bin/python --out results/X.Y.Z.json` with that environment's Python, then run `make_table.py`, which rewrites `results/summary.md` from all the JSON files; copy the new rows into this file. Run the measurements on the same machine on mains power, one at a time, and rerun the older releases too if the machine, operating system or NumPy and numba versions change.
