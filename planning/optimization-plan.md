# SOM Optimization Plan: Inference at GRMHD Scale

- **Branch:** `dev` (from `public-release` @ `305c752`)
- **Created:** 2026-10-12
- **Status:** Proposed. Open questions below need answers before phases 3–4.

## Goal

Find current sheets in, and track them through, very high-resolution GRMHD simulation output, ideally in real time. Each snapshot holds about 10¹⁰–10¹² points if everything is interpolated to the finest refinement level.

- **Training** happens once, on a sample. Speeding it up is a bonus.
- **Inference** runs on every snapshot. This is the priority.

## Baseline measurements

**Setup:** Apple M1 Max with 10 threads, numpy 2.5.4, numba 0.68.0. The lattice is 64×37 = 2,368 nodes, matching the fiducial plasma run. The data is 2,097,152 random Gaussian points, and the lattice is random rather than trained.

To reproduce, run `python planning/benchmarks/bench_inference.py [N] [F]`.

### Inference (finding each point's best-matching node)

| Method | F = 4 | F = 10 | Labels match current? |
|---|---|---|---|
| Current `Lattice.best_match` (f64) | 0.19 M pts/s | 0.09 M pts/s | — |
| Compiled loop, no per-point memory allocation (f64) | 2.66 M pts/s (14×) | 1.33 M pts/s (15×) | 100% |
| Compiled loop, no per-point memory allocation (f32) | 2.07 M pts/s | 1.76 M pts/s (20×) | 100% |
| Compiled loop, nodes in the inner loop (f64) | 2.98 M pts/s (16×) | 1.30 M pts/s | 100% |
| `scipy.spatial.cKDTree`, all threads | 8.11 M pts/s (43×) | 0.66 M pts/s (7×) | 100% |
| Chunked matrix multiply (f32) | 0.54 M pts/s | 0.54 M pts/s | 99.9997% (ties) |

- **Why the current code is slow:** `best_match` (`src/aweSOM/som.py:456`) creates a new `nodes × F` temporary array for every point inside the parallel loop. It also writes its output as `(N, 1)` float64 and prints progress from inside the compiled loop.
- **The KD-tree depends on the number of features.** It is fastest at low F and drops below the plain loop by F = 10. These numbers come from random data, so they need re-checking on real trained lattices and features.

**Projected time for one 10¹⁰-point snapshot on this laptop:**

| Method | Time |
|---|---|
| Current | ~15 h |
| Compiled loop | ~1 h |
| KD-tree, F = 4 | ~20 min |

**GPU estimate, not measured:** checking every point against every node at 10¹⁰ points is about 3×10¹⁴ operations. That is roughly 15–30 s on one A100 in FP32. 10¹² points needs many GPUs and/or a search that skips most nodes.

### Training

Reproduce with `planning/benchmarks/bench_train.py` and `planning/benchmarks/bench_train_njit.py`.

| Version | Time per step |
|---|---|
| Current `fast_som` | 186 µs |
| Compiled prototype, neighbourhood covers the whole lattice (nsize = 65) | 16.8 µs (11×) |
| Compiled prototype, small neighbourhood (nsize = 8) | 6.4 µs (29×) |

Where the current time goes:
- About 30% is spent computing the U-matrix (the map of distances between neighbouring nodes) 202 times. That's only needed for diagnostics.
- Most of the rest is Python-level numpy work on every step, such as `np.outer` and temporary arrays.

## Problems at this scale beyond kernel speed

1. **Each snapshot is renormalized with its own statistics.** `manual_scaling` (`src/aweSOM/run_som.py:67`) uses the mean and standard deviation of whatever data it receives. Labels then shift between snapshots, which breaks tracking. The scaling from training must be frozen and saved with the model.
2. **Memory.** The current API keeps the whole dataset in `som.data_array`. It also builds two full-size intermediate arrays: node indices as `(N, 1)` float64, then coordinates as `(N, 2)` int32. 10¹⁰ points × 4 features in float32 is already 160 GB. Inference must read data in chunks and write labels directly, as uint8 or uint16.
3. **Interpolating to the finest level.** Inference treats each point on its own, so it can run on each refinement level's native cells. That usually means far fewer points. However, features built from derivatives or convolutions depend on resolution, and this physics question needs checking.
4. **Feature extraction cost.** The convolutions in `examples/plasma-turbulence/obtain_features.py` may cost more per snapshot than the SOM itself. This is out of scope for now but should be profiled.
5. **Model persistence.** Inference needs a small model file: lattice weights, the node-to-cluster table, scaler parameters and feature names. It must not include training data or history.

## Plan

### Phase 1: CPU inference (does not depend on the open questions)

- [ ] Add one inference function that goes from features to labels: `predict(data, chunk_size=...)`.
  - Combines finding the best node with looking up its cluster. Returns uint8/uint16 cluster ids, with no intermediate arrays.
  - Supports float32 and processes input in chunks, so it works on memory-mapped or streamed HDF5 data.
- [ ] Fix `Lattice.best_match` so it stops allocating per point, without changing its signature or output, for backward compatibility. Remove the print inside the loop.
- [ ] Freeze the scaling: record the training mean and standard deviation, then reuse them at inference.
- [ ] Add a small model save/load format, separate from `save_som_object`.
- [ ] Add tests that new labels match current `best_match` + `assign_cluster_to_data` exactly.
- [ ] Add a benchmark in `tests/performance/` using a real trained lattice and real features.

### Phase 2: Training

- [ ] Compile the whole training loop with numba. Update only the nodes inside the current neighbourhood, keeping the same maths as `fast_som` and `Gamma`.
- [ ] Make the U-matrix and lattice history snapshots optional, off by default or at a configurable frequency.
- [ ] Optionally, add a batch-SOM mode for parallel training on large samples.
- [ ] Check that trained lattices are statistically equivalent to the current trainer's.

### Phase 3: GPU and scale-out

- [ ] Add a GPU inference path. JAX is already used by SCE; alternatives are CuPy and numba-cuda. It should process data in tiles and stream labels out.
- [ ] Split the work across GPUs and nodes (MPI, or one process per data block). Inference treats each point independently, so this is simple.
- [ ] Optionally, run inside the simulation while it runs, or directly on its native AMR blocks.

### Phase 4: Faster search (if still needed)

- [ ] Use a KD-tree or ball tree when the feature count is low.
- [ ] Start each point's search from its neighbouring cell's (or the previous snapshot's) best node. Search outward over the lattice and confirm the result is exact.
- [ ] Downstream tracking: connected-component labelling with `scipy.ndimage.label` or `cc3d`, then match regions across snapshots. `origin/active-dev` has a pure-Python version in `combine_clumps.py` that will not scale.

## Open questions

1. How many features and roughly what lattice size will the new project use? This decides between the KD-tree and brute force.
2. What hardware: NVIDIA or AMD GPUs, and MPI? Which GRMHD code and AMR output format?
3. What does "real-time" mean in practice: a target time per snapshot? Is running inference inside the simulation an option?
4. Must labels match the current method exactly, or are small differences at cluster boundaries acceptable for speed?

## Change log

| Date | Change |
|---|---|
| 2026-10-12 | Plan created, with baseline benchmarks and benchmark scripts in `planning/benchmarks/` |
