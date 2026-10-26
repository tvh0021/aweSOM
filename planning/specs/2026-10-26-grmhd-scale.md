# Spec: aweSOM at GRMHD scale — frozen model artifact, exact inference, parallel batch training

Label: `ready-for-agent`
Branch: `research/grmhd-scale` (checkout `/Users/tvh0021/git_repos/aweSOM-research`)
Date: 2026-10-26
Sources: `planning/notes/2026-10-26-source-ledger.md`, `planning/notes/2026-10-26-self-grill.md`,
`planning/optimization-plan.md`, `/Users/tvh0021/git_repos/asterx/audit/ideas.md` §4.

## Problem Statement

A researcher trains a self-organizing map (SOM) once on a sample of plasma
simulation data and then wants to classify every cell of every later
snapshot, including inside a running AsterX (CarpetX, C++, CPU/CUDA/ROCm)
simulation, so that current sheets can be tracked through time. Today the
only trained-model artifact is a Python pickle of the whole `Lattice`
object, each snapshot is re-normalized with its own statistics (so labels
drift between snapshots), inference allocates per point and runs at
0.1-0.2 M points/s, and training is a serial loop. Nothing a C++ code can
consume exists.

## Solution

aweSOM gains (1) a frozen, language-neutral model artifact with the scaler,
lattice, node-to-cluster map, uncertainty thresholds and provenance; (2) an
exact, chunked, allocation-free inference function in Python and a
header-only C++ reference implementation that reproduces it bitwise; (3) a
batch-SOM trainer in JAX that shards data across all visible GPUs (or CPU
devices) with the numba online trainer kept as the reference; (4) a
behaviour-preserving speed-up of the online trainer and of centroid
merging; (5) a design document for the `AsterCurrentSheets` thorn that
consumes the artifact.

## User Stories

1. As a plasma researcher, I want to save a trained SOM as a small file without training data, so that I can archive and share the model.
2. As a plasma researcher, I want the scaler fitted on the training sample stored in the model, so that every later snapshot is scaled identically and labels are comparable across time.
3. As a plasma researcher, I want `predict(features)` to return cluster ids directly, so that I do not have to call three methods and hold two full-size intermediate arrays.
4. As a plasma researcher, I want `predict` to accept float32 input and process it in chunks, so that a 10^9-point snapshot fits in memory.
5. As a plasma researcher, I want the BMU distance returned alongside the label, so that I can flag cells unlike anything seen in training.
6. As a plasma researcher, I want the model file to carry per-node and global distance thresholds, so that the runtime detector can emit a confidence flag without re-deriving statistics.
7. As a plasma researcher, I want the model file to carry provenance (aweSOM version, git commit, dataset description, creation time, merge threshold, seed), so that a published figure can be traced to the exact model.
8. As an AsterX developer, I want a plain-text twin of the HDF5 model, so that a thorn without HDF5 can still load the lattice.
9. As an AsterX developer, I want a header-only C++ classifier that reads the artifact and reproduces the Python labels exactly, so that offline/runtime equivalence is testable.
10. As an AsterX developer, I want a design document for `AsterCurrentSheets` that follows AsterAnalysis conventions, so that the thorn can be written without re-indexing AsterX.
11. As a researcher with a multi-GPU node, I want to train a SOM on the full sample using all GPUs, so that training a large lattice on 10^8 points takes minutes.
12. As a researcher on a laptop, I want the batch trainer to run on CPU with no code change, so that I can develop and test locally.
13. As a researcher, I want the batch trainer validated against the online trainer with quantization and topographic error, so that I know what "equivalent" means.
14. As a researcher running SCE ensembles, I want the online trainer to run faster without changing its result, so that 30-100 realizations finish sooner.
15. As a researcher, I want U-matrix history snapshots to be optional, so that long training runs do not spend 30% of their time on diagnostics.
16. As a researcher, I want centroid merging to never hit the recursion limit on large lattices while giving identical merges, so that a 128x74 lattice can be clustered.
17. As a researcher, I want a given seed to reproduce a training run exactly, so that results can be regenerated.
18. As a maintainer, I want `import aweSOM` to work without JAX, so that the CPU-only install stays light.
19. As a maintainer, I want every new behaviour covered by a test that runs in seconds on CPU, so that the suite stays usable.
20. As a maintainer, I want the existing public API and on-disk formats untouched, so that notebooks and SCE scripts keep working.
21. As a cluster user, I want a SLURM example for multi-GPU batch training, so that I can submit without reading JAX docs.
22. As a reader, I want a benchmark script and measured numbers for inference and training on real-sized inputs, so that claims are checkable.

## Implementation Decisions

- **Model artifact (`aweSOM.model.SOMModel`)**: a dataclass with `xdim`, `ydim`, `weights` (nodes x F float64), `node_cluster` (nodes int32, -1 allowed for unassigned), `feature_names`, `scaler_kind` ("none" | "manual" | "minmax"), `scaler_offset`, `scaler_scale` (F each, scaled = (x - offset)/scale), `node_dist_q99` (nodes), `reject_distance` (scalar), `metadata` (dict of strings). `from_lattice(lattice, scaler=None, merge_cost=...)` builds it from a trained `Lattice`; `save(path)` writes HDF5 (`format_version` attribute = 1) and `save_text(path)` writes a self-describing plain-text file (header lines `# key value`, then the weight rows); `load` handles both by extension. Node index convention is the existing `rowix = x + y * xdim`.
- **Scaler (`aweSOM.model.Scaler`)**: `fit(data)` for "manual" (offset = mean, scale = 2*std*bulk_range, matching `manual_scaling`) and "minmax"; `transform`, `inverse_transform`. `manual_scaling` and `inv_manual_scaling` stay as they are.
- **Inference (`aweSOM.inference`)**: `best_match_nodes(weights, data, chunk)` numba kernel: no per-point allocation, float64 accumulation in the sequential feature order, strict `<` scan ascending over node index, returns int32 node ids and float64 squared distances. `predict(model_or_lattice, data, chunk=2**20, return_distance=False, dtype of labels int32)`. `Lattice.best_match` keeps its signature and output shape but delegates to the new kernel (same results, no print in the loop). `Lattice.predict` convenience wrapper.
- **Exactness contract**: reference path = float64 direct differences, sequential sum, strict `<`. Python `predict`, `Lattice.best_match` and the C++ reference agree bitwise on labels for float64 input. Float32 input is allowed and documented as approximate (ties may flip).
- **C++ reference (`cpp/awesom_classify.hpp`, `cpp/test_classify.cpp`, `cpp/CMakeLists.txt`)**: header-only; `load_text(path)` and, if `AWESOM_HAVE_HDF5`, `load_hdf5(path)`; `classify(const double* x, int& node, double& dist2)` and a scaled variant; no dependencies beyond the standard library. A pytest builds it with the system compiler if available (skips otherwise) and compares labels on a fixture written by Python.
- **Batch trainer (`aweSOM.batch`)**: `train_batch(data, xdim, ydim, epochs=40, radius_schedule=None, chunk=65536, seed=42, init="sampling", mesh=None, precision="highest", dtype=float32)` returns `(weights, history)`; factorised update `num = H^T S`, `den = H^T n` with `S, n` from `segment_sum` per device and `psum` across the mesh; `jax.shard_map` + `jax.make_mesh`; degenerate nodes keep their weight; radius schedule = r_max for 50% of epochs, linear to r_min over 25%, r_min for 25%; neighbourhood identical to `Lattice.Gamma` with alpha = 1. `Lattice.train_lattice_batch(...)` wraps it and fills `self.lattice`, `self.umat`, `self.epoch`. JAX is imported lazily; a clear `ImportError` otherwise. Multi-host initialization is left to the caller.
- **Batch acceptance**: `tests/batch_test.py` runs on 4 virtual CPU devices (`jax_num_cpu_devices` set before import in a subprocess or via `XLA_FLAGS`), checks: shape/finite, sharded result equals single-device result to 1e-5, QE and TE (definitions in the ledger) against the online trainer on a synthetic 3-cluster set over 5 paired seeds with the Sol bounds (QE ratio UCB <= 1.03, TE diff UCB <= 0.01) marked `slow`.
- **Online trainer speed-up (`Lattice.fast_som`)**: new `history` argument on `train_lattice` (default True keeps today's snapshots); the per-step BMU search and neighbourhood update move into one numba kernel that only touches nodes within the radius; arithmetic order preserved so a seeded run gives a bitwise-identical lattice to the current loop (test compares against a frozen copy of the old loop kept in the test file).
- **Merge (`Lattice.merge_similar_centroids`)**: replaced by the equivalent non-recursive loop with an ordered-pair cost cache (algorithm in the ledger §E); identical results tested against the old recursive implementation kept in the test file on 20 random maps.
- **Thorn design (`docs/source/astercurrentsheets.md`)**: parameter list (`model_file`, `cadence`, `features`), schedule (`AT analysis`, after `AsterAnalysis_MHD`), grid functions (`sheet_label` int as CCTK_REAL, `sheet_dist`), table upload via `The_Managed_Arena` + `Gpu::copy`, kernel = the C++ reference per cell, feature evaluation from `HydroBaseX::Bvec` with a cell-centred curl-B proxy; explicitly out of scope for this repo to implement.
- **No change** to SCE, to `save_som_object`, `save_cluster_labels`, label file names, pickle attributes, or `Lattice` constructor signature.

## Testing Decisions

- Tests check external behaviour: labels, distances, file round-trips, shapes, numerical agreement; never internal call order.
- Prior art: `tests/som_test.py` fixtures (`map`, `trained_map`), `tests/regression_test.py` (small seeded runs under `redirect_stdout`), `tests/sce_test.py` (`tmp_path`).
- New files: `tests/model_test.py`, `tests/inference_test.py`, `tests/batch_test.py`, `tests/merge_test.py`, `tests/cpp_test.py`; each runs in < 30 s on CPU; GPU-only paths are skipped not failed.
- Equivalence fixtures: a small trained lattice saved under `tests/fixtures/` (HDF5 + text) with 1000 float64 points and their reference labels.

## Out of Scope

AsterX thorn source code; KD-tree or early-exit BMU search; connected components and temporal tracking; changes to the SCE algorithm; README rewrite; packaging changes beyond an optional `[gpu]` extra; multi-host launch orchestration.

## Further Notes

Decisions were taken autonomously (user asleep); each is flagged in the final report. The thorn name `AsterCurrentSheets` and the artifact contents come from the user's own audit note in the asterx repo.
