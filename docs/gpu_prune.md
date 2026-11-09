# Exact CUDA box pruning

`aweSOM._gpu_prune.best_match_nodes_cuda_prune(weights, data,
lattice_shape=None, hint=None, chunk=2**20)` returns int32 node IDs and float64
squared distances. `inference.predict(backend="cuda_prune", hint=...)` and
`stream.predict_stream` expose the same search. Hints must be valid int32 node
IDs, one per input row. They change the work, never the answer.

Reuse `PruneGpuSearcher` for a fixed lattice across successive time steps.
The convenience function constructs a searcher each call. The streamed API
constructs one per pass. No new dependencies are required beyond optional CuPy.

## Design

Partition the lattice into 8x8 tiles. Without a lattice shape, use contiguous
64-node runs. Each block stores exact float64 coordinate minima and maxima.
These extrema select existing weights and introduce no rounding error. Packed
node IDs retain the original indexing convention, `x + y * xdim`.

With a hint, evaluate that node first. Otherwise, evaluate all box bounds and
scan the block with the smallest bound first. Then visit other blocks in
ascending block order, skipping blocks with bounds strictly greater than the
current best squared distance. This deliberately avoids a heap or sort per
thread, whose storage and divergent control flow would be expensive. The
implementation does not globally sort the blocks. A good warm start removes
the initial bound sweep altogether.

One thread handles one row. Box metadata uses shared memory when it fits in
32 KiB, otherwise global memory. The weight matrix and partition stay on the
GPU for the searcher's lifetime. Nearby input rows can take similar branches;
randomly ordered rows may diverge. The present host API stages chunks
synchronously, unlike the double-buffered brute-force backend.

When block node evaluations exceed half the lattice, abandon pruning and restart an
ascending full scan. The threshold is checked after each block, so the total
node work is at most `N + floor(N/2) + 65`, including a possible hint.
The bound work is at most three visits per block, counting seed selection.
The fallback bounds work but does not promise baseline throughput: its full
scan reads global weights and does not use the baseline's shared weight tiles.

## Exactness without a heuristic margin

All weights and finite rows are float64, with float32 inputs converted exactly.
For feature f, let `[lo, hi]` contain every weight in the block. Compute a gap g
as `RN(lo - x)` if x < lo, `RN(x - hi)` if x > hi, and +0 otherwise. RN is IEEE
round-to-nearest, ties-to-even. Subtraction is monotone, and sign reversal
preserves the magnitude of a rounded subtraction. Therefore, for every member
w, `0 <= g <= abs(RN(w - x))`. This holds through gradual underflow and overflow.

Squaring nonnegative values with RN is monotone. Accumulating nonnegative
squares with RN in the same ascending feature order is also monotone. By
induction, the computed box bound is no greater than the reference's computed
squared distance for *each* member, including when either overflows to infinity.
Thus the rounding safety margin is exactly zero: the lower bound is already a
bound on the rounded reference result, rather than an approximation to a
real-valued Euclidean norm. There is no square root, centroid rounding, relative
error estimate, or restriction on the magnitude of finite inputs. Subnormal
squares rounding to zero remain safe and are covered in the tests.

The device uses `__dsub_rn`, `__dmul_rn`, and `__dadd_rn`, with FMA contraction
disabled. Every visited node uses precisely the baseline feature order. The
CPU emulation uses Numba without fast math. Skip only when `bound > best`,
never on equality. Update on smaller distance, or equal distance with a lower
node ID. Initialize `(best, ID)` to `(inf, 0)`, as in the reference. Consequently
infinite distances never replace the initial state. This preserves the lowest
index in ties, including zero distances and duplicate nodes across blocks.

Nonfinite weights disable pruning for the whole lattice; nonfinite input rows
disable it per row. The ascending reference scan then preserves NaN-ignore
semantics and the `(0, inf)` result when no finite distance exists. Hints are
ignored on those paths. Fallback also discards the seed and restarts in reference
order.

## Local pruning measurements

With deterministic test seed 41, the 32x32 smooth seven-feature lattice checked
6.2500% of nodes per row without hints and 6.3477% with nearby previous-node
hints. A 16x16 lattice trained with the repository's streamed trainer for four
epochs checked 31.2500% without hints and 31.6406% with exact current-BMU hints.
Counts include the one initial hint evaluation and any fallback work. In these
cases both starts reach the same candidate blocks; hints save the initial
metadata sweep rather than node evaluations. These are CPU-emulated operation
counts, not measured GPU speedups.

## Validation and GH200 follow-up

Local validation on the development Mac: 47 backend tests passed and 16
CUDA tests skipped. The complete suite finished with 947 passed, 23 skipped,
2 expected failures and 2 failures in 314 seconds. Both failures are
`tests/sce_test.py::test_jax_gpu_available` and `::test_jax_cpu_available`, which
unconditionally import absent JAX through `unittest.mock.patch`. Both also fail
on unchanged base commit `6e891e9` in the supplied environment. The focused
backend rerun includes the 35 CPU cases and one CUDA case added after the full
suite began collecting. No existing tests were modified or deselected.

On a machine without CUDA:

```sh
PYTHONPATH=src python -m pytest tests/gpu_prune_test.py -q -s
PYTHONPATH=src python -m pytest -q
```

On Tigris, use the existing CUDA/CuPy environment and a reviewed allocation
under the cluster rules. No submission script is included or required by this
change. Inside that allocation:

```sh
PYTHONPATH=src python -m pytest tests/gpu_prune_test.py tests/gpu_test.py -q
PYTHONPATH=src python planning/benchmarks/bench_gpu_prune.py --rows 1000000 --side 100
```

The probe builds metadata outside timing, warms the kernels, asserts bitwise
parity, and reports synchronous end-to-end rates for full scan, unseeded prune,
nearby synthetic previous BMUs and wrong hints. Supply `--weights lattice.npy --data
scaled_rows.npy --hint previous_bmu.npy` for actual trained data. Without
`--hint`, real data uses ideal current BMUs as an upper-bound warm-start proxy. Keep `--side` consistent with the
lattice shape. The synthetic sheet is a pruning proxy, not a GRMHD workload.
The reference baseline measured elsewhere was 48.7 Mrows/s on a GH200; compare
the two backends in the same allocation before drawing a speed conclusion.

A provisional target is 1-4x end-to-end speedup on a smooth trained 100x100
lattice with good hints. This is an unvalidated engineering target, not a
prediction supported by GPU timing. Random weights, broad overlapping boxes,
far-away rows, divergent warps, or transfer overhead may erase the gain or
cause a slowdown. Node-check fraction is not a throughput multiplier.

The CUDA kernel has not been compiled or executed on the development Mac.
Hardware parity, register pressure at 32 features, shared/global metadata paths,
and throughput remain GPU validation items. The raw kernel already accepts
device pointers, but a public device-resident API, stream ownership, persistent
staging, and an AsterX C++/CUDA integration are future work.
