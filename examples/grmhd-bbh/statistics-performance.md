# Production statistics memory and I/O

`export_statistics` reads float32 raw features once in sequential row blocks.
It builds stable counting-sort permutations for nodes and cluster labels once
apiece. It gathers each feature into a reusable float64 buffer in RAM and
computes the same NumPy percentiles and medians as before. Cluster dictionary
insertion order and per-node partitioning before the rejection percentile are
preserved. The function creates no scratch files and does not modify its inputs.
`PhaseClock`, the worker supervisor and all training settings are unchanged.

## Memory

For N rows, D features, input element size S and row block size B, the feature
stage holds N*D*S bytes of raw features, 8*N bytes of permutation and 8*N bytes
of float64 statistics buffer. The temporary row read needs at most B*D*S bytes;
feature gathering needs at most B*S bytes. These temporaries do not overlap.
The node stage releases its permutation and distance buffer before loading raw.
Counting a group uses a temporary int64 cast of the int32 keys in NumPy's
`bincount`; the cast ends before permutation allocation. Group arrays cost
O(number of nodes or clusters). No comparison-sort workspace is needed.

At N=20M, D=7, S=4, the main live heap arrays total 880 MB. Existing labels,
nodes and distances can add 320 MB of resident file mappings, and the default
B=2**21 row read adds 59 MB. Allow about 2 GB per seed for this stage, including
allocator retention, NumPy temporaries and interpreter overhead. This is a
planning allowance, not a whole-job RSS cap; training and the OS cache are extra.
Two statistics stages fit comfortably within the 180 GB production allocation.

At 1e9 rows the same arrays need 44 GB, plus up to 16 GB of resident input
mappings and block temporaries per seed. The 2 GB allowance scales to roughly
100 GB per seed. Two concurrent seeds would require a different memory budget
or exact partitioning on node-local storage. This implementation is linear in
row count and feature count; it is not an out-of-core solution for that target.

## Reproduce the comparison

Run `benchmark_statistics.py DIRECTORY --rows 3000000` with the repository's
`src` on `PYTHONPATH`. Put DIRECTORY on the storage being measured. It creates
a contiguous, uncompressed float32 `pool.h5`, production-style mapped labels,
nodes and distances, and compares the frozen old implementation from tests
against the current driver. An intermediate variant changes only scratch
placement to RAM. The run order is old, RAM scratch, new, new, RAM scratch, old. It excludes Numba's first
compilation, checks output bytes and writes `timings.json`. It records wall and
process CPU times, raw HDF5 read time, grouping time, percentile time and median
time. `--uncached` applies macOS `F_NOCACHE` to the HDF5 descriptor; it does not
evict every existing cached page or emulate a remote network filesystem.

## Local measurements on 2026-10-29

The existing local production pool has `raw.shape == (20000000, 7)`, dtype
float32, `chunks=None` and no compression. The suspected row-chunking is absent.
The old code instead selects each column separately from contiguous row-major
storage and writes a mapped float64 statistics file eight times in the output
directory. The feature scans span the raw matrix seven times. Scratch payload
writes total at least 64*N bytes before in-place percentile and median writes.
For 20M rows that is 1.28 GB of scratch payload, with scattered writes grouped
by node or cluster. The output directory on the production cluster is lagoon.

An isolated 3M-row benchmark on `/Volumes/ExternalNVMe`, with 131072-row blocks
and `NUMBA_NUM_THREADS=2`, produced these means of two runs per variant:

| Implementation | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| Old, mapped scratch | 1.1226 | 0.8122 |
| Old, only scratch changed to RAM | 0.7087 | 0.7057 |
| New, row reads and RAM statistics | 0.5982 | 0.5980 |

The new implementation is 1.88 times faster locally. Moving only scratch to RAM
removes 0.414 seconds of wall time. HDF5 reads take 0.197 seconds with column
selection and 0.00696 seconds with row blocks, a 28.2-fold reduction. Percentile
and median CPU times remain similar, so arithmetic is not the local bottleneck.
The RAM-scratch ablation and raw-read timers overlap in the table. Their savings
must not be added to predict the total speedup.

Every result in the comparison is bitwise identical. Numba compilation is
excluded for both implementations. The benchmark runs under `nice -n 10` with
the research environment's Python. OS caches are uncontrolled. Logs and JSON
are stored under
`/Volumes/ExternalNVMe/awesom-results/2026-10-29/fast-statistics/benchmark-release`.
With `--uncached`, old and new means are 1.131 and 0.595 seconds wall,
and 0.820 and 0.595 seconds CPU. The RAM-scratch variant takes 0.695 seconds
wall and 0.695 seconds CPU. Its JSON is in the sibling
`benchmark-release-uncached` directory.

This local drive does not reproduce lagoon's 2020–2369 second stalls. The
measurements identify both avoidable I/O paths, but cannot establish their
relative contributions on the remote filesystem without cluster access.
