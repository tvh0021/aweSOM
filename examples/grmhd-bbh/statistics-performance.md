# Production statistics memory and I/O

`export_statistics` reads float32 raw features once in sequential row blocks.
It builds stable counting-sort permutations for nodes and cluster labels once
apiece. It gathers each feature into a reusable float64 buffer in RAM and
computes the same NumPy percentiles and medians as before. Cluster dictionary
insertion order and per-node partitioning before the rejection percentile are
preserved. The function creates no scratch files and does not modify its inputs.
`PhaseClock`, the worker supervisor and all training settings are unchanged.

## Why the old code was slow on lagoon

The old `export_statistics`, frozen in
`tests/production_pool_statistics_reference.py`, grouped values through a
float64 scratch file. It created that file with `open_memmap` in the output
directory, which on the production cluster is lagoon, a network filesystem. It
did this eight times: once for node distances and once for each of the seven
features. Each pass wrote 8*N bytes, 160 MB at 20M rows, and the writes were
scattered. Every row went to the slot reserved for its node or cluster, so
consecutive rows landed in different parts of the file. The percentile and
median calls then rewrote the same pages in place.

The old code also read the raw matrix one column at a time. The pool stores
rows contiguously, so reading one feature for all rows touches the whole
matrix with a stride of seven values. Seven features meant seven strided passes
over the same 560 MB.

On a local drive these patterns cost little: the old code was 1.88 times slower
on 3M rows (see below). On a network filesystem, each dirty page written back
from a scattered memmap and each strided read can become a separate small
request to the server. Many small round trips leave the process waiting. That
fits the measured stage wall time, which was 4.3 to 6.9 times its CPU time in
five of six seeds and 2.7 times in the sixth. Old CPU time was also 56 to 303
times the new CPU time in those five seeds. The likely reason is kernel time
spent on page faults and writeback, since `time.process_time` counts system
time. This is inferred, not profiled.

The new code reads the raw matrix once, in sequential row blocks, and does all
grouping, percentiles and medians in RAM. It writes nothing to lagoon until the
final outputs. On the cluster its wall time equals its CPU time.

Measured: per-seed wall and CPU time for both versions on the same pool, and
the local timings below. Inferred: that small network requests cause the gap.
Without profiling on the cluster we cannot split the old wall time between the
scratch writes and the column reads.

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

This local drive does not reproduce lagoon's stalls of 2000 seconds or more.
The local runs identify both avoidable I/O paths. The cluster runs below
measure the total effect on lagoon, but not the share of each path.

## Production cluster measurements, 2026-10-29

Jobs 8014 and 8047 ran the same production pool: 20M rows by 7 features,
float32, contiguous HDF5, k=10, r_min=8, lattices 91x64 and 48x34, seeds 42, 43
and 44. Both wrote their outputs to lagoon. Job 8014 ran the old code at
0a006be; job 8047 ran the new code at cb3db68. Times are the `wall_seconds`
and `cpu_seconds` fields of each job's `timing.log`, pulled to
`/Volumes/ExternalNVMe/awesom-results/cluster/HHR_flux_production_pool_k10_r8_<job>/`.

| Seed | Old stats wall | Old stats CPU | New stats wall | New stats CPU | Wall speedup | Training wall, old / new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 91x64-s42 | 2369.4 | 537.7 | 5.27 | 5.24 | 450 | 399 / 404 |
| 91x64-s43 | 2020.6 | 293.1 | 5.28 | 5.26 | 382 | 405 / 395 |
| 91x64-s44 | 10127.5 | 1528.9 | 5.14 | 5.13 | 1970 | 213 / 193 |
| 48x34-s42 | 2326.2 | 540.3 | 5.77 | 5.76 | 403 | 217 / 216 |
| 48x34-s43 | 10075.1 | 1551.4 | 5.14 | 5.12 | 1961 | 210 / 207 |
| 48x34-s44 | 73.1 | 27.2 | 4.86 | 4.85 | 15 | 118 / 116 |
| Total | 26992.0 | 4478.6 | 31.47 | 31.37 | 858 | 1561 / 1531 |

All times are in seconds. The new statistics stage is 15 to 1970 times faster
in wall time per seed, and 382 to 1970 times faster on five of the six seeds.
Over all six seeds the old stage took 7.5 hours and the new one 31 seconds. The
new stage takes about 5 seconds per seed, roughly 1 to 4 percent of training
time. Except for 48x34-s44, the old stage took 5 to 48 times as long as
training. The new stage's wall time is within 0.4 percent of its CPU time.

The old timings vary more than the code explains. Three seeds took 2020 to 2369
seconds, two took about 10,000 seconds, and 48x34-s44 took 73 seconds. Each
lattice runs s42 and s43 together with 16 threads each, then s44 alone with 32
threads. A plausible reading is that 48x34-s44 was fast because no other seed
in the job was writing to lagoon. That is inferred, not measured. It does not
explain 91x64-s44, which also ran alone in its slot and took 10,128 seconds.
Load from other jobs on lagoon and the state of the page cache are other
possible causes; the logs do not record either. 48x34-s43 did share its slot
with s42, but so did the three seeds near 2,000 seconds. Treat the per-seed
spread as uncontrolled. The new stage varied only from 4.86 to 5.77 seconds.

Job 8047 started at 23:11:49 UTC and wrote `VALIDATED 0` at 23:30:00 UTC, about
18 minutes in total. Job 8014 started at 17:47:41 UTC and wrote `VALIDATED 0`
at 00:24:14 UTC the next day, about 6.6 hours. The earlier job 8012 ran the same
pipeline with the old code and spent about 8 hours outside its epoch loop, which
is consistent with this stage.

## Output identity on the cluster

The `run/results/` directories of 8014 and 8047 match wherever results are
stored. All six `*-labels.npy` and all six `*-umat.npy` files are byte-for-byte
identical. In every model `.h5` and `frozen-*.h5` file, the datasets `weights`,
`node_cluster`, `node_dist_q99` and `scaler` are identical, and `feature_names`
has identical values. The only attribute that differs is
`metadata.aweSOM_commit`. In the model JSON files, every field matches,
including `cluster_medians`, `reject_distance`, `merge_cost_used` and
`sheet_rule`. The exceptions are timings, hostname, pid, `SLURM_JOB_ID` and
peak RSS. The freeze manifest differs in the same fields, plus
`aweSOM_commit` and `implementation_sha256`. Sheet cluster IDs match: [4], [7]
and [4] for the 91x64 seeds, and none for the 48x34 seeds. `merge_cost_used`
matches per seed, for example 0.147779242479924 for 91x64-s42.

## Sheet-candidate descriptors, 2026-10-29

`export_statistics` now also returns the descriptors of the consensus
sheet-candidate rule: one-vs-rest rank effects for all seven features,
per-cluster quartiles, rows and shares per training snapshot, and the joint
favourable fraction. Each feature needs one sort of N rows. A float32 column is
sorted as one uint64 array, with the order-preserving value bits above the
label. That is about 0.8 s per feature locally, against 2.3 s for an argsort.

Measured locally on the job 8014 pool (2e7 rows, all six seeds), the statistics
stage took 9.1 to 9.9 s per model, against 2.1 to 2.2 s before. The medians
stayed bitwise identical to the previous code and to the job's JSON. The extra
peak memory is about 28 bytes per row, about 0.6 GB at 2e7 rows.
