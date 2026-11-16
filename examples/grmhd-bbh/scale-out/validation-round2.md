# Round 2 validation

Starting commit was `7738ed0`, confirmed before editing. The four fixes in that commit remain intact. `src/aweSOM/stream.py` is unchanged. No GPU or MPI imports were added; nothing was pushed or submitted.

## Changes

1. Restored the scaler's float64 column memmap and `np.percentile(overwrite_input=True)`, with the private `scaler-column.npy` under the selected scratch root. This needs 8 bytes per row, without a full sort.
2. Float32 feature runs use order-preserving value bits above a uint32 label in one uint64 key, including the old signed-zero normalization. Float64 distances use stable argsort and gather. One Numba heap merges all runs through one input file descriptor and fixed buffers, directly into the percentile/rank scan. There are no intermediate merge generations. Feature jobs run in spawned single-threaded processes, capped by `min(threads_per_seed, features)`. Read-only file-backed input is reopened independently in each process; in-memory callers use the serial path. Serial and concurrent output representations agree exactly in the added regression test, including ties, subnormal values and empty groups.
3. Reused-pool fits check scratch and output free space before creating OUT or starting training. Each check honors the existing minimum-free floor. When both roots share a filesystem, each check budgets their combined demand. The regular pool-path output estimate also accounts for the larger concurrent scratch demand.
4. `--collect` without `--reuse-pool` exits through `argparse.error` before reading inputs or creating output.
5. Renamed the driver fixtures to `scale_out_driver_cpu.npz` and `scale_out_driver_cpu.json` and updated the test reference.

Output statistics keys, model metadata keys and file formats are unchanged. The new optional `workers` argument changes execution only. The small fixes already supplied by the parent were not reverted.

## Tests

```text
NUMBA_NUM_THREADS=2 .venv/bin/pytest tests/production_pool_test.py tests/production_pool_scaleout_test.py tests/stream_test.py -q
117 passed, 2 skipped in 52.47s
```

The two skips are the existing JAX comparisons because JAX is unavailable. The final combined suite ran after the merge-buffer sizing change. Shell syntax checks for the fit launchers and `git diff --check` passed.

Added coverage checks serial/concurrent statistics, scratch/output capacity failures and the free-space floor, and the collect/reuse dependency. Existing explicit-scratch fit/collect coverage passes. The first parallel test attempted to reopen a file still open for writing and encountered HDF5's file lock. The test now closes the writer and exercises the production read-only path; live-writer callers remain on the serial path.

A fresh 100,000-row main/prepare/fit/collect probe also passed with two-thread fits, exercising feature process pools nested inside the normal spawned fit workers. Its two workflows produced identical weight hashes. Full main took 14.687 s; pool preparation 1.480 s; separate fits 10.519 and 9.790 s; collection 1.369 s. See `local-probe-round2.json` in the external report directory.

## Exact 10-million-row differential probe

The probe compares all returned quantities with `reference_pool.py`, the complete implementation from `82dee49`. Inputs retain the original seed, float32 values, tie fixture and empty cluster. This round adds per-snapshot counts and shares to the comparisons and runs the actual read-only HDF5/mapped-label path with seven workers. NaN masks agree for every quantity. Every maximum absolute difference is exactly zero.

| Quantity | Max absolute difference | Max relative difference |
|---|---:|---:|
| node_q99 | 0 | 0 |
| reject | 0 | 0 |
| medians | 0 | 0 |
| cluster_effects | 0 | 0 |
| cluster_quartiles | 0 | 0 |
| pooled_medians | 0 | 0 |
| cluster_joint_favorable_fraction | 0 | 0 |
| cluster_rows/all | 0 | 0 |
| cluster_rows/1 | 0 | 0 |
| cluster_rows/2 | 0 | 0 |
| cluster_share/all | 0 | 0 |
| cluster_share/1 | 0 | 0 |
| cluster_share/2 | 0 | 0 |

## Measured statistics time and memory

| Method | Wall seconds |
|---|---:|
| In-memory reference, fresh differential run | 4.944 |
| Round 1 external structured sort and pairwise merge | 71.458 |
| Round 2 packed runs, k-way merge, seven feature workers | 8.394 |
| Round 2 isolated parallel process | 7.725 |

The isolated new-only parent peaked at 1,443,332,096 bytes RSS. The sampled peak for the entire process tree was 3,745,480,704 bytes, or 3.745 decimal GB. The largest individual child peak was 328,728,576 bytes. Process-tree sampling uses `ps` at 100 ms intervals, so it can miss shorter peaks and counts shared pages in each process. Parent peak includes input generation. These are local macOS results on a ten-logical-CPU machine; they are not GP measurements.

The previous isolated 1.264 GB figure covered a single process. Comparing it directly with the new parent-only figure would omit the seven feature workers, so use the process-tree figure for this round. Merge value/label buffers have a fixed 2**24-record ceiling, at most 268,435,456 bytes per process, plus a bounded output buffer, run-formation buffers and metadata. Metadata is capped at 2**18 runs; exceeding that limit raises an error requiring a larger chunk. With the production 2**21-row chunk there are only 477 runs at 1e9 and 4769 at 1e10. Linux mapped-page release remains in use. Billion-row peak RSS still needs a cluster measurement.

## Scratch and 32-core planning projections

A packed feature writes and reads one 8-byte key per row. Distance records are 16 bytes per row and run before the features. Peak scratch is `max(16, 8*W)*N` bytes, where W is the concurrent feature count. On a 32-core fit W=7, so peak statistics scratch is **56 bytes per row**, not 16 bytes per row. Serial or two-feature concurrency peaks at 16 bytes per row. No merged output generation is written. Scratch traffic across the distance stage and all seven features is `(2*16 + 7*2*8)*N = 144*N` bytes.

| Rows | Projected statistics wall time, 32 allocated CPUs | Peak scratch, seven workers | Fit output vectors | Scratch read + write traffic |
|---:|---:|---:|---:|---:|
| 1e9 | 27–54 minutes | 56,000,000,000 B | 16,000,000,000 B | 144,000,000,000 B |
| 1e10 | 6–12 hours | 560,000,000,000 B | 160,000,000,000 B | 1,440,000,000,000 B |

These are conditional planning estimates, not measured GP performance. The straight-line extrapolation of the 8.394 s measurement alone is 14.0 minutes and 2.33 hours. For the planning ranges above, multiply CPU work by the extra heap-depth ratios, 1.68 and 2.31, and allow per-worker throughput equal to the local machine through two times slower. This slowdown interval is an explicit assumption, not a measured Xeon correction. Then add a cold-I/O scenario of 432*N bytes at an assumed aggregate 1–2 GB/s, conservatively adding CPU and I/O without overlap. Seven workers are used; there is no fictitious 32-fold speedup. More production clusters can increase scan CPU cost beyond this three-occupied-cluster fixture.

The cold-I/O scenario includes 144*N scratch bytes and up to 288*N logical source bytes. The latter includes counts, labels, distances and the favourable-fraction pass, and allows seven separate scans of the contiguous seven-feature HDF5 input. Actual physical reads can be much lower when page cache is effective. Local-scratch I/O and shared-pool I/O may have different rates; measure both before treating these projections as a schedule. The rework removes logarithmically repeated scratch I/O; it does **not** prove the review's under-one-hour expectation at 1e10.

At 1e10, 560 GB of local scratch may itself be unavailable. If scratch and OUT share a filesystem the guard requires at least 720 GB free for one seven-worker fit, before the configured minimum-free floor. Existing pool storage is already occupied and is not double-counted. At 1e9 the combined requirement is 72 GB. Scaler selection separately needs 8 GB or 80 GB of scratch; it does not overlap fit statistics. Scaler memmap residency is governed by the operating system, as in the requested original algorithm.

## Evidence and remaining work

The external report directory contains `statistics_probe_round2.py`, `statistics-round2-probe.json`, `statistics-round2-new-only.json`, `local_probe_round2.py` and `local-probe-round2.json`. Round 1 scripts/results were preserved. Reproduce the differential and isolated statistics measurements with:

```bash
NUMBA_NUM_THREADS=7 .venv/bin/python /path/to/report-dir/statistics_probe_round2.py "$PWD"
NUMBA_NUM_THREADS=7 .venv/bin/python /path/to/report-dir/statistics_probe_round2.py "$PWD" --new-only
```

No 1e9/1e10 execution or 32-core GP benchmark was attempted because this child cannot access the cluster and the local probe is intentionally 1e7 rows. Scratch throughput/capacity, shared-storage read amplification, Linux aggregate RSS, and real Xeon wall time remain unqualified. No GPU, MPI, stream or checkpoint changes were made because those remain outside this stream's ownership. The review's existing float64 rank-sum rounding above 2**53 was left unchanged as requested; the order statistics and tie policy remain exact, and parallel workers use exactly the same scan arithmetic as serial execution.
