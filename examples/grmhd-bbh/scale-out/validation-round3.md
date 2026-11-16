# Round 3 validation

Completed on `impl/scale-out-driver-implementation`, starting from `2b45bc4`. The implementation and scheduler fixes tested here are at `c2cb19bde405f36da053d5092d0729e82babb14b`. Current head is `c2cb19bde405f36da053d5092d0729e82babb14b`. Nothing was pushed, rebased or merged. `src/aweSOM/stream.py` is unchanged, and no GPU or MPI imports were added.

## Commits

```text
aef3c2f Check transform equality on every available epoch read path
f479e2d Allow one GiB merge buffers without changing summary arithmetic
ab6d6e9 Check scratch and output space on full training runs
eb9041c Support exact deferred statistics with immutable source fits
c2cb19b Allow GP debug fits to run on its available node
```

Item 4 is one implementation commit, `eb9041c`, containing the driver, statistics shell and sbatch launchers, strengthened tests and README. The existing GP fit launcher's exclusion removal is separable and committed as `c2cb19b`; the new statistics launcher was created without that exclusion in item 4. This follows the supplied debug topology, where excluding lm-n0001 would exclude its only node. No scheduler topology query or submission was performed.

## Changes and equality contract

- `aef3c2f` covers transformed `__call__`, and callable `rows(ix)` and `lazy_chunks()` when present. The test compares bytes against both stored scaled rows and transforming the unscaled reads, with repeated lazy reads and duplicate/unsorted sample indices. It passes on this branch's old source. The same test was also executed against the EpochSource class extracted from `f182ca3`, exercising all three merged read paths without changing either checkout.
- `f479e2d` raises the merge refill cap to 2**26 records, at most 1 GiB of value/label buffers per worker. Float32 and float64 tests compare output bytes under small and large refill caps.
- `ab6d6e9` checks scratch capacity before full-run pooling/training as well as reused-pool fits. Full-run estimates account for pool creation, multiple fits, concurrent scratch and retained vectors.
- `eb9041c` adds exact inline/deferred statistics and fresh-output completion. Shared `statistics_fields()` constructs both model and JSON fields. Deferred fits retain their BMU artifacts. Completion does not retrain or search for BMUs, preserves training provenance and the complete source directory, and records the statistics job separately. Collection rejects still-deferred fits.

Astra's escalation review found the remaining serialization defect: placeholder key order differed from inline construction. The corrected STATISTIC_FIELDS order now matches `statistics_fields()`, so replacing placeholders preserves JSON order. The stronger test compares unsorted serialized final JSON after excluding only the six documented trainer durations, fit wall time and stage timing containers. Backend selection is fixed to patch for this controlled comparison. Execution metadata and peak RSS are test-controlled instead of being silently excluded. The six trainer durations are epoch_time, io_wait, bmu_time, acc_time, update_time and final_time.

The tests compare every statistics field, all model metadata, numeric HDF5 dataset dtype/shape/bytes, root attributes, exact feature-name strings and label/U-matrix NPY bytes. Whole HDF5 container layout is not required to match. Parameterized CLI tests cover inline and later workflows, deferred collection refusal, completion of both seeds, successful collection and source immutability. The latest pre-escalation test adjustment was not known to fail; Astra confirmed it passed, then strengthened the contract and corrected key ordering.

## Owner verification

```text
NUMBA_NUM_THREADS=2 .venv/bin/pytest tests/production_pool_scaleout_test.py::test_deferred_statistics_matches_inline -q
1 passed in 5.04s

NUMBA_NUM_THREADS=2 .venv/bin/pytest tests/production_pool_test.py tests/production_pool_scaleout_test.py -q
68 passed in 74.05s (0:01:14)

NUMBA_NUM_THREADS=2 .venv/bin/pytest tests/production_pool_test.py tests/production_pool_scaleout_test.py tests/stream_test.py -q
122 passed, 2 skipped in 74.51s (0:01:14)
```

The two skips are the existing JAX comparisons because JAX is unavailable. No requested CPU check failed. `bash -n` passed for the fit/statistics shell and sbatch launchers and every Bash block in the README. `git diff --check` passed.

## Exact 1e7-row probe

The probe was copied to `statistics_probe_round3.py` before running, and its output names were changed to Round 3. The reference remains `reference_pool.py` from `82dee49`. The seven-worker HDF5/mapped-label run uses the same seed, ties, empty cluster, two snapshots and 2**18-row chunks as Round 2. All 13 quantities match exactly, with identical NaN masks.

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

The in-memory reference took 4.932 s; the current bounded statistics path took 7.117 s. An isolated new-only run took 7.361 s, with parent peak RSS 1,442,856,960 bytes and sampled process-tree peak RSS 3,714,859,008 bytes. Process-tree sampling uses ps at 100 ms intervals and can miss shorter peaks or count shared pages more than once. The parent peak includes input generation. These are local measurements, not GP or Tigris qualification.

Evidence files are `statistics_probe_round3.py`, `statistics-round3-probe.json` and `statistics-round3-new-only.json` beside this report. The copied probe changes only the evidence filenames. SHA256 checks confirmed that the three saved Round 2 script/JSON files remain unchanged; hashes are in `round3-preserved-evidence-sha256.json`. Owner test logs are `round3-owner-equivalence.log`, `round3-owner-production.log` and `round3-owner-all.log`.

Reproduce the probe from this checkout with:

```bash
NUMBA_NUM_THREADS=7 .venv/bin/python /path/to/report-dir/statistics_probe_round3.py "$PWD"
NUMBA_NUM_THREADS=7 .venv/bin/python /path/to/report-dir/statistics_probe_round3.py "$PWD" --new-only
```

These rerun commands overwrite only the Round 3 JSON outputs. Make another named copy before a later investigation if these Round 3 observations must be preserved.

## Deferred-statistics workflow and artifact state

Keep the original pool available throughout training and statistics completion.
The following commands assume two validated inputs with at least five billion
rows each. Use the same input paths, anchor, holdout, sampling and scaling
settings when preparing, fitting and collecting the pool.

Set the paths and create the output parents. Every individual OUT below must
be fresh, including a statistics retry's OUT.

```bash
export REPO=/absolute/aweSOM VENV=/absolute/venv
export INPUT_1=/absolute/first/features.h5 INPUT_2=/absolute/second/features.h5
export POOL_OUT=/absolute/pool-1e10
export POOL="$POOL_OUT/results"
export FIT_ROOT=/absolute/fits-1e10 COMPLETE_ROOT=/absolute/statistics-1e10
export ROWS_PER_SNAPSHOT=5000000000 ROWS_PER_EPOCH=10000000000 EPOCHS=40
mkdir -p "$FIT_ROOT" "$COMPLETE_ROOT"
```

Prepare the pool once. Run this command inside a suitably provisioned allocation;
scaler selection needs 80 GB of scratch at 1e10 rows. The default pool needs
280 GB for seven float32 features. Set `SCRATCH` to an existing scratch directory
on the node executing the command.

```bash
export SCRATCH=/absolute/node-local/scratch
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor 94208 --held-out 405504 \
  --out "$POOL_OUT" --pool-only --scratch "$SCRATCH" \
  --rows-per-snapshot "$ROWS_PER_SNAPSHOT" --rows-per-epoch "$ROWS_PER_EPOCH" \
  --epochs "$EPOCHS" --r-min 8
```

Submit the fits with statistics deferred. The GP templates use debug without an
exclusion of its only node, lm-n0001. These commands submit work; they do not
establish that a 1e10-row training run fits the requested wall time or memory.
Use the trainer's supported checkpoint/resume workflow when required.

```bash
for LATTICE in 91x64 48x34; do
  for SEED in 42 43; do
    export LATTICE SEED OUT="$FIT_ROOT/${LATTICE}-s${SEED}"
    sbatch "$REPO/examples/grmhd-bbh/scale-out/production-fit-gp.sbatch" \
      --statistics later
  done
done
```

Wait for each fit to finish before submitting its statistics job. The statistics
launcher defaults to three single-threaded feature workers. At 1e10 rows,
`THREADS=3` needs up to 240 GB of scratch; `THREADS=1` needs up to 160 GB.
Completion copies 40 GB of labels per model into the new OUT, in addition to
small model and JSON files. When OUT and scratch share a filesystem, the space
guard adds both demands. Set scratch to a directory that exists on each
statistics node, or omit SCRATCH to use SLURM_TMPDIR, TMPDIR, then the output
filesystem. Confirm the chosen root's capacity before submission.

```bash
export THREADS=3
for LATTICE in 91x64 48x34; do
  for SEED in 42 43; do
    export FIT_OUT="$FIT_ROOT/${LATTICE}-s${SEED}"
    export OUT="$COMPLETE_ROOT/${LATTICE}-s${SEED}"
    sbatch "$REPO/examples/grmhd-bbh/scale-out/production-fit-statistics-gp.sbatch"
  done
done
```

The direct equivalent for one fit is:

```bash
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --statistics-only "$FIT_ROOT/91x64-s42" \
  --out "$COMPLETE_ROOT/91x64-s42" \
  --threads-per-seed 3 --scratch "$SCRATCH"
```

A retry reads the same deferred source and uses a different fresh OUT.
Completion restores the saved chunk size and tube thresholds. It reads saved
weights, labels, nodes and distances and performs no training or BMU search.
The original pool, deferred model, `fit-results.json`, `pool-state.json`, label
and U-matrix files must remain available, together with the retained
`{tag}-nodes.npy` and `{tag}-distances.npy`. Those two files use 120 GB per model
at 1e10 rows; labels use another 40 GB.

Deferred fit JSON has `statistics: "deferred"`. The following fields are null:

- `cluster_medians`, `pooled_medians`, `cluster_effects`, `cluster_quartiles`,
  `cluster_rows`, `cluster_share`, `cluster_joint_favorable_fraction` and
  `sheet_candidates`.
- `legacy_threshold_rule_diagnostic`, `tube_rule` and `reject_distance`.
- `selected_sheet_clusters`, `selection_audit` and `seed_overlap`.
- `verification.saved_components_equal.distance_quantiles` and
  `verification.saved_components_equal.reject_distance`.

`sheet_selection_status` is `"deferred"`. In the model HDF5 file,
`node_dist_q99` and the root `reject_distance` attribute contain NaNs. Metadata
has `statistics="deferred"` and a deferred selection status/audit. Metadata
omits `cluster_effects`, `cluster_quartiles`, `cluster_rows`, `cluster_share`,
`cluster_joint_favorable_fraction`, `sheet_candidates`, `tube_clusters` and
`tube_rule`. Weights, scaler and cluster assignments are present. Use the
completed artifacts for distance thresholds and final selection.

Completion restores all numerical statistics and metadata to the inline
values, sets JSON `statistics: "complete"`, removes the model's temporary
`statistics` metadata marker and restores the two verification booleans. Sheet
selection returns to pending until collection. The completed fit preserves
training provenance; `statistics-job.json` records the completion job separately.
Only documented timing observations differ in the controlled equivalence test.

Completion leaves the entire deferred source unchanged. Its fresh results
contain labels, U-matrix, model and JSON, but no node/distance copies. Deleting
retained source nodes/distances is an operator step after validating and
preserving the completed artifacts. Once deleted, that source cannot be used
to retry statistics. No launcher or completion command deletes them.

After all four statistics jobs finish, collect their completed outputs. A
still-deferred fit is rejected with a clear `run --statistics-only first` error.
Collection copies artifacts and creates the frozen models and manifest, so its
OUT needs another 160 GB for the four label vectors at this pool size.

```bash
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor 94208 --held-out 405504 \
  --rows-per-snapshot "$ROWS_PER_SNAPSHOT" --rows-per-epoch "$ROWS_PER_EPOCH" \
  --epochs "$EPOCHS" --r-min 8 --reuse-pool "$POOL" \
  --out /absolute/fresh/collection-1e10 \
  --collect "$COMPLETE_ROOT/91x64-s42/results" "$COMPLETE_ROOT/91x64-s43/results" \
            "$COMPLETE_ROOT/48x34-s42/results" "$COMPLETE_ROOT/48x34-s43/results"
```

A full driver run can also use `--statistics later`. It emits a deferred
`fit-results.json` bundle instead of selecting frozen models. Complete that
bundle with `--statistics-only`, then collect its completed results with the
same training settings.

## Remaining qualification

No 1e9/1e10 run, actual cluster submission, aarch64 installation or Linux billion-row RSS measurement was attempted. This task permits local implementation and kits, not cluster execution. The exact fixture/probe results establish numerical equivalence and lifecycle behavior, not a twelve-hour training/statistics schedule or scratch capacity on a particular node. The remaining deployment work is for the orchestrator. The older incomplete Round 3 status is archived separately as `round3-incomplete-before-astra.md`; this section replaces it.
