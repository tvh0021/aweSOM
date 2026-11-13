# Run one fit per node

Create a pool once with the same input, sampling, and scaling options used by the
fits. Pass `--pool-only --out /absolute/fresh/pool-output` to `production_pool.py`.
The prepared pool and `pool-state.json` are in `pool-output/results`.
The default pool contains float32 `raw`; add `--write-scaled` to retain the old
float64 dataset for comparison, then use `--check-scaling N` during a fit.

Export the shared configuration before submitting the GP jobs (one per lattice and seed):

```bash
export REPO=/absolute/aweSOM VENV=/absolute/venv
export POOL=/absolute/pool-output/results
export INPUT_1=/absolute/first/features.h5 INPUT_2=/absolute/second/features.h5
export ROWS_PER_SNAPSHOT=500000000 ROWS_PER_EPOCH=1000000000
for LATTICE in 100x100; do
  for SEED in 42 43; do
    export LATTICE SEED OUT=/absolute/fits/${LATTICE}-s${SEED}
    sbatch "$REPO/examples/grmhd-bbh/scale-out/production-fit-gp.sbatch"
  done
done
```

Use an existing output parent and a fresh OUT for each job. Set `SCRATCH` to a
node-local directory if the scheduler does not provide `SLURM_TMPDIR` or `TMPDIR`.
Exact statistics sort packed float32 features into 8-byte keys per row.
A single k-way merge reads each run once, without writing another generation.
With W=min(threads-per-seed, features) concurrent features, peak scratch is
max(16, 8*W) bytes per row for float32 pools, or 56 bytes per row with seven
concurrent features. Serial features need 16 bytes per row including the
separate float64 distance stage. Scaler selection needs 8 bytes per row.
Each worker reserves at most 1 GiB for merge value/label buffers, plus a
4 MiB output buffer, bounded run-formation arrays and its Python/Numba runtime.
Seven concurrent workers can reserve 7 GiB of merge buffers alone.
Merge buffers and metadata have fixed ceilings; more than 2**18 runs requires
a larger --chunk. Each statistics process uses one Numba/BLAS thread.
The driver removes its private scratch directory after success or an exception.
A job killed by the scheduler (time limit, SIGTERM, SIGKILL) leaves it behind;
remove `awesom-stat-*` directories under the scratch root by hand.

After all jobs finish, collect their results with the same training options:

```bash
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor 94208 --held-out 405504 \
  --rows-per-snapshot "$ROWS_PER_SNAPSHOT" --rows-per-epoch "$ROWS_PER_EPOCH" \
  --r-min 8 --reuse-pool "$POOL" --out /absolute/fresh/collection \
  --collect /absolute/fits/100x100-s42/results /absolute/fits/100x100-s43/results
```

Collection copies fit artifacts before updating selection metadata. It retains
source fit files and refuses missing seeds, duplicate fits, and mismatched pool
or training settings. Collection currently copies label vectors as well, so
budget four bytes per row per model on the destination filesystem.

## Run one fit over four nodes (MPI)

`production-fit-mpi-gp.sbatch` trains one fit on four gp-default nodes, one MPI
rank per node with 32 threads. Every rank trains on the pool windows it owns.
Rank 0 alone labels the pool, computes statistics and writes OUT; the other
ranks write no persistent outputs (only their node-local Numba cache and stdout
lines in the job log). Set `MPI_SITE` to the mpi4py 4.1.2 wheel install
(`/lagoon/trungha/bbh-output/for-awesom-2026-10-28/scale_out_mpi_kit/mpi-site`
on GP). The job runs the affinity pre-flight of
`examples/slurm_scripts/scale_out/common.sh` first and writes it to `$OUT-launch`.
Both seeds share the job name `so-sweep1` with `--dependency=singleton`, so the
second waits for the first. The job runs in, and writes its Slurm log to,
`/lagoon/trungha/bbh-output/for-awesom-2026-10-28`; keep OUT and the collection
under that root too. Submit both seeds, then collect them:

```bash
export REPO=/absolute/aweSOM VENV=/absolute/venv POOL=/absolute/pool-output/results
export MPI_SITE=/lagoon/trungha/bbh-output/for-awesom-2026-10-28/scale_out_mpi_kit/mpi-site
export INPUT_1=/absolute/first/features.h5 INPUT_2=/absolute/second/features.h5 LATTICE=100x100
FITS=/lagoon/trungha/bbh-output/for-awesom-2026-10-28/fits-mpi; mkdir -p "$FITS"
for SEED in 42 43; do
  SEED=$SEED OUT=$FITS/$LATTICE-s$SEED sbatch "$REPO/examples/grmhd-bbh/scale-out/production-fit-mpi-gp.sbatch"
done
# after both jobs end:
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor 94208 --held-out 405504 \
  --rows-per-snapshot 500000000 --rows-per-epoch 1000000000 --r-min 8 \
  --lattices "$LATTICE" --reuse-pool "$POOL" --out "$FITS/collection-$LATTICE" \
  --collect "$FITS/$LATTICE-s42/results" "$FITS/$LATTICE-s43/results"
```

Wall time: training takes about `EPOCHS * ROWS_PER_EPOCH / (2.0e6 * 0.93 * 4)`
seconds (one node measured about 2.0e6 rows/s on real data at 40 epochs; four
nodes at about 0.93 efficiency): 2.2 h for the default 60 epochs of 1e9 rows
(phases 20 at r_max, 10 tapering, 30 at r_min; `--phases P,T,M` sets others). Rank 0 then
labels the whole pool on one node, about `POOL_ROWS / 2.0e6` seconds, and inline
statistics add about an hour at 1e9 rows. The six-hour limit covers 1e9 rows.
The fit record carries `ranks`, `rank_hostnames`, `rank_numba_threads` and
`affinity_restored` (null on a rank where MPI_Init left its CPU set alone).
`--mpi` makes a launch without a multi-rank communicator fail; an MPI launch
runs only `--fit` with `--reuse-pool`.

For GH200, submit `production-fit-tigris.sbatch` with the site's partition name
on the `sbatch` command line. It requests 72 CPUs and 400 GB. The
`production-fit-grace.sbatch` variant requests 144 CPUs and 400 GB. Confirm the
site's memory limits first. Neither template assumes an x86 instruction set or
requires `numactl`. Install NumPy, SciPy, h5py, scikit-learn, and Numba wheels for
linux-aarch64, or build them on site. MPI and CUDA dependencies remain optional;
the driver imports mpi4py only under `--mpi` or an MPI launcher.

`--groups 64` preserves accumulator rounding. Set `GROUPS_COUNT=128` on 144 cores
to try more parallel accumulator tasks; this changes the last bits. Two workers
on a 144-core allocation can use `--seed-workers 2 --threads-per-seed 72` in the
full driver. The per-fit templates deliberately run one worker.

`--verify-labels` checks every pool label; the default verifies the first million
rows and checks backend parity on up to 8192 rows. `--resume PATH` requires a
trainer with a `resume` parameter. The driver passes a per-fit checkpoint path
when the installed trainer exposes `checkpoint`. A resumed fit still requires a
fresh OUT. Reuse the earlier checkpoint through `--resume PATH`.

## Defer statistics for a 1e10-row pool

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
export ROWS_PER_SNAPSHOT=5000000000 ROWS_PER_EPOCH=10000000000 EPOCHS=60
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

Submit the fits with statistics deferred. The training template uses debug on
lm-n0001 for fast turnaround, not for correctness. These commands do not establish
that a 1e10-row training run fits the requested wall time or memory.
Use the trainer's supported checkpoint/resume workflow when required.

```bash
for LATTICE in 100x100; do
  for SEED in 42 43; do
    export LATTICE SEED OUT="$FIT_ROOT/${LATTICE}-s${SEED}"
    sbatch "$REPO/examples/grmhd-bbh/scale-out/production-fit-gp.sbatch" \
      --statistics later
  done
done
```

Wait for each fit to finish before submitting its statistics job. The statistics
launcher reads the feature count from the deferred fit's `pool-state.json`
and defaults to that many single-threaded workers, capped at
`SLURM_CPUS_PER_TASK`. An explicit `THREADS` overrides the default.
The statistics template uses gp-default with 8 CPUs, 96G and an eight-hour limit,
enough for a 1e9 completion (about an hour). For 1e10 pass `--time=2-00:00:00`
on the sbatch command line (the gp-default maximum).
GP nodes have 32 cores and 192 GB of RAM. At 1e9 rows, seven workers need
56 GB of scratch plus at most 7 GiB of merge buffers. The 96G request leaves
about 37 GiB for run-formation arrays, file cache and runtimes.

Measured on 2026-11-05, node-local `/tmp` on every GP node is a tmpfs RAM disk
charged to the job's memory: 94 GB on gp nodes and 378 GB on lm-n0001.
The 56 GB scratch at 1e9 rows fits on a gp node. At 1e10 rows, seven workers
need about 560 GB, so put scratch under the job's lagoon directory with
`SCRATCH`, which the launcher passes as `--scratch`. Measured lagoon throughput
was about 1 GB/s writing and 1.5–3.5 GB/s reading.
Before any 1e10 statistics submission, time a 1e9 `--statistics-only` run
to check the wall-time request.

Completion copies 40 GB of labels per model at 1e10 rows into the new OUT,
in addition to small model and JSON files. When OUT and scratch share a
filesystem, the space guard adds both demands. If SCRATCH is omitted, the
launcher uses SLURM_TMPDIR, TMPDIR, then the output filesystem.
Confirm the chosen root's capacity before submission.

```bash
export SCRATCH=/absolute/lagoon/job-directory/scratch
# Leave THREADS unset to use the feature count, or export THREADS explicitly.
for LATTICE in 100x100; do
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
  --statistics-only "$FIT_ROOT/100x100-s42" \
  --out "$COMPLETE_ROOT/100x100-s42" \
  --threads-per-seed 7 --scratch "$SCRATCH"
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

After both statistics jobs finish, collect their completed outputs. A
still-deferred fit is rejected with a clear `run --statistics-only first` error.
Collection copies artifacts and creates the frozen models and manifest, so its
OUT needs another 80 GB for the two label vectors (4 bytes per row each) at this pool size.

```bash
"$VENV/bin/python" "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor 94208 --held-out 405504 \
  --rows-per-snapshot "$ROWS_PER_SNAPSHOT" --rows-per-epoch "$ROWS_PER_EPOCH" \
  --epochs "$EPOCHS" --r-min 8 --reuse-pool "$POOL" \
  --out /absolute/fresh/collection-1e10 \
  --collect "$COMPLETE_ROOT/100x100-s42/results" "$COMPLETE_ROOT/100x100-s43/results"
```

A full driver run can also use `--statistics later`. It emits a deferred
`fit-results.json` bundle instead of selecting frozen models. Complete that
bundle with `--statistics-only`, then collect its completed results with the
same training settings.
