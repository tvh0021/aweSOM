# Run one fit per node

Create a pool once with the same input, sampling, and scaling options used by the
fits. Pass `--pool-only --out /absolute/fresh/pool-output` to `production_pool.py`.
The prepared pool and `pool-state.json` are in `pool-output/results`.
The default pool contains float32 `raw`; add `--write-scaled` to retain the old
float64 dataset for comparison, then use `--check-scaling N` during a fit.

Export the shared configuration before submitting the four GP jobs:

```bash
export REPO=/absolute/aweSOM VENV=/absolute/venv
export POOL=/absolute/pool-output/results
export INPUT_1=/absolute/first/features.h5 INPUT_2=/absolute/second/features.h5
export ROWS_PER_SNAPSHOT=500000000 ROWS_PER_EPOCH=1000000000
for LATTICE in 91x64 48x34; do
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
  --collect /absolute/fits/91x64-s42/results /absolute/fits/91x64-s43/results \
            /absolute/fits/48x34-s42/results /absolute/fits/48x34-s43/results
```

Collection copies fit artifacts before updating selection metadata. It retains
source fit files and refuses missing seeds, duplicate fits, and mismatched pool
or training settings. Collection currently copies label vectors as well, so
budget four bytes per row per model on the destination filesystem.

For GH200, submit `production-fit-tigris.sbatch` with the site's partition name
on the `sbatch` command line. It requests 72 CPUs and 400 GB. The
`production-fit-grace.sbatch` variant requests 144 CPUs and 400 GB. Confirm the
site's memory limits first. Neither template assumes an x86 instruction set or
requires `numactl`. Install NumPy, SciPy, h5py, scikit-learn, and Numba wheels for
linux-aarch64, or build them on site. MPI and CUDA dependencies remain optional
and are not imported by this driver.

`--groups 64` preserves accumulator rounding. Set `GROUPS_COUNT=128` on 144 cores
to try more parallel accumulator tasks; this changes the last bits. Two workers
on a 144-core allocation can use `--seed-workers 2 --threads-per-seed 72` in the
full driver. The per-fit templates deliberately run one worker.

`--verify-labels` checks every pool label; the default verifies the first million
rows and checks backend parity on up to 8192 rows. `--resume PATH` requires a
trainer with a `resume` parameter. The driver passes a per-fit checkpoint path
when the installed trainer exposes `checkpoint`. A resumed fit still requires a
fresh OUT. Reuse the earlier checkpoint through `--resume PATH`.
