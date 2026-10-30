# Submit the k10 production rerun

Copy this entire kit into `/lagoon/trungha/bbh-output/for-awesom-2026-10-28/production-rerun-k10` using your usual transfer method. The kit contains the driver, its source manifest, and the complete `aweSOM-src/src/aweSOM` package. Keep the existing `check7-venv` and input extraction directories beside this kit.

From a cluster login shell, submit both lattices with this exact command:

```bash
sbatch /lagoon/trungha/bbh-output/for-awesom-2026-10-28/production-rerun-k10/production_pool.sbatch
```

To run 91x64 only, submit:

```bash
sbatch /lagoon/trungha/bbh-output/for-awesom-2026-10-28/production-rerun-k10/production_pool.sbatch --91x64-only
```

Both modes use snapshots 94208 and 98304, hold out 405504, and fit seeds 42, 43, and 44. The default fits 91x64 first, then 48x34, both at r8 and target_clusters=10. The batch file requests an exclusive 32-core gp-default node, 180 GB of its 192 GB, and 24 hours. It excludes lm-n0001 and retains both required mail directives.

Allow roughly 8 to 12 hours for both lattices when planning submission. Job 8012 took 8.14 hours, mostly outside the recorded epoch loop. A precise 32-core ETA is unavailable because time outside its epoch loop was not partitioned. The local small test measured four fits in 10.50 seconds sequentially versus 6.56 seconds with two workers, a 1.60x speedup on the second repeat. This supports two workers, but does not predict production speedup. The scheduler limit is 24 hours. The 91x64-only mode has no reliable total ETA because 8012's slowest seed belonged to 48x34.

Two seeds use 16 Numba threads each. The last seed uses all 32 threads. The driver limits native BLAS and OpenMP libraries to one thread and records the actual Numba count in each model JSON. To compare sequential execution on the same node, export `SEED_WORKERS=1` before submission.

Outputs go beside the old run, to `HHR_flux_production_pool_k10_r8_JOBID`. The script refuses an existing output path. Read `production-pool-k10-JOBID.out`, `timing.log`, and `status/JOBID-0/final.tsv` before transferring results.

Replace `JOBID` and `USER_CLUSTER_HOST` in these transfer commands. Use your established login hostname. These commands are instructions for the user and have not been executed by the implementation agent.

```bash
rsync -a USER_CLUSTER_HOST:/lagoon/trungha/bbh-output/for-awesom-2026-10-28/HHR_flux_production_pool_k10_r8_JOBID/ /Volumes/ExternalNVMe/awesom-results/cluster/HHR_flux_production_pool_k10_r8_JOBID/
rsync -a USER_CLUSTER_HOST:/PATH_TO_SUBMIT_DIRECTORY/production-pool-k10-JOBID.out /Volumes/ExternalNVMe/awesom-results/cluster/
```

Transfer the whole output directory, including `run/results`, `timing.log`, and `status`. The result directory includes `pool.h5`, frozen and per-seed models, labels, U-matrices, model JSONs, and `freeze-manifest.json`. A failed run retains `results.partial` for diagnosis and requires a fresh output path for a new submission.

Read `models[].merge_cost_used` in `freeze-manifest.json` for each selected merge threshold. `merge_cost` remains the 0.05 floor. `target_clusters` and `n_clusters` accompany the selected cost in both the model sidecar JSON and HDF5 metadata. Use `selection[].medoid_seed` to identify the frozen model's record. Ties can produce fewer than ten clusters.

The existing APPROVED checks remain unchanged. An explicit `--r-min 8` differs from their `r_min=None`, and the 48x34 default resolves to r5. Therefore this requested kit records `diagnostic_only=true`, including in the two-lattice mode. The single-lattice mode also records the lattice departure. `VALIDATED/0` confirms execution and numerical checks, not approval of changed scientific settings. Do not erase these reasons.

Each model JSON retains the existing per-epoch timings and adds `execution` and `phase_timings`. The log emits a tagged wall and process CPU duration after training, clustering, labelling, statistics, export, and verification. It also records the hostname, CPU affinity where supported, thread environment, source commit, and input hashes.
