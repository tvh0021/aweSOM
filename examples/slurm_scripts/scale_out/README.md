# Scale-out timing kit: threads, nodes and rows for `train_batch_stream`

These jobs time the streaming SOM trainer on synthetic gmm_bbh7 rows. The generator and seeds (data seed 20261103, SOM seed 42) are the ones from jobs 8238 and 8243. Every job runs 5 epochs with `final_pass="none"` on the 91x64 and 160x100 lattices, both with r_min 8. Multi-rank runs use `train_batch_stream(comm=...)`: there is one rank per node (or per socket), chunk `i` belongs to rank `i % ranks`, and the per-epoch sums are Allgathered and added in rank order. Each rank generates only the chunks it owns, so nothing is read from disk during training.

## Files

| File | Role |
| --- | --- |
| `probe.sbatch` | environment probe, mpi4py install into `mpi-site/`, MPI launcher test, 20 GB lagoon read test; submit first |
| `threads.sbatch` | 1 node, 1e8 rows, one lattice per job (`LATTICE`): 1x32, 1x16, 1x8 threads, then 2 ranks x 16 threads with one rank per socket |
| `nodes.sbatch` | 1e8 rows, both lattices, 1, 2 and 4 nodes, 1 rank x 32 threads per node; runs parity, the summary and `gate.json` |
| `rows.sbatch` | 1 node, 32 threads, both lattices, 1e7 and 1e8 rows |
| `rows_1e9.sbatch` | 1 node, 32 threads, 1e9 rows, one lattice per job; supplies the parity reference for `nodes_1e9` |
| `nodes_1e9.sbatch` | 1e9 rows, 2 and 4 nodes, both lattices; refuses to start unless `gate.json` shows >= 70 % node efficiency at 2 and 4 nodes for every lattice in `LATTICES` (or `FORCE=1`) |
| `common.sh` | shared code: OUT guards, environment, probes, launcher choice, `run_one` |
| `scale_out.py` | one timed run, 1 or more ranks; writes `run.json`, `rank<r>.json`, `weights.npy`, `checkpoint.npz` and `progress.log` |
| `parity.py` | compares multi-rank weights with the 1-rank weights of the same setup at rtol 1e-4 and atol 1e-12; exits 1 on failure |
| `summarize.py` | builds `summary.md` (rates, efficiencies, reduction, io_wait, RSS, I/O, and a labelled projection to 72 and 144 cores) and `gate.json`; marks UNDER-PINNED runs (a rank with fewer CPUs than threads) and leaves them out of efficiencies and the gate; lists unfinished runs. Works on a job killed mid-sweep: `python summarize.py --out DIR <job OUT>` |
| `probe.py` | node environment JSON and the file write/read rate tests |
| `bind_probe.sbatch`, `bind_probe.py` | 1 node, 20 min: where a rank loses its CPUs in `MPI_Init` (cases below); writes `probe/bind-<case>.txt` and one summary line per case to the job log |
| `gmm_bbh7.npz` | generator parameters, copied from the Z-scale kit |
| `aweSOM-src/`, `wheels/`, `kit-manifest.json` | vendored aweSOM source, the offline mpi4py wheel (and source-build files), and the hashes of every file; only in a kit built by `make_kit.sh` |

In the repository, `[WHEELS=DIR] make_kit.sh DEST [REVISION]` builds a kit from a committed revision. `scale_out.py` checks the vendored package against `kit-manifest.json` and records the result in `run.json` under `source.package_matches_manifest`.

## Copy to the cluster (run on the laptop)

```bash
rsync -av --exclude __pycache__ --exclude '*.pyc' --exclude '*.nbc' --exclude '*.nbi' --exclude .DS_Store \
  /Volumes/ExternalNVMe/awesom-results/planning/2026-11-05/scale-out/mpi/kit/ \
  gp:/lagoon/trungha/bbh-output/for-awesom-2026-10-28/scale_out_mpi_kit/
```

Keep the kit on lagoon. OUT defaults to `$KIT/runs/<job>-<jobid>`, which puts the I/O test on lagoon as well.

## Submit (on the cluster, in this order)

```bash
K=/lagoon/trungha/bbh-output/for-awesom-2026-10-28/scale_out_mpi_kit
cd $K
sbatch --output=$K/%x-%j.out --partition=gp-default --nodes=2 --export=ALL,KIT=$K $K/probe.sbatch
# after the probe: read runs/probe-<id>/probe/{mpi-install.txt,hello-*.txt,sinfo.txt}
sbatch --output=$K/%x-%j.out --export=ALL,KIT=$K,LATTICE=91x64 $K/threads.sbatch
sbatch --output=$K/%x-%j.out --export=ALL,KIT=$K,LATTICE=160x100 $K/threads.sbatch
sbatch --output=$K/%x-%j.out --export=ALL,KIT=$K $K/rows.sbatch
sbatch --output=$K/%x-%j.out --partition=gp-default --export=ALL,KIT=$K $K/nodes.sbatch
sbatch --output=$K/%x-%j.out --export=ALL,KIT=$K,LATTICE=91x64 $K/rows_1e9.sbatch
sbatch --output=$K/%x-%j.out --export=ALL,KIT=$K,LATTICE=160x100 $K/rows_1e9.sbatch
# only if the gate passes: $VENV/bin/python $K/summarize.py --check-gate $K/runs/nodes-<id>/summary/gate.json
sbatch --output=$K/%x-%j.out --partition=gp-default --export=ALL,KIT=$K,GATE=$K/runs/nodes-<id>/summary/gate.json,REF_DIRS=$K/runs/rows1e9-91x64-<id>:$K/runs/rows1e9-160x100-<id> $K/nodes_1e9.sbatch
```

Before you submit, run `env -u SCALE_N -u LATTICES -u LATTICE -u OUT -u FORCE -u RESUME_RUN` so stray test settings stay out of the jobs. Each job refuses to start if its OUT already exists.

## Partition

Each file sets `--partition=debug`, as the brief asks. The 2026-10-28 `sinfo` listed one debug node, lm-n0001 (48 cores, 770 GB). On debug:

- `nodes.sbatch` and `nodes_1e9.sbatch` cannot start, because they need 4 nodes.
- `probe.sbatch` asks for 1 or 2 nodes and gets 1, which leaves no second node for the 2-node launcher test or the cold read (job 8401).
- the single-node jobs run on a different CPU from jobs 8238 and 8243 (gp-n0013, Xeon Gold 6130).

The probe's `sinfo.txt` shows the current layout. If debug still has one node, submit the node sweeps with `--partition=gp-default`, as in the commands above. Each gp node has 32 cores and 192 GB. Submit the probe with `--partition=gp-default --nodes=2`, as above, so the 2-node launcher test and the cold read run. mpi4py is installed once, by the batch shell on the first node, before any `srun`, so the two nodes never write `mpi-site/` at the same time. For thread and row rates comparable with 8238, submit threads, rows and rows_1e9 the same way. Every job asks for explicit CPUs and memory (64 GB at 1e8 rows, 120 GB at 1e9) and never takes all of lm-n0001.

## Cluster notes (probe 8407, 2026-11-05, gp-n0021 + gp-n0022)

| Partition | Nodes | CPUs, memory per node | Time limit |
| --- | --- | --- | --- |
| gp-default / gp-long | 40, gp-n[0001-0040] | 32, 192 GB | 2 days / 20 days |
| wl-default / wl-long | 24, wl-n[0001-0024] | 32, 192 GB | 2 days / 20 days |
| all-default | 64, the gp and wl nodes | 32, 192 GB | none |
| debug / lm-largemem | 1, lm-n0001 | 48, 770 GB | 12 h / 1 day |

- GRES is `(null)` on every partition and node, so the cluster has no GPUs. The GPU probe cannot run here; it needs Tigris.
- gp nodes: Xeon Gold 6130, 2 sockets x 16 cores, 2 NUMA nodes.
- Compute nodes have no internet and no C headers, so pip cannot download and nothing can be compiled there (see MPI).
- srun MPI plugins: `pmix` (`pmix_v4`) and `pmi2`.
- Multi-rank runs on GP lost CPUs in `MPI_Init` until `launch_cmd` set `HFI_NO_CPUAFFINITY=1` (with `PSM3_NO_CPUAFFINITY=1 IPATH_NO_CPUAFFINITY=1`). GP nodes have Intel Omni-Path (`hfi1_0`). In `MPI_Init`, Open MPI's `btl ofi` opens libfabric's `psmx2` provider, and libpsm2's `hfi_userinit` pins a process whose mask is the whole node to one CPU (jobs 8413 and 8444: 1 rank x 32 CPUs per node, 32 -> 1). Half-node masks (2 ranks x 16, job 8409) are left alone. Job 8448 (`bind_probe.sbatch`) proved it: its strace names libpsm2 as the caller of `sched_setaffinity(0, [0])`, case J (the three variables) and case K (`pml ob1`, `btl self,vader,tcp`) keep 32 CPUs, and Open MPI's binding policy (`none` or `core`) makes no difference. `--mpi=pmi2` and `pml ucx` abort in `MPI_Init` on GP. The 1-rank path of `run_one` (plain srun, no MPI) is not affected. The guard stays: `scale_out.py` saves its CPU set before MPI starts and sets it back right after `MPI_Init` if it shrank, records `affinity_restored` in `rank<r>.json` (shown as `32 (restored)` in the summary) and writes one stderr line per affected rank. `affinity_preflight` does the same, prints `restored=<CPUs after the restore>` (0: not needed) on each `AFFINITY` line, and writes a `WARNING` line to the job log if the restore fired. `nodes`, `nodes_1e9` and `threads` stop with exit 3 if any rank is still short after the restore, and `scale_out.py` refuses to train on every rank in that case. Analysis: `planning/2026-11-05/scale-out/mpi/affinity-round2.md`.
- I/O (20 GB files): lagoon cold read 1.5 GB/s, warm reread 3.5 GB/s, two nodes reading at once 1.4–1.5 GB/s each, lagoon write about 1 GB/s; node-local `/tmp` read 3.2–3.4 GB/s.

## MPI

Compute nodes have no internet (probe 8401 could not resolve pypi.org) and no C headers (probe 8407: `stdlib.h: No such file or directory`), so nothing can be compiled there. `probe.sbatch` therefore installs the binary mpi4py wheel offline:

1. Unless `from mpi4py import MPI` already works and names the module's Open MPI version (step 2), it moves any old `mpi-site/` to `mpi-site.stale-<job>-<time>`, loads the newest `openmpi` module (`openmpi4/4.1.6` on GP) and runs `pip install --no-index --find-links $KIT/wheels --target $KIT/mpi-site --only-binary mpi4py mpi4py`. The PyPI wheel holds an Open MPI and an MPICH build of `mpi4py.MPI` and dlopens the MPI library at import. The probe takes the library from the module (`$(dirname $(dirname $(command -v mpicc)))/lib/libmpi.so.40`), refuses to go on if it is missing, and records it in `mpi-site/LIBMPI`. It records the module in `mpi-site/MODULE`. Every job loads that module, exports `MPI4PY_LIBMPI` from `mpi-site/LIBMPI`, and puts `mpi-site` on `PYTHONPATH`. `srun` passes the whole environment to the ranks (nothing in the kit filters it), and the `mpirun` lines add `-x MPI4PY_LIBMPI`.
2. The probe then checks that `MPI.Get_library_version()` names the module's Open MPI version and that `mpi4py.MPI.__file__` is not the MPICH build, and writes both to `probe/mpi4py-version.txt`. A failed or mislinked install is moved to `mpi-site.failed-<job>-<time>`.
3. `MPI_BUILD=source` (in `--export`) compiles mpi4py from the sdist in `$KIT/wheels` against the module's `mpicc` instead. Use it only where compute nodes have a compiler, such as Tigris. PyPI also has aarch64 wheels (`manylinux2014_aarch64`), so on Tigris the default wheel route works too once that wheel is in `WHEELS`.
4. Only if the kit has no `wheels/`, it tries the same install from PyPI, then the PyPI `mpi4py` and `openmpi` wheels. Both need network access from the node, which GP compute nodes do not have.

Do not use the `py3-mpi4py` module: it is built for the system python, not the 3.11 venv.

### Offline wheels

`make_kit.sh` copies `WHEELS` (default `/Volumes/ExternalNVMe/awesom-results/planning/2026-11-05/scale-out/mpi/wheels`) into the kit as `wheels/`, checks each file against `WHEELS/SHA256SUMS`, and lists each file's sha256 under `wheels` in `kit-manifest.json`. The wheels stay out of git. `make_kit.sh` requires the binary wheel `mpi4py-4.1.2-cp311-cp311-manylinux1_x86_64.manylinux_2_5_x86_64.whl`, which the default route installs on GP. The rest is the `MPI_BUILD=source` set and is optional: the mpi4py sdist and its build-isolation requirements, `setuptools >= 42` and `build` (from mpi4py's `pyproject.toml`), their dependencies `packaging` and `pyproject_hooks`, and `cython >= 3.0.1`, which the build needs because the sdist has no generated `MPI.c`. If the sdist is there, `make_kit.sh` requires a cython wheel too. Cython comes twice: the cp311 manylinux x86_64 wheel and the pure-Python wheel as a slower fallback. For Tigris, add the `manylinux2014_aarch64` mpi4py wheel (`--platform manylinux2014_aarch64` below).

To rebuild the directory, run this on a machine with internet:

```bash
W=/Volumes/ExternalNVMe/awesom-results/planning/2026-11-05/scale-out/mpi/wheels
pip download --no-deps --only-binary :all: --python-version 3.11 --implementation cp --abi cp311 \
  --platform manylinux1_x86_64 --platform manylinux_2_5_x86_64 -d $W mpi4py==4.1.2
# optional, MPI_BUILD=source only:
pip download --no-deps --no-binary :all: -d $W mpi4py==4.1.2
pip download --no-deps --only-binary :all: -d $W setuptools==84.2.1 build==1.6.1 packaging==26.3 pyproject_hooks==1.3.3
pip download --no-deps --only-binary :all: --python-version 3.11 --implementation cp --abi cp311 \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 -d $W cython==3.3.1
pip download --no-deps --only-binary :all: --python-version 3.11 --implementation py --abi none \
  --platform any -d $W cython==3.3.1
(cd $W && shasum -a 256 *.whl *.tar.gz > SHA256SUMS)
```

To check the source set on the laptop, run `pip install --no-index --find-links $W --target /tmp/x --no-binary mpi4py mpi4py`; it builds against Homebrew Open MPI if `mpicc` is on PATH. The Linux wheel cannot be installed on macOS; PyPI's macOS wheel of the same version (`--only-binary :all:` into a scratch directory) exercises the same dlopen route with `MPI4PY_LIBMPI=/opt/homebrew/lib/libmpi.40.dylib`.

If the build still fails, run the same `pip install --target` command on a login node. Multi-rank jobs try `srun --mpi=pmix`, `pmix_v4`, `pmi2`, then `mpirun`. Each is tested with `python -m mpi4py.bench helloworld`, and the first one that starts the right number of ranks is used. Set `LAUNCH=` to force one. `launch_cmd` in `common.sh` builds the multi-rank command once for `run_one` and the affinity pre-flight. The srun form is `srun --mpi=pmix --nodes=N --ntasks=R --ntasks-per-node=P --cpus-per-task=C --distribution=block:block --cpu-bind=verbose,cores` with `HFI_NO_CPUAFFINITY=1 PSM3_NO_CPUAFFINITY=1 IPATH_NO_CPUAFFINITY=1` in the environment, which stops libpsm2 from pinning a rank in `MPI_Init` (cluster notes); mpirun passes the same variables with `-x`. The mpirun form binds each rank itself with `--map-by ppr:P:node:pe=C` (or `ppr:1:package:pe=C` per socket) and `--bind-to core --report-bindings`.

### Binding probe

`bind_probe.sbatch` runs `bind_probe.py` (CPU count before, after loading libmpi, after `MPI_Init`, and per thread; no restore) under one `srun` per case, 60 s each: A is `launch_cmd` without the three `*_NO_CPUAFFINITY` variables, B adds ess/hwloc/pmix/pml/mtl verbosity, C uses `--mpi=pmi2`, D `--cpu-bind=none`, E `OMPI_MCA_orte_bound_at_launch=1`, F adds binding policy `core`, G 2 ranks x 16 (the 8409 shape), H the restore, I the task's environment and mask without MPI.
J sets `HFI_NO_CPUAFFINITY=1 PSM3_NO_CPUAFFINITY=1 IPATH_NO_CPUAFFINITY=1` (`launch_cmd` as `run_one` uses it), K uses `pml ob1` with `btl self,vader,tcp` (no fabric library), L uses `pml ucx` only, and M runs A under `strace -f -k -e trace=sched_setaffinity` to name the caller. `probe/fabric.txt` lists the HCA devices, MPI components and transport libraries.
A 32-CPU task covers a whole gp node but only 32 of lm-n0001's 48 CPUs, so the job warns on `debug`. To reproduce the 8413/8444 condition, submit it with `--partition=gp-default`.

## Time estimates

These scale from job 8238: 1e9 rows, 91x64, 32 threads, epoch 1 933 s, later epochs 176–312 s. For 160x100, assume 2–2.75 times as long.

| Job | Estimate | Limit |
| --- | --- | --- |
| probe | 10–20 min (20 GB written twice, read 3 ways) | 1 h |
| threads 91x64 / 160x100 | ~30 / ~80 min | 2 h |
| rows | ~20 min | 2 h |
| nodes | ~30 min | 2 h |
| rows_1e9 91x64 / 160x100 | ~35 min / 1.5–2 h | 2 h |
| nodes_1e9 | ~1.5 h | 4 h |

If rows_1e9 hits its limit at 160x100, resubmit it with `RESUME_RUN=$K/runs/rows1e9-160x100-<id>/runs/160x100-n1e9-1x32`. Training continues from the last finished epoch. The resumed weights are bitwise equal to those of an uninterrupted run.

`IO_TEST_GB=0` skips the 20 GB lagoon test in the timing jobs. The brief asks for it at the start of every job, so it runs by default.

## Outputs per job (`$OUT`)

| Path | Content |
| --- | --- |
| `runs/<run>/run.json` | settings; per-epoch seconds, rows, rows/s, BMU, accumulate, update, reduction and checkpoint times; per-rank rows and pass times; nodes; peak RSS per rank; git commit; manifest check |
| `summary/summary.md`, `summary/gate.json` | tables and the 1e9 gate |
| `parity/parity.md` | max rel diff of each multi-rank run against its 1-rank reference |
| `probe/` | sinfo (job nodes, then every partition and node with gres), modules, `srun --mpi=list`, lscpu, `free -g`, `df -h`, `numactl -H`, environment JSON per node, mpi4py import, launcher tests, I/O JSONs; `affinity-<case>.txt` (CPUs per rank before and after `MPI_Init`, and after the restore) |
| `logs/<run>.log`, `status/<job>-<restart>/final.tsv` | run logs; VALIDATED or FAILED |
