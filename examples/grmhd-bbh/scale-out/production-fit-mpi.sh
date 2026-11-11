#!/bin/bash
# One fit, one MPI rank per node (production_pool.py --mpi). Every rank trains on the pool
# windows it owns; rank 0 alone labels the pool, computes statistics and writes OUT.
# The launch is launch_cmd of examples/slurm_scripts/scale_out/common.sh (srun-pmix), after
# its affinity pre-flight. Pre-flight output: $OUT-launch/probe/affinity-fit.txt.
set -eo pipefail  # not -u: common.sh's module setup sources the site's lmod scripts
: "${OUT:?set a fresh absolute output directory}"
: "${VENV:?set the environment directory}"
: "${REPO:?set the aweSOM checkout}"
: "${MPI_SITE:?set the directory holding the mpi4py 4.1.2 wheel install}"
: "${POOL:?set the prepared results directory}"
: "${INPUT_1:?set the first extraction}"
: "${INPUT_2:?set the second extraction}"
: "${SEED:?set 42 or 43}"
LATTICE=${LATTICE:-100x100}
EPOCHS=${EPOCHS:-40}
LAUNCH_DIR=$OUT-launch
case "$OUT" in /*) ;; *) echo 'OUT must be absolute' >&2; exit 2;; esac
for d in "$OUT" "$LAUNCH_DIR"; do
  [ ! -e "$d" ] && [ ! -L "$d" ] || { echo "$d exists" >&2; exit 2; }
done
[ -d "$(dirname "$OUT")" ] || { echo 'OUT parent missing' >&2; exit 2; }
[ -x "$VENV/bin/python" ] || { echo 'VENV python missing' >&2; exit 2; }
[ -d "$MPI_SITE/mpi4py" ] || { echo "MPI_SITE has no mpi4py: $MPI_SITE" >&2; exit 2; }
NN=${SLURM_JOB_NUM_NODES:?}
CPUS=${SLURM_CPUS_PER_TASK:?}
THREADS=${THREADS:-$CPUS}
[ "$THREADS" -ge 1 ] && [ "$THREADS" -le "$CPUS" ] || { echo 'invalid thread budget' >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$MPI_SITE"
# Node-local numba cache: one rank per node, so no two ranks share it.
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/awesom-numba-${SLURM_JOB_ID:-local}
# Statistics scratch on rank 0's node, as in production-fit.sh (GP /tmp is RAM, 94 GB).
scratch=${SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-}}}
scratch_args=()
if [ -n "$scratch" ]; then scratch_args=(--scratch "$scratch"); fi

. "$REPO/examples/slurm_scripts/scale_out/common.sh"
PY="$VENV/bin/python"
LAUNCH=srun-pmix
kit_modules  # loads $MPI_SITE/MODULE, exports MPI4PY_LIBMPI from $MPI_SITE/LIBMPI
mkdir -p "$LAUNCH_DIR/probe"
# Every rank must keep its CPUS CPUs through MPI_Init (jobs 8413, 8444 got 1).
( OUT=$LAUNCH_DIR; affinity_preflight fit "$NN" 1 "$CPUS" none )
launch_cmd "$NN" 1 "$CPUS" none
echo "launch: ${CMD[*]}"
"${CMD[@]}" "$PY" -m mpi4py "$REPO/examples/grmhd-bbh/production_pool.py" \
  --fit "$LATTICE" "$SEED" --reuse-pool "$POOL" --out "$OUT" --mpi \
  --inputs "$INPUT_1" "$INPUT_2" --anchor "${ANCHOR:-94208}" --held-out "${HELD_OUT:-405504}" \
  --rows-per-snapshot "${ROWS_PER_SNAPSHOT:-500000000}" --rows-per-epoch "${ROWS_PER_EPOCH:-1000000000}" \
  --epochs "$EPOCHS" --r-min 8 --threads-per-seed "$THREADS" \
  --groups "${GROUPS_COUNT:-64}" ${scratch_args[@]+"${scratch_args[@]}"} "$@"
