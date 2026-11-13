#!/bin/bash
set -euo pipefail
: "${OUT:?set a fresh absolute output directory}"
: "${VENV:?set the environment directory}"
: "${REPO:?set the aweSOM checkout}"
: "${POOL:?set the prepared results directory}"
: "${INPUT_1:?set the first extraction}"
: "${INPUT_2:?set the second extraction}"
: "${LATTICE:?set the lattice, e.g. 100x100}"
: "${SEED:?set 42 or 43}"
case "$OUT" in /*) ;; *) echo 'OUT must be absolute' >&2; exit 2;; esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || { echo 'OUT exists' >&2; exit 2; }
[ -d "$(dirname "$OUT")" ] || { echo 'OUT parent missing' >&2; exit 2; }
[ -x "$VENV/bin/python" ] || { echo 'VENV python missing' >&2; exit 2; }
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:?}}
[ "$THREADS" -ge 1 ] && [ "$THREADS" -le "$SLURM_CPUS_PER_TASK" ] || { echo 'invalid thread budget' >&2; exit 2; }
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# The driver creates NUMBA_CACHE_DIR per worker under OUT, separately for each rank.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# Prefer node-local scratch. Driver falls back to the output filesystem if none exists.
scratch=${SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-}}}
scratch_args=()
if [ -n "$scratch" ]; then scratch_args=(--scratch "$scratch"); fi
# Default groups=64 preserves summation order. On 144 cores, 128 is reasonable
# but changes the last bits; opt in with GROUPS_COUNT=128.
"$VENV/bin/python" -u "$REPO/examples/grmhd-bbh/production_pool.py" \
  --inputs "$INPUT_1" "$INPUT_2" --anchor "${ANCHOR:-94208}" --held-out "${HELD_OUT:-405504}" \
  --out "$OUT" --reuse-pool "$POOL" --fit "$LATTICE" "$SEED" \
  --rows-per-snapshot "${ROWS_PER_SNAPSHOT:-500000000}" --rows-per-epoch "${ROWS_PER_EPOCH:-1000000000}" \
  --epochs "${EPOCHS:-60}" --r-min 8 --threads-per-seed "$THREADS" \
  --groups "${GROUPS_COUNT:-64}" ${scratch_args[@]+"${scratch_args[@]}"} "$@"
