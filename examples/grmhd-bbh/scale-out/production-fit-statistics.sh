#!/bin/bash
set -euo pipefail
: "${OUT:?set a fresh absolute output directory}"
: "${FIT_OUT:?set the deferred fit output directory}"
: "${VENV:?set the environment directory}"
: "${REPO:?set the aweSOM checkout}"
case "$OUT" in /*) ;; *) echo 'OUT must be absolute' >&2; exit 2;; esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || { echo 'OUT exists' >&2; exit 2; }
[ -d "$(dirname "$OUT")" ] || { echo 'OUT parent missing' >&2; exit 2; }
[ -d "$FIT_OUT" ] || { echo 'FIT_OUT missing' >&2; exit 2; }
[ -x "$VENV/bin/python" ] || { echo 'VENV python missing' >&2; exit 2; }
# Three concurrent feature workers need 24 bytes/row of scratch; one needs 16.
THREADS=${THREADS:-3}
[ "$THREADS" -ge 1 ] && [ "$THREADS" -le "${SLURM_CPUS_PER_TASK:?}" ] || { echo 'invalid thread budget' >&2; exit 2; }
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
scratch=${SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-}}}
scratch_args=()
if [ -n "$scratch" ]; then scratch_args=(--scratch "$scratch"); fi
"$VENV/bin/python" -u "$REPO/examples/grmhd-bbh/production_pool.py" \
  --statistics-only "$FIT_OUT" --out "$OUT" --threads-per-seed "$THREADS" \
  ${scratch_args[@]+"${scratch_args[@]}"} "$@"
