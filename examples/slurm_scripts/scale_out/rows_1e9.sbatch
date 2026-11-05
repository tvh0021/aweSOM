#!/bin/bash
#SBATCH --job-name=so-rows1e9
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# 1e9 rows on one node, 32 threads, one lattice per job (LATTICE=91x64 default, or
# 160x100). Its weights are the parity reference of nodes_1e9.sbatch (REF_DIRS).
# Estimated 35 min (91x64) and 1.5-2 h (160x100). If the time limit stops it, resubmit
# with RESUME_RUN=<OUT of the first job>/runs/<run>: training continues from
# checkpoint.npz (bitwise equal to an uninterrupted run); logs go to a fresh OUT.
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
LATTICE=${LATTICE:-91x64}; N=1e9
kit_init "rows1e9-$LATTICE"
env_probe
io_probe "${IO_TEST_GB:-20}"
if [ -n "$RESUME_RUN" ]; then
  [ -f "$RESUME_RUN/checkpoint.npz" ] || { echo "no checkpoint.npz in RESUME_RUN=$RESUME_RUN" >&2; exit 2; }
  try_run "$RESUME_RUN" 1 1 32 none -- --n "$N" --lattice "$LATTICE" --tag rows-1e9 --resume
else
  try_run "$OUT/runs/$LATTICE-n$N-1x32" 1 1 32 none -- --n "$N" --lattice "$LATTICE" --tag rows-1e9
fi
summarize_job ${RESUME_RUN:+"$RESUME_RUN"}
end_sweep
