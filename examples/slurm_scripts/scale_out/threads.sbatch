#!/bin/bash
#SBATCH --job-name=so-threads
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# Thread scaling on one node, SCALE_N rows (default 1e8), one lattice per job
# (LATTICE=91x64 default, or 160x100): 1 rank x 32, 16, 8 threads, then 2 ranks x 16
# threads, one rank per socket (MPI). Estimated 30 min (91x64) and 80 min (160x100).
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
LATTICE=${LATTICE:-91x64}; N=${SCALE_N:-1e8}
kit_init "threads-$LATTICE"
env_probe
io_probe "${IO_TEST_GB:-20}"
# Launcher and affinity pre-flight first, so a bad binding stops the job in seconds.
MPI_OK=no
if pick_launcher 1 2; then
  affinity_preflight 2x16-socket 1 2 16 socket || exit 3
  MPI_OK=yes
fi
for t in 32 16 8; do
  try_run "$OUT/runs/$LATTICE-n$N-1x$t" 1 1 "$t" none -- --n "$N" --lattice "$LATTICE" --tag "threads-1x$t"
done
if [ "$MPI_OK" = yes ]; then
  try_run "$OUT/runs/$LATTICE-n$N-2x16-socket" 1 2 16 socket -- --n "$N" --lattice "$LATTICE" --tag threads-2x16-socket
  check_pinning "$OUT/runs/$LATTICE-n$N-2x16-socket" || true
else
  FAILS=$((FAILS + 1))
fi
$PY "$KIT/parity.py" --out "$OUT/parity" "$OUT/runs" || FAILS=$((FAILS + 1))
summarize_job
end_sweep
