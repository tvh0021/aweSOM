#!/bin/bash
#SBATCH --job-name=so-rows
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# Row scaling on one node, 32 threads, both lattices: 1e7 and 1e8 rows (1e9 is
# rows_1e9.sbatch). Estimated 20 min.
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
LATTICES=${LATTICES:-"91x64 160x100"}
kit_init rows
env_probe
io_probe "${IO_TEST_GB:-20}"
for L in $LATTICES; do
  for N in 1e7 1e8; do
    try_run "$OUT/runs/$L-n$N-1x32" 1 1 32 none -- --n "$N" --lattice "$L" --tag "rows-$N"
  done
done
summarize_job
end_sweep
