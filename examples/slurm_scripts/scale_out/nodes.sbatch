#!/bin/bash
#SBATCH --job-name=so-nodes
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# Node scaling, SCALE_N rows (default 1e8), both lattices: 1, 2 and 4 nodes, one rank
# per node, 32 threads. Then parity of the 2- and 4-node weights against the 1-node
# weights (the job fails above 1e-4) and summary/gate.json (>= 70 % node efficiency at
# 2 and 4 nodes releases nodes_1e9.sbatch). The 2026-10-28 sinfo listed one debug node;
# see README for the partition override. Estimated 30 min.
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
N=${SCALE_N:-1e8}; LATTICES=${LATTICES:-"91x64 160x100"}
kit_init nodes
[ "${#NODES[@]}" -ge 4 ] || { echo "nodes.sbatch needs 4 nodes, got ${#NODES[@]}" >&2; exit 2; }
env_probe
io_probe "${IO_TEST_GB:-20}"
pick_launcher 4 4 || exit 1
for L in $LATTICES; do
  try_run "$OUT/runs/$L-n$N-nodes1" 1 1 32 none -- --n "$N" --lattice "$L" --tag nodes-1
  for nn in 2 4; do
    try_run "$OUT/runs/$L-n$N-nodes$nn" "$nn" 1 32 none -- --n "$N" --lattice "$L" --tag "nodes-$nn"
  done
done
$PY "$KIT/parity.py" --out "$OUT/parity" "$OUT/runs" || FAILS=$((FAILS + 1))
summarize_job
end_sweep
