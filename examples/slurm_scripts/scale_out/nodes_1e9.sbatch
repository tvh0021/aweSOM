#!/bin/bash
#SBATCH --job-name=so-nodes1e9
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# 1e9 rows on 2 and 4 nodes, one rank per node, 32 threads, both lattices. Submit only
# when the nodes.sbatch gate passes for every lattice it runs: GATE=<nodes OUT>/summary/gate.json
# (FORCE=1 skips the check). REF_DIRS = the rows_1e9 OUT dirs (1-node reference
# weights, colon separated); parity fails the job above 1e-4. Estimated 1.5 h.
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
LATTICES=${LATTICES:-"91x64 160x100"}; N=1e9
if [ "${FORCE:-0}" != 1 ]; then
  [ -f "$GATE" ] || { echo "set GATE=<nodes OUT>/summary/gate.json (or FORCE=1)" >&2; exit 2; }
  # Every lattice this job runs needs >= 70 % node efficiency at 2 and 4 nodes in the gate.
  "${VENV:-/lagoon/trungha/bbh-output/for-awesom-2026-10-28/check7-venv}/bin/python" \
    "${KIT:-$SLURM_SUBMIT_DIR}/summarize.py" --check-gate "$GATE" --lattices $LATTICES \
    || { echo "gate not met for LATTICES=$LATTICES; not running 1e9 (FORCE=1 overrides)" >&2; exit 2; }
fi
[ -n "$REF_DIRS" ] || { echo "set REF_DIRS to the rows_1e9 OUT dir(s), colon separated" >&2; exit 2; }
IFS=: read -r -a REFS <<< "$REF_DIRS"
kit_init nodes1e9
[ "${#NODES[@]}" -ge 4 ] || { echo "nodes_1e9.sbatch needs 4 nodes, got ${#NODES[@]}" >&2; exit 2; }
env_probe
io_probe "${IO_TEST_GB:-20}"
pick_launcher 4 4 || exit 1
# Every multi-rank launch first proves each rank keeps its 32 CPUs (job 8413 got 1).
for nn in 2 4; do affinity_preflight "nodes$nn" "$nn" 1 32 none || exit 3; done
for L in $LATTICES; do
  for nn in 2 4; do
    try_run "$OUT/runs/$L-n$N-nodes$nn" "$nn" 1 32 none -- --n "$N" --lattice "$L" --tag "nodes-$nn"
  done
done
$PY "$KIT/parity.py" --out "$OUT/parity" "$OUT/runs" --ref "${REFS[@]}" || FAILS=$((FAILS + 1))
summarize_job "${REFS[@]}"
end_sweep
