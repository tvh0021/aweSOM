#!/bin/bash
#SBATCH --job-name=so-sweep1
#SBATCH --dependency=singleton
#SBATCH --partition=gp-default
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=you@example.org
# One fit over four gp nodes, one MPI rank per node; every rank writes its own label shard
# under <tag>-<key>.parts/ and rank 0 publishes the manifests, statistics and other outputs.
# --job-name=so-sweep1 with --dependency=singleton: a second seed (and the scale-out
# kit's sweeps) waits until this job ends.
# Submit with --chdir and --output on a shared filesystem (see README.md):
#   sbatch --chdir="$WORKDIR" --output="$WORKDIR/%x-%j.out" production-fit-mpi-gp.sbatch
set -euo pipefail
: "${REPO:?set the checkout}"
exec bash "$REPO/examples/grmhd-bbh/scale-out/production-fit-mpi.sh" "$@"
