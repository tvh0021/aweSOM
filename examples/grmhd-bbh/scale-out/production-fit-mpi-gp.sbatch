#!/bin/bash
#SBATCH --job-name=so-sweep1
#SBATCH --dependency=singleton
#SBATCH --partition=gp-default
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --chdir=/lagoon/trungha/bbh-output/for-awesom-2026-10-28
#SBATCH --output=/lagoon/trungha/bbh-output/for-awesom-2026-10-28/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# One fit over four gp nodes, one MPI rank per node; rank 0 writes every output.
# --job-name=so-sweep1 with --dependency=singleton: a second seed (and the scale-out
# kit's sweeps) waits until this job ends.
set -euo pipefail
: "${REPO:?set the checkout}"
exec bash "$REPO/examples/grmhd-bbh/scale-out/production-fit-mpi.sh" "$@"
