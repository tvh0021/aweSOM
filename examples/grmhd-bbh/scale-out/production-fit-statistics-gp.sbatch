#!/bin/bash
#SBATCH --job-name=awesom-statistics-gp
#SBATCH --partition=gp-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
set -euo pipefail
: "${REPO:?set the checkout}"
exec bash "$REPO/examples/grmhd-bbh/scale-out/production-fit-statistics.sh" "$@"
