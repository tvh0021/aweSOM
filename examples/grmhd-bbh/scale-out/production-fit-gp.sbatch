#!/bin/bash
#SBATCH --job-name=awesom-fit-gp
#SBATCH --partition=debug
#SBATCH --exclude=lm-n0001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
set -euo pipefail
: "${REPO:?set the checkout}"
exec bash "$REPO/examples/grmhd-bbh/scale-out/production-fit.sh" "$@"
