#!/bin/bash
#SBATCH --job-name=awesom-fit-grace
# Set the site partition at submission: sbatch --partition=SITE_PARTITION ...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=you@example.org
set -euo pipefail
: "${REPO:?set the checkout}"
exec bash "$REPO/examples/grmhd-bbh/scale-out/production-fit.sh" "$@"
