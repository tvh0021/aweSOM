#!/bin/bash
#SBATCH -J 2j1b1e0r
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH -t 01:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=ALL --mail-user=tvha@umass.edu
#SBATCH -p genx

# activate threading
# export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=true
# export HDF5_USE_FILE_LOCKING=FALSE

source [path_to_aweSOM]/aweSOM/.venv/bin/activate

python3 [path_to_aweSOM]/aweSOM/examples/plasma-turbulence/run_plasma_som.py --train 1000000 2097152 4000000 --ratio 0.4 0.6 0.8 1. --alpha_0 0.1 0.2 0.4
