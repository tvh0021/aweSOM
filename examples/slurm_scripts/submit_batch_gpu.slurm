#!/bin/bash
#SBATCH -J aweSOM-batch-gpu
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH -t 2:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 8
#SBATCH --mem=200G

# Multi-GPU batch SOM training with aweSOM.batch, one process per GPU.
#
# UNTESTED ON GPUs: written without GPU access. What has been run: the same
# Python code path (jax.distributed.initialize + process-local rows +
# train_batch) with two CPU processes on one machine. Not run: CUDA, SLURM
# auto-detection inside jax.distributed.initialize, the GPU binding below,
# reading the HDF5 slices with h5py, and more than one node (raise --nodes;
# nothing else changes). Check the first run's log for the device count.
#
# Install: pip install 'aweSOM[gpu]'   (adds jax[cuda12])
#
# Inputs (edit or export before sbatch):
#   FEATURES_FILE  HDF5 file with a 2-D float dataset (N, F) of features
#   DATASET        dataset name inside the file
#   XDIM, YDIM     lattice dimensions
#   EPOCHS         number of batch epochs
#   OUTPUT         .npy file for the trained weights (written by process 0)

export FEATURES_FILE=${FEATURES_FILE:-features.h5}
export DATASET=${DATASET:-features}
export XDIM=${XDIM:-64}
export YDIM=${YDIM:-37}
export EPOCHS=${EPOCHS:-40}
export OUTPUT=${OUTPUT:-lattice_batch_${XDIM}x${YDIM}.npy}

export PYTHONDONTWRITEBYTECODE=true
# bitwise reproducible scatter-adds (atomics are otherwise order dependent)
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"
# let JAX allocate on demand instead of pre-allocating 75% of each GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
module load cuda # if your cluster needs it

# srun forwards stdin only to task 0, so write the driver to a file first
DRIVER=${SLURM_SUBMIT_DIR:-.}/train_batch_${SLURM_JOB_ID:-local}.py
cat > "$DRIVER" <<'EOF'
import os

import h5py
import jax
import numpy as np

# One process per GPU. If SLURM exposes all GPUs of the node to every task,
# pin this process to the GPU with its node-local rank; if SLURM already
# binds one GPU per task, let JAX use the single visible device.
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
local_ids = None if visible and "," not in visible else [int(os.environ["SLURM_LOCALID"])]
jax.distributed.initialize(local_device_ids=local_ids)  # coordinator etc. from SLURM env

from aweSOM.batch import train_batch

rank = int(os.environ["SLURM_PROCID"])
size = int(os.environ["SLURM_NTASKS"])
with h5py.File(os.environ["FEATURES_FILE"], "r") as f:
    dset = f[os.environ["DATASET"]]
    n = dset.shape[0]
    start, stop = n * rank // size, n * (rank + 1) // size
    local = dset[start:stop].astype(np.float32)

if rank == 0:
    print(f"processes {jax.process_count()}, global devices {jax.device_count()}, "
          f"rows {n}, features {local.shape[1]}", flush=True)

# train_batch treats `local` as this process's rows and assembles the global
# array with jax.make_array_from_process_local_data; every process must call it.
weights, info = train_batch(
    local,
    int(os.environ["XDIM"]),
    int(os.environ["YDIM"]),
    epochs=int(os.environ["EPOCHS"]),
    verbose=(rank == 0),
)

if rank == 0:
    np.save(os.environ["OUTPUT"], weights)
    print(f"QE after each epoch: {np.round(info['qe'], 5).tolist()}", flush=True)
    print(f"seconds per epoch: {np.round(info['epoch_time'], 3).tolist()}", flush=True)
EOF

srun python "$DRIVER"
