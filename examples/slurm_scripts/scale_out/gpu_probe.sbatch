#!/bin/bash
#SBATCH --job-name=awesom-gpu-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
# For A100-40GB replace the preceding line with --constraint=a100-40gb.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
set -euo pipefail
KIT=${KIT:-$SLURM_SUBMIT_DIR}
REPO=${REPO:-$KIT/aweSOM-src}
OUT=${OUT:?Set OUT to a fresh absolute output directory}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-8}}
[[ "$OUT" = /* && ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'OUT must be fresh and absolute'; exit 2; }
[[ "$THREADS" -le "${SLURM_CPUS_PER_TASK:-8}" ]] || { echo 'THREADS exceeds allocation'; exit 2; }
[[ -f "$KIT/gpu_probe.py" && -f "$KIT/trained-8238.npy" && -d "$REPO/src/aweSOM" ]] || exit 2
# VENV defaults inside the fresh run. An explicit existing environment is rejected.
VENV=${VENV:-$OUT/venv}
[[ ! -e "$VENV" ]] || { echo 'VENV exists; choose a fresh environment'; exit 2; }
mkdir "$OUT"
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$OUT/numba-cache/rank-0" HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
mkdir -p "$NUMBA_CACHE_DIR"
{
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  free -g
  nproc
  lscpu | head || true
  df -h "${TMPDIR:-/tmp}" /tmp
  module avail 2>&1 | grep -i -E 'cuda|openmpi|mpi' || true
  command -v mpirun || true
  mpirun --version || true
} > "$OUT/hardware.log" 2>&1
python3 -m venv "$VENV"
"$VENV/bin/pip" install --report "$OUT/install.json" -e "$REPO[cuda]" pytest > "$OUT/install.log" 2>&1
"$VENV/bin/python" -c 'import importlib.util; print("mpi4py available:", importlib.util.find_spec("mpi4py") is not None)' >> "$OUT/hardware.log"
"$VENV/bin/python" "$KIT/gpu_probe.py" --kit "$KIT" --out "$OUT/results" > "$OUT/probe.log" 2>&1
"$VENV/bin/pytest" -q "$REPO/tests/gpu_test.py" > "$OUT/pytest.log" 2>&1
