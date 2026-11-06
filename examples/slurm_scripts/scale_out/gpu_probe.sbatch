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
# GP-only survey: refuse to run on Tigris (no host-state reads there, user rule 2026-11-06).
case "$(printf '%s' "${SLURM_CLUSTER_NAME:-}" | tr '[:upper:]' '[:lower:]')|$PWD|${SLURM_SUBMIT_DIR:-}|${KIT:-}|${OUT:-}" in
  tigris\|*|*/shared/rc/*) echo "GP-only probe: refusing to run on Tigris" >&2; exit 2;;
esac
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
  nvidia-smi -L  # sysinfo-ok: GP-only probe, refuses on Tigris
  nvidia-smi topo -m  # sysinfo-ok: GP-only probe, refuses on Tigris
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv  # sysinfo-ok: GP-only probe, refuses on Tigris
  free -g  # sysinfo-ok: GP-only probe, refuses on Tigris
  nproc
  lscpu | head || true  # sysinfo-ok: GP-only probe, refuses on Tigris
  df -h "${TMPDIR:-/tmp}" /tmp  # sysinfo-ok: GP-only probe, refuses on Tigris
  module avail 2>&1 | grep -i -E 'cuda|openmpi|mpi' || true  # sysinfo-ok: GP-only probe, refuses on Tigris
  command -v mpirun || true
  mpirun --version || true
} > "$OUT/hardware.log" 2>&1
# CuPy compiles kernels with NVRTC and needs the CUDA headers: either a cuda
# module or the cupy-cuda12x[ctk] wheels that bundle them. Both are tried.
module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || true
PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "PYTHON=$PYTHON is older than 3.10; set PYTHON to a newer interpreter"; exit 2; }
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --report "$OUT/install.json" -e "$REPO[cuda]" 'cupy-cuda12x[ctk]' pytest > "$OUT/install.log" 2>&1
"$VENV/bin/python" -c 'import cupy; cupy.show_config()' >> "$OUT/hardware.log" 2>&1 || echo 'cupy.show_config failed' >> "$OUT/hardware.log"
"$VENV/bin/python" -c 'import importlib.util; print("mpi4py available:", importlib.util.find_spec("mpi4py") is not None)' >> "$OUT/hardware.log"
"$VENV/bin/python" "$KIT/gpu_probe.py" --kit "$KIT" --out "$OUT/results" > "$OUT/probe.log" 2>&1
"$VENV/bin/pytest" -q "$REPO/tests/gpu_test.py" > "$OUT/pytest.log" 2>&1
