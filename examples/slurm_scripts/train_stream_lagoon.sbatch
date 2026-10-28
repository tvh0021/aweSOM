#!/bin/bash
#SBATCH -J awesom-stream
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -p lm-largemem
#SBATCH --nodelist=lm-n0001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH -t 4:00:00

# SOM training on up to ~10^10 rows on one CPU node (Green Prairies lm-n0001:
# 2 x 24 cores, 770 GB, no GPU) with aweSOM.stream, then the BMU of every row.
# Not submitted or run on the cluster yet; the same driver ran on a laptop on a
# small HDF5 file (see planning/benchmarks/bench_stream.py and the 2026-10-28
# scale report for the timings this is based on).
#
# The partition can also be "debug" (12 h limit); lm-largemem allows 24 h.
#
# Required:
#   AWESOM_FEATURES  HDF5 file with the (N, F) features, e.g.
#                    sbatch --export=ALL,AWESOM_FEATURES=/lagoon/$USER/features.h5 train_stream_lagoon.sbatch
# Optional (defaults in train_stream_lagoon.py):
#   AWESOM_DATASET (features)  AWESOM_OUT  XDIM YDIM (91 64)  EPOCHS (40)
#   MODE (sample | full)  SAMPLE_ROWS (1e8)  SAMPLE_FRACTION  SAMPLE_BLOCK (4096)
#   LOAD_TO_RAM (0/1)  SCALE (manual | none)  LABELS (1)  CHUNK  SEED
#
# Expected wall time for N = 10^10, F = 5, 91 x 64, 48 cores (extrapolated from
# a 10-core M1 Max, see the report): MODE=sample (10^8-row sample, 40 epochs,
# then labels for all rows) well under 1 h if reading the file keeps up
# (200 GB per labelling pass); MODE=full with all rows per epoch only a few
# epochs per hour, so use SAMPLE_FRACTION=0.01 there.

set -euo pipefail

: "${AWESOM_FEATURES:?set AWESOM_FEATURES to the HDF5 feature file}"
export AWESOM_OUT=${AWESOM_OUT:-$PWD/awesom_stream_${SLURM_JOB_ID:-local}}
export AWESOM_VENV=${AWESOM_VENV:-$HOME/git_repos/aweSOM/.venv}
export AWESOM_REPO=${AWESOM_REPO:-$HOME/git_repos/aweSOM}

# one numba worker per allocated core; no nested BLAS threads
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# numba compiles once and caches next to the sources; keep the cache on a writable path
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$AWESOM_OUT/numba_cache}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

mkdir -p "$AWESOM_OUT"
source "$AWESOM_VENV/bin/activate"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  features $AWESOM_FEATURES  out $AWESOM_OUT"
python -c "import numba, numpy, h5py, aweSOM; print('numba', numba.__version__, 'numpy', numpy.__version__, 'h5py', h5py.__version__, 'aweSOM', aweSOM.__file__)"

TIMER=""
if [ -x /usr/bin/time ] && /usr/bin/time -v true >/dev/null 2>&1; then TIMER="/usr/bin/time -v"; fi
$TIMER python "$AWESOM_REPO/examples/slurm_scripts/train_stream_lagoon.py" 2>&1 | tee "$AWESOM_OUT/log.txt"
