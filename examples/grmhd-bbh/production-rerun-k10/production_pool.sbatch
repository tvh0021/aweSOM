#!/bin/bash
#SBATCH --job-name=production-pool-k10
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --exclude=lm-n0001
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
set -eo pipefail
DATA_KIT=${DATA_KIT:-/lagoon/trungha/bbh-output/for-awesom-2026-10-28}
KIT=${KIT:-$DATA_KIT/production-rerun-k10}
# For four separate node jobs, use ../scale-out/production-fit-gp.sbatch.
# groups=64 preserves rounding; --groups 128 can help 144-core nodes but changes last bits.
LATTICES='91x64 48x34'
case "${1:-}" in
  '') ;;
  --91x64-only) LATTICES='91x64' ;;
  *) echo 'usage: sbatch production_pool.sbatch [--91x64-only]' >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 2; }
OUT=${OUT:-$DATA_KIT/HHR_flux_production_pool_k10_r8_${SLURM_JOB_ID}}
# Validate before creating anything. Refuse existing output, even a failed run.
case "$OUT" in /*) ;; *) echo 'OUT must be absolute' >&2; exit 2;; esac
case "/$OUT/" in */../*|*/./*) echo 'OUT contains dot components' >&2; exit 2;; esac
p=$OUT
while [ "$p" != / ]; do
  [ ! -L "$p" ] || { echo "symlink in OUT: $p" >&2; exit 2; }
  p=$(dirname "$p")
done
anc=$OUT; while [ ! -d "$anc" ]; do anc=$(dirname "$anc"); done
resolved=$(cd "$anc" && pwd -P)
case "$resolved/" in "$HOME/"*|/lagoon/trungha/*) ;; *) echo 'OUT must resolve under HOME or /lagoon/trungha' >&2; exit 2;; esac
[ ! -e "$OUT" ] || { echo 'OUT exists; use a fresh output directory' >&2; exit 2; }
[ -d "$(dirname "$OUT")" ] || { echo "OUT parent does not exist: $(dirname "$OUT")" >&2; exit 2; }
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd -P)
STATUS_DIR="$OUT/status/${SLURM_JOB_ID:-local}-${SLURM_RESTART_COUNT:-0}"
completed=no
finish() {
  rc=$1; trap - EXIT TERM INT HUP; set +e
  mkdir -p "$STATUS_DIR"
  state=FAILED
  if [ "$rc" -eq 0 ] && [ "$completed" = yes ]; then state=VALIDATED; elif [ "$rc" -eq 0 ]; then rc=1; fi
  printf '%s\t%s\t%s\n' "$state" "$rc" "$(date -u +%FT%TZ)" > "$STATUS_DIR/final.tmp"
  mv "$STATUS_DIR/final.tmp" "$STATUS_DIR/final.tsv"
  echo "production-pool: $state exit=$rc"
  exit "$rc"
}
trap 'finish "$?"' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
mkdir -p "$STATUS_DIR"
printf 'STARTED\t%s\n' "$(date -u +%FT%TZ)" > "$STATUS_DIR/start.tsv"
cd "$OUT"
source "${VENV:-$DATA_KIT/check7-venv}/bin/activate"
export PYTHONPATH="$KIT/aweSOM-src/src:${ETK:-$HOME/etk-analysis}"
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$OUT/numba-cache" MPLCONFIGDIR="$OUT/mpl" TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-$OUT/tmp}}"
export HDF5_USE_FILE_LOCKING=FALSE
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:?} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u "$KIT/production_pool.py" --verify-source-only
mkdir -p "$TMPDIR"
INPUT_94208=${INPUT_94208:-$DATA_KIT/HHR_flux_it94208_region7/features_HHR_flux_94208.h5}
INPUT_98304=${INPUT_98304:-$DATA_KIT/HHR_flux_it98304_region7_anchor94208/features_HHR_flux_98304.h5}
# 405504 remains the named holdout. A third training snapshot needs a different holdout.
python -u "$KIT/production_pool.py" --inputs "$INPUT_94208" "$INPUT_98304" \
  --anchor "${ANCHOR:-94208}" --held-out "${HELD_OUT:-405504}" --out "$OUT/run" \
  --rows-per-snapshot "${ROWS_PER_SNAPSHOT:-10000000}" \
  --rows-per-epoch "${ROWS_PER_EPOCH:-20000000}" \
  --lattices $LATTICES --seeds 42 43 \
  --target-clusters 10 --r-min 8 \
  --seed-workers "${SEED_WORKERS:-2}" \
  --epochs "${EPOCHS:-40}" --chunk "${CHUNK:-2097152}" \
  --min-free-gb "${MIN_FREE_GB:-10}" > "$OUT/timing.log" 2>&1
completed=yes

