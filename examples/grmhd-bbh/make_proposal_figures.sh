#!/usr/bin/env bash
# Reproduce the proposal figure (current sheet between the BBH jets).
#   bash make_proposal_figures.sh [all|plot] [ITERATIONS]   (default: all "180224 135168")
# "plot" only redraws from the saved .npz files (a few seconds per iteration).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../.." && pwd)"
OUT=/Volumes/ExternalNVMe/awesom-results/2026-10-28/proposal-figure
AW_PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
ETK=/Volumes/ExternalNVMe/etk-analysis
ETK_PY=$ETK/.venv/bin/python
RUN=/Volumes/ExternalNVMe/simulation-data/michi-bbh/output-0003
INDEX=/Volumes/ExternalNVMe/som-proto-f5/idx3d_0003.json
SOM=$ETK/som_results/som_object.bbh_0003_180224-91-64-0.5-10000000-1u.pkl
TRAIN=/Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/features_bbh_0003-180224-N10000000.h5
MC=0.1
MODE="${1:-all}"
ITS="${2:-180224 135168}"

aw() { PYTHONPATH="$WT/src" NUMBA_NUM_THREADS=6 nice -n 10 "$AW_PY" "$HERE/proposal_figure.py" "$@"; }
etk() { PYTHONPATH="$ETK" nice -n 10 "$ETK_PY" "$@"; }

grid_of() {
  case "$1" in
    180224) echo /Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/grid_LR_flux_180224.h5 ;;
    135168) echo "$OUT/grid-135168/grid_LR_flux_135168.h5" ;;
  esac
}

for IT in $ITS; do
  TAG=it$IT
  GRID=$(grid_of $IT)
  if [[ "$MODE" == all ]]; then
    if [[ ! -f "$GRID" ]]; then  # only it 135168 needs one made (~2 min, 2.4 GB)
      # the lattice was trained on features transformed with the 180224 training
      # set's thresholds: another snapshot must be extracted with the same b2_min
      # and s_v, unclipped; proposal_figure.py clips to the training winsor limits
      read -r B2MIN SV < <("$AW_PY" - "$TRAIN" <<'PY'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    print(repr(float(f.attrs["b2_min"])), repr(float(f.attrs["s_v"])))
PY
)
      mkdir -p "$(dirname "$GRID")"
      (cd "$(dirname "$GRID")" && etk -m etkplot som-features "$INDEX" --it $IT -N 1e7 \
         --parfile "$RUN/parfile.par" -o . --grid-out "$(basename "$GRID")" \
         --b2-min "$B2MIN" --s-v "$SV" --winsor 0 1)
    fi
    aw labels --grid "$GRID" --train-features "$TRAIN" --som "$SOM" --merge-cost $MC \
       --ref-labels "$ETK/som_results/labels.bbh_0003_180224-91-64-0.5-10000000-1u-mc0.1.npy" \
       --out "$OUT" --tag $TAG
    etk "$HERE/sample_raw_planes.py" --index "$INDEX" --parfile "$RUN/parfile.par" --it $IT \
       -o "$OUT/raw_planes_$TAG.npz"
    aw sample --grid "$GRID" --raw-planes "$OUT/raw_planes_$TAG.npz" --out "$OUT" --tag $TAG
  fi
done

plot_it() {
  # panel (b) arrow labels are placed by hand per snapshot (plane coordinates, M)
  case "$1" in
    180224) aw plot --out "$OUT" --tag it180224 --elev 14 --azim -58 \
              --annotate "7.8,-15,-1,-21,sheet between jets" \
              --annotate "-2,0.6,-2,9,equatorial sheet" ;;
    135168) aw plot --out "$OUT" --tag it135168 --elev 14 --azim -58 \
              --annotate "2.2,6.6,2,17,sheet between jets" \
              --annotate "7,0.3,4,-16,equatorial sheet" ;;
  esac
}
for IT in $ITS; do plot_it $IT; done
