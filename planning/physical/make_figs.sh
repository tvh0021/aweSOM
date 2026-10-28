#!/bin/sh
# Report figures. sh make_figs.sh [variants|mc|slice]...
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
export PYTHONPATH=$HERE/../../src
RES=/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical
FIG=$RES/figures
mkdir -p "$FIG"
cd "$HERE" || exit 1
S="-a0.5-s1-it180224"

variants() {
  nice -n 10 $PY figures.py variants --mc 0.1 --out "$FIG/fig3_variants_mc0.1.png" \
    --configs base5-N1e+06-rule$S nodiv4-N1e+06-rule$S cb_swap5-N1e+06-rule$S chic_swap5-N1e+06-rule$S \
      cb_add6-N1e+06-rule$S chic_add6-N1e+06-rule$S base5-N1e+06-half$S nodiv4-N1e+06-half$S \
      cb_swap5-N1e+06-half$S chic_swap5-N1e+06-half$S cb_add6-N1e+06-half$S chic_add6-N1e+06-half$S \
    --labels "base5 (baseline), rule" "nodiv4 (drop div v), rule" "cb_swap5 (div v to C_B), rule" \
      "chic_swap5 (div v to chi contrast), rule" "cb_add6 (+C_B), rule" "chic_add6 (+chi contrast), rule" \
      "base5, half lattice" "nodiv4, half" "cb_swap5, half" "chic_swap5, half" "cb_add6, half" "chic_add6, half"
}

mc() {
  nice -n 10 $PY figures.py mc --out "$FIG/fig2_score_vs_merge_cost.png" \
    --configs base5-N1e+06-rule$S nodiv4-N1e+06-rule$S cb_swap5-N1e+06-rule$S \
      base5-N1e+07-rule$S nodiv4-N1e+07-rule$S cb_swap5-N1e+07-rule$S \
    --labels "base5 1e6" "nodiv4 1e6" "cb_swap5 1e6" "base5 1e7" "nodiv4 1e7" "cb_swap5 1e7"
}

slice() {
  L=$RES/labels
  P=$RES/postproc
  T=$SLICE_TAG
  nice -n 10 $PY figures.py slice --it 135168 --sheet-cluster "$SLICE_SHEET" \
    --labels "$L/${T}__it135168_mc$SLICE_MC.npy" "$P/${T}__it135168_mc${SLICE_MC}_min27.npy" \
      "$P/${T}__it135168_mc${SLICE_MC}_maj1.npy" \
    --titles "raw labels" "min_size=27" "majority 3^3, 1 pass" \
    --suptitle "$SLICE_TITLE" --out "$FIG/fig1_slice_${T}_mc$SLICE_MC.png"
}

for k in "$@"; do $k; done
