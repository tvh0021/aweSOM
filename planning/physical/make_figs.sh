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
      base5-N1e+07-rule$S nodiv4-N1e+07-rule$S cb_swap5-N1e+07-rule$S nodiv4-N1e+07-rule1e6$S cb_swap5-N1e+07-rule1e6$S \
    --labels "base5, 1e6 rows, 51x36" "nodiv4, 1e6 rows, 48x34" "cb_swap5, 1e6 rows, 51x36" "base5, 1e7 rows, 91x64" \
      "nodiv4, 1e7 rows, 86x61" "cb_swap5, 1e7 rows, 91x64" "nodiv4, 1e7 rows, 48x34" "cb_swap5, 1e7 rows, 51x36"
}

slice_one() {  # TAG MC SHEET_CLUSTER TITLE
  L=$RES/labels
  P=$RES/postproc
  nice -n 10 $PY figures.py slice --it 135168 --sheet-cluster "$3" --window=-23,23,-12.5,12.5 --zoom=-21,-9,-9,1 \
    --labels "$L/${1}__it135168_mc$2.npy" "$P/${1}__it135168_mc${2}_min27.npy" "$P/${1}__it135168_mc${2}_maj1.npy" \
    --titles "raw labels" "clean_labels(min_size=27)" "majority 3^3, 1 pass" \
    --suptitle "$4" --out "$FIG/fig1_slice_${1%%-a0.5*}_mc$2.png"
}

slice() {
  slice_one nodiv4-N1e+06-rule-a0.5-s1-seed0-it180224 0.1 3 "nodiv4, N=1e6, 48x34 (rule) lattice, merge cost 0.1 (it 135168, z = 0)"
}

for k in "$@"; do $k; done
