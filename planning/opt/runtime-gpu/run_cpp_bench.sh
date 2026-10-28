#!/bin/sh
# Reproduce the C++ kernel benchmark: prep data, compile, run. Prints one JSON line per data set.
# usage: sh planning/opt/runtime-gpu/run_cpp_bench.sh [real|uniform6|sheet6|uniform10|sheet10 ...]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PY=${PY:-/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python}
OUT=${OUT:-/tmp/awesom_rtgpu_bench}
THREADS=${THREADS:-10}
OMPFLAGS=${OMPFLAGS:--Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp}
mkdir -p "$OUT"
/usr/bin/clang++ -O2 -std=c++17 -ffp-contract=off $OMPFLAGS -o "$OUT/bench_cpp" "$HERE/bench_cpp.cpp"
/usr/bin/clang++ -O2 -std=c++17 -ffp-contract=off -o "$OUT/test_fast" "$HERE/test_fast.cpp"
"$OUT/test_fast"
for set in ${@:-real}; do
  case $set in
    real) F=4; args="real"; NS=65536; NP=524288 ;;
    uniform*) F=${set#uniform}; args="uniform $F 524288"; NS=32768; NP=262144 ;;
    sheet*) F=${set#sheet}; args="sheet $F 524288"; NS=32768; NP=262144 ;;
  esac
  [ -f "$OUT/$set/W.bin" ] || PYTHONPATH="$ROOT/src" "$PY" "$HERE/prep_data.py" "$OUT/$set" $args >/dev/null
  (cd "$OUT/$set" && "$OUT/bench_cpp" W.bin X.bin ref_idx.bin ref_d2.bin $F $NS 5 $THREADS $NP)
done
