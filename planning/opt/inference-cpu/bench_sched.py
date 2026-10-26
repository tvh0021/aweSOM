"""Usage: python bench_sched.py REPS
Screen kernel scheduling on real F=4 data: numba threads 8 vs 10, default vs
numba.set_parallel_chunksize(4) (dynamic tiles), against the leafsize-64 KD-tree. One JSON line."""
import json, sys
import numpy as np, numba
from aweSOM import inference as inf
from common import load_real, interleaved

w, x = load_real(); N = len(x)


def run(T, cs):
    def f():
        numba.set_num_threads(T)
        old = numba.set_parallel_chunksize(cs)
        try:
            return inf.best_match_nodes_screen(w, x)
        finally:
            numba.set_parallel_chunksize(old)
    return f


fns = {"kdtree_ls64_w10": lambda: inf.best_match_nodes_kdtree(w, x, workers=10),
       "kdtree_ls64_w8": lambda: inf.best_match_nodes_kdtree(w, x, workers=8)}
for T in (8, 10):
    for cs in (0, 4):
        fns[f"screen_T{T}_cs{cs}"] = run(T, cs)
med, _ = interleaved(fns, reps=int(sys.argv[1]))
print(json.dumps({"N": N, "pts_per_s": {k: round(N / v) for k, v in med.items()},
                  "ratio_vs_kdtree_ls64_w10": {k: round(med["kdtree_ls64_w10"] / v, 3) for k, v in med.items()}}))
