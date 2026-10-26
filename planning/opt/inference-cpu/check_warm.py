import json, sys, time
import numpy as np
from aweSOM import inference as inf
from common import load_real, interleaved
from warm import best_match_nodes_warm, build_table

w, x = load_real()
N = x.shape[0]
t0 = time.perf_counter(); tabK = {K: build_table(w, K) for K in (64, 256, 2368)}; tb = time.perf_counter() - t0
ref_i, ref_d = inf.best_match_nodes(w, x)
res = {"table_build_s_all3": tb}
for K, tab in tabK.items():
    i, d = best_match_nodes_warm(w, x, tab)
    res[f"K{K}_idx_diff"] = int((i != ref_i).sum())
    res[f"K{K}_d2_bitdiff"] = int((d.view(np.int64) != ref_d.view(np.int64)).sum())
fns = {"numba": lambda: inf.best_match_nodes(w, x), "kdtree": lambda: inf.best_match_nodes_kdtree(w, x)}
for K, tab in tabK.items():
    fns[f"warm_K{K}"] = (lambda tab=tab: best_match_nodes_warm(w, x, tab))
med, _ = interleaved(fns, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
res["pts_per_s"] = {k: N / v for k, v in med.items()}
print(json.dumps(res))
