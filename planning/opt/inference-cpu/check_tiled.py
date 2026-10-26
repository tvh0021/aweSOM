import json, sys
import numpy as np
from aweSOM import inference as inf
from common import load_real, interleaved
from tiled import best_match_nodes_tiled

w, x = load_real()
N = x.shape[0]
ref_i, ref_d = inf.best_match_nodes(w, x)
res = {}
for B in (16, 64, 256):
    i, d = best_match_nodes_tiled(w, x, B)
    res[f"B{B}_idx_diff"] = int((i != ref_i).sum()); res[f"B{B}_d2_bitdiff"] = int((d.view(np.int64) != ref_d.view(np.int64)).sum())
fns = {"numba": lambda: inf.best_match_nodes(w, x), "kdtree": lambda: inf.best_match_nodes_kdtree(w, x)}
for B in (16, 64, 256):
    fns[f"tiled_B{B}"] = (lambda B=B: best_match_nodes_tiled(w, x, B))
med, _ = interleaved(fns, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
res["pts_per_s"] = {k: N / v for k, v in med.items()}
print(json.dumps(res))
