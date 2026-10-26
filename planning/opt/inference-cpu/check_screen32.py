import json, sys
import numpy as np
from aweSOM import inference as inf
from common import load_real, interleaved
from tiled import best_match_nodes_tiled
from screen32 import best_match_nodes_screen32

w, x = load_real()
N = x.shape[0]
x32 = x.astype(np.float32)
ref_i, ref_d = inf.best_match_nodes(w, x)
ref_i32, ref_d32 = inf.best_match_nodes(w, x32)
res = {}
for B in (128, 256, 512):
    i, d, nfb = best_match_nodes_screen32(w, x, B, True)
    res[f"B{B}_idx_diff"] = int((i != ref_i).sum()); res[f"B{B}_d2_bitdiff"] = int((d.view(np.int64) != ref_d.view(np.int64)).sum()); res[f"B{B}_fallbacks"] = nfb
i, d, nfb = best_match_nodes_screen32(w, x32, 256, True)
res["f32in_idx_diff"] = int((i != ref_i32).sum()); res["f32in_d2_bitdiff"] = int((d.view(np.int64) != ref_d32.view(np.int64)).sum()); res["f32in_fallbacks"] = nfb
fns = {"numba": lambda: inf.best_match_nodes(w, x), "kdtree": lambda: inf.best_match_nodes_kdtree(w, x),
       "tiled64_B256": lambda: best_match_nodes_tiled(w, x, 256)}
for B in (128, 256, 512):
    fns[f"screen32_B{B}"] = (lambda B=B: best_match_nodes_screen32(w, x, B))
med, _ = interleaved(fns, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
res["pts_per_s"] = {k: N / v for k, v in med.items()}
print(json.dumps(res))
