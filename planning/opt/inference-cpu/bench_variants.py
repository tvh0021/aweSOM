"""Usage: python bench_variants.py THREADS N_ROWS REPS [F]

Interleaved timings of the exact-scan variants against the reference numba
scan and the KD-tree, real fiducial data (F=4) or synthetic smooth fields
(F != 4). Prints one JSON line.
"""
import json, sys
import numpy as np
import numba
from aweSOM import inference as inf
from common import load_real, synthetic, interleaved
from tiled import best_match_nodes_tiled
from screen32 import best_match_nodes_screen32
from screen32r import best_match_nodes_screen32r
from screen2 import best_match_nodes_screen2

T = int(sys.argv[1]); NR = int(sys.argv[2]); REPS = int(sys.argv[3]); F = int(sys.argv[4]) if len(sys.argv) > 4 else 4
numba.set_num_threads(T)
if F == 4:
    w, x = load_real(); data = "real fiducial 128^3"
else:
    w, x = synthetic(F); data = f"synthetic smooth fields F={F}, lattice = 2368 sampled rows"
x = np.ascontiguousarray(x[:NR]); N = x.shape[0]
ref_i, ref_d = inf.best_match_nodes(w, x)
kd_i, _ = inf.best_match_nodes_kdtree(w, x, workers=T)
res = {"threads": T, "N": N, "F": F, "data": data, "kdtree_label_diff_vs_ref": int((kd_i != ref_i).sum())}
cands = {
    "tiled64": lambda: best_match_nodes_tiled(w, x, 256),
    "screen32": lambda: best_match_nodes_screen32(w, x, 512),
    "screen2_direct": lambda: best_match_nodes_screen2(w, x, expand=False),
}
for k, f in cands.items():
    out = f()
    res[k + "_label_diff"] = int((out[0] != ref_i).sum())
    if k.startswith("screen2"):
        res[k + "_fallbacks"] = best_match_nodes_screen2(w, x, expand=k.endswith("expand"), return_fallbacks=True)[2]
    res[k + "_d2_bitdiff"] = int((out[1].view(np.int64) != ref_d.view(np.int64)).sum())
fns = {"numba_ref": lambda: inf.best_match_nodes(w, x), "kdtree": lambda: inf.best_match_nodes_kdtree(w, x, workers=T)}
fns.update(cands)
med, samples = interleaved(fns, reps=REPS)
res["pts_per_s"] = {k: round(N / v) for k, v in med.items()}
res["ratio_vs_numba_ref"] = {k: round(med["numba_ref"] / v, 3) for k, v in med.items()}
cm = interleaved.cpu_median
res["cpu_s_per_Mpts"] = {k: round(v / N * 1e6, 4) for k, v in cm.items()}
res["cpu_ratio_vs_numba_ref"] = {k: round(cm["numba_ref"] / v, 3) for k, v in cm.items()}
res["cpu_ratio_vs_kdtree"] = {k: round(cm["kdtree"] / v, 3) for k, v in cm.items()}
res["ratio_vs_kdtree"] = {k: round(med["kdtree"] / v, 3) for k, v in med.items()}
print(json.dumps(res))
