"""Usage: python bench_claims.py F N REPS [THREADS]

Head-to-head of the CPU inference paths, interleaved, median of REPS after one
warm-up (JIT excluded). F=4: real fiducial data (2,097,152 x 4 scaled features,
trained 64x37 lattice); other F: synthetic smooth Gaussian fields on a 128^3 box,
lattice = 2368 sampled rows. Prints one JSON line.

  numba_ref      inference.best_match_nodes (the bitwise reference)
  kdtree_ls16    inference.best_match_nodes_kdtree(leafsize=16)  (old default)
  kdtree_ls64    inference.best_match_nodes_kdtree(leafsize=64)  (new default)
  kdtree_k2cert  leafsize-64 tree, k=2, accept the nearest node only if the
                 tree's second distance exceeds it by 1e-9 relative, else the
                 full reference scan (prototype here, not in the library)
  screen         inference.best_match_nodes_screen (float32 screen + float64 certificate)
"""
import json, sys
import numpy as np
import numba
from scipy.spatial import cKDTree
import aweSOM
from aweSOM import inference as inf
from common import load_real, synthetic, interleaved

F = int(sys.argv[1]); NR = int(sys.argv[2]); REPS = int(sys.argv[3])
T = int(sys.argv[4]) if len(sys.argv) > 4 else 10
numba.set_num_threads(T)
w, x = load_real() if F == 4 else synthetic(F)
x = np.ascontiguousarray(x[:NR]); N = len(x)


def kdtree_k2cert(w, x):
    tree = cKDTree(w, leafsize=64)
    d, i = tree.query(x, k=2, workers=T)
    idx = i[:, 0].astype(np.int32)
    ok = d[:, 1] > d[:, 0] * (1 + 1e-9) + 1e-150
    bad = ~ok
    nb = int(bad.sum())
    if nb:
        idx[bad], _ = inf.best_match_nodes(w, x[bad])
    d2 = np.empty(len(x))
    inf._dist2_kernel(w, x, idx, d2)
    kdtree_k2cert.fallbacks = nb
    return idx, d2


ref = inf.best_match_nodes(w, x)
fns = {
    "numba_ref": lambda: inf.best_match_nodes(w, x),
    "kdtree_ls16": lambda: inf.best_match_nodes_kdtree(w, x, workers=T, leafsize=16),
    "kdtree_ls64": lambda: inf.best_match_nodes_kdtree(w, x, workers=T, leafsize=64),
    "kdtree_k2cert": lambda: kdtree_k2cert(w, x),
    "screen": lambda: inf.best_match_nodes_screen(w, x),
}
res = {"aweSOM": aweSOM.__file__, "F": F, "N": N, "threads": T,
       "data": "real fiducial" if F == 4 else "synthetic smooth fields, lattice = sampled rows"}
for k, f in fns.items():
    i, d = f()
    res[f"{k}_label_diff"] = int((i != ref[0]).sum())
    res[f"{k}_d2_bitdiff"] = int((d.view(np.int64) != ref[1].view(np.int64)).sum())
res["screen_fallbacks"] = inf.best_match_nodes_screen(w, x, return_fallbacks=True)[2]
res["kdtree_k2cert_fallbacks"] = kdtree_k2cert.fallbacks
med, _ = interleaved(fns, reps=REPS)
cm = interleaved.cpu_median
res["pts_per_s"] = {k: round(N / v) for k, v in med.items()}
res["ratio_vs_numba_ref"] = {k: round(med["numba_ref"] / v, 3) for k, v in med.items()}
res["ratio_vs_kdtree_ls16"] = {k: round(med["kdtree_ls16"] / v, 3) for k, v in med.items()}
res["cpu_ratio_vs_kdtree_ls16"] = {k: round(cm["kdtree_ls16"] / v, 3) for k, v in cm.items()}
print(json.dumps(res))
