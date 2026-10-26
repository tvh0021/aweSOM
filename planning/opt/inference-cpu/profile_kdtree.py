"""Where does time go in predict(backend="kdtree") on the real fiducial data?"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
import aweSOM
from aweSOM import inference as inf
from common import load_real, interleaved

w, x = load_real()
N = x.shape[0]
x32 = x.astype(np.float32)
cl = np.arange(w.shape[0]) % 7
tree = cKDTree(w)
idx = tree.query(x, k=1, workers=-1)[1].astype(np.int32)
d2 = np.empty(N)
fns = {
    "tree_build": lambda: cKDTree(w),
    "query_f64": lambda: tree.query(x, k=1, workers=-1),
    "query_f64_eps0_p2_dub": lambda: tree.query(x, k=1, workers=-1, distance_upper_bound=np.inf),
    "dist2_kernel": lambda: inf._dist2_kernel(w, x, idx, d2),
    "gather": lambda: cl[idx],
    "kdtree_full_f64": lambda: inf.best_match_nodes_kdtree(w, x),
    "kdtree_full_f32in": lambda: inf.best_match_nodes_kdtree(w, x32),
    "numba_full_f64": lambda: inf.best_match_nodes(w, x),
}
med, _ = interleaved(fns, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
print(json.dumps({"file": aweSOM.__file__, "N": N, "F": w.shape[1],
                  "median_s": med, "pts_per_s": {k: N / v for k, v in med.items()}}))
