"""Usage: python bench_leafsize.py REPS F N
cKDTree leafsize sweep through the library function (query + exact distance), 10 threads.
F=4 uses the real fiducial data, other F synthetic smooth fields. One JSON line."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from aweSOM import inference as inf
from common import load_real, synthetic, interleaved

REPS = int(sys.argv[1]); F = int(sys.argv[2]); NR = int(sys.argv[3])
w, x = load_real() if F == 4 else synthetic(F)
x = np.ascontiguousarray(x[:NR]); N = len(x)
ref = inf.best_match_nodes(w, x)[0]
out = {"F": F, "N": N, "data": "real" if F == 4 else "synthetic"}
fns = {}
for ls in (16, 32, 64, 128, 256):
    t = cKDTree(w, leafsize=ls)
    out[f"ls{ls}_label_diff_vs_numba"] = int((t.query(x, workers=-1)[1] != ref).sum())
    fns[f"ls{ls}"] = (lambda t=t: t.query(x, workers=-1))
med, _ = interleaved(fns, reps=REPS)
out["pts_per_s"] = {k: round(N / v) for k, v in med.items()}
out["ratio_vs_ls16"] = {k: round(med["ls16"] / v, 3) for k, v in med.items()}
print(json.dumps(out))
