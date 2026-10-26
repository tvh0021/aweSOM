"""Usage: python bench_kdtree_params.py REPS
cKDTree build options and the scaler cost on real fiducial data (F=4, 10 threads). One JSON line."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from aweSOM import inference as inf
from aweSOM.model import Scaler
from common import load_real, interleaved

w, x = load_real(); N = len(x)
trees = {f"ls{ls}_bal{int(b)}_cmp{int(c)}": cKDTree(w, leafsize=ls, balanced_tree=b, compact_nodes=c)
         for ls in (4, 16, 64) for b in (False, True) for c in (True,)}
ref = trees["ls16_bal1_cmp1"].query(x, workers=-1)[1]
diffs = {k: int((t.query(x[:200000], workers=-1)[1] != ref[:200000]).sum()) for k, t in trees.items()}
sc = Scaler(kind="manual", offset=np.full(4, 0.1), scale=np.full(4, 2.0))
fns = {k: (lambda t=t: t.query(x, workers=-1)) for k, t in trees.items()}
if sc is not None:
    fns["scaler_transform"] = lambda: sc.transform(x)
med, _ = interleaved(fns, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
print(json.dumps({"N": N, "pts_per_s": {k: round(N / v) for k, v in med.items()},
                  "ratio_vs_default_ls16_bal1": {k: round(med["ls16_bal1_cmp1"] / v, 3) for k, v in med.items()},
                  "idx_diff_first200k_vs_default": diffs}))
