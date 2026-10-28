"""B4: SOMModel.from_lattice distance thresholds, distance_quantiles (argsort + 2368 np.percentile
calls) vs distance_quantiles_fast (counting sort + per-node sort, compiled). Real fiducial data
(2,097,152 x 4) and trained 64 x 37 lattice; BMUs from inference.best_match_nodes (exact).
usage: python bench_quantiles.py [reps=5]
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, numba
from _common import load_fiducial, median_interleaved
import aweSOM
from aweSOM.inference import best_match_nodes
from aweSOM.model import distance_quantiles, distance_quantiles_fast

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
fid = load_fiducial()
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
w = np.asarray(fid.lattice, np.float64)
idx, d2 = best_match_nodes(w, data); dist = np.sqrt(d2)
t = median_interleaved({"base": lambda: distance_quantiles(idx, dist, w.shape[0]),
                        "fast": lambda: distance_quantiles_fast(idx, dist, w.shape[0])}, reps=reps, warm=1)
med = {k: float(np.median(v)) for k, v in t.items()}
a, ra = distance_quantiles(idx, dist, w.shape[0]); b, rb = distance_quantiles_fast(idx, dist, w.shape[0])
# synthetic check with small and tied groups
rng = np.random.default_rng(0); si = rng.integers(0, 300, 20000).astype(np.int32); sd = np.round(rng.random(20000), 2)
sa, _ = distance_quantiles(si, sd, 310); sb, _ = distance_quantiles_fast(si, sd, 310)
print(json.dumps(dict(base_s=med["base"], fast_s=med["fast"], ratio=med["base"] / med["fast"], threads=numba.get_num_threads(),
    bitwise_equal_real=bool(np.array_equal(a, b, equal_nan=True) and ra == rb),
    bitwise_equal_synthetic=bool(np.array_equal(sa, sb, equal_nan=True)), times=t, aweSOM=aweSOM.__file__)))
