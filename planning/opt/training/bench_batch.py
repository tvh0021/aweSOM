"""B3: batch trainer, JAX train_batch (float32 matmul BMU, all CPU devices) vs train_batch_kdtree
(exact float64 KD-tree BMU, numpy sums). Real fiducial plasma data, all 2,097,152 x 4 points,
64 x 37 lattice, same seed/init/schedule. Both are warmed up (JIT) on a 65,536-row subset first.
Reports per-epoch median (epochs 2..E), wall time, and QE/TE of both results on all points
(QE/TE computed with the exact compiled scan, bitwise equal to batch.quantization_error).
usage: python bench_batch.py [epochs=40] [rounds=1]
"""
import json, sys, os, time, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _common import load_fiducial
import aweSOM, jax
from aweSOM.batch import train_batch, train_batch_kdtree, quantization_error_fast, topographic_error_fast

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
fid = load_fiducial()
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
xdim, ydim = fid.xdim, fid.ydim
train_batch(data[:65536], xdim, ydim, epochs=2); train_batch_kdtree(data[:65536], xdim, ydim, epochs=2)
res = {"jax": [], "kdtree": []}
for r in range(rounds):
    for name, fn in (("jax", train_batch), ("kdtree", train_batch_kdtree)):
        t = time.perf_counter(); w, info = fn(data, xdim, ydim, epochs=epochs); wall = time.perf_counter() - t
        res[name].append(dict(wall_s=wall, epoch_median_s=float(np.median(info["epoch_time"][1:])),
                              qe=quantization_error_fast(w, data), te=topographic_error_fast(w, data, xdim, ydim),
                              n_dead=int((info["hits"] == 0).sum()), w=w))
out = {k: [{kk: vv for kk, vv in d.items() if kk != "w"} for d in v] for k, v in res.items()}
wj, wk = res["jax"][-1]["w"], res["kdtree"][-1]["w"]
out["max_abs_weight_diff"] = float(np.abs(wj - wk).max())
out["ratio_wall"] = float(np.median([d["wall_s"] for d in res["jax"]]) / np.median([d["wall_s"] for d in res["kdtree"]]))
out["ratio_epoch"] = float(np.median([d["epoch_median_s"] for d in res["jax"]]) / np.median([d["epoch_median_s"] for d in res["kdtree"]]))
out.update(epochs=epochs, n=data.shape[0], jax_devices=len(jax.devices()), aweSOM=aweSOM.__file__)
print(json.dumps(out))
