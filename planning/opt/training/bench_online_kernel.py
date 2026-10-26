"""B1: online step kernel, reference (9f29e9b) vs candidate (transposed lattice + Gaussian table).

Real fiducial plasma data (2,097,152 x 4), 64 x 37 lattice, fresh-run schedule with
n_steps steps (same proportions as the 2M-step production run). Single thread (the
kernel is serial). Also checks bitwise equality of the final lattice, and for F = 6, 8, 10
on synthetic data.
usage: python bench_online_kernel.py [n_steps=100000] [reps=5]
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _common import load_fiducial, median_interleaved, som_train_steps_ref, online_setup
import aweSOM
from aweSOM.som import _som_train_steps

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
fid = load_fiducial()
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
xdim, ydim = fid.xdim, fid.ydim
lat0, xk, alphas, nsizes = online_setup(data, xdim, ydim, fid.alpha_0, n_steps)

out = {}
def run(kernel):
    lat = lat0.copy(); kernel(lat, xk, xdim, alphas, nsizes, 0, n_steps); return lat
fns = {"ref": lambda: run(som_train_steps_ref), "cand": lambda: run(_som_train_steps)}
t = median_interleaved(fns, reps=reps, warm=1)
med = {k: float(np.median(v)) for k, v in t.items()}
out["F4_real"] = dict(ref_us_per_step=1e6 * med["ref"] / n_steps, cand_us_per_step=1e6 * med["cand"] / n_steps,
                      ratio=med["ref"] / med["cand"], bitwise_equal=bool(np.array_equal(run(som_train_steps_ref), run(_som_train_steps))),
                      ref_times=t["ref"], cand_times=t["cand"])

# other feature counts on synthetic Gaussian data (fewer steps)
rng = np.random.default_rng(1)
for F in (6, 8, 10):
    d = rng.standard_normal((200_000, F))
    m = 50_000
    l0, xs, al, ns = online_setup(d, xdim, ydim, fid.alpha_0, m)
    def runF(kernel):
        lat = l0.copy(); kernel(lat, xs, xdim, al, ns, 0, m); return lat
    tt = median_interleaved({"ref": lambda: runF(som_train_steps_ref), "cand": lambda: runF(_som_train_steps)}, reps=reps, warm=1)
    mm = {k: float(np.median(v)) for k, v in tt.items()}
    out[f"F{F}_synthetic"] = dict(ref_us_per_step=1e6 * mm["ref"] / m, cand_us_per_step=1e6 * mm["cand"] / m,
                                  ratio=mm["ref"] / mm["cand"], bitwise_equal=bool(np.array_equal(runF(som_train_steps_ref), runF(_som_train_steps))))
out["n_steps"] = n_steps; out["aweSOM"] = aweSOM.__file__
print(json.dumps(out))
