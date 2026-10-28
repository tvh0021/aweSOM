"""B2: quantization/topographic error, NumPy einsum blocks (baseline) vs compiled top-2 scan.

Timing on the first n points of the real fiducial plasma data with the fiducial trained
64 x 37 lattice; exactness on all 2,097,152 points against the values the baseline
functions produced in planning/runs/plasma-2026-10-26/results.json (fiducial entry),
and on the timing subset directly.
usage: python bench_qe_te.py [n=200000] [reps=5]
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, numba
from _common import load_fiducial, median_interleaved
import aweSOM
from aweSOM.batch import quantization_error, topographic_error, quantization_error_fast, topographic_error_fast

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
fid = load_fiducial()
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
w = np.asarray(fid.lattice, np.float64); xdim, ydim = fid.xdim, fid.ydim
sub = data[:n]
nt = numba.get_num_threads()

def base(): return quantization_error(w, sub), topographic_error(w, sub, xdim, ydim)
def fast(): return quantization_error_fast(w, sub), topographic_error_fast(w, sub, xdim, ydim)
def fast1():
    numba.set_num_threads(1)
    try: return fast()
    finally: numba.set_num_threads(nt)
t = median_interleaved({"base": base, "fast": fast, "fast_1thread": fast1}, reps=reps, warm=1)
med = {k: float(np.median(v)) for k, v in t.items()}
qb, tb = base(); qf, tf = fast()
tt = time.perf_counter(); qa, ta = quantization_error_fast(w, data), topographic_error_fast(w, data, xdim, ydim); t_full = time.perf_counter() - tt
print(json.dumps(dict(
    n_timing=n, threads=nt, base_s=med["base"], fast_s=med["fast"], fast_1thread_s=med["fast_1thread"],
    ratio=med["base"] / med["fast"], ratio_1thread=med["base"] / med["fast_1thread"],
    subset_qe_equal=qb == qf, subset_te_equal=tb == tf, subset_qe_reldiff=abs(qb - qf) / qb,
    full_fast_s=t_full, full_qe=qa, full_te=ta,
    full_qe_equal_to_results_json=qa == 0.3238546019656085, full_te_equal_to_results_json=ta == 0.007632255554199219,
    times=t, aweSOM=aweSOM.__file__), default=lambda o: o.item()))
