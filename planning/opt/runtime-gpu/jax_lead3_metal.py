"""Lead 3: jax-metal (Apple GPU, float32): library float32 path and the exact prefilter
(candidate stage on Metal, f64 recheck on host with numba) vs numba float64.

Usage: python jax_lead3_metal.py [unused] [reps] [chunk]   (run with the .venv-metal interpreter)

Device stage (float32, direct differences, same order as the reference):
    d32[p, i] = sum_f fl32(fl32(w_if) - fl32(x_pf))^2   (sequential over f)
    m = min_i d32,  a = argmin_i d32 (first index),
    cnt = #{i : d32[p, i] <= thr(p)},  thr = (m + 2 * err(p)) * (1 + 2**-20)
Error bound (u = 2**-24, u64 = 2**-53, R_f = |x_f| + max_i |w_if|):
    for every node, |d32 - D64| <= err(p) with
    err = S * (4u' + 4u'^2 + g_F (1 + 2u')^2 + g64) + F * 2**-126,
    S = sum_f R_f^2, u' = u (1 + u), g_F = F u / (1 - F u), g64 = (F+1) u64 / (1 - (F+1) u64).
  (rounding of x and w to f32 and of the subtraction: |d32_f - d_f| <= 2u' R_f;
   squaring: |d32_f^2 - d_f^2| <= 4u' R_f^2 + 4u'^2 R_f^2; F roundings of
   square+sum: g_F * sum (R_f (1+2u'))^2; reference D64 vs exact: g64 * S;
   f32 flush-to-zero of tiny squares: F * 2**-126.) err itself is computed
   in f32 and inflated by 1.001, and thr by (1 + 2**-20), covering the
   handful of roundings in evaluating err and m + 2 err.
  => the reference winner j* (minimal D64, lowest index) has d32_j* <= m + 2 err,
     so cnt == 1 implies a == j*. Points with cnt != 1 are rescanned in full with
     the numba kernel; dist2 for all points is recomputed with the exact f64
     kernel (_dist2_kernel), so labels and dist2 must be bitwise equal to numba.
Prints one JSON line.
"""
import json
import os
import sys
import time

NDEV = 1
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 4096


import pickle

import numpy as np
import jax
import jax.numpy as jnp

from aweSOM.inference import _dist2_kernel, best_match_nodes, best_match_nodes_jax

lat = pickle.load(open("/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/"
                       "som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl", "rb"))
X = np.ascontiguousarray(lat.data_array, dtype=np.float64)
F = X.shape[1]
W = np.ascontiguousarray(np.asarray(lat.lattice, dtype=np.float64).reshape(-1, F))
N, M = X.shape[0], W.shape[0]
assert np.all(np.isfinite(W)) and np.all(np.isfinite(X))
ref_idx, ref_d2 = best_match_nodes(W, X)
devs = jax.local_devices()

u = 2.0 ** -24
up = u * (1 + u)
gF = F * u / (1 - F * u)
g64 = (F + 1) * 2.0 ** -53 / (1 - (F + 1) * 2.0 ** -53)
COEF = np.float32((4 * up + 4 * up * up + gF * (1 + 2 * up) ** 2 + g64) * 1.001)
TINY = np.float32(F * 2.0 ** -126)
WMAX = np.abs(W).max(axis=0).astype(np.float32)


@jax.jit
def prefilter(w, block, wmax):
    d = block[:, None, 0] - w[None, :, 0]
    acc = d * d
    for f in range(1, F):
        d = block[:, None, f] - w[None, :, f]
        acc = acc + d * d
    m = jnp.min(acc, axis=1)
    a = jnp.argmin(acc, axis=1).astype(jnp.int32)
    R = jnp.abs(block) + wmax[None, :]
    err = jnp.sum(R * R, axis=1) * COEF + TINY
    thr = (m + 2 * err) * np.float32(1 + 2.0 ** -20)
    cnt = jnp.sum(acc <= thr[:, None], axis=1).astype(jnp.int32)
    return a, cnt


def run_prefilter(devices, chunk=CHUNK, stats=None):
    X32 = X.astype(np.float32)
    a_all = np.empty(N, np.int32)
    c_all = np.empty(N, np.int32)
    wd = [jax.device_put(W.astype(np.float32), d) for d in devices]
    wm = [jax.device_put(WMAX, d) for d in devices]
    pad = np.zeros((chunk, F), np.float32)
    pend = []

    def col(it):
        s, e, a, c = it
        a_all[s:e] = np.asarray(a)[: e - s]
        c_all[s:e] = np.asarray(c)[: e - s]

    for k, s in enumerate(range(0, N, chunk)):
        e = min(s + chunk, N)
        b = X32[s:e]
        if e - s < chunk:
            pad[: e - s] = b
            pad[e - s:] = 0
            b = pad
        dv = k % len(devices)
        pend.append((s, e, *prefilter(wd[dv], jax.device_put(b, devices[dv]), wm[dv])))
        if len(pend) > len(devices):
            col(pend.pop(0))
    for it in pend:
        col(it)
    redo = np.flatnonzero(c_all != 1)
    if redo.size:
        a_all[redo], _ = best_match_nodes(W, X[redo])
    d2 = np.empty(N, np.float64)
    _dist2_kernel(W, X, a_all, d2)
    if stats is not None:
        stats["cnt_hist"] = {str(k): int(v) for k, v in zip(*np.unique(np.minimum(c_all, 9), return_counts=True))}
        stats["n_recheck"] = int(redo.size)
        stats["max_cnt"] = int(c_all.max())
    return a_all, d2


stats = {}
cases = {
    "numba": lambda: best_match_nodes(W, X),
    "lib_f32_metal_default": lambda: best_match_nodes_jax(W, X, precision="float32", devices=devs[:1]),
    "lib_f32_metal_chunk%d" % CHUNK: lambda: best_match_nodes_jax(W, X, chunk=CHUNK, precision="float32", devices=devs[:1]),
    "prefilter_1dev": lambda: run_prefilter(devs[:1]),
}
if len(devs) > 1:
    cases[f"lib_f64_{len(devs)}dev_chunk{CHUNK}"] = lambda: best_match_nodes_jax(W, X, chunk=CHUNK, devices=devs)
    cases[f"prefilter_{len(devs)}dev"] = lambda: run_prefilter(devs)

res = {}
for name, fn in cases.items():
    i, d = fn() if name != "prefilter_1dev" else run_prefilter(devs[:1], stats=stats)
    res[name] = {"label_mismatch": int(np.count_nonzero(i != ref_idx)),
                 "d2_mismatch": int(np.count_nonzero(np.asarray(d, np.float64) != ref_d2)), "t": []}
for _ in range(REPS):
    for name, fn in cases.items():
        t0 = time.perf_counter()
        fn()
        res[name]["t"].append(time.perf_counter() - t0)

# exact-tie count in the reference (other nodes with D64 equal to the min), on a sample
tb = float(np.median(res["numba"]["t"]))
out = {"lead": 3, "device": devs[0].platform, "ndev": len(devs), "jax": jax.__version__,
       "N": N, "F": F, "nodes": M, "chunk": CHUNK, "reps": REPS,
       "err_coef": float(COEF), "median_err_abs": float(np.median(((np.abs(X) + WMAX) ** 2).sum(1) * COEF)),
       "median_ref_d2": float(np.median(ref_d2)), **stats, "results": {}}
for name, r in res.items():
    t = float(np.median(r["t"]))
    out["results"][name] = {"pts_per_s": round(N / t), "vs_numba": round(tb / t, 3),
                            "label_mismatch": r["label_mismatch"], "d2_mismatch": r["d2_mismatch"]}
print(json.dumps(out))
