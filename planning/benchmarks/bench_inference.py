"""Usage: python bench_inference.py [N] [F]

Benchmark candidate SOM inference (best-matching-unit) kernels against the current one."""

import sys
import time
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from aweSOM import Lattice

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_097_152
F = int(sys.argv[2]) if len(sys.argv) > 2 else 4
XDIM, YDIM = 64, 37  # fiducial plasma lattice

rng = np.random.default_rng(0)
data64 = rng.normal(size=(N, F)) * 0.5
lattice64 = rng.normal(size=(XDIM * YDIM, F)) * 0.5
data32 = data64.astype(np.float32)
lattice32 = lattice64.astype(np.float32)


@njit(parallel=True, fastmath=True, cache=True)
def bmu_fused(lattice, obs):
    n_nodes, n_feat = lattice.shape
    out = np.empty(obs.shape[0], dtype=np.int32)
    for i in prange(obs.shape[0]):
        best = np.inf
        best_j = 0
        for j in range(n_nodes):
            s = 0.0
            for f in range(n_feat):
                d = lattice[j, f] - obs[i, f]
                s += d * d
            if s < best:
                best = s
                best_j = j
        out[i] = best_j
    return out


def bmu_matmul(lattice, obs, chunk=1 << 16):
    # argmin_j ||x - w_j||^2 = argmin_j (||w_j||^2 - 2 x.w_j)
    w2 = np.einsum("ij,ij->i", lattice, lattice)
    out = np.empty(obs.shape[0], dtype=np.int32)
    for s in range(0, obs.shape[0], chunk):
        d = w2 - 2.0 * (obs[s : s + chunk] @ lattice.T)
        out[s : s + chunk] = np.argmin(d, axis=1)
    return out


def bmu_kdtree(lattice, obs):
    tree = cKDTree(lattice)
    _, idx = tree.query(obs, k=1, workers=-1)
    return idx.astype(np.int32)


def timeit(label, fn, ref=None, repeat=2):
    fn()  # warm up / JIT compile
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    res = np.asarray(res).ravel().astype(np.int64)
    agree = "" if ref is None else f"  agreement={np.mean(res == ref):.6f}"
    print(f"{label:<34s} {best:8.3f} s  {N / best / 1e6:8.2f} Mpts/s{agree}", flush=True)
    return res


@njit(parallel=True, fastmath=True, cache=True)
def bmu_simd(lattice_T, w2, obs, block=256):
    # lattice_T is [F, nodes] so the inner loop runs contiguously over nodes and vectorizes
    n_feat, n_nodes = lattice_T.shape
    n = obs.shape[0]
    out = np.empty(n, dtype=np.int32)
    n_blocks = (n + block - 1) // block
    for b in prange(n_blocks):
        d = np.empty(n_nodes, dtype=lattice_T.dtype)
        for i in range(b * block, min((b + 1) * block, n)):
            for j in range(n_nodes):
                d[j] = w2[j]
            for f in range(n_feat):
                xf = -2.0 * obs[i, f]
                for j in range(n_nodes):
                    d[j] += xf * lattice_T[f, j]
            out[i] = np.argmin(d)
    return out


print(f"N={N:,}  F={F}  nodes={XDIM * YDIM}")
ref = timeit("current Lattice.best_match (f64)", lambda: Lattice.best_match(lattice64, data64), repeat=1)
timeit("fused numba loop (f64)", lambda: bmu_fused(lattice64, data64), ref)
timeit("fused numba loop (f32)", lambda: bmu_fused(lattice32, data32), ref)
timeit("chunked matmul (f32)", lambda: bmu_matmul(lattice32, data32), ref)
timeit("scipy cKDTree (f64)", lambda: bmu_kdtree(lattice64, data64), ref)
for lat, dat, tag in ((lattice64, data64, "f64"), (lattice32, data32, "f32")):
    lt = np.ascontiguousarray(lat.T)
    w2 = np.einsum("ij,ij->i", lat, lat)
    timeit(f"SIMD numba, nodes-inner ({tag})", lambda: bmu_simd(lt, w2, dat), ref)
