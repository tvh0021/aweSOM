"""Prototype: exact brute-force BMU scan, vectorised across points.

Same arithmetic per (point, node) as inference._bmu_kernel (float64 direct
differences, features summed in ascending order starting from 0.0, strict <
over nodes in ascending order), but the loop order is node-outer /
point-inner on a tile of B points stored feature-major, so LLVM can vectorise
across points. Each point still sees the nodes in ascending order with a
strict <, so labels and distances are bitwise those of the reference.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def tiled_kernel(weights, data, out_idx, out_dist2, B):
    n = data.shape[0]
    n_nodes, F = weights.shape
    ntiles = (n + B - 1) // B
    for t in prange(ntiles):
        start = t * B
        m = min(B, n - start)
        xT = np.zeros((F, B))
        for p in range(m):
            for f in range(F):
                xT[f, p] = np.float64(data[start + p, f])
        best = np.full(B, np.inf)
        bidx = np.zeros(B, dtype=np.int32)
        acc = np.empty(B)
        for i in range(n_nodes):
            w0 = weights[i, 0]
            for p in range(B):
                d = w0 - xT[0, p]
                acc[p] = 0.0 + d * d
            for f in range(1, F):
                wf = weights[i, f]
                for p in range(B):
                    d = wf - xT[f, p]
                    acc[p] = acc[p] + d * d
            ii = np.int32(i)
            for p in range(B):
                a = acc[p]
                c = a < best[p]
                best[p] = np.fmin(a, best[p])
                bidx[p] = ii if c else bidx[p]
        for p in range(m):
            out_idx[start + p] = bidx[p]
            out_dist2[start + p] = best[p]


def best_match_nodes_tiled(weights, data, B=64):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    data = np.ascontiguousarray(data)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    tiled_kernel(w, data, idx, d2, B)
    return idx, d2
