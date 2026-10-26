"""Prototype: exact warm-start BMU search with a node-node neighbour table.

For node b, nbr[b, :K] lists nodes sorted by distance from w_b, rad[b, k] is a
lower bound on that distance. Point x starts at b = previous point's BMU.
Triangle inequality: |x - w_i| >= |w_b - w_i| - |x - w_b|, so once
rad[b, k] > |x - w_b| + sqrt(best) (with margin) no later node can tie or win.
The scan keeps the reference contract (min float64 acc, ties to lowest index).
If the cutoff is not reached within K neighbours, or x is non-finite, the
point gets the full reference scan.
"""
import numpy as np
from numba import njit, prange

REL = 1e-9
ABS = 1e-150


def build_table(weights, K=None):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    n = w.shape[0]
    K = n if K is None else min(K, n)
    diff = w[:, None, :] - w[None, :, :]
    D = np.sqrt(np.einsum("ijf,ijf->ij", diff, diff))
    order = np.argsort(D, axis=1, kind="stable")[:, :K]
    # make sure b itself is first (D[b,b] == 0; duplicates also 0 -> any order fine)
    rad = np.take_along_axis(D, order, axis=1)
    # conservative: shrink by a relative margin so it is a strict lower bound
    rad = rad * (1.0 - 1e-12)
    itype = np.int16 if n < 32768 else np.int32
    return order.astype(itype), np.ascontiguousarray(rad, dtype=np.float64)


@njit(inline="always")
def _full_scan(weights, data, p):
    best = np.inf
    best_idx = 0
    for i in range(weights.shape[0]):
        acc = 0.0
        for f in range(weights.shape[1]):
            d = weights[i, f] - np.float64(data[p, f])
            acc += d * d
        if acc < best:
            best = acc
            best_idx = i
    return best_idx, best


@njit(parallel=True, cache=True)
def warm_kernel(weights, nbr, rad, data, out_idx, out_dist2, block):
    n = data.shape[0]
    F = weights.shape[1]
    K = nbr.shape[1]
    nblocks = (n + block - 1) // block
    for blk in prange(nblocks):
        start = blk * block
        stop = min(start + block, n)
        b = -1
        for p in range(start, stop):
            finite = True
            for f in range(F):
                if not np.isfinite(np.float64(data[p, f])):
                    finite = False
            if b < 0 or not finite:
                bi, bd = _full_scan(weights, data, p)
                out_idx[p] = bi
                out_dist2[p] = bd
                if finite:
                    b = bi
                continue
            acc = 0.0
            for f in range(F):
                d = weights[b, f] - np.float64(data[p, f])
                acc += d * d
            best = acc
            best_idx = b
            db = np.sqrt(acc)
            cut = (db + np.sqrt(best)) * (1.0 + REL) + ABS
            done = False
            for k in range(1, K):
                if rad[b, k] > cut:
                    done = True
                    break
                i = np.int64(nbr[b, k])
                acc = 0.0
                for f in range(F):
                    d = weights[i, f] - np.float64(data[p, f])
                    acc += d * d
                if acc < best or (acc == best and i < best_idx):
                    best = acc
                    best_idx = i
                    cut = (db + np.sqrt(best)) * (1.0 + REL) + ABS
            if not done and K < weights.shape[0]:
                bi, bd = _full_scan(weights, data, p)
                best_idx = bi
                best = bd
            out_idx[p] = best_idx
            out_dist2[p] = best
            b = best_idx


def best_match_nodes_warm(weights, data, table=None, K=None, block=4096):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if table is None:
        table = build_table(w, K)
    nbr, rad = table
    data = np.ascontiguousarray(data)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    warm_kernel(w, nbr, rad, data, idx, d2, block)
    return idx, d2
