"""Prototype: float32 screening scan + float64 certification (exact).

Pass (per tile of B points, node-outer / point-inner so it vectorises 4-wide
on NEON): float32 direct-difference squared distances, keeping per point the
smallest value s1 (node j1, strict <, ascending nodes) and the second smallest
s2 over all other nodes.

Certification: with u = 2**-24, for every node i the true Euclidean distance
r_i satisfies |sqrt(acc32_i) - r_i| <= e(x) := 2u(Wmax + |x|) + tiny (input
rounding + subtraction rounding), and acc32_i is within a relative
gamma = (F + 2) u of the exact sum of the rounded squares. If
    sqrt(s2 / (1 + g)) - e  >  (sqrt(s1 / (1 - g)) + e) * (1 + 1e-12)
then every node other than j1 is strictly farther than j1 by a relative margin
far above float64 rounding, so the float64 reference scan picks j1 too. The
float64 distance of j1 is then computed with the reference arithmetic. Points
failing the test (near-ties, non-finite, overflow) get the full float64
reference scan. Labels and distances are therefore bitwise those of
inference.best_match_nodes.
"""
import numpy as np
from numba import njit, prange

U32 = 2.0 ** -24


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
def screen_kernel(weights, w32, wmax, data, out_idx, out_dist2, B, out_nfb):
    n = data.shape[0]
    n_nodes, F = weights.shape
    g = (F + 2) * U32 * 1.01
    ntiles = (n + B - 1) // B
    nfb = np.zeros(ntiles, dtype=np.int64)
    for t in prange(ntiles):
        start = t * B
        m = min(B, n - start)
        xT = np.zeros((F, B), dtype=np.float32)
        for p in range(m):
            for f in range(F):
                xT[f, p] = np.float32(data[start + p, f])
        s1 = np.full(B, np.float32(np.inf))
        s2 = np.full(B, np.float32(np.inf))
        j1 = np.zeros(B, dtype=np.int32)
        acc = np.empty(B, dtype=np.float32)
        for i in range(n_nodes):
            w0 = w32[i, 0]
            for p in range(B):
                d = w0 - xT[0, p]
                acc[p] = d * d
            for f in range(1, F):
                wf = w32[i, f]
                for p in range(B):
                    d = wf - xT[f, p]
                    acc[p] = acc[p] + d * d
            ii = np.int32(i)
            for p in range(B):
                a = acc[p]
                b1 = s1[p]
                b2 = s2[p]
                c = a < b1
                c2 = a < b2
                n2 = b1 if c else (a if c2 else b2)
                s1[p] = np.fmin(a, b1)
                j1[p] = ii if c else j1[p]
                s2[p] = n2
        for p in range(m):
            q = start + p
            xn = 0.0
            for f in range(F):
                v = np.float64(data[q, f])
                xn += v * v
            e = 2.0 * U32 * (wmax + np.sqrt(xn)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2[p]) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1[p]) / (1.0 - g)) + e) * (1.0 + 1e-12)
            if lo2 > hi1:
                j = j1[p]
                a64 = 0.0
                for f in range(F):
                    d = weights[j, f] - np.float64(data[q, f])
                    a64 += d * d
                out_idx[q] = j
                out_dist2[q] = a64
            else:
                bi, bd = _full_scan(weights, data, q)
                out_idx[q] = bi
                out_dist2[q] = bd
                nfb[t] += 1
    out_nfb[0] = nfb.sum()


def best_match_nodes_screen32(weights, data, B=256, return_fallbacks=False):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e18:
        raise ValueError("finite, moderate weights required")
    w32 = w.astype(np.float32)
    wmax = float(np.sqrt((w * w).sum(1)).max())
    data = np.ascontiguousarray(data)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    nfb = np.zeros(1, dtype=np.int64)
    screen_kernel(w, w32, wmax, data, idx, d2, B, nfb)
    if return_fallbacks:
        return idx, d2, int(nfb[0])
    return idx, d2
