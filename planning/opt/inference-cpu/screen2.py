"""Two-kernel float32 screen + float64 certification (exact), with FMA contraction.

screen_direct: float32 direct differences (as screen32.py), fastmath contract.
screen_expand: float32 t_i = |w_i|^2 - 2 w_i . x via FMAs (|x|^2 dropped,
    it is the same for every node), fastmath contract.
certify: plain (no fastmath) float64 kernel. Accepts j1 if the float32 gap
    between the best and second-best screen values exceeds a rigorous bound on
    the screen's rounding error, else runs the full float64 reference scan.
    The returned node and squared distance are computed with the reference
    arithmetic, so labels and distances are bitwise those of
    inference.best_match_nodes.

Error bounds, u = 2**-24, applied with a 1.01 factor and a tiny absolute slack:
  direct: |sqrt(acc32_i) - r_i| <= 2u(Wmax + |x|), acc32 relative (F+2)u.
  expand: |t32_i - (r_i^2 - |x|^2)| <= (F+4)u(Wmax^2 + 2 Wmax |x|).
A 1e-12 relative margin on top keeps float64 rounding of the reference scan
from mattering.
"""
import numpy as np
from numba import njit, prange

U32 = 2.0 ** -24


@njit(parallel=True, fastmath={"contract"}, cache=True)
def screen_direct(w32, data, s1o, s2o, j1o, B):
    n = data.shape[0]
    n_nodes, F = w32.shape
    ntiles = (n + B - 1) // B
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
            s1o[start + p] = s1[p]
            s2o[start + p] = s2[p]
            j1o[start + p] = j1[p]


@njit(parallel=True, fastmath={"contract"}, cache=True)
def screen_expand(m32, n32, data, s1o, s2o, j1o, B):
    """m32 = -2 w (float32), n32 = |w|^2 (float32)."""
    n = data.shape[0]
    n_nodes, F = m32.shape
    ntiles = (n + B - 1) // B
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
            ni = n32[i]
            m0 = m32[i, 0]
            for p in range(B):
                acc[p] = ni + m0 * xT[0, p]
            for f in range(1, F):
                mf = m32[i, f]
                for p in range(B):
                    acc[p] = acc[p] + mf * xT[f, p]
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
            s1o[start + p] = s1[p]
            s2o[start + p] = s2[p]
            j1o[start + p] = j1[p]


@njit(parallel=True, cache=True)
def certify(weights, wmax, data, s1, s2, j1, expand, out_idx, out_dist2, out_nfb):
    n = data.shape[0]
    n_nodes, F = weights.shape
    nfb = np.zeros(n, dtype=np.uint8)
    for q in prange(n):
        xn2 = 0.0
        for f in range(F):
            v = np.float64(data[q, f])
            xn2 += v * v
        xn = np.sqrt(xn2)
        a1 = np.float64(s1[q])
        a2 = np.float64(s2[q])
        if expand:
            E = (F + 4) * U32 * 1.01 * (wmax * wmax + 2.0 * wmax * xn) + 1e-30
            eta = 1e-12 * (abs(a1) + xn2 + E) + 1e-300
            ok = (a2 - E) > (a1 + E + eta)
        else:
            g = (F + 2) * U32 * 1.01
            e = 2.0 * U32 * (wmax + xn) * 1.01 + 1e-18
            lo2 = np.sqrt(a2 / (1.0 + g)) - e
            hi1 = (np.sqrt(a1 / (1.0 - g)) + e) * (1.0 + 1e-12)
            ok = lo2 > hi1
        if ok:
            j = j1[q]
            acc = 0.0
            for f in range(F):
                d = weights[j, f] - np.float64(data[q, f])
                acc += d * d
            out_idx[q] = j
            out_dist2[q] = acc
        else:
            best = np.inf
            bi = 0
            for i in range(n_nodes):
                acc = 0.0
                for f in range(F):
                    d = weights[i, f] - np.float64(data[q, f])
                    acc += d * d
                if acc < best:
                    best = acc
                    bi = i
            out_idx[q] = bi
            out_dist2[q] = best
            nfb[q] = 1
    out_nfb[0] = nfb.sum()


def best_match_nodes_screen2(weights, data, expand=True, B=512, chunk=2**20, return_fallbacks=False):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e15:
        raise ValueError("finite, moderate weights required")
    wmax = float(np.sqrt((w * w).sum(1)).max())
    w32 = w.astype(np.float32)
    m32 = (-2.0 * w).astype(np.float32)
    n32 = (w * w).sum(1).astype(np.float32)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    nfb_total = 0
    c = min(chunk, max(n, 1))
    s1 = np.empty(c, dtype=np.float32); s2 = np.empty(c, dtype=np.float32); j1 = np.empty(c, dtype=np.int32)
    nfb = np.zeros(1, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blk = np.ascontiguousarray(data[start:stop])
        k = stop - start
        if expand:
            screen_expand(m32, n32, blk, s1[:k], s2[:k], j1[:k], B)
        else:
            screen_direct(w32, blk, s1[:k], s2[:k], j1[:k], B)
        certify(w, wmax, blk, s1[:k], s2[:k], j1[:k], expand, idx[start:stop], d2[start:stop], nfb)
        nfb_total += int(nfb[0])
    if return_fallbacks:
        return idx, d2, nfb_total
    return idx, d2
