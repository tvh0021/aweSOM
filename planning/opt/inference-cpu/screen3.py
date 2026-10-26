"""screen2 with the per-node point loop fused and the feature loop unrolled (code-generated per F).

Certification kernel is screen2.certify (plain float64, no fastmath), so the
exactness argument of screen2.py applies unchanged.
"""
import numpy as np
from numba import njit, prange

from screen2 import certify

_T = '''
def kern(W, n32, data, s1o, s2o, j1o, B):
    n = data.shape[0]
    n_nodes = W.shape[0]
    ntiles = (n + B - 1) // B
    for t in prange(ntiles):
        start = t * B
        m = min(B, n - start)
        xT = np.zeros(({F}, B), dtype=np.float32)
        for p in range(m):
            for f in range({F}):
                xT[f, p] = np.float32(data[start + p, f])
{XR}
        s1 = np.full(B, np.float32(np.inf))
        s2 = np.full(B, np.float32(np.inf))
        j1 = np.zeros(B, dtype=np.int32)
        for i in range(n_nodes):
{WL}
            ii = np.int32(i)
            for p in range(B):
{BODY}
                b1 = s1[p]
                c = a < b1
                s2[p] = min(s2[p], max(b1, a))
                s1[p] = min(a, b1)
                j1[p] = ii if c else j1[p]
        for p in range(m):
            s1o[start + p] = s1[p]
            s2o[start + p] = s2[p]
            j1o[start + p] = j1[p]
'''
_cache = {}


def get(F, expand):
    key = (F, expand)
    if key in _cache:
        return _cache[key]
    xr = "\n".join(f"        x{f} = xT[{f}]" for f in range(F))
    wl = ["            ni = n32[i]"] if expand else []
    wl += [f"            w{f} = W[i, {f}]" for f in range(F)]
    if expand:
        body = ["                a = ni + w0 * x0[p]"] + [f"                a = a + w{f} * x{f}[p]" for f in range(1, F)]
    else:
        body = ["                d = w0 - x0[p]", "                a = d * d"]
        for f in range(1, F):
            body += [f"                d = w{f} - x{f}[p]", "                a = a + d * d"]
    src = _T.format(F=F, XR=xr, WL="\n".join(wl), BODY="\n".join(body))
    ns = {"np": np, "prange": prange}
    exec(src, ns)
    k = njit(parallel=True, fastmath={"contract"})(ns["kern"])
    _cache[key] = k
    return k


def best_match_nodes_screen3(weights, data, expand=False, B=256, chunk=2**20, return_fallbacks=False):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e15:
        raise ValueError("finite, moderate weights required")
    F = w.shape[1]
    wmax = float(np.sqrt((w * w).sum(1)).max())
    W = ((-2.0 * w) if expand else w).astype(np.float32)
    n32 = (w * w).sum(1).astype(np.float32)
    kern = get(F, expand)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    c = min(chunk, max(n, 1))
    s1 = np.empty(c, dtype=np.float32); s2 = np.empty(c, dtype=np.float32); j1 = np.empty(c, dtype=np.int32)
    nfb = np.zeros(1, dtype=np.int64)
    tot = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blk = np.ascontiguousarray(data[start:stop])
        k = stop - start
        kern(W, n32, blk, s1[:k], s2[:k], j1[:k], B)
        certify(w, wmax, blk, s1[:k], s2[:k], j1[:k], expand, idx[start:stop], d2[start:stop], nfb)
        tot += int(nfb[0])
    if return_fallbacks:
        return idx, d2, tot
    return idx, d2
