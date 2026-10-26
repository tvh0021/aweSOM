"""Prototype: tiled exact scan with the feature loop unrolled per F (code-generated)."""
import numpy as np
from numba import njit, prange

_TEMPLATE = '''
def kern(weights, data, out_idx, out_dist2, B):
    n = data.shape[0]
    n_nodes = weights.shape[0]
    ntiles = (n + B - 1) // B
    for t in prange(ntiles):
        start = t * B
        m = min(B, n - start)
        xT = np.zeros(({F}, B))
        for p in range(m):
            for f in range({F}):
                xT[f, p] = np.float64(data[start + p, f])
{XROWS}
        best = np.full(B, np.inf)
        bidx = np.zeros(B, dtype=np.int32)
        for i in range(n_nodes):
{WLOAD}
            ii = np.int32(i)
            for p in range(B):
{BODY}
                c = a < best[p]
                best[p] = a if c else best[p]
                bidx[p] = ii if c else bidx[p]
        for p in range(m):
            out_idx[start + p] = bidx[p]
            out_dist2[start + p] = best[p]
'''

_cache = {}


def get_kernel(F):
    if F in _cache:
        return _cache[F]
    xrows = "\n".join(f"        x{f} = xT[{f}]" for f in range(F))
    wload = "\n".join(f"            w{f} = weights[i, {f}]" for f in range(F))
    body = ["                d = w0 - x0[p]", "                a = 0.0 + d * d"]
    for f in range(1, F):
        body += [f"                d = w{f} - x{f}[p]", "                a = a + d * d"]
    src = _TEMPLATE.format(F=F, XROWS=xrows, WLOAD=wload, BODY="\n".join(body))
    ns = {"np": np, "prange": prange}
    exec(src, ns)
    k = njit(parallel=True, cache=False)(ns["kern"])
    _cache[F] = k
    return k


def best_match_nodes_tiledF(weights, data, B=128):
    w = np.ascontiguousarray(weights, dtype=np.float64)
    data = np.ascontiguousarray(data)
    n = data.shape[0]
    idx = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.float64)
    get_kernel(w.shape[1])(w, data, idx, d2, B)
    return idx, d2
