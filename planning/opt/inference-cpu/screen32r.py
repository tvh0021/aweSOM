"""Register-blocked variant of screen32: P points held in scalars, code-generated per (F, P).

Same certification as screen32.py (labels and distances bitwise equal to the
float64 reference scan); only the float32 screening loop differs.
"""
import numpy as np
from numba import njit, prange

from screen32 import U32, _full_scan

_TEMPLATE = '''
def kern(weights, w32, wmax, data, out_idx, out_dist2, out_nfb):
    n = data.shape[0]
    n_nodes = weights.shape[0]
    F = {F}
    g = (F + 2) * U32 * 1.01
    ntiles = (n + {P} - 1) // {P}
    nfb = np.zeros(ntiles, dtype=np.int64)
    inf32 = np.float32(np.inf)
    for t in prange(ntiles):
        start = t * {P}
        m = min({P}, n - start)
{XLOAD}
{INIT}
        for i in range(n_nodes):
{WLOAD}
            ii = np.int32(i)
{BODY}
        s1a = np.empty({P}, dtype=np.float32)
        s2a = np.empty({P}, dtype=np.float32)
        j1a = np.empty({P}, dtype=np.int32)
{STORE}
        for p in range(m):
            q = start + p
            xn = 0.0
            for f in range(F):
                v = np.float64(data[q, f])
                xn += v * v
            e = 2.0 * U32 * (wmax + np.sqrt(xn)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2a[p]) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1a[p]) / (1.0 - g)) + e) * (1.0 + 1e-12)
            if lo2 > hi1:
                j = j1a[p]
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
'''

_cache = {}


def get_kernel(F, P):
    key = (F, P)
    if key in _cache:
        return _cache[key]
    xl = []
    for p in range(P):
        for f in range(F):
            xl.append(f"        x{p}_{f} = np.float32(data[min(start + {p}, n - 1), {f}])")
    init = []
    for p in range(P):
        init += [f"        s1_{p} = inf32", f"        s2_{p} = inf32", f"        j1_{p} = np.int32(0)"]
    wl = [f"            w{f} = w32[i, {f}]" for f in range(F)]
    body = []
    for p in range(P):
        body.append(f"            d = w0 - x{p}_0")
        body.append(f"            a = d * d")
        for f in range(1, F):
            body.append(f"            d = w{f} - x{p}_{f}")
            body.append(f"            a = a + d * d")
        body += [
            f"            c = a < s1_{p}",
            f"            c2 = a < s2_{p}",
            f"            s2_{p} = s1_{p} if c else (a if c2 else s2_{p})",
            f"            j1_{p} = ii if c else j1_{p}",
            f"            s1_{p} = a if c else s1_{p}",
        ]
    store = []
    for p in range(P):
        store += [f"        s1a[{p}] = s1_{p}", f"        s2a[{p}] = s2_{p}", f"        j1a[{p}] = j1_{p}"]
    src = _TEMPLATE.format(F=F, P=P, XLOAD="\n".join(xl), INIT="\n".join(init),
                           WLOAD="\n".join(wl), BODY="\n".join(body), STORE="\n".join(store))
    ns = {"np": np, "prange": prange, "U32": U32, "_full_scan": _full_scan}
    exec(src, ns)
    k = njit(parallel=True)(ns["kern"])
    _cache[key] = k
    return k


def best_match_nodes_screen32r(weights, data, P=8, return_fallbacks=False):
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
    get_kernel(w.shape[1], P)(w, w32, wmax, data, idx, d2, nfb)
    if return_fallbacks:
        return idx, d2, int(nfb[0])
    return idx, d2
